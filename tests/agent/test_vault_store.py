"""``VaultStore._read_all`` must surface every decryptable-but-malformed vault
file as ``VaultError`` so callers that catch ``VaultError`` (the ``hermes vault``
commands, ``vault.*`` RPC handlers) get the designed error path instead of a raw
traceback. Corruption arrives via partial-write survivors, manual edits of the
envelope, or format drift."""

import pytest

from agent.vault_store import VaultError, VaultStore, normalize_origin


def _corrupt(store: VaultStore, payload: bytes) -> None:
    store._vault_path.write_bytes(store._fernet().encrypt(payload))


def _store(tmp_path) -> VaultStore:
    store = VaultStore(tmp_path / "vault")
    store.add_item(
        "login",
        "Example login",
        {"identifier_type": "email", "identifier": "u@example.com", "password": "pw"},
        origin="https://example.com",
    )
    return store


@pytest.mark.parametrize(
    "op",
    ["list_items", "get_meta", "remove_item", "resolve_secret", "add_item"],
)
def test_corrupt_vault_raises_vault_error_on_every_reader(tmp_path, op):
    """Every ``_read_all`` consumer must see the same VaultError contract so
    callers that catch VaultError all get the designed error path."""
    store = _store(tmp_path)
    _corrupt(store, b"{not json")
    call = {
        "list_items": lambda: store.list_items(),
        "get_meta": lambda: store.get_meta("vault_x"),
        "remove_item": lambda: store.remove_item("vault_x"),
        "resolve_secret": lambda: store.resolve_secret("vault_x"),
        "add_item": lambda: store.add_item(
            "login",
            "x",
            {"identifier_type": "email", "identifier": "u", "password": "p"},
            origin="https://x.example",
        ),
    }[op]
    with pytest.raises(VaultError):
        call()


@pytest.mark.parametrize("pasted", [
    "https://example.com\u200b",
    "\ufeffhttps://example.com/login\u200b",
    "\u200b https://example.com \u200b",
])
def test_pasted_invisible_edge_chars_bind_to_the_real_origin(tmp_path, pasted):
    """A zero-width char copied along with a URL is invisible in every UI, yet left in the
    hostname it binds the login to an origin no page ever has: the item never fills."""
    meta = VaultStore(tmp_path / "vault").add_item(
        "login",
        "Example login",
        {"identifier_type": "email", "identifier": "u@example.com", "password": "pw"},
        origin=pasted,
    )
    assert meta.origin == normalize_origin("https://example.com/") == "https://example.com"


@pytest.mark.parametrize("value", ["https://exa\u200bmple.com", "https://example.com\u200b.evil.test"])
def test_embedded_invisible_chars_are_rejected_not_folded(value):
    """Inside the value an invisible char is not paste residue; folding it away would bind
    the item to a host the user never saw spelled out."""
    with pytest.raises(VaultError, match="invisible"):
        normalize_origin(value)


@pytest.mark.parametrize("value", ["\u202ehttps://evil.example", "https://evil.example\u202e"])
def test_bidi_override_at_edge_is_not_silently_stripped(value):
    with pytest.raises(VaultError):
        normalize_origin(value)
