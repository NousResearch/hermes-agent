"""``VaultStore._read_all`` must surface every decryptable-but-malformed vault
file as ``VaultError`` so callers that catch ``VaultError`` (the ``hermes vault``
commands, ``vault.*`` RPC handlers) get the designed error path instead of a raw
traceback. Corruption arrives via partial-write survivors, manual edits of the
envelope, or format drift."""

import pytest

from agent.vault_store import VaultError, VaultStore


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


def test_malformed_json_vault_file_raises_vault_error(tmp_path):
    store = _store(tmp_path)
    _corrupt(store, b"{not json")
    with pytest.raises(VaultError, match="corrupted"):
        store.list_items()


def test_non_dict_json_vault_file_raises_vault_error(tmp_path):
    store = _store(tmp_path)
    _corrupt(store, b"[1, 2]")
    with pytest.raises(VaultError, match="corrupted"):
        store.list_items()


def test_non_utf8_vault_file_raises_vault_error(tmp_path):
    store = _store(tmp_path)
    _corrupt(store, b"\xff\xfe\x00binary")
    with pytest.raises(VaultError, match="corrupted"):
        store.list_items()


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
