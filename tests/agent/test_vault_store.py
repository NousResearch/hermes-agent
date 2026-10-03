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


@pytest.mark.parametrize(
    "op",
    ["list_items", "get_meta", "remove_item", "resolve_secret", "add_item", "replace_login_password"],
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
        "replace_login_password": lambda: store.replace_login_password("vault_x", "new-pw"),
    }[op]
    with pytest.raises(VaultError):
        call()


class TestReplaceLoginPassword:
    """In-place password replacement (#123915): the stored secret changes, everything else stays.

    The old delete + re-add path forced re-entering the authenticator key (shown once at
    enrolment) and churned the vault identity fills bind to; the replacement keeps the same
    handle, origin binding, identifier metadata and TOTP seed, atomically."""

    def _login(self, store: VaultStore):
        return store.add_item(
            "login",
            "Canva",
            {
                "identifier_type": "email",
                "identifier": "u@example.com",
                "password": "old-pw-123",
                "otp_secret": "JBSWY3DPEHPK3PXP",
            },
            origin="https://canva.com",
        )

    def test_replace_keeps_handle_origin_identifier_and_otp(self, tmp_path):
        store = VaultStore(tmp_path / "vault")
        meta = self._login(store)
        out = store.replace_login_password(meta.id, "new-pw-456")
        assert out.id == meta.id
        assert out.origin == "https://canva.com"
        assert out.identifier == "u@example.com"
        assert out.has_otp is True
        assert store.resolve_secret(meta.id)["password"] == "new-pw-456"
        assert store.resolve_secret(meta.id)["otp_secret"] == "JBSWY3DPEHPK3PXP"

    def test_replace_unknown_id_raises_and_writes_nothing(self, tmp_path):
        store = self._login_store(tmp_path)
        before = store.list_items()
        with pytest.raises(VaultError):
            store.replace_login_password("vault_nope", "new-pw")
        assert [m.to_dict() for m in store.list_items()] == [m.to_dict() for m in before]

    def test_replace_rejects_empty_password_without_touching_the_entry(self, tmp_path):
        store = VaultStore(tmp_path / "vault")
        meta = self._login(store)
        with pytest.raises(VaultError):
            store.replace_login_password(meta.id, "  ")
        assert store.resolve_secret(meta.id)["password"] == "old-pw-123"

    def test_replace_rejects_non_login_kinds(self, tmp_path):
        store = VaultStore(tmp_path / "vault")
        meta = store.add_item(
            kind="payment", label="Card", origin="https://shop.test",
            secret={"card_number": "4111111111111111", "cardholder_name": "A",
                    "exp_month": "7", "exp_year": "2029", "cvc": "123"},
        )
        with pytest.raises(VaultError):
            store.replace_login_password(meta.id, "new-pw")

    def _login_store(self, tmp_path) -> VaultStore:
        store = VaultStore(tmp_path / "vault")
        self._login(store)
        return store
