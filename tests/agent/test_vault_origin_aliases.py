"""Exact-origin aliases must not require replacing a saved password."""
import pytest

from agent.vault_store import VaultError, VaultStore


@pytest.fixture
def login(tmp_path):
    store = VaultStore(tmp_path / "vault")
    meta = store.add_item("login", "Example", {
        "identifier_type": "email", "identifier": "person@example.com",
        "password": "synthetic-password", "otp_secret": "JBSWY3DPEHPK3PXP",
    }, origin="https://example.com")
    return store, meta


def test_add_origin_preserves_secret_identity_and_primary(login):
    store, before = login
    original_secret = store.resolve_secret(before.id)
    after = store.add_origin(before.id, "https://WWW.example.com:443/login")
    assert after.id == before.id
    assert after.origin == before.origin
    assert after.created_at == before.created_at
    assert after.identifier == before.identifier
    assert after.has_otp
    assert after.allowed_origins == ("https://example.com", "https://www.example.com")
    assert store.resolve_secret(before.id) == original_secret
    assert store.get_meta(before.id) == after
    assert store.list_items() == [after]
    assert "secret" not in after.to_dict()
    assert after.to_dict()["allowed_origins"] == list(after.allowed_origins)


def test_add_origin_idempotent_and_persistent(login):
    store, meta = login
    store.add_origin(meta.id, "https://www.example.com")
    store.add_origin(meta.id, "https://www.example.com:443/")
    store.add_origin(meta.id, "https://example.com")
    reopened = VaultStore(store._base)
    assert reopened.get_meta(meta.id).allowed_origins == (
        "https://example.com", "https://www.example.com")


@pytest.mark.parametrize("origin", [
    "example.com", "https://*.example.com", "ftp://example.com",
    "https://user:pass@example.com", "https://example.com\\evil", "https://exa mple.com",
])
def test_reject_unsafe_alias_without_writing(login, origin):
    store, meta = login
    before = store._vault_path.read_bytes()
    with pytest.raises(VaultError):
        store.add_origin(meta.id, origin)
    assert store._vault_path.read_bytes() == before


def test_missing_handle_does_not_write(login):
    store, _ = login
    before = store._vault_path.read_bytes()
    with pytest.raises(VaultError):
        store.add_origin("vault_missing", "https://www.example.com")
    assert store._vault_path.read_bytes() == before


def test_non_login_alias_not_supported(login):
    store, _ = login
    meta = store.add_item("address", "Address", {
        "address_line1": "1 Test Road", "city": "Test", "postal_code": "12345", "country": "GB",
    }, origin="https://example.com")
    with pytest.raises(VaultError):
        store.add_origin(meta.id, "https://www.example.com")
