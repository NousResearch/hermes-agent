"""``VaultStore._read_all`` must surface every decryptable-but-malformed vault
file as ``VaultError`` so callers that catch ``VaultError`` (the ``hermes vault``
commands, ``vault.*`` RPC handlers) get the designed error path instead of a raw
traceback. Corruption arrives via partial-write survivors, manual edits of the
envelope, or format drift."""

import pytest
from datetime import datetime

from agent.vault_store import VaultError, VaultStore


CHECKOUT_SECRETS = {
    "payment": {"card_number": "4111111111111111", "exp_month": "7", "exp_year": "2029", "cvc": "123"},
    "address": {"address_line1": "1 Main St", "city": "Springfield", "postal_code": "12345", "country": "US"},
}


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


def test_local_origins_are_primary_first_stable_and_reads_do_not_migrate(tmp_path):
    store = _store(tmp_path)
    [meta] = store.list_items()
    before = store._vault_path.read_bytes()
    assert meta.allowed_origins == ("https://example.com",)
    assert store.get_meta(meta.id).allowed_origins == meta.allowed_origins
    assert store._vault_path.read_bytes() == before

    records = store._read_all()
    records[0]["allowed_origins"] = ["https://second.test", "https://example.com",
                                      "https://second.test", "http://second.test:8080"]
    store._write_all(records)
    assert store.get_meta(meta.id).allowed_origins == (
        "https://example.com", "https://second.test", "http://second.test:8080")
    assert store.get_meta(meta.id).to_dict()["allowed_origins"] == [
        "https://example.com", "https://second.test", "http://second.test:8080"]


@pytest.mark.parametrize("kind", ["payment", "address"])
def test_authorize_and_revoke_persist_without_resolving_or_changing_secrets(tmp_path, monkeypatch, kind):
    store = VaultStore(tmp_path / "vault")
    meta = store.add_item(kind, "Checkout", CHECKOUT_SECRETS[kind], origin="https://primary.test")
    monkeypatch.setattr(store, "resolve_secret", lambda *_: pytest.fail("origin operations must not resolve secrets"))
    assert store.origin_events(meta.id) == []
    authorized = store.authorize_origin(meta.id, "HTTPS://Second.TEST:443/")
    assert authorized.allowed_origins == ("https://primary.test", "https://second.test")
    after_add = store._vault_path.read_bytes()
    assert store.authorize_origin(meta.id, "https://second.test").allowed_origins == authorized.allowed_origins
    assert store.authorize_origin(meta.id, "https://primary.test").allowed_origins == authorized.allowed_origins
    assert store._vault_path.read_bytes() == after_add
    store.authorize_origin(meta.id, "http://second.test:8080")
    revoked = store.revoke_origin(meta.id, "https://SECOND.test:443/")
    assert revoked.allowed_origins == ("https://primary.test", "http://second.test:8080")
    fresh = VaultStore(tmp_path / "vault")
    assert fresh.get_meta(meta.id).allowed_origins == revoked.allowed_origins
    assert fresh.resolve_secret(meta.id) == CHECKOUT_SECRETS[kind]
    events = fresh.origin_events(meta.id)
    assert [(e["action"], e["origin"]) for e in events] == [
        ("add", "https://second.test"), ("add", "http://second.test:8080"), ("remove", "https://second.test")]
    for event in events:
        assert set(event) == {"timestamp", "action", "origin"}
        assert datetime.fromisoformat(event["timestamp"]).tzinfo is not None
    public = str(events) + str(fresh.get_meta(meta.id).to_dict())
    for field, value in CHECKOUT_SECRETS[kind].items():
        assert field not in public
        if len(value) >= 6:
            assert value not in public
    assert b"https://second.test" not in fresh._vault_path.read_bytes()
    events[0]["origin"] = "https://unapproved.test"
    assert fresh.origin_events(meta.id)[0]["origin"] == "https://second.test"


@pytest.mark.parametrize("url", [
    "", "second.test", "/checkout", "ftp://second.test", "file:///checkout", "https://*.test",
    "https://user@second.test", "https://user:password@second.test", "https://second.test/checkout",
    "https://second.test//", "https://second.test/?x=1", "https://second.test/#frag",
    "https://second.test?", "https://second.test#", "https://second.test:bad", "https://[invalid",
])
def test_origin_mutations_reject_non_origin_input_without_echo_or_write(tmp_path, url):
    store = VaultStore(tmp_path / "vault")
    meta = store.add_item("payment", "Card", CHECKOUT_SECRETS["payment"], origin="https://primary.test")
    before = store._vault_path.read_bytes()
    for method in (store.authorize_origin, store.revoke_origin):
        with pytest.raises(VaultError) as error:
            method(meta.id, url)
        assert "password" not in str(error.value)
        assert store._vault_path.read_bytes() == before


def test_origin_mutations_require_existing_checkout_with_primary_and_protect_primary(tmp_path):
    store = _store(tmp_path)
    [login] = store.list_items()
    unbound = store.add_item("address", "Unbound", CHECKOUT_SECRETS["address"])
    card = store.add_item("payment", "Card", CHECKOUT_SECRETS["payment"], origin="https://primary.test")
    before = store._vault_path.read_bytes()
    for handle in ("vault_missing", login.id, unbound.id):
        for method in (store.authorize_origin, store.revoke_origin):
            with pytest.raises(VaultError):
                method(handle, "https://second.test")
    for url in ("https://primary.test:443/", "https://missing.test"):
        with pytest.raises(VaultError):
            store.revoke_origin(card.id, url)
    with pytest.raises(VaultError):
        store.origin_events("vault_missing")
    assert store._vault_path.read_bytes() == before


@pytest.mark.parametrize(
    "op",
    ["list_items", "get_meta", "remove_item", "resolve_secret", "add_item",
     "authorize_origin", "revoke_origin", "origin_events"],
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
        "authorize_origin": lambda: store.authorize_origin("vault_x", "https://second.test"),
        "revoke_origin": lambda: store.revoke_origin("vault_x", "https://second.test"),
        "origin_events": lambda: store.origin_events("vault_x"),
        "add_item": lambda: store.add_item(
            "login",
            "x",
            {"identifier_type": "email", "identifier": "u", "password": "p"},
            origin="https://x.example",
        ),
    }[op]
    with pytest.raises(VaultError):
        call()
