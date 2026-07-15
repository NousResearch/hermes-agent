"""Keychain OAuth grant store: alias gating, schema-versioned payloads,
redacted errors. Synthetic tokens and fake backends only."""

from __future__ import annotations

import json

import pytest

from agent.oauth_broker.grant_store import (
    KeychainGrantStore,
    grant_to_payload,
)
from agent.oauth_broker.models import (
    ACCOUNT_ALIASES,
    GRANT_SCHEMA_VERSION,
    OAUTH_GRANT_KEYCHAIN_SERVICE,
    GrantStoreError,
    OAuthGrant,
)

SYNTHETIC_ACCESS = "synthetic-access-token-A1"
SYNTHETIC_REFRESH = "synthetic-refresh-token-A1"


def _grant(**overrides):
    payload = dict(
        access_token=SYNTHETIC_ACCESS,
        refresh_token=SYNTHETIC_REFRESH,
        expires_at=4102444800.0,  # 2100-01-01, synthetic far future
        account_id="acct-synthetic-a",
    )
    payload.update(overrides)
    return OAuthGrant(**payload)


def _store(fake_keychain):
    return KeychainGrantStore(backend=fake_keychain)


# ---------------------------------------------------------------------------
# Constants and alias gating
# ---------------------------------------------------------------------------


def test_account_aliases_are_exactly_a_b_c():
    assert ACCOUNT_ALIASES == ("A", "B", "C")
    assert OAUTH_GRANT_KEYCHAIN_SERVICE == "ai.hermes.oauth-broker.openai-codex"


@pytest.mark.parametrize("bad_alias", ["D", "a", "", "AB", " A", None])
def test_operations_reject_unknown_aliases(fake_keychain, bad_alias):
    store = _store(fake_keychain)
    for operation in (
        lambda: store.load(bad_alias),
        lambda: store.replace(bad_alias, _grant()),
        lambda: store.delete(bad_alias),
    ):
        with pytest.raises(GrantStoreError) as excinfo:
            operation()
        assert excinfo.value.category == "invalid_alias"
    assert fake_keychain.read_calls == []
    assert fake_keychain.write_calls == []
    assert fake_keychain.delete_calls == []


# ---------------------------------------------------------------------------
# Round trip and payload schema
# ---------------------------------------------------------------------------


def test_replace_then_load_round_trip(fake_keychain):
    store = _store(fake_keychain)
    grant = _grant()
    store.replace("A", grant)
    assert store.load("A") == grant
    assert fake_keychain.write_calls == [(OAUTH_GRANT_KEYCHAIN_SERVICE, "A")]
    assert fake_keychain.read_calls == [(OAUTH_GRANT_KEYCHAIN_SERVICE, "A")]


def test_serialized_payload_is_versioned_and_minimal(fake_keychain):
    store = _store(fake_keychain)
    store.replace("B", _grant())
    raw = fake_keychain.items[(OAUTH_GRANT_KEYCHAIN_SERVICE, "B")]
    payload = json.loads(raw)
    assert payload["schema_version"] == GRANT_SCHEMA_VERSION
    assert set(payload) == {
        "schema_version",
        "access_token",
        "refresh_token",
        "expires_at",
        "account_id",
    }


def test_replace_overwrites_previous_grant_atomically(fake_keychain):
    store = _store(fake_keychain)
    store.replace("A", _grant())
    rotated = _grant(
        access_token="synthetic-access-token-A2",
        refresh_token="synthetic-refresh-token-A2",
    )
    store.replace("A", rotated)
    assert store.load("A") == rotated
    # One keychain item per alias — the write path never creates a second copy.
    assert list(fake_keychain.items) == [(OAUTH_GRANT_KEYCHAIN_SERVICE, "A")]


def test_delete_removes_grant(fake_keychain):
    store = _store(fake_keychain)
    store.replace("C", _grant())
    store.delete("C")
    assert fake_keychain.items == {}
    with pytest.raises(GrantStoreError) as excinfo:
        store.load("C")
    assert excinfo.value.category == "not_found"


def test_load_missing_grant_reports_alias_and_category(fake_keychain):
    store = _store(fake_keychain)
    with pytest.raises(GrantStoreError) as excinfo:
        store.load("A")
    assert excinfo.value.alias == "A"
    assert excinfo.value.category == "not_found"
    assert "A" in str(excinfo.value)


def test_delete_missing_grant_is_not_found(fake_keychain):
    store = _store(fake_keychain)
    with pytest.raises(GrantStoreError) as excinfo:
        store.delete("B")
    assert excinfo.value.category == "not_found"


# ---------------------------------------------------------------------------
# Payload rejection paths — errors never leak token substrings
# ---------------------------------------------------------------------------


def _seed_raw(fake_keychain, alias, raw):
    fake_keychain.items[(OAUTH_GRANT_KEYCHAIN_SERVICE, alias)] = raw


def test_load_rejects_wrong_schema_version(fake_keychain):
    store = _store(fake_keychain)
    _seed_raw(
        fake_keychain,
        "A",
        json.dumps(
            {
                "schema_version": 99,
                "access_token": SYNTHETIC_ACCESS,
                "refresh_token": SYNTHETIC_REFRESH,
                "expires_at": 4102444800.0,
                "account_id": "acct-synthetic-a",
            }
        ),
    )
    with pytest.raises(GrantStoreError) as excinfo:
        store.load("A")
    assert excinfo.value.category == "schema_version"
    assert SYNTHETIC_ACCESS not in str(excinfo.value)


def test_load_rejects_unknown_secret_bearing_field(fake_keychain):
    store = _store(fake_keychain)
    _seed_raw(
        fake_keychain,
        "A",
        json.dumps(
            {
                "schema_version": GRANT_SCHEMA_VERSION,
                "access_token": SYNTHETIC_ACCESS,
                "refresh_token": SYNTHETIC_REFRESH,
                "expires_at": 4102444800.0,
                "account_id": "acct-synthetic-a",
                "id_token": "synthetic-id-token-should-not-echo",
            }
        ),
    )
    with pytest.raises(GrantStoreError) as excinfo:
        store.load("A")
    text = str(excinfo.value)
    assert excinfo.value.category == "unknown_field"
    assert "id_token" in text
    assert "[REDACTED]" in text
    assert "synthetic-id-token-should-not-echo" not in text


def test_load_rejects_unknown_benign_field(fake_keychain):
    store = _store(fake_keychain)
    _seed_raw(
        fake_keychain,
        "A",
        json.dumps(
            {
                "schema_version": GRANT_SCHEMA_VERSION,
                "access_token": SYNTHETIC_ACCESS,
                "refresh_token": SYNTHETIC_REFRESH,
                "expires_at": 4102444800.0,
                "account_id": "acct-synthetic-a",
                "note": "benign-forward-compat-field",
            }
        ),
    )
    with pytest.raises(GrantStoreError) as excinfo:
        store.load("A")
    assert excinfo.value.category == "unknown_field"
    assert "note" in str(excinfo.value)


@pytest.mark.parametrize("bad_version", [True, 1.0])
def test_load_rejects_non_integer_schema_version(fake_keychain, bad_version):
    _seed_raw(
        fake_keychain,
        "A",
        json.dumps(
            {
                "schema_version": bad_version,
                "access_token": SYNTHETIC_ACCESS,
                "refresh_token": SYNTHETIC_REFRESH,
                "expires_at": 4102444800.0,
                "account_id": "acct-synthetic-a",
            }
        ),
    )
    with pytest.raises(GrantStoreError) as excinfo:
        _store(fake_keychain).load("A")
    assert excinfo.value.category == "schema_version"


def test_load_rejects_duplicate_json_keys(fake_keychain):
    raw = (
        '{"schema_version":1,'
        '"access_token":"synthetic-first",'
        '"access_token":"synthetic-second",'
        '"refresh_token":"synthetic-refresh",'
        '"expires_at":4102444800.0,'
        '"account_id":"acct-synthetic-a"}'
    )
    _seed_raw(fake_keychain, "A", raw)
    with pytest.raises(GrantStoreError) as excinfo:
        _store(fake_keychain).load("A")
    assert excinfo.value.category == "duplicate_field"
    assert "synthetic-first" not in str(excinfo.value)
    assert "synthetic-second" not in str(excinfo.value)


def test_load_rejects_non_json_payload_without_echoing_it(fake_keychain):
    store = _store(fake_keychain)
    _seed_raw(fake_keychain, "A", "synthetic-not-json-" + SYNTHETIC_ACCESS)
    with pytest.raises(GrantStoreError) as excinfo:
        store.load("A")
    text = str(excinfo.value)
    assert excinfo.value.category == "invalid_payload"
    assert "[REDACTED]" in text
    assert SYNTHETIC_ACCESS not in text


def test_load_rejects_missing_required_field_with_redacted_text(fake_keychain):
    store = _store(fake_keychain)
    _seed_raw(
        fake_keychain,
        "A",
        json.dumps(
            {
                "schema_version": GRANT_SCHEMA_VERSION,
                "refresh_token": SYNTHETIC_REFRESH,
                "expires_at": 4102444800.0,
                "account_id": "acct-synthetic-a",
            }
        ),
    )
    with pytest.raises(GrantStoreError) as excinfo:
        store.load("A")
    text = str(excinfo.value)
    assert excinfo.value.category == "invalid_grant_payload"
    assert "[REDACTED]" in text
    assert SYNTHETIC_REFRESH not in text


# ---------------------------------------------------------------------------
# OAuthGrant validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_expiry", [float("nan"), float("inf"), float("-inf")])
def test_grant_rejects_non_finite_expiry(bad_expiry):
    with pytest.raises(ValueError, match="finite"):
        _grant(expires_at=bad_expiry)


def test_grant_payload_json_encoder_forbids_non_finite_numbers():
    grant = object.__new__(OAuthGrant)
    object.__setattr__(grant, "access_token", SYNTHETIC_ACCESS)
    object.__setattr__(grant, "refresh_token", SYNTHETIC_REFRESH)
    object.__setattr__(grant, "expires_at", float("nan"))
    object.__setattr__(grant, "account_id", "acct-synthetic-a")
    with pytest.raises(ValueError):
        grant_to_payload(grant)


@pytest.mark.parametrize(
    "overrides",
    [
        {"access_token": ""},
        {"access_token": "   "},
        {"refresh_token": ""},
        {"account_id": ""},
        {"expires_at": "soon"},
        {"expires_at": None},
        {"expires_at": True},  # bool is not a timestamp
    ],
)
def test_grant_validation_rejects_bad_fields(overrides):
    with pytest.raises(ValueError) as excinfo:
        _grant(**overrides)
    text = str(excinfo.value)
    assert "[REDACTED]" in text
    assert SYNTHETIC_ACCESS not in text
    assert SYNTHETIC_REFRESH not in text
