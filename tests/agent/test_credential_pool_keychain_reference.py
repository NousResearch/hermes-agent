"""Keychain-reference credentials in the pool: runtime-only resolution,
disk stripping, and fail-closed behavior.

All secrets are synthetic. The Keychain reader is always patched; no test
touches a real Keychain.
"""

from __future__ import annotations

import json

import pytest

import agent.keychain_secret as keychain_secret_mod
from agent.credential_pool import CredentialPool, PooledCredential
from agent.keychain_secret import KeychainRef, KeychainUnavailable

SYNTHETIC_CLIENT_KEY = "synthetic-broker-client-key-42"
CLIENT_URI = "keychain://ai.hermes.oauth-broker.client/local"


def _reference_error_type():
    """Imported lazily so the module collects before implementation exists."""
    from agent.credential_pool import KeychainReferenceUnresolved

    return KeychainReferenceUnresolved


def _broker_entry(**overrides):
    payload = dict(
        provider="openai-codex",
        id="broker-A",
        label="broker-A",
        auth_type="api_key",
        priority=0,
        source="keychain_reference",
        access_token="MUST_NOT_PERSIST",
        base_url="http://127.0.0.1:17880/accounts/A/backend-api/codex",
        extra={"secret_source": CLIENT_URI},
    )
    payload.update(overrides)
    return PooledCredential(**payload)


def _patch_reader(monkeypatch, result=SYNTHETIC_CLIENT_KEY):
    calls = []

    def fake_read(ref, *, backend=None):
        calls.append(ref)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(keychain_secret_mod, "read_keychain_secret", fake_read)
    return calls


# ---------------------------------------------------------------------------
# Disk boundary
# ---------------------------------------------------------------------------


def test_to_dict_strips_access_token_and_keeps_reference():
    entry = _broker_entry()
    payload = entry.to_dict()
    assert "access_token" not in payload
    assert "refresh_token" not in payload
    assert payload["secret_source"] == CLIENT_URI
    assert payload["base_url"] == "http://127.0.0.1:17880/accounts/A/backend-api/codex"
    assert payload["source"] == "keychain_reference"


def test_to_dict_writes_only_one_way_fingerprint():
    payload = _broker_entry().to_dict()
    fingerprint = payload.get("secret_fingerprint")
    assert isinstance(fingerprint, str) and fingerprint.startswith("sha256:")
    assert "MUST_NOT_PERSIST" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------


def test_runtime_api_key_resolves_reference_at_access_time(monkeypatch):
    calls = _patch_reader(monkeypatch)
    entry = _broker_entry()
    assert entry.runtime_api_key == SYNTHETIC_CLIENT_KEY
    assert entry.runtime_api_key == SYNTHETIC_CLIENT_KEY
    # Resolved on every access — no cached or persisted copies.
    assert calls == [
        KeychainRef(service="ai.hermes.oauth-broker.client", account="local"),
        KeychainRef(service="ai.hermes.oauth-broker.client", account="local"),
    ]


def test_runtime_resolution_survives_disk_round_trip(monkeypatch):
    _patch_reader(monkeypatch)
    reloaded = PooledCredential.from_dict("openai-codex", _broker_entry().to_dict())
    assert reloaded.source == "keychain_reference"
    assert reloaded.runtime_api_key == SYNTHETIC_CLIENT_KEY


def test_fail_closed_when_keychain_unavailable_and_no_env_fallback(monkeypatch):
    _patch_reader(monkeypatch, result=KeychainUnavailable(message="locked"))
    monkeypatch.setenv("OPENAI_API_KEY", "env-value-must-never-be-used")
    entry = _broker_entry()
    with pytest.raises(_reference_error_type()):
        entry.runtime_api_key


def test_fail_closed_when_secret_source_missing(monkeypatch):
    _patch_reader(monkeypatch)
    entry = _broker_entry(extra={})
    with pytest.raises(_reference_error_type()):
        entry.runtime_api_key


def test_fail_closed_on_malformed_secret_source(monkeypatch):
    _patch_reader(monkeypatch)
    entry = _broker_entry(extra={"secret_source": "keychain://only-service"})
    with pytest.raises(_reference_error_type()):
        entry.runtime_api_key


def test_reference_error_text_has_no_secret_material(monkeypatch):
    _patch_reader(monkeypatch, result=KeychainUnavailable(message="locked"))
    entry = _broker_entry()
    with pytest.raises(_reference_error_type()) as excinfo:
        entry.runtime_api_key
    text = str(excinfo.value)
    assert SYNTHETIC_CLIENT_KEY not in text
    assert "MUST_NOT_PERSIST" not in text


# ---------------------------------------------------------------------------
# Non-reference credentials keep current behavior
# ---------------------------------------------------------------------------


def test_manual_credential_ignores_secret_source(monkeypatch):
    calls = _patch_reader(monkeypatch)
    entry = _broker_entry(
        id="manual-1",
        source="manual",
        access_token="synthetic-manual-token",
        extra={"secret_source": CLIENT_URI},
    )
    assert entry.runtime_api_key == "synthetic-manual-token"
    assert calls == []


def test_oauth_credential_keeps_access_token_behavior(monkeypatch):
    calls = _patch_reader(monkeypatch)
    entry = _broker_entry(
        id="oauth-1",
        source="device_code",
        auth_type="oauth",
        access_token="synthetic-oauth-access",
        extra={},
    )
    assert entry.runtime_api_key == "synthetic-oauth-access"
    assert calls == []


# ---------------------------------------------------------------------------
# Pool selection fails closed per entry, not per pool
# ---------------------------------------------------------------------------


def test_selection_skips_unresolvable_reference_entry(monkeypatch):
    _patch_reader(monkeypatch, result=KeychainUnavailable(message="locked"))
    reference = _broker_entry()
    manual = _broker_entry(
        id="manual-1",
        label="manual-1",
        source="manual",
        priority=1,
        access_token="synthetic-manual-token",
        extra={},
    )
    pool = CredentialPool("openai-codex", [reference, manual])
    selected = pool.select()
    assert selected is not None and selected.id == "manual-1"


def test_selection_returns_none_when_only_unresolvable_references(monkeypatch):
    _patch_reader(monkeypatch, result=KeychainUnavailable(message="locked"))
    pool = CredentialPool("openai-codex", [_broker_entry()])
    assert pool.select() is None


def _three_shared_key_broker_entries():
    order = ("B", "C", "A")
    return [
        _broker_entry(
            id=f"broker-{alias}",
            label=f"broker-{alias}",
            priority=priority,
            base_url=(
                f"http://127.0.0.1:17880/accounts/{alias}/backend-api/codex"
            ),
        )
        for priority, alias in enumerate(order)
    ]


def test_shared_key_rotation_uses_base_url_to_mark_only_failed_alias(monkeypatch):
    _patch_reader(monkeypatch)
    pool = CredentialPool("openai-codex", _three_shared_key_broker_entries())
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: None)

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint=SYNTHETIC_CLIENT_KEY,
        base_url_hint="http://127.0.0.1:17880/accounts/B/backend-api/codex/",
    )

    by_id = {entry.id: entry for entry in pool.entries()}
    assert by_id["broker-B"].last_status == "exhausted"
    assert by_id["broker-A"].last_status is None
    assert by_id["broker-C"].last_status is None
    assert next_entry is not None and next_entry.id == "broker-C"


def test_ambiguous_shared_key_without_current_or_route_fails_closed(monkeypatch):
    _patch_reader(monkeypatch)
    pool = CredentialPool("openai-codex", _three_shared_key_broker_entries())
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: None)

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint=SYNTHETIC_CLIENT_KEY,
    )

    assert next_entry is None
    assert all(entry.last_status is None for entry in pool.entries())


def test_ambiguous_shared_key_with_current_still_fails_closed(monkeypatch):
    _patch_reader(monkeypatch)
    pool = CredentialPool("openai-codex", _three_shared_key_broker_entries())
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: None)
    selected = pool.select()
    assert selected is not None and selected.id == "broker-B"

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint=SYNTHETIC_CLIENT_KEY,
    )

    assert next_entry is None
    assert all(entry.last_status is None for entry in pool.entries())
