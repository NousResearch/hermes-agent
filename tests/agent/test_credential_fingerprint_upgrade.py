"""A digest-format upgrade must not revive an unchanged, unusable credential."""

import hashlib
import hmac
import json

import pytest


def _old_fingerprint(token, prefix):
    data = token.encode()
    readers = {
        "sha256": lambda: hashlib.sha256(data).hexdigest(),
        "hmac-sha256": lambda: hmac.new(
            b"hermes-non-verifier-fingerprint-v2", data, hashlib.sha256
        ).hexdigest(),
        "sha3-256": lambda: hashlib.sha3_256(data).hexdigest(),
    }
    return prefix + ":" + readers[prefix]()[:16]


@pytest.mark.parametrize("prefix", ["sha256", "hmac-sha256", "sha3-256"])
def test_legacy_sanitized_row_keeps_exhaustion_after_rehydration(prefix):
    from agent.credential_pool import PooledCredential, _upsert_entry

    token = "synthetic-unchanged-key"
    old_fingerprint = _old_fingerprint(token, prefix)
    entries = [PooledCredential.from_dict("openrouter", {
        "id": "upgrade-test",
        "auth_type": "api_key",
        "source": "env:OPENROUTER_API_KEY",
        "secret_fingerprint": old_fingerprint,
        "last_status": "exhausted",
        "last_error_code": 429,
        "last_error_reset_at": 9999999999.0,
    })]

    _upsert_entry(entries, "openrouter", "env:OPENROUTER_API_KEY", {
        "source": "env:OPENROUTER_API_KEY",
        "auth_type": "api_key",
        "access_token": token,
    })

    assert entries[0].access_token == token
    assert entries[0].last_status == "exhausted"
    assert entries[0].last_error_code == 429
    assert entries[0].to_dict()["secret_fingerprint"].startswith("pbkdf2-sha256:")

    _upsert_entry(entries, "openrouter", "env:OPENROUTER_API_KEY", {
        "source": "env:OPENROUTER_API_KEY",
        "auth_type": "api_key",
        "access_token": "synthetic-genuinely-rotated-key",
    })
    assert entries[0].last_status is None


@pytest.mark.parametrize("prefix", ["sha256", "hmac-sha256", "sha3-256"])
def test_legacy_spent_rotation_stays_quarantined(tmp_path, monkeypatch, prefix):
    from agent import anthropic_credentials as credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = "synthetic-spent-refresh-token"
    old_fingerprint = _old_fingerprint(token, prefix)
    source_path = tmp_path / "synthetic-singleton.json"
    sidecar = credentials._spent_rotation_sidecar_path(source_path)
    sidecar.write_text(json.dumps({"version": 1, "fingerprints": [old_fingerprint]}),
                       encoding="utf-8")
    credentials._SPENT_ROTATION_FINGERPRINTS.clear()

    assert credentials.is_rotation_consumed_uncommitted(token, source_path=source_path)
    assert not credentials.is_rotation_consumed_uncommitted(
        "synthetic-independent-token", source_path=source_path)
