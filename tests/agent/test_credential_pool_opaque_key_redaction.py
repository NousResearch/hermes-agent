"""An echoed key with no recognisable prefix must not reach ``auth.json`` either (#71994).

The pattern redactor cannot see an opaque key (random hex from a custom provider), so the pool
masks the failing row's own secret values verbatim before it records the provider's error text.
"""
from __future__ import annotations

import json

from agent.credential_pool import CredentialPool, PooledCredential

_OPAQUE_KEY = "9f3c" + "a1b2c3d4e5f60718293a4b5c6d7e8f90"


def _pool(source: str) -> CredentialPool:
    return CredentialPool("anthropic", [
        PooledCredential(id="k1", provider="anthropic", auth_type="api_key", access_token=_OPAQUE_KEY,
                         label="a", source=source, priority=0),
        PooledCredential(id="k2", provider="anthropic", auth_type="api_key", access_token="other" + "x" * 20,
                         label="b", source="manual", priority=1),
    ])


def test_borrowed_row_echoed_opaque_key_is_masked_on_disk(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    pool = _pool("env:ANTHROPIC_API_KEY")
    entry = pool._entries[0]

    pool._mark_exhausted(entry, 401, {"message": f"Invalid API key: {_OPAQUE_KEY}", "reason": _OPAQUE_KEY})

    raw = (home / "auth.json").read_text()
    assert _OPAQUE_KEY not in raw
    row = next(r for r in json.loads(raw)["credential_pool"]["anthropic"] if r["id"] == "k1")
    assert row["last_error_message"] == "Invalid API key: ***"
    assert "access_token" not in row  # borrowed: the sanitizer still strips the secret itself


def test_mark_dead_masks_the_rows_own_refresh_token(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    pool = _pool("manual")
    entry = pool._entries[0]
    entry.refresh_token = "rt" + _OPAQUE_KEY
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)

    pool._mark_dead_refresh_grant(entry, RuntimeError(f"invalid_grant: {entry.refresh_token}"))

    message = pool._entries[0].last_error_message or ""
    assert _OPAQUE_KEY not in message and message.startswith("invalid_grant")
