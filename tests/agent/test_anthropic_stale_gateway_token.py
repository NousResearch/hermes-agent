"""A gateway holding a superseded on-disk token must recover, not 401 forever.

Regression: context compression ran through ``auxiliary.compression`` on a long-lived
gateway whose cached Anthropic client kept an access token that another process had
since rotated. ``_refresh_anthropic_credentials`` returned False for any key that was
not byte-identical to the on-disk one, so no eviction happened and every compression
attempt aborted with ``summary_generation_aborted`` — silently, forever.
"""
import json
import time

import pytest

from agent import anthropic_credentials as ac
from agent import auxiliary_client as aux


def _seed_disk_login(tmp_path, monkeypatch, *, expires_at_ms):
    monkeypatch.setattr(ac.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ac, "_first_env", lambda *names: "")
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_keychain", lambda: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({"credential_pool": {}}))
    creds = tmp_path / ".claude" / ".credentials.json"
    creds.parent.mkdir(exist_ok=True)
    creds.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "fresh-disk-token", "refreshToken": "disk-refresh",
        "expiresAt": expires_at_ms,
    }}))
    return creds


def _forbid_rotation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Another process's refresh rotation was spent")
    monkeypatch.setattr(ac, "_refresh_oauth_token", forbidden)


def test_superseded_in_memory_token_recovers_from_disk(tmp_path, monkeypatch):
    """Stale cached token + valid newer token on disk -> recoverable (evict & rebuild)."""
    _seed_disk_login(tmp_path, monkeypatch, expires_at_ms=int(time.time() * 1000) + 3_600_000)
    _forbid_rotation(monkeypatch)
    assert aux._refresh_anthropic_credentials("stale-token-held-in-memory") is True


def test_superseded_token_not_recoverable_when_disk_login_is_also_expired(tmp_path, monkeypatch):
    """No usable credential anywhere -> report failure instead of an eviction loop."""
    _seed_disk_login(tmp_path, monkeypatch, expires_at_ms=1)
    _forbid_rotation(monkeypatch)
    assert aux._refresh_anthropic_credentials("stale-token-held-in-memory") is False


def test_no_failed_key_is_not_recoverable(tmp_path, monkeypatch):
    _seed_disk_login(tmp_path, monkeypatch, expires_at_ms=int(time.time() * 1000) + 3_600_000)
    _forbid_rotation(monkeypatch)
    assert aux._refresh_anthropic_credentials("") is False
