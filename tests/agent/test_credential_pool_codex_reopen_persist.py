"""A Codex bench lifted by the live usage probe is persisted, not just cleared in memory.

``_MARK_OK`` drops ``last_status_at``, so the recency merge in ``write_credential_pool`` kept the
older on-disk bench: every other process, ``hermes auth list`` and the desktop provider chip went on
reporting the account as rate-limited until the stale ``resets_at``, although its quota was back.
"""

import json
import time
from unittest.mock import patch

from agent.credential_pool import load_pool

MODEL = "gpt-5.5"


def _write_exhausted_codex_pool(home, **extra):
    entry = {
        "id": "codex-1",
        "label": "codex-login",
        "auth_type": "oauth",
        "priority": 0,
        "source": "device_code",
        "access_token": "token",
        "refresh_token": "refresh",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "last_status": "exhausted",
        "last_status_at": time.time(),
        "last_error_code": 429,
        "last_error_reason": "usage_limit_reached",
        "last_error_message": "The usage limit has been reached",
        "last_error_reset_at": time.time() + 3 * 86400,
        **extra,
    }
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"openai-codex": [entry]}}))


def _stored_status(home):
    return json.loads((home / "auth.json").read_text())["credential_pool"]["openai-codex"][0]["last_status"]


def test_probe_confirmed_reopen_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_exhausted_codex_pool(tmp_path)

    with patch("hermes_cli.auth._probe_codex_quota_restored", return_value=True):
        assert load_pool("openai-codex").lift_reopened_cooldowns(model=MODEL) is True

    assert _stored_status(tmp_path) == "ok"
    assert load_pool("openai-codex").has_available(model=MODEL) is True


def test_a_newer_peer_429_still_wins_over_an_older_reopen(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_exhausted_codex_pool(tmp_path)
    pool = load_pool("openai-codex")
    with patch("hermes_cli.auth._probe_codex_quota_restored", return_value=True):
        assert pool.lift_reopened_cooldowns(model=MODEL) is True

    _write_exhausted_codex_pool(tmp_path, last_status_at=time.time() + 5)
    pool._persist()

    assert _stored_status(tmp_path) == "exhausted"
