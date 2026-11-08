"""#97315: a rotation that can only hand back the entry that was just marked
exhausted is no recovery — it must return None so the turn fails.

A sole-credential ``openai-codex`` pool hitting a 429 ``usage_limit_reached``
writes a correct bench (``last_error_reset_at`` days out), yet the selection
path can re-admit the just-marked entry within milliseconds (auth-store sync
adopting fresher tokens, a false-positive quota probe). ``mark_exhausted_and_rotate``
then reports a successful rotation, and the caller retries the same throttled
credential forever (~2 req/s for hours, gateway wedged until killed by hand).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace


def _entry(idx: int, *, token: str, refresh: str) -> dict:
    return {
        "id": f"codex-{idx}",
        "label": f"codex-login-{idx}",
        "auth_type": "oauth",
        "priority": idx,
        "source": "device_code",
        "access_token": token,
        "refresh_token": refresh,
    }


def _load(tmp_path, monkeypatch, entries: list[dict]):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"openai-codex": entries}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    return load_pool("openai-codex")


def _revive_entry(pool, entry):
    """Simulate the auth-store sync adopting fresher tokens: the bench is cleared
    mid-selection, exactly as ``_sync_entry_from_auth_store`` does for a
    changed token pair."""
    updated = replace(
        entry,
        access_token=entry.access_token + "-adopted",
        refresh_token=(entry.refresh_token or "") + "-adopted",
        last_status=None,
        last_status_at=None,
        last_error_code=None,
        last_error_reason=None,
        last_error_message=None,
        last_error_reset_at=None,
    )
    pool._replace_entry(entry, updated)
    return updated


def _mark(pool, credential_id: str):
    return pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "usage_limit_reached", "reset_at": time.time() + 95.5 * 3600},
        credential_id=credential_id,
    )


def test_revived_sole_entry_is_no_recovery(tmp_path, monkeypatch):
    pool = _load(tmp_path, monkeypatch, [_entry(1, token="tok-a", refresh="rf-a")])
    pool._sync_entry_from_auth_store = lambda entry: _revive_entry(pool, entry)

    assert _mark(pool, "codex-1") is None


def test_multi_entry_pool_still_rotates_to_healthy_sibling(tmp_path, monkeypatch):
    pool = _load(
        tmp_path, monkeypatch,
        [_entry(1, token="tok-a", refresh="rf-a"), _entry(2, token="tok-b", refresh="rf-b")],
    )
    pool._sync_entry_from_auth_store = lambda entry: entry

    nxt = _mark(pool, "codex-1")

    assert nxt is not None and nxt.id == "codex-2"
