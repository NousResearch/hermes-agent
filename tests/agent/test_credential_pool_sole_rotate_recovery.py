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
from types import SimpleNamespace
from unittest.mock import Mock


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


def test_stale_session_pool_reloads_new_sibling_after_exhausted_rotation(tmp_path, monkeypatch):
    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.error_classifier import FailoverReason
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    old_pool = _load(tmp_path, monkeypatch, [_entry(1, token="tok-a", refresh="rf-a")])
    old_pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "usage_limit_reached", "reset_at": time.time() + 3600},
        credential_id="codex-1",
    )
    auth_path = tmp_path / "hermes" / "auth.json"
    store = json.loads(auth_path.read_text())
    store["credential_pool"]["openai-codex"].append(
        _entry(2, token="tok-b", refresh="rf-b")
    )
    auth_path.write_text(json.dumps(store))

    swap = Mock(return_value=True)
    agent = SimpleNamespace(
        _credential_pool=old_pool,
        _credential_pool_entry_id="codex-1",
        _credential_pool_revert_id=None,
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.3-codex",
        api_key="tok-a",
        _swap_credential=swap,
    )
    real_load_pool = load_pool
    reloads = []

    def tracked_load_pool(provider):
        reloads.append(provider)
        return real_load_pool(provider)

    monkeypatch.setattr("agent.credential_pool.load_pool", tracked_load_pool)

    recovered, retried = recover_with_credential_pool(
        agent,
        status_code=429,
        has_retried_429=False,
        classified_reason=FailoverReason.rate_limit,
        error_context={"reason": "usage_limit_reached", "reset_at": time.time() + 3600},
    )

    assert (recovered, retried) == (True, False)
    assert reloads == ["openai-codex"]
    assert agent._credential_pool is not old_pool
    assert {entry.id for entry in agent._credential_pool.entries()} == {"codex-1", "codex-2"}
    assert next(entry for entry in agent._credential_pool.entries() if entry.id == "codex-1").last_status == STATUS_EXHAUSTED
    swap.assert_called_once()
    assert swap.call_args.args[0].id == "codex-2"
