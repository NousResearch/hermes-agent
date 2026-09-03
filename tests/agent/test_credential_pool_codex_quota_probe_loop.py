"""A false-positive Codex quota probe must not hot-loop a benched credential.

The ``/usage`` quota probe lifts an ``openai-codex`` cooldown when the upstream
window looks reopened.  It reports window utilisation only, so it can read
"restored" while the account is still refused for a limit that endpoint does
not expose.  A positive result is also cached for 5 minutes, so before this
guard every selection during that window handed the benched credential straight
back: the pool never reached "no available entries", ``rotate()`` never returned
None, and the ``fallback_providers`` chain was never reached.

Observed 2026-08-26: ~8,000 requests over 2.5h against a one-entry
``openai-codex`` pool, replaying the same 429 at ~1 req/s until the real quota
window reopened.

The probe now buys exactly one optimistic retry per entry per quota window.
"""

from __future__ import annotations

import json
import time

import pytest


RESET_AT_FAR_FUTURE_SECONDS = 3 * 60 * 60


def _load(tmp_path, monkeypatch, *, last_status_at: float, reset_at: float):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "455eb2",
                    "label": "device_code",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual",
                    "access_token": "***",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "last_status": "exhausted",
                    "last_status_at": last_status_at,
                    "last_error_code": 429,
                    "last_error_reason": "usage_limit_reached",
                    "last_error_message": "The usage limit has been reached",
                    "last_error_reset_at": reset_at,
                }
            ]
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.credential_pool import load_pool

    return load_pool("openai-codex")


@pytest.fixture
def always_restored(monkeypatch):
    """Force the probe to insist the quota is restored, and count the calls."""
    import agent.credential_pool as cp

    calls: list[int] = []

    def _probe(token, *, base_url=None, **kwargs):
        calls.append(1)
        return True

    monkeypatch.setattr(cp.auth_mod, "_probe_codex_quota_restored", _probe)
    monkeypatch.setattr(
        cp.auth_mod,
        "_is_codex_rate_limit_shaped",
        lambda code, reason, message: True,
    )
    return calls


def test_positive_probe_still_grants_one_optimistic_retry(tmp_path, monkeypatch, always_restored):
    """The #43747 early-reopen behaviour is preserved for the first attempt."""
    now = time.time()
    pool = _load(
        tmp_path,
        monkeypatch,
        last_status_at=now - 5,
        reset_at=now + RESET_AT_FAR_FUTURE_SECONDS,
    )

    entry = pool.select()

    assert entry is not None, "a positive probe should lift the stale cooldown once"
    assert entry.id == "455eb2"
    assert entry.last_status == "ok"
    assert len(always_restored) == 1


def test_probe_is_not_retrusted_after_the_account_429s_again(
    tmp_path, monkeypatch, always_restored
):
    """The loop terminates: pool empties, so provider fallback can engage."""
    now = time.time()
    pool = _load(
        tmp_path,
        monkeypatch,
        last_status_at=now - 5,
        reset_at=now + RESET_AT_FAR_FUTURE_SECONDS,
    )

    assert pool.select() is not None  # optimistic retry granted

    # ...and upstream refuses it anyway, exactly as it did for 2.5 hours.
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429,
        credential_id="455eb2",
        error_context={
            "reason": "usage_limit_reached",
            "message": "The usage limit has been reached",
            "reset_at": now + RESET_AT_FAR_FUTURE_SECONDS,
        },
    )

    assert rotated is None, (
        "a one-entry pool whose only credential just 429'd must report no "
        "rotation target so the caller falls through to fallback_providers"
    )
    # Every subsequent selection stays empty rather than replaying the 429.
    for _ in range(25):
        assert pool.select() is None
    assert len(always_restored) == 1, (
        "the discredited probe must not be consulted again for this window"
    )


def test_marker_clears_once_the_real_cooldown_elapses(tmp_path, monkeypatch, always_restored):
    """A later quota window gets its own early-reopen detection."""
    import agent.credential_pool as cp

    now = time.time()
    pool = _load(
        tmp_path,
        monkeypatch,
        last_status_at=now - 5,
        reset_at=now + RESET_AT_FAR_FUTURE_SECONDS,
    )
    assert pool.select() is not None
    assert (
        pool.mark_exhausted_and_rotate(
            status_code=429,
            credential_id="455eb2",
            error_context={"reason": "usage_limit_reached"},
        )
        is None
    )
    assert pool._codex_probe_honored_at

    # The genuine reset lands: cooldown elapsed, entry recovers on its own.
    with pool._lock:
        stale = pool._entries[0]
        pool._replace_entry(
            stale,
            cp.replace(
                stale,
                last_status_at=now - RESET_AT_FAR_FUTURE_SECONDS,
                last_error_reset_at=now - 60,
            ),
        )

    entry = pool.select()
    assert entry is not None
    assert entry.last_status == "ok"
    assert not pool._codex_probe_honored_at, "marker should not survive a real recovery"
