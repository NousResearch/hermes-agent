"""AccountSlot: per-account refresh serialization, terminal errors, and
redacted status. Synthetic grants, fake stores, injected refresh functions —
no Keychain, no network."""

from __future__ import annotations

import asyncio
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time

import pytest

from agent.oauth_broker.account_slot import (
    DEFAULT_REFRESH_SKEW_SECONDS,
    AccountSlot,
    SlotRefreshError,
    account_process_lock,
)
from agent.oauth_broker.models import GrantStoreError, OAuthGrant

NOW = 1_800_000_000.0  # fixed synthetic clock epoch

ACCESS_1 = "synthetic-access-1"
REFRESH_1 = "synthetic-refresh-1"
ACCESS_2 = "synthetic-access-2"
REFRESH_2 = "synthetic-refresh-2"


def _acquire_account_lock_in_child(state_dir, started, acquired):
    started.set()
    with account_process_lock(Path(state_dir), "A"):
        acquired.set()


class FakeGrantStore:
    """Dict-backed stand-in for KeychainGrantStore, recording events."""

    def __init__(self, grants=None, events=None):
        self.grants = dict(grants or {})
        self.events = events if events is not None else []
        self.load_count = 0

    def load(self, alias):
        self.load_count += 1
        self.events.append(f"load:{alias}")
        try:
            return self.grants[alias]
        except KeyError:
            raise GrantStoreError(
                alias=alias, category="not_found", detail="no grant provisioned"
            ) from None

    def replace(self, alias, grant):
        self.events.append(f"replace:{alias}:{grant.refresh_token}")
        self.grants[alias] = grant

    def delete(self, alias):
        self.grants.pop(alias, None)


class FakeAuthError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(f"synthetic auth failure ({code})")


def _grant(alias_suffix="1", *, expires_at=NOW + 3600.0):
    return OAuthGrant(
        access_token=f"synthetic-access-{alias_suffix}",
        refresh_token=f"synthetic-refresh-{alias_suffix}",
        expires_at=expires_at,
        account_id="acct-synthetic-a",
    )


def _slot(
    tmp_path,
    *,
    alias="A",
    store=None,
    refresh_fn=None,
    clock=lambda: NOW,
):
    store = store if store is not None else FakeGrantStore({alias: _grant()})
    calls = []

    def default_refresh(access_token, refresh_token):
        calls.append((access_token, refresh_token))
        return {
            "access_token": ACCESS_2,
            "refresh_token": REFRESH_2,
            "expires_at": NOW + 7200.0,
        }

    slot = AccountSlot(
        alias,
        grant_store=store,
        state_dir=tmp_path,
        refresh_fn=refresh_fn or default_refresh,
        clock=clock,
    )
    return slot, store, calls


# ---------------------------------------------------------------------------
# Skew behavior
# ---------------------------------------------------------------------------


def test_no_refresh_when_token_outside_skew(tmp_path):
    slot, _store, calls = _slot(tmp_path)
    token = asyncio.run(slot.get_access_token())
    assert token == ACCESS_1
    assert calls == []


def test_proactive_refresh_inside_skew(tmp_path):
    expiring = FakeGrantStore(
        {"A": _grant(expires_at=NOW + DEFAULT_REFRESH_SKEW_SECONDS - 1)}
    )
    slot, store, calls = _slot(tmp_path, store=expiring)
    token = asyncio.run(slot.get_access_token())
    assert token == ACCESS_2
    assert calls == [(ACCESS_1, REFRESH_1)]
    assert store.grants["A"].refresh_token == REFRESH_2


# ---------------------------------------------------------------------------
# Refresh mechanics
# ---------------------------------------------------------------------------


def test_refresh_rereads_keychain_after_lock_and_uses_fresh_refresh_token(tmp_path):
    """The refresh call must consume the refresh token read under the lock,
    not any value captured before lock acquisition."""
    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)})
    observed = []

    def refresh_fn(access_token, refresh_token):
        observed.append(refresh_token)
        return {
            "access_token": ACCESS_2,
            "refresh_token": REFRESH_2,
            "expires_at": NOW + 7200.0,
        }

    slot, store, _ = _slot(tmp_path, store=store, refresh_fn=refresh_fn)

    async def run():
        # Simulate another process rotating the grant on disk before our
        # coroutine acquires the slot lock: swap the stored refresh token.
        store.grants["A"] = OAuthGrant(
            access_token=ACCESS_1,
            refresh_token="synthetic-refresh-rotated-elsewhere",
            expires_at=NOW - 10,
            account_id="acct-synthetic-a",
        )
        return await slot.get_access_token()

    asyncio.run(run())
    assert observed == ["synthetic-refresh-rotated-elsewhere"]


def test_rotated_grant_written_to_store_before_token_returned(tmp_path):
    events = []
    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)}, events=events)

    def refresh_fn(access_token, refresh_token):
        events.append("refresh")
        return {
            "access_token": ACCESS_2,
            "refresh_token": REFRESH_2,
            "expires_at": NOW + 7200.0,
        }

    slot, _store, _ = _slot(tmp_path, store=store, refresh_fn=refresh_fn)

    async def run():
        token = await slot.get_access_token()
        events.append("returned")
        return token

    token = asyncio.run(run())
    assert token == ACCESS_2
    refresh_idx = events.index("refresh")
    replace_idx = events.index(f"replace:A:{REFRESH_2}")
    returned_idx = events.index("returned")
    assert refresh_idx < replace_idx < returned_idx


def test_twenty_concurrent_requests_cause_exactly_one_refresh(tmp_path):
    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)})
    refresh_count = 0

    def refresh_fn(access_token, refresh_token):
        nonlocal refresh_count
        refresh_count += 1
        time.sleep(0.02)  # widen the race window across the thread hop
        return {
            "access_token": ACCESS_2,
            "refresh_token": REFRESH_2,
            "expires_at": NOW + 7200.0,
        }

    slot, _store, _ = _slot(tmp_path, store=store, refresh_fn=refresh_fn)

    async def run():
        return await asyncio.gather(
            *(slot.get_access_token() for _ in range(20))
        )

    tokens = asyncio.run(run())
    assert tokens == [ACCESS_2] * 20
    assert refresh_count == 1


def test_concurrent_forced_refreshes_deduplicate(tmp_path):
    """Two 401 handlers forcing a refresh at once must consume the refresh
    token only once (single-use upstream semantics)."""
    store = FakeGrantStore({"A": _grant()})
    refresh_count = 0

    def refresh_fn(access_token, refresh_token):
        nonlocal refresh_count
        refresh_count += 1
        time.sleep(0.02)
        return {
            "access_token": ACCESS_2,
            "refresh_token": REFRESH_2,
            "expires_at": NOW + 7200.0,
        }

    slot, _store, _ = _slot(tmp_path, store=store, refresh_fn=refresh_fn)

    async def run():
        return await asyncio.gather(slot.force_refresh(), slot.force_refresh())

    tokens = asyncio.run(run())
    assert tokens == [ACCESS_2, ACCESS_2]
    assert refresh_count == 1


def test_delayed_401_for_stale_access_token_does_not_rotate_again(tmp_path):
    store = FakeGrantStore({"A": _grant()})
    slot, _store, calls = _slot(tmp_path, store=store)

    async def run():
        stale = await slot.get_access_token()
        first = await slot.refresh_after_unauthorized(stale)
        delayed = await slot.refresh_after_unauthorized(stale)
        return stale, first, delayed

    assert asyncio.run(run()) == (ACCESS_1, ACCESS_2, ACCESS_2)
    assert calls == [(ACCESS_1, REFRESH_1)]


# ---------------------------------------------------------------------------
# Terminal and transient failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code", ["invalid_grant", "refresh_token_reused", "token_revoked"]
)
def test_terminal_oauth_errors_mark_slot_terminal(tmp_path, code):
    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)})
    call_count = 0

    def refresh_fn(access_token, refresh_token):
        nonlocal call_count
        call_count += 1
        raise FakeAuthError(code)

    slot, _store, _ = _slot(tmp_path, store=store, refresh_fn=refresh_fn)

    async def run():
        with pytest.raises(SlotRefreshError) as excinfo:
            await slot.get_access_token()
        assert excinfo.value.terminal is True
        assert excinfo.value.category == code
        # Terminal slots never retry the refresh endpoint.
        with pytest.raises(SlotRefreshError) as second:
            await slot.get_access_token()
        assert second.value.terminal is True

    asyncio.run(run())
    assert call_count == 1
    status = slot.status()
    assert status.healthy is False
    assert status.terminal_category == code


def test_refresh_timeout_is_terminal_when_rotation_outcome_is_unknown(tmp_path):
    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)})

    def refresh_fn(access_token, refresh_token):
        raise TimeoutError("synthetic timeout")

    slot, _store, _ = _slot(tmp_path, store=store, refresh_fn=refresh_fn)

    async def run():
        with pytest.raises(SlotRefreshError) as excinfo:
            await slot.get_access_token()
        assert excinfo.value.terminal is True
        assert excinfo.value.category == "refresh_outcome_unknown"
        assert slot.status().healthy is False
        with pytest.raises(SlotRefreshError) as second:
            await slot.get_access_token()
        assert second.value.category == "refresh_outcome_unknown"

    asyncio.run(run())
    assert (tmp_path / "recovery" / "A.json").exists()


def test_explicit_retryable_refresh_response_clears_preflight_marker(tmp_path):
    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)})
    behavior = ["explicit-error"]

    def refresh_fn(access_token, refresh_token):
        if behavior:
            behavior.pop()
            raise FakeAuthError("codex_refresh_failed")
        return {
            "access_token": ACCESS_2,
            "refresh_token": REFRESH_2,
            "expires_at": NOW + 7200.0,
        }

    slot, _store, _ = _slot(tmp_path, store=store, refresh_fn=refresh_fn)

    async def run():
        with pytest.raises(SlotRefreshError) as excinfo:
            await slot.get_access_token()
        assert excinfo.value.terminal is False
        assert excinfo.value.category == "codex_refresh_failed"
        assert not (tmp_path / "recovery" / "A.json").exists()
        return await slot.get_access_token()

    assert asyncio.run(run()) == ACCESS_2


def test_missing_grant_is_fail_closed_but_not_terminal(tmp_path):
    slot, _store, calls = _slot(tmp_path, store=FakeGrantStore({}))

    async def run():
        with pytest.raises(SlotRefreshError) as excinfo:
            await slot.get_access_token()
        assert excinfo.value.terminal is False
        assert excinfo.value.category == "not_found"

    asyncio.run(run())
    assert calls == []


# ---------------------------------------------------------------------------
# Cross-account independence
# ---------------------------------------------------------------------------


def test_a_refresh_lock_never_blocks_b(tmp_path):
    release_a = threading.Event()

    def blocking_refresh(access_token, refresh_token):
        release_a.wait(timeout=5)
        return {
            "access_token": ACCESS_2,
            "refresh_token": REFRESH_2,
            "expires_at": NOW + 7200.0,
        }

    slot_a, _sa, _ = _slot(
        tmp_path,
        alias="A",
        store=FakeGrantStore({"A": _grant(expires_at=NOW - 10)}),
        refresh_fn=blocking_refresh,
    )
    slot_b, _sb, calls_b = _slot(
        tmp_path, alias="B", store=FakeGrantStore({"B": _grant()})
    )

    async def run():
        a_task = asyncio.create_task(slot_a.get_access_token())
        await asyncio.sleep(0.05)  # let A enter its refresh
        token_b = await asyncio.wait_for(slot_b.get_access_token(), timeout=1)
        release_a.set()
        token_a = await asyncio.wait_for(a_task, timeout=5)
        return token_a, token_b

    token_a, token_b = asyncio.run(run())
    assert token_a == ACCESS_2
    assert token_b == ACCESS_1
    assert calls_b == []


def test_terminal_a_does_not_affect_b(tmp_path):
    def failing_refresh(access_token, refresh_token):
        raise FakeAuthError("invalid_grant")

    slot_a, _sa, _ = _slot(
        tmp_path,
        alias="A",
        store=FakeGrantStore({"A": _grant(expires_at=NOW - 10)}),
        refresh_fn=failing_refresh,
    )
    slot_b, _sb, _ = _slot(
        tmp_path, alias="B", store=FakeGrantStore({"B": _grant()})
    )

    async def run():
        with pytest.raises(SlotRefreshError):
            await slot_a.get_access_token()
        return await slot_b.get_access_token()

    assert asyncio.run(run()) == ACCESS_1


# ---------------------------------------------------------------------------
# Advisory lock files and redacted status/logs
# ---------------------------------------------------------------------------


def test_refresh_creates_per_alias_lock_file_with_owner_only_mode(tmp_path):
    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, _ = _slot(tmp_path, store=store)
    asyncio.run(slot.get_access_token())
    lock_path = tmp_path / "locks" / "A.lock"
    assert lock_path.exists()
    assert (lock_path.stat().st_mode & 0o777) == 0o600


def test_account_lock_rejects_planted_symlink_without_chmodding_target(tmp_path):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"synthetic-victim")
    victim.chmod(0o644)
    (lock_dir / "A.lock").symlink_to(victim)
    slot = AccountSlot("A", grant_store=FakeGrantStore({}), state_dir=tmp_path)

    with pytest.raises(OSError):
        slot._acquire_file_lock()

    assert victim.read_bytes() == b"synthetic-victim"
    assert (victim.stat().st_mode & 0o777) == 0o644


def test_account_process_lock_serializes_a_spawned_process(tmp_path):
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    acquired = context.Event()
    process = context.Process(
        target=_acquire_account_lock_in_child,
        args=(str(tmp_path), started, acquired),
    )

    with account_process_lock(tmp_path, "A"):
        process.start()
        assert started.wait(timeout=3.0)
        assert not acquired.wait(timeout=0.2)

    assert acquired.wait(timeout=3.0)
    process.join(timeout=3.0)
    assert process.exitcode == 0


def test_keyboard_interrupt_after_flock_closes_lock_fd(tmp_path, monkeypatch):
    import agent.oauth_broker.account_slot as account_slot_mod

    slot, _store, _calls = _slot(tmp_path)
    real_flock = account_slot_mod.fcntl.flock
    captured = []

    def interrupt_after_lock(fd, operation):
        real_flock(fd, operation)
        if operation & fcntl.LOCK_EX:
            captured.append(fd)
            raise KeyboardInterrupt

    monkeypatch.setattr(account_slot_mod.fcntl, "flock", interrupt_after_lock)
    with pytest.raises(KeyboardInterrupt):
        slot._acquire_file_lock()

    assert captured
    with pytest.raises(OSError):
        os.fstat(captured[0])


def test_cancelled_async_lock_waiter_does_not_orphan_flock(tmp_path):
    slot, _store, _calls = _slot(tmp_path)
    held_fd = slot._acquire_file_lock()

    async def scenario():
        task = asyncio.create_task(slot.force_refresh())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        slot._release_file_lock(held_fd)
        await asyncio.sleep(0.1)

    asyncio.run(scenario())

    probe_fd = os.open(slot._lock_path, os.O_RDWR)
    try:
        fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe_fd, fcntl.LOCK_UN)
    finally:
        os.close(probe_fd)


def test_invalid_alias_rejected(tmp_path):
    with pytest.raises(ValueError):
        AccountSlot(
            "D",
            grant_store=FakeGrantStore({}),
            state_dir=tmp_path,
            refresh_fn=lambda a, r: {},
        )


# ---------------------------------------------------------------------------
# Refresh-persistence invariant (design §八.7): rotated grant survives a
# Keychain outage as the in-memory authority
# ---------------------------------------------------------------------------


class FlakyGrantStore(FakeGrantStore):
    """Grant store whose replace() fails while fail_replace is set."""

    def __init__(self, grants=None, *, fail_replace=True):
        super().__init__(grants)
        self.fail_replace = fail_replace
        self.replace_calls = 0

    def replace(self, alias, grant):
        self.replace_calls += 1
        if self.fail_replace:
            raise GrantStoreError(
                alias=alias, category="os_error", detail="synthetic keychain outage"
            )
        super().replace(alias, grant)


def _persistence_retry_attempts():
    from agent.oauth_broker import account_slot as slot_mod

    return slot_mod.PERSISTENCE_RETRY_ATTEMPTS


def _persistence_retry_cooldown():
    from agent.oauth_broker import account_slot as slot_mod

    return slot_mod.PERSISTENCE_RETRY_COOLDOWN_SECONDS


def test_replace_failure_serves_rotated_token_with_bounded_retries(tmp_path):
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, calls = _slot(tmp_path, store=store)
    token = asyncio.run(slot.get_access_token())
    # The rotated grant is served from memory — never raised away, never lost.
    assert token == ACCESS_2
    assert calls == [(ACCESS_1, REFRESH_1)]
    # Persistence was retried a bounded number of times, then gave up.
    assert store.replace_calls == _persistence_retry_attempts()
    status = slot.status()
    assert status.persistence_degraded is True
    assert status.healthy is True  # degraded, not terminal
    assert status.last_refresh_result == "ok"


def test_degraded_persistence_retries_are_coalesced_across_request_burst(tmp_path):
    box = {"now": NOW}
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, _ = _slot(tmp_path, store=store, clock=lambda: box["now"])

    async def run():
        await slot.get_access_token()
        assert store.replace_calls == _persistence_retry_attempts()
        await asyncio.gather(*(slot.get_access_token() for _ in range(20)))
        within_cooldown = store.replace_calls
        box["now"] += _persistence_retry_cooldown() + 1
        await asyncio.gather(*(slot.get_access_token() for _ in range(20)))
        return within_cooldown, store.replace_calls

    within_cooldown, after_one_due_batch = asyncio.run(run())
    assert within_cooldown == _persistence_retry_attempts()
    assert after_one_due_batch == _persistence_retry_attempts() * 2


def test_recovery_marker_blocks_restart_from_reusing_stale_refresh_token(tmp_path):
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, _ = _slot(tmp_path, store=store)
    assert asyncio.run(slot.get_access_token()) == ACCESS_2

    marker = tmp_path / "recovery" / "A.json"
    assert marker.exists()
    assert marker.stat().st_mode & 0o777 == 0o600
    marker_text = marker.read_text()
    for secret in (ACCESS_1, REFRESH_1, ACCESS_2, REFRESH_2):
        assert secret not in marker_text

    restarted_refreshes = []

    def restarted_refresh(access_token, refresh_token):
        restarted_refreshes.append(refresh_token)
        return {
            "access_token": "must-not-be-used",
            "refresh_token": "must-not-be-used",
            "expires_at": NOW + 7200,
        }

    restarted = AccountSlot(
        "A",
        grant_store=store,
        state_dir=tmp_path,
        refresh_fn=restarted_refresh,
        clock=lambda: NOW,
    )
    with pytest.raises(SlotRefreshError) as excinfo:
        asyncio.run(restarted.get_access_token())
    assert excinfo.value.category == "persistence_recovery_required"
    assert excinfo.value.terminal is True
    assert restarted_refreshes == []


def test_preloaded_stale_grant_is_blocked_before_fast_path(tmp_path):
    stale = _grant(expires_at=NOW + 9000)
    store = FakeGrantStore({"A": stale})
    marker_writer = AccountSlot(
        "A", grant_store=store, state_dir=tmp_path, clock=lambda: NOW
    )
    marker_writer._write_recovery_marker(stale.refresh_token)

    restarted = AccountSlot(
        "A",
        grant_store=store,
        state_dir=tmp_path,
        initial_grant=stale,
        clock=lambda: NOW,
    )

    status = restarted.status()
    assert status.healthy is False
    assert status.terminal_category == "persistence_recovery_required"
    with pytest.raises(SlotRefreshError) as excinfo:
        asyncio.run(restarted.get_access_token())
    assert excinfo.value.category == "persistence_recovery_required"
    assert store.load_count == 0


def test_recovery_marker_symlink_fails_closed_without_touching_target(tmp_path):
    recovery_dir = tmp_path / "recovery"
    recovery_dir.mkdir()
    victim = tmp_path / "victim-marker.json"
    victim_content = json.dumps(
        {
            "version": 1,
            "state": "refresh_persistence_pending",
            "stale_refresh_fingerprint": "sha256:" + "0" * 64,
        }
    )
    victim.write_text(victim_content, encoding="utf-8")
    (recovery_dir / "A.json").symlink_to(victim)

    slot = AccountSlot(
        "A",
        grant_store=FakeGrantStore({"A": _grant(expires_at=NOW + 9000)}),
        state_dir=tmp_path,
        initial_grant=_grant(expires_at=NOW + 9000),
        clock=lambda: NOW,
    )

    assert slot.status().terminal_category == "persistence_recovery_required"
    assert victim.read_text(encoding="utf-8") == victim_content
    assert (recovery_dir / "A.json").is_symlink()


def test_recovery_marker_keyboard_interrupt_cleans_fd_and_temp(tmp_path, monkeypatch):
    import agent.oauth_broker.account_slot as account_slot_mod

    slot = AccountSlot("A", grant_store=FakeGrantStore({}), state_dir=tmp_path)
    real_write = account_slot_mod.os.write
    captured = []

    def write_then_interrupt(fd, data):
        captured.append(fd)
        real_write(fd, data)
        raise KeyboardInterrupt

    monkeypatch.setattr(account_slot_mod.os, "write", write_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        slot._write_recovery_marker(REFRESH_1)

    assert captured
    with pytest.raises(OSError):
        os.fstat(captured[0])
    recovery_dir = tmp_path / "recovery"
    assert list(recovery_dir.glob(".A.json.*.tmp")) == []
    assert not (recovery_dir / "A.json").exists()


def test_recovery_marker_write_rejects_symlink_directory(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "recovery").symlink_to(external, target_is_directory=True)
    slot = AccountSlot("A", grant_store=FakeGrantStore({}), state_dir=tmp_path)

    with pytest.raises(OSError):
        slot._write_recovery_marker(REFRESH_1)

    assert list(external.iterdir()) == []


def test_preloaded_external_reauth_grant_clears_stale_marker(tmp_path):
    stale = _grant(expires_at=NOW + 9000)
    external = OAuthGrant(
        access_token="synthetic-access-preloaded-external",
        refresh_token="synthetic-refresh-preloaded-external",
        expires_at=NOW + 9000,
        account_id="acct-synthetic-a",
    )
    store = FakeGrantStore({"A": external})
    marker_writer = AccountSlot(
        "A", grant_store=store, state_dir=tmp_path, clock=lambda: NOW
    )
    marker_writer._write_recovery_marker(stale.refresh_token)

    restarted = AccountSlot(
        "A",
        grant_store=store,
        state_dir=tmp_path,
        initial_grant=external,
        clock=lambda: NOW,
    )

    assert restarted.status().healthy is True
    assert asyncio.run(restarted.get_access_token()) == external.access_token
    assert not (tmp_path / "recovery" / "A.json").exists()
    assert store.load_count == 0


def test_external_reauth_wins_over_degraded_in_memory_pending_grant(tmp_path):
    box = {"now": NOW}
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, _ = _slot(tmp_path, store=store, clock=lambda: box["now"])
    assert asyncio.run(slot.get_access_token()) == ACCESS_2

    external = OAuthGrant(
        access_token="synthetic-access-external-reauth",
        refresh_token="synthetic-refresh-external-reauth",
        expires_at=NOW + 9000,
        account_id="acct-synthetic-a",
    )
    store.grants["A"] = external
    store.fail_replace = False
    box["now"] += _persistence_retry_cooldown() + 1

    token = asyncio.run(slot.get_access_token())
    assert token == external.access_token
    assert store.grants["A"] == external
    assert not (tmp_path / "recovery" / "A.json").exists()


def test_external_reauth_wins_when_degraded_pending_grant_has_expired(tmp_path):
    box = {"now": NOW}
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, calls = _slot(tmp_path, store=store, clock=lambda: box["now"])
    assert asyncio.run(slot.get_access_token()) == ACCESS_2

    external = OAuthGrant(
        access_token="synthetic-access-human-login",
        refresh_token="synthetic-refresh-human-login",
        expires_at=NOW + 9000,
        account_id="acct-synthetic-a",
    )
    store.grants["A"] = external
    store.fail_replace = False
    box["now"] += 7201

    token = asyncio.run(slot.get_access_token())

    assert token == external.access_token
    assert store.grants["A"] == external
    assert calls == [(ACCESS_1, REFRESH_1)]


def test_external_reauth_wins_over_bare_force_refresh_from_degraded_slot(tmp_path):
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, calls = _slot(tmp_path, store=store)
    assert asyncio.run(slot.get_access_token()) == ACCESS_2

    external = OAuthGrant(
        access_token="synthetic-access-human-force-login",
        refresh_token="synthetic-refresh-human-force-login",
        expires_at=NOW + 9000,
        account_id="acct-synthetic-a",
    )
    store.grants["A"] = external
    store.fail_replace = False
    replace_calls_before_force = store.replace_calls

    token = asyncio.run(slot.force_refresh())

    assert token == external.access_token
    assert store.grants["A"] == external
    assert store.replace_calls == replace_calls_before_force
    assert calls == [(ACCESS_1, REFRESH_1)]
    assert slot.status().persistence_degraded is False


def test_external_logout_is_not_overwritten_by_expired_degraded_pending(tmp_path):
    box = {"now": NOW}
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, calls = _slot(tmp_path, store=store, clock=lambda: box["now"])
    assert asyncio.run(slot.get_access_token()) == ACCESS_2
    del store.grants["A"]
    store.fail_replace = False
    box["now"] += 7201

    with pytest.raises(SlotRefreshError) as excinfo:
        asyncio.run(slot.get_access_token())

    assert excinfo.value.category == "not_found"
    assert excinfo.value.terminal is False
    assert "A" not in store.grants
    assert calls == [(ACCESS_1, REFRESH_1)]
    assert slot.status().present is False
    assert slot.status().healthy is True


def test_old_pending_process_cannot_overwrite_reauth_after_new_process_clears_marker(
    tmp_path,
):
    box = {"now": NOW}
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    old_slot, _store, _ = _slot(
        tmp_path, store=store, clock=lambda: box["now"]
    )
    assert asyncio.run(old_slot.get_access_token()) == ACCESS_2

    external = OAuthGrant(
        access_token="synthetic-access-external-new-process",
        refresh_token="synthetic-refresh-external-new-process",
        expires_at=NOW + 9000,
        account_id="acct-synthetic-a",
    )
    store.grants["A"] = external
    store.fail_replace = False

    # A newly started broker sees the external grant, proves that the old
    # marker is stale, and removes it.
    new_slot = AccountSlot(
        "A",
        grant_store=store,
        state_dir=tmp_path,
        clock=lambda: box["now"],
    )
    assert asyncio.run(new_slot.get_access_token()) == external.access_token
    assert not (tmp_path / "recovery" / "A.json").exists()

    replace_calls_before_old_retry = store.replace_calls
    box["now"] += _persistence_retry_cooldown() + 1
    assert asyncio.run(old_slot.get_access_token()) == external.access_token
    assert store.grants["A"] == external
    assert store.replace_calls == replace_calls_before_old_retry


def test_degraded_slot_rereads_but_never_reuses_consumed_refresh_token(tmp_path):
    box = {"now": NOW}
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    observed = []

    def refresh_fn(access_token, refresh_token):
        observed.append(refresh_token)
        n = len(observed)
        return {
            "access_token": f"synthetic-access-r{n}",
            "refresh_token": f"synthetic-refresh-r{n}",
            "expires_at": box["now"] + 7200.0,
        }

    slot, _store, _ = _slot(
        tmp_path, store=store, refresh_fn=refresh_fn, clock=lambda: box["now"]
    )

    async def run():
        first = await slot.get_access_token()
        assert first == "synthetic-access-r1"
        box["now"] += 7200.0  # the rotated access token now needs refresh too
        second = await slot.get_access_token()
        assert second == "synthetic-access-r2"

    asyncio.run(run())
    # The second refresh re-read Keychain only to compare the stale-token
    # fingerprint, then consumed the in-memory rotated token rather than the
    # already-consumed original.
    assert observed == [REFRESH_1, "synthetic-refresh-r1"]
    assert store.load_count == 2
    assert store.grants["A"].refresh_token == REFRESH_1  # keychain still stale
    assert slot.status().persistence_degraded is True


def test_later_request_retries_persistence_and_recovers(tmp_path):
    box = {"now": NOW}
    store = FlakyGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, _ = _slot(tmp_path, store=store, clock=lambda: box["now"])

    async def run():
        await slot.get_access_token()
        assert slot.status().persistence_degraded is True
        store.fail_replace = False  # keychain heals
        box["now"] += _persistence_retry_cooldown() + 1
        return await slot.get_access_token()  # due retry persists cached grant

    token = asyncio.run(run())
    assert token == ACCESS_2
    assert store.grants["A"].refresh_token == REFRESH_2
    assert slot.status().persistence_degraded is False
    assert store.replace_calls == _persistence_retry_attempts() + 1


def test_blocked_keychain_load_for_a_does_not_block_b_event_loop(tmp_path):
    load_started = threading.Event()
    release_load = threading.Event()
    b_completed_before_release = []

    class BlockingStore(FakeGrantStore):
        def load(self, alias):
            load_started.set()
            release_load.wait(timeout=2.0)
            return super().load(alias)

    store_a = BlockingStore({"A": _grant("a", expires_at=NOW + 3600)})
    slot_a = AccountSlot(
        "A", grant_store=store_a, state_dir=tmp_path, clock=lambda: NOW
    )
    slot_b = AccountSlot(
        "B",
        grant_store=FakeGrantStore(),
        state_dir=tmp_path,
        clock=lambda: NOW,
        initial_grant=_grant("b", expires_at=NOW + 3600),
    )

    async def run():
        loop = asyncio.get_running_loop()

        def controller():
            assert load_started.wait(timeout=1.0)
            future = asyncio.run_coroutine_threadsafe(
                slot_b.get_access_token(), loop
            )
            try:
                b_completed_before_release.append(
                    future.result(timeout=0.10) == "synthetic-access-b"
                )
            except TimeoutError:
                b_completed_before_release.append(False)
            finally:
                release_load.set()

        thread = threading.Thread(target=controller)
        thread.start()
        token_a = await slot_a.get_access_token()
        await asyncio.sleep(0)  # run B if it was queued behind a blocked loop
        thread.join(timeout=1.0)
        return token_a

    token_a = asyncio.run(run())

    assert b_completed_before_release == [True]
    assert token_a == "synthetic-access-a"


def test_initial_grant_preloads_slot_without_store_reads(tmp_path):
    store = FakeGrantStore({"A": _grant()})
    slot = AccountSlot(
        "A",
        grant_store=store,
        state_dir=tmp_path,
        refresh_fn=lambda a, r: {},
        clock=lambda: NOW,
        initial_grant=_grant(),
    )
    assert slot.status().present is True
    token = asyncio.run(slot.get_access_token())
    assert token == ACCESS_1
    assert store.load_count == 0  # served from the preloaded grant


def test_status_is_redacted_and_logs_contain_no_secrets(tmp_path, caplog):
    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, _ = _slot(tmp_path, store=store)
    with caplog.at_level("DEBUG"):
        token = asyncio.run(slot.get_access_token())
    assert token == ACCESS_2

    status = slot.status()
    assert status.alias == "A"
    assert status.present is True
    assert status.healthy is True
    assert status.expires_at == NOW + 7200.0
    assert status.last_refresh_result == "ok"
    assert isinstance(status.access_token_fingerprint, str)
    assert status.access_token_fingerprint.startswith("sha256:")

    rendered = repr(status) + caplog.text
    for secret in (ACCESS_1, REFRESH_1, ACCESS_2, REFRESH_2):
        assert secret not in rendered


def test_untrusted_relogin_error_code_is_normalized_before_log_and_status(
    tmp_path, caplog
):
    leaked_code = "synthetic-secret-like-code\nforged-log-line"

    class ReloginError(RuntimeError):
        code = leaked_code
        relogin_required = True

    def refresh_fn(access_token, refresh_token):
        raise ReloginError("synthetic upstream detail")

    store = FakeGrantStore({"A": _grant(expires_at=NOW - 10)})
    slot, _store, _ = _slot(tmp_path, store=store, refresh_fn=refresh_fn)
    with caplog.at_level("DEBUG"):
        with pytest.raises(SlotRefreshError) as excinfo:
            asyncio.run(slot.get_access_token())

    assert excinfo.value.category == "auth_relogin_required"
    status = slot.status()
    assert status.terminal_category == "auth_relogin_required"
    rendered = str(excinfo.value) + repr(status) + caplog.text
    assert leaked_code not in rendered
    assert "forged-log-line" not in rendered
