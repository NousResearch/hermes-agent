"""Regression for #113690: a pinned async-delegation completion must not publish a route it
resolved across a session boundary.

``_resolve_async_delegation_session`` awaits the spawning session's DB row, so by the time the
lookup returns the route may have moved: ``/new`` or ``/stop`` revoked the run generation, or a
concurrent ``/resume`` repointed the key. Both are boundaries the resolution snapshotted before —
the stale pin must fail closed instead of overwriting the newer route.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _runner_with_suspended_lookup(tmp_path, entered, release, pinned_row=None):
    """A GatewayRunner over a real SessionStore whose spawned-session lookup blocks."""
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore, SessionSource, SessionStore

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="test-chat",
        chat_type="dm",
        user_id="test-user",
    )
    entry = store.get_or_create_session(source)

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)

    row = pinned_row if pinned_row is not None else {"ended_at": None}

    async def get_session(session_id):
        entered.set()
        await release.wait()
        return {"id": session_id, **row}

    runner._session_db = SimpleNamespace(get_session=AsyncMock(side_effect=get_session))
    return runner, store, entry


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["none", "revoke", "replace"])
async def test_pending_pin_respects_concurrent_boundary(tmp_path, boundary):
    """The pin only moves the route while the run and the route are both still current."""
    entered, release = asyncio.Event(), asyncio.Event()
    runner, store, entry = _runner_with_suspended_lookup(tmp_path, entered, release)
    generation = runner._begin_session_run_generation(entry.session_key)
    task = asyncio.create_task(
        runner._resolve_async_delegation_session(entry, "test-pinned"),
    )
    await asyncio.wait_for(entered.wait(), 3)

    expected = "test-pinned"
    if boundary != "none":
        # A /stop or /new style boundary revokes the in-flight run token.
        runner._invalidate_session_run_generation(
            entry.session_key, reason="test boundary"
        )
        assert not runner._is_session_run_current(entry.session_key, generation)
        expected = entry.session_id
    if boundary == "replace":
        # A concurrent /resume repoints the key while the lookup is still suspended.
        store.switch_session(entry.session_key, "test-replacement")
        expected = "test-replacement"
        assert store.lookup_by_session_key(entry.session_key).session_id == expected

    release.set()
    result = await asyncio.wait_for(task, 3)

    assert store.lookup_by_session_key(entry.session_key).session_id == expected
    if boundary == "none":
        assert result.session_id == expected
    else:
        assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["identity", "compression"])
async def test_revocation_is_sampled_where_the_transition_commits(tmp_path, shape):
    """A boundary arriving at the commit is observed, not outrun by an earlier validation.

    The two shapes reach their effect by a different route than a plain repoint: the identity fast
    path hands back the entry it already has, and the compression path advances the route through
    ``advance_compression_session``. Validating the run token once after the lookup and then acting
    would let a ``/stop`` landing in that second window publish anyway. The transition samples the
    token under the authority the bump takes, so a boundary arriving at that moment still wins.
    """
    entered, release = asyncio.Event(), asyncio.Event()
    pinned_row = {"ended_at": "2026-07-08T00:00:00", "end_reason": "idle"}
    if shape == "compression":
        pinned_row["end_reason"] = "compression"
    runner, store, entry = _runner_with_suspended_lookup(
        tmp_path,
        entered,
        release,
        pinned_row=pinned_row,
    )
    if shape == "compression":
        runner._resolve_compression_lineage_target = AsyncMock(return_value="sess_tip")
    generation = runner._begin_session_run_generation(entry.session_key)

    task = asyncio.create_task(
        runner._resolve_async_delegation_session(entry, "test-pinned"),
    )
    await asyncio.wait_for(entered.wait(), 3)

    # Inject the boundary at the instant the transition samples the run token. A shape that had
    # validated earlier and mutated without re-sampling would publish the stale pin here.
    seen = []
    real_is_current = runner._is_session_run_current

    def invalidate_as_the_commit_samples(key, gen):
        if not seen:
            seen.append(True)
            runner._invalidate_session_run_generation(
                key,
                reason="boundary at the commit",
            )
        return real_is_current(key, gen)

    with patch.object(
        runner, "_is_session_run_current", invalidate_as_the_commit_samples
    ):
        release.set()
        result = await asyncio.wait_for(task, 3)

    assert seen, "the transition never sampled the run token"
    assert result is None
    assert store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
    assert not runner._is_session_run_current(entry.session_key, generation)
