"""Strict internal delivery drops dead contexts and superseded runs before side effects."""

from typing import Any, cast

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource


def _strict_event(source, entry):
    return MessageEvent(
        text="injected", message_type=MessageType.TEXT, source=source, internal=True,
        allow_gateway_control=False,
        metadata={
            "gateway_session_key": entry.session_key,
            "gateway_session_id": entry.session_id,
            "gateway_session_strict": True,
        },
    )


@pytest.fixture
def strict_runner(tmp_path, monkeypatch):
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="strict-lane", chat_type="dm")
        entry = store.get_or_create_session(source)
        cached = []
        original_cache = runner._cache_session_source

        def record_cache(session_key, src):
            cached.append(session_key)
            return original_cache(session_key, src)

        monkeypatch.setattr(runner, "_cache_session_source", record_cache)
        try:
            yield runner, store, source, entry, cached
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()


def _add_turn(store, session_id):
    db = cast(Any, store._db)
    db.append_message(session_id, "user", "hello")
    db.append_message(session_id, "assistant", "hi")


@pytest.mark.asyncio
async def test_strict_event_on_live_context_resolves(strict_runner):
    runner, store, source, entry, cached = strict_runner
    _add_turn(store, entry.session_id)
    generation = runner._begin_session_run_generation(entry.session_key)

    resolved = await runner._hmwa_resolve_session(_strict_event(source, entry), source, generation)

    assert resolved is not None and resolved[1].session_id == entry.session_id
    assert cached == [entry.session_key]


@pytest.mark.asyncio
async def test_strict_event_on_empty_transcript_drops_before_side_effects(strict_runner):
    runner, _store, source, entry, cached = strict_runner
    generation = runner._begin_session_run_generation(entry.session_key)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source, generation) is None
    assert cached == []


@pytest.mark.asyncio
async def test_strict_event_on_ended_session_drops_before_side_effects(strict_runner):
    runner, store, source, entry, cached = strict_runner
    _add_turn(store, entry.session_id)
    cast(Any, store._db).end_session(entry.session_id, end_reason="user_reset")
    generation = runner._begin_session_run_generation(entry.session_key)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source, generation) is None
    assert cached == []


@pytest.mark.asyncio
async def test_failed_context_read_fails_closed(strict_runner, monkeypatch):
    runner, store, source, entry, cached = strict_runner
    _add_turn(store, entry.session_id)
    generation = runner._begin_session_run_generation(entry.session_key)

    def broken(_session_id):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(runner._session_db, "get_session_context_state", broken)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source, generation) is None
    assert cached == []


@pytest.mark.asyncio
async def test_superseded_run_drops_after_recheck_and_before_side_effects(strict_runner, monkeypatch):
    """A newer run that starts while the recheck is awaited must not reach the source cache."""
    runner, store, source, entry, cached = strict_runner
    _add_turn(store, entry.session_id)
    generation = runner._begin_session_run_generation(entry.session_key)
    lookup = runner.async_session_store.lookup_by_session_key

    async def lookup_then_supersede(session_key):
        found = await lookup(session_key)
        runner._begin_session_run_generation(session_key)
        return found

    monkeypatch.setattr(runner.async_session_store, "lookup_by_session_key", lookup_then_supersede)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source, generation) is None
    assert cached == []


@pytest.mark.asyncio
async def test_direct_callers_without_a_run_generation_keep_working(strict_runner):
    runner, store, source, entry, cached = strict_runner
    _add_turn(store, entry.session_id)

    resolved = await runner._hmwa_resolve_session(_strict_event(source, entry), source)

    assert resolved is not None and cached == [entry.session_key]
