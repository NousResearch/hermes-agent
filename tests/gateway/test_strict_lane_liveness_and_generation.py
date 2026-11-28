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


def _add_turn(runner, session_id):
    from persistence.conversation.messages import NewMessage
    db = cast(Any, runner._session_db)
    db.messages.append(session_id, NewMessage.from_live({"role": "user", "content": "hello"}))
    db.messages.append(session_id, NewMessage.from_live({"role": "assistant", "content": "hi"}))


@pytest.mark.asyncio
async def test_strict_event_on_live_context_resolves(strict_runner):
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    runner._begin_session_run_generation(entry.session_key)

    resolved = await runner._hmwa_resolve_session(_strict_event(source, entry), source)

    assert resolved is not None and resolved[1].session_id == entry.session_id
    assert cached == [entry.session_key]


@pytest.mark.asyncio
async def test_strict_event_on_empty_transcript_drops_before_side_effects(strict_runner):
    runner, _store, source, entry, cached = strict_runner
    runner._begin_session_run_generation(entry.session_key)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []


@pytest.mark.asyncio
async def test_strict_event_on_ended_session_drops_before_side_effects(strict_runner):
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    cast(Any, runner._session_db).sessions.end(entry.session_id, "user_reset")
    runner._begin_session_run_generation(entry.session_key)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []


@pytest.mark.asyncio
async def test_failed_context_read_fails_closed(strict_runner, monkeypatch):
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    runner._begin_session_run_generation(entry.session_key)

    def broken(_session_id):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(runner._session_db.sessions, "list_row", broken)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []


@pytest.mark.asyncio
async def test_superseded_run_drops_after_recheck_and_before_side_effects(strict_runner, monkeypatch):
    """A newer run that starts while the recheck is awaited must not reach the source cache."""
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    runner._begin_session_run_generation(entry.session_key)
    lookup = runner.async_session_store.lookup_by_session_key

    async def lookup_then_supersede(session_key):
        found = await lookup(session_key)
        runner._begin_session_run_generation(session_key)
        return found

    monkeypatch.setattr(runner.async_session_store, "lookup_by_session_key", lookup_then_supersede)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []


@pytest.mark.asyncio
async def test_strict_event_before_any_run_resolves(strict_runner):
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)

    resolved = await runner._hmwa_resolve_session(_strict_event(source, entry), source)

    assert resolved is not None and cached == [entry.session_key]


@pytest.mark.asyncio
async def test_session_ended_during_awaited_lookup_drops_before_side_effects(strict_runner, monkeypatch):
    """The persisted row ends while the route lookup is suspended; the in-memory mapping and the
    run generation are unchanged, so only a persisted read taken after that await can see it."""
    runner, store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    generation = runner._begin_session_run_generation(entry.session_key)
    lookup = runner.async_session_store.lookup_by_session_key

    async def lookup_then_end(session_key):
        found = await lookup(session_key)
        cast(Any, runner._session_db).sessions.end(found.session_id, "user_reset")
        return found

    monkeypatch.setattr(runner.async_session_store, "lookup_by_session_key", lookup_then_end)

    assert store.peek_session_id(entry.session_key) == entry.session_id
    assert runner._is_session_run_current(entry.session_key, generation)
    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []


@pytest.mark.asyncio
async def test_session_still_live_after_awaited_lookup_resolves(strict_runner, monkeypatch):
    """Positive control for the ordering above: the same suspended lookup without a termination."""
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    runner._begin_session_run_generation(entry.session_key)
    lookup = runner.async_session_store.lookup_by_session_key

    async def lookup_only(session_key):
        return await lookup(session_key)

    monkeypatch.setattr(runner.async_session_store, "lookup_by_session_key", lookup_only)

    resolved = await runner._hmwa_resolve_session(_strict_event(source, entry), source)
    assert resolved is not None and cached == [entry.session_key]


@pytest.mark.asyncio
async def test_route_moved_during_final_read_drops_before_side_effects(strict_runner, monkeypatch):
    runner, store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    sessions = cast(Any, runner._session_db).sessions
    read = sessions.list_row
    moved = {"on": False}
    peek = store.peek_session_id

    def read_then_move(session_id):
        row = read(session_id)
        moved["on"] = True
        return row

    monkeypatch.setattr(sessions, "list_row", read_then_move)
    monkeypatch.setattr(store, "peek_session_id", lambda key: "another-session" if moved["on"] else peek(key))

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []


@pytest.mark.asyncio
async def test_run_superseded_during_final_read_drops_before_side_effects(strict_runner, monkeypatch):
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    runner._begin_session_run_generation(entry.session_key)
    sessions = cast(Any, runner._session_db).sessions
    read = sessions.list_row

    def read_then_supersede(session_id):
        row = read(session_id)
        runner._begin_session_run_generation(entry.session_key)
        return row

    monkeypatch.setattr(sessions, "list_row", read_then_supersede)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []


@pytest.mark.asyncio
async def test_missing_persisted_row_fails_closed(strict_runner, monkeypatch):
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    monkeypatch.setattr(cast(Any, runner._session_db).sessions, "list_row", lambda _session_id: None)

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []


@pytest.mark.asyncio
async def test_missing_session_db_fails_closed(strict_runner):
    runner, _store, source, entry, cached = strict_runner
    _add_turn(runner, entry.session_id)
    runner._session_db = None

    assert await runner._hmwa_resolve_session(_strict_event(source, entry), source) is None
    assert cached == []
