"""``rotate-session`` control verb: external, event-driven rotation of one chat/thread (#125590).

Covers the three contracts the issue asked for:
  1. the verb rotates the targeted chat/thread — new session id returned, old row ended with
     ``end_reason="session_reset"`` (real socket, real SessionStore, real SQLite rows);
  2. a second channel on the same gateway is untouched;
  3. an unknown chat/thread answers ``rotated: false`` with no error and no state change.
Plus the /new-parity invariants an external rotation must not break: generation bump, cached-agent
eviction, conversation-scope clear and the session-boundary hooks all fire for the rotated key only.
"""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.control_socket import GatewayControlServer, query_gateway_control
from gateway.run_session_rotate import rotate_session_verb
from gateway.session import SessionSource, SessionStore

pytestmark = pytest.mark.platforms("posix")  # Unix-socket transport, like test_control_socket.py


def _source(chat_id: str, thread_id=None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="group", thread_id=thread_id)


class _FunnelRecorder:
    """Fake runner surface: only what the verb touches, recording funnel calls per key."""

    def __init__(self, store: SessionStore):
        self.session_store = store
        self.calls: dict[str, list[str]] = {}

    def _record(self, key: str, name: str) -> None:
        self.calls.setdefault(key, []).append(name)

    def _session_key_for_source(self, source: SessionSource) -> str:
        return self.session_store._generate_session_key(source)

    @property
    def async_session_store(self):
        store = self.session_store

        class _Async:
            async def reset_session(self, key):
                return await asyncio.to_thread(store.reset_session, key)

        return _Async()

    def _invalidate_session_run_generation(self, key, *, reason: str = "") -> int:
        self._record(key, "generation")
        return 1

    def _release_running_agent_state(self, key) -> None:
        self._record(key, "release")

    async def _cleanup_old_agent_for_reset(self, key) -> None:
        self._record(key, "cleanup")

    def _evict_cached_agent(self, key) -> None:
        self._record(key, "evict")

    def _clear_conversation_scope(self, key, *, reason: str) -> None:
        self._record(key, "scope")

    async def _fire_session_reset_hooks(self, source, key, old_sid, new_sid) -> None:
        self._record(key, f"hooks:{old_sid}:{new_sid}")

    def _is_telegram_topic_lane(self, source) -> bool:
        return False

    def _record_telegram_topic_binding(self, source, entry) -> None:  # pragma: no cover - lane off
        self._record(self._session_key_for_source(source), "topic")


def _make_store(tmp_path: Path) -> SessionStore:
    return SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


def test_rotate_session_rotates_target_and_ends_old_row(home: Path, tmp_path: Path):
    store = _make_store(tmp_path)
    source = _source("-100200")
    entry = store.get_or_create_session(source, force_new=True, touch_activity=False)
    old_sid = entry.session_id
    key = store._generate_session_key(source)

    async def scenario():
        runner = _FunnelRecorder(store)
        server = GatewayControlServer(
            home, verb_handlers={"rotate-session": rotate_session_verb(runner)})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: query_gateway_control(
                    home, "rotate-session", params={"platform": "telegram", "chat_id": "-100200"}))
            return result
        finally:
            await server.stop()

    result = asyncio.run(scenario())
    assert result is not None
    assert result["rotated"] is True
    assert result["old_session_id"] == old_sid
    assert result["end_reason"] == "session_reset"
    new_sid = result["new_session_id"]
    assert new_sid and new_sid != old_sid

    # Old row durably ended with the reset reason; new row created under the same key.
    db = cast(Any, store._db)
    assert db is not None
    old_row = db.get_session(old_sid)
    new_row = db.get_session(new_sid)
    assert old_row is not None and old_row.get("end_reason") == "session_reset"
    assert old_row.get("ended_at") is not None
    assert new_row is not None and new_row.get("session_key") == key

    # The live routing index now serves the fresh id (funnel parity is asserted in its own test).
    assert store.peek_session_id(key) == new_sid


def test_rotate_session_funnel_parity_matches_new(home: Path, tmp_path: Path):
    store = _make_store(tmp_path)
    source = _source("-100300")
    store.get_or_create_session(source, force_new=True, touch_activity=False)

    recorder: dict[str, list[str]] = {}

    async def scenario():
        class _Runner(_FunnelRecorder):
            def _record(self, key, name):
                recorder.setdefault(key, []).append(name)
                super()._record(key, name)

        runner = _Runner(store)
        server = GatewayControlServer(
            home, verb_handlers={"rotate-session": rotate_session_verb(runner)})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: query_gateway_control(
                    home, "rotate-session", params={"platform": "telegram", "chat_id": "-100300"}))
        finally:
            await server.stop()

    result = asyncio.run(scenario())
    assert result and result["rotated"] is True
    key = store._generate_session_key(source)
    calls = recorder.get(key, [])
    # Same order as /new: generation bump, slot release, cleanup, eviction, scope clear, hooks.
    assert [c.split(":")[0] for c in calls] == [
        "generation", "release", "cleanup", "evict", "scope", "hooks"]
    assert no_other_keys(recorder, key)


def no_other_keys(recorder: dict, key: str) -> bool:
    return all(k == key for k in recorder)


def test_rotate_session_fires_on_session_reset_plugin_hook(home: Path, tmp_path: Path, monkeypatch):
    """/new parity: after the new session exists, plugins learn of the rotation through the
    same ``on_session_reset`` lifecycle hook /new fires (old id -> new id). A miss (nothing
    routed) fires nothing — there is no rotation to report."""
    import hermes_cli.lifecycle as lifecycle

    store = _make_store(tmp_path)
    source = _source("-100400")
    entry = store.get_or_create_session(source, force_new=True, touch_activity=False)
    old_sid = entry.session_id

    hook_calls: list[dict] = []
    real_invoke = lifecycle.invoke_hook

    def spy_invoke(name, **kwargs):
        if name == "on_session_reset":
            hook_calls.append(kwargs)
        return real_invoke(name, **kwargs)

    monkeypatch.setattr(lifecycle, "invoke_hook", spy_invoke)

    async def scenario(params: dict):
        runner = _FunnelRecorder(store)
        server = GatewayControlServer(
            home, verb_handlers={"rotate-session": rotate_session_verb(runner)})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "rotate-session", params=params))
        finally:
            await server.stop()

    result = asyncio.run(scenario({"platform": "telegram", "chat_id": "-100400"}))
    assert result and result["rotated"] is True
    assert len(hook_calls) == 1
    call = hook_calls[0]
    assert call["old_session_id"] == old_sid
    assert call["new_session_id"] == result["new_session_id"] != old_sid
    assert call["session_id"] == result["new_session_id"]
    assert call["platform"] == "telegram"

    # Unknown chat: rotated False, no session created, so no hook event either.
    miss = asyncio.run(scenario({"platform": "telegram", "chat_id": "-100401"}))
    assert miss is not None and miss["rotated"] is False
    assert len(hook_calls) == 1


def test_rotate_session_leaves_second_channel_untouched(home: Path, tmp_path: Path):
    store = _make_store(tmp_path)
    other = _source("-100999")
    other_entry = store.get_or_create_session(other, force_new=True, touch_activity=False)
    other_key = store._generate_session_key(other)
    target = _source("-100111")
    store.get_or_create_session(target, force_new=True, touch_activity=False)

    async def scenario():
        runner = _FunnelRecorder(store)
        server = GatewayControlServer(
            home, verb_handlers={"rotate-session": rotate_session_verb(runner)})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: query_gateway_control(
                    home, "rotate-session", params={"platform": "telegram", "chat_id": "-100111"}))
        finally:
            await server.stop()

    result = asyncio.run(scenario())
    assert result and result["rotated"] is True
    # The other channel's live id and DB row are both untouched.
    assert store.peek_session_id(other_key) == other_entry.session_id
    other_row = cast(Any, store._db).get_session(other_entry.session_id)
    assert other_row is not None and other_row.get("ended_at") is None


def test_rotate_session_unknown_chat_is_miss_not_error(home: Path, tmp_path: Path):
    store = _make_store(tmp_path)

    async def scenario():
        runner = _FunnelRecorder(store)
        server = GatewayControlServer(
            home, verb_handlers={"rotate-session": rotate_session_verb(runner)})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: query_gateway_control(
                    home, "rotate-session", params={"platform": "telegram", "chat_id": "-424242"}))
        finally:
            await server.stop()

    result = asyncio.run(scenario())
    assert result is not None
    assert result["rotated"] is False
    assert "error" not in result
    # No session materialized for the polled key: a miss creates nothing.
    assert store.has_any_sessions() is False


def test_rotate_session_missing_params_report_error(home: Path, tmp_path: Path):
    store = _make_store(tmp_path)

    async def scenario():
        runner = _FunnelRecorder(store)
        server = GatewayControlServer(
            home, verb_handlers={"rotate-session": rotate_session_verb(runner)})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            no_platform = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "rotate-session", params={"chat_id": "1"}))
            bad_platform = await loop.run_in_executor(
                None, lambda: query_gateway_control(
                    home, "rotate-session", params={"platform": "nope", "chat_id": "1"}))
            no_chat = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "rotate-session", params={"platform": "telegram"}))
            return no_platform, bad_platform, no_chat
        finally:
            await server.stop()

    no_platform, bad_platform, no_chat = asyncio.run(scenario())
    for result in (no_platform, bad_platform, no_chat):
        assert result is not None and result.get("error")
    assert "platform" in cast(dict, no_platform)["error"]
    assert "nope" in cast(dict, bad_platform)["error"]
    assert "chat_id" in cast(dict, no_chat)["error"]


def test_rotate_session_thread_key_targets_thread_only(home: Path, tmp_path: Path):
    """A thread_id rotation must not rotate the parent chat's own session."""
    store = _make_store(tmp_path)
    chat = _source("-100500")
    chat_entry = store.get_or_create_session(chat, force_new=True, touch_activity=False)
    thread = _source("-100500", thread_id="77")
    thread_entry = store.get_or_create_session(thread, force_new=True, touch_activity=False)

    async def scenario():
        runner = _FunnelRecorder(store)
        server = GatewayControlServer(
            home, verb_handlers={"rotate-session": rotate_session_verb(runner)})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: query_gateway_control(
                    home, "rotate-session",
                    params={"platform": "telegram", "chat_id": "-100500", "thread_id": "77"}))
        finally:
            await server.stop()

    result = asyncio.run(scenario())
    assert result and result["rotated"] is True
    assert result["old_session_id"] == thread_entry.session_id
    assert result["new_session_id"] != thread_entry.session_id
    # The parent chat's session is untouched.
    assert store.peek_session_id(store._generate_session_key(chat)) == chat_entry.session_id
