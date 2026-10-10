"""Which title stage is allowed to spend a platform rename.

Titling is two-stage: a derived slice of the user's own words lands inline, and
the model's version replaces it a moment later. A local sidebar wants both. A
Discord thread or a Telegram topic wants only the second — renaming twice lands
on the same name at twice the cost, and Discord allows two channel renames per
ten minutes, so the throwaway can be the one that survives.
"""

from __future__ import annotations

import json
import types
import weakref

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner


def _attach(lane):
    """Attach the title callback for *lane* and return (callback, renames)."""
    renames: list = []
    source = types.SimpleNamespace(platform=Platform.DISCORD, chat_id="chan-1")

    runner = types.SimpleNamespace(
        _is_telegram_topic_lane=lambda src: lane == "telegram",
        _is_discord_thread_lane=lambda src: lane == "discord",
        _is_discord_auto_thread_lane=lambda src: lane == "discord",
        _is_relay_discord_channel_lane=lambda src: False,
        _recover_discord_auto_thread_source=lambda src, key: src,
        _schedule_telegram_topic_title_rename=(
            lambda src, sid, title: renames.append(title)
        ),
        _schedule_discord_semantic_thread_rename=(
            lambda src, sid, title: renames.append(title)
        ),
    )
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = types.SimpleNamespace(session_id="sess-1")
    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=source, session_key="key-1")
    )
    return agent._on_session_title, renames


@pytest.mark.parametrize("lane", ["telegram", "discord"])
def test_the_rename_waits_for_the_model_title(lane):
    callback, renames = _attach(lane)

    callback("fix the flaky auth test in log", "derived")
    assert renames == []

    callback("Fix flaky auth test", "llm")
    assert renames == ["Fix flaky auth test"]


def test_user_created_discord_thread_registers_title_rename():
    callback, renames = _attach("discord")

    # The callback helper represents an existing user-created thread; unlike a
    # bot-created thread it has no auto-thread markers to consult.
    callback("Rename this thread", "llm")

    assert renames == ["Rename this thread"]


def _thread_source(thread_id="thread-1", **fields):
    return SessionSource(
        platform=Platform.DISCORD, chat_id=thread_id, chat_type="thread", thread_id=thread_id, **fields,
    )


_OPENING = _thread_source(
    message_id="opening-message", auto_thread_created=True, auto_thread_initial_name="Opening words",
)


def _attach_in_thread(current, entry_origin, row_origin=None, get_session=None):
    """Attach the title callback for an unmarked in-thread follow-up *current* whose routing entry
    holds *entry_origin* (and whose session row holds *row_origin*), through the REAL lane
    predicate and recovery. Returns (agent, scheduled renames)."""
    scheduled: list = []
    entry = types.SimpleNamespace(session_id="sess-1", origin=entry_origin)
    row = {"origin_json": json.dumps(row_origin.to_dict())} if row_origin else None
    db = types.SimpleNamespace(get_session=get_session or {"sess-1": row}.get)
    runner = types.SimpleNamespace(
        session_store=types.SimpleNamespace(
            lookup_by_session_key={"key-1": entry}.get, _db_for_key={"key-1": db}.get,
        ),
        _is_telegram_topic_lane=lambda src: False,
        _is_relay_discord_channel_lane=lambda src: False,
        _schedule_discord_semantic_thread_rename=lambda src, sid, title: scheduled.append((src, sid, title)),
    )
    for name in ("_is_discord_thread_lane", "_is_discord_auto_thread_lane", "_recover_discord_auto_thread_source"):
        setattr(runner, name, types.MethodType(getattr(GatewayRunner, name), runner))
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = types.SimpleNamespace(session_id="sess-1")
    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=current, session_key="key-1")
    )
    return agent, scheduled


@pytest.mark.parametrize("rebuilt_entry", [False, True], ids=["routing-entry", "rebuilt-entry-falls-back-to-row"])
def test_discord_title_retry_recovers_auto_thread_origin(rebuilt_entry):
    """A fresh agent on a later in-thread turn still wires the semantic rename (#127667), keeping
    the live event's message id and transport owner."""
    adapter = type("Adapter", (), {})()
    current = _thread_source(message_id="follow-up", profile="runtime-profile")
    current._transport_adapter_ref = weakref.ref(adapter)
    if rebuilt_entry:  # routing index lost: the entry was rebuilt from an unmarked event
        agent, scheduled = _attach_in_thread(current, entry_origin=_thread_source(), row_origin=_OPENING)
    else:
        agent, scheduled = _attach_in_thread(current, entry_origin=_OPENING)
    agent._on_session_title("Recovered semantic title", "llm")

    [(recovered, session_id, title)] = scheduled
    assert recovered.message_id == "follow-up"
    assert recovered._transport_adapter_ref() is adapter
    assert (recovered.auto_thread_created, recovered.auto_thread_initial_name) == (True, "Opening words")
    assert (session_id, title) == ("sess-1", "Recovered semantic title")


_UNREADABLE_CALLS: list = []


def _unreadable(session_id):
    _UNREADABLE_CALLS.append(session_id)
    raise RuntimeError("state.db unavailable")


@pytest.mark.parametrize("entry_origin,row_origin,get_session,expect_callback", [
    pytest.param(
        _thread_source("thread-2", auto_thread_created=True, auto_thread_initial_name="Other"),
        _thread_source("thread-2", auto_thread_created=True, auto_thread_initial_name="Other"),
        None, True, id="other-thread",
    ),
    pytest.param(_thread_source(message_id="opening-message"), None, None, True, id="user-created-thread"),
    pytest.param(_thread_source(), None, _unreadable, False, id="unreadable-row"),
])
def test_discord_title_retry_never_borrows_markers_from_another_origin(entry_origin, row_origin, get_session, expect_callback):
    _UNREADABLE_CALLS.clear()
    agent, scheduled = _attach_in_thread(
        _thread_source(message_id="follow-up"), entry_origin, row_origin, get_session,
    )
    if not expect_callback:  # the row read ran; attach's try/except drops the callback, as on main
        assert not hasattr(agent, "_on_session_title")
        assert scheduled == []
        assert _UNREADABLE_CALLS == ["sess-1"]
        return
    # With the broadened thread lane (#127841) an existing native thread earns the semantic
    # rename even when recovery cannot attach auto-thread markers — but never borrowed ones:
    # the scheduled rename targets THIS thread and carries no foreign initial name.
    assert hasattr(agent, "_on_session_title")
    agent._on_session_title("Recovered semantic title", "llm")
    [(src, session_id, title)] = scheduled
    assert str(src.thread_id) == "thread-1"
    assert getattr(src, "auto_thread_created", None) is not True
    assert getattr(src, "auto_thread_initial_name", None) != "Other"
    assert (session_id, title) == ("sess-1", "Recovered semantic title")


@pytest.mark.anyio
async def test_native_thread_rename_passes_only_the_initial_name_guard():
    """The shared rename lane must honor the strict native adapter contract."""
    calls: list[tuple[str, str, str | None]] = []

    class StrictNativeAdapter:
        async def rename_thread(
            self,
            thread_id: str,
            name: str,
            *,
            only_if_current_name: str | None = None,
        ) -> bool:
            calls.append((thread_id, name, only_if_current_name))
            return True

    class NativeRenameRunner:
        _is_discord_thread_lane = GatewayRunner._is_discord_thread_lane
        _is_discord_auto_thread_lane = GatewayRunner._is_discord_auto_thread_lane
        _sanitize_discord_thread_title = GatewayRunner._sanitize_discord_thread_title
        _rename_discord_auto_thread_for_session_title = (
            GatewayRunner._rename_discord_auto_thread_for_session_title
        )

        def __init__(self, adapter):
            self.adapters = {Platform.DISCORD: adapter}

        def _delivery_adapter_for(self, source):
            return self.adapters[source.platform]

    source = types.SimpleNamespace(
        platform=Platform.DISCORD,
        chat_id="999",
        chat_type="thread",
        thread_id="999",
        auto_thread_created=True,
        auto_thread_initial_name="Initial words",
    )

    runner = NativeRenameRunner(StrictNativeAdapter())
    await runner._rename_discord_auto_thread_for_session_title(
        source,
        "session-1",
        "Semantic Session Title",
    )

    assert calls == [("999", "Semantic Session Title", "Initial words")]


def test_title_thread_copy_preserves_transport_adapter_ref(monkeypatch):
    """Multiplex-routed sources must keep their transport owner for side effects."""
    captured_sources = []

    class Adapter:
        pass

    adapter = Adapter()

    async def noop():
        return None

    def fake_schedule(coro, loop, logger=None, log_message=None):
        coro.close()

    monkeypatch.setattr("gateway.run.safe_schedule_threadsafe", fake_schedule)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        profile="runtime-profile",
        auto_thread_created=True,
        auto_thread_initial_name="Initial words",
    )
    source._transport_adapter_ref = weakref.ref(adapter)

    runner = types.SimpleNamespace(
        _gateway_loop=types.SimpleNamespace(is_closed=lambda: False),
        _schedule_rename_from_title_thread=GatewayRunner._schedule_rename_from_title_thread,
    )

    runner._schedule_rename_from_title_thread(
        runner,
        source,
        lambda copied: captured_sources.append(copied) or noop(),
        "Discord semantic thread rename",
    )

    assert len(captured_sources) == 1
    copied = captured_sources[0]
    assert copied is not source
    assert copied.profile == "runtime-profile"
    assert copied._transport_adapter_ref() is adapter
