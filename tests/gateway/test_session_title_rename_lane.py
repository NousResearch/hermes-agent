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
        _is_discord_auto_thread_lane=lambda src: lane == "discord",
        _is_relay_discord_channel_lane=lambda src: False,
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
        holder, agent, types.SimpleNamespace(source=source)
    )
    return agent._on_session_title, renames


@pytest.mark.parametrize("lane", ["telegram", "discord"])
def test_the_rename_waits_for_the_model_title(lane):
    callback, renames = _attach(lane)

    callback("fix the flaky auth test in log", "derived")
    assert renames == []

    callback("Fix flaky auth test", "llm")
    assert renames == ["Fix flaky auth test"]


def _recovery_runner(scheduled):
    """Fake runner binding the REAL lane predicate, like NativeRenameRunner below."""

    class RecoveryRunner:
        _is_telegram_topic_lane = lambda self, source: False  # noqa: E731
        _is_relay_discord_channel_lane = lambda self, source: False  # noqa: E731
        _is_discord_auto_thread_lane = GatewayRunner._is_discord_auto_thread_lane

        def _schedule_discord_semantic_thread_rename(self, source, session_id, title):
            scheduled.append((source, session_id, title))

    return RecoveryRunner()


def _recover_agent(session_db, origin):
    return types.SimpleNamespace(
        session_id="sess-1",
        _session_db=session_db
        or types.SimpleNamespace(
            get_session=lambda session_id: {
                "origin_json": json.dumps(origin.to_dict()) if origin else "{}",
            }
        ),
    )


def test_discord_title_retry_recovers_auto_thread_origin():
    """A fresh agent on a later thread turn still wires the semantic rename."""
    scheduled = []
    current = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        message_id="follow-up",
    )
    origin = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        message_id="opening-message",
        auto_thread_created=True,
        auto_thread_initial_name="Opening words",
    )
    runner = _recovery_runner(scheduled)
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = _recover_agent(None, origin)

    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=current)
    )
    agent._on_session_title("Recovered semantic title", "llm")

    assert len(scheduled) == 1
    recovered, session_id, title = scheduled[0]
    assert recovered.message_id == "follow-up"
    assert recovered.auto_thread_created is True
    assert recovered.auto_thread_initial_name == "Opening words"
    assert session_id == "sess-1"
    assert title == "Recovered semantic title"


@pytest.mark.parametrize(
    "origin_source",
    [
        pytest.param(None, id="no-origin-markers"),
        pytest.param(
            SessionSource(
                platform=Platform.DISCORD,
                chat_id="thread-1",
                chat_type="thread",
                thread_id="thread-2",
                message_id="opening-message",
                auto_thread_created=True,
                auto_thread_initial_name="Other thread",
            ),
            id="other-thread-id",
        ),
        pytest.param(
            SessionSource(
                platform=Platform.DISCORD,
                chat_id="thread-1",
                chat_type="thread",
                thread_id="thread-1",
                message_id="opening-message",
            ),
            id="user-created-thread",
        ),
    ],
)
def test_discord_title_retry_recovery_no_ops_for_untrusted_origins(origin_source):
    """Wrong thread, manual thread, or blank origin: the callback is never wired."""
    scheduled = []
    current = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        message_id="follow-up",
    )
    runner = _recovery_runner(scheduled)
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = _recover_agent(None, origin_source)

    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=current)
    )

    assert not hasattr(agent, "_on_session_title")
    assert scheduled == []


def test_discord_title_retry_recovery_survives_db_failure():
    """A broken session DB degrades to main's behaviour, not a wiring crash."""
    scheduled = []
    current = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        message_id="follow-up",
    )

    class BrokenDB:
        def get_session(self, session_id):
            raise RuntimeError("state.db unavailable")

    runner = types.SimpleNamespace(
        _is_telegram_topic_lane=lambda source: False,
        _is_discord_auto_thread_lane=GatewayRunner._is_discord_auto_thread_lane,
        _is_relay_discord_channel_lane=lambda source: False,
        _schedule_discord_semantic_thread_rename=(
            lambda source, session_id, title: scheduled.append(
                (source, session_id, title)
            )
        ),
    )
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = types.SimpleNamespace(
        session_id="sess-1", _session_db=BrokenDB(),
    )

    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=current)
    )

    # Recovery failed, the marker-less source stays off every lane: no callback,
    # exactly like main — and the failure never escaped the attach.
    assert not hasattr(agent, "_on_session_title")
    assert scheduled == []


def test_discord_title_retry_recovery_preserves_transport_adapter_ref():
    """The recovered source keeps the live event's transport owner (multiplex)."""
    scheduled = []
    adapter = type("Adapter", (), {})()
    current = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        message_id="follow-up",
        profile="runtime-profile",
    )
    current._transport_adapter_ref = weakref.ref(adapter)
    origin = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_type="thread",
        thread_id="thread-1",
        message_id="opening-message",
        auto_thread_created=True,
        auto_thread_initial_name="Opening words",
    )
    runner = _recovery_runner(scheduled)
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = _recover_agent(None, origin)

    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=current)
    )
    agent._on_session_title("Recovered semantic title", "llm")

    recovered = scheduled[0][0]
    assert recovered.message_id == "follow-up"
    assert recovered._transport_adapter_ref() is adapter


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
        return None

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
