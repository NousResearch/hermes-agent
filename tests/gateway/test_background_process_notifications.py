"""Tests for configurable background process notification modes.

The gateway process watcher pushes status updates to users' chats when
background terminal commands run.  ``display.background_process_notifications``
controls verbosity: off | result | error | all (default).

Contributed by @PeterFile (PR #593), reimplemented on current main.
"""

import asyncio
import queue
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _parse_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRegistry:
    """Return pre-canned sessions, then None once exhausted."""

    def __init__(self, sessions, consumed=False):
        self._sessions = list(sessions)
        self._consumed = consumed

    def get(self, session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, session_id):
        return self._consumed


def _build_runner(monkeypatch, tmp_path, mode: str) -> GatewayRunner:
    """Create a GatewayRunner with a fake config for the given mode."""
    (tmp_path / "config.yaml").write_text(
        f"display:\n  background_process_notifications: {mode}\n",
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner


def _watcher_dict(session_id="proc_test", thread_id=""):
    d = {
        "session_id": session_id,
        "check_interval": 0,
        "platform": "telegram",
        "chat_id": "123",
    }
    if thread_id:
        d["thread_id"] = thread_id
    return d


def _watch_event(session_id="proc_watch", thread_id="42"):
    return {
        "type": "watch_match",
        "session_id": session_id,
        "session_key": f"agent:main:telegram:dm:123:{thread_id}",
        "pattern": "READY",
        "command": "build",
        "output": "READY\n",
    }


# ---------------------------------------------------------------------------
# _load_background_notifications_mode unit tests
# ---------------------------------------------------------------------------

class TestLoadBackgroundNotificationsMode:

    def test_defaults_to_concise(self, monkeypatch, tmp_path):
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "concise"

    def test_unknown_mode_falls_back_to_concise(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: bogus\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "concise"

    def test_reads_config_yaml(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: error\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "error"


# ---------------------------------------------------------------------------
# _run_process_watcher integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumed_completion_skips_raw_notification(monkeypatch, tmp_path):
    """#65379: after process(wait) already returned the completion inline,
    the gateway watcher must NOT also push the raw
    "[Background process ... finished with exit code ...]" message.

    The agent-notify branch already honored _completion_consumed, but its
    skip fell through to the text-notification branch, double-delivering the
    same output to the chat (observed on Slack with
    background_process_notifications: all)."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="done\n", exited=True, exit_code=0, command="sleep 1; echo done",
    )]
    monkeypatch.setattr(
        pr_module, "process_registry", _FakeRegistry(sessions, consumed=True)
    )

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    # notify_on_complete=True mirrors the reported scenario: the watcher's
    # agent-notify skip must not fall through to a raw adapter.send().
    watcher = _watcher_dict()
    watcher["notify_on_complete"] = True
    await runner._run_process_watcher(watcher)

    adapter.send.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_consumed_completion_skips_raw_notification_without_agent_notify(
    monkeypatch, tmp_path
):
    """#65379 variant: same double-delivery guard for plain watchers
    (notify_on_complete=False) — wait/log consumption suppresses the raw
    completion message in every mode."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="done\n", exited=True, exit_code=0, command="echo done",
    )]
    monkeypatch.setattr(
        pr_module, "process_registry", _FakeRegistry(sessions, consumed=True)
    )

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_inject_watch_notification_routes_from_session_store_origin(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="123",
            user_name="Emiliyan",
        )
    )

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:group:-100:42",
    }

    await runner._inject_watch_notification("[SYSTEM: Background process matched]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.internal is True
    assert synth_event.source.platform == Platform.TELEGRAM
    assert synth_event.source.chat_id == "-100"
    assert synth_event.source.chat_type == "group"
    assert synth_event.source.thread_id == "42"
    assert synth_event.source.user_id == "123"
    assert synth_event.source.user_name == "Emiliyan"


@pytest.mark.asyncio
async def test_post_turn_watch_drain_off_consumes_without_injecting(monkeypatch, tmp_path):
    runner = _build_runner(monkeypatch, tmp_path, "off")
    adapter = runner.adapters[Platform.TELEGRAM]
    completion_queue = queue.Queue()
    completion_queue.put(_watch_event("proc_one"))
    completion_queue.put(_watch_event("proc_two"))
    async_event = {"type": "async_delegation", "session_id": "delegate_one"}
    completion_queue.put(async_event)

    await runner._drain_watch_notifications(completion_queue)

    adapter.handle_message.assert_not_awaited()
    assert completion_queue.qsize() == 1
    assert completion_queue.get_nowait() is async_event


@pytest.mark.asyncio
async def test_post_turn_watch_drain_all_injects_from_queued_event_origin(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:dm:123:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            thread_id="42",
            user_id="proc_owner",
            user_name="alice",
        )
    )
    completion_queue = queue.Queue()
    completion_queue.put(_watch_event())
    async_event = {"type": "async_delegation", "session_id": "delegate_one"}
    completion_queue.put(async_event)

    await runner._drain_watch_notifications(completion_queue)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.source.thread_id == "42"
    assert synth_event.source.user_id == "proc_owner"
    assert completion_queue.qsize() == 1
    assert completion_queue.get_nowait() is async_event


@pytest.mark.asyncio
async def test_inject_watch_notification_carries_message_id_reply_anchor(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:dm:123:24296"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            thread_id="24296",
            user_id="1",
            user_name="Fabio",
        )
    )

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:dm:123:24296",
        "message_id": "777",
    }

    await runner._inject_watch_notification("[SYSTEM: Background process matched]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.message_id == "777"
    assert synth_event.source.thread_id == "24296"


@pytest.mark.asyncio
async def test_inject_watch_notification_loads_session_store_off_loop(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:dm:123:24296"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            thread_id="24296",
            user_id="1",
            user_name="Fabio",
        )
    )
    loop_thread = threading.get_ident()
    load_threads = []
    real_ensure_loaded = runner.session_store._ensure_loaded

    def spy_ensure_loaded():
        load_threads.append(threading.get_ident())
        return real_ensure_loaded()

    monkeypatch.setattr(runner.session_store, "_ensure_loaded", spy_ensure_loaded)

    await runner._inject_watch_notification(
        "[SYSTEM: Background process matched]",
        {
            "session_id": "proc_watch",
            "session_key": "agent:main:telegram:dm:123:24296",
            "message_id": "777",
        },
    )

    adapter.handle_message.assert_awaited_once()
    assert load_threads
    assert all(thread_id != loop_thread for thread_id in load_threads)


@pytest.mark.asyncio
async def test_inject_watch_notification_ignores_foreground_event_source(monkeypatch, tmp_path):
    """Negative test: watch notification must NOT route to the foreground thread."""
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    # Session store has the process's original thread (thread 42)
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="proc_owner",
            user_name="alice",
        )
    )

    # The evt dict carries the correct session_key — NOT a foreground event
    evt = {
        "session_id": "proc_cross_thread",
        "session_key": "agent:main:telegram:group:-100:42",
    }

    await runner._inject_watch_notification("[SYSTEM: watch match]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    # Must route to thread 42 (process origin), NOT some other thread
    assert synth_event.source.thread_id == "42"
    assert synth_event.source.user_id == "proc_owner"


# ---------------------------------------------------------------------------
# concise mode — pretty one-liner instead of the raw output dump
# ---------------------------------------------------------------------------


class TestConciseFormatter:

    def test_success_is_one_line_without_output(self):
        from gateway.run import _format_concise_process_notification
        text = _format_concise_process_notification(
            "proc_abc", "python3 scan_fleet.py --all", 0,
            "1300\n1400\n1500\n{...huge json...}",
            duration_seconds=754,
        )
        assert text.startswith("✅ Background task finished")
        assert "scan_fleet.py" in text
        assert "12m 34s" in text
        # The raw output must NOT appear on success
        assert "1300" not in text
        assert "\n" not in text

    def test_failure_appends_short_tail(self):
        from gateway.run import _format_concise_process_notification
        out = "\n".join(f"line{i}" for i in range(50)) + "\nTraceback: boom"
        text = _format_concise_process_notification(
            "proc_abc", "make build", 2, out,
        )
        assert text.startswith("❌ Background task failed (exit 2)")
        assert "Traceback: boom" in text
        # Only a short tail, not the whole output
        assert "line0" not in text

    def test_long_command_is_truncated(self):
        from gateway.run import _format_concise_process_notification
        text = _format_concise_process_notification(
            "proc_abc", "x" * 300, 0, "",
        )
        assert "…" in text
        assert len(text) < 200


@pytest.mark.asyncio
async def test_concise_mode_sends_pretty_message_not_raw_dump(monkeypatch, tmp_path):
    """Default mode: a finished process produces the one-line status message,
    never the '[Background process ... Here's the final output: ...]' wall."""
    import tools.process_registry as pr_module

    big_output = "\n".join(str(i * 100) for i in range(60))
    sessions = [SimpleNamespace(
        output_buffer=big_output, exited=True, exit_code=0,
        command="python3 scan.py", started_at=None,
    )]
    monkeypatch.setattr(
        pr_module, "process_registry", _FakeRegistry(sessions, consumed=False)
    )

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "concise")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    adapter.send.assert_awaited_once()
    sent_text = adapter.send.await_args.args[1]
    assert sent_text.startswith("✅ Background task finished")
    assert "Here's the final output" not in sent_text
    assert "5000" not in sent_text


@pytest.mark.asyncio
async def test_concise_mode_failure_includes_tail(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="starting\nfatal: repo not found\n", exited=True,
        exit_code=128, command="git clone x", started_at=None,
    )]
    monkeypatch.setattr(
        pr_module, "process_registry", _FakeRegistry(sessions, consumed=False)
    )

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "concise")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    adapter.send.assert_awaited_once()
    sent_text = adapter.send.await_args.args[1]
    assert sent_text.startswith("❌ Background task failed (exit 128)")
    assert "fatal: repo not found" in sent_text


@pytest.mark.asyncio
async def test_concise_mode_no_interim_output_updates(monkeypatch, tmp_path):
    """concise never pushes 'is still running~ New output' interim updates."""
    import tools.process_registry as pr_module

    running = SimpleNamespace(
        output_buffer="chunk one\n", exited=False, exit_code=None,
        command="sleep 100", started_at=None,
    )
    done = SimpleNamespace(
        output_buffer="chunk one\nchunk two\n", exited=True, exit_code=0,
        command="sleep 100", started_at=None,
    )
    monkeypatch.setattr(
        pr_module, "process_registry", _FakeRegistry([running, done], consumed=False)
    )

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "concise")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    # Exactly one send: the final concise message; no interim updates.
    adapter.send.assert_awaited_once()
    sent_text = adapter.send.await_args.args[1]
    assert "is still running" not in sent_text
    assert sent_text.startswith("✅ Background task finished")


# ---------------------------------------------------------------------------
# _parse_session_key helper
# ---------------------------------------------------------------------------


def test_parse_session_key_with_extra_parts():
    """6th part in a group key may be a user_id, not a thread_id — omit it."""
    result = _parse_session_key("agent:main:discord:group:chan123:thread456")
    assert result == {"platform": "discord", "chat_type": "group", "chat_id": "chan123"}


# ---------------------------------------------------------------------------
# api_server (stateless) wake routing — gateway/wake.py self-post path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inject_watch_notification_raw_session_key_self_posts(monkeypatch, tmp_path):
    """An event whose session_key is a RAW api_server session id (not an
    agent:main:... structured key) must wake the real session via the
    /v1/chat/completions self-post instead of being dropped for missing
    routing metadata."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    api_adapter = SimpleNamespace(
        supports_async_delivery=False,
        handle_message=AsyncMock(),
        _host="127.0.0.1", _port=8642, _api_key="k", _model_name="m",
    )
    runner.adapters[Platform.API_SERVER] = api_adapter

    posts = []

    async def fake_self_post(adapter, *, text, session_id):
        posts.append({"text": text, "session_id": session_id})

    import gateway.wake as wake_mod
    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)

    evt = {
        "session_id": "proc_watch",
        "session_key": "raw-hq-session-id",  # no agent:main:... structure
    }
    result = await runner._inject_watch_notification("[SYSTEM: subagent finished]", evt)

    assert result is True
    api_adapter.handle_message.assert_not_awaited()
    assert posts == [
        {"text": "[SYSTEM: subagent finished]", "session_id": "raw-hq-session-id"}
    ]


@pytest.mark.asyncio
async def test_inject_watch_notification_origin_session_id_wins(monkeypatch, tmp_path):
    """origin_session_id (stamped at dispatch time by async_delegation) takes
    precedence as the wake target."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    api_adapter = SimpleNamespace(
        supports_async_delivery=False,
        handle_message=AsyncMock(),
        _host="127.0.0.1", _port=8642, _api_key="k", _model_name="m",
    )
    runner.adapters[Platform.API_SERVER] = api_adapter

    posts = []

    async def fake_self_post(adapter, *, text, session_id):
        posts.append(session_id)

    import gateway.wake as wake_mod
    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)

    evt = {
        "session_id": "proc_watch",
        "session_key": "",
        "origin_session_id": "raw-origin-sid",
    }
    result = await runner._inject_watch_notification("[SYSTEM: done]", evt)
    assert result is True
    assert posts == ["raw-origin-sid"]


@pytest.mark.asyncio
async def test_message_agent_completion_apiserver_persists_without_self_post(
    monkeypatch, tmp_path,
):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    api_adapter = SimpleNamespace(supports_async_delivery=False)
    runner.adapters[Platform.API_SERVER] = api_adapter

    import gateway.wake as wake_mod

    posts = []
    persisted = []

    async def fake_self_post(adapter, *, text, session_id):
        posts.append(session_id)

    async def fake_persist(adapter, *, text, session_id, profile_home, evt=None):
        persisted.append((text, session_id, profile_home, evt))

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    monkeypatch.setattr(wake_mod, "persist_message_agent_delivery", fake_persist)

    evt = {
        "type": "completion",
        "session_id": "proc_reply",
        "session_key": "raw-sender-sid",
        "platform": "api_server",
        "chat_type": "dm",
        "chat_id": "raw-sender-sid",
        "parent_session_id": "raw-sender-sid",
        "completion_delivery": "message_agent",
        "completion_delivery_home": str(tmp_path / "profiles" / "sender"),
    }
    result = await runner._inject_watch_notification("reply", evt)

    assert result is True
    assert posts == []
    assert persisted == [(
        "reply",
        "raw-sender-sid",
        str(tmp_path / "profiles" / "sender"),
        evt,
    )]


@pytest.mark.asyncio
async def test_async_delegation_apiserver_persists_delivery_not_self_post(
    monkeypatch, tmp_path,
):
    """#85957: an async_delegation completion targeting a stateless api_server
    session must be persisted as a durable DELIVERY row — never self-POSTed
    to /v1/chat/completions as a new role=user prompt (which starts an
    unauthorized agent turn after the client-owned parent turn ended)."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    api_adapter = SimpleNamespace(
        supports_async_delivery=False,
        handle_message=AsyncMock(),
        _host="127.0.0.1", _port=8642, _api_key="k", _model_name="m",
    )
    runner.adapters[Platform.API_SERVER] = api_adapter

    import gateway.wake as wake_mod

    posts = []

    async def fake_self_post(adapter, *, text, session_id):
        posts.append(session_id)

    persisted = []

    async def fake_persist(adapter, *, text, session_id, evt=None):
        persisted.append({"text": text, "session_id": session_id, "evt": evt})

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    monkeypatch.setattr(wake_mod, "persist_delegation_delivery", fake_persist)

    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_85957",
        "session_key": "raw-hq-session-id",  # no agent:main:... structure
        "origin_session_id": "raw-hq-session-id",
        "status": "completed",
    }
    result = await runner._inject_watch_notification(
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_85957]", evt,
    )

    assert result is True
    assert posts == []  # the self-POST user-turn wake must NOT fire
    api_adapter.handle_message.assert_not_awaited()
    assert len(persisted) == 1
    assert persisted[0]["session_id"] == "raw-hq-session-id"
    assert persisted[0]["evt"]["delegation_id"] == "deleg_85957"


@pytest.mark.asyncio
async def test_async_delegation_apiserver_persist_failure_is_retryable(
    monkeypatch, tmp_path,
):
    """A failed delivery persist returns False so the durable claim is
    released and the completion retried — never silently lost."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    api_adapter = SimpleNamespace(
        supports_async_delivery=False,
        handle_message=AsyncMock(),
        _host="127.0.0.1", _port=8642, _api_key="k", _model_name="m",
    )
    runner.adapters[Platform.API_SERVER] = api_adapter

    import gateway.wake as wake_mod

    async def fail_persist(adapter, *, text, session_id, evt=None):
        raise RuntimeError("state.db unavailable")

    monkeypatch.setattr(wake_mod, "persist_delegation_delivery", fail_persist)

    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_fail",
        "session_key": "raw-sid",
    }
    result = await runner._inject_watch_notification("[BATCH COMPLETE]", evt)
    assert result is False


def test_gateway_drain_retains_and_formats_overflow_events():
    """watch_overflow_* events must survive the gateway drain and render
    their summary — previously they were discarded at the drain (only
    watch_match/watch_disabled were retained) and had no formatter branch."""
    import asyncio
    from gateway.run import (
        _drain_gateway_watch_events,
        _format_gateway_process_notification,
    )

    queue = asyncio.Queue()
    tripped = {
        "type": "watch_overflow_tripped",
        "message": "watch flood detected: 47 notifications suppressed for pattern 'ERROR'",
        "session_id": "proc_a1b2",
    }
    released = {
        "type": "watch_overflow_released",
        "message": "watch flood released: notifications resumed for pattern 'ERROR'",
        "session_id": "proc_a1b2",
    }
    queue.put_nowait(tripped)
    queue.put_nowait(released)

    retained = _drain_gateway_watch_events(queue)
    assert retained == [tripped, released]

    out_tripped = _format_gateway_process_notification(tripped)
    assert "47 notifications suppressed" in out_tripped
    assert "exit code" not in out_tripped
    out_released = _format_gateway_process_notification(released)
    assert "notifications resumed" in out_released
    assert "exit code" not in out_released
