"""Tests for configurable background process notification modes.

The gateway process watcher pushes status updates to users' chats when
background terminal commands run.  ``display.background_process_notifications``
controls verbosity: off | result | error | all (default).

Contributed by @PeterFile (PR #593), reimplemented on current main.
"""

import asyncio
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

    def __init__(self, sessions):
        self._sessions = list(sessions)

    def get(self, session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, session_id):
        return False


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


# ---------------------------------------------------------------------------
# _load_background_notifications_mode unit tests
# ---------------------------------------------------------------------------

class TestLoadBackgroundNotificationsMode:

    def test_defaults_to_all(self, monkeypatch, tmp_path):
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "all"

    def test_reads_config_yaml(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: error\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "error"

    def test_env_var_overrides_config(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: error\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "off")
        assert GatewayRunner._load_background_notifications_mode() == "off"

    def test_false_value_maps_to_off(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: false\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "off"

    def test_invalid_value_defaults_to_all(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: banana\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "all"


# ---------------------------------------------------------------------------
# _run_process_watcher integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "sessions", "expected_calls", "expected_fragment"),
    [
        # all mode: running output → sends update
        (
            "all",
            [
                SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None),
                None,  # process disappears → watcher exits
            ],
            1,
            "is still running",
        ),
        # result mode: running output → no update
        (
            "result",
            [
                SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None),
                None,
            ],
            0,
            None,
        ),
        # off mode: exited process → no notification
        (
            "off",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            0,
            None,
        ),
        # result mode: exited → notifies
        (
            "result",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            1,
            "completed normally (exit code 0)",
        ),
        # error mode: exit 0 → no notification
        (
            "error",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            0,
            None,
        ),
        # error mode: exit 1 → notifies
        (
            "error",
            [SimpleNamespace(output_buffer="traceback\n", exited=True, exit_code=1)],
            1,
            "exited (exit code 1)",
        ),
        # all mode: exited → notifies
        (
            "all",
            [SimpleNamespace(output_buffer="ok\n", exited=True, exit_code=0)],
            1,
            "completed normally (exit code 0)",
        ),
    ],
)
async def test_run_process_watcher_respects_notification_mode(
    monkeypatch, tmp_path, mode, sessions, expected_calls, expected_fragment
):
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    # Patch asyncio.sleep to avoid real delays
    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, mode)
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    assert adapter.send.await_count == expected_calls, (
        f"mode={mode}: expected {expected_calls} sends, got {adapter.send.await_count}"
    )
    if expected_fragment is not None:
        sent_message = adapter.send.await_args.args[1]
        assert expected_fragment in sent_message


@pytest.mark.asyncio
async def test_thread_id_passed_to_send(monkeypatch, tmp_path):
    """thread_id from watcher dict is forwarded as metadata to adapter.send()."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict(thread_id="42"))

    assert adapter.send.await_count == 1
    _, kwargs = adapter.send.call_args
    assert kwargs["metadata"] == {"thread_id": "42"}


@pytest.mark.asyncio
async def test_no_thread_id_sends_no_metadata(monkeypatch, tmp_path):
    """When thread_id is empty, metadata should be None (general topic)."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    assert adapter.send.await_count == 1
    _, kwargs = adapter.send.call_args
    assert kwargs["metadata"] is None


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
async def test_agent_notification_carries_message_id_reply_anchor(monkeypatch, tmp_path):
    """notify_on_complete injection carries the triggering message_id so the
    synthetic event can be reply-anchored back into a Telegram DM topic.

    Without an anchor, Telegram private-chat topic sends fall back to the main
    chat (see _thread_kwargs_for_send / telegram_dm_topic_reply_fallback)."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="SMOKE_OK\n", exited=True, exit_code=0, command="sleep 1",
    )]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    watcher = {
        "session_id": "proc_anchor",
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123:24296",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "24296",
        "message_id": "555",
        "notify_on_complete": True,
    }
    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.internal is True
    assert synth_event.message_id == "555"
    assert synth_event.source.thread_id == "24296"


@pytest.mark.asyncio
async def test_agent_notification_no_message_id_is_tolerated(monkeypatch, tmp_path):
    """A watcher dict without message_id (CLI spawn, pre-upgrade checkpoint)
    still injects — message_id is simply None."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="done\n", exited=True, exit_code=0, command="sleep 1",
    )]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    watcher = {
        "session_id": "proc_anchorless",
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123:24296",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "24296",
        "notify_on_complete": True,
    }
    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.message_id is None


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


def test_build_process_event_source_falls_back_to_session_key_chat_type(monkeypatch, tmp_path):
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:group:-100:42",
        "platform": "telegram",
        "chat_id": "-100",
        "thread_id": "42",
        "user_id": "123",
        "user_name": "Emiliyan",
    }

    source = runner._build_process_event_source(evt)

    assert source is not None
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100"
    assert source.chat_type == "group"
    assert source.thread_id == "42"
    assert source.user_id == "123"
    assert source.user_name == "Emiliyan"


def test_build_process_event_source_uses_cached_live_source_before_session_key_parse(
    monkeypatch, tmp_path
):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._cache_session_source(
        "agent:main:telegram:group:-100:42",
        SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="proc_owner",
            user_name="alice",
        ),
    )

    source = runner._build_process_event_source(
        {
            "session_id": "proc_watch",
            "session_key": "agent:main:telegram:group:-100:42",
        }
    )

    assert source is not None
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100"
    assert source.chat_type == "group"
    assert source.thread_id == "42"
    assert source.user_id == "proc_owner"
    assert source.user_name == "alice"


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


def test_build_process_event_source_returns_none_for_empty_evt(monkeypatch, tmp_path):
    """Missing session_key and no platform metadata → None (drop notification)."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    source = runner._build_process_event_source({"session_id": "proc_orphan"})
    assert source is None


def test_build_process_event_source_returns_none_for_invalid_platform(monkeypatch, tmp_path):
    """Invalid platform string → None."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_bad",
        "platform": "not_a_real_platform",
        "chat_type": "dm",
        "chat_id": "123",
    }
    source = runner._build_process_event_source(evt)
    assert source is None


def test_build_process_event_source_returns_none_for_short_session_key(monkeypatch, tmp_path):
    """Session key with <5 parts doesn't parse, falls through to empty metadata → None."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_short",
        "session_key": "agent:main:telegram",  # Too few parts
    }
    source = runner._build_process_event_source(evt)
    assert source is None


# ---------------------------------------------------------------------------
# _parse_session_key helper
# ---------------------------------------------------------------------------

def test_parse_session_key_valid():
    result = _parse_session_key("agent:main:telegram:group:-100")
    assert result == {"platform": "telegram", "chat_type": "group", "chat_id": "-100"}


def test_parse_session_key_with_extra_parts():
    """6th part in a group key may be a user_id, not a thread_id — omit it."""
    result = _parse_session_key("agent:main:discord:group:chan123:thread456")
    assert result == {"platform": "discord", "chat_type": "group", "chat_id": "chan123"}


def test_parse_session_key_with_user_id_part():
    """Group keys with per-user isolation have user_id as 6th part — don't return as thread_id."""
    result = _parse_session_key("agent:main:telegram:group:chat1:user99")
    assert result == {"platform": "telegram", "chat_type": "group", "chat_id": "chat1"}


def test_parse_session_key_dm_with_thread():
    """DM keys use parts[5] as thread_id unambiguously."""
    result = _parse_session_key("agent:main:telegram:dm:chat1:topic42")
    assert result == {"platform": "telegram", "chat_type": "dm", "chat_id": "chat1", "thread_id": "topic42"}


def test_parse_session_key_thread_chat_type():
    """Thread-typed keys use parts[5] as thread_id unambiguously."""
    result = _parse_session_key("agent:main:discord:thread:chan1:thread99")
    assert result == {"platform": "discord", "chat_type": "thread", "chat_id": "chan1", "thread_id": "thread99"}


def test_parse_session_key_too_short():
    assert _parse_session_key("agent:main:telegram") is None
    assert _parse_session_key("") is None


def test_parse_session_key_wrong_prefix():
    assert _parse_session_key("cron:main:telegram:dm:123") is None
    assert _parse_session_key("agent:cron:telegram:dm:123") is None


# ---------------------------------------------------------------------------
# Chat-facing notice content: output omitted, states retained, commands redacted
# ---------------------------------------------------------------------------

from gateway.run import (  # noqa: E402
    _format_process_completion_notice,
    _format_process_running_notice,
)

_SECRET = "sk-abc123456789012345678901234567890123"
_SECRET_CMD = f'curl -H "Authorization: Bearer {_SECRET}" https://api.example.com/v1'


class TestProcessNoticeContent:

    def test_completion_notice_omits_output_and_points_at_log(self):
        notice = _format_process_completion_notice(
            session_id="proc_1", command="npm run build", exit_code=0
        )
        assert "proc_1 completed normally (exit code 0)" in notice
        assert "Command: npm run build" in notice
        assert 'process(action="log", session_id="proc_1")' in notice

    def test_completion_notice_retains_kill_diagnostics(self):
        notice = _format_process_completion_notice(
            session_id="proc_2",
            command="sleep 600",
            exit_code=-15,
            completion_reason="killed",
            termination_source="process.kill",
        )
        assert "terminated by process.kill" in notice
        assert "exit code -15, SIGTERM" in notice

    def test_completion_notice_reports_lost_backend(self):
        notice = _format_process_completion_notice(
            session_id="proc_3",
            command="sleep 600",
            exit_code=None,
            completion_reason="lost",
        )
        assert "marked lost because the process backend disappeared" in notice

    def test_completion_notice_redacts_command_credentials(self):
        notice = _format_process_completion_notice(
            session_id="proc_4", command=_SECRET_CMD, exit_code=1
        )
        assert _SECRET not in notice
        assert "proc_4" in notice

    def test_running_notice_redacts_and_omits_output(self):
        notice = _format_process_running_notice(
            session_id="proc_5", command=_SECRET_CMD
        )
        assert _SECRET not in notice
        assert "proc_5 is still running" in notice
        assert 'process(action="log", session_id="proc_5")' in notice

    def test_long_commands_are_shortened_after_redaction(self):
        long_cmd = "echo " + "x" * 600
        notice = _format_process_running_notice(session_id="proc_6", command=long_cmd)
        command_line = next(
            line for line in notice.splitlines() if line.startswith("Command: ")
        )
        assert len(command_line) <= 310
        assert command_line.endswith("…")
