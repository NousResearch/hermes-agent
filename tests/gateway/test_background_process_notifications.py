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


# ---------------------------------------------------------------------------
# _coalesce_and_inject_watch_events tests (post-turn drain coalescing)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_watch_event_passes_through_unchanged(monkeypatch, tmp_path):
    """A lone watch event should be injected as-is, no coalescing needed."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.handle_message = AsyncMock()

    evt = {
        "type": "watch_match",
        "session_id": "proc_abc",
        "session_key": "agent:main:telegram:dm:123:42",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "42",
        "pattern": "DONE",
        "output": "Build DONE",
        "command": "make build",
    }

    await runner._coalesce_and_inject_watch_events([evt])

    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_multiple_watch_match_same_session_key_are_coalesced(
    monkeypatch, tmp_path,
):
    """Multiple watch_match events for the same session_key produce exactly
    one injection."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.handle_message = AsyncMock()

    session_key = "agent:main:telegram:dm:123:42"
    events = [
        {
            "type": "watch_match",
            "session_id": f"proc_{i}",
            "session_key": session_key,
            "platform": "telegram",
            "chat_id": "123",
            "thread_id": "42",
            "pattern": f"PATTERN_{i}",
            "output": f"output_{i}",
            "command": f"cmd_{i}",
        }
        for i in range(4)
    ]

    await runner._coalesce_and_inject_watch_events(events)

    # 4 events → 1 coalesced injection
    assert adapter.handle_message.await_count == 1
    injected_text = adapter.handle_message.await_args.args[0].text
    assert "4 background processes matched watch patterns" in injected_text
    assert "batched" in injected_text.lower()
    for i in range(4):
        assert f"proc_{i}" in injected_text


@pytest.mark.asyncio
async def test_watch_match_different_session_keys_not_coalesced(
    monkeypatch, tmp_path,
):
    """watch_match events for different session_keys remain separate."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.handle_message = AsyncMock()

    events = [
        {
            "type": "watch_match",
            "session_id": f"proc_a{i}",
            "session_key": f"agent:main:telegram:dm:100:{i}",
            "platform": "telegram",
            "chat_id": "100",
            "thread_id": str(i),
            "pattern": "DONE",
            "output": f"out_{i}",
            "command": f"cmd_{i}",
        }
        for i in range(3)
    ]

    await runner._coalesce_and_inject_watch_events(events)

    # Different session_keys → separate injections
    assert adapter.handle_message.await_count == 3


@pytest.mark.asyncio
async def test_mixed_watch_types_coalesced_by_type_and_key(
    monkeypatch, tmp_path,
):
    """watch_match and watch_disabled events for the same session_key
    are coalesced separately by type."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.handle_message = AsyncMock()

    session_key = "agent:main:telegram:dm:123:42"
    events = [
        {"type": "watch_match", "session_id": "proc_w1",
         "session_key": session_key, "platform": "telegram",
         "chat_id": "123", "thread_id": "42",
         "pattern": "DONE", "output": "ok", "command": "cmd"},
        {"type": "watch_match", "session_id": "proc_w2",
         "session_key": session_key, "platform": "telegram",
         "chat_id": "123", "thread_id": "42",
         "pattern": "ERROR", "output": "fail", "command": "cmd2"},
        {"type": "watch_disabled", "session_id": "proc_d1",
         "session_key": session_key, "platform": "telegram",
         "chat_id": "123", "thread_id": "42",
         "message": "Watch disabled for proc_d1"},
        {"type": "watch_disabled", "session_id": "proc_d2",
         "session_key": session_key, "platform": "telegram",
         "chat_id": "123", "thread_id": "42",
         "message": "Watch disabled for proc_d2"},
    ]

    await runner._coalesce_and_inject_watch_events(events)

    # 2 watch_match coalesced + 2 watch_disabled coalesced = 2 injections
    assert adapter.handle_message.await_count == 2
    texts = [c.args[0].text for c in adapter.handle_message.await_args_list]

    # One should be the watch_match batch
    match_texts = [t for t in texts if "matched watch patterns" in t]
    assert len(match_texts) == 1
    assert "2 background processes matched watch patterns" in match_texts[0]

    # One should be the watch_disabled batch
    disabled_texts = [t for t in texts if "disabled" in t]
    assert len(disabled_texts) == 1
    assert "2 background processes had watch patterns disabled" in disabled_texts[0]


@pytest.mark.asyncio
async def test_coalesced_ids_truncated_after_5(monkeypatch, tmp_path):
    """When >5 processes, first 5 IDs are listed, rest shown as count."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.handle_message = AsyncMock()

    session_key = "agent:main:telegram:dm:123:42"
    events = [
        {
            "type": "watch_match",
            "session_id": f"proc_{i}",
            "session_key": session_key,
            "platform": "telegram",
            "chat_id": "123",
            "thread_id": "42",
            "pattern": "DONE",
            "output": f"out_{i}",
            "command": "cmd",
        }
        for i in range(8)
    ]

    await runner._coalesce_and_inject_watch_events(events)

    assert adapter.handle_message.await_count == 1
    injected_text = adapter.handle_message.await_args.args[0].text
    assert "8 background processes matched watch patterns" in injected_text
    for i in range(5):
        assert f"proc_{i}" in injected_text
    assert "...and 3 more" in injected_text
    assert "proc_5" not in injected_text


@pytest.mark.asyncio
async def test_empty_events_list_is_noop(monkeypatch, tmp_path):
    """Empty event list should not cause any injection."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.handle_message = AsyncMock()

    await runner._coalesce_and_inject_watch_events([])

    adapter.handle_message.assert_not_awaited()
