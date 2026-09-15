"""Launch-time arming for gateway process-completion watchers (#112033)."""

from types import SimpleNamespace

from tools.process_registry import ProcessRegistry
from tools.terminal_tool_background import _register_completion_watcher


def _session():
    return SimpleNamespace(
        id="proc_active", watcher_platform="telegram", watcher_chat_id="123",
        watcher_user_id="u", watcher_user_name="Ada", watcher_thread_id="",
        watcher_message_id="", parent_session_id="parent",
    )


def test_launch_arms_watcher_immediately_when_gateway_is_live():
    registry = ProcessRegistry()
    armed = []
    registry.set_watcher_scheduler(lambda watcher: armed.append(watcher) or True)

    _register_completion_watcher(registry, _session(), "agent:main:telegram:dm:123")

    assert [watcher["session_id"] for watcher in armed] == ["proc_active"]
    assert registry.pending_watchers == []


def test_launch_keeps_watcher_pending_when_no_gateway_is_live():
    registry = ProcessRegistry()

    _register_completion_watcher(registry, _session(), "agent:main:telegram:dm:123")

    assert [watcher["session_id"] for watcher in registry.take_pending_watchers()] == ["proc_active"]
    assert registry.pending_watchers == []


def test_failed_live_schedule_returns_watcher_to_recovery_queue():
    registry = ProcessRegistry()
    registry.set_watcher_scheduler(lambda watcher: False)

    _register_completion_watcher(registry, _session(), "agent:main:telegram:dm:123")

    assert [watcher["session_id"] for watcher in registry.take_pending_watchers()] == ["proc_active"]
