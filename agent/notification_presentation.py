"""Turn-local presentation of diagnostic-only wakes; execution and controls stay live."""
from contextlib import contextmanager
from contextvars import ContextVar


_muted_surface: ContextVar[str | None] = ContextVar("muted_notification_surface", default=None)
# These are text/media UI events, not approval/clarify/connection requests or outcomes.
_FREEFORM_EVENTS = frozenset({
    "message.start", "message.delta", "message.interim", "message.complete",
    "reasoning.start", "reasoning.delta", "reasoning.end", "thinking.delta", "status.update", "notification.show",
    "tool.start", "tool.complete", "tool.generating", "error", "reaction",
})
_PRESENTATION_CALLBACKS = (
    "stream_delta_callback", "interim_assistant_callback", "reasoning_callback", "reasoning_event_callback",
    "tool_progress_callback",
    "tool_start_callback", "tool_complete_callback", "tool_gen_callback", "reaction_callback",
)


def event_presentation_muted(event: str, session_id: str) -> bool:
    return _muted_surface.get() == session_id and event in _FREEFORM_EVENTS


def diagnostic_process_event(event: dict) -> bool:
    """Early failure/monitor diagnostics, not the explicitly requested final result."""
    return bool(event.get("task_failure_notice")) or event.get("type") in {
        "watch_disabled", "watch_overflow_tripped", "watch_overflow_released",
    }


def notification_config_snapshot():
    """Read the owning effective config once per turn (the loader already returns a fresh copy)."""
    from gateway.warning_notifications import effective_user_config
    return effective_user_config()


@contextmanager
def notification_policy_snapshot(agent, platform, config):
    """Bind one foreground policy for callbacks, including worker threads. ``config`` is read-only."""
    missing = object()
    saved = {key: getattr(agent, key, missing)
             for key in ("_notification_config", "_notification_platform")}
    try:
        agent._notification_config = config
        agent._notification_platform = platform
        yield
    finally:
        for key, value in saved.items():
            if value is missing:
                delattr(agent, key)
            else:
                setattr(agent, key, value)


@contextmanager
def notification_turn(agent, *, muted: bool, session_id: str = ""):
    """Freeze the current turn's presentation without touching prompts or tool schemas."""
    if not muted:
        yield
        return
    missing = object()
    keys = (*_PRESENTATION_CALLBACKS, "suppress_status_output", "_mute_notification_reply")
    saved = {key: getattr(agent, key, missing) for key in keys}
    token = _muted_surface.set(session_id)
    try:
        for key in _PRESENTATION_CALLBACKS:
            setattr(agent, key, None)
        agent.suppress_status_output = True
        agent._mute_notification_reply = True
        yield
    finally:
        for key, value in saved.items():
            if value is missing:
                delattr(agent, key)
            else:
                setattr(agent, key, value)
        _muted_surface.reset(token)
