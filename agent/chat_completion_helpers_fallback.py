"""Fallback route notices shared by the completion runtime."""

def _buffer_fallback_notice(agent, notice: str) -> None:
    """Buffer the switch notice for terminal failure AND retain it as a durable one-shot for
    _emit_pending_fallback_notice (a successful fallback clears retry chatter)."""
    agent._buffer_diagnostic_status(notice)
    pending = getattr(agent, "_pending_fallback_notice", None)
    if isinstance(pending, list):
        pending.append(notice)
    else:
        agent._pending_fallback_notice = [str(pending), notice] if pending else [notice]
