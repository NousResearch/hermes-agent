"""One-shot compaction outcomes shared by core status and adapter observers."""

from typing import Any


def _emit_compaction_terminal(callback: Any, status: str) -> None:
    """Notify lifecycle observers without changing compaction or legacy status semantics."""
    from agent.conversation_compression import _swallow

    if callable(callback):
        with _swallow('compaction lifecycle observer failed', exc_info=True):
            callback(status)


class _CompactionLifecycle:
    """Owns the one-shot terminal edge of the compaction status lifecycle.
    ``commit_status`` is rebound to "committed" only on success and read at ``complete()`` time, so abort
    paths keep the legacy status edge suppressed; observers still receive the outcome."""

    def __init__(self, agent: Any, status_emitted: bool) -> None:
        self._agent = agent
        self._terminal_callback = getattr(agent, "compaction_callback", None)
        self.cancelled = False
        self.status_emitted = status_emitted
        self._done_emitted = False
        self.commit_status = "aborted"

    def complete(self, *, force_terminal: bool = False) -> None:
        from agent.conversation_compression import _emit_compaction_done

        if self._done_emitted:
            return
        self._done_emitted = True
        # Suppressed start → no terminal edge. Non-compacting aborts (lock contender,
        # cancelled fence) opt in via force_terminal so clients can retire their phase.
        # Failure warnings go through _emit_warning and are never suppressed here.
        interrupted = getattr(self._agent, "_interrupt_requested", False) is True
        outcome = "completed" if self.commit_status == "committed" else (
            "cancelled" if force_terminal or interrupted or self.cancelled else "failed"
        )
        _emit_compaction_terminal(self._terminal_callback, outcome)
        if self.status_emitted and (self.commit_status == "committed" or force_terminal):
            _emit_compaction_done(self._agent)
