"""Busy-input action menu state for gateway adapters.

This module intentionally owns only prompt/action state and generic labels.  The
GatewayRunner remains responsible for side effects (queueing, steering,
interrupting, stopping, replaying events), while platform adapters render the
menu and route callbacks back to the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import secrets
import time
from typing import Any, Dict, Iterable, Optional


DEFAULT_BUSY_PROMPT_TTL_SECONDS = 15 * 60


@dataclass(frozen=True)
class BusyInputAction:
    """A single action exposed by a busy-input prompt."""

    key: str
    label: str
    description: str = ""
    style: str = "default"


@dataclass
class BusyInputPrompt:
    """Server-side state for a platform inline busy-input menu."""

    prompt_id: str
    session_key: str
    event: Any
    run_generation: int = 0
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    status: str = "open"  # open | resolved | expired | cancelled
    resolved_action: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        return bool(self.expires_at and now >= self.expires_at)


class BusyInputManager:
    """In-memory busy-input prompt registry with TTL and idempotency.

    Prompt payloads include MessageEvent objects, so persistence is deliberately
    not attempted here.  After a gateway restart, callbacks resolve as unknown /
    expired, which is safer than replaying partially serialized user input.
    """

    ACTIVE_ACTIONS = (
        BusyInputAction("queue", "⏳ Queue", "Run after the current task finishes."),
        BusyInputAction("context", "💬 Context", "Try to inject as context; otherwise queue."),
        BusyInputAction("steer", "⏩ Steer", "Inject into the current run if possible."),
        BusyInputAction("interrupt", "⚡ Interrupt", "Interrupt the current task and run this next.", "danger"),
        BusyInputAction("stop", "🛑 Stop", "Stop the current task and discard this message.", "danger"),
        BusyInputAction("cancel", "✖ Cancel", "Discard this message."),
    )

    SUBAGENT_ACTIONS = (
        BusyInputAction("queue", "⏳ Queue safely", "Run after subagents finish."),
        BusyInputAction("context", "💬 Context", "Add context without cancelling subagents."),
        BusyInputAction("stop", "🛑 Stop everything", "Force-stop the current task and subagents.", "danger"),
        BusyInputAction("cancel", "✖ Cancel", "Discard this message."),
    )

    FINISHED_ACTIONS = (
        BusyInputAction("run_now", "▶️ Run now", "The previous run finished; process this now."),
        BusyInputAction("context", "💬 Add as context", "Process this as a follow-up/context note."),
        BusyInputAction("cancel", "✖ Discard", "Discard this message."),
    )

    VALID_ACTIONS = {
        "queue",
        "context",
        "steer",
        "interrupt",
        "stop",
        "cancel",
        "run_now",
        "delete_queued",
    }

    def __init__(self, ttl_seconds: float = DEFAULT_BUSY_PROMPT_TTL_SECONDS):
        self.ttl_seconds = float(ttl_seconds or DEFAULT_BUSY_PROMPT_TTL_SECONDS)
        self._prompts: Dict[str, BusyInputPrompt] = {}

    def new_prompt(
        self,
        *,
        session_key: str,
        event: Any,
        run_generation: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BusyInputPrompt:
        self.prune_expired()
        prompt_id = secrets.token_urlsafe(9)
        while prompt_id in self._prompts:
            prompt_id = secrets.token_urlsafe(9)
        now = time.time()
        prompt = BusyInputPrompt(
            prompt_id=prompt_id,
            session_key=session_key,
            event=event,
            run_generation=int(run_generation or 0),
            created_at=now,
            expires_at=now + self.ttl_seconds,
            metadata=dict(metadata or {}),
        )
        self._prompts[prompt_id] = prompt
        return prompt

    def get(self, prompt_id: str) -> Optional[BusyInputPrompt]:
        prompt = self._prompts.get(prompt_id)
        if prompt and prompt.status == "open" and prompt.is_expired():
            prompt.status = "expired"
        return prompt

    def resolve(
        self,
        prompt_id: str,
        action: str,
        *,
        resolved_by: Optional[str] = None,
    ) -> tuple[str, Optional[BusyInputPrompt]]:
        """Mark a prompt resolved.

        Returns ``(status, prompt)`` where status is one of:
        ``invalid_action``, ``missing``, ``expired``, ``already_resolved``,
        or ``resolved``.
        """
        action = str(action or "").strip().lower()
        if action not in self.VALID_ACTIONS:
            return "invalid_action", None
        prompt = self.get(prompt_id)
        if prompt is None:
            return "missing", None
        if prompt.status == "expired":
            return "expired", prompt
        if prompt.status != "open":
            return "already_resolved", prompt
        prompt.status = "cancelled" if action == "cancel" else "resolved"
        prompt.resolved_action = action
        prompt.resolved_by = resolved_by
        prompt.resolved_at = time.time()
        return "resolved", prompt

    def expire(self, prompt_id: str) -> bool:
        prompt = self._prompts.get(prompt_id)
        if not prompt:
            return False
        prompt.status = "expired"
        return True

    def prune_expired(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        pruned = 0
        for prompt_id, prompt in list(self._prompts.items()):
            if prompt.status == "open" and prompt.is_expired(now):
                prompt.status = "expired"
            # Keep resolved prompts briefly for duplicate-click idempotency.
            terminal_age = now - (prompt.resolved_at or prompt.expires_at or prompt.created_at)
            if prompt.status in {"resolved", "cancelled"} and terminal_age > 300:
                self._prompts.pop(prompt_id, None)
                pruned += 1
            elif prompt.status == "expired" and now - prompt.expires_at > 300:
                self._prompts.pop(prompt_id, None)
                pruned += 1
        return pruned

    @classmethod
    def actions_for_state(cls, *, has_active_run: bool, has_subagents: bool = False) -> Iterable[BusyInputAction]:
        if not has_active_run:
            return cls.FINISHED_ACTIONS
        if has_subagents:
            return cls.SUBAGENT_ACTIONS
        return cls.ACTIVE_ACTIONS
