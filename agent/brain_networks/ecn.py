"""Executive Control Network (ECN) — task focus and cognitive control.

The ECN manages:
- Task switching and focus
- Working memory maintenance
- Inhibition of distractions
- Goal-directed behavior
- Persistent session focus (SQLite via persistence module)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExecutiveControlNetwork:
    """Executive Control Network implementation with explicit task-state machine."""

    TASK_STATES = ["idle", "focused", "interrupted", "recovering", "switching"]

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.current_task: Optional[str] = None
        self.task_stack: List[str] = []
        self.focus_level: float = 0.5  # 0.0-1.0
        self.distraction_count: int = 0
        self.state: str = "idle"
        self.max_stack: int = 10
        self.session_id: str = ""
        self.pinned: bool = False
        self._persist: bool = True

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize ECN."""
        self.config = config or {}
        self.max_stack = int(self.config.get("max_task_stack", 10))
        self._persist = bool(self.config.get("persist", True))
        if self.state not in self.TASK_STATES:
            self.state = "idle"
        return True

    def bind_session(self, session_id: str, *, load: bool = True) -> None:
        """Bind to a conversation session and optionally hydrate from disk."""
        sid = (session_id or "").strip()
        if sid and sid == self.session_id and self.current_task is not None:
            return
        self.session_id = sid
        if load and self._persist and sid:
            self.load_persisted()

    def load_persisted(self) -> bool:
        """Hydrate from SQLite. Returns True if a row was loaded."""
        if not self.session_id:
            return False
        try:
            from agent.brain_networks.persistence import load_ecn_state

            data = load_ecn_state(self.session_id)
        except Exception as exc:
            logger.debug("ECN load_persisted failed: %s", exc)
            return False
        if not data:
            return False
        self.current_task = data.get("current_task")
        self.task_stack = list(data.get("task_stack") or [])
        self.focus_level = float(data.get("focus_level", 0.5))
        self.distraction_count = int(data.get("distraction_count", 0))
        self.state = str(data.get("state") or "idle")
        self.pinned = bool(data.get("pinned"))
        return True

    def persist(self) -> None:
        """Write current focus state to SQLite (best-effort)."""
        if not self._persist or not self.session_id:
            return
        try:
            from agent.brain_networks.persistence import save_ecn_state

            save_ecn_state(
                self.session_id,
                current_task=self.current_task,
                task_stack=self.task_stack,
                focus_level=self.focus_level,
                distraction_count=self.distraction_count,
                state=self.state,
                pinned=self.pinned,
            )
        except Exception as exc:
            logger.debug("ECN persist failed: %s", exc)

    def set_focus(self, task: str, *, pinned: bool = True) -> Dict[str, Any]:
        """Explicitly set the standing focus task (used by /focus)."""
        task = (task or "").strip()
        if not task:
            return self.get_state()
        if self.current_task and self.current_task != task:
            self._push_task(self.current_task)
        self.current_task = task[:200]
        self.focus_level = 1.0
        self.distraction_count = 0
        self.state = "focused"
        self.pinned = bool(pinned)
        self.persist()
        return self.get_state()

    def clear_focus(self) -> Dict[str, Any]:
        """Clear standing focus and optional persistence row."""
        self.current_task = None
        self.task_stack = []
        self.focus_level = 0.5
        self.distraction_count = 0
        self.state = "idle"
        self.pinned = False
        if self.session_id and self._persist:
            try:
                from agent.brain_networks.persistence import clear_ecn_state

                clear_ecn_state(self.session_id)
            except Exception as exc:
                logger.debug("ECN clear_focus persist failed: %s", exc)
        return self.get_state()

    def evaluate_focus(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate and maintain task focus with explicit state transitions.

        Returns:
            Focus state information including state-machine state and reminder
        """
        # Ensure session binding when context carries session_id
        sid = str(context.get("session_id") or "").strip()
        if sid:
            self.bind_session(sid, load=True)

        user_message = context.get("user_message", "")

        # Pinned focus: do not auto-switch unless explicit switch language
        task_switched = self._detect_task_switch(user_message)
        is_distraction = self._is_distraction(user_message)

        if self.pinned and self.current_task and not task_switched:
            # Maintain pinned focus; still track mild distraction
            if is_distraction:
                self.distraction_count += 1
                self.focus_level = max(0.4, self.focus_level - 0.05)
                self.state = "interrupted" if self.distraction_count > 2 else "focused"
            else:
                self.focus_level = min(1.0, self.focus_level + 0.05)
                self.distraction_count = max(0, self.distraction_count - 1)
                self.state = "focused"
            result = self._focus_result(task_switched=False)
            self.persist()
            return result

        if task_switched:
            if self.current_task:
                self._push_task(self.current_task)
            self.current_task = user_message[:100]
            self.focus_level = 1.0
            self.distraction_count = 0
            self.state = "switching"
            self.pinned = False
        elif is_distraction:
            self.distraction_count += 1
            self.focus_level = max(0.2, self.focus_level - 0.15)
            if self.distraction_count > 2:
                self.state = "interrupted"
            elif self.state == "interrupted":
                self.state = "recovering"
            else:
                self.state = "focused" if self.focus_level > 0.5 else "recovering"
        else:
            # Maintain or degrade focus
            self.focus_level = max(0.3, self.focus_level - 0.05)
            self.distraction_count = max(0, self.distraction_count - 1)
            self.state = "focused" if self.focus_level > 0.5 else "recovering"
            if self.current_task is None:
                # First substantive message becomes the implicit task
                if user_message and len(user_message.strip()) >= 10 and not is_distraction:
                    self.current_task = user_message[:100]
                    self.focus_level = 0.85
                    self.state = "focused"
                else:
                    self.state = "idle"

        result = self._focus_result(task_switched=task_switched)
        self.persist()
        return result

    def _focus_result(self, *, task_switched: bool) -> Dict[str, Any]:
        reminder = ""
        if self.current_task:
            pin = " (pinned)" if self.pinned else ""
            reminder = (
                f"Stay on: {self.current_task}{pin} "
                f"[focus={self.focus_level:.2f} state={self.state}]"
            )
        return {
            "current_task": self.current_task,
            "task_stack_depth": len(self.task_stack),
            "focus_level": round(self.focus_level, 2),
            "distraction_count": self.distraction_count,
            "task_switched": task_switched,
            "should_inhibit": self.distraction_count > 2,
            "state": self.state,
            "pinned": self.pinned,
            "reminder": reminder,
            "focus": reminder,
        }

    def _detect_task_switch(self, message: str) -> bool:
        """Detect if user is switching tasks."""
        switch_indicators = [
            "instead", "rather", "change", "switch", "now",
            "different", "forget that", "new topic", "by the way",
        ]

        message_lower = message.lower()
        return any(indicator in message_lower for indicator in switch_indicators)

    def _is_distraction(self, message: str) -> bool:
        """Check if message is potentially distracting."""
        # Very short messages might be distractions
        if len(message) < 10:
            return True

        # Off-topic indicators
        off_topic = ["unrelated", "random", "just curious", "not important"]
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in off_topic)

    def get_working_memory(self) -> List[str]:
        """Get current working memory contents."""
        return self.task_stack[-3:] if self.task_stack else []

    def _push_task(self, task: str) -> None:
        """Push task onto stack, respecting max depth."""
        if task:
            self.task_stack.append(task)
            if len(self.task_stack) > self.max_stack:
                self.task_stack.pop(0)

    def push_task(self, task: str) -> None:
        """Push task onto stack."""
        self._push_task(self.current_task)
        self.current_task = task
        self.focus_level = 1.0
        self.state = "focused"
        self.persist()

    def pop_task(self) -> Optional[str]:
        """Pop task from stack."""
        if self.task_stack:
            self.current_task = self.task_stack.pop()
            self.focus_level = min(1.0, self.focus_level + 0.2)
            self.state = "recovering"
            self.persist()
            return self.current_task
        self.state = "idle"
        self.persist()
        return None

    def get_state(self) -> Dict[str, Any]:
        """Get ECN state."""
        return {
            "current_task": self.current_task,
            "task_stack_depth": len(self.task_stack),
            "focus_level": round(self.focus_level, 2),
            "distraction_count": self.distraction_count,
            "state": self.state,
            "pinned": self.pinned,
            "session_id": self.session_id,
            "working_memory": self.get_working_memory(),
            "reminder": (
                f"Stay on: {self.current_task}" if self.current_task else ""
            ),
        }
