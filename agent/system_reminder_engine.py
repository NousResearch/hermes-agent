"""Cyclic system-reminder injection engine.

Injects configurable ``<system-reminder>`` blocks into the tool-result stream
on a tool-call cadence.  Designed to coexist with ``/steer`` — the steer
injection fires once per user request; this engine fires on a counter-based
cadence across tool calls.

Design invariants (see AGENTS.md):
- Does NOT mutate past context — only appends to the most recent tool
  result in the current batch, same as ``/steer``.
- Does NOT break role alternation — reminders are appended to existing
  tool-role messages, never inserted as new synthetic messages.
- Does NOT make an API call — zero network, zero file I/O.
- Does NOT depend on AIAgent internals beyond what's documented on the
  ``agent`` parameter to ``maybe_inject()``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Default block definitions ──────────────────────────────────────────

DEFAULT_REMINDER_BLOCKS: List[Dict[str, str]] = [
    {
        "id": "tool-use-enforcement",
        "content": (
            "<system-reminder>\n"
            "You have access to tools for web search, terminal, file "
            "operations, and more.  When the user asks a question that "
            "requires up-to-date information, use web_search instead of "
            "relying on your training data.  Use terminal for system tasks, "
            "read_file for examining files, and write_file for creating them.\n"
            "</system-reminder>"
        ),
    },
    {
        "id": "task-completion",
        "content": (
            "<system-reminder>\n"
            "Task completion guidance: when you finish the user's task, "
            "provide a clear summary of what was done, list the artifacts "
            "you created or modified with their absolute paths, and note "
            "any decisions or trade-offs made during implementation.\n"
            "</system-reminder>"
        ),
    },
    {
        "id": "memory-usage",
        "content": (
            "<system-reminder>\n"
            "You have persistent memory across sessions.  Save durable "
            "facts using the memory tool: user preferences, environment "
            "details, tool quirks, and stable conventions.  Do NOT save "
            "task progress, session outcomes, or temporary TODO state.\n"
            "</system-reminder>"
        ),
    },
    {
        "id": "file-delivery",
        "content": (
            "<system-reminder>\n"
            "File delivery: when the user asks you to build, run, or "
            "verify something, the deliverable is a working artifact backed "
            "by real tool output — not a description of one.  Keep working "
            "until you have actually exercised the code or produced the "
            "requested result.\n"
            "</system-reminder>"
        ),
    },
    {
        "id": "kanban-lifecycle",
        "content": (
            "<system-reminder>\n"
            "Kanban task lifecycle: orient with kanban_show() → work inside "
            "workspace → heartbeat on long ops → block on ambiguity → "
            "complete with structured handoff.  Do NOT complete a task you "
            "didn't actually finish — block it instead.\n"
            "</system-reminder>"
        ),
    },
]


class SystemReminderEngine:
    """Cyclic reminder injection engine.

    Parameters
    ----------
    enabled : bool
        Master on/off switch.  When False, ``should_inject()`` always
        returns False and ``build_reminder()`` returns None.
    cadence : int
        Inject a reminder every *N* tool calls.  Default 5.
    blocks : list[dict]
        List of reminder block dicts, each with ``id`` and ``content``
        keys.  The engine cycles through them on each injection tick.
    position : str
        Where to inject: ``"last_user"`` (appends to the most recent
        user message), ``"new_user"`` (new user-role message), or
        ``"assistant_prefix"`` (prepended to the next assistant turn).
    """

    def __init__(
        self,
        enabled: bool = True,
        cadence: int = 5,
        blocks: Optional[List[Dict[str, str]]] = None,
        position: str = "last_user",
    ) -> None:
        self.enabled = enabled
        self.cadence = max(1, cadence)  # guard against zero / negative
        self.blocks: List[Dict[str, str]] = (
            list(blocks) if blocks is not None else list(DEFAULT_REMINDER_BLOCKS)
        )
        self.position = position
        self._counter = 0
        self._block_index = 0

    # ── Public API ────────────────────────────────────────────────────

    def should_inject(self) -> bool:
        """Check whether it's time to inject a reminder.

        Call after each tool result has been appended.  Returns True
        when the tool-call counter has reached the cadence threshold.

        Thread-safety: the engine is typically called from a single
        thread (the tool-executor loop).  If called from multiple
        threads, the caller must hold a lock around both the
        ``should_inject() -> build_reminder()`` pair.
        """
        if not self.enabled or not self.blocks:
            return False
        self._counter += 1
        if self._counter >= self.cadence:
            self._counter = 0
            return True
        return False

    def build_reminder(self) -> Optional[Dict[str, Any]]:
        """Build the next reminder content block.

        Returns a dict with keys ``content`` (str) and, for the
        ``"last_user"`` position, ``action`` (``"append_to_last_user"``).
        Returns None when disabled or no blocks are configured.

        The return value is designed to be consumed by the
        ``agent_inject_system_reminder()`` module-level function,
        not interpreted by the caller directly.
        """
        if not self.enabled or not self.blocks:
            return None
        block = self.blocks[self._block_index % len(self.blocks)]
        self._block_index += 1
        content = block["content"]

        if self.position == "last_user":
            return {"action": "append_to_last_user", "content": "\n\n" + content}
        elif self.position == "new_user":
            return {"role": "user", "content": content}
        else:  # assistant_prefix
            return {"action": "assistant_prefix", "content": content}

    def maybe_inject(self) -> Optional[Dict[str, Any]]:
        """Convenience: ``should_inject()`` + ``build_reminder()`` in one call.

        Thread-safety: same as calling the two methods separately —
        NOT atomic across threads.
        """
        if self.should_inject():
            return self.build_reminder()
        return None

    # ── State helpers ─────────────────────────────────────────────────

    def reset_counter(self) -> None:
        """Reset the tool-call counter (e.g. after a user message)."""
        self._counter = 0

    @property
    def state(self) -> Dict[str, Any]:
        """Serialisable snapshot for logging / diagnostics."""
        return {
            "enabled": self.enabled,
            "cadence": self.cadence,
            "num_blocks": len(self.blocks),
            "position": self.position,
            "counter": self._counter,
            "block_index": self._block_index,
            "next_block_id": (
                self.blocks[self._block_index % len(self.blocks)]["id"]
                if self.blocks
                else None
            ),
        }


# ── Injection helper (consumed by tool_executor.py) ────────────────────


def agent_inject_system_reminder(
    agent: Any,
    messages: list,
    num_tool_msgs: int = 0,
) -> None:
    """Check the agent's ``_system_reminder_engine`` and inject if due.

    Designed to be called from ``tool_executor.py`` in the same spots
    where ``/steer`` is injected:

    * After each individual tool result is appended (per-tool drain).
    * After the final tool result of a batch (batch drain).

    The injection appends to the last ``role:"tool"`` message's content
    (same pattern as ``/steer``) so role alternation is preserved.

    Parameters
    ----------
    agent:
        The AIAgent instance.  Must have a ``_system_reminder_engine``
        attribute (set by ``agent_init.py``).
    messages:
        The running messages list.  Modified in-place when a reminder
        is injected.
    num_tool_msgs:
        Number of tool-result messages appended in this batch.  Used to
        locate the last tool-role message safely.  Pass 0 to skip.
    """
    if num_tool_msgs <= 0 or not messages:
        return
    engine: Optional[SystemReminderEngine] = getattr(
        agent, "_system_reminder_engine", None
    )
    if engine is None:
        return
    reminder = engine.maybe_inject()
    if reminder is None:
        return
    action = reminder.get("action")
    if action == "append_to_last_user":
        # Same pattern as /steer: find the last tool-role message in
        # the recent tail and append the reminder content.
        target_idx = None
        for j in range(
            len(messages) - 1,
            max(len(messages) - num_tool_msgs - 1, -1),
            -1,
        ):
            msg = messages[j]
            if isinstance(msg, dict) and msg.get("role") == "tool":
                target_idx = j
                break
        if target_idx is not None:
            existing = messages[target_idx].get("content", "")
            if isinstance(existing, str):
                messages[target_idx]["content"] = existing + reminder["content"]
            else:
                # Anthropic multimodal content blocks — append a text block.
                blocks = list(existing) if existing else []
                blocks.append({"type": "text", "text": reminder["content"]})
                messages[target_idx]["content"] = blocks
            logger.info(
                "Injected system-reminder block (action=%s, %d chars)",
                action,
                len(reminder["content"]),
            )
        else:
            logger.debug(
                "System-reminder skipped: no tool-role message found in tail"
            )
    elif action == "assistant_prefix":
        logger.info(
            "System-reminder assistant_prefix requested but not yet implemented"
        )
    # new_user position is intentionally not handled in tool_executor
    # context — those reminders are delivered via conversation_loop.py
    # before the next user turn.
