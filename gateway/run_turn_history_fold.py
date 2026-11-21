"""Long-turn history fold (``display.progress_fold_after``, opt-in).

A long gateway turn stacks progress bubbles and interim commentary in the chat. Once the turn has
more than ``fold_after`` of those messages on screen, the oldest ones collapse: the first folded
message is edited into one compact history summary (``<narration> · 3 steps`` lines, capped at
``HISTORY_LINES`` lines / ``HISTORY_CHARS`` characters with a ``… N earlier`` head) and each later
folded message is deleted. The newest ``fold_after`` messages stay as they are.

Pure delivery: nothing here touches the transcript or the prompt. Needs an adapter with real
``edit_message`` and ``delete_message``; an adapter may render the summary natively through
``edit_progress_history(chat_id, message_id, lines, metadata=...)`` (Slack: a grey context block).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from typing import Any, Optional

from agent.i18n import t
from agent.interrupt_compat import _accepts_keyword
from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.run")

HISTORY_LINES = 6
HISTORY_CHARS = 1200
_NARRATION_CHARS = 160


@dataclasses.dataclass
class _Entry:
    message_id: str
    narration: str = ""
    steps: int = 0


def _clip(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= _NARRATION_CHARS else text[: _NARRATION_CHARS - 1] + "…"


def count_steps(text: str) -> int:
    """Progress items in a bubble: one per line outside code fences, plus one per fenced block that
    has no header line of its own (back-to-back terminal commands drop the repeated header)."""
    steps, in_fence, prev_outside = 0, False, False
    for line in str(text).splitlines():
        if line.strip().startswith("```"):
            if not in_fence and not prev_outside:
                steps += 1
            in_fence, prev_outside = not in_fence, False
        elif not in_fence and line.strip():
            steps, prev_outside = steps + 1, True
    return steps


def _steps_label(count: int) -> str:
    return t("gateway.progress.history_step") if count == 1 else t("gateway.progress.history_steps", count=count)


def summarize(entries: list[_Entry]) -> list[str]:
    """One line per narration, with the count of progress steps that followed it."""
    lines: list[str] = []
    steps = 0

    def flush() -> None:
        nonlocal steps
        if not steps:
            return
        if lines:
            lines[-1] += f" · {_steps_label(steps)}"
        else:
            lines.append(_steps_label(steps))
        steps = 0

    for entry in entries:
        narration = _clip(entry.narration)
        if narration:
            flush()
            lines.append(narration)
        steps += entry.steps
    flush()
    return lines


def cap_lines(lines: list[str]) -> list[str]:
    """Keep the newest lines within the line/char budget; name how many earlier ones were dropped."""
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        if len(kept) >= HISTORY_LINES or size + len(line) + 1 > HISTORY_CHARS:
            break
        kept.insert(0, line)
        size += len(line) + 1
    dropped = len(lines) - len(kept)
    return [t("gateway.progress.history_earlier", count=dropped), *kept] if dropped else kept


class HistoryFold:
    """Per-turn fold of this turn's progress/commentary messages. All methods run on the gateway loop."""

    def __init__(self, adapter: Any, chat_id: str, fold_after: int, metadata: Optional[dict] = None):
        self._adapter, self._chat_id, self._fold_after, self._metadata = adapter, chat_id, fold_after, metadata
        self._live: list[_Entry] = []
        self._folded: list[_Entry] = []
        self._dead: set[str] = set()
        self._pending: list[str] = []
        self._history_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    @classmethod
    def for_turn(cls, adapter: Any, chat_id: str, fold_after: Any, metadata: Optional[dict]) -> Optional[HistoryFold]:
        """The turn's fold, or None when off (the default) or the adapter cannot edit and delete."""
        if not isinstance(fold_after, int) or fold_after <= 0 or adapter is None:
            return None
        for name in ("edit_message", "delete_message"):
            if getattr(type(adapter), name, None) in (None, getattr(BasePlatformAdapter, name)):
                return None
        return cls(adapter, chat_id, fold_after, metadata)

    def is_folded(self, message_id: Any) -> bool:
        return str(message_id) in self._dead or str(message_id) == self._history_id

    def note_progress(self, message_id: Any, text: str) -> None:
        """A progress bubble was sent or edited: each progress item in it is one step."""
        self._note(str(message_id), "", count_steps(text))

    def note_narration(self, message_id: Any, text: str) -> None:
        """An interim commentary / narration segment landed as its own message."""
        self._note(str(message_id), str(text or ""), 0)

    def _note(self, message_id: str, narration: str, steps: int) -> None:
        if not message_id or message_id == "None" or self.is_folded(message_id):
            return
        live = next((e for e in self._live if e.message_id == message_id), None)
        if live is not None:
            live.narration, live.steps = narration or live.narration, steps or live.steps
            return
        self._live.append(_Entry(message_id, narration, steps))
        while len(self._live) > self._fold_after:
            entry = self._live.pop(0)
            self._folded.append(entry)
            self._dead.add(entry.message_id)
            self._pending.append(entry.message_id)
        if self._pending:
            task = asyncio.get_running_loop().create_task(self._apply())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _apply(self) -> None:
        async with self._lock:
            pending, self._pending = self._pending, []
            if not pending:
                return
            for message_id in pending:
                if self._history_id is None:
                    self._history_id = message_id
                    self._dead.discard(message_id)
                    continue
                try:
                    await self._adapter.delete_message(self._chat_id, message_id)
                except Exception:  # health: allow BLE001 -- a refused delete leaves that message; the fold goes on
                    logger.debug("History fold: delete of %s failed", message_id, exc_info=True)
            await self._render(cap_lines(summarize(self._folded)))

    async def _render(self, lines: list[str]) -> None:
        native = getattr(self._adapter, "edit_progress_history", None)
        try:
            if native is not None:
                await native(self._chat_id, self._history_id, lines, metadata=self._metadata)
                return
            kwargs: dict[str, Any] = {"chat_id": self._chat_id, "message_id": self._history_id,
                                      "content": "\n".join(lines)}
            if self._metadata and _accepts_keyword(self._adapter.edit_message, "metadata"):
                kwargs["metadata"] = self._metadata
            await self._adapter.edit_message(**kwargs)
        except Exception:  # health: allow BLE001 -- the summary edit is cosmetic; the turn must not fail on it
            logger.debug("History fold: summary edit failed", exc_info=True)
