"""Telegram plan-aware progress card for long-running turns (#124600).

A single card is sent when the first clean model commentary segment arrives and is
edited in place: new commentary renders as ``Now:`` under the pinned ``Plan:`` block,
with an edit floor between edits and a fallback refresh of the elapsed header during
silence. Tool telemetry, thinking scratch, and internal paths never render; with no
clean commentary the card stays silent (renders ``None``).
"""

from __future__ import annotations

import re
import time

#: Minimum seconds between card edits (also the documented default).
EDIT_FLOOR_SECONDS = 60.0
#: Seconds of silence after the last edit that still refresh the elapsed header.
FALLBACK_REFRESH_SECONDS = 300.0

#: Max rendered card length (Telegram caps messages at 4096 chars).
_MAX_CARD_CHARS = 3500

#: A line that is only an absolute path (optionally quoted) is internal noise, not prose.
_PATH_ONLY_RE = re.compile(r"""^["']?/(?:[\w.\-]+/)*[\w.\-]+\.\w+["']?$""")
#: Absolute paths embedded in prose are redacted to their basename. The lookbehind
#: keeps URL path segments (``https://host/a/b``) intact: a ``/`` preceded by ``/`` or
#: ``:`` starts a URL, not an absolute path, and mangling a URL is worse than leaving
#: an internal path in (#124976).
_EMBEDDED_PATH_RE = re.compile(r"(?<![\w:/])/(?:[\w.\-]+/)+([\w.\-]+)")
#: Tool-status chatter, e.g. "Calling web_search..." / "Running tests…".
_TELEMETRY_RE = re.compile(
    r"(calling|running|executing|invoking|searching|reading|writing|fetching|querying)\b.{0,60}(\.\.\.|…)",
    re.IGNORECASE,
)
#: Thinking scratch about what to do next, e.g. "considering which tool to call".
_THINKING_RE = re.compile(
    r"\b(considering|deciding|weighing|thinking about|figuring out|planning)\b"
    r".*\b(tool|call|next|whether|which)\b",
    re.IGNORECASE,
)


def clean_commentary_segment(text: object) -> str:
    """Keep model-authored prose; drop telemetry/scratch lines and redact internal paths.

    Returns ``""`` when nothing speakable remains (the card stays silent)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _PATH_ONLY_RE.match(line):
            continue
        if line and not line[0].isalnum() and line[0] not in ("(", "[", '"', "'"):
            # Emoji/symbol-prefixed status line (progress pips, thinking bubbles) — but
            # markdown markers ("- step", "**Step 1**") introduce real prose, and the
            # opening commentary is often a bulleted plan, so strip them and judge again
            # instead of dropping the line (#124976).
            stripped = line.lstrip("-*+># \t")
            if not stripped or (
                not stripped[0].isalnum() and stripped[0] not in ("(", "[", '"', "'")
            ):
                continue
            line = stripped
        if _TELEMETRY_RE.search(line) or _THINKING_RE.search(line):
            continue
        line = _EMBEDDED_PATH_RE.sub(r"\1", line)
        if line and re.search(r"[A-Za-z0-9]", line):
            kept.append(line)
    return "\n".join(kept).strip()


def render_progress_card(plan: object, now: object = None, elapsed_seconds: float = 0.0) -> str | None:
    """Render the card text, or ``None`` when there is no plan (card stays silent).

    ``now`` is the latest update text (rendered as ``Now:``); only the elapsed header
    changes during silence, which is what the fallback refresh rewrites.
    """
    if not isinstance(plan, str) or not plan.strip():
        return None
    try:
        elapsed = max(0.0, float(elapsed_seconds or 0.0))
    except (TypeError, ValueError):
        elapsed = 0.0
    lines = [f"⏳ Working — {int(elapsed // 60)} min", "", "Plan:", plan.strip()]
    if isinstance(now, str) and now.strip():
        lines += ["", f"Now: {' '.join(now.split())}"]
    text = "\n".join(lines)
    return text[:_MAX_CARD_CHARS]


class TelegramProgressCard:
    """Accumulates commentary into one pinned-plan + latest-update card."""

    def __init__(
        self,
        *,
        edit_floor: float = EDIT_FLOOR_SECONDS,
        fallback: float = FALLBACK_REFRESH_SECONDS,
    ) -> None:
        self.edit_floor = float(edit_floor)
        self.fallback = float(fallback)
        self._plan: str | None = None
        self._current: str | None = None
        self._started_at: float | None = None
        self._message_id: str | None = None
        self._last_edit_at: float | None = None
        self._last_sent_snapshot: tuple | None = None

    def observe(self, text: object, now: float | None = None) -> bool:
        """Fold a raw commentary segment in; True when card content changed."""
        cleaned = clean_commentary_segment(text)
        if not cleaned:
            return False
        if now is None:
            now = time.time()
        if self._plan is None:
            self._plan = cleaned
            self._started_at = now
            return True
        flat = " ".join(cleaned.split())
        if flat in (" ".join(self._plan.split()), self._current):
            return False
        self._current = flat
        return True

    def should_send(self, now: float | None = None) -> bool:
        """Sendable as soon as the first clean segment arrived (no tick wait)."""
        return self._plan is not None and self._message_id is None

    def mark_sent(self, message_id: object, now: float | None = None) -> None:
        """Record a send/edit round-trip; resets the floor and fallback clocks."""
        if now is None:
            now = time.time()
        self._message_id = str(message_id)
        self._last_edit_at = now
        self._last_sent_snapshot = (self._plan, self._current)

    def render(self, now: float | None = None) -> str | None:
        """Current card text, or ``None`` when still silent."""
        if self._plan is None:
            return None
        if now is None:
            now = time.time()
        elapsed = max(0.0, now - self._started_at) if self._started_at is not None else 0.0
        return render_progress_card(self._plan, self._current, elapsed)

    def due_for_refresh(self, now: float | None = None) -> bool:
        """True when new content cleared the edit floor, or the fallback elapsed.

        The dirty check compares plan/update content only (not the elapsed header),
        so the header ticking over a minute boundary alone never forces an edit.
        """
        if now is None:
            now = time.time()
        if self._message_id is None:
            return self.should_send(now)
        last = self._last_edit_at if self._last_edit_at is not None else now
        since_edit = max(0.0, now - last)
        if since_edit >= self.fallback:
            return True
        dirty = (self._plan, self._current) != self._last_sent_snapshot
        return bool(dirty and since_edit >= self.edit_floor)
