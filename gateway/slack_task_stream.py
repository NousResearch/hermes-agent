"""Slack-native "Thinking Steps" task-card streaming for tool progress.

Slack's chat.startStream / chat.appendStream / chat.stopStream trio lets a
bot render tool-call progress as a native, collapsible task-card timeline
inside a message — the same UX Slack's own AI features use — instead of the
plain markdown text bubbles the gateway edits by default. This module wraps
that lifecycle for a single agent turn so gateway/run.py can drive it with
the same tool-start/tool-finish events it already emits for the markdown
progress path.

Layering contract: this module owns ALL presentation logic (labels,
categories, previews, summaries, source chips, error sniffing) as pure
helpers plus the ``SlackTaskStream`` lifecycle class; gateway/run.py only
correlates tool events and schedules coroutines. Keep it that way — the
module is self-contained (no gateway imports) so it can be reused verbatim
if the wiring moves (e.g. onto GatewayEventDispatcher, upstream PR #54522).

Opt-in (``display.platforms.slack.tool_progress_native``) and strictly
additive: every other platform, and Slack installs that don't set the flag,
keep the existing markdown progress-bubble behavior untouched. Tuning knobs
(rollover thresholds, reasoning/output caps) resolve like any other display
setting — see gateway/display_config.py.

Empirically measured Slack API limits (2026-07-05, ehoy
scripts/carnie/slack_stream_probe.py):
  * a streamed message dies ~306s after startStream even with active
    appends (absolute lifetime, not inactivity) → proactive rollover;
  * msg_too_long is per-chunk (a >12k field), NOT cumulative — 62k chars
    of cumulative task_update content passed clean → per-field ceilings.

Reference:
  * https://docs.slack.dev/reference/methods/chat.startStream
  * https://slack.dev/slack-thinking-steps-ai-agents/
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("gateway.slack_task_stream")

# Per-tool presentation metadata: (friendly verb, coarse category).
# The verb mirrors OpenClaw's progress lines ("Exec — which gog && gog
# --help", "Read — foo.py") instead of repeating raw tool names; the
# category feeds the auto-generated turn header ("Searched · edited files ·
# ran commands"). Unlisted tools fall back to their raw name / no category.
_TOOL_META: dict[str, tuple[str, Optional[str]]] = {
    "terminal": ("Exec", "ran commands"),
    "process": ("Process", "ran commands"),
    "execute_code": ("Run code", "ran commands"),
    "read_file": ("Read", "read files"),
    "search_files": ("Search files", "read files"),
    "write_file": ("Write", "edited files"),
    "patch": ("Edit", "edited files"),
    "web_search": ("Web search", "searched"),
    "web_extract": ("Fetch", "searched"),
    "x_search": ("X search", "searched"),
    "session_search": ("Recall", "searched"),
    "browser_navigate": ("Browse", "browsed"),
    "browser_click": ("Browse", "browsed"),
    "browser_type": ("Browse", "browsed"),
    "browser_snapshot": ("Browse", "browsed"),
    "browser_vision": ("Browse", "browsed"),
    "browser_scroll": ("Browse", "browsed"),
    "delegate_task": ("Delegate", "delegated"),
    "image_generate": ("Generate image", "generated images"),
    "vision_analyze": ("Analyze image", "analyzed images"),
    "text_to_speech": ("Speak", None),
    "todo": ("Plan", "planned"),
    "memory": ("Memory", "updated memory"),
    "skill_view": ("Load skill", "loaded skills"),
    "skills_list": ("List skills", "loaded skills"),
    "clarify": ("Ask", None),
    "cronjob": ("Schedule", None),
}

# Tools whose primary argument is file content — surface a snippet of it in
# the card's collapsible ``details`` field, mirroring mixlayer's Molly bot
# (each Write renders as a collapsible card previewing the file body).
_CONTENT_ARG_BY_TOOL = {
    "write_file": "content",
    "patch": "new_string",
    "execute_code": "code",
}


def tool_label(tool_name: str) -> str:
    """Friendly verb for a tool ("terminal" → "Exec"). Falls back to name."""
    meta = _TOOL_META.get(tool_name)
    return meta[0] if meta else tool_name


def _word_trim(text: str, limit: int) -> str:
    """Trim to ``limit`` chars at a word boundary, appending an ellipsis.

    Slicing raw delta buffers mid-word rendered titles like "run.py first.I've"
    — trim back to the last space when one exists reasonably close to the cap.
    """
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    sp = cut.rfind(" ")
    if sp > limit * 0.6:
        cut = cut[:sp]
    return cut.rstrip() + "…"


def clean_output_preview(result: Any, limit: int = 300) -> Optional[str]:
    """Turn a raw tool result into a compact, human-readable card preview.

    Tool results arrive as JSON-wrapped strings ({"output": "..."},
    {"result": "..."}), often with escaped newlines and a lot of bulk. The
    card ``output`` field is for a glanceable summary, not a data dump —
    unwrap the common envelopes, collapse whitespace, and keep the head.
    """
    if result is None:
        return None
    text = result
    if isinstance(text, str):
        s = text.strip()
        # Unwrap {"output": "..."} / {"result": "..."} / {"content": "..."}
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    for k in ("output", "result", "content", "text", "stdout"):
                        v = obj.get(k)
                        if isinstance(v, str) and v.strip():
                            s = v.strip()
                            break
            except Exception:
                pass
        text = s
    else:
        text = str(text)
    # Collapse escaped + real whitespace runs into single spaces.
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = " ".join(text.split())
    if not text:
        return None
    return _word_trim(text, limit)


def summarize_tool_title(tool_name: str, args: Any, result: Any) -> Optional[str]:
    """A short, human phrase describing what a tool call accomplished.

    Used to give collapsible cards a descriptive title beyond the raw
    argument echo (e.g. "Web search — 5 results for 'intel stock'" instead
    of the query string alone). Returns None to fall back to the arg preview.
    """
    if tool_name in {"web_search", "x_search"} and isinstance(result, str):
        n = result.count("\"url\":") or result.count("http")
        q = ""
        if isinstance(args, dict):
            q = str(args.get("query") or "").strip()
        if n:
            base = f"{n} results" + (f" for “{q[:40]}”" if q else "")
            return base
    if tool_name == "web_extract" and isinstance(args, dict):
        urls = args.get("urls")
        if isinstance(urls, list) and urls:
            try:
                dom = urlparse(str(urls[0])).netloc
            except Exception:
                dom = str(urls[0])
            extra = f" +{len(urls)-1}" if len(urls) > 1 else ""
            return f"{dom}{extra}"
    if tool_name in {"read_file", "write_file", "patch"} and isinstance(args, dict):
        path = args.get("path") or args.get("file_path")
        if path:
            return str(path).split("/")[-1]
    if tool_name == "terminal":
        clean = clean_output_preview(result, limit=90)
        if clean:
            return clean[:60]
    return None


def tool_category(tool_name: str) -> Optional[str]:
    """Human bucket for a tool ("web_search" → "searched"), or None."""
    meta = _TOOL_META.get(tool_name)
    return meta[1] if meta else None


def _split_complete_sentences(text: str) -> tuple[str, str]:
    """Split ``text`` into (complete sentences, trailing incomplete tail).

    Used to flush reasoning at sentence boundaries: the flush timer and
    tool-call finalize land at arbitrary stream positions, and cutting
    there splits a sentence across two 💭 cards (observed live: card N
    ending "…cron reminders.Both" with the sentence continuing on card
    N+1). The last sentence-terminating punctuation wins; everything
    after it is the held tail.
    """
    best = -1
    for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = text.rfind(sep)
        if idx > best:
            best = idx
    if best < 0:
        # Terminal punctuation at the very end counts as complete.
        if text.rstrip().endswith((".", "!", "?", ":")):
            return text, ""
        return "", text
    cut = best + 1  # include the punctuation, not the following space
    return text[:cut], text[cut:].lstrip()


def tool_details_from_args(tool_name: str, args: Any) -> Optional[str]:
    """Optional collapsible-body preview extracted from the tool's args."""
    key = _CONTENT_ARG_BY_TOOL.get(tool_name)
    if not key or not isinstance(args, dict):
        return None
    val = args.get(key)
    if not isinstance(val, str) or not val.strip():
        return None
    return val.strip()[:300]


# Tools whose results naturally carry URL attributions worth surfacing as
# clickable ``sources`` on the task card.
_SOURCE_TOOLS = {
    "web_search", "web_extract", "x_search", "browser_navigate",
    # image_generate returns a hosted result URL — chip links to the image.
    "image_generate",
}
_URL_RE = re.compile(r"https?://[^\s'\"\)\]>,\\]+")

# MCP tools return failures as result TEXT without raising (is_error stays
# False on the lifecycle event), so a 403 would render with a ✓. Cheap sniff
# on the head of the result for common error shapes.
_ERROR_TEXT_RE = re.compile(
    r'^\s*(?:\{"result":\s*")?(?:Error\b|\[?Error\]?[:\s]|HTTP (?:4|5)\d\d)'
)


def result_looks_like_error(result: Any) -> bool:
    """True when a tool result *string* reads as an error despite ok status."""
    if not isinstance(result, str):
        return False
    head = result.strip()[:80]
    return bool(head and _ERROR_TEXT_RE.match(head))


def tool_sources(tool_name: str, args: Any, result_text: Any, limit: int = 3) -> Optional[List[dict]]:
    """Build task_update ``sources`` entries (clickable URL chips) for a call.

    URLs come from the args (web_extract's url list, browser_navigate's
    target) and from the result body (search hits). Deduped by URL, capped
    at ``limit``, labeled with the domain.
    """
    if tool_name not in _SOURCE_TOOLS:
        return None
    urls: List[str] = []
    if isinstance(args, dict):
        arg_urls = args.get("urls")
        if isinstance(arg_urls, list):
            urls.extend(u for u in arg_urls if isinstance(u, str))
        for key in ("url", "image_url"):
            val = args.get(key)
            if isinstance(val, str) and val.startswith("http"):
                urls.append(val)
    if isinstance(result_text, str):
        urls.extend(_URL_RE.findall(result_text))
    out: List[dict] = []
    seen = set()
    for u in urls:
        u = u.rstrip(".,;:!?")
        if not u.startswith("http"):
            continue
        try:
            domain = urlparse(u).netloc or u
        except Exception:
            domain = u
        # Dedupe by domain, not full URL — two pages from the same site
        # render as identical-looking chips (observed: doubled www.intc.com).
        if domain in seen:
            continue
        seen.add(domain)
        out.append({"type": "url", "text": domain[:60], "url": u})
        if len(out) >= limit:
            break
    return out or None


class SlackTaskStream:
    """Manage the lifecycle of one Slack native task-card streaming message.

    One instance covers a single agent turn: ``ensure_started()`` opens the
    stream lazily on the first tool event, ``task_started``/``task_finished``
    push ``task_update`` chunks keyed by a stable per-tool-call id, and
    ``stop()`` closes the stream when the turn completes.

    Long turns outlive a single streamed message — Slack closes streams
    after ~5 minutes (``message_not_in_streaming_state``) and caps the
    cumulative message size (``msg_too_long``). Rather than falling back to
    markdown, the stream ROLLS OVER: the current card is closed cleanly and
    a fresh streaming message continues the timeline, replaying any
    still-in-progress tasks so their completion lands on the new card.
    Rollover happens proactively (age/size thresholds, so the user never
    sees an error) and reactively (on either recoverable API error).

    All public methods swallow their own errors (log at info) and flip
    ``self.disabled`` on unrecoverable failure so the caller can fall back
    to the markdown progress path for the rest of the turn instead of
    retrying a broken stream on every subsequent event.
    """

    # Tuning defaults (overridable per-instance via __init__, which run.py
    # feeds from the display config keys tool_progress_native_*):
    #
    # Proactive rollover thresholds, measured live 2026-07-05 via probe
    # (ehoy scripts/carnie/slack_stream_probe.py): a stream dies ~306s after startStream
    # even with appends every 20s — an ABSOLUTE lifetime, not inactivity —
    # so roll at 240s (~80%, margin for jitter/slow appends). Cumulative
    # size probed clean past 61,920 chars (earlier msg_too_long failures
    # were single oversized chunks, since capped per-field), so the char
    # threshold is a loose backstop, not the binding constraint.
    ROLLOVER_MAX_AGE_S = 240.0
    ROLLOVER_MAX_CHARS = 40_000
    # Cap on accumulated 💭 reasoning text per card. 0 = uncapped, bounded
    # only by SLACK_FIELD_CEILING below.
    REASONING_MAX_CHARS = 0
    # Ceiling on the locally-kept reasoning copy (used for rollover replay).
    # Probed 2026-07-05: single details chunks up to 32k accepted with no
    # error — the documented 12k limit applies to markdown_text, not
    # task_update details. 30k keeps one card's replay chunk comfortably
    # under the rollover size budget while being far beyond any realistic
    # single thinking burst.
    SLACK_FIELD_CEILING = 30_000
    # Minimum accumulated chars before a 💭 card is opened/updated.
    # The flush timer can cut a burst mid-word (observed: a card containing
    # just "I"); tiny fragments carry no signal, so hold them until the
    # burst has substance. A pending fragment below this at finalize time
    # is carried into the next burst rather than emitted as its own card.
    REASONING_MIN_CHARS = 40
    # Per-tool result preview length on finished cards.
    OUTPUT_PREVIEW_CHARS = 120
    # Runaway guard: a turn pathological enough to need more fresh streams
    # than this should fall back to markdown instead. Sized generously —
    # age-based rollover alone consumes one per ~4 min, so a legitimate
    # 40-60 min turn can use 10-15 (observed turns run 20+ min).
    MAX_ROLLOVERS = 20

    def __init__(
        self,
        client: Any,
        channel: str,
        thread_ts: str,
        task_display_mode: str = "plan",
        recipient_team_id: Optional[str] = None,
        recipient_user_id: Optional[str] = None,
        rollover_age_s: Optional[float] = None,
        rollover_chars: Optional[int] = None,
        reasoning_chars: Optional[int] = None,
        output_chars: Optional[int] = None,
    ) -> None:
        self.client = client
        self.channel = channel
        self.thread_ts = thread_ts
        self.recipient_team_id = recipient_team_id
        self.recipient_user_id = recipient_user_id
        self.task_display_mode = task_display_mode
        # Config-driven tuning (None → class default). reasoning cap of 0
        # means uncapped; it is still clamped to SLACK_FIELD_CEILING.
        if rollover_age_s is not None and rollover_age_s > 0:
            self.ROLLOVER_MAX_AGE_S = float(rollover_age_s)
        if rollover_chars is not None and rollover_chars > 0:
            self.ROLLOVER_MAX_CHARS = int(rollover_chars)
        if reasoning_chars is not None:
            self.REASONING_MAX_CHARS = max(0, int(reasoning_chars))
        if output_chars is not None and output_chars > 0:
            self.OUTPUT_PREVIEW_CHARS = int(output_chars)
        self.ts: Optional[str] = None
        self.disabled = False
        self._started = False
        self._stopped = False
        # Turn stats for the final header title ("4 steps · 12s").
        self._task_count = 0
        self._total_duration = 0.0
        # Distinct tool-category buckets seen this turn, in first-seen order,
        # for the auto-generated header ("Searched · edited files · …").
        self._categories: list[str] = []
        self._last_header: str = ""
        # Start-time title per task id: the finish update falls back to it
        # when no descriptive summary is available (title REPLACES on each
        # task_update; details APPENDS — see module docstring).
        self._titles: dict[int, str] = {}
        # Interleaved reasoning cards: each burst of thinking between tool
        # calls gets its own 💭 card in the timeline (updated in place while
        # the burst continues, finalized when the next tool starts). The
        # title carries the rolling tail of the thought; the card's DETAILS
        # carries the full burst text (titles are capped ~255 by Slack,
        # details can hold far more — that's where full reasoning lives).
        self._reasoning_open_id: Optional[str] = None
        self._reasoning_title: str = ""
        self._reasoning_details: str = ""
        self._reasoning_unsent: str = ""
        self._reasoning_carry: str = ""
        self._reasoning_count = 0
        # Per-subagent state (card id → {tools, number, start-time}) for the
        # numbered, timed delegate cards.
        self._subagents: dict[str, dict[str, Any]] = {}
        # Rollover state: when the current streamed message ages/fills out,
        # it's closed and a fresh one continues the timeline. Tasks still
        # in_progress at rollover are tracked so they can be replayed onto
        # the new card (their task ids don't exist there otherwise).
        self._stream_opened_at = 0.0
        self._sent_chars = 0
        # Last-sent size per card id, for net-delta size accounting (a
        # task_update replaces its card, so only growth counts).
        self._chunk_sizes: dict[str, int] = {}
        self._rollovers = 0
        self._in_progress: dict[str, dict[str, Any]] = {}  # task_id → last-sent chunk
        # Serializes the open: tool events arrive back-to-back and each
        # task_started() awaits ensure_started(), so without a lock two
        # coroutines can both pass the ``_started`` check before either
        # completes chat.startStream — opening two streams.
        self._start_lock = asyncio.Lock()
        # Serializes appends (OpenClaw does the same via a promise chain):
        # events are scheduled as independent coroutines, so without this a
        # fast tool's "finished" append could overtake its own "started"
        # append mid-HTTP and leave the card stuck showing in_progress.
        # asyncio.Lock wakes waiters FIFO, so scheduling order is preserved.
        self._send_lock = asyncio.Lock()

    async def ensure_started(self) -> bool:
        """Open the stream once (idempotent). Returns True if usable."""
        if self.disabled:
            return False
        if self._started:
            return True
        async with self._start_lock:
            return await self._start_locked()

    async def _start_locked(self) -> bool:
        if self.disabled:
            return False
        if self._started:
            return True
        try:
            await self._open_stream()
            self._started = True
            return True
        except Exception as e:  # SlackApiError or any transport failure
            logger.warning("chat.startStream failed, disabling native task cards: %s", e)
            self.disabled = True
            return False

    async def _open_stream(self) -> None:
        """Raw chat.startStream call — shared by first open and rollover.

        Raises on failure; callers decide whether that's fatal. Takes no
        locks (callers already hold whichever lock is appropriate).
        """
        kwargs: dict[str, Any] = {
            "channel": self.channel,
            "thread_ts": self.thread_ts,
            "task_display_mode": self.task_display_mode,
        }
        # Slack requires BOTH recipient ids for streams opened by bot
        # tokens, despite the API docs listing them as optional
        # (empirically: omitting team → missing_recipient_team_id,
        # omitting user → missing_recipient_user_id). Pass when known.
        if self.recipient_team_id:
            kwargs["recipient_team_id"] = self.recipient_team_id
        if self.recipient_user_id:
            kwargs["recipient_user_id"] = self.recipient_user_id
        result = await self.client.chat_startStream(**kwargs)
        self.ts = result.get("ts") if result else None
        if not self.ts:
            raise RuntimeError("chat.startStream returned no ts")
        self._stream_opened_at = time.monotonic()
        self._sent_chars = 0
        self._chunk_sizes = {}

    async def task_started(
        self,
        index: int,
        tool_name: str,
        preview: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Emit/update a task_update chunk for a tool call that just started.

        The title carries a friendly verb plus the args preview
        ("Exec — grep foo …") so each line is scannable without expanding —
        mirroring OpenClaw's progress lines. ``details`` (optional) fills the
        collapsible card body, e.g. file content for Write/Edit calls.
        """
        if self.disabled:
            return
        if not await self.ensure_started():
            return
        label = tool_label(tool_name)
        title = f"{label} — {preview}" if preview else label
        title = title[:250]
        self._titles[index] = title
        self._task_count += 1
        # Track tool categories for the auto-generated turn header.
        cat = tool_category(tool_name)
        if cat and cat not in self._categories:
            self._categories.append(cat)
        # Close out any open 💭 card first so the timeline reads
        # thought ✓ → tool, in order.
        await self._finalize_reasoning_card()
        await self._append_task_update(
            index, title, status="in_progress", details=details,
        )
        # Header = an auto-generated summary of what the turn is doing,
        # composed from the distinct tool categories seen so far
        # ("Searched · edited files · ran commands"), refreshed as new
        # categories appear. Beats echoing the first tool's raw args
        # (the old behavior rendered "Exec — date +%H:%M + 1 command").
        await self._refresh_turn_header()

    async def _refresh_turn_header(self) -> None:
        """Set the collapsible header to a phrase summarizing the turn."""
        if not self._categories:
            return
        # Capitalize the first bucket, join the rest with " · ".
        cats = list(self._categories)
        cats[0] = cats[0][:1].upper() + cats[0][1:]
        header = " · ".join(cats)
        if header != self._last_header:
            self._last_header = header
            await self.set_plan_title(header)

    async def task_finished(
        self,
        index: int,
        tool_name: str,
        duration: float = 0.0,
        ok: bool = True,
        output: Optional[str] = None,
        sources: Optional[List[dict]] = None,
        summary: Optional[str] = None,
    ) -> None:
        """Emit/update a task_update chunk for a tool call that just finished."""
        if self.disabled:
            return
        if not await self.ensure_started():
            return
        # Prefer a descriptive summary ("Read → 5 results for X") over the
        # start-time arg echo; fall back to the stored start title.
        if summary:
            base = f"{tool_label(tool_name)} → {summary}"
        else:
            base = self._titles.get(index) or tool_label(tool_name)
        title = f"{base} · {duration:.1f}s"[:250] if duration else base[:250]
        # Failed tool calls render as "complete" with a ✗ suffix instead of
        # Slack's "error" status: the pink warning triangle reads as "the
        # agent broke" when in reality a failed call is routine — the agent
        # sees the error and adapts. Reserving the triangle for genuine
        # breakage (stream abandoned mid-turn) keeps it meaningful.
        if not ok:
            title = f"{base} · ✗ failed"[:250]
        self._total_duration += duration or 0.0
        # +30 slack vs the run.py-side preview cap so a summary suffix fits.
        out = str(output)[: self.OUTPUT_PREVIEW_CHARS + 30] if output else None
        # details APPEND server-side (measured 2026-07-05), so the start-time
        # content preview persists on its own — re-sending it here would
        # duplicate it. Send no details on the finish update.
        await self._append_task_update(
            index, title, status="complete",
            output=out, sources=sources,
        )

    async def subagent_event(
        self,
        event_type: str,
        subagent_key: str,
        goal: str = "",
        tool_name: Optional[str] = None,
        ok: bool = True,
        number: Optional[int] = None,
    ) -> None:
        """Render delegated subagents as their own live cards.

        delegate_task relays child lifecycle events to the parent's progress
        callback (subagent.start / subagent.tool / subagent.complete). Each
        child gets one card keyed by subagent_id so parallel children update
        independently. The card shows a stable number (#1, #2…), the goal,
        a live tool count, and elapsed time. Time advances on each relayed
        event rather than ticking continuously — Slack cards only redraw when
        a chunk is sent, so "rolling" means "updates whenever the child does
        something", which is the useful signal (a frozen count == stuck).
        """
        if self.disabled:
            return
        if not await self.ensure_started():
            return
        sid = f"sub_{subagent_key}"
        label = (goal or "subagent").strip()
        if len(label) > 70:
            label = label[:67] + "…"
        st = self._subagents.setdefault(
            sid, {"tools": [], "n": number, "t0": time.monotonic()}
        )
        if number is not None:
            st["n"] = number
        num = f"#{st['n']} " if st.get("n") is not None else ""
        elapsed = time.monotonic() - st["t0"]

        def _title(status_suffix: str = "") -> str:
            ntools = len(st["tools"])
            bits = []
            if ntools:
                bits.append(f"{ntools} tool{'s' if ntools != 1 else ''}")
            if elapsed >= 1:
                bits.append(f"{elapsed:.0f}s")
            meta = f" · {' · '.join(bits)}" if bits else ""
            return f"🔀 Delegate {num}— {label}{meta}{status_suffix}"[:250]

        if event_type == "subagent.start":
            self._task_count += 1
            await self._append_raw_task(sid, _title(), status="in_progress")
        elif event_type == "subagent.tool" and tool_name:
            st["tools"].append(tool_label(tool_name))
            # details APPEND server-side — send only the new tool label and
            # let Slack accumulate the trail ("A → B → C" grows one arrow
            # per update instead of re-sending the whole trail).
            step = tool_label(tool_name)
            delta = step if len(st["tools"]) == 1 else f" → {step}"
            await self._append_raw_task(
                sid, _title(), status="in_progress", details=delta,
            )
        elif event_type == "subagent.complete":
            suffix = "" if ok else " · ✗ failed"
            await self._append_raw_task(sid, _title(suffix), status="complete")

    async def reasoning_update(self, text: str) -> None:
        """Render the model's thinking as interleaved 💭 cards in the timeline.

        Each burst of reasoning between tool calls gets its own card,
        positioned exactly where the thinking happened (thought → tool →
        thought → tool, the Anthropic/OpenAI webapp rhythm). The open card
        updates in place while the burst continues; the next task_started
        finalizes it. Header (plan title) stays owned by the first tool call
        — an earlier version routed reasoning through the header, which
        wiped it (plan_update replaces the title wholesale).

        Field split (title vs details): a card's ``title`` is the
        always-visible line and Slack hard-caps it (~255); ``details`` is
        the collapsible body (~12k). The title is set ONCE per card from
        the head of the thought and never rewritten — an earlier version
        showed the rolling tail, which churned on every flush and read as
        confusing mid-sentence fragments (user-reported 2026-07-05).
        Everything overflowing the title lives in details, which carries
        the full accumulated burst (uncapped by default).

        Before the stream opens (thinking that precedes the first tool call
        — i.e. every turn's opening thought), the burst is BUFFERED rather
        than dropped: state is updated but no API call is made, and the
        pending 💭 card is flushed by the first task_started, so the card
        still reads thought ✓ → tool. A turn with zero tool calls never
        opens a stream, so its reasoning is never rendered — intentional.
        """
        if self.disabled:
            return
        line = " ".join(str(text).split())
        if not line:
            return
        if self._reasoning_open_id is None:
            self._reasoning_count += 1
            self._reasoning_open_id = f"think{self._reasoning_count}"
            # Carry any sub-threshold fragment from the previous burst
            # (see _finalize_reasoning_card) instead of starting empty.
            self._reasoning_details = self._reasoning_carry
            self._reasoning_unsent = self._reasoning_carry
            self._reasoning_carry = ""
            self._reasoning_title = ""
        # ``details`` APPENDS across task_update chunks with the same id —
        # measured live 2026-07-05 (probe: two updates "AAA"/"BBB" stored
        # as "AAABBB"); title/status REPLACE. So each flush must send ONLY
        # the not-yet-sent delta, and Slack accumulates server-side.
        # Sending the full running text each flush rendered the
        # burst₁+(burst₁+burst₂)+… staircase duplication.
        # _reasoning_details keeps the full local copy (rollover replay +
        # finalize-before-stream-opens need it); _reasoning_unsent is the
        # pending tail.
        self._reasoning_details = (self._reasoning_details + " " + line).strip()
        self._reasoning_unsent = (self._reasoning_unsent + " " + line).strip()
        cap = self.SLACK_FIELD_CEILING
        if self.REASONING_MAX_CHARS > 0:
            cap = min(self.REASONING_MAX_CHARS, cap)
        if len(self._reasoning_details) > cap:
            clipped = self._reasoning_details[-cap:]
            sp = clipped.find(" ")
            self._reasoning_details = "…" + clipped[sp + 1 if 0 <= sp < 40 else 0:]
        # Hold sub-threshold bursts: the flush timer can slice mid-word
        # ("I" as a whole card). Don't open/update the card until the
        # burst has substance; held text flushes with the next update or
        # carries into the next burst at finalize.
        if len(self._reasoning_details) < self.REASONING_MIN_CHARS:
            return
        # Title: short TLDR-style header (Claude-app rhythm — headers are
        # sub-sentence). First sentence of the thought, capped ~80 chars,
        # set once and never rewritten; the full text lives in details.
        if not self._reasoning_title:
            head = self._reasoning_details
            for sep in (". ", "! ", "? ", " — "):
                idx = head.find(sep)
                if 0 < idx < 120:
                    head = head[: idx + 1].rstrip(" —")
                    break
            self._reasoning_title = f"💭 {_word_trim(head, 80)}"[:250]
        if not self._started:
            return  # buffered — flushed by the first task_started
        # Flush at sentence boundaries only: send the complete-sentence
        # prefix of the unsent buffer, hold the incomplete tail for the
        # next flush (or the finalize/carry path). Cutting at raw timer
        # positions split sentences across cards. Join with a TRAILING
        # space — probe #4: Slack preserves trailing whitespace at chunk
        # joins but strips leading whitespace at some element boundaries
        # (the "reminders.Both" jam).
        sendable, tail = _split_complete_sentences(self._reasoning_unsent)
        if not sendable:
            return
        self._reasoning_unsent = tail
        await self._append_raw_task(
            self._reasoning_open_id, self._reasoning_title,
            status="in_progress", details=sendable.rstrip() + " ",
        )

    async def _finalize_reasoning_card(self) -> None:
        """Settle the open 💭 card when the next tool starts.

        Sends any still-unsent text (complete or not — the burst is over,
        so the remainder belongs to THIS card; only sub-threshold bursts
        that never rendered are carried into the next burst instead of
        emitting a fragment card).
        """
        if self._reasoning_open_id is None:
            return
        if len(self._reasoning_details) < self.REASONING_MIN_CHARS:
            # Nothing was ever sent for this burst — roll it forward.
            self._reasoning_carry = self._reasoning_details
            self._reasoning_open_id = None
            self._reasoning_details = ""
            self._reasoning_unsent = ""
            return
        rid, title = self._reasoning_open_id, self._reasoning_title
        tail, self._reasoning_unsent = self._reasoning_unsent, ""
        self._reasoning_open_id = None
        self._reasoning_details = ""
        await self._append_raw_task(
            rid, title, status="complete",
            details=(tail.rstrip() + " ") if tail.strip() else None,
        )

    async def set_plan_title(self, title: str) -> None:
        """Set/update the card's collapsible header via a plan_update chunk.

        Slack's default header ("Thinking completed" / "Something went
        wrong") is generic; a plan_update replaces it with our own text.
        """
        if self.disabled or not self._started:
            return
        try:
            async with self._send_lock:
                if self.disabled:
                    return
                await self.client.chat_appendStream(
                    channel=self.channel,
                    ts=self.ts,
                    chunks=[{"type": "plan_update", "title": str(title)[:250]}],
                )
        except Exception as e:
            logger.info("plan_update failed (non-fatal): %s", e)

    async def _append_task_update(
        self,
        index: int,
        title: str,
        *,
        status: str,
        details: Optional[str] = None,
        output: Optional[str] = None,
        sources: Optional[List[dict]] = None,
    ) -> None:
        await self._append_raw_task(
            f"t{index}", title, status=status,
            details=details, output=output, sources=sources,
        )

    async def _append_raw_task(
        self,
        task_id: str,
        title: str,
        *,
        status: str,
        details: Optional[str] = None,
        output: Optional[str] = None,
        sources: Optional[List[dict]] = None,
    ) -> None:
        try:
            chunk: dict[str, Any] = {
                "type": "task_update",
                "id": task_id,
                "title": title,
                "status": status,
            }
            if details:
                chunk["details"] = details
            if output:
                chunk["output"] = output
            if sources:
                chunk["sources"] = sources
            # Track open tasks for rollover replay: an in_progress card must
            # be re-created on the fresh stream or its completion update
            # would reference a task id that doesn't exist there.
            if status == "in_progress":
                self._in_progress[task_id] = dict(chunk)
            else:
                self._in_progress.pop(task_id, None)
            async with self._send_lock:
                if self.disabled:
                    return
                # Proactive rollover: refresh the stream *before* Slack's
                # ~5-min stream lifetime or per-message size cap kill it,
                # so the user never sees an error state.
                if (
                    time.monotonic() - self._stream_opened_at > self.ROLLOVER_MAX_AGE_S
                    or self._sent_chars > self.ROLLOVER_MAX_CHARS
                ):
                    await self._rollover_locked()
                try:
                    await self._send_chunk_locked(chunk)
                except Exception as e:
                    # Reactive rollover: both errors mean "this message can't
                    # take more content" — recoverable with a fresh stream.
                    if any(m in str(e) for m in ("message_not_in_streaming_state", "msg_too_long")):
                        await self._rollover_locked()
                        await self._send_chunk_locked(chunk)
                    else:
                        raise
        except Exception as e:
            logger.warning("chat.appendStream failed, disabling native task cards: %s", e)
            self.disabled = True

    async def _send_chunk_locked(self, chunk: dict[str, Any]) -> None:
        """Send one chunk on the current stream. Caller holds _send_lock."""
        await self.client.chat_appendStream(
            channel=self.channel,
            ts=self.ts,
            chunks=[chunk],
        )
        # Approximate the message-size budget by NET RENDERED content: a
        # task_update with a known id REPLACES that card, so count only the
        # size delta vs what that card previously held. Summing raw appends
        # would explode on 💭 cards (each update re-sends the whole
        # accumulated burst) and trigger spurious rollovers.
        size = sum(len(str(v)) for v in chunk.values())
        key = chunk.get("id") or f"__{chunk.get('type', 'chunk')}__"
        prev = self._chunk_sizes.get(key, 0)
        self._chunk_sizes[key] = size
        self._sent_chars += max(0, size - prev)

    async def _rollover_locked(self) -> None:
        """Close the current stream and continue on a fresh one.

        Caller holds _send_lock. Raises if the fresh stream can't be opened
        (caller's outer except then disables cards — correct: no stream to
        write to). The old card is stopped best-effort with a continuation
        footer; still-open tasks and the turn header are replayed onto the
        new card so the timeline visually continues.
        """
        self._rollovers += 1
        if self._rollovers > self.MAX_ROLLOVERS:
            raise RuntimeError(f"exceeded {self.MAX_ROLLOVERS} stream rollovers this turn")
        try:
            # Settle still-open tasks on the OLD card before closing it —
            # stopping a stream with in_progress tasks makes Slack stamp
            # them with red warning triangles ("something went wrong"),
            # which reads as breakage when it's just a continuation
            # (observed live 2026-07-05). Mark them complete with a ⤵
            # suffix here; they're replayed as in_progress on the fresh
            # card below.
            for tid, chunk in list(self._in_progress.items()):
                settled = dict(chunk)
                settled["status"] = "complete"
                settled["title"] = f"{str(chunk.get('title', ''))[:240]} ⤵"
                try:
                    await self.client.chat_appendStream(
                        channel=self.channel, ts=self.ts, chunks=[settled],
                    )
                except Exception:
                    break  # old stream already dead — skip the rest
            await self.client.chat_stopStream(
                channel=self.channel,
                ts=self.ts,
                blocks=[{
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": "⤵ continued below"}],
                }],
            )
        except Exception as e:
            # Old stream may already be dead (that's why we're here).
            logger.info("rollover: closing old stream failed (non-fatal): %s", e)
        await self._open_stream()
        logger.info(
            "rollover: continued task cards on fresh stream ts=%s (rollover #%d)",
            self.ts, self._rollovers,
        )
        # Replay the turn header and any in-flight tasks on the new card.
        # The fresh card starts empty, so replayed chunks need FULL content:
        # for the open 💭 card the tracked chunk only carries the last sent
        # delta (details append server-side) — substitute the full local
        # accumulated burst.
        replay: List[dict] = []
        if self._last_header:
            replay.append({"type": "plan_update", "title": self._last_header[:250]})
        for c in self._in_progress.values():
            chunk = dict(c)
            if (
                self._reasoning_open_id is not None
                and chunk.get("id") == self._reasoning_open_id
                and self._reasoning_details
            ):
                chunk["details"] = self._reasoning_details
                self._reasoning_unsent = ""
            replay.append(chunk)
        for chunk in replay:
            await self._send_chunk_locked(chunk)

    async def stop(self, final_text: Optional[str] = None) -> None:
        """Close the streaming message. No-op if never started or already stopped."""
        if self._stopped or not self._started or self.disabled:
            self._stopped = True
            return
        # Settle any open 💭 card so nothing is left in_progress at close
        # (Slack renders unfinished tasks as warnings + "Something went wrong").
        try:
            await self._finalize_reasoning_card()
        except Exception:
            pass
        self._stopped = True
        try:
            # Wait for any in-flight append to land before closing, so the
            # last task's status update isn't racing chat.stopStream.
            async with self._send_lock:
                kwargs: dict[str, Any] = {"channel": self.channel, "ts": self.ts}
                if final_text:
                    kwargs["markdown_text"] = final_text
                # Footer: a context block with turn stats, attached to the
                # final message via stopStream's blocks parameter. Plain
                # context blocks need no interactivity handler (unlike
                # buttons, which require an events endpoint to be useful).
                if self._task_count:
                    _secs = (
                        f" · {self._total_duration:.0f}s tool time"
                        if self._total_duration >= 1 else ""
                    )
                    _plural = "tool call" if self._task_count == 1 else "tool calls"
                    kwargs["blocks"] = [
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"⚙ {self._task_count} {_plural}{_secs}",
                                }
                            ],
                        }
                    ]
                await self.client.chat_stopStream(**kwargs)
        except Exception as e:
            logger.info("chat.stopStream failed: %s", e)


__all__ = [
    "SlackTaskStream",
    "clean_output_preview",
    "result_looks_like_error",
    "summarize_tool_title",
    "tool_category",
    "tool_details_from_args",
    "tool_label",
    "tool_sources",
]
