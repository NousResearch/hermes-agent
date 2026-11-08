"""Rich Slack task cards (Slack "Thinking Steps") for gateway tool progress.

Slack's chat.startStream / chat.appendStream / chat.stopStream trio lets a bot render tool-call
progress as a native, collapsible task-card timeline inside a message. The adapter's stock native
card (``platforms.slack.extra.native_task_cards``) renders one row per tool call. This module is the
opt-in richer publisher (``display.tool_progress_native``): per-subagent child streams, live and
completed reasoning rows, tool detail / source / output rows, proactive stream rollover, secret
redaction before anything reaches Slack, and a guarded client that fences every write against stopped,
stale or interrupted turns.

Layering contract: ``SlackTaskStream`` owns presentation and a single-stream lifecycle;
``RichTaskCardSession`` publishes the events ``TurnRunner`` already queues for the stock card (plus
reasoning and subagent events when the flag is on). The stock publisher and this one are mutually
exclusive for a turn and share the stock ID correlation, text fallback and egress-refusal state
(see ``gateway/run_turn_task_cards.py``). Direct Slack transport uses the adapter's workspace
resolver and outbound guard; relay connectors keep their native adapter seam and the stock card.

Opt-in and strictly additive: every other platform, and Slack installs that don't set the flag, keep
the stock behaviour untouched. Tuning knobs (rollover thresholds, reasoning/output caps) resolve like
any other display setting, see ``gateway/display_config.py``.

Slack API limits this relies on, measured against the live API:
  * a streamed message dies ~306 s after startStream even with active appends (an absolute
    lifetime, not inactivity), hence proactive rollover;
  * msg_too_long is per chunk (a >12k field), NOT cumulative: 62k chars of cumulative task_update
    content passed clean, hence per-field ceilings.

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
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger("gateway.slack_task_stream")


def _safe_card_error(error: BaseException) -> str:
    """Redact vendor errors before they reach logs or progress fallbacks."""
    from agent.redact import redact_for_egress

    return redact_for_egress(str(error))


# Per-tool presentation metadata: (friendly verb, coarse category).
# The verb ("Exec — grep foo", "Read — foo.py") replaces the raw tool name;
# the category feeds the auto-generated turn header ("Searched · edited files ·
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
            except json.JSONDecodeError:
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
            except ValueError:
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
        best = max(best, idx)
    if best < 0:
        # Terminal punctuation at the very end counts as complete.
        if text.rstrip().endswith((".", "!", "?", ":")):
            return text, ""
        return "", text
    cut = best + 1  # include the punctuation, not the following space
    return text[:cut], text[cut:].lstrip()


def _reasoning_body_after_preview(text: str, preview: str) -> str:
    """Remove only the exact source prefix already shown in a prose title."""
    normalized: list[str] = []
    source_ends: list[int] = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if normalized and not in_whitespace:
                normalized.append(" ")
                source_ends.append(index + 1)
            elif normalized and in_whitespace:
                source_ends[-1] = index + 1
            in_whitespace = True
        else:
            normalized.append(char)
            source_ends.append(index + 1)
            in_whitespace = False
        if len(normalized) >= len(preview):
            break

    rendered = "".join(normalized)
    prefix = preview
    if not rendered.startswith(prefix) and preview.endswith("…") and rendered.startswith(preview[:-1]):
        # _reasoning_preview adds this ellipsis when it trims the source.
        prefix = preview[:-1]
    if not prefix or not rendered.startswith(prefix):
        return text
    boundary = source_ends[len(prefix) - 1]
    return text[boundary:]
_REASONING_INLINE_SPAN_RE = re.compile(
    r"`[^`]+`"
    r"|(?<!!)\[[^\]]+\]\([^()\s]+(?:\([^()]*\)[^()\s]*)*\)"
    r"|(?:\*\*|__)(.+?)(?:\*\*|__)"
    r"|~~.+?~~"
    r"|(?<![\*_])(?:\*|_)(?![\*_\s]).+?(?<![\*_\s])(?:\*|_)(?![\*_])"
)


def _reasoning_preview(text: str) -> str:
    """Keep only a plain source prefix in the title; rich spans belong in details."""
    head = " ".join(text.split())
    # Inspect the full text before sentence clipping: punctuation inside a span
    # must not hide its closing delimiter from the boundary decision.
    span = _REASONING_INLINE_SPAN_RE.search(head)
    plain = head[:span.start()].rstrip() if span else head
    if not plain and span:
        return "Thinking…"
    clipped_sentence = False
    for sep in (". ", "! ", "? ", " — "):
        idx = plain.find(sep)
        if 0 < idx < 120:
            plain = plain[:idx + 1].rstrip(" —")
            clipped_sentence = True
            break
    preview = _word_trim(plain, 80)
    if span and not clipped_sentence and preview == plain and not preview.endswith("…"):
        preview += "…"
    return preview


_REASONING_HEADING_RE = re.compile(r"\*\*([^*\n]+)\*\*|^#{1,6}[ \t]+([^\n]+)", re.MULTILINE)


def _completed_reasoning_rows(text: str) -> list[tuple[str, str]]:
    """Separate explicit summary headings only after whole-burst redaction.

    Providers may concatenate heading-only parts without a delimiter. Split
    those only when the ENTIRE burst is headings; bold phrases inside prose
    are not new steps. Unrecognized text remains intact in the body.
    """
    headings = list(_REASONING_HEADING_RE.finditer(text))
    only_headings = headings and not _REASONING_HEADING_RE.sub("", text).strip()
    parts = [match[0] for match in headings] if only_headings else [text]
    rows = []
    for part in parts:
        heading = _REASONING_HEADING_RE.match(part)
        label = (heading[1] or heading[2]).strip() if heading else ""
        remainder = part[heading.end():].lstrip(" \t") if heading else ""
        if heading and label and (not remainder or remainder.startswith(("\n", "\r\n"))):
            # Strip heading-line spacing separately so body indentation survives.
            body = remainder.lstrip("\r\n")
            title = _word_trim(label, 80)
            if title != label:
                # Only the undisplayed suffix belongs in details, not a second
                # copy of the visible heading. Preserve overflow even without prose.
                body = label[len(title) - 1:] + ("\n\n" + body if body else "")
            rows.append((title, body))
        else:
            rows.append(("", part))
    return rows


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


def tool_sources(tool_name: str, args: Any, result_text: Any, limit: int = 3) -> Optional[list[dict]]:
    """Build task_update ``sources`` entries (clickable URL chips) for a call.

    URLs come from the args (web_extract's url list, browser_navigate's
    target) and from the result body (search hits). Deduped by URL, capped
    at ``limit``, labeled with the domain.
    """
    if tool_name not in _SOURCE_TOOLS:
        return None
    urls: list[str] = []
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
    out: list[dict] = []
    seen = set()
    for u in urls:
        u = u.rstrip(".,;:!?")
        if not u.startswith("http"):
            continue
        try:
            domain = urlparse(u).netloc or u
        except ValueError:
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

    # Tuning defaults (overridable per-instance via __init__, which
    # RichTaskCardSession feeds from the display config keys tool_progress_native_*):
    #
    # Proactive rollover thresholds, measured live: a stream dies ~306s after startStream
    # even with appends every 20s — an ABSOLUTE lifetime, not inactivity —
    # so roll at 290s (~95% — post drain-fix, tool events arrive densely
    # enough; a proactive miss triggers the tested reactive path). Cumulative
    # size probed clean past 61,920 chars (earlier msg_too_long failures
    # were single oversized chunks, since capped per-field), so the char
    # threshold is a loose backstop, not the binding constraint.
    ROLLOVER_MAX_AGE_S = 290.0
    ROLLOVER_MAX_CHARS = 40_000
    # Cap on accumulated 💭 details, including separator spaces. A positive
    # budget covers the logical card/burst even across continuation messages;
    # 0 preserves uncapped bursts (the per-chunk ceiling below still applies).
    REASONING_MAX_CHARS = 0
    # Ceiling on the locally-kept reasoning copy and each details chunk.
    # Probed live: single details chunks up to 32k accepted with no
    # error — the documented 12k limit applies to markdown_text, not
    # task_update details. This is a probed ceiling, not a documented
    # public API guarantee.
    SLACK_FIELD_CEILING = 30_000
    # Minimum accumulated chars before a 💭 card is opened/updated.
    # The flush timer can cut a burst mid-word (observed: a card containing
    # just "I"); tiny fragments carry no signal, so hold them until the
    # burst has substance. A semantic/tool/turn boundary settles even a short
    # burst: carrying it forward merges unrelated steps or loses the last one.
    REASONING_MIN_CHARS = 40
    # Reasoning bodies are not replayed at rollover: even a bounded tail
    # duplicates text across messages. The title/continuation marker carries
    # context; only not-yet-delivered reasoning continues on the fresh card.
    # Per-tool result preview length on finished cards. 0 (default) =
    # no API output field — result-derived summary titles remain enabled.
    # Output previews get skimmed past while reasoning is the signal, so
    # they are off by default; set tool_progress_native_output_chars to re-enable.
    OUTPUT_PREVIEW_CHARS = 0
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
        header_label: Optional[str] = None,
        is_subagent: bool = False,
    ) -> None:
        self.client = client
        self.channel = channel
        self.thread_ts = thread_ts
        self.recipient_team_id = recipient_team_id
        self.recipient_user_id = recipient_user_id
        # Slack documents plan/timeline only. Keep the legacy dense setting
        # as a plan alias; there is no separate local dense renderer to preserve.
        self.task_display_mode = "plan" if task_display_mode == "dense" else task_display_mode
        # Identity prefix for the card header ("agent · a3f2c1 · 14:32").
        self.header_label = header_label
        self._is_subagent = is_subagent
        # Config-driven tuning (None → class default). A positive reasoning
        # budget is clamped to SLACK_FIELD_CEILING; 0 keeps uncapped bursts.
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
        self._titles: dict[int | str, str] = {}
        # Interleaved reasoning cards: each burst of thinking between tool
        # calls gets its own 💭 card in the timeline (updated in place while
        # the burst continues, finalized when the next tool starts). The
        # title carries an explicit summary heading or a first-sentence preview;
        # details hold the explanation/overflow without repeating a heading.
        self._reasoning_open_id: Optional[str] = None
        self._reasoning_title: str = ""
        self._reasoning_details: str = ""
        self._reasoning_unsent: str = ""
        self._reasoning_completed = False
        self._reasoning_count = 0
        # Actual appended details, including separator spaces, for this burst.
        # A configured cap is not a per-delta allowance; rollover does not
        # replenish it or re-expose a prefix that already exists above.
        self._reasoning_sent_chars = 0
        # Per-subagent state (card id → {tools, number, start-time}) for the
        # numbered, timed delegate cards.
        self._subagents: dict[str, dict[str, Any]] = {}
        # Rollover state: when the current streamed message ages/fills out,
        # it's closed and a fresh one continues the timeline. Tasks still
        # in_progress at rollover are tracked so they can be replayed onto
        # the new card (their task ids don't exist there otherwise).
        self._stream_opened_at = 0.0
        self._last_append_at: Optional[float] = None
        self._sent_chars = 0
        # Rendered size per field on this stream. Details accumulate; omitted
        # fields persist; title/status/output replace only when present.
        self._chunk_sizes: dict[tuple[str, str], dict[str, int]] = {}
        self._rollovers = 0
        self._in_progress: dict[str, dict[str, Any]] = {}  # task_id → last-sent chunk
        # Serializes the open: tool events arrive back-to-back and each
        # task_started() awaits ensure_started(), so without a lock two
        # coroutines can both pass the ``_started`` check before either
        # completes chat.startStream — opening two streams.
        self._start_lock = asyncio.Lock()
        # Serializes appends:
        # events are scheduled as independent coroutines, so without this a
        # fast tool's "finished" append could overtake its own "started"
        # append mid-HTTP and leave the card stuck showing in_progress.
        # asyncio.Lock wakes waiters FIFO, so scheduling order is preserved.
        self._send_lock = asyncio.Lock()
        from gateway.slack_plan_snapshot import NativePlan
        self._plan = NativePlan(self)

    async def ensure_started(self) -> bool:
        """Open the stream once (idempotent). Returns True if usable."""
        if self.disabled or self._stopped:
            return False
        if self._started:
            return True
        async with self._start_lock:
            return await self._start_locked()

    async def _start_locked(self) -> bool:
        if self.disabled or self._stopped:
            return False
        if self._started:
            return True
        from slack_sdk.errors import SlackClientError
        from aiohttp import ClientError

        if not isinstance(self.recipient_user_id, str) or not re.fullmatch(r"[UW][A-Z0-9]+", self.recipient_user_id):
            self.disabled = True
            return False
        try:
            await self._open_stream()
            self._started = True
            return True
        except (SlackClientError, ClientError, OSError, TimeoutError, RuntimeError) as e:
            logger.warning("chat.startStream failed, disabling native task cards: %s", _safe_card_error(e))
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
        self._last_append_at = None
        self._sent_chars = 0
        self._chunk_sizes = {}

    async def task_started(
        self,
        index: int | str,
        tool_name: str,
        preview: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Emit/update a task_update chunk for a tool call that just started.

        The title carries a friendly verb plus the args preview
        ("Exec — grep foo …") so each line is scannable without expanding.
        ``details`` (optional) fills the
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
        """Set the collapsible header to a phrase summarizing the turn.

        Prefixed with the identity label when one was provided
        ("agent · a3f2c1 · 14:32 — Searched · edited files") so multiple
        cards in one thread — main turn + per-subagent streams — are
        attributable at a glance.
        """
        if not self._categories:
            return
        # Capitalize the first bucket, join the rest with " · ".
        cats = list(self._categories)
        cats[0] = cats[0][:1].upper() + cats[0][1:]
        header = " · ".join(cats)
        if self.header_label:
            header = f"{self.header_label} — {header}"
        # Continuation marker: after a rollover this stream's card is a
        # fresh message continuing an earlier one, so the header must
        # self-identify as such — an unmarked replayed header reads as a
        # duplicate card. The header_label
        # (identity + number, e.g. "SUBAGENT #2 · 01:37") is preserved,
        # so out-of-order continuation messages stay attributable.
        if self._rollovers:
            header = f"⤵ {header}"
        if header != self._last_header:
            self._last_header = header
            await self.set_plan_title(header)

    async def task_finished(
        self,
        index: int | str,
        tool_name: str,
        duration: float = 0.0,
        ok: bool = True,
        output: Optional[str] = None,
        sources: Optional[list[dict]] = None,
        summary: Optional[str] = None,
    ) -> None:
        """Emit/update a task_update chunk for a tool call that just finished."""
        if self.disabled:
            return
        if not await self.ensure_started():
            return
        # Completion-only events can be the first tool boundary on a runtime.
        await self._finalize_reasoning_card()
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
        # OUTPUT_PREVIEW_CHARS <= 0 disables output previews entirely.
        if self.OUTPUT_PREVIEW_CHARS <= 0:
            out = None
        else:
            out = str(output)[: self.OUTPUT_PREVIEW_CHARS + 30] if output else None
        # details APPEND server-side (measured live), so the start-time
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

    async def reasoning_update(self, text: str, *, completed: bool = False) -> None:
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
        confusing mid-sentence fragments.
        Everything overflowing the title lives in details, which carries
        the full accumulated burst (uncapped by default).

        ``completed`` is reserved for the rich publisher's whole redacted burst
        at a semantic/tool/turn boundary, not individual provider callbacks.
        Explicit headings are split from their bodies only in that path, before
        any append-only details escape. Paragraphs remain intact.

        Before the stream opens (thinking that precedes the first tool call
        — i.e. every turn's opening thought), the burst is BUFFERED rather
        than dropped: state is updated but no API call is made, and the
        pending 💭 card is flushed by the first task_started, so the card
        still reads thought ✓ → tool. A turn with zero tool calls never
        opens a stream, so its reasoning is never rendered — intentional.
        """
        if self.disabled or self._stopped:
            return
        line = str(text).strip() if completed else " ".join(str(text).split())
        if not line:
            return
        if self._reasoning_open_id is None:
            self._reasoning_count += 1
            self._reasoning_open_id = f"think{self._reasoning_count}"
            self._reasoning_details = ""
            self._reasoning_unsent = ""
            self._reasoning_title = ""
            self._reasoning_sent_chars = 0
        # ``details`` APPENDS across task_update chunks with the same id —
        # measured live (probe: two updates "AAA"/"BBB" stored
        # as "AAABBB"); title/status REPLACE. So each flush must send ONLY
        # the not-yet-sent delta, and Slack accumulates server-side.
        # Sending the full running text each flush rendered the
        # burst₁+(burst₁+burst₂)+… staircase duplication.
        # _reasoning_details keeps the local copy for title/threshold checks;
        # _reasoning_unsent is the pending tail, including before stream open.
        self._reasoning_details = (self._reasoning_details + " " + line).strip()
        self._reasoning_unsent = (self._reasoning_unsent + " " + line).strip()
        self._reasoning_completed = completed
        cap = self.SLACK_FIELD_CEILING
        if self.REASONING_MAX_CHARS > 0:
            cap = min(self.REASONING_MAX_CHARS, cap)
        if self.REASONING_MAX_CHARS > 0:
            # Slack cannot replace an already-rendered prefix. Keep the head
            # and admit only what still fits after successful details appends.
            self._reasoning_details = self._reasoning_details[:cap]
            remaining = max(0, cap - self._reasoning_sent_chars)
            self._reasoning_unsent = self._reasoning_unsent[:remaining]
        elif len(self._reasoning_details) > cap:
            clipped = self._reasoning_details[-cap:]
            sp = clipped.find(" ")
            self._reasoning_details = "…" + clipped[sp + 1 if 0 <= sp < 40 else 0:]
        if completed:
            # Rich publication supplies whole redacted bursts, never timer/token
            # fragments. Defer title/body separation until the semantic boundary.
            if self._started:
                await self._finalize_reasoning_card()
            return
        # Hold sub-threshold bursts: the flush timer can slice mid-word
        # ("I" as a whole card). Don't open/update the card until the
        # burst has substance; held text flushes with the next update or
        # settles at the next semantic boundary.
        if len(self._reasoning_details) < min(self.REASONING_MIN_CHARS, cap):
            return
        # Title: short TLDR-style header (Claude-app rhythm — headers are
        # sub-sentence). First sentence of the thought, capped ~80 chars,
        # set once and never rewritten; the full text lives in details.
        if not self._reasoning_title:
            self._reasoning_title = f"💭 {_reasoning_preview(self._reasoning_details)}"[:250]
        if not self._started:
            return  # buffered — flushed by the first task_started
        # Flush at sentence boundaries only: send the complete-sentence
        # prefix of the unsent buffer, hold the incomplete tail for the
        # next flush (or finalization). Cutting at raw timer
        # positions split sentences across cards. Join with a TRAILING
        # space — probe #4: Slack preserves trailing whitespace at chunk
        # joins but strips leading whitespace at some element boundaries
        # (the "reminders.Both" jam).
        sendable, tail = _split_complete_sentences(self._reasoning_unsent)
        if self.REASONING_MAX_CHARS > 0 and len(self._reasoning_unsent) == remaining:
            # At the budget boundary no future sentence can fit. A small cap
            # must still render its prefix instead of waiting forever for 40
            # characters or sentence punctuation outside the allowed budget.
            sendable, tail = self._reasoning_unsent, ""
        if not sendable:
            return
        self._reasoning_unsent = tail
        delta = sendable.rstrip() + " "
        if self.REASONING_MAX_CHARS > 0:
            delta = delta[:remaining]
        await self._append_reasoning_details(delta, status="in_progress")

    async def _append_reasoning_details(self, delta: str, *, status: str) -> None:
        """Split only already-sanitized text; the field ceiling is not a burst cap.

        Keep the reasoning identity live until the LAST chunk is accepted, even
        during finalization: rollover must replay identity, never delivered body.
        """
        for offset in range(0, max(1, len(delta)), self.SLACK_FIELD_CEILING):
            part = delta[offset:offset + self.SLACK_FIELD_CEILING]
            last = offset + self.SLACK_FIELD_CEILING >= len(delta)
            await self._append_raw_task(
                self._reasoning_open_id, self._reasoning_title,
                status=status if last else "in_progress", details=part or None,
            )
            if self.disabled:
                return
            self._reasoning_sent_chars += len(part)

    async def _finalize_reasoning_card(self) -> None:
        """Settle the open 💭 card when the next tool starts.

        Sends any still-unsent text, including short completed bursts. The
        threshold holds partial updates, not complete semantic steps.
        """
        if self._reasoning_open_id is None:
            return
        cap = self.SLACK_FIELD_CEILING
        if self.REASONING_MAX_CHARS > 0:
            cap = min(cap, self.REASONING_MAX_CHARS)
        tail, self._reasoning_unsent = self._reasoning_unsent, ""
        rows = _completed_reasoning_rows(tail) if self._reasoning_completed else [("", tail)]
        for index, (label, body) in enumerate(rows):
            if index:
                self._reasoning_count += 1
                self._reasoning_open_id = f"think{self._reasoning_count}"
            if self._reasoning_completed or not self._reasoning_title:
                head = label or _reasoning_preview(body if self._reasoning_completed else self._reasoning_details)
                self._reasoning_title = f"💭 {head}"[:250]
            if self._reasoning_completed and not label:
                body = _reasoning_body_after_preview(body, _reasoning_preview(body))
            delta = (body.rstrip() + " ") if body.strip() else ""
            if self.REASONING_MAX_CHARS > 0:
                delta = delta[:max(0, cap - self._reasoning_sent_chars)]
            await self._append_reasoning_details(delta, status="complete")
        self._reasoning_open_id = None
        self._reasoning_details = ""
        self._reasoning_completed = False

    async def set_plan_title(self, title: str) -> None:
        """Set/update the card's collapsible header via a plan_update chunk.

        Slack's default header ("Thinking completed" / "Something went
        wrong") is generic; a plan_update replaces it with our own text.
        """
        if self.disabled or not self._started:
            return
        from slack_sdk.errors import SlackClientError
        from aiohttp import ClientError

        try:
            async with self._send_lock:
                if self.disabled:
                    return
                await self._send_chunk_locked({"type": "plan_update", "title": str(title)[:250]})
        except (SlackClientError, ClientError, OSError, TimeoutError, RuntimeError) as e:
            logger.info("plan_update failed (non-fatal): %s", _safe_card_error(e))

    async def _append_task_update(
        self,
        index: int | str,
        title: str,
        *,
        status: str,
        details: Optional[str] = None,
        output: Optional[str] = None,
        sources: Optional[list[dict]] = None,
    ) -> None:
        await self._append_raw_task(
            str(index) if isinstance(index, str) else f"t{index}", title, status=status,
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
        sources: Optional[list[dict]] = None,
    ) -> None:
        from slack_sdk.errors import SlackClientError
        from aiohttp import ClientError

        try:
            chunk: dict[str, Any] = {
                "type": "task_update",
                "id": task_id,
                "title": title,
                "status": status,
            }
            if details:
                # Reasoning has already spent its cumulative budget at the
                # reasoning boundary. Provider tool ids can also start with
                # "think"; their bodies must not inherit the reasoning cap.
                chunk["details"] = str(details)[-self.SLACK_FIELD_CEILING:]
            if output:
                chunk["output"] = str(output)[:self.SLACK_FIELD_CEILING]
            if sources:
                chunk["sources"] = sources
            # Track only DELIVERED tasks. Tracking this new chunk before rollover
            # replayed it once there and again below (details APPEND in Slack).
            async with self._send_lock:
                if self.disabled or self._stopped:
                    return
                # Proactive rollover: refresh the stream *before* Slack's
                # ~5-min stream lifetime or per-message size cap kill it,
                # so the user never sees an error state.
                if not self._plan.editing and (
                    time.monotonic() - self._stream_opened_at > self.ROLLOVER_MAX_AGE_S
                    or self._sent_chars > self.ROLLOVER_MAX_CHARS or not self._plan.fits(chunk)
                ):
                    await self._plan.activate()
                try:
                    if not await self._send_chunk_locked(chunk):
                        return
                except (SlackClientError, ClientError, OSError, TimeoutError, RuntimeError) as e:
                    if self._stopped:
                        return
                    # These explicit rejections did not accept the chunk: a
                    # fresh stream can continue it once. Missing-message errors
                    # otherwise abandon native cards for the rest of a live turn.
                    # Unknown transport outcomes must never replay append-only text.
                    response = getattr(e, "response", None)
                    code = response.get("error") if response is not None else None
                    if response is None and type(e) is RuntimeError:
                        # _GuardedCardClient turns an explicit ok:false result
                        # into an exact RuntimeError code, unlike transport errors.
                        code = str(e)
                    if code in ("message_not_in_streaming_state", "msg_too_long", "message_not_found"):
                        now = time.monotonic()
                        logger.info(
                            "task-card stream rejected: code=%s channel=%s thread=%s ts=%s "
                            "team=%s age_s=%.3f last_append_age_s=%s chunk=%s/%s; continuing once",
                            code, self.channel, self.thread_ts, self.ts, self.recipient_team_id,
                            now - self._stream_opened_at,
                            round(now - self._last_append_at, 3) if self._last_append_at is not None else None,
                            chunk.get("type"), chunk.get("id"),
                        )
                        if code == "message_not_in_streaming_state":
                            await self._plan.activate(expired=True)
                        else:
                            await self._rollover_locked()
                        if not await self._send_chunk_locked(chunk):
                            return
                    else:
                        raise
                if status == "in_progress":
                    prior = self._in_progress.get(task_id, {})
                    replay = dict(chunk)
                    if task_id == self._reasoning_open_id:
                        # Continue reasoning by identity/title/status only. Its
                        # body already lives on the previous message; even a
                        # short tail replay duplicates text and spends budget.
                        replay.pop("details", None)
                    else:
                        text = str(prior.get("details") or "") + str(chunk.get("details") or "")
                        if text:
                            replay["details"] = text[-self.SLACK_FIELD_CEILING:]
                    self._in_progress[task_id] = replay
                else:
                    self._in_progress.pop(task_id, None)
        except (SlackClientError, ClientError, OSError, TimeoutError, RuntimeError) as e:
            logger.warning("chat.appendStream failed, disabling native task cards: %s", _safe_card_error(e))
            self.disabled = True

    async def _send_chunk_locked(self, chunk: dict[str, Any]) -> bool:
        """Send while live; return whether accepted. Caller holds _send_lock."""
        # stop() latches outside this lock, even while a previous send awaits.
        # Already-dispatched writes cannot be unsent, but no new ones may start.
        if self._stopped:
            return False
        if self._plan.editing:
            await self._plan.offer(chunk)
            return True
        await self.client.chat_appendStream(
            channel=self.channel,
            ts=self.ts,
            chunks=[chunk],
        )
        self._last_append_at = time.monotonic()
        self._plan.apply(chunk)
        # Approximate rendered characters, not wire traffic: details APPEND
        # even when every delta has the same length. Other fields REPLACE,
        # and an omitted field leaves its prior value intact. Count only
        # accepted writes, so a rejected chunk retried after rollover is once.
        key = (chunk.get("type", "chunk"), chunk.get("id", ""))
        sizes = self._chunk_sizes.setdefault(key, {})
        for field, value in chunk.items():
            size = len(str(value))
            if field == "details" and chunk.get("type") == "task_update":
                self._sent_chars += size
                sizes[field] = sizes.get(field, 0) + size
            else:
                self._sent_chars += size - sizes.get(field, 0)
                sizes[field] = size
        return True

    async def _rollover_locked(self) -> None:
        """Close the current stream and continue on a fresh one.

        Caller holds _send_lock. Raises if the fresh stream can't be opened
        (caller's outer except then disables cards — correct: no stream to
        write to). The old card is stopped best-effort with a continuation
        footer; still-open tasks and the turn header are replayed onto the
        new card so the timeline visually continues.
        """
        from slack_sdk.errors import SlackClientError
        from aiohttp import ClientError

        self._rollovers += 1
        if self._rollovers > self.MAX_ROLLOVERS:
            raise RuntimeError(f"exceeded {self.MAX_ROLLOVERS} stream rollovers this turn")
        try:
            # Settle still-open tasks on the OLD card before closing it —
            # stopping a stream with in_progress tasks makes Slack stamp
            # them with red warning triangles ("something went wrong"),
            # which reads as breakage when it's just a continuation
            # (observed live). Unfinished children must remain
            # pending, including dedicated child streams and fallback entries.
            # The ⤵ suffix hands off to in_progress on the fresh card below;
            # ordinary tool/reasoning segments retain their complete settlement.
            for tid, chunk in list(self._in_progress.items()):
                if self._stopped:
                    return
                # Settlement updates replace only title/status. Resending details
                # here duplicates the whole card body on the old message.
                settled = {k: chunk[k] for k in ("type", "id", "title")}
                settled["status"] = "pending" if self._is_subagent or tid in self._subagents else "complete"
                settled["title"] = f"{str(chunk.get('title', ''))[:240]} ⤵"
                try:
                    await self.client.chat_appendStream(
                        channel=self.channel, ts=self.ts, chunks=[settled],
                    )
                except (SlackClientError, ClientError, OSError, TimeoutError, RuntimeError):
                    break  # old stream already dead — skip the rest
            if self._stopped:
                return
            await self.client.chat_stopStream(
                channel=self.channel,
                ts=self.ts,
                blocks=[{
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": "⤵ continued below"}],
                }],
            )
        except (SlackClientError, ClientError, OSError, TimeoutError, RuntimeError) as e:
            # Old stream may already be dead (that's why we're here).
            logger.info("rollover: closing old stream failed (non-fatal): %s", _safe_card_error(e))
        if self._stopped:
            return
        await self._open_stream()
        self._plan.reset_stream()
        # If stop latched while startStream was in flight, it will close the
        # returned ts after we release _send_lock; do not publish onto it.
        if self._stopped:
            return
        logger.info(
            "rollover: continued task cards on fresh stream ts=%s (rollover #%d)",
            self.ts, self._rollovers,
        )
        # Replay the header and in-flight task state on the new message.
        # Reasoning replay state deliberately excludes already-sent details,
        # including when rollover occurs inside the final completion update.
        replay: list[dict] = []
        if self._last_header:
            # Stamp the replayed header as a continuation (see
            # _refresh_turn_header) — the pre-rollover _last_header has no
            # marker, and headers are only re-sent when a new tool category
            # appears, so without stamping here the continuation card
            # usually keeps an unmarked header and reads as a duplicate.
            _title = self._last_header
            if not _title.startswith("⤵"):
                _title = f"⤵ {_title}"
            self._last_header = _title
            replay.append({"type": "plan_update", "title": _title[:250]})
        for c in self._in_progress.values():
            chunk = dict(c)
            replay.append(chunk)
        for chunk in replay:
            if not await self._send_chunk_locked(chunk):
                return

    async def flush_pending_plan(self) -> None:
        """The owning queue consumer drives coalesced edits; no detached writer."""
        if not self._stopped and not self.disabled:
            async with self._send_lock:
                await self._plan.flush()

    async def handoff_pending(self, task_ids=None) -> None:
        """End this turn's live display without claiming the child finished.

        Slack marks in_progress tasks as errors at stopStream. Its documented
        pending state can instead retain the unknown outcome, with an explicit
        handoff title. Never replay append-only details or open a fresh stream
        during cleanup. Every write still crosses the guarded client's fences.
        """
        if self.disabled or self._stopped or not self._started:
            return
        try:
            async with self._send_lock:
                for tid, chunk in list(self._in_progress.items()):
                    if task_ids is not None and tid not in task_ids:
                        continue
                    await self._send_chunk_locked({
                        "type": "task_update", "id": tid, "status": "pending",
                        "title": f"{str(chunk.get('title', ''))[:190]} · ⤵ result pending (live updates ended)",
                    })
                    self._in_progress.pop(tid, None)
        except Exception:
            logger.debug("child task-card handoff failed", exc_info=True)

    async def stop(self, final_text: Optional[str] = None, *, flush_reasoning: bool = True) -> None:
        """Close the streaming message. No-op if never started or already stopped."""
        if self._stopped or not self._started:
            self._stopped = True
            return
        # Settle any open 💭 card so nothing is left in_progress at close
        # (Slack renders unfinished tasks as warnings + "Something went wrong").
        try:
            if flush_reasoning:
                await self._finalize_reasoning_card()
        except Exception:
            logger.debug("reasoning card cleanup failed", exc_info=True)
        if flush_reasoning and self._plan.editing and not self._stopped:
            async with self._send_lock:
                await self._plan.flush(final=True)
        self._stopped = True
        if self._plan.closed_stream:
            return
        from slack_sdk.errors import SlackClientError
        from aiohttp import ClientError

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
        except (SlackClientError, ClientError, OSError, TimeoutError, RuntimeError) as e:
            logger.info("chat.stopStream failed: %s", _safe_card_error(e))


def _redact_card_value(value: Any) -> Any:
    """Scrub display copies before clipping; keep the canonical outbound policy."""
    try:
        from agent.redact import _JSON_KEY_NAMES, _mask_token, _should_redact_assignment, redact_for_egress, redact_registered_vault_values
        if isinstance(value, str):
            # Preserve complete vault matches before JSON splits their contents.
            value = redact_registered_vault_values(value)
            # Tool results usually arrive serialized. Decode BEFORE text regexes
            # can split escaped secret values; retain a string display copy.
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                # Ordinary prose/markers are not JSON; no lossy pass ran yet.
                return redact_for_egress(value)
            # JSON strings can hide escaped vault values too. Each decoded
            # string is strictly shorter, so recursive unwrapping terminates.
            if isinstance(decoded, (dict, list, str)):
                return json.dumps(_redact_card_value(decoded), ensure_ascii=False)
            return redact_for_egress(value)
        if isinstance(value, list):
            return [_redact_card_value(item) for item in value]
        if isinstance(value, dict):
            # Apply the native named-field policy to unescaped values. Regex
            # substitutions over serialized JSON can leave dangling escaped
            # quotes/backslashes; that text is neither safe nor parseable JSON.
            # Exact vault sanitation must run before lossy head/tail masking:
            # once masked, the registered full value cannot be recognized again.
            return {
                _redact_card_value(key): _mask_token(_redact_card_value(item))
                if (isinstance(key, str) and isinstance(item, str)
                    and re.fullmatch(_JSON_KEY_NAMES, key, re.IGNORECASE)
                    and _should_redact_assignment(key, item, check_keyword=False))
                else _redact_card_value(item)
                for key, item in value.items()
            }
        return value
    except Exception:  # health: allow BLE001 -- Fail closed if redaction fails; logging a traceback could disclose the original card value.
        # A display-only redactor failure must not kill the event consumer or
        # reuse tainted data (including exception messages) on the fallback rail.
        return "[redacted: display unavailable]"


def _redact_card_event(raw: dict) -> dict:
    """Build a safe display copy for BOTH rich publication and text fallback.

    Reasoning deltas are buffered separately until a semantic boundary. Complete
    tool arguments are scrubbed before previews are shortened, so opaque vault
    values cannot lose their recognizable suffix before reaching the redactor.
    """
    from agent.display import build_tool_preview
    raw = dict(raw)
    for field in ("args", "result", "preview", "summary", "goal", "tool_name"):
        if field in raw:
            raw[field] = _redact_card_value(raw[field])
    if raw.get("args") and raw.get("tool_name"):
        if not isinstance(raw["args"], dict):
            # A failed structural pass returns a safe placeholder, not args.
            # The earlier preview may contain a clipped, unrecognizable secret;
            # never reuse it when its authoritative arguments were discarded.
            raw["args"] = {}
            raw["preview"] = "[redacted: display unavailable]"
        else:
            try:
                raw["preview"] = build_tool_preview(raw["tool_name"], raw["args"], max_len=64) or raw.get("preview")
            except Exception:  # health: allow BLE001 -- Preview generation uses redacted arguments; discard them on any failure without logging values.
                # Preview builders can run their own redaction (browser input).
                # Treat that failure just like discarded authoritative args.
                raw["args"] = {}
                raw["preview"] = "[redacted: display unavailable]"
    return raw


class _GuardedCardClient:
    """Use the direct adapter's workspace resolver and outbound guard on every call.

    Relay adapters MUST stay on their own guarded native-card method. They do not
    expose these direct-Slack seams; lack of support is a normal editable fallback,
    never permission to borrow a client or bypass connector egress authorization.
    """

    def __init__(self, adapter, channel, team_id, ctx):
        self.adapter, self.channel, self.team_id = adapter, channel, team_id
        self.ctx = ctx
        self.failure = None
        self.halted = False

    def is_live(self):
        agent = self.ctx.agent_holder[0] if self.ctx.agent_holder else None
        if not self.ctx._run_still_current() or (agent is not None and getattr(agent, "is_interrupted", False)):
            self.halted = True
        return not self.halted

    async def _call(self, method, payload):
        from gateway.platforms.base import SendResult
        from gateway.relay.egress import declined_send
        if self.failure is not None and declined_send(self.failure):
            raise RuntimeError(self.failure.error)
        blocked = self.adapter._outbound_blocked(self.channel, "native task cards")
        if blocked is not None:
            self.failure = blocked
            raise RuntimeError(blocked.error)
        try:
            client = self.adapter._get_client(self.channel, team_id=self.team_id)
            # Routing / task identity stays byte-stable. Only the newly exposed
            # display fields pass through the forced user-facing redactor.
            payload = dict(payload)
            if "chunks" in payload:
                payload["chunks"] = [
                    {key: value if key in {"type", "id", "status"} else _redact_card_value(value)
                     for key, value in chunk.items()}
                    for chunk in payload["chunks"]
                ]
            for field in ("blocks", "markdown_text"):
                if field in payload:
                    payload[field] = _redact_card_value(payload[field])
            if not self.is_live():
                if method != "chat.stopStream":
                    raise RuntimeError("Task-card turn no longer live")
                # stopStream can append content too. A dead turn may only release
                # an existing stream, never send a footer, reasoning or final text.
                payload = {key: payload[key] for key in ("channel", "ts")}
            result = await client.api_call(method, json=payload)
            if result.get("ok") is False:
                raise RuntimeError(result.get("error") or "Slack card request failed")
            # Successful reactive rollover is recovery, unlike authorization refusal.
            self.failure = None
            return result
        except Exception as exc:
            self.failure = SendResult(success=False, error=str(exc), retryable=True)
            raise

    async def chat_startStream(self, **kwargs):
        return await self._call("chat.startStream", kwargs)

    async def chat_appendStream(self, **kwargs):
        return await self._call("chat.appendStream", kwargs)

    async def chat_stopStream(self, **kwargs):
        return await self._call("chat.stopStream", kwargs)

    async def chat_update(self, **kwargs):
        return await self._call("chat.update", kwargs)


class RichTaskCardSession:
    """Rich publisher for TurnRunner's EXISTING native-card queue/fallback rail.

    The stock opt-in still uses the adapter's basic cards. The display opt-in
    chooses this publisher instead, not an additional progress task.
    All lifecycle events are consumed serially, including reasoning boundaries.
    """

    def __init__(self, adapter, ctx):
        from datetime import datetime
        from gateway.display_config import resolve_display_setting
        self.ctx = ctx
        self.main = None
        self.client = None
        self.started_at = {}
        self.children = {}
        self.child_steps = {}
        self.child_completed = set()
        self.reasoning = ""
        self.closed = False
        # A relay adapter's guarded send_native_task_card_progress is its only
        # authorized egress seam; direct transport is deliberately unavailable.
        required = ("_native_task_card_key", "_outbound_blocked", "_get_client")
        if not all(callable(getattr(type(adapter), name, None)) for name in required):
            return
        key = adapter._native_task_card_key(ctx.source.chat_id, ctx._progress_reply_to, ctx._progress_metadata)
        if key is None:
            return
        md = ctx._progress_metadata or {}
        self.client = _GuardedCardClient(adapter, ctx.source.chat_id, key[0], ctx)
        cfg = ctx.user_config or {}
        settings = {name: resolve_display_setting(cfg, "slack", "tool_progress_native_" + suffix)
                    for name, suffix in {
                        "task_display_mode": "mode", "rollover_age_s": "rollover_age_s",
                        "rollover_chars": "rollover_chars", "reasoning_chars": "reasoning_chars",
                        "output_chars": "output_chars",
                    }.items()}
        patterns = (adapter.config.extra or {}).get("mention_patterns") or []
        callsign = str(patterns[0]) if patterns else "agent"
        label = f"{callsign} · {str(ctx.session_id or '')[-6:]} · {datetime.now().strftime('%H:%M')}"
        self.main = SlackTaskStream(
            self.client, ctx.source.chat_id, key[2], header_label=label,
            recipient_team_id=md.get("recipient_team_id") or key[0],
            recipient_user_id=md.get("recipient_user_id") or md.get("user_id") or ctx.source.user_id, **settings,
        )

    async def _subagent_event(self, raw):
        from datetime import datetime
        from tools.delegate_tool_progress import SUBAGENT_FAILURE_STATUSES
        event = raw["type"]
        # task_index repeats across batches. Prefer the current relay's stable child
        # identity/session; only the legacy fallback needs a batch-scoped index.
        key = str(raw.get("subagent_id") or raw.get("child_session_id")
                  or f"{raw.get('delegation_id') or ''}:{raw.get('task_index', 0)}")
        if key in self.child_completed:
            return
        terminal = event == "subagent.complete"
        ok = raw.get("status") not in SUBAGENT_FAILURE_STATUSES
        if terminal:
            # Terminal state belongs to the child, not to the transport chosen
            # below. A failed dedicated stream must not let late events revive it.
            self.child_completed.add(key)
        goal = str(raw.get("goal") or raw.get("preview") or "subagent")
        child = self.children.get(key)
        if child is None:
            # The single queue serializes opens: main then children, numbering in
            # actual Slack thread order rather than the racing task_index order.
            await self.main.ensure_started()
            child = SlackTaskStream(
                self.client, self.main.channel, self.main.thread_ts,
                task_display_mode=self.main.task_display_mode,
                recipient_team_id=self.main.recipient_team_id, recipient_user_id=self.main.recipient_user_id,
                rollover_age_s=self.main.ROLLOVER_MAX_AGE_S, rollover_chars=self.main.ROLLOVER_MAX_CHARS,
                reasoning_chars=self.main.REASONING_MAX_CHARS, output_chars=self.main.OUTPUT_PREVIEW_CHARS,
                header_label=f"🔀 SUBAGENT #{len(self.children) + 1} · {datetime.now().strftime('%H:%M')} · {goal[:60]}",
                is_subagent=True,
            )
            self.children[key] = child
            self.child_steps[key] = 0
            await child.task_started(0, "delegate_task", goal[:120])
        # Dict insertion order is the same serial open order used by child
        # headers. Preserve that number if this child moves to the main stream.
        number = list(self.children).index(key) + 1
        if not child.disabled and event == "subagent.tool" and raw.get("tool_name"):
            self.child_steps[key] += 1
            await child.task_started(
                self.child_steps[key], raw["tool_name"], raw.get("preview"),
                details=tool_details_from_args(raw["tool_name"], raw.get("args") or {}),
            )
        elif not child.disabled and terminal:
            count = self.child_steps[key]
            for index in range(count + 1):
                await child.task_finished(index, "step", ok=ok)
            summary = str(raw.get("summary") or raw.get("preview") or "")
            summary = " ".join(re.sub(r"^\s*#{1,6}\s*", "", summary).split())[:400]
            await child.task_started(count + 1, "delegate_task", "result", details=summary or None)
            await child.task_finished(count + 1, "delegate_task", float(raw.get("duration_seconds") or 0), ok,
                                      summary="✅ completed" if ok else "failed")
        if child.disabled:
            # Also catch a stream that failed DURING the current event, not just
            # one that was already disabled on entry. Main-stream recovery must
            # carry the actual terminal outcome instead of ok=True by default.
            await self.main.subagent_event(event, key, goal, raw.get("tool_name"), ok=ok, number=number)
        if terminal:
            await child.stop()

    async def flush_pending(self):
        if self.closed or self.client is None or not self.client.is_live():
            return
        for stream in [self.main, *self.children.values()]:
            if stream is not None:
                await stream.flush_pending_plan()

    async def _flush_reasoning(self):
        pending, self.reasoning = self.reasoning, ""
        if pending and self.main is not None:
            await self.main.reasoning_update(_redact_card_value(pending), completed=True)

    async def publish(self, events):
        from gateway.platforms.base import SendResult
        from slack_sdk.errors import SlackClientError
        from aiohttp import ClientError

        if self.closed or self.main is None or self.client is None:
            return SendResult(success=False, error="Rich cards require a direct Slack thread target")
        try:
            for raw in events:
                if not self.client.is_live():
                    return self.client.failure or SendResult(success=False, error="Task-card turn no longer live")
                event = raw["type"]
                if event == "reasoning.delta":
                    self.reasoning += raw["text"]
                    # Canonical redaction is whole-text (including multi-line
                    # keys). Timer flushes can publish a prefix before its next
                    # delta identifies it as a secret. Flush at the tool/turn
                    # boundary, before the renderer normalizes or clips text.
                    continue
                if not event.startswith("subagent."):
                    # Child progress can interleave a main-model delta; it is
                    # not a semantic boundary of that reasoning burst.
                    await self._flush_reasoning()
                raw = _redact_card_event(raw)
                if event.startswith("subagent."):
                    await self._subagent_event(raw)
                    continue
                call_id, name = str(raw.get("tool_call_id") or ""), raw.get("tool_name") or "tool"
                args = raw.get("args") or {}
                if event == "tool.started":
                    self.started_at[call_id] = raw.get("timestamp", time.monotonic())
                    await self.main.task_started(call_id, name, raw.get("preview"), details=tool_details_from_args(name, args))
                elif event == "tool.completed":
                    result = raw.get("result")
                    start = self.started_at.pop(call_id, None)
                    duration = max(0, raw.get("timestamp", time.monotonic()) - start) if start is not None else 0
                    await self.main.task_finished(
                        call_id, name, duration, not (raw.get("is_error") or result_looks_like_error(result)),
                        output=clean_output_preview(result, limit=self.main.OUTPUT_PREVIEW_CHARS),
                        sources=tool_sources(name, args, result), summary=summarize_tool_title(name, args, result),
                    )
            if self.client.failure is not None:
                return self.client.failure
            return SendResult(success=not self.main.disabled, message_id=self.main.ts, error="Rich stream disabled" if self.main.disabled else None)
        except (SlackClientError, ClientError, OSError, TimeoutError, RuntimeError) as exc:
            safe_error = _safe_card_error(exc)
            logger.warning("Rich task-card publication failed: %s", safe_error)
            return SendResult(success=False, error=safe_error, retryable=True)

    async def stop(self, *, flush_reasoning=True):
        if self.closed:
            return
        self.closed = True
        flush_reasoning = bool(flush_reasoning and self.client is not None and self.client.is_live())
        if self.client is not None and not flush_reasoning:
            self.client.halted = True
        if self.main is not None:
            if flush_reasoning:
                await self._flush_reasoning()
                # A failed child stream may have moved its progress onto main.
                await self.main.handoff_pending({
                    f"sub_{key}" for key in self.children if key not in self.child_completed
                })
            self.reasoning = ""
            await self.main.stop(flush_reasoning=flush_reasoning)
        # These streams belong to this turn's queue, not the background child.
        # Closing every stream prevents orphan rollovers after /stop or agent reuse;
        # later async-child results use the upstream completion-notification turn.
        for key, child in self.children.items():
            if flush_reasoning and key not in self.child_completed:
                await child.handoff_pending()
            await child.stop(flush_reasoning=flush_reasoning)
        if self.client is not None:
            self.client.halted = True


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
