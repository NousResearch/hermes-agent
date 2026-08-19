"""MemoryManager — orchestrates memory providers for the agent.

Single integration point in run_agent.py. Replaces scattered per-backend
code with one manager that delegates to registered providers.

Only ONE external plugin provider is allowed at a time — attempting to
register a second external provider is rejected with a warning.  This
prevents tool schema bloat and conflicting memory backends.

Usage in run_agent.py:
    self._memory_manager = MemoryManager()
    # Only ONE of these:
    self._memory_manager.add_provider(plugin_provider)

    # System prompt
    prompt_parts.append(self._memory_manager.build_system_prompt())

    # Pre-turn
    context = self._memory_manager.prefetch_all(user_message)

    # Post-turn
    self._memory_manager.sync_all(user_msg, assistant_response)
    self._memory_manager.queue_prefetch_all(user_msg)
"""

from __future__ import annotations

import json
import logging
import re
import inspect
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from agent.skill_commands import extract_user_instruction_from_skill_message
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# How long shutdown_all() waits for in-flight background sync/prefetch work
# to drain before abandoning it. A wedged provider must never block process
# teardown indefinitely — the worker threads are daemon, so anything still
# running past this window dies with the interpreter.
_SYNC_DRAIN_TIMEOUT_S = 5.0
_EXTERNAL_PREFETCH_TIMEOUT_S = 8.0


def normalize_tool_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """Return a function-tool dict with a resolvable top-level ``name``.

    Context engines and memory providers expose tool schemas via
    ``get_tool_schemas()``. The expected shape is a bare function schema
    (``{"name": ..., "description": ..., "parameters": ...}``) which callers
    wrap as ``{"type": "function", "function": schema}``.

    Some providers instead return an entry that is *already* in OpenAI tool
    form (``{"type": "function", "function": {"name": ...}}``). Wrapping that
    a second time produces ``{"type": "function", "function": {"type":
    "function", "function": {...}}}`` whose ``function`` has no top-level
    ``name``. Strict providers (e.g. DeepSeek) reject the *entire* request
    with ``tools[N].function: missing field name`` (HTTP 400), so one bad
    schema disables the whole toolset and breaks every turn (#47707).

    This helper normalizes both shapes to the bare function schema and
    returns ``None`` for anything without a resolvable name, so callers can
    skip-with-warning rather than appending a nameless tool.
    """
    if not isinstance(schema, dict):
        return None
    # Unwrap an already-wrapped OpenAI tool entry.
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
        if not isinstance(schema, dict):
            return None
    name = schema.get("name", "")
    if not name or not isinstance(name, str):
        return None
    return schema


def memory_provider_tools_enabled(
    enabled_toolsets: Optional[List[str]],
    disabled_toolsets: Optional[List[str]] = None,
    *,
    memory_tool_present: bool = False,
) -> bool:
    """Return whether external memory-provider tools should be exposed."""
    if disabled_toolsets and "memory" in disabled_toolsets:
        return False
    if memory_tool_present:
        return True
    if enabled_toolsets is None:
        return True
    if not enabled_toolsets:
        return False
    if "memory" in enabled_toolsets:
        return True

    try:
        from toolsets import resolve_toolset

        return any("memory" in resolve_toolset(name) for name in enabled_toolsets)
    except Exception:
        logger.debug("Failed to resolve enabled toolsets for memory-provider tools", exc_info=True)
        return False


def inject_memory_provider_tools(agent: Any) -> int:
    """Append external memory-provider tool schemas to an agent tool surface."""
    memory_manager = getattr(agent, "_memory_manager", None)
    tools = getattr(agent, "tools", None)
    if not memory_manager or tools is None:
        return 0

    existing_tool_names = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool, dict)
    }
    if not memory_provider_tools_enabled(
        getattr(agent, "enabled_toolsets", None),
        getattr(agent, "disabled_toolsets", None),
        memory_tool_present="memory" in existing_tool_names,
    ):
        return 0

    get_schemas = getattr(memory_manager, "get_all_tool_schemas", None)
    if not callable(get_schemas):
        return 0

    valid_tool_names = getattr(agent, "valid_tool_names", None)
    if valid_tool_names is None:
        valid_tool_names = set()
        agent.valid_tool_names = valid_tool_names

    added = 0
    for raw_schema in get_schemas():
        schema = normalize_tool_schema(raw_schema)
        if schema is None:
            logger.warning(
                "Memory provider returned a tool schema with no resolvable "
                "name; skipping to avoid poisoning the request (%r)",
                raw_schema,
            )
            continue
        tool_name = schema["name"]
        if tool_name in existing_tool_names:
            continue
        tools.append({"type": "function", "function": schema})
        valid_tool_names.add(tool_name)
        existing_tool_names.add(tool_name)
        added += 1

    return added


# ---------------------------------------------------------------------------
# Context fencing helpers
# ---------------------------------------------------------------------------

_CONTEXT_TAG_PADDING_LIMIT = 64
_CONTEXT_TAG_PADDING_RE = rf"\s{{0,{_CONTEXT_TAG_PADDING_LIMIT}}}"
_CONTEXT_OPEN_TAG_RE = re.compile(
    rf'<{_CONTEXT_TAG_PADDING_RE}memory-context{_CONTEXT_TAG_PADDING_RE}>',
    re.IGNORECASE,
)
_CONTEXT_CLOSE_TAG_RE = re.compile(
    rf'</{_CONTEXT_TAG_PADDING_RE}memory-context{_CONTEXT_TAG_PADDING_RE}>',
    re.IGNORECASE,
)
# Only ``<`` followed by a character that can begin the supported tag grammar
# needs Python-level classification.  Let the regex engine skip runs of
# ordinary angle brackets in linear time.
_CONTEXT_TAG_CANDIDATE_RE = re.compile(r"<(?=[/mM\s]|$)")
_INTERNAL_NOTE_RE = re.compile(
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*',
    re.IGNORECASE,
)


def sanitize_context(text: str, *, strict: bool = True) -> str:
    """Strip fence tags, injected context blocks, and system notes.

    ``strict=True`` is the provider-output mode: an ambiguous inline opener
    with no block terminator fails closed at EOF so recalled payload cannot be
    exposed. Transcript/display projections can opt into ``strict=False`` to
    preserve the suffix of an unmatched inline mention such as
    ``"Explain <memory-context> to me"``; complete and block-shaped fences are
    still removed in either mode.
    """
    # Use the streaming parser for complete strings too.  Keeping one grammar
    # prevents a response from being safe in the final path but unsafe while
    # streamed (or vice versa), including mixed-case/whitespace tag variants.
    scrubber = StreamingContextScrubber(strict=strict)
    text = scrubber.feed(text) + scrubber.flush()
    text = _INTERNAL_NOTE_RE.sub('', text)
    return text


def sanitize_context_for_transcript(text: str) -> str:
    """Remove recalled-context fences while preserving ordinary inline prose."""
    return sanitize_context(text, strict=False)


class StreamingContextScrubber:
    """Stateful scrubber for streaming text that may contain split memory-context spans.

    The one-shot ``sanitize_context`` regex cannot survive chunk boundaries:
    a ``<memory-context>`` opened in one delta and closed in a later delta
    leaks its payload to the UI because the non-greedy block regex needs
    both tags in one string.  This scrubber runs a small state machine
    across deltas, holding back partial-tag tails and discarding
    everything inside a span (including the system-note line).

    Usage::

        scrubber = StreamingContextScrubber()
        for delta in stream:
            visible = scrubber.feed(delta)
            if visible:
                emit(visible)
        trailing = scrubber.flush()  # at end of stream
        if trailing:
            emit(trailing)

    The scrubber is re-entrant per agent instance.  Callers building new
    top-level responses (new turn) should create a fresh scrubber or call
    ``reset()``.

    Tag padding is accepted up to ``_MAX_TAG_PADDING`` whitespace characters
    on each side of the name. Longer tag-like prefixes are retained only while
    the bounded parser can still distinguish them from ordinary prose; a
    confirmed over-budget memory-context opener fails closed.

    Inline openers are held only up to ``_MAX_AMBIGUOUS_INLINE_LEN``. At EOF,
    strict provider-output mode discards every unmatched opener. Lenient
    transcript/user mode uses a small anchored grammar to restore ordinary
    prose that explicitly documents the tag. This keeps chunk boundaries from
    changing the decision and prevents arbitrary provider payload from
    bypassing the fence.
    """

    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"
    _MAX_TAG_PADDING = _CONTEXT_TAG_PADDING_LIMIT
    _MAX_TAG_LEN = len(_CLOSE_TAG) + (2 * _MAX_TAG_PADDING)
    _MAX_AMBIGUOUS_INLINE_LEN = 512
    # A standalone closer needs a little more lookahead than an individual
    # tag candidate: the following opener can begin just beyond the candidate
    # budget. Keep that decision finite, and fail closed if it is exhausted.
    _MAX_POST_CLOSE_LEN = 2 * _MAX_AMBIGUOUS_INLINE_LEN

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = bool(strict)
        self._in_span: bool = False
        self._span_depth: int = 0
        self._buf: str = ""
        self._ambiguous_inline: bool = False
        self._after_standalone_close: bool = False
        self._post_close_prefix: str = ""
        self._post_close_buf: str = ""
        self._discard_post_close: bool = False
        self._at_block_boundary: bool = True
        self._pending_at_block_boundary: bool = True
        # ``_feed_once`` can discover another already-buffered suffix that
        # needs to be parsed in a different state.  Keep that transition on an
        # explicit queue rather than recursively re-entering ``feed``: a long
        # response containing many inline fences must not consume one Python
        # stack frame per fence.
        self._feed_queue: deque[str] = deque()
        self._feed_active: bool = False

    def reset(self) -> None:
        self._in_span = False
        self._span_depth = 0
        self._buf = ""
        self._ambiguous_inline = False
        self._after_standalone_close = False
        self._post_close_prefix = ""
        self._post_close_buf = ""
        self._discard_post_close = False
        self._at_block_boundary = True
        self._pending_at_block_boundary = True
        self._feed_queue.clear()
        self._feed_active = False

    def feed(self, text: str) -> str:
        """Return visible text while iteratively draining state transitions."""
        if not text:
            return ""

        if self._feed_active:
            # Internal state transitions historically called ``feed`` again.
            # Queue the continuation at the front so it is consumed before
            # any later input, while the outermost call owns visible output.
            self._feed_queue.appendleft(text)
            return ""

        self._feed_active = True
        self._feed_queue.append(text)
        out: list[str] = []
        try:
            while self._feed_queue:
                out.append(self._feed_once(self._feed_queue.popleft()))
        finally:
            self._feed_active = False
            self._feed_queue.clear()
        return "".join(out)

    def _feed_once(self, text: str) -> str:
        """Return the visible portion of ``text`` after scrubbing.

        Any trailing fragment that could be the start of an open/close tag
        is held back in the internal buffer and surfaced on the next
        ``feed()`` call or discarded/emitted by ``flush()``.
        """
        if not text:
            return ""

        if self._after_standalone_close:
            return self._feed_after_standalone_close(text)
        if self._discard_post_close:
            return ""
        if self._ambiguous_inline:
            return self._feed_ambiguous(text)

        had_pending = bool(self._buf)
        pending_at_block_boundary = self._pending_at_block_boundary
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []
        cursor = 0

        while cursor < len(buf):
            if self._in_span:
                # Treat supported openers inside a fenced span as nested.
                # Provider-controlled memory can contain delimiter-like text;
                # accepting the first closer would expose everything between
                # an inner close and the matching outer close.
                tag_match = _CONTEXT_TAG_CANDIDATE_RE.search(buf, cursor)
                if tag_match is None:
                    self._buf = self._partial_span_tag_suffix(buf[cursor:])
                    return "".join(out)
                cursor = tag_match.start()
                closing = cursor + 1 < len(buf) and buf[cursor + 1] == "/"
                status, tag_end = self._classify_tag_prefix(
                    buf, closing=closing, start=cursor
                )
                if status == "partial":
                    self._buf = buf[cursor:]
                    return "".join(out)
                if status == "over_budget":
                    # A complete tag-like opener with unsupported padding is
                    # still untrusted provider context.  Count it as nested so
                    # its closer cannot terminate the enclosing fence and
                    # expose the rest of the outer payload.  Unsupported
                    # closers remain inert: accepting one would reduce the
                    # depth based on a delimiter outside the supported grammar.
                    if not closing:
                        self._span_depth += 1
                    cursor += max(tag_end, 1)
                    continue
                if status == "invalid":
                    cursor += max(tag_end, 1)
                    continue
                cursor += tag_end
                if closing:
                    self._span_depth -= 1
                    if self._span_depth == 0:
                        self._in_span = False
                        # A pending fragment consumed while inside the span
                        # describes the closer, not the next candidate.  Do
                        # not let its saved block-boundary state misclassify
                        # an immediately following lenient inline reference.
                        had_pending = False
                else:
                    self._span_depth += 1
                continue

            tag_match = _CONTEXT_TAG_CANDIDATE_RE.search(buf, cursor)
            if tag_match is None:
                self._append_visible(out, buf[cursor:])
                return "".join(out)
            tag_idx = tag_match.start()

            if tag_idx > cursor:
                self._append_visible(out, buf[cursor:tag_idx])
                cursor = tag_idx
                had_pending = False

            candidate_at_block_boundary = (
                pending_at_block_boundary if had_pending else self._at_block_boundary
            )
            had_pending = False

            closing = cursor + 1 < len(buf) and buf[cursor + 1] == "/"
            status, tag_end = self._classify_tag_prefix(
                buf, closing=closing, start=cursor
            )

            if status == "invalid":
                self._append_visible(out, buf[cursor])
                cursor += 1
                continue

            if status == "partial":
                candidate = buf[cursor:]
                if len(candidate) > self._MAX_AMBIGUOUS_INLINE_LEN:
                    # A prefix that remains tag-like after the absolute buffer
                    # budget is untrusted. This caps retained memory and avoids
                    # quadratic copies under character-wise streaming.
                    if closing:
                        self._append_visible(out, candidate)
                    else:
                        self._enter_span()
                    return "".join(out)
                self._buf = candidate
                self._pending_at_block_boundary = candidate_at_block_boundary
                return "".join(out)

            if status == "over_budget":
                if closing:
                    # Unsupported standalone closers cannot disclose content
                    # and remain ordinary text in both one-shot and streaming
                    # paths.
                    self._append_visible(out, buf[cursor : cursor + tag_end])
                    cursor += tag_end
                    continue
                # Once the complete name confirms that an over-budget prefix
                # is a memory fence, preserve only preceding visible text.
                self._enter_span()
                cursor += tag_end
                continue

            if closing:
                # A standalone closer may be an attempt by recalled provider
                # data to escape a fence before reopening it. Hold the suffix
                # until EOF or a following opener makes that distinction
                # unambiguous; the bounded strict path fails closed rather
                # than releasing attacker-controlled padding.
                self._after_standalone_close = True
                return "".join(out) + self._feed_after_standalone_close(
                    buf[cursor + tag_end :]
                )

            after_start = cursor + tag_end
            line_start = after_start
            while line_start < len(buf) and buf[line_start] in " \t":
                line_start += 1
            if candidate_at_block_boundary or (
                line_start < len(buf) and buf[line_start] in "\r\n"
            ):
                self._enter_span()
                cursor = after_start
                continue

            # When the decisive delimiter is already in this input buffer,
            # stay in the current cursor loop.  Re-queuing ``buf[after_start:]``
            # would copy the entire remaining suffix once per inline fence,
            # making a long response with many complete fence pairs quadratic.
            if _CONTEXT_OPEN_TAG_RE.search(
                buf, after_start
            ) is not None or _CONTEXT_CLOSE_TAG_RE.search(buf, after_start) is not None:
                self._enter_span()
                cursor = after_start
                continue

            if len(buf) - cursor > self._MAX_AMBIGUOUS_INLINE_LEN:
                # The closer may arrive in a later provider chunk. Releasing
                # an over-budget suffix here would therefore make lenient
                # sanitization chunk-dependent and could expose a complete
                # recalled-context payload. Supported documentation
                # references are intentionally bounded and resolve at EOF.
                self._enter_span()
                cursor = after_start
                continue

            # A complete inline opener is ambiguous: it may begin a leaked
            # span or be a short documentation reference. Hold the bounded
            # candidate until a closer, a line break, the budget, or EOF makes
            # the decision independent of provider chunking.
            self._buf = buf[cursor:]
            self._ambiguous_inline = True
            self._pending_at_block_boundary = candidate_at_block_boundary
            return "".join(out)

        return "".join(out)

    def _feed_after_standalone_close(self, text: str) -> str:
        """Hold a standalone-close suffix until a later opener or EOF.

        ``</memory-context>INJECTED<memory-context>...`` is an escape attempt:
        both the injected middle and reopened suffix must be discarded. A
        closer with no later opener retains the historical behavior of
        stripping only the delimiter when the stream is flushed.
        """
        candidate = self._post_close_prefix + self._post_close_buf + text
        self._post_close_prefix = ""
        self._post_close_buf = ""
        cursor = 0

        while cursor < len(candidate):
            tag_match = _CONTEXT_TAG_CANDIDATE_RE.search(candidate, cursor)
            if tag_match is None:
                break
            tag_idx = tag_match.start()
            if tag_idx > self._MAX_POST_CLOSE_LEN:
                # A visible projection cannot release a long suffix and later
                # retract it if an opener arrives. Once the finite
                # lookahead is exhausted, both modes discard the remainder of
                # this response. Lenient mode cannot release an earlier prefix
                # either: a later opener would make it attacker-controlled
                # close/reopen content. This is deliberately response-scoped
                # state, not retained attacker text.
                self._after_standalone_close = False
                self._discard_post_close = True
                return ""
            closing = tag_idx + 1 < len(candidate) and candidate[tag_idx + 1] == "/"
            status, tag_end = self._classify_tag_prefix(
                candidate, closing=closing, start=tag_idx
            )
            if status == "partial":
                if len(candidate) > self._MAX_POST_CLOSE_LEN:
                    # The complete retained suffix is no longer eligible for
                    # the bounded close/open documentation grammar.  Discard
                    # its decided prefix, but keep parsing the unresolved tag
                    # candidate: releasing it as prose (lenient) or blindly
                    # discarding all later chunks (strict) would make the
                    # result depend on where the provider split the tag.
                    self._after_standalone_close = False
                    self._discard_post_close = True
                    return ""
                self._store_post_close_candidate(candidate)
                return ""
            if not closing and status in {"complete", "over_budget"}:
                if (
                    not self._strict
                    and not self._discard_post_close
                    and status == "complete"
                    and len(candidate) <= self._MAX_AMBIGUOUS_INLINE_LEN
                ):
                    # A transcript can legitimately document both delimiter
                    # spellings in one sentence. Keep the bounded suffix until
                    # EOF so the explicit-reference grammar can distinguish
                    # that prose from a close/reopen escape attempt without
                    # making the answer depend on provider chunking.
                    cursor = tag_idx + tag_end
                    continue
                self._after_standalone_close = False
                self._discard_post_close = False
                self._enter_span()
                return self.feed(candidate[tag_idx + tag_end :])
            cursor = tag_idx + max(tag_end, 1)

        if self._discard_post_close:
            # The ambiguity budget was crossed while a later tag was still
            # incomplete. If its continuation proves invalid, the discarded
            # prefix cannot be reconstructed safely; fail closed for the
            # remainder of this response without retaining attacker input.
            self._after_standalone_close = False
            self._post_close_prefix = ""
            self._post_close_buf = ""
            return ""

        if len(candidate) > self._MAX_POST_CLOSE_LEN:
            self._after_standalone_close = False
            self._discard_post_close = True
            return ""

        self._store_post_close_candidate(candidate)
        return ""

    def _store_post_close_candidate(self, candidate: str) -> None:
        """Retain bounded close-suffix lookahead with a bounded tag tail."""
        split_at = max(0, len(candidate) - self._MAX_AMBIGUOUS_INLINE_LEN)
        self._post_close_prefix = candidate[:split_at]
        self._post_close_buf = candidate[split_at:]

    def _feed_ambiguous(self, text: str) -> str:
        """Advance a bounded inline-opener candidate."""
        candidate = self._buf + text
        self._buf = ""

        open_status, open_end = self._classify_tag_prefix(candidate, closing=False)
        if open_status != "complete":
            # This is defensive: ambiguous mode is entered only after a
            # complete supported opener.
            self._ambiguous_inline = False
            self._enter_span()
            return ""

        if _CONTEXT_OPEN_TAG_RE.search(candidate, open_end) is not None or (
            _CONTEXT_CLOSE_TAG_RE.search(candidate, open_end) is not None
        ):
            self._ambiguous_inline = False
            self._enter_span()
            return self.feed(candidate[open_end:])

        after = candidate[open_end:]
        if after.lstrip(" \t").startswith(("\r", "\n")):
            self._ambiguous_inline = False
            self._enter_span()
            return self.feed(after)

        if len(candidate) > self._MAX_AMBIGUOUS_INLINE_LEN:
            self._ambiguous_inline = False
            self._enter_span()
            # Parse the capped candidate tail once in span mode before
            # discarding it.  It may already contain a complete nested opener
            # (including an over-padding-budget opener) followed by only a
            # *partial* closer.  Keeping just that partial suffix would forget
            # the nested depth, so the completed closer in the next chunk
            # could terminate the outer fence and expose its remaining
            # payload.  The in-span parser retains only a bounded tag suffix.
            return self.feed(after)

        self._buf = candidate
        return ""

    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream.

        If we're still inside an unterminated span the remaining content is
        discarded (safer: leaking partial memory context is worse than a
        truncated answer).  A held partial-tag tail is emitted verbatim, but a
        complete ambiguous opener is restored only for explicit inline
        tag-reference prose in lenient transcript/user mode.
        """
        if self._in_span:
            self._buf = ""
            self._in_span = False
            self._span_depth = 0
            self._ambiguous_inline = False
            self._after_standalone_close = False
            self._post_close_prefix = ""
            self._post_close_buf = ""
            self._discard_post_close = False
            return ""
        if self._discard_post_close:
            self._discard_post_close = False
            self._after_standalone_close = False
            self._post_close_prefix = ""
            self._post_close_buf = ""
            return ""
        if self._after_standalone_close:
            tail = self._post_close_prefix + self._post_close_buf
            self._after_standalone_close = False
            self._post_close_prefix = ""
            self._post_close_buf = ""
            if not self._strict:
                # A close/reopen sequence is indistinguishable from an escape
                # when its intervening prose is attacker controlled.  Do not
                # guess from sentence shape: re-run every candidate through
                # the strict standalone-close path. Ordinary standalone closer
                # prose (with no later opener) remains visible there.
                strict_scrubber = type(self)(strict=True)
                tail = (
                    strict_scrubber.feed(self._CLOSE_TAG + tail)
                    + strict_scrubber.flush()
                )
            if tail:
                self._update_block_boundary(tail)
            return tail
        tail = self._buf
        self._buf = ""
        ambiguous_inline = self._ambiguous_inline
        self._ambiguous_inline = False
        if ambiguous_inline:
            # Provider output cannot prove that prose following an unmatched
            # opener is documentation rather than recalled private payload.
            # Strict mode therefore never restores it. Transcript/user mode
            # retains the bounded documentation grammar for compatibility.
            if self._strict:
                return ""
            if not self._is_explicit_inline_tag_reference(tail):
                open_status, open_end = self._classify_tag_prefix(
                    tail, closing=False
                )
                if open_status != "complete":
                    return tail
                tail = tail[open_end:]
        if tail:
            self._update_block_boundary(tail)
        return tail

    @classmethod
    def _classify_tag_prefix(
        cls, candidate: str, *, closing: bool, start: int = 0
    ) -> tuple[str, int]:
        """Classify a candidate beginning with ``<``.

        The result is ``complete``, ``partial``, ``over_budget``, or
        ``invalid`` plus the end offset for complete forms. A partial may
        contain more than the supported padding while the name is still
        unknown; the absolute candidate budget limits that ambiguity. Only
        the finite grammar prefix is inspected, even when ``candidate`` also
        contains a large ordinary-text suffix.
        """
        if start >= len(candidate) or candidate[start] != "<":
            return "invalid", 0
        pos = start + 1
        if closing:
            if pos >= len(candidate):
                return "partial", 0
            if candidate[pos] != "/":
                return "invalid", 0
            pos += 1
        elif pos < len(candidate) and candidate[pos] == "/":
            return "invalid", 0

        target = "memory-context"
        padding_start = pos
        while pos < len(candidate) and candidate[pos].isspace():
            pos += 1
            if pos - start > cls._MAX_AMBIGUOUS_INLINE_LEN:
                return "over_budget", pos - start
        leading_padding = pos - padding_start
        if pos == len(candidate):
            return "partial", 0

        name_len = min(len(candidate) - pos, len(target))
        if candidate[pos : pos + name_len].lower() != target[:name_len]:
            return "invalid", 0
        if name_len < len(target):
            return "partial", 0
        pos += len(target)
        if pos - start > cls._MAX_AMBIGUOUS_INLINE_LEN:
            return "over_budget", pos - start

        padding_start = pos
        while pos < len(candidate) and candidate[pos].isspace():
            pos += 1
            if pos - start > cls._MAX_AMBIGUOUS_INLINE_LEN:
                return "over_budget", pos - start
        trailing_padding = pos - padding_start
        if pos >= len(candidate):
            return "partial", 0
        if candidate[pos] != ">":
            return "invalid", 0

        status = (
            "over_budget"
            if leading_padding > cls._MAX_TAG_PADDING
            or trailing_padding > cls._MAX_TAG_PADDING
            else "complete"
        )
        return status, pos + 1 - start

    @classmethod
    def _partial_tag_suffix(cls, buf: str, *, closing: bool) -> str:
        """Return the bounded suffix that can still become a supported tag."""
        idx = buf.rfind("<")
        if idx < 0:
            return ""
        candidate = buf[idx:]
        status, _ = cls._classify_tag_prefix(candidate, closing=closing)
        if status != "partial" or len(candidate) > cls._MAX_TAG_LEN:
            return ""
        return candidate

    @classmethod
    def _partial_span_tag_suffix(cls, buf: str) -> str:
        """Return a supported opener/closer fragment at a span boundary."""
        idx = buf.rfind("<")
        if idx < 0:
            return ""
        candidate = buf[idx:]
        closing = len(candidate) > 1 and candidate[1] == "/"
        status, _ = cls._classify_tag_prefix(candidate, closing=closing)
        if status != "partial" or len(candidate) > cls._MAX_TAG_LEN:
            return ""
        return candidate

    def _enter_span(self) -> None:
        """Enter a new outer memory-context span."""
        self._in_span = True
        self._span_depth = 1

    def _append_visible(self, out: list[str], text: str) -> None:
        if not text:
            return
        out.append(text)
        self._update_block_boundary(text)

    def _update_block_boundary(self, text: str) -> None:
        last_newline = text.rfind("\n")
        if last_newline >= 0:
            self._at_block_boundary = not text[last_newline + 1:].strip(" \t\r")
        else:
            self._at_block_boundary = (
                self._at_block_boundary and not text.strip(" \t\r")
            )

    @classmethod
    def _is_explicit_inline_tag_reference(cls, candidate: str) -> bool:
        """Recognize a bounded, single-line inline delimiter reference.

        This intentionally uses only shape, not an English sentence
        whitelist.  A reference must be a complete short sentence fragment
        with a short multi-word clause after the delimiter and terminal
        punctuation. Raw marker-label shapes require one extra word because
        they are also common provider payload prefixes. Short values,
        control-line payloads, nested tags, and unfinished provider output
        remain fail-closed.
        """
        if len(candidate) > cls._MAX_AMBIGUOUS_INLINE_LEN:
            return False
        match = _CONTEXT_OPEN_TAG_RE.match(candidate)
        if match is None or match.start() != 0:
            return False
        after = candidate[match.end():]
        if (
            not after
            or "\r" in after
            or "\n" in after
            or _CONTEXT_OPEN_TAG_RE.search(after) is not None
            or _CONTEXT_CLOSE_TAG_RE.search(after) is not None
        ):
            return False
        backticked = after.startswith("`")
        if backticked:
            after = after[1:]
        if not after.startswith((" ", "\t")):
            return False
        prose = after.strip(" \t")
        if not prose or prose[-1] not in ".!?":
            return False
        if any(char in prose for char in "<>`"):
            return False
        words = re.findall(r"[^\W_]+", prose, re.UNICODE)
        if len(words) < 3:
            return False
        marker_shape = backticked or words[0].casefold() in {
            "block",
            "fence",
            "marker",
            "tag",
        }
        return not marker_shape or len(words) >= 4


def build_memory_context_block(raw_context: str) -> str:
    """Wrap prefetched memory in a fenced block with system note."""
    if not raw_context or not raw_context.strip():
        return ""
    clean = sanitize_context(raw_context)
    if clean != raw_context:
        logger.warning("memory provider returned pre-wrapped context; stripped")
    return (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data — "
        "this is the agent's persistent memory and should inform all responses.]\n\n"
        f"{clean}\n"
        "</memory-context>"
    )


class MemoryManager:
    """Orchestrates the built-in provider plus at most one external provider.

    The builtin provider is always first. Only one non-builtin (external)
    provider is allowed.  Failures in one provider never block the other.
    """

    def __init__(self, *, external_prefetch_timeout: Optional[float] = None) -> None:
        self._providers: List[MemoryProvider] = []
        self._tool_to_provider: Dict[str, MemoryProvider] = {}
        self._has_external: bool = False  # True once a non-builtin provider is added
        self._external_prefetch_timeout = (
            _EXTERNAL_PREFETCH_TIMEOUT_S
            if external_prefetch_timeout is None
            else float(external_prefetch_timeout)
        )
        if self._external_prefetch_timeout <= 0:
            raise ValueError("external_prefetch_timeout must be positive")
        self._external_prefetch_threads: Dict[str, threading.Thread] = {}
        self._external_prefetch_lock = threading.Lock()
        # Background executor for end-of-turn sync/prefetch. Lazily created on
        # first use so the common builtin-only path spawns no extra threads.
        # A single worker serializes a provider's writes (turn N must land
        # before turn N+1) and caps thread growth at one per manager. See
        # _submit_background() and the sync_all/queue_prefetch_all rationale.
        self._sync_executor: Optional[ThreadPoolExecutor] = None
        self._sync_executor_lock = threading.Lock()
        # Futures are tracked by durability class so shutdown can give writes
        # a bounded FIFO drain, then explicitly report anything abandoned.
        self._background_futures: Dict[Future, str] = {}
        self._shutting_down = False
        self._shutdown_drain_state: Dict[str, Any] = {
            "status": "not_started",
            "abandoned_writes": 0,
            "abandoned_prefetches": 0,
            "active_tasks": 0,
        }

    # -- Registration --------------------------------------------------------

    def add_provider(self, provider: MemoryProvider) -> None:
        """Register a memory provider.

        Built-in provider (name ``"builtin"``) is always accepted.
        Only **one** external (non-builtin) provider is allowed — a second
        attempt is rejected with a warning.
        """
        is_builtin = provider.name == "builtin"

        if not is_builtin:
            if self._has_external:
                existing = next(
                    (p.name for p in self._providers if p.name != "builtin"), "unknown"
                )
                logger.warning(
                    "Rejected memory provider '%s' — external provider '%s' is "
                    "already registered. Only one external memory provider is "
                    "allowed at a time. Configure which one via memory.provider "
                    "in config.yaml.",
                    provider.name, existing,
                )
                return
            self._has_external = True

        self._providers.append(provider)

        # Core tool names are reserved — a memory provider must never register
        # a tool that shadows a built-in (e.g. ``clarify``, ``delegate_task``).
        # Built-ins always win, so such a tool is dropped at agent init and
        # would otherwise linger in ``_tool_to_provider`` and hijack dispatch
        # (#40466). Reject it here, at the door, so it never enters the routing
        # table at all — matching the built-ins-always-win invariant used by
        # the TTS/browser/search provider registries.
        from toolsets import _HERMES_CORE_TOOLS

        _core_tool_names = set(_HERMES_CORE_TOOLS)

        # Index tool names → provider for routing
        for raw_schema in provider.get_tool_schemas():
            schema = normalize_tool_schema(raw_schema)
            if schema is None:
                continue
            tool_name = schema["name"]
            if tool_name in _core_tool_names:
                logger.warning(
                    "Memory provider '%s' tool '%s' shadows a reserved core "
                    "tool name; registration ignored. Core tools always win — "
                    "rename the provider's tool to something unique.",
                    provider.name, tool_name,
                )
                continue
            if tool_name and tool_name not in self._tool_to_provider:
                self._tool_to_provider[tool_name] = provider
            elif tool_name in self._tool_to_provider:
                logger.warning(
                    "Memory tool name conflict: '%s' already registered by %s, "
                    "ignoring from %s",
                    tool_name,
                    self._tool_to_provider[tool_name].name,
                    provider.name,
                )

        logger.info(
            "Memory provider '%s' registered (%d tools)",
            provider.name,
            len(provider.get_tool_schemas()),
        )

    @property
    def providers(self) -> List[MemoryProvider]:
        """All registered providers in order."""
        return list(self._providers)

    def get_provider(self, name: str) -> Optional[MemoryProvider]:
        """Get a provider by name, or None if not registered."""
        for p in self._providers:
            if p.name == name:
                return p
        return None

    # -- System prompt -------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Collect system prompt blocks from all providers.

        Returns combined text, or empty string if no providers contribute.
        Each non-empty block is labeled with the provider name.
        """
        blocks = []
        for provider in self._providers:
            try:
                block = provider.system_prompt_block()
                if block and block.strip():
                    blocks.append(block)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' system_prompt_block() failed: %s",
                    provider.name, e,
                )
        return "\n\n".join(blocks)

    # -- Prefetch / recall ---------------------------------------------------

    @staticmethod
    def _strip_skill_scaffolding(text: str) -> Optional[str]:
        """Return memory-worthy user text, or None to skip the turn.

        When a user invokes a /skill or /bundle, Hermes expands the turn into
        a model-facing message that embeds the entire skill body. Feeding that
        verbatim to memory providers pollutes their stores/embeddings with
        prompt scaffolding instead of what the user actually asked. We recover
        just the user's instruction here, once, for every provider — so this
        is fixed for the whole provider fan-out, not per backend.

        - Non-skill messages pass through unchanged.
        - Skill turns with a user instruction return that instruction.
        - Bare skill invocations (no instruction) return None → callers skip
          the turn, since there is no user content worth remembering.
        """
        return extract_user_instruction_from_skill_message(text)

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Collect prefetch context from all providers.

        Returns merged context text labeled by provider. Empty providers
        are skipped. Failures in one provider don't block others.
        """
        clean_query = self._strip_skill_scaffolding(query)
        if not clean_query:
            return ""
        parts = []
        for provider in self._providers:
            try:
                result = self._prefetch_provider(provider, clean_query, session_id=session_id)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' prefetch failed (non-fatal): %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    def _prefetch_provider(
        self, provider: MemoryProvider, query: str, *, session_id: str = ""
    ) -> str:
        if provider.name == "builtin":
            return provider.prefetch(query, session_id=session_id)

        result_box: Dict[str, str] = {}
        error_box: Dict[str, Exception] = {}

        def _run() -> None:
            try:
                result_box["value"] = provider.prefetch(query, session_id=session_id) or ""
            except Exception as exc:  # pragma: no cover - re-raised by caller
                error_box["value"] = exc

        # Propagate the caller's contextvars (profile HERMES_HOME override)
        # to the prefetch thread — see _submit_background.
        import contextvars
        from functools import partial

        thread = threading.Thread(
            target=partial(contextvars.copy_context().run, _run),
            daemon=True,
            name=f"memory-prefetch-{provider.name}",
        )
        with self._external_prefetch_lock:
            existing = self._external_prefetch_threads.get(provider.name)
            if existing is not None:
                if existing.is_alive():
                    logger.debug(
                        "Memory provider '%s' prefetch is still running; skipping this turn",
                        provider.name,
                    )
                    return ""
                self._external_prefetch_threads.pop(provider.name, None)
            self._external_prefetch_threads[provider.name] = thread
            thread.start()

        thread.join(self._external_prefetch_timeout)
        if thread.is_alive():
            logger.warning(
                "Memory provider '%s' prefetch timed out after %.1fs; skipping it until "
                "the stuck call returns",
                provider.name,
                self._external_prefetch_timeout,
            )
            return ""

        with self._external_prefetch_lock:
            if self._external_prefetch_threads.get(provider.name) is thread:
                self._external_prefetch_threads.pop(provider.name, None)
        if error_box:
            raise error_box["value"]
        return result_box.get("value", "")

    def describe_recall(self) -> str:
        """Build a deterministic, model-independent recall indicator line.

        Call right after :meth:`prefetch_all` on the turn thread. Collects each
        provider's :meth:`MemoryProvider.recall_status` and renders a single
        status string (e.g. ``"🧠 Provider — recalled 3 memories"``) so the
        user SEES memory was used regardless of whether the model mentions it.
        Returns ``""`` when no provider injected memory this turn — callers can
        emit the result unconditionally.
        """
        segments: List[str] = []
        for provider in self._providers:
            try:
                status = provider.recall_status()
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' recall_status failed (non-fatal): %s",
                    provider.name, e,
                )
                continue
            if status is None:
                continue
            if status.count == 1:
                detail = "recalled 1 memory"
            elif status.count > 1:
                detail = f"recalled {status.count} memories"
            else:
                # count <= 0 → content injected but no discrete count (reflect).
                detail = "recalled relevant memory"
            segments.append(f"{status.glyph} {status.provider_label} — {detail}")
        return "  ".join(segments)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch on all providers for the next turn.

        Provider work is dispatched to a background worker so a slow or
        wedged provider can never block the caller. See ``sync_all`` for
        the full rationale (agent stuck "running" minutes after a turn).
        """
        providers = list(self._providers)
        if not providers:
            return

        clean_query = self._strip_skill_scaffolding(query)
        if not clean_query:
            return

        def _run() -> None:
            for provider in providers:
                try:
                    provider.queue_prefetch(clean_query, session_id=session_id)
                except Exception as e:
                    logger.debug(
                        "Memory provider '%s' queue_prefetch failed (non-fatal): %s",
                        provider.name, e,
                    )

        self._submit_background(_run, kind="prefetch")

    # -- Sync ----------------------------------------------------------------

    @staticmethod
    def _provider_sync_accepts_messages(provider: MemoryProvider) -> bool:
        """Return whether sync_turn accepts a messages keyword."""
        try:
            signature = inspect.signature(provider.sync_turn)
        except (TypeError, ValueError):
            return True
        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return True
        return "messages" in signature.parameters

    def sync_all(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Sync a completed turn to all providers.

        Runs on a background worker thread, NOT inline on the
        turn-completion path. A provider's ``sync_turn`` may make a
        blocking network/daemon call (a misconfigured Hindsight daemon
        was observed blocking ~298s before failing); doing that inline
        held ``run_conversation`` open long after the user saw their
        response, so every interface (CLI, TUI, gateway) kept the agent
        marked "running" for minutes and any follow-up message triggered
        an aggressive interrupt. Dispatching off-thread means a slow or
        broken provider can never stall the turn — the sync simply
        completes (or fails, logged) in the background.

        Writes are serialized through a single worker so turn N lands
        before turn N+1; provider implementations don't need their own
        ordering guarantees.
        """
        providers = list(self._providers)
        if not providers:
            return

        clean_user_content = self._strip_skill_scaffolding(user_content)
        if not clean_user_content:
            return
        user_content = clean_user_content

        def _run() -> None:
            for provider in providers:
                try:
                    if messages is not None and self._provider_sync_accepts_messages(provider):
                        provider.sync_turn(
                            user_content,
                            assistant_content,
                            session_id=session_id,
                            messages=messages,
                        )
                    else:
                        provider.sync_turn(
                            user_content,
                            assistant_content,
                            session_id=session_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Memory provider '%s' sync_turn failed: %s",
                        provider.name, e,
                    )

        self._submit_background(_run)

    # -- Background dispatch -------------------------------------------------

    def _submit_background(self, fn, *, kind: str = "write") -> None:
        """Queue ``fn`` on the serialized worker and track its durability class.

        The submitted callable is wrapped with the CALLER's contextvars:
        profile isolation in multi-profile processes (gateway multiplexer,
        dashboard, cron) is a ContextVar-scoped HERMES_HOME override, and
        executor worker threads start with empty contexts — without the
        wrap, a provider resolving ambient state (config paths, secrets)
        from the worker would silently land on the default profile.
        """
        import contextvars
        from functools import partial

        ctx = contextvars.copy_context()
        fn = partial(ctx.run, fn)
        executor = self._get_sync_executor()
        if executor is None:
            if self._shutting_down:
                logger.warning("Memory manager is shutting down; rejecting late %s task", kind)
                return
            # Creation failure outside shutdown: preserve the historical
            # fail-safe behavior and run the operation inline.
            try:
                fn()
            except Exception as e:  # pragma: no cover - fn guards internally
                logger.debug("Inline memory background task failed: %s", e)
            return
        try:
            # Make submit+tracking atomic with the shutdown snapshot. The
            # callback is attached after releasing the lock because an already
            # completed future invokes callbacks synchronously.
            with self._sync_executor_lock:
                if self._shutting_down:
                    logger.warning("Memory manager is shutting down; rejecting late %s task", kind)
                    return
                future = executor.submit(fn)
                self._background_futures[future] = kind
            future.add_done_callback(self._forget_background_future)
        except RuntimeError:
            if self._shutting_down:
                logger.warning("Memory manager shut down during %s submission; task rejected", kind)
                return
            try:
                fn()
            except Exception as e:  # pragma: no cover - fn guards internally
                logger.debug("Inline memory background task failed: %s", e)

    def _forget_background_future(self, future: Future) -> None:
        with self._sync_executor_lock:
            self._background_futures.pop(future, None)

    def _get_sync_executor(self) -> Optional[ThreadPoolExecutor]:
        """Lazily create the single-worker background executor."""
        if self._shutting_down:
            return None
        if self._sync_executor is not None:
            return self._sync_executor
        with self._sync_executor_lock:
            if self._shutting_down:
                return None
            if self._sync_executor is None:
                try:
                    # Daemon workers (see tools.daemon_pool): a provider wedged
                    # on a network call must never block interpreter exit.
                    from tools.daemon_pool import DaemonThreadPoolExecutor
                    self._sync_executor = DaemonThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="mem-sync",
                    )
                except Exception as e:  # pragma: no cover - resource exhaustion
                    logger.warning("Failed to create memory sync executor: %s", e)
                    return None
            return self._sync_executor

    def flush_pending(self, timeout: Optional[float] = None) -> bool:
        """Block until queued sync/prefetch work has drained.

        Single-worker executor means submitting a sentinel and waiting on
        it guarantees every previously-submitted task has run. Returns
        True if the barrier completed within ``timeout`` (or no executor
        exists), False on timeout. Used at real session boundaries and by
        tests that need to assert provider state deterministically.
        """
        executor = self._sync_executor
        if executor is None:
            return True
        try:
            fut = executor.submit(lambda: None)
        except RuntimeError:
            # Executor already shut down — nothing pending.
            return True
        try:
            fut.result(timeout=timeout)
            return True
        except Exception:
            return False

    # -- Tools ---------------------------------------------------------------

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Collect tool schemas from all providers.

        Reserved core tool names (``clarify``, ``delegate_task``, etc.) are
        skipped — they are rejected from the routing table in
        :meth:`add_provider`, so the manager must not advertise a schema it
        will never route. Built-ins always win (#40466).
        """
        from toolsets import _HERMES_CORE_TOOLS

        _core_tool_names = set(_HERMES_CORE_TOOLS)
        schemas = []
        seen = set()
        for provider in self._providers:
            try:
                for raw_schema in provider.get_tool_schemas():
                    schema = normalize_tool_schema(raw_schema)
                    if schema is None:
                        logger.warning(
                            "Memory provider '%s' returned a tool schema with "
                            "no resolvable name; skipping (%r)",
                            provider.name, raw_schema,
                        )
                        continue
                    name = schema["name"]
                    if name in _core_tool_names:
                        continue
                    if name not in seen:
                        schemas.append(schema)
                        seen.add(name)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' get_tool_schemas() failed: %s",
                    provider.name, e,
                )
        return schemas

    def get_all_tool_names(self) -> set:
        """Return set of all tool names across all providers."""
        return set(self._tool_to_provider.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if any provider handles this tool."""
        return tool_name in self._tool_to_provider

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        """Route a tool call to the correct provider.

        Returns JSON string result. Raises ValueError if no provider
        handles the tool.
        """
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return tool_error(f"No memory provider handles tool '{tool_name}'")
        try:
            return provider.handle_tool_call(tool_name, args, **kwargs)
        except Exception as e:
            logger.error(
                "Memory provider '%s' handle_tool_call(%s) failed: %s",
                provider.name, tool_name, e,
            )
            return tool_error(f"Memory tool '{tool_name}' failed: {e}")

    # -- Lifecycle hooks -----------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Notify all providers of a new turn.

        kwargs may include: remaining_tokens, model, platform, tool_count.
        """
        for provider in self._providers:
            try:
                provider.on_turn_start(turn_number, message, **kwargs)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_turn_start failed: %s",
                    provider.name, e,
                )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Notify all providers of session end."""
        for provider in self._providers:
            try:
                provider.on_session_end(messages)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' on_session_end failed: %s",
                    provider.name, e,
                    exc_info=True,
                )

    def commit_session_boundary_async(
        self,
        messages: List[Dict[str, Any]],
        *,
        new_session_id: str,
        parent_session_id: str = "",
        reason: str = "new_session",
    ) -> None:
        """Queue old-session extraction + provider rebinding as ONE serialized task.

        Session rotation (/new) must deliver ``on_session_end`` (end-of-session
        extraction — an LLM-bound call that can take seconds) strictly BEFORE
        ``on_session_switch`` (which rebinds provider-internal ``_session_id`` /
        turn buffers to the new session). Running extraction inline blocked the
        /new command for the whole LLM round-trip (#16454); running it on an
        ad-hoc thread raced the inline switch — providers key off internal
        state, so a late ``on_session_end`` ran against post-switch bindings
        (transcript misattributed to the new session id, double-ingest of the
        old turn buffer, new-session buffers cleared).

        Submitting BOTH hooks as one task on the manager's single background
        worker gives both properties at a single chokepoint: the caller returns
        immediately, and the worker's FIFO order serializes end→switch against
        every other provider write (per-turn ``sync_all``, prefetches), which
        already share the same worker. If the executor is unavailable,
        ``_submit_background`` degrades to inline execution — the pre-#16454
        synchronous behavior, slow but correct.
        """
        if not self._providers:
            return
        snapshot = list(messages or [])

        def _run() -> None:
            try:
                self.on_session_end(snapshot)
            except Exception as e:  # pragma: no cover - on_session_end guards per-provider
                logger.warning("Session-boundary extraction failed: %s", e)
            try:
                self.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=True,
                    reason=reason,
                )
            except Exception as e:  # pragma: no cover - on_session_switch guards per-provider
                logger.warning("Session-boundary switch failed: %s", e)

        self._submit_background(_run)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Notify all providers that the agent's session_id has rotated.

        Fires on ``/resume``, ``/branch``, ``/reset``, ``/new``, and
        context compression — any path that reassigns
        ``AIAgent.session_id`` without tearing the provider down.

        Providers keep running; they only need to refresh cached
        per-session state so subsequent writes land in the correct
        session's record. See ``MemoryProvider.on_session_switch`` for
        the full contract.

        ``rewound=True`` signals that session_id is unchanged but the
        transcript was truncated; providers caching per-turn document
        state should invalidate.
        """
        if not new_session_id:
            return
        # Only forward ``rewound`` when it's actually set. Passing it
        # unconditionally would inject ``rewound=False`` into every
        # provider's **kwargs for the common /resume, /branch, /new, and
        # compression paths, polluting providers that capture extra kwargs
        # (and breaking exact-dict assertions). The /undo path sets
        # rewound=True explicitly; everyone else stays clean.
        if rewound:
            kwargs["rewound"] = True
        for provider in self._providers:
            try:
                provider.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=reset,
                    **kwargs,
                )
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_session_switch failed: %s",
                    provider.name, e,
                )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Notify all providers before context compression.

        Returns combined text from providers to include in the compression
        summary prompt. Empty string if no provider contributes.
        """
        parts = []
        for provider in self._providers:
            try:
                result = provider.on_pre_compress(messages)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_pre_compress failed: %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    @staticmethod
    def _provider_memory_write_metadata_mode(provider: MemoryProvider) -> str:
        """Return how to pass metadata to a provider's memory-write hook."""
        try:
            signature = inspect.signature(provider.on_memory_write)
        except (TypeError, ValueError):
            return "keyword"

        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return "keyword"
        if "metadata" in signature.parameters:
            return "keyword"

        accepted = [
            p for p in params
            if p.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        if len(accepted) >= 4:
            return "positional"
        return "legacy"

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify external providers when the built-in memory tool writes.

        Skips the builtin provider itself (it's the source of the write).
        """
        for provider in self._providers:
            if provider.name == "builtin":
                continue
            try:
                metadata_mode = self._provider_memory_write_metadata_mode(provider)
                if metadata_mode == "keyword":
                    provider.on_memory_write(
                        action, target, content, metadata=dict(metadata or {})
                    )
                elif metadata_mode == "positional":
                    provider.on_memory_write(action, target, content, dict(metadata or {}))
                else:
                    provider.on_memory_write(action, target, content)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_memory_write failed: %s",
                    provider.name, e,
                )

    # Actions the bridge mirrors to external providers. The built-in memory
    # tool can also return non-mutating shapes (errors, staged-for-approval
    # records); those are filtered out by ``notify_memory_tool_write`` before
    # we ever reach a provider.
    _MIRRORED_MEMORY_ACTIONS = {"add", "replace", "remove"}

    @staticmethod
    def _memory_tool_result_succeeded(result: Any) -> bool:
        """True only when the built-in memory tool actually committed a write.

        Fails closed: a string that isn't JSON, a non-dict result, a missing
        ``success``, or a write staged for approval (``staged is True``) all
        return False so external providers are never told about a write that
        did not land.
        """
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                return False
        if not isinstance(result, dict):
            return False
        return result.get("success") is True and result.get("staged") is not True

    def notify_memory_tool_write(
        self,
        tool_result: Any,
        tool_args: Dict[str, Any],
        *,
        build_metadata: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Mirror a built-in memory tool call to external providers.

        This is the single entry point the agent loop calls after running the
        built-in ``memory`` tool. All the decisions about *whether* and *what*
        to mirror live here, behind the manager interface — the loop only hands
        over the raw tool result and args:

        * gate on a committed (non-staged, successful) write,
        * expand the single-op and batched (``operations``) shapes,
        * keep only mutating actions (add/replace/remove),
        * build per-op provenance metadata and forward ``old_text``.

        ``build_metadata`` is an optional agent-side callable (the loop knows
        session/task/tool-call provenance the manager does not) invoked once per
        mirrored op.
        """
        if not self._memory_tool_result_succeeded(tool_result):
            return

        target = str(tool_args.get("target") or "memory")
        operations = tool_args.get("operations")
        if isinstance(operations, list) and operations:
            raw_operations = operations
        else:
            raw_operations = [{
                "action": tool_args.get("action"),
                "content": tool_args.get("content"),
                "old_text": tool_args.get("old_text"),
            }]

        for op in raw_operations:
            if not isinstance(op, dict):
                continue
            action = str(op.get("action") or "")
            if action not in self._MIRRORED_MEMORY_ACTIONS:
                continue
            try:
                metadata = dict(build_metadata() if build_metadata else {})
                old_text = op.get("old_text")
                if old_text:
                    metadata["old_text"] = str(old_text)
                self.on_memory_write(
                    action,
                    target,
                    str(op.get("content") or ""),
                    metadata=metadata,
                )
            except Exception as e:
                logger.debug("notify_memory_tool_write failed for op %s: %s", action, e)

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Notify all providers that a subagent completed."""
        for provider in self._providers:
            try:
                provider.on_delegation(
                    task, result, child_session_id=child_session_id, **kwargs
                )
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_delegation failed: %s",
                    provider.name, e,
                )

    def shutdown_all(self) -> None:
        """Shut down all providers (reverse order for clean teardown).

        Drains the background sync/prefetch executor first (bounded by
        ``_SYNC_DRAIN_TIMEOUT_S``) so a turn's final sync has a chance to
        land before providers are torn down. The worker threads are
        daemon, so anything still wedged past the drain window dies with
        the interpreter rather than blocking exit.
        """
        self._drain_sync_executor()
        for provider in reversed(self._providers):
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' shutdown failed: %s",
                    provider.name, e,
                )

    @property
    def shutdown_drain_state(self) -> Dict[str, Any]:
        """Snapshot of the most recent bounded shutdown drain outcome."""
        with self._sync_executor_lock:
            return dict(self._shutdown_drain_state)

    def _drain_sync_executor(self) -> None:
        """Give queued FIFO work a bounded chance, then abandon explicitly."""
        with self._sync_executor_lock:
            self._shutting_down = True
            executor = self._sync_executor
            self._sync_executor = None
            tracked = dict(self._background_futures)
            self._shutdown_drain_state = {
                "status": "draining" if executor is not None else "drained",
                "abandoned_writes": 0,
                "abandoned_prefetches": 0,
                "active_tasks": sum(not future.done() for future in tracked),
            }
        if executor is None:
            return

        # shutdown(wait=False) closes submission without touching the FIFO.
        # Waiting on the tracked futures lets the real single-worker executor
        # run every queued write/boundary task in order up to the deadline.
        executor.shutdown(wait=False, cancel_futures=False)
        _, pending = wait(tuple(tracked), timeout=_SYNC_DRAIN_TIMEOUT_S)
        if not pending:
            with self._sync_executor_lock:
                self._shutdown_drain_state.update(status="drained", active_tasks=0)
            return

        abandoned_writes = 0
        abandoned_prefetches = 0
        active_tasks = 0
        for future in pending:
            kind = tracked[future]
            if future.cancel():
                if kind == "prefetch":
                    abandoned_prefetches += 1
                else:
                    abandoned_writes += 1
            else:
                active_tasks += 1

        with self._sync_executor_lock:
            self._shutdown_drain_state.update(
                status="timed_out",
                abandoned_writes=abandoned_writes,
                abandoned_prefetches=abandoned_prefetches,
                active_tasks=active_tasks,
            )
        logger.warning(
            "Memory shutdown drain timed out after %.2fs; abandoning %d queued "
            "memory write(s) and %d queued prefetch(es); %d active task(s) remain detached",
            _SYNC_DRAIN_TIMEOUT_S,
            abandoned_writes,
            abandoned_prefetches,
            active_tasks,
        )

    def initialize_all(self, session_id: str, **kwargs) -> None:
        """Initialize all providers.

        Automatically injects ``hermes_home`` into *kwargs* so that every
        provider can resolve profile-scoped storage paths without importing
        ``get_hermes_home()`` themselves.
        """
        if "hermes_home" not in kwargs:
            from hermes_constants import get_hermes_home
            kwargs["hermes_home"] = str(get_hermes_home())
        for provider in self._providers:
            try:
                provider.initialize(session_id=session_id, **kwargs)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' initialize failed: %s",
                    provider.name, e,
                )
