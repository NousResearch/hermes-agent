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

import logging
import re
import inspect
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context fencing helpers
# ---------------------------------------------------------------------------

_INTERNAL_CONTEXT_TAG_NAMES = (
    r"memory-context",
    r"recalled[_-]memory[_-]context",
    r"supermemory-context",
    r"ship[_-]mode[_-]guard",
)
_INTERNAL_CONTEXT_TAG_RE = r"(?:" + "|".join(_INTERNAL_CONTEXT_TAG_NAMES) + r")"
_FENCE_TAG_RE = re.compile(rf'</?\s*{_INTERNAL_CONTEXT_TAG_RE}\s*>', re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(
    rf'<\s*{_INTERNAL_CONTEXT_TAG_RE}\s*>[\s\S]*?</\s*{_INTERNAL_CONTEXT_TAG_RE}\s*>',
    re.IGNORECASE,
)
_UNTERMINATED_INTERNAL_CONTEXT_RE = re.compile(
    rf'<\s*{_INTERNAL_CONTEXT_TAG_RE}\s*>\s*'
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.'
    r'[\s\S]*$',
    re.IGNORECASE,
)
_UNTERMINATED_SHIP_MODE_TAG_RE = re.compile(
    r'<\s*ship[_-]mode[_-]guard\s*>[\s\S]*$',
    re.IGNORECASE,
)
_UNTERMINATED_CONTEXT_BLOCK_RE = re.compile(
    rf'(^|\n)[ \t]*<\s*{_INTERNAL_CONTEXT_TAG_RE}\s*>[ \t]*\r?\n[\s\S]*$',
    re.IGNORECASE,
)
_INTERNAL_NOTE_RE = re.compile(
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*'
    r'(?:Treat as (?:informational background data|authoritative reference data[^\]]*)\.|'
    r'Recalled memory is useful context, not authoritative;[^\]]*)\]\s*',
    re.IGNORECASE,
)
_SHIP_MODE_GUARD_RE = re.compile(
    r'\[\s*Ship-mode routing guard:[\s\S]*?\]\s*',
    re.IGNORECASE,
)
_RAW_MEMORY_HEADING_RE = re.compile(
    r'^##\s*(?:Honcho Context|User Representation|Explicit Observations|User Peer Card|'
    r'AI Self-Representation|Recalled assistant context|AI Identity Card)\s*$',
    re.IGNORECASE | re.MULTILINE,
)
_COMPACT_MEMORY_CONTEXT_RE = re.compile(
    r'(?ims)(^|\n)[ \t]*#\s*Memory context pointers\s*\n[\s\S]*?(?=\n{2,}(?![ \t]*(?:[-*]|##\s*Compact peer preferences|Knowledge Store:|Tech Stack:|Design Preference:|Model Routing:|Active Project:))|\Z)'
)
_COMPACT_PEER_PREFERENCES_RE = re.compile(
    r'(?ims)(^|\n)[ \t]*##\s*Compact peer preferences\s*\n[\s\S]*?(?=\n{2,}(?![ \t]*(?:Knowledge Store:|Tech Stack:|Design Preference:|Model Routing:|Active Project:))|\Z)'
)
_HONCHO_CONTEXT_SUMMARY_RE = re.compile(
    r'(?ims)(^|\n)[ \t]*(?:The current conversation context|The most relevant context to (?:our|the) current conversation)\b[\s\S]*?(?=\n{2,}|\Z)'
)
_HONCHO_LEAK_SUMMARY_LINE_RE = re.compile(
    r'(?im)^.*\b(?:foundational context|observations indicate)\b.*(?:\n|$)'
)
_AIVS_AUTONOMOUS_LOOP_RE = re.compile(
    r'(?im)^.*Check\s+the\s+AIVS\s+Hermes\s+Kanban\s+board\s+on\s+the\s+VPS\s+'
    r'and\s+keep\s+the\s+autonomous\s+dev\s+loop\s+moving[^\n]*(?:\n(?!\s*$).*)*\n?',
)
_LEADING_COMPACTION_FALLBACK_RE = re.compile(
    r'^\s*\[CONTEXT COMPACTION\s+[^\]]*\]'
    r'[\s\S]*?'
    r'(?:Summary generation was unavailable\.[^\n]*(?:\n|$))'
    r'(?:[^\n]*removed to free context space[^\n]*(?:\n|$))?'
    r'(?:[^\n]*messages contained earlier work[^\n]*(?:\n|$))?',
    re.IGNORECASE,
)
_LEADING_GATEWAY_SYSTEM_NOTE_RE = re.compile(
    r'^\s*\[System note:\s*Your previous turn(?: in this session)? (?:was interrupted|in this session was interrupted)[^\]]*\]\s*',
    re.IGNORECASE,
)


def _strip_loose_context_tags(text: str) -> str:
    """Strip loose internal fence tags without deleting normal prose mentions."""
    text = re.sub(rf'</\s*{_INTERNAL_CONTEXT_TAG_RE}\s*>', '', text, flags=re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        start, end = match.span()
        prev_char = text[start - 1] if start > 0 else ''
        next_char = text[end] if end < len(text) else ''
        # Preserve literal documentation/prose mentions such as
        # "`<memory-context>`", "The <memory-context> tag", and
        # "<memory-context> is the literal tag name". Real leaked blocks are
        # removed above when followed by a newline; suspicious inline opener
        # escapes are stripped.
        if prev_char == '`' or next_char == '`':
            return match.group(0)
        if next_char.isspace() and next_char not in '\r\n':
            return match.group(0)
        return ''

    return re.sub(rf'<\s*{_INTERNAL_CONTEXT_TAG_RE}\s*>', repl, text, flags=re.IGNORECASE)


def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, and system notes from provider output."""
    text = _LEADING_COMPACTION_FALLBACK_RE.sub('', text)
    text = _LEADING_GATEWAY_SYSTEM_NOTE_RE.sub('', text)
    text = _SHIP_MODE_GUARD_RE.sub('', text)
    text = _COMPACT_MEMORY_CONTEXT_RE.sub(lambda m: m.group(1), text)
    text = _COMPACT_PEER_PREFERENCES_RE.sub(lambda m: m.group(1), text)
    text = _HONCHO_CONTEXT_SUMMARY_RE.sub(lambda m: m.group(1), text)
    text = _HONCHO_LEAK_SUMMARY_LINE_RE.sub('', text)
    text = _AIVS_AUTONOMOUS_LOOP_RE.sub('', text)
    text = _INTERNAL_CONTEXT_RE.sub('', text)
    text = _UNTERMINATED_INTERNAL_CONTEXT_RE.sub('', text)
    text = _UNTERMINATED_SHIP_MODE_TAG_RE.sub('', text)
    text = _UNTERMINATED_CONTEXT_BLOCK_RE.sub(lambda m: m.group(1), text)
    text = _INTERNAL_NOTE_RE.sub('', text)
    text = _RAW_MEMORY_HEADING_RE.sub('## Recalled context', text)
    text = _strip_loose_context_tags(text)
    return text


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
    """

    _TAG_SPANS = tuple(
        (f"<{name}>", f"</{name}>", True)
        for name in (
            "memory-context",
            "recalled_memory_context",
            "recalled-memory-context",
            "supermemory-context",
            "ship_mode_guard",
            "ship-mode-guard",
        )
    )

    def __init__(self) -> None:
        self._in_span: bool = False
        self._close_tags: tuple[str, ...] = ()
        self._buf: str = ""
        self._at_block_boundary: bool = True

    def reset(self) -> None:
        self._in_span = False
        self._close_tags = ()
        self._buf = ""
        self._at_block_boundary = True

    def feed(self, text: str) -> str:
        """Return the visible portion of ``text`` after scrubbing.

        Any trailing fragment that could be the start of an open/close tag
        is held back in the internal buffer and surfaced on the next
        ``feed()`` call or discarded/emitted by ``flush()``.
        """
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_span:
                match = self._find_earliest_tag(buf, self._close_tags)
                if match is None:
                    # Hold back a potential partial close tag; drop the rest
                    held = self._max_partial_suffix_any(buf, self._close_tags)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                idx, close_tag = match
                # Found close — skip span content + tag, continue
                buf = buf[idx + len(close_tag):]
                self._in_span = False
                self._close_tags = ()
            else:
                match = self._find_boundary_open_span(buf)
                if match is None:
                    # No open tag — hold back a potential partial open tag
                    held = (
                        self._max_pending_open_suffix(buf)
                        or self._max_partial_suffix_any(
                            buf,
                            tuple(open_tag for open_tag, _close_tag, _requires_newline in self._TAG_SPANS),
                        )
                    )
                    if held:
                        self._append_visible(out, buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        self._append_visible(out, buf)
                    return "".join(out)
                idx, open_tag, close_tag = match
                # Emit text before the tag, enter span
                if idx > 0:
                    self._append_visible(out, buf[:idx])
                buf = buf[idx + len(open_tag):]
                self._in_span = True
                self._close_tags = (close_tag,)

        return "".join(out)

    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream.

        If we're still inside an unterminated span the remaining content is
        discarded (safer: leaking partial memory context is worse than a
        truncated answer).  Otherwise the held-back partial-tag tail is
        emitted after one final sanitizer pass.
        """
        if self._in_span:
            self._buf = ""
            self._in_span = False
            self._close_tags = ()
            return ""
        tail = self._buf
        self._buf = ""
        return sanitize_context(tail)

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Return the length of the longest buf-suffix that is a tag-prefix.

        Case-insensitive.  Returns 0 if no suffix could start the tag.
        """
        tag_lower = tag.lower()
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), len(tag_lower) - 1)
        for i in range(max_check, 0, -1):
            if tag_lower.startswith(buf_lower[-i:]):
                return i
        return 0

    @classmethod
    def _max_partial_suffix_any(cls, buf: str, tags: tuple[str, ...]) -> int:
        return max((cls._max_partial_suffix(buf, tag) for tag in tags), default=0)

    @staticmethod
    def _find_earliest_tag(buf: str, tags: tuple[str, ...]) -> tuple[int, str] | None:
        buf_lower = buf.lower()
        best: tuple[int, str] | None = None
        for tag in tags:
            idx = buf_lower.find(tag)
            if idx == -1:
                continue
            if best is None or idx < best[0]:
                best = (idx, tag)
        return best

    def _find_boundary_open_span(self, buf: str) -> tuple[int, str, str] | None:
        """Find an opening fence only when it starts a block-like span."""
        buf_lower = buf.lower()
        best: tuple[int, str, str] | None = None
        for open_tag, close_tag, requires_newline in self._TAG_SPANS:
            search_start = 0
            while True:
                idx = buf_lower.find(open_tag, search_start)
                if idx == -1:
                    break
                if self._is_block_boundary(buf, idx) and (
                    not requires_newline or self._has_block_opener_suffix(buf, idx, open_tag)
                ):
                    candidate = (idx, open_tag, close_tag)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                    break
                search_start = idx + 1
        return best

    def _max_pending_open_suffix(self, buf: str) -> int:
        """Hold a complete boundary tag until the following char confirms it."""
        buf_lower = buf.lower()
        for open_tag, _close_tag, requires_newline in self._TAG_SPANS:
            if not requires_newline or not buf_lower.endswith(open_tag):
                continue
            idx = len(buf) - len(open_tag)
            if self._is_block_boundary(buf, idx):
                return len(open_tag)
        return 0

    def _has_block_opener_suffix(self, buf: str, idx: int, open_tag: str) -> bool:
        after_idx = idx + len(open_tag)
        if after_idx >= len(buf):
            return False
        return buf[after_idx] in "\r\n"

    def _is_block_boundary(self, buf: str, idx: int) -> bool:
        if idx == 0:
            return self._at_block_boundary
        preceding = buf[:idx]
        last_newline = preceding.rfind("\n")
        if last_newline == -1:
            return self._at_block_boundary and preceding.strip() == ""
        return preceding[last_newline + 1:].strip() == ""

    def _append_visible(self, out: list[str], text: str) -> None:
        if not text:
            return
        text = sanitize_context(text)
        if not text:
            return
        out.append(text)
        self._update_block_boundary(text)

    def _update_block_boundary(self, text: str) -> None:
        last_newline = text.rfind("\n")
        if last_newline != -1:
            self._at_block_boundary = text[last_newline + 1:].strip() == ""
        else:
            self._at_block_boundary = self._at_block_boundary and text.strip() == ""


_SECTION_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_SAFE_CONTEXT_SECTION_NAMES = {
    "session summary",
    "compact peer preferences",
    "recalled user context (recent/relevant)",
}
_USER_PEER_CARD_SECTION_NAMES = {"user peer card", "user profile", "peer card"}
_HONCHO_LABEL_RE = re.compile(r"^(attribute|instruction)\s*:\s*(.+)$", re.IGNORECASE)
_SAFE_PEER_PREFERENCE_LABELS = {
    "active project",
    "design preference",
    "knowledge store",
    "model routing",
    "tech stack",
}


def _normalize_peer_preference_label(label: str) -> str:
    """Normalize compact peer-card labels before fail-closed allowlisting."""
    label = label.strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", label)


def _safe_peer_preference_line(line: str) -> str | None:
    """Return a safe compact preference line, or None for unapproved labels.

    Peer cards can contain arbitrary identity fields (names, handles,
    employers, relationships, locations, dates, etc.).  This parser is
    fail-closed: only explicitly approved preference labels survive, including
    when Honcho prefixes the line with ATTRIBUTE:/INSTRUCTION:.
    """
    stripped = line.strip()
    if not stripped:
        return None

    label_match = _HONCHO_LABEL_RE.match(stripped)
    if label_match:
        stripped = label_match.group(2).strip()

    label, separator, value = stripped.partition(":")
    if separator != ":":
        return None
    label = re.sub(r"\s+", " ", label.strip())
    value = value.strip()
    if not label or not value:
        return None
    if _normalize_peer_preference_label(label) not in _SAFE_PEER_PREFERENCE_LABELS:
        return None
    return f"{label}: {value}"


def compact_user_peer_card(body: str) -> str:
    """Keep only allowlisted compact peer preference labels from peer-card text."""
    lines: list[str] = []
    for line in sanitize_context(body).splitlines():
        safe_line = _safe_peer_preference_line(line)
        if safe_line:
            lines.append(safe_line)
    return _truncate_memory_context("\n".join(lines), 1200)


def _truncate_memory_context(text: str, limit: int) -> str:
    """Bound auto-injected recall text before it reaches the active prompt."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("; "))
    if boundary >= int(limit * 0.65):
        cut = cut[:boundary].rstrip()
    return cut.rstrip() + " …"


def _extract_safe_memory_sections(raw_context: str) -> list[str]:
    """Keep only compact current/session + peer-card sections from recall.

    Honcho/context backends may return raw representations, old explicit
    observations, and assistant self-representation. Those remain recoverable
    through memory tools / durable stores, but normal provider prompts get only
    a compact peer/session slice plus file-backed retrieval pointers.
    """
    matches = list(_SECTION_HEADING_RE.finditer(raw_context or ""))
    sections: list[str] = []
    for idx, match in enumerate(matches):
        name = match.group(1).strip()
        section_name = name.lower()
        if section_name not in _SAFE_CONTEXT_SECTION_NAMES and section_name not in _USER_PEER_CARD_SECTION_NAMES:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_context)
        body = sanitize_context(raw_context[start:end]).strip()
        if not body:
            continue
        if section_name in _USER_PEER_CARD_SECTION_NAMES:
            body = compact_user_peer_card(body)
            if not body:
                continue
            sections.append(f"## Compact peer preferences\n{body}")
            continue
        sections.append(f"## {name}\n{_truncate_memory_context(body, 1200)}")
    return sections


def build_memory_context_block(raw_context: str) -> str:
    """Return no automatic recall context for the active provider prompt.

    Gateway-visible sessions proved that even previously "safe" compact
    memory pointers can be quoted back to users. Memory remains available via
    explicit Honcho/wiki/session tools, but no recalled context block is pasted
    into ordinary user messages at API-call time.
    """
    return ""


class MemoryManager:
    """Orchestrates the built-in provider plus at most one external provider.

    The builtin provider is always first. Only one non-builtin (external)
    provider is allowed.  Failures in one provider never block the other.
    """

    def __init__(self) -> None:
        self._providers: List[MemoryProvider] = []
        self._tool_to_provider: Dict[str, MemoryProvider] = {}
        self._has_external: bool = False  # True once a non-builtin provider is added

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
        for schema in provider.get_tool_schemas():
            tool_name = schema.get("name", "")
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

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Collect prefetch context from all providers.

        Returns merged context text labeled by provider. Empty providers
        are skipped. Failures in one provider don't block others.
        """
        parts = []
        for provider in self._providers:
            try:
                result = provider.prefetch(query, session_id=session_id)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' prefetch failed (non-fatal): %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch on all providers for the next turn."""
        for provider in self._providers:
            try:
                provider.queue_prefetch(query, session_id=session_id)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' queue_prefetch failed (non-fatal): %s",
                    provider.name, e,
                )

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
        """Sync a completed turn to all providers."""
        for provider in self._providers:
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
                for schema in provider.get_tool_schemas():
                    name = schema.get("name", "")
                    if name in _core_tool_names:
                        continue
                    if name and name not in seen:
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
                logger.debug(
                    "Memory provider '%s' on_session_end failed: %s",
                    provider.name, e,
                )

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
        """Shut down all providers (reverse order for clean teardown)."""
        for provider in reversed(self._providers):
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' shutdown failed: %s",
                    provider.name, e,
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
