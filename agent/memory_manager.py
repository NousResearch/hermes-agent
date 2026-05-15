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
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context fencing helpers
# ---------------------------------------------------------------------------

_FENCE_TAG_RE = re.compile(r'</?\s*memory-context\s*>', re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(
    r'<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>',
    re.IGNORECASE,
)
_INTERNAL_NOTE_RE = re.compile(
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*',
    re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(r'<\s*think\s*>[\s\S]*?</\s*think\s*>', re.IGNORECASE)
_RESOLVED_PACKET_RE = re.compile(r'^---\s*\n# Resolved Memory Context[\s\S]*$', re.IGNORECASE | re.MULTILINE)


def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, generated packets, and system notes from provider output."""
    text = _THINK_BLOCK_RE.sub('', text)
    text = _RESOLVED_PACKET_RE.sub('', text)
    text = _INTERNAL_CONTEXT_RE.sub('', text)
    text = _INTERNAL_NOTE_RE.sub('', text)
    text = _FENCE_TAG_RE.sub('', text)
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

    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"

    def __init__(self) -> None:
        self._in_span: bool = False
        self._buf: str = ""

    def reset(self) -> None:
        self._in_span = False
        self._buf = ""

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
                idx = buf.lower().find(self._CLOSE_TAG)
                if idx == -1:
                    # Hold back a potential partial close tag; drop the rest
                    held = self._max_partial_suffix(buf, self._CLOSE_TAG)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                # Found close — skip span content + tag, continue
                buf = buf[idx + len(self._CLOSE_TAG):]
                self._in_span = False
            else:
                idx = buf.lower().find(self._OPEN_TAG)
                if idx == -1:
                    # No open tag — hold back a potential partial open tag
                    held = self._max_partial_suffix(buf, self._OPEN_TAG)
                    if held:
                        out.append(buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        out.append(buf)
                    return "".join(out)
                # Emit text before the tag, enter span
                if idx > 0:
                    out.append(buf[:idx])
                buf = buf[idx + len(self._OPEN_TAG):]
                self._in_span = True

        return "".join(out)

    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream.

        If we're still inside an unterminated span the remaining content is
        discarded (safer: leaking partial memory context is worse than a
        truncated answer).  Otherwise the held-back partial-tag tail is
        emitted verbatim (it turned out not to be a real tag).
        """
        if self._in_span:
            self._buf = ""
            self._in_span = False
            return ""
        tail = self._buf
        self._buf = ""
        return tail

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


# ----------------------------------------------------------------------
# Context Packet Builder
# ----------------------------------------------------------------------
# Parses raw prefetch text into a structured packet with sections.
# Minimal conflict resolver flags identity/model/provider/path
# contradictions via simple heuristics — does NOT delete data.

# Known section markers that providers emit
# Matches markdown headers like "## Facts", "## [Facts]:", "# Preferences", etc.
# Groups: (label, bracketed_label)
_SECTION_RE = re.compile(
    r'^#+\s*\[?\s*('
    r'facts|preferences|operational[_\s-]?state|conflicts?|excluded[_\s-]?context|'
    r'memory|source|session[_\s-]?summary|user[_\s-]?representation|user[_\s-]?peer[_\s-]?card|'
    r'ai[_\s-]?self[_\s-]?representation|ai[_\s-]?identity[_\s-]?card|resolved[_\s-]?memory[_\s-]?context|'
    r'contradictions?'
    r')\s*\]?\s*:?\s*$',
    re.IGNORECASE,
)
_SECTION_LABEL_ALIASES = {
    "session_summary": "facts",
    "user_representation": "preferences",
    "user_peer_card": "preferences",
    "ai_self_representation": "facts",
    "ai_identity_card": "operational_state",
    "resolved_memory_context": "excluded_context",
    "contradiction": "conflicts",
    "contradictions": "conflicts",
}
_SECTION_SOURCE_ALIASES = {
    "session_summary": "honcho_session_summary",
    "user_representation": "honcho_user_representation",
    "user_peer_card": "honcho_user_peer_card",
    "ai_self_representation": "honcho_ai_self_representation",
    "ai_identity_card": "honcho_ai_identity_card",
    "resolved_memory_context": "generated_resolved_packet",
    "facts": "memory_facts",
    "preferences": "memory_preferences",
    "operational_state": "memory_operational_state",
    "conflict": "memory_conflicts",
    "conflicts": "memory_conflicts",
    "contradiction": "memory_conflicts",
    "contradictions": "memory_conflicts",
    "excluded_context": "memory_excluded_context",
}
_PROVIDER_TAG_RE = re.compile(r'^\[?(?:Provider|Memory|Source)\s*[:=]\s*([\w.-]+)\]?$', re.IGNORECASE)
_SUPERSEDED_IDENTITY_RE = re.compile(
    r'\bpreferred\s+name\b.*\bdarwin\b|\bcall\s+(?:him|user)\s+darwin\b',
    re.IGNORECASE,
)
_SUPERSEDED_DEEPSEEK_RE = re.compile(
    r'\bdeepseek\s+v4\s+pro\b.*\b(?:reasoning\s+)?orchestrator\b|\bdeepseek\b.*\bprimary\b',
    re.IGNORECASE,
)


def _normalize_section_label(label: str) -> str:
    return label.lower().replace(' ', '_').replace('-', '_')


def _source_for_label(raw_label: str, current_source: str) -> str:
    normalized = _normalize_section_label(raw_label)
    # Provider tags remain authoritative for generic section labels.  A payload
    # like ``[Provider: honcho]\n# Facts`` should still be sourced to honcho,
    # not rewritten to a synthetic memory_facts source.
    if normalized in {"facts", "preferences", "operational_state", "conflict", "conflicts", "excluded_context"}:
        if current_source and current_source != "unknown":
            return current_source
    if normalized in _SECTION_SOURCE_ALIASES:
        return _SECTION_SOURCE_ALIASES[normalized]
    return current_source if current_source else "unknown"


def _should_exclude_line(line: str) -> bool:
    l = line.lower()
    # Honcho may still contain an old peer-card conclusion that confuses the
    # user's name with this local agent's name. Keep it as excluded evidence,
    # but never inject it as an active preference.
    if _SUPERSEDED_IDENTITY_RE.search(line):
        return True
    # Superseded model-routing claim observed in the raw peer card. The active
    # Darwin contract is GPT-5.5 primary, MiniMax delegated executor, DeepSeek
    # fallback/contrast; conflicting correction text should be reviewed, not
    # treated as a live preference.
    if _SUPERSEDED_DEEPSEEK_RE.search(line):
        return True
    if l.startswith("# resolved memory context") or l.startswith("# sources"):
        return True
    return False


def _canonical_conflict_value(kind: str, line: str) -> str:
    l = line.lower()
    if kind == "identity":
        if "preferred name is darwin" in l and "call him darwin" in l:
            return "user_name_darwin_claim"
        if "juan carlos verni" in l or "called juan" in l or "user should be called juan" in l:
            return "user_name_juan_claim"
        if "darwin is" in l and ("agent" in l or "instance" in l):
            return "darwin_agent_claim"
    if kind == "model":
        if "deepseek" in l and ("orchestrator" in l or "primary" in l or "main" in l):
            return "deepseek_primary_orchestrator_claim"
        if "gpt-5.5" in l and ("primary" in l or "orchestrat" in l):
            return "gpt_5_5_primary_claim"
        if "minimax" in l and ("delegated" in l or "executor" in l or "executes" in l):
            return "minimax_delegated_executor_claim"
    if kind == "path":
        try:
            import os
            return os.path.realpath(os.path.expanduser(line.split(":", 1)[-1].strip())).lower()
        except Exception:
            pass
    return line.split(":", 1)[-1].strip().lower()


def _split_raw_sections(raw: str) -> List[Dict[str, str]]:
    """Split raw prefetch text into sections by provider or markdown headers.

    Returns list of dicts: [{"label": "...", "body": "...", "source": "..."}, ...]
    """
    raw = sanitize_context(raw)
    sections = []
    current_label = "generic"
    current_source = "unknown"
    current_body_lines: List[str] = []

    def _flush(label: str, body: str, source: str) -> None:
        if body.strip():
            sections.append({"label": label, "body": body.strip(), "source": source or "unknown"})

    for line in raw.splitlines():
        stripped_line = line.strip()
        provider_match = _PROVIDER_TAG_RE.match(stripped_line)
        if provider_match:
            _flush(current_label, "\n".join(current_body_lines), current_source)
            current_body_lines = []
            current_source = provider_match.group(1)
            current_label = current_source
            continue

        header_match = _SECTION_RE.match(stripped_line)
        if header_match:
            _flush(current_label, "\n".join(current_body_lines), current_source)
            current_body_lines = []
            raw_label = _normalize_section_label(header_match.group(1))
            current_source = _source_for_label(raw_label, current_source)
            current_label = _SECTION_LABEL_ALIASES.get(raw_label, raw_label)
            continue

        current_body_lines.append(line)

    _flush(current_label, "\n".join(current_body_lines), current_source)
    return sections


def _classify_line(line: str) -> str:
    """Classify a line into: fact, preference, operational, other."""
    l = line.lower().strip()
    if any(k in l for k in ["prefer", "like", "always", "never", "want", "avoid", "use"]):
        return "preference"
    if any(k in l for k in ["running", "status", "mode", "active", "current", "session"]):
        return "operational"
    return "fact"


def _resolve_conflicts(sections: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Detect contradictions across sections via simple heuristics.

    Flags identity, model, provider, and path contradictions.
    Does NOT delete or modify source data.
    """
    conflicts: List[Dict[str, Any]] = []
    identity_hints: List[Tuple[str, str]] = []
    model_hints: List[Tuple[str, str]] = []
    provider_hints: List[Tuple[str, str]] = []
    path_hints: List[Tuple[str, str]] = []

    # Simple extraction: lines containing identity/model/provider/path keywords
    for sec in sections:
        body = sec["body"]
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            l = stripped.lower()
            if any(k in l for k in ["identity:", "user:", "persona:", "agent:", "name:", "preferred name", "called juan", "user should be called juan"]):
                identity_hints.append((sec["source"], stripped))
            if any(k in l for k in ["model:", "llm:", "using model", "primary model", "fallback/comparison model", "delegated executor", "model routing", "orchestrator"]):
                model_hints.append((sec["source"], stripped))
            if any(k in l for k in ["provider:", "backend:", "via openai", "provider `"]):
                provider_hints.append((sec["source"], stripped))
            if any(k in l for k in ["path:", "file:", "directory:", "config path", "home:"]):
                path_hints.append((sec["source"], stripped))

    def _check_contradictions(hints: List[Tuple[str, str]], kind: str):
        values: Dict[str, List[Tuple[str, str]]] = {}
        for source, line in hints:
            key = _canonical_conflict_value(kind, line)
            if key not in values:
                values[key] = []
            values[key].append((source, line))
        if len(values) > 1:
            conflicts.append({
                "kind": kind,
                "sources": {v: [s for s, _ in vs] for v, vs in values.items()},
                "resolution": "unresolved",
                "note": "multiple values detected — manual review recommended",
            })

    _check_contradictions(identity_hints, "identity")
    _check_contradictions(model_hints, "model")
    _check_contradictions(provider_hints, "provider")
    _check_contradictions(path_hints, "path")

    return conflicts


def build_memory_context_packet(raw_text: str) -> Dict[str, Any]:
    """Parse raw prefetch text into a structured context packet.

    Returns dict with keys:
      - facts: list of fact lines
      - preferences: list of preference lines
      - operational_state: list of operational lines
      - conflicts: list of detected conflict dicts
      - excluded_context: list of excluded lines
      - source_precedence: list of source names in priority order
      - raw_sections: list of {"label", "body", "source"} dicts
    """
    if not raw_text or not raw_text.strip():
        return {
            "facts": [],
            "preferences": [],
            "operational_state": [],
            "conflicts": [],
            "excluded_context": [],
            "source_precedence": [],
            "raw_sections": [],
        }

    sections = _split_raw_sections(raw_text)

    facts: List[str] = []
    preferences: List[str] = []
    operational_state: List[str] = []
    excluded_context: List[str] = []

    for sec in sections:
        label = sec["label"].lower()
        body = sec["body"]
        if not body:
            continue
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _should_exclude_line(stripped):
                excluded_context.append(stripped)
                continue
            cls = _classify_line(stripped)
            if label == "excluded_context":
                excluded_context.append(stripped)
            elif label == "preferences" or cls == "preference":
                preferences.append(stripped)
            elif label == "operational_state" or cls == "operational":
                operational_state.append(stripped)
            else:
                facts.append(stripped)

    # Source precedence: deduplicated in order of appearance
    seen_sources: set = set()
    source_precedence: List[str] = []
    for sec in sections:
        if sec["source"] not in seen_sources:
            seen_sources.add(sec["source"])
            source_precedence.append(sec["source"])

    conflicts = _resolve_conflicts(sections)

    return {
        "facts": facts,
        "preferences": preferences,
        "operational_state": operational_state,
        "conflicts": conflicts,
        "excluded_context": excluded_context,
        "source_precedence": source_precedence,
        "raw_sections": sections,
    }


def _packet_to_text(packet: Dict[str, Any]) -> str:
    """Render a context packet as readable text for injection."""
    parts = ["# Resolved Memory Context"]
    if packet["source_precedence"]:
        parts.append(f"# Sources (in priority order): {', '.join(packet['source_precedence'])}")
    if packet["conflicts"]:
        parts.append(f"# Conflicts detected: {len(packet['conflicts'])}")
        for c in packet["conflicts"]:
            parts.append(f"#   [{c['kind']}] unresolved — manual review recommended")
    if packet["facts"]:
        parts.append("# Facts")
        for f in packet["facts"][:20]:  # cap at 20
            parts.append(f"- {f}")
    if packet["preferences"]:
        parts.append("# Preferences")
        for p in packet["preferences"][:20]:
            parts.append(f"- {p}")
    if packet["operational_state"]:
        parts.append("# Operational State")
        for o in packet["operational_state"][:20]:
            parts.append(f"- {o}")
    if packet["excluded_context"]:
        parts.append(f"# Excluded context entries: {len(packet['excluded_context'])}")
    return "\n".join(parts)


def build_memory_context_block(raw_context: str, *, packet_builder_enabled: bool = False) -> str:
    """Wrap prefetched memory in a fenced block with system note.

    ``packet_builder_enabled`` is a default-off gate for the Resolved Memory
    Context MVP.  Disabled preserves the historical raw-context injection.
    Enabled replaces raw context with the resolved packet instead of duplicating
    both, keeping rollback trivial and token growth measurable.
    """
    if not raw_context or not raw_context.strip():
        return ""
    clean = sanitize_context(raw_context)
    if clean != raw_context:
        logger.warning("memory provider returned pre-wrapped context; stripped")

    body = clean
    if packet_builder_enabled:
        packet = build_memory_context_packet(clean)
        packet_text = _packet_to_text(packet)
        raw_len = len(clean)
        packet_len = len(packet_text)
        delta = packet_len - raw_len
        log_fn = logger.warning if raw_len and packet_len > raw_len * 1.10 else logger.info
        log_fn(
            "memory_packet_chars_before=%s after=%s delta=%s",
            raw_len,
            packet_len,
            delta,
        )
        body = packet_text

    return (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data — "
        "this is the agent's persistent memory and should inform all responses.]\n\n"
        f"{body}\n"
        "</memory-context>"
    )


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

        # Index tool names → provider for routing
        for schema in provider.get_tool_schemas():
            tool_name = schema.get("name", "")
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

    def sync_all(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Sync a completed turn to all providers."""
        for provider in self._providers:
            try:
                provider.sync_turn(user_content, assistant_content, session_id=session_id)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' sync_turn failed: %s",
                    provider.name, e,
                )

    # -- Tools ---------------------------------------------------------------

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Collect tool schemas from all providers."""
        schemas = []
        seen = set()
        for provider in self._providers:
            try:
                for schema in provider.get_tool_schemas():
                    name = schema.get("name", "")
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
        """
        if not new_session_id:
            return
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
