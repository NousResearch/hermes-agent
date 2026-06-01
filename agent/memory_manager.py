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

_FENCE_TAG_RE = re.compile(r'</?\s*memory-context\s*>', re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(
    r'<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>',
    re.IGNORECASE,
)
_REVERSED_CONTEXT_RE = re.compile(
    r'</\s*memory-context\s*>[\s\S]*?<\s*memory-context\s*>',
    re.IGNORECASE,
)
_ORPHAN_CLOSE_CONTEXT_RE = re.compile(
    r'</\s*memory-context\s*>[\s\S]*',
    re.IGNORECASE,
)
# Greedy fallback: unclosed <memory-context> → strip everything after open tag
_UNCLOSED_CONTEXT_RE = re.compile(
    r'<\s*memory-context\s*>[\s\S]*',
    re.IGNORECASE,
)
_INTERNAL_NOTE_RE = re.compile(
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*',
    re.IGNORECASE,
)
_RESTART_RECOVERY_NOTE_RE = re.compile(
    r'\[System note:\s*Your previous turn(?: in this session)? was interrupted.*?\]\s*',
    re.IGNORECASE | re.DOTALL,
)
# Detect echoed system-prompt memory blocks (no <memory-context> tags).
_ECHOED_MEMORY_HEADER_RE = re.compile(
    r'^[═]{20,}\s*\n'
    r'MEMORY \(your personal notes\).*\n'
    r'[═]{20,}\s*\n'
    r'[\s\S]*?'
    r'(?=\n[═]{20,}\s*\n[^\n]*USER PROFILE|'
    r'\n[═]{20,}\s*\n[^\n]*MEMORY \(|'
    r'\n\n[A-Z][a-z]+[\s:—]|'
    r'\Z)',
    re.MULTILINE,
)
_ECHOED_USER_PROFILE_RE = re.compile(
    r'^[═]{20,}\s*\n'
    r'USER PROFILE \(who the user is\).*\n'
    r'[═]{20,}\s*\n'
    r'[\s\S]*?'
    r'(?=\n[═]{20,}\s*\n[^\n]*(?:USER PROFILE|MEMORY)|'
    r'\n\n[A-Z][a-z]+[\s:—]|'
    r'\Z)',
    re.MULTILINE,
)
_ECHOED_HINDSIGHT_RE = re.compile(
    r'\[Hindsight Sidecar Recall[^\]]*\]\n[\s\S]*?'
    r'(?=\n\[(?:Provider Recall|Hindsight|Persona|Atoms|Scenario)\]|\Z)',
    re.MULTILINE,
)
_ECHOED_PROVIDER_RECALL_RE = re.compile(
    r'\[Provider Recall\]\n[\s\S]*?'
    r'(?=\n\[(?:Hindsight|Persona|Atoms|Scenario)\]|\Z)',
    re.MULTILINE,
)
_ECHOED_LAYERS_RE = re.compile(
    r'\[(?:Persona|Atoms|Scenario)\]\n[\s\S]*?'
    r'(?=\n\[(?:Persona|Atoms|Scenario|Provider Recall|Hindsight)\]|\Z)',
    re.MULTILINE,
)


def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, and system notes from provider output."""
    _orig = text
    # 1. Balanced <memory-context>...</memory-context> blocks
    text = _INTERNAL_CONTEXT_RE.sub('', text)
    # 1b. Reversed/orphan close→open spans are malformed but still unsafe.
    text = _REVERSED_CONTEXT_RE.sub('', text)
    # 1c. Orphan close tags are malformed but unsafe; fail closed by dropping
    #     the rest of the output rather than exposing possible context payload.
    text = _ORPHAN_CLOSE_CONTEXT_RE.sub('', text)
    # 2. System notes
    text = _INTERNAL_NOTE_RE.sub('', text)
    text = _RESTART_RECOVERY_NOTE_RE.sub('', text)
    # 3. Unclosed <memory-context> — greedy: strip everything after open tag.
    #    This MUST run before generic fence-tag removal; otherwise the opener
    #    would be deleted first and the payload after it would leak.
    text = _UNCLOSED_CONTEXT_RE.sub('', text)
    # 4. Orphan fence tags (for example stray closing tags) are stripped as
    #    inert markup after all payload-bearing spans have been removed.
    text = _FENCE_TAG_RE.sub('', text)
    # 5. Echoed system-prompt memory content (no tags, just raw content)
    text = _ECHOED_MEMORY_HEADER_RE.sub('', text)
    text = _ECHOED_USER_PROFILE_RE.sub('', text)
    text = _ECHOED_HINDSIGHT_RE.sub('', text)
    text = _ECHOED_PROVIDER_RECALL_RE.sub('', text)
    text = _ECHOED_LAYERS_RE.sub('', text)
    # 6. Collapse excessive blank lines left by removals
    text = re.sub(r'\n{3,}', '\n\n', text)
    result = text.strip()
    if len(_orig) != len(result):
        logger.warning("sanitize_context: %d -> %d chars (stripped %d)", len(_orig), len(result), len(_orig) - len(result))
    return result


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
    _TAG_NAME = "memory-context"

    def __init__(self) -> None:
        self._in_span: bool = False
        self._drop_until_reset: bool = False
        self._buf: str = ""
        self._at_block_boundary: bool = True

    def reset(self) -> None:
        self._in_span = False
        self._drop_until_reset = False
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
        if self._drop_until_reset:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_span:
                close_match = self._find_tag(buf, closing=True)
                if close_match is None:
                    # Hold back a potential partial close tag; drop the rest
                    held = self._max_possible_tag_suffix(buf)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                # Found close — skip span content + tag, continue
                _, end = close_match
                buf = buf[end:]
                self._in_span = False
            else:
                open_match = self._find_tag(buf, closing=False)
                close_match = self._find_tag(buf, closing=True)
                if close_match is not None and (
                    open_match is None or close_match[0] < open_match[0]
                ):
                    start, end = close_match
                    if start > 0:
                        self._append_visible(out, buf[:start])
                    self._buf = ""
                    self._in_span = False
                    self._drop_until_reset = True
                    return "".join(out)
                if open_match is None:
                    # No open tag — hold back a potential partial open tag
                    held = self._max_possible_tag_suffix(buf)
                    if held:
                        self._append_visible(out, buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        self._append_visible(out, buf)
                    return "".join(out)
                # Emit text before the tag, enter span
                start, end = open_match
                if start > 0:
                    self._append_visible(out, buf[:start])
                buf = buf[end:]
                self._in_span = True

        return "".join(out)

    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream.

        If we're still inside an unterminated span the remaining content is
        discarded (safer: leaking partial memory context is worse than a
        truncated answer).  Otherwise the held-back partial-tag tail is
        emitted verbatim (it turned out not to be a real tag).
        """
        if self._drop_until_reset:
            self._buf = ""
            return ""
        if self._in_span:
            self._buf = ""
            self._in_span = False
            return ""
        tail = self._buf
        self._buf = ""
        return tail

    def _find_tag(self, buf: str, *, closing: bool) -> tuple[int, int] | None:
        """Find a complete memory-context tag using sanitizer-compatible grammar.

        Matches ``<\\s*memory-context\\s*>`` and
        ``</\\s*memory-context\\s*>`` case-insensitively.  If a partial tag is
        present at the end of ``buf`` it returns ``None`` so the caller can
        hold the tail for the next chunk.
        """
        lower = buf.lower()
        tag_len = len(self._TAG_NAME)
        search_start = 0
        while True:
            idx = lower.find("<", search_start)
            if idx == -1:
                return None
            pos = idx + 1
            if pos >= len(buf):
                return None
            if closing:
                if lower[pos] != "/":
                    search_start = idx + 1
                    continue
                pos += 1
                if pos >= len(buf):
                    return None
            else:
                if lower[pos] == "/":
                    search_start = idx + 1
                    continue
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos + tag_len > len(buf):
                return None
            if lower[pos:pos + tag_len] != self._TAG_NAME:
                search_start = idx + 1
                continue
            pos += tag_len
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos >= len(buf):
                return None
            if buf[pos] == ">":
                return idx, pos + 1
            search_start = idx + 1

    def _max_possible_tag_suffix(self, buf: str) -> int:
        """Hold a tail that could become a whitespace-tolerant fence tag.

        Whitespace inside the final sanitizer's tag regex is unbounded, so a
        small fixed suffix cap is unsafe: ``<`` followed by a long whitespace
        run can be split across chunks and must not be emitted before the tag
        name arrives.  Scan every suffix; the strings are tiny streaming
        deltas, and privacy beats micro-optimizing this path.
        """
        for size in range(len(buf), 0, -1):
            tail = buf[-size:]
            if self._could_be_tag_prefix(tail):
                return size
        return 0

    def _could_be_tag_prefix(self, text: str) -> bool:
        lower = text.lower()
        if not lower.startswith("<"):
            return False
        pos = 1
        if pos < len(lower) and lower[pos] == "/":
            pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        typed = lower[pos:]
        # If the tag name is complete, only whitespace or the final '>' may
        # follow.  If incomplete, it must be a prefix of memory-context.
        if len(typed) <= len(self._TAG_NAME):
            return self._TAG_NAME.startswith(typed)
        return typed.startswith(self._TAG_NAME) and all(
            ch.isspace() or ch == ">" for ch in text[pos + len(self._TAG_NAME):]
        )

    def _append_visible(self, out: list[str], text: str) -> None:
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
