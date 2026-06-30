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
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from agent.skill_commands import extract_user_instruction_from_skill_message
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# How long shutdown_all() waits for in-flight background sync/prefetch work
# to drain before abandoning it. A wedged provider must never block process
# teardown indefinitely — the worker threads are daemon, so anything still
# running past this window dies with the interpreter.
_SYNC_DRAIN_TIMEOUT_S = 5.0


def memory_provider_tools_enabled(enabled_toolsets: Optional[List[str]]) -> bool:
    """Return whether external memory-provider tools should be exposed."""
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
    if (
        "memory" not in existing_tool_names
        and not memory_provider_tools_enabled(getattr(agent, "enabled_toolsets", None))
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
    for schema in get_schemas():
        if not isinstance(schema, dict):
            continue
        tool_name = schema.get("name", "")
        if not tool_name or tool_name in existing_tool_names:
            continue
        tools.append({"type": "function", "function": schema})
        valid_tool_names.add(tool_name)
        existing_tool_names.add(tool_name)
        added += 1

    return added


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


def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, and system notes from provider output."""
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
        self._at_block_boundary: bool = True

    def reset(self) -> None:
        self._in_span = False
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
                idx = self._find_boundary_open_tag(buf)
                if idx == -1:
                    # No open tag — hold back a potential partial open tag
                    held = (
                        self._max_pending_open_suffix(buf)
                        or self._max_partial_suffix(buf, self._OPEN_TAG)
                    )
                    if held:
                        self._append_visible(out, buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        self._append_visible(out, buf)
                    return "".join(out)
                # Emit text before the tag, enter span
                if idx > 0:
                    self._append_visible(out, buf[:idx])
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

    def _find_boundary_open_tag(self, buf: str) -> int:
        """Find an opening fence only when it starts a block-like span."""
        buf_lower = buf.lower()
        search_start = 0
        while True:
            idx = buf_lower.find(self._OPEN_TAG, search_start)
            if idx == -1:
                return -1
            if self._is_block_boundary(buf, idx) and self._has_block_opener_suffix(buf, idx):
                return idx
            search_start = idx + 1

    def _max_pending_open_suffix(self, buf: str) -> int:
        """Hold a complete boundary tag until the following char confirms it."""
        if not buf.lower().endswith(self._OPEN_TAG):
            return 0
        idx = len(buf) - len(self._OPEN_TAG)
        if not self._is_block_boundary(buf, idx):
            return 0
        return len(self._OPEN_TAG)

    def _has_block_opener_suffix(self, buf: str, idx: int) -> bool:
        after_idx = idx + len(self._OPEN_TAG)
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
        # Background executor for end-of-turn sync/prefetch. Lazily created on
        # first use so the common builtin-only path spawns no extra threads.
        # A single worker serializes a provider's writes (turn N must land
        # before turn N+1) and caps thread growth at one per manager. See
        # _submit_background() and the sync_all/queue_prefetch_all rationale.
        self._sync_executor: Optional[ThreadPoolExecutor] = None
        self._sync_executor_lock = threading.Lock()

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
                result = provider.prefetch(clean_query, session_id=session_id)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' prefetch failed (non-fatal): %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    def prefetch_all_scored(
        self,
        intent_context: Any,
        *,
        session_id: str = "",
        builtin_entries: Optional[List[str]] = None,
        memory_store: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Collect scored memory candidates from all providers + builtin store.

        Stage 2-4 of the relevance pipeline:
          1. Each provider returns scored candidates via ``prefetch_with_intent``
          2. If builtin_entries provided, score them against the intent context
          3. Fuse all candidates (dedup + trust weighting + score threshold)

        Returns a deduplicated, threshold-filtered, score-ranked list of
        candidate dicts ready for gating + injection formatting.

        Empty list means nothing worth injecting.
        """
        from agent.memory.fusion import fuse, apply_score_threshold
        from agent.memory.gating import _DEFAULT_MIN_RELEVANCE

        all_candidates: List[Dict[str, Any]] = []

        # Stage 2a: External providers (structured scoring)
        for provider in self._providers:
            if provider.name == "builtin":
                continue
            try:
                candidates = provider.prefetch_with_intent(
                    intent_context, session_id=session_id,
                )
                if candidates:
                    all_candidates.extend(candidates)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' prefetch_with_intent failed: %s",
                    provider.name, e,
                )

        # Stage 2b: Builtin store — score raw entries against intent
        if builtin_entries:
            # Prefer instance method (FTS5+embedding) if MemoryStore available,
            # fall back to static classmethod (heuristic only).
            if memory_store is not None and hasattr(memory_store, "scored_prefetch_instance"):
                try:
                    builtin_candidates = memory_store.scored_prefetch_instance(
                        task_type=intent_context.task_type,
                        task_confidence=intent_context.task_confidence,
                        keywords=intent_context.keywords,
                        expanded_query=intent_context.expanded_query,
                    )
                    if builtin_candidates:
                        all_candidates.extend(builtin_candidates)
                except Exception as e:
                    logger.debug("Builtin scored_prefetch_instance failed: %s", e)
            else:
                from tools.memory_tool import MemoryStore
                try:
                    builtin_candidates = MemoryStore.scored_prefetch(
                        builtin_entries,
                        task_type=intent_context.task_type,
                        task_confidence=intent_context.task_confidence,
                        keywords=intent_context.keywords,
                        expanded_query=intent_context.expanded_query,
                    )
                    if builtin_candidates:
                        all_candidates.extend(builtin_candidates)
                except Exception as e:
                    logger.debug("Builtin scored_prefetch failed: %s", e)

        if not all_candidates:
            return []

        # Stage 3: Cross-provider fusion (dedup + trust weighting)
        from agent.memory.scored_memory import ScoredMemory

        # Convert dicts to ScoredMemory objects for fusion
        scored_objects = []
        for c in all_candidates:
            sm = ScoredMemory(
                content=c.get("content", ""),
                provider=c.get("provider", "unknown"),
                score=c.get("score", 0.0),
                context_tag=c.get("context_tag"),
                signal_strength=c.get("signal_strength", 0.5),
                score_semantic=c.get("score_semantic", 0.0),
                score_keyword=c.get("score_keyword", 0.0),
                score_context=c.get("score_context", 0.0),
                score_entity=c.get("score_entity", 0.0),
                score_temporal=c.get("score_temporal", 1.0),
                score_signal=c.get("score_signal", 0.5),
                source_file=c.get("source_file"),
                entry_index=c.get("entry_index"),
            )
            scored_objects.append(sm)

        fused = fuse(scored_objects)

        # Stage 4: Apply score threshold
        passed = apply_score_threshold(fused, _DEFAULT_MIN_RELEVANCE)

        # Convert back to dicts for the caller
        result = []
        for sm in passed:
            result.append({
                "content": sm.content,
                "score": sm.score,
                "provider": sm.provider,
                "context_tag": sm.context_tag,
                "signal_strength": sm.signal_strength,
                "source_file": sm.source_file,
                "entry_index": sm.entry_index,
                "score_semantic": sm.score_semantic,
                "score_keyword": sm.score_keyword,
                "score_context": sm.score_context,
                "score_entity": sm.score_entity,
                "score_temporal": sm.score_temporal,
                "score_signal": sm.score_signal,
            })

        return result

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

        self._submit_background(_run)

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

    def _submit_background(self, fn) -> None:
        """Run ``fn`` on the manager's background worker.

        The executor is created lazily and shared across calls. If the
        executor can't be created or has already been shut down, ``fn``
        runs inline as a last-resort fallback — losing the async benefit
        but never losing the write itself. ``fn`` must do its own
        per-provider error handling; this wrapper only guards executor
        plumbing.
        """
        executor = self._get_sync_executor()
        if executor is None:
            # Executor unavailable (shut down / creation failed) — run
            # inline rather than drop the work. Slow, but correct.
            try:
                fn()
            except Exception as e:  # pragma: no cover - fn guards internally
                logger.debug("Inline memory background task failed: %s", e)
            return
        try:
            executor.submit(fn)
        except RuntimeError:
            # Executor was shut down between the get and the submit
            # (teardown race). Fall back to inline.
            try:
                fn()
            except Exception as e:  # pragma: no cover - fn guards internally
                logger.debug("Inline memory background task failed: %s", e)

    def _get_sync_executor(self) -> Optional[ThreadPoolExecutor]:
        """Lazily create the single-worker background executor."""
        if self._sync_executor is not None:
            return self._sync_executor
        with self._sync_executor_lock:
            if self._sync_executor is None:
                try:
                    self._sync_executor = ThreadPoolExecutor(
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
                logger.warning(
                    "Memory provider '%s' on_session_end failed: %s",
                    provider.name, e,
                    exc_info=True,
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

    def _drain_sync_executor(self) -> None:
        """Shut down the background executor, waiting briefly for drain.

        Bounded by ``_SYNC_DRAIN_TIMEOUT_S``: a wedged provider must never
        hang process/session teardown. We stop accepting new work and
        cancel anything still queued, then wait at most the drain timeout
        for the currently-running task on a watcher thread. The worker is
        daemon, so an over-running task dies with the interpreter.
        """
        with self._sync_executor_lock:
            executor = self._sync_executor
            self._sync_executor = None
        if executor is None:
            return
        try:
            # Stop accepting new work and drop anything still queued, but
            # do NOT block here — cancel_futures cancels not-yet-started
            # tasks; the in-flight one keeps running on its daemon thread.
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Older Python without cancel_futures kwarg.
            try:
                executor.shutdown(wait=False)
            except Exception as e:  # pragma: no cover
                logger.debug("Memory sync executor shutdown failed: %s", e)
            return
        except Exception as e:  # pragma: no cover
            logger.debug("Memory sync executor shutdown failed: %s", e)
            return
        # Give an in-flight sync a bounded chance to finish on a watcher
        # thread so we don't block the caller past the drain timeout.
        drainer = threading.Thread(
            target=lambda: self._bounded_executor_wait(executor),
            daemon=True,
            name="mem-sync-drain",
        )
        drainer.start()
        drainer.join(timeout=_SYNC_DRAIN_TIMEOUT_S)

    @staticmethod
    def _bounded_executor_wait(executor: ThreadPoolExecutor) -> None:
        try:
            executor.shutdown(wait=True)
        except Exception as e:  # pragma: no cover
            logger.debug("Memory sync executor drain wait failed: %s", e)

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
