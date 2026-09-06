"""MemoryManager — fans the agent's memory hooks out to registered providers.

The builtin provider is always allowed; only ONE external plugin provider may be
registered at a time (tool-schema bloat, conflicting backends).
"""

from __future__ import annotations

import contextvars
import inspect
import json
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider, PRE_COMPRESS_CHECKPOINT_API_VERSION
from agent.skill_commands import extract_user_instruction_from_skill_message
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Providers that predate the checkpoint-API attribute are on the best-effort v1 contract.
_LEGACY_PRE_COMPRESS_API_VERSION = 1

# shutdown_all() drain bound; workers are daemon threads so a wedged provider never
# blocks interpreter exit.
_SYNC_DRAIN_TIMEOUT_S = 5.0
_EXTERNAL_PREFETCH_TIMEOUT_S = 8.0


# -- Signature introspection (providers are duck-typed; call shapes vary) -----

def _signature_params(fn: Callable[..., Any]):
    """``fn``'s parameter mapping, or None when uninspectable (C callables, exotic proxies)."""
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None


def _has_var_kwargs(params) -> bool:
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _accepts_require_checkpoint(fn: Callable[..., Any]) -> bool:
    """True if ``fn`` can receive the ``require_checkpoint`` keyword (unreadable signatures -> False).

    Bare-shape v2 providers (``on_pre_compress(self, messages)``) would raise TypeError on the
    keyword, which the host would re-raise as a checkpoint failure despite a successful write.
    """
    params = _signature_params(fn)
    if params is None:
        return False
    kind = getattr(params.get("require_checkpoint"), "kind", None)
    return _has_var_kwargs(params) or kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _ctx_bound(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Bind ``fn`` to the CALLER's contextvars for another thread: profile isolation is a
    ContextVar-scoped HERMES_HOME override, and an unbound worker would silently use the default profile."""
    return partial(contextvars.copy_context().run, fn)


# -- Tool-schema plumbing -----------------------------------------------------

def normalize_tool_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """Return a bare function-tool dict with a resolvable top-level ``name``, else None.

    Providers should return ``{"name", "description", "parameters"}`` but some return the
    wrapped OpenAI form; wrapping that twice yields a nameless ``function`` and strict
    providers (DeepSeek) reject the ENTIRE request, so both shapes are normalized here.
    """
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
    name = schema.get("name", "")
    return schema if name and isinstance(name, str) else None


def memory_provider_tools_enabled(enabled_toolsets: Optional[List[str]], disabled_toolsets: Optional[List[str]] = None,
                                  *, memory_tool_present: bool = False) -> bool:
    """Return whether external memory-provider tools should be exposed."""
    if disabled_toolsets and "memory" in disabled_toolsets:
        return False
    if memory_tool_present or enabled_toolsets is None:
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


def _tool_name(tool: Any) -> Any:
    return tool.get("function", {}).get("name") if isinstance(tool, dict) else None


def memory_provider_tools_exposed(agent: Any) -> bool:
    """Whether external memory-provider tools are exposed on ``agent``.

    Same gate as ``inject_memory_provider_tools`` so a provider's ``system_prompt_block()``
    never advertises tools absent from the tool surface.
    """
    tools = getattr(agent, "tools", None)
    present = isinstance(tools, (list, tuple)) and any(_tool_name(t) == "memory" for t in tools)
    enabled, disabled = getattr(agent, "enabled_toolsets", None), getattr(agent, "disabled_toolsets", None)
    return memory_provider_tools_enabled(enabled, disabled, memory_tool_present=present)


def inject_memory_provider_tools(agent: Any) -> int:
    """Append external memory-provider tool schemas to an agent tool surface; return count added."""
    memory_manager = getattr(agent, "_memory_manager", None)
    tools = getattr(agent, "tools", None)
    if not memory_manager or tools is None:
        return 0

    if not memory_provider_tools_exposed(agent):
        # Say so once: a silent 0 leaves the provider looking "half on" with no clue which
        # config key (platform_toolsets / disabled_toolsets) gated it.
        # See #81014.
        _providers = [p for p in getattr(memory_manager, "providers", None) or []
                      if getattr(p, "name", "") != "builtin"]
        if _providers:
            logger.info(
                "Memory provider(s) %s configured but the 'memory' toolset is "
                "gated off for this session (platform_toolsets / "
                "agent.disabled_toolsets) — provider tools and system-prompt "
                "block are both withheld.",
                [getattr(p, "name", type(p).__name__) for p in _providers],
            )
        return 0

    get_schemas = getattr(memory_manager, "get_all_tool_schemas", None)
    if not callable(get_schemas):
        return 0

    if getattr(agent, "valid_tool_names", None) is None:
        agent.valid_tool_names = set()
    existing_tool_names = {_tool_name(tool) for tool in tools if isinstance(tool, dict)}
    added = 0
    for raw_schema in get_schemas():
        schema = normalize_tool_schema(raw_schema)
        if schema is None:
            logger.warning(
                "Memory provider returned a tool schema with no resolvable "
                "name; skipping to avoid poisoning the request (%r)", raw_schema,
            )
        elif schema["name"] not in existing_tool_names:
            tools.append({"type": "function", "function": schema})
            agent.valid_tool_names.add(schema["name"])
            existing_tool_names.add(schema["name"])
            added += 1
    return added


# -- Context fencing helpers --------------------------------------------------

_FENCE_TAG_RE = re.compile(r'</?\s*memory-context\s*>', re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(r'<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>', re.IGNORECASE)
_INTERNAL_NOTE_RE = re.compile(
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*',
    re.IGNORECASE,
)


def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, and system notes from provider output."""
    for pattern in (_INTERNAL_CONTEXT_RE, _INTERNAL_NOTE_RE, _FENCE_TAG_RE):
        text = pattern.sub('', text)
    return text


class StreamingContextScrubber:
    """Stateful scrubber for streaming text whose memory-context spans may straddle deltas.

    ``sanitize_context`` needs both tags in one string, so a split span would leak to the UI;
    this holds back partial-tag tails between ``feed()`` calls and drops span interiors.
    One scrubber (or ``reset()``) per top-level response; call ``flush()`` at end of stream.
    """

    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._in_span: bool = False
        self._buf: str = ""
        self._at_block_boundary: bool = True

    def feed(self, text: str) -> str:
        """Return the visible portion of ``text``; a possible partial tag tail is held for the next call."""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []
        while buf:
            if self._in_span:
                tag = self._CLOSE_TAG
                idx = buf.lower().find(tag)
                held = self._max_partial_suffix(buf, tag)  # potential partial close tag
            else:
                tag = self._OPEN_TAG
                idx = self._find_boundary_open_tag(buf)
                # A complete boundary tag at the buffer end is held until the next char confirms it.
                n = len(tag)
                pending = n if buf.lower().endswith(tag) and self._ends_at_block_boundary(buf[:-n]) else 0
                held = pending or self._max_partial_suffix(buf, tag)
            if idx == -1:
                # Hold back the possible partial tag; inside a span the rest is dropped.
                if not self._in_span:
                    self._append_visible(out, buf[:-held] if held else buf)
                self._buf = buf[-held:] if held else ""
                break
            if not self._in_span:
                self._append_visible(out, buf[:idx])
            buf = buf[idx + len(tag):]
            self._in_span = not self._in_span
        return "".join(out)

    def flush(self) -> str:
        """Emit the held-back tail at end-of-stream; inside an unterminated span it is discarded
        (leaking partial memory context is worse than a truncated answer)."""
        tail = "" if self._in_span else self._buf
        self._buf = ""
        self._in_span = False
        return tail

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Length of the longest buf-suffix that is a (case-insensitive) prefix of ``tag``, else 0."""
        tag_lower, buf_lower = tag.lower(), buf.lower()
        span = range(min(len(buf_lower), len(tag_lower) - 1), 0, -1)
        return next((i for i in span if tag_lower.startswith(buf_lower[-i:])), 0)

    def _find_boundary_open_tag(self, buf: str) -> int:
        """Find an opening fence only when it starts a block-like span (own line, newline after)."""
        buf_lower, tag_len = buf.lower(), len(self._OPEN_TAG)
        idx = buf_lower.find(self._OPEN_TAG)
        while idx != -1:
            after_idx = idx + tag_len
            if self._ends_at_block_boundary(buf[:idx]) and after_idx < len(buf) and buf[after_idx] in "\r\n":
                return idx
            idx = buf_lower.find(self._OPEN_TAG, idx + 1)
        return -1

    def _ends_at_block_boundary(self, text: str) -> bool:
        """Whether emitting ``text`` leaves the stream at a line start (blank tail after the last newline;
        no newline at all -> only whitespace and already at a boundary)."""
        head, sep, tail = text.rpartition("\n")
        return tail.strip() == "" and (bool(sep) or self._at_block_boundary)

    def _append_visible(self, out: list[str], text: str) -> None:
        if text:
            out.append(text)
            self._at_block_boundary = self._ends_at_block_boundary(text)


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
    """Builtin provider (always first) plus at most one external provider.

    Failures in one provider never block the other: every fan-out hook logs and
    swallows per-provider exceptions.
    """

    def __init__(self, *, external_prefetch_timeout: Optional[float] = None) -> None:
        self._providers: List[MemoryProvider] = []
        self._tool_to_provider: Dict[str, MemoryProvider] = {}
        self._has_external: bool = False
        timeout = external_prefetch_timeout
        timeout = _EXTERNAL_PREFETCH_TIMEOUT_S if timeout is None else float(timeout)
        if timeout <= 0:
            raise ValueError("external_prefetch_timeout must be positive")
        self._external_prefetch_timeout = timeout
        self._external_prefetch_threads: Dict[str, threading.Thread] = {}
        self._external_prefetch_lock = threading.Lock()
        # Single-worker background executor for end-of-turn sync/prefetch, created lazily so
        # the builtin-only path spawns no threads; one worker serializes a provider's writes.
        self._sync_executor: Optional[ThreadPoolExecutor] = None
        self._sync_executor_lock = threading.Lock()
        # Futures by durability class ("write" / "prefetch") so shutdown can drain FIFO
        # within a bound, then report exactly what it abandoned.
        self._background_futures: Dict[Future, str] = {}
        self._shutting_down = False
        self._shutdown_drain_state: Dict[str, Any] = {
            "status": "not_started", "abandoned_writes": 0, "abandoned_prefetches": 0, "active_tasks": 0,
        }

    def _each_provider(self, label: str, call: Callable[[MemoryProvider], Any], *, level: int = logging.DEBUG,
                       providers: Optional[List[MemoryProvider]] = None, exc_info: bool = False) -> List[Any]:
        """Call ``call(provider)`` per provider, logging+swallowing failures; returns successes in order.
        ``label`` completes the log line ``Memory provider '<name>' <label>: <exc>``."""
        results: List[Any] = []
        for provider in self._providers if providers is None else providers:
            try:
                results.append(call(provider))
            except Exception as e:
                logger.log(level, "Memory provider '%s' %s: %s", provider.name, label, e, exc_info=exc_info)
        return results

    def add_provider(self, provider: MemoryProvider) -> None:
        """Register a provider; builtin always accepted, only ONE external allowed."""
        if provider.name != "builtin":
            if self._has_external:
                existing = next((p.name for p in self._providers if p.name != "builtin"), "unknown")
                logger.warning(
                    "Rejected memory provider '%s' — external provider '%s' is "
                    "already registered. Only one external memory provider is "
                    "allowed at a time. Configure which one via memory.provider "
                    "in config.yaml.", provider.name, existing,
                )
                return
            self._has_external = True

        self._providers.append(provider)

        # Core tool names are reserved: built-ins always win at agent init, so a shadowing
        # provider tool would linger in ``_tool_to_provider`` and hijack dispatch.
        # ``clarify``, ``delegate_task``). Reject it here, at the door, so it never enters the routing table
        # at all — matching the built-ins-always-win invariant used by the TTS/browser/search provider
        # registries. See #40466.
        from toolsets import _HERMES_CORE_TOOLS

        for raw_schema in provider.get_tool_schemas():
            schema = normalize_tool_schema(raw_schema)
            if schema is None:
                continue
            tool_name = schema["name"]
            if tool_name in _HERMES_CORE_TOOLS:
                logger.warning(
                    "Memory provider '%s' tool '%s' shadows a reserved core "
                    "tool name; registration ignored. Core tools always win — "
                    "rename the provider's tool to something unique.", provider.name, tool_name,
                )
            elif tool_name in self._tool_to_provider:
                logger.warning(
                    "Memory tool name conflict: '%s' already registered by %s, "
                    "ignoring from %s", tool_name, self._tool_to_provider[tool_name].name, provider.name,
                )
            else:
                self._tool_to_provider[tool_name] = provider

        logger.info("Memory provider '%s' registered (%d tools)", provider.name, len(provider.get_tool_schemas()))

    @property
    def providers(self) -> List[MemoryProvider]:
        return list(self._providers)

    def get_provider(self, name: str) -> Optional[MemoryProvider]:
        return next((p for p in self._providers if p.name == name), None)

    def build_system_prompt(self) -> str:
        """Join every provider's non-empty ``system_prompt_block()`` with blank lines."""
        blocks = self._each_provider("system_prompt_block() failed", lambda p: p.system_prompt_block(),
                                      level=logging.WARNING)
        return "\n\n".join(b for b in blocks if b and b.strip())

    # A /skill or /bundle turn embeds the whole skill body in the model-facing message;
    # providers get just the user's instruction (None for a bare invocation).
    _strip_skill_scaffolding = staticmethod(extract_user_instruction_from_skill_message)

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Merge non-empty prefetch context from all providers (failures are non-fatal)."""
        clean_query = self._strip_skill_scaffolding(query)
        if not clean_query:
            return ""
        parts = self._each_provider(
            "prefetch failed (non-fatal)", lambda p: self._prefetch_provider(p, clean_query, session_id=session_id),
        )
        return "\n\n".join(p for p in parts if p and p.strip())

    def _prefetch_provider(self, provider: MemoryProvider, query: str, *, session_id: str = "") -> str:
        """Run one provider's prefetch; external providers are bounded by a timeout. A stuck external
        call keeps running on its daemon thread and the provider is skipped on later turns until it returns."""
        if provider.name == "builtin":
            return provider.prefetch(query, session_id=session_id)

        result_box: Dict[str, Any] = {}

        def _run() -> None:
            try:
                result_box["value"] = provider.prefetch(query, session_id=session_id) or ""
            except Exception as exc:  # pragma: no cover - re-raised by caller
                result_box["error"] = exc

        thread = threading.Thread(target=_ctx_bound(_run), daemon=True, name=f"memory-prefetch-{provider.name}")
        with self._external_prefetch_lock:
            existing = self._external_prefetch_threads.get(provider.name)
            if existing is not None and existing.is_alive():
                logger.debug("Memory provider '%s' prefetch is still running; skipping this turn", provider.name)
                return ""
            self._external_prefetch_threads[provider.name] = thread
            thread.start()

        thread.join(self._external_prefetch_timeout)
        if thread.is_alive():
            logger.warning(
                "Memory provider '%s' prefetch timed out after %.1fs; skipping it until "
                "the stuck call returns", provider.name, self._external_prefetch_timeout,
            )
            return ""

        with self._external_prefetch_lock:
            if self._external_prefetch_threads.get(provider.name) is thread:
                self._external_prefetch_threads.pop(provider.name, None)
        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("value", "")

    def describe_recall(self) -> str:
        """Deterministic recall indicator line (e.g. ``"🧠 Provider — recalled 3 memories"``); ``""`` if none.
        Call right after :meth:`prefetch_all` so the user SEES memory was used even if the model is silent."""
        segments: List[str] = []
        for status in self._each_provider("recall_status failed (non-fatal)", lambda p: p.recall_status()):
            if status is None:
                continue
            # count <= 0: content injected but no discrete count (reflect)
            detail = ("recalled 1 memory" if status.count == 1 else f"recalled {status.count} memories"
                      if status.count > 1 else "recalled relevant memory")
            segments.append(f"{status.glyph} {status.provider_label} — {detail}")
        return "  ".join(segments)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch on all providers for the next turn (see ``sync_all``)."""
        providers = list(self._providers)
        clean_query = self._strip_skill_scaffolding(query) if providers else None
        if not clean_query:
            return
        self._submit_background(lambda: self._each_provider(
            "queue_prefetch failed (non-fatal)", lambda p: p.queue_prefetch(clean_query, session_id=session_id),
            providers=providers,
        ), kind="prefetch")

    @staticmethod
    def _provider_sync_accepts_messages(provider: MemoryProvider) -> bool:
        """Whether ``sync_turn`` accepts a ``messages`` keyword (uninspectable → assume yes)."""
        params = _signature_params(provider.sync_turn)
        return params is None or _has_var_kwargs(params) or "messages" in params

    def sync_all(self, user_content: str, assistant_content: str, *, session_id: str = "",
                 messages: Optional[List[Dict[str, Any]]] = None) -> None:
        """Sync a completed turn to all providers on the background worker.

        Never inline: a provider's ``sync_turn`` may block for minutes, which kept ``run_conversation``
        open after the user saw the response. The single worker also serializes writes (turn N before N+1).
        """
        providers = list(self._providers)
        clean_user_content = self._strip_skill_scaffolding(user_content) if providers else None
        if not clean_user_content:
            return

        def _sync(provider: MemoryProvider) -> None:
            kwargs: Dict[str, Any] = {"session_id": session_id}
            if messages is not None and self._provider_sync_accepts_messages(provider):
                kwargs["messages"] = messages
            provider.sync_turn(clean_user_content, assistant_content, **kwargs)

        self._submit_background(
            lambda: self._each_provider("sync_turn failed", _sync, level=logging.WARNING, providers=providers)
        )

    def _submit_background(self, fn, *, kind: str = "write") -> None:
        """Queue ``fn`` on the serialized worker (created lazily; None once shutting down) and track its
        durability class. Runs under the caller's contextvars (``_ctx_bound``). If the executor is
        unavailable outside shutdown, run inline — the historical fail-safe."""
        fn = _ctx_bound(fn)
        executor = None if self._shutting_down else self._sync_executor
        if executor is None and not self._shutting_down:
            with self._sync_executor_lock:
                if self._sync_executor is None and not self._shutting_down:
                    try:
                        # Daemon workers: a wedged provider must never block interpreter exit.
                        from tools.daemon_pool import DaemonThreadPoolExecutor
                        self._sync_executor = DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="mem-sync")
                    except Exception as e:  # pragma: no cover - resource exhaustion
                        logger.warning("Failed to create memory sync executor: %s", e)
                executor = self._sync_executor
        future = None
        try:
            # Submit+track atomically with the shutdown snapshot. The callback is attached
            # outside the lock: an already-completed future invokes callbacks synchronously.
            with self._sync_executor_lock:
                if self._shutting_down:
                    logger.warning("Memory manager is shutting down; rejecting late %s task", kind)
                    return
                if executor is not None:
                    future = executor.submit(fn)
                    self._background_futures[future] = kind
        except RuntimeError:
            if self._shutting_down:
                logger.warning("Memory manager shut down during %s submission; task rejected", kind)
                return
        if future is not None:
            future.add_done_callback(self._forget_background_future)
            return
        try:
            fn()
        except Exception as e:  # pragma: no cover - fn guards internally
            logger.debug("Inline memory background task failed: %s", e)

    def _forget_background_future(self, future: Future) -> None:
        with self._sync_executor_lock:
            self._background_futures.pop(future, None)

    def flush_pending(self, timeout: Optional[float] = None) -> bool:
        """Block until queued sync/prefetch work has drained (False on timeout).
        With a single worker, a sentinel task completing proves every earlier task ran."""
        executor = self._sync_executor
        if executor is None:
            return True
        try:
            executor.submit(lambda: None).result(timeout=timeout)
        except Exception as e:
            return isinstance(e, RuntimeError)  # executor already shut down — nothing pending
        return True

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Collect deduplicated tool schemas from all providers; reserved core tool names are
        skipped because :meth:`add_provider` refuses to route them."""
        from toolsets import _HERMES_CORE_TOOLS

        schemas: List[Dict[str, Any]] = []
        seen = set()

        def _collect(provider: MemoryProvider) -> None:
            for raw_schema in provider.get_tool_schemas():
                schema = normalize_tool_schema(raw_schema)
                if schema is None:
                    logger.warning(
                        "Memory provider '%s' returned a tool schema with "
                        "no resolvable name; skipping (%r)", provider.name, raw_schema,
                    )
                elif schema["name"] not in _HERMES_CORE_TOOLS and schema["name"] not in seen:
                    schemas.append(schema)
                    seen.add(schema["name"])

        self._each_provider("get_tool_schemas() failed", _collect, level=logging.WARNING)
        return schemas

    def get_all_tool_names(self) -> set:
        return set(self._tool_to_provider)

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_to_provider

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Route a tool call to its provider; returns a JSON string (tool_error on failure)."""
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return tool_error(f"No memory provider handles tool '{tool_name}'")
        try:
            return provider.handle_tool_call(tool_name, args, **kwargs)
        except Exception as e:
            logger.error("Memory provider '%s' handle_tool_call(%s) failed: %s", provider.name, tool_name, e)
            return tool_error(f"Memory tool '{tool_name}' failed: {e}")

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._each_provider("on_turn_start failed", lambda p: p.on_turn_start(turn_number, message, **kwargs))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._each_provider("on_session_end failed", lambda p: p.on_session_end(messages), level=logging.WARNING,
                            exc_info=True)

    def commit_session_boundary_async(self, messages: List[Dict[str, Any]], *, new_session_id: str,
                                      parent_session_id: str = "", reason: str = "new_session") -> None:
        """Queue old-session extraction + provider rebinding as ONE serialized task.

        ``on_session_end`` (LLM-bound, seconds) must run strictly BEFORE ``on_session_switch`` rebinds
        provider state; an ad-hoc thread raced the inline switch and misattributed transcripts.

        Running extraction inline blocked the /new command for the whole LLM round-trip (#16454); running it
        on an ad-hoc thread raced the inline switch — providers key off internal state, so a late
        ``on_session_end`` ran against post-switch bindings (transcript misattributed to the new session id,
        double-ingest of the old turn buffer, new-session buffers cleared).
        Submitting BOTH hooks as one task on the manager's single background worker gives both properties at
        a single chokepoint: the caller returns immediately, and the worker's FIFO order serializes
        end→switch against every other provider write (per-turn ``sync_all``, prefetches), which already
        share the same worker. If the executor is unavailable, ``_submit_background`` degrades to inline
        execution — the pre-#16454 synchronous behavior, slow but correct.
        """
        if not self._providers:
            return
        snapshot = list(messages or [])

        def _run() -> None:  # both hooks already guard per-provider
            try:
                self.on_session_end(snapshot)
            except Exception as e:  # pragma: no cover
                logger.warning("Session-boundary extraction failed: %s", e)
            try:
                self.on_session_switch(new_session_id, parent_session_id=parent_session_id, reset=True, reason=reason)
            except Exception as e:  # pragma: no cover
                logger.warning("Session-boundary switch failed: %s", e)

        self._submit_background(_run)

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        """Notify providers that ``AIAgent.session_id`` rotated without teardown
        (``/resume``, ``/branch``, ``/reset``, ``/new``, compression). ``rewound=True``
        (``/undo``): same id, truncated transcript."""
        if not new_session_id:
            return
        if rewound:  # forward only when set so it never pollutes providers' **kwargs
            kwargs["rewound"] = True
        self._each_provider(
            "on_session_switch failed",
            lambda p: p.on_session_switch(new_session_id, parent_session_id=parent_session_id, reset=reset, **kwargs),
        )

    @staticmethod
    def _checkpoint_api_version(provider: MemoryProvider) -> Optional[int]:
        """Provider's advertised pre-compress checkpoint API version; None if unparseable."""
        try:
            return int(getattr(provider, "pre_compress_checkpoint_api_version", _LEGACY_PRE_COMPRESS_API_VERSION))
        except (TypeError, ValueError):
            return None

    def supports_pre_compress_checkpoint(self, api_version: int = PRE_COMPRESS_CHECKPOINT_API_VERSION) -> bool:
        """Return whether an active provider guarantees checkpoint API support."""
        versions = (self._checkpoint_api_version(p) for p in self._providers)
        return any(v is not None and v >= api_version for v in versions)

    def on_pre_compress(self, messages: List[Dict[str, Any]], *,
                        evidence_messages: Optional[List[Dict[str, Any]]] = None, require_checkpoint: bool = False,
                        checkpoint_api_version: int = PRE_COMPRESS_CHECKPOINT_API_VERSION) -> str:
        """Notify providers before compression; return their combined summary-prompt text.

        ``messages`` is the raw v1 transcript; ``evidence_messages`` is the host-normalized list handed
        only to checkpoint (v2+) providers. With ``require_checkpoint`` at least one checkpoint provider
        must succeed — its exception propagates so the caller keeps the uncompressed transcript.
        """
        parts = []
        checkpoint_succeeded = False
        for provider in self._providers:
            version = self._checkpoint_api_version(provider)
            if version is None:
                version = _LEGACY_PRE_COMPRESS_API_VERSION
            is_checkpoint_provider = version >= checkpoint_api_version
            use_evidence = is_checkpoint_provider and evidence_messages is not None
            provider_messages = evidence_messages if use_evidence else messages
            kwargs: Dict[str, Any] = {}
            # v1 providers and bare-shape v2 providers never see the signal.
            if is_checkpoint_provider and _accepts_require_checkpoint(provider.on_pre_compress):
                kwargs["require_checkpoint"] = require_checkpoint
            try:
                result = provider.on_pre_compress(provider_messages, **kwargs)
                if result and result.strip():
                    parts.append(result)
                checkpoint_succeeded = checkpoint_succeeded or is_checkpoint_provider
            except Exception as e:
                logger.debug("Memory provider '%s' on_pre_compress failed: %s", provider.name, e)
                if require_checkpoint and is_checkpoint_provider:
                    raise
        if require_checkpoint and not checkpoint_succeeded:
            raise RuntimeError(
                f"No active memory provider completed pre-compress checkpoint API v{checkpoint_api_version}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _provider_memory_write_metadata_mode(provider: MemoryProvider) -> str:
        """How to pass metadata to ``on_memory_write``: "keyword", "positional", or "legacy" (none)."""
        params = _signature_params(provider.on_memory_write)
        if params is None or _has_var_kwargs(params) or "metadata" in params:
            return "keyword"
        accepted = sum(p.kind is not inspect.Parameter.VAR_POSITIONAL for p in params.values())
        return "positional" if accepted >= 4 else "legacy"

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Notify external providers when the built-in memory tool writes (skips builtin, the source)."""

        def _notify(provider: MemoryProvider) -> None:
            mode = self._provider_memory_write_metadata_mode(provider)
            if mode == "legacy":
                provider.on_memory_write(action, target, content)
            elif mode == "positional":
                provider.on_memory_write(action, target, content, dict(metadata or {}))
            else:
                provider.on_memory_write(action, target, content, metadata=dict(metadata or {}))

        external = [p for p in self._providers if p.name != "builtin"]
        self._each_provider("on_memory_write failed", _notify, providers=external)

    # Actions mirrored to external providers; non-mutating results (errors, staged) are
    # filtered by ``notify_memory_tool_write`` first.
    _MIRRORED_MEMORY_ACTIONS = {"add", "replace", "remove"}

    @staticmethod
    def _memory_tool_result_succeeded(result: Any) -> bool:
        """True only when the built-in memory tool actually committed a write. Fails closed (non-JSON,
        non-dict, missing ``success``, staged for approval) so providers never mirror a write that did not land."""
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                return False
        return isinstance(result, dict) and result.get("success") is True and result.get("staged") is not True

    def notify_memory_tool_write(self, tool_result: Any, tool_args: Dict[str, Any], *,
                                 build_metadata: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
        """Mirror a built-in memory tool call to external providers.

        Gates on a committed write, expands single-op and batched ``operations`` shapes, keeps only
        mutating actions, and forwards ``old_text`` plus provenance from ``build_metadata`` (the loop
        knows session/task/tool-call identity; we do not).
        """
        if not self._memory_tool_result_succeeded(tool_result):
            return
        target = str(tool_args.get("target") or "memory")
        operations = tool_args.get("operations")
        for op in operations if isinstance(operations, list) and operations else [tool_args]:
            action = str(op.get("action") or "") if isinstance(op, dict) else ""
            if action not in self._MIRRORED_MEMORY_ACTIONS:
                continue
            try:
                metadata = dict(build_metadata() if build_metadata else {})
                old_text = op.get("old_text")
                if old_text:
                    metadata["old_text"] = str(old_text)
                self.on_memory_write(action, target, str(op.get("content") or ""), metadata=metadata)
            except Exception as e:
                logger.debug("notify_memory_tool_write failed for op %s: %s", action, e)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        self._each_provider(
            "on_delegation failed",
            lambda p: p.on_delegation(task, result, child_session_id=child_session_id, **kwargs),
        )

    def shutdown_all(self) -> None:
        """Drain the background executor (bounded), then shut providers down in reverse order."""
        self._drain_sync_executor()
        self._each_provider("shutdown failed", lambda p: p.shutdown(), level=logging.WARNING,
                            providers=self._providers[::-1])

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
                "abandoned_writes": 0, "abandoned_prefetches": 0,
                "active_tasks": sum(not future.done() for future in tracked),
            }
        if executor is None:
            return

        # shutdown(wait=False) closes submission without touching the FIFO; waiting on the
        # tracked futures lets the worker run every queued task in order up to the deadline.
        executor.shutdown(wait=False, cancel_futures=False)
        _, pending = wait(tuple(tracked), timeout=_SYNC_DRAIN_TIMEOUT_S)
        cancelled = [tracked[future] for future in pending if future.cancel()]
        active_tasks = len(pending) - len(cancelled)
        abandoned_prefetches = cancelled.count("prefetch")
        abandoned_writes = len(cancelled) - abandoned_prefetches
        with self._sync_executor_lock:
            self._shutdown_drain_state.update(
                status="timed_out" if pending else "drained", abandoned_writes=abandoned_writes,
                abandoned_prefetches=abandoned_prefetches, active_tasks=active_tasks,
            )
        if not pending:
            return
        logger.warning(
            "Memory shutdown drain timed out after %.2fs; abandoning %d queued "
            "memory write(s) and %d queued prefetch(es); %d active task(s) remain detached",
            _SYNC_DRAIN_TIMEOUT_S, abandoned_writes, abandoned_prefetches, active_tasks,
        )

    def initialize_all(self, session_id: str, **kwargs) -> None:
        """Initialize all providers, injecting ``hermes_home`` so they resolve profile-scoped paths."""
        if "hermes_home" not in kwargs:
            from hermes_constants import get_hermes_home
            kwargs["hermes_home"] = str(get_hermes_home())
        self._each_provider("initialize failed", lambda p: p.initialize(session_id=session_id, **kwargs),
                            level=logging.WARNING)
