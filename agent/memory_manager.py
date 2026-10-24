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
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from agent.redact import redact_for_egress

from agent.memory_caller_scope import bound_scope, caller_scope, claim_instance, in_owner_home, refused_tool_message
# Moved to agent.memory_context / agent.memory_manager_tools; still importable here because agent.memory_manager is a public
# module for plugins (hermes_cli/plugin_validate_private_imports.py).
from agent.memory_context import StreamingContextScrubber, build_memory_context_block, sanitize_context
from agent.memory_manager_tools import (inject_memory_provider_tools, memory_provider_tools_enabled,
                                        memory_provider_tools_exposed, normalize_tool_schema)
from agent.memory_provider import MemoryProvider, PRE_COMPRESS_CHECKPOINT_API_VERSION, ctx_bound, spawn_context_thread
from agent.skill_commands import extract_user_instruction_from_skill_message
from hermes_constants import hermes_home_key
from tools.hook_output_spill import get_spill_config, spill_if_oversized
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Providers that predate the checkpoint-API attribute are on the best-effort v1 contract.
_LEGACY_PRE_COMPRESS_API_VERSION = 1

# shutdown_all() drain bound; workers are daemon threads so a wedged provider never
# blocks interpreter exit.
_SYNC_DRAIN_TIMEOUT_S = 5.0
_EXTERNAL_PREFETCH_TIMEOUT_S = 8.0
# A background provider call (sync_turn / queue_prefetch) running longer than this is treated as
# stuck. Well above the shutdown drain bound and the prefetch timeout so a slow-but-healthy backend
# is never counted.
_BACKGROUND_CALL_STALL_S = 30.0
# Admission for external provider calls per (home, provider); see _StuckCallGate. Python cannot stop a
# thread, so every stuck call pins one until it returns. New calls stop once this many calls that started
# after the provider's last successful return are overdue; older orphans belong to an outage it has
# since recovered from.
_MAX_STUCK_CALLS = 4
# While the gate is closed, one call per interval goes through, so a recovered backend proves itself
# without waiting for an orphan to return.
_PROBE_INTERVAL_S = 60.0
# Hard cap on calls in flight per (home, provider), orphans included: failed probes pin one more thread
# each, so admission (probes too) stops here and a backend that stays wedged pins a bounded number.
_MAX_INFLIGHT_CALLS = 16


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


# -- Provider egress scrubbing -----------------------------------------------

def _scrub(text: str) -> str:
    """The shared egress scrub. Like chat-platform and cron delivery it holds even when
    ``security.redact_secrets`` is off (that setting governs local logs, not what leaves the agent)
    and fails closed."""
    return redact_for_egress(text)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-./+]{16,}")


def _redact_for_provider(*values: Any) -> Any:
    """Copies of ``values`` with every string scrubbed before it reaches a memory provider.

    Providers archive whatever they are handed (#115104), so each manager fan-out (turn sync, recall
    queries, session-end and pre-compress transcripts, memory-tool mirrors, delegation results, provider
    tool args) goes through ``_scrub``. Walks dicts/lists/tuples so tool-call arguments and multimodal text
    parts are covered; ``data:`` URLs (inline media) pass through. A value the scrub masked anywhere in the
    hand-off is then masked everywhere in it, so a key detected in a tool result is also caught where the
    model repeats it bare in prose. Never mutates the caller's objects: unchanged values come back as-is,
    changed containers as shallow copies. One value in → one value out; several → a tuple.
    """
    learned: set = set()

    def walk(value: Any, fix) -> Any:
        if isinstance(value, str):
            return value if not value or value.startswith("data:") else fix(value)
        if isinstance(value, dict):
            out = {k: walk(v, fix) for k, v in value.items()}
            return value if all(out[k] is value[k] for k in value) else out
        if isinstance(value, (list, tuple)):
            items = [walk(v, fix) for v in value]
            if all(new is old for new, old in zip(items, value)):
                return value
            return tuple(items) if isinstance(value, tuple) else items
        return value

    def scrub(text: str) -> str:
        out = _scrub(text)
        if out != text:
            learned.update(set(_TOKEN_RE.findall(text)) - set(_TOKEN_RE.findall(out)))
        return out

    scrubbed = [walk(v, scrub) for v in values]
    if learned:
        pattern = re.compile("|".join(re.escape(t) for t in sorted(learned, key=len, reverse=True)))
        scrubbed = [walk(v, lambda t: pattern.sub("[redacted]", t) if pattern.search(t) else t) for v in scrubbed]
    return scrubbed[0] if len(scrubbed) == 1 else tuple(scrubbed)


def _redact_messages_for_egress(messages: Optional[list[Any]]) -> Optional[list[Any]]:
    return None if messages is None else list(_redact_for_provider(messages))


class _StuckCallGate:
    """Process-wide admission for one kind of external provider call, keyed by (Hermes home, provider).

    A new MemoryManager is built per agent, so a per-instance view would let every fresh manager start
    one more call against a hung backend and pin one more thread. Shared, the gate closes once
    _MAX_STUCK_CALLS calls that started after the key's last successful return are past their deadline,
    admits one half-open probe per _PROBE_INTERVAL_S while closed, and never lets more than
    _MAX_INFLIGHT_CALLS be in flight. Calls not yet overdue are not counted, so healthy concurrent
    sessions run side by side. Each call removes only its own entry.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: dict[tuple[str, str], dict[object, tuple[float, float]]] = {}  # token -> (start, deadline)
        self.last_ok: dict[tuple[str, str], float] = {}
        self.probe_at: dict[tuple[str, str], float] = {}  # interval start: gate close, then each probe
        self.skips: dict[tuple[str, str], int] = {}

    def enter(self, key: tuple[str, str], token: object, timeout: float) -> Optional[tuple[int, int, int]]:
        """Register ``token`` and return None, or ``(stuck, in_flight, skips)`` when the call is skipped."""
        now = time.monotonic()
        with self.lock:
            calls = self.calls.setdefault(key, {})
            last_ok = self.last_ok.get(key, float("-inf"))
            stuck = sum(start > last_ok and now > deadline for start, deadline in calls.values())
            if len(calls) >= _MAX_INFLIGHT_CALLS:
                admit = False
            elif stuck < _MAX_STUCK_CALLS:
                admit = True
                self.probe_at.pop(key, None)
                self.skips.pop(key, None)  # gate open: the next outage warns on its first skip
            else:
                admit = now - self.probe_at.setdefault(key, now) >= _PROBE_INTERVAL_S
                if admit:
                    self.probe_at[key] = now
            if not admit:
                skips = self.skips[key] = self.skips.get(key, 0) + 1
                return stuck, len(calls), skips
            calls[token] = (now, now + timeout)
            return None

    def leave(self, key: tuple[str, str], token: object, *, ok: bool) -> None:
        with self.lock:
            calls = self.calls.get(key)
            if calls is None or calls.pop(token, None) is None:
                return
            if calls:
                if ok:
                    self.last_ok[key] = time.monotonic()
                return
            for state in (self.calls, self.last_ok, self.probe_at, self.skips):
                state.pop(key, None)


class MemoryManager:
    """Builtin provider (always first) plus at most one external provider.

    Failures in one provider never block the other: every fan-out hook logs and
    swallows per-provider exceptions.
    """

    # External prefetch threads (#134392) and background sync_turn / queue_prefetch calls. The home is
    # part of the key because one gateway process can serve several profiles, each with its own config
    # and credentials for a same-named provider: one profile's hung backend must not silence another's.
    _PREFETCH_GATE = _StuckCallGate()
    _BACKGROUND_GATE = _StuckCallGate()

    def __init__(self, *, external_prefetch_timeout: Optional[float] = None) -> None:
        self._providers: list[MemoryProvider] = []
        self._tool_to_provider: dict[str, MemoryProvider] = {}
        self._external_prefetch_spill_config: Optional[dict[str, Any]] = None
        self._has_external: bool = False
        timeout = external_prefetch_timeout
        timeout = _EXTERNAL_PREFETCH_TIMEOUT_S if timeout is None else float(timeout)
        if timeout <= 0:
            raise ValueError("external_prefetch_timeout must be positive")
        self._external_prefetch_timeout = timeout
        # Captured at build time, inside the profile scope the agent is built for.
        self._home_key = hermes_home_key()
        # Single-worker background executor for end-of-turn sync/prefetch, created lazily so
        # the builtin-only path spawns no threads; one worker serializes a provider's writes.
        self._sync_executor: Optional[ThreadPoolExecutor] = None
        self._sync_executor_lock = threading.Lock()
        # Futures by durability class ("write" / "prefetch") so shutdown can drain FIFO
        # within a bound, then report exactly what it abandoned.
        self._background_futures: dict[Future, str] = {}
        self._shutting_down = False
        self._teardown_claim: Optional[str] = None  # see claim_teardown()
        self._shutdown_drain_state: dict[str, Any] = {
            "status": "not_started", "abandoned_writes": 0, "abandoned_prefetches": 0, "active_tasks": 0,
        }
        # The profile ``initialize_all`` bound the providers to; tool calls must come from it (#140058).
        self._bound_scope: Optional[dict[str, Any]] = None
        self._tool_call_keywords_cache: dict[int, Optional[frozenset[str]]] = {}

    def _each_provider(self, label: str, call: Callable[[MemoryProvider], Any], *, level: int = logging.DEBUG,
                       providers: Optional[list[MemoryProvider]] = None, exc_info: bool = False) -> list[Any]:
        """Call ``call(provider)`` per provider, logging+swallowing failures; returns successes in order.
        ``label`` completes the log line ``Memory provider '<name>' <label>: <exc>``."""
        results: list[Any] = []
        for provider in self._providers if providers is None else providers:
            try:
                results.append(call(provider))
            except Exception as e:
                logger.log(level, "Memory provider '%s' %s: %s", provider.name, label, e, exc_info=exc_info)
        return results

    def _owner_allows(self, hook: str) -> bool:
        """Whether a lifecycle hook may reach the providers: only from the profile they were initialized
        for (#140058; see ``agent/memory_caller_scope.py``). Nothing to reach, nothing to refuse."""
        return not self._providers or caller_scope(self._bound_scope, f"hook {hook}") is not None

    def _tool_call_keywords(self, provider: MemoryProvider) -> Optional[frozenset[str]]:
        """The keywords ``provider.handle_tool_call`` declares; None when it takes any (``**kwargs``, or
        a signature that can't be read). Worked out once per provider, not on every tool call."""
        if id(provider) not in self._tool_call_keywords_cache:
            params = _signature_params(provider.handle_tool_call)
            self._tool_call_keywords_cache[id(provider)] = (
                None if params is None or _has_var_kwargs(params) else frozenset(params))
        return self._tool_call_keywords_cache[id(provider)]

    def _drop_provider(self, provider: MemoryProvider) -> None:
        self._tool_call_keywords_cache.pop(id(provider), None)
        self._providers = [p for p in self._providers if p is not provider]
        self._tool_to_provider = {t: p for t, p in self._tool_to_provider.items() if p is not provider}
        self._has_external = any(p.name != "builtin" for p in self._providers)

    def add_provider(self, provider: MemoryProvider) -> None:
        """Register a provider; builtin always accepted, only ONE external allowed."""
        if provider.name != "builtin" and self._has_external:
            existing = next((p.name for p in self._providers if p.name != "builtin"), "unknown")
            logger.warning(
                "Rejected memory provider '%s' — external provider '%s' is "
                "already registered. Only one external memory provider is "
                "allowed at a time. Configure which one via memory.provider "
                "in config.yaml.", provider.name, existing,
            )
            return

        # Load schemas BEFORE mutating any manager state: a provider whose schema
        # load raises must leave `_providers` / `_has_external` untouched, otherwise
        # it blocks every later external provider in this process (#9948).
        schemas = list(provider.get_tool_schemas())

        if provider.name != "builtin":
            from hermes_cli.config import load_config

            # Recall is provider-ranked: keep it whole unless this profile opts in.
            # Snapshot both gates at registration, never re-read them mid-conversation.
            memory = load_config().get("memory", {})
            spill_config = get_spill_config()
            if not (isinstance(memory, dict) and memory.get("prefetch_spill_enabled", False) is True):
                # Off by default, but keep a ceiling so a runaway provider can't overflow the context.
                spill_config["max_chars"] = max(spill_config["max_chars"] * 10, 100_000)
            self._external_prefetch_spill_config = spill_config
            self._has_external = True

        self._providers.append(provider)

        # Core tool names are reserved: built-ins always win at agent init, so a shadowing
        # provider tool would linger in ``_tool_to_provider`` and hijack dispatch.
        # ``clarify``, ``delegate_task``). Reject it here, at the door, so it never enters the routing table
        # at all — matching the built-ins-always-win invariant used by the TTS/browser/search provider
        # registries. See #40466.
        from toolsets import _HERMES_CORE_TOOLS

        for raw_schema in schemas:
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

        logger.info("Memory provider '%s' registered (%d tools)", provider.name, len(schemas))

    @property
    def providers(self) -> list[MemoryProvider]:
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
        if not clean_query or not self._owner_allows("prefetch"):
            return ""
        clean_query = _redact_for_provider(clean_query)
        parts = self._each_provider(
            "prefetch failed (non-fatal)", lambda p: self._prefetch_provider(p, clean_query, session_id=session_id),
        )
        return "\n\n".join(p for p in parts if p and p.strip())

    def _prefetch_provider(self, provider: MemoryProvider, query: str, *, session_id: str = "") -> str:
        """Run one provider's prefetch; external providers are bounded by a timeout. A stuck external
        call keeps running on its daemon thread; _PREFETCH_GATE decides whether later turns call again."""
        if provider.name == "builtin":
            return provider.prefetch(query, session_id=session_id)

        result_box: dict[str, Any] = {}

        key = self._registry_key(provider)

        def _run() -> None:
            try:
                result_box["value"] = provider.prefetch(query, session_id=session_id) or ""
            except Exception as exc:  # pragma: no cover - re-raised by caller
                result_box["error"] = exc
            finally:
                self._PREFETCH_GATE.leave(key, thread, ok="error" not in result_box)

        thread = spawn_context_thread(_run, name=f"memory-prefetch-{provider.name}")
        skipped = self._PREFETCH_GATE.enter(key, thread, self._external_prefetch_timeout)
        if skipped:
            logger.debug("Memory provider '%s' prefetch skipped (%d stuck, %d in flight)",
                         provider.name, *skipped[:2])
            return ""
        try:
            thread.start()
        except RuntimeError:
            # Never started, so its finally never runs: drop the entry or it would count forever.
            self._PREFETCH_GATE.leave(key, thread, ok=False)
            raise

        thread.join(self._external_prefetch_timeout)
        if thread.is_alive():
            logger.warning(
                "Memory provider '%s' prefetch timed out after %.1fs; the call keeps running in the "
                "background (later turns may skip it while the provider stays stuck)", provider.name,
                self._external_prefetch_timeout,
            )
            return ""

        if "error" in result_box:
            raise result_box["error"]
        result = result_box.get("value", "")
        if result and result.strip():
            # Opt-in spill limits recall replayed with the user turn's api_content;
            # the registration snapshot leaves provider-ranked results intact by default.
            result = spill_if_oversized(
                result, session_id=session_id, source=f"{provider.name} memory prefetch",
                config=self._external_prefetch_spill_config,
            )
        return result

    def describe_recall(self) -> str:
        """Deterministic recall indicator line (e.g. ``"🧠 Provider — recalled 3 memories"``); ``""`` if none.
        Call right after :meth:`prefetch_all` so the user SEES memory was used even if the model is silent."""
        segments: list[str] = []
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
        if not clean_query or not self._owner_allows("queue_prefetch"):
            return
        clean_query = _redact_for_provider(clean_query)
        self._submit_background(lambda: self._each_provider(
            "queue_prefetch failed (non-fatal)",
            lambda p: self._call_unless_stalled(p, lambda: p.queue_prefetch(clean_query, session_id=session_id)),
            providers=providers,
        ), kind="prefetch")

    @staticmethod
    def _provider_sync_accepts(provider: MemoryProvider, keyword: str) -> bool:
        """Whether ``sync_turn`` accepts ``keyword`` (uninspectable → assume yes)."""
        params = _signature_params(provider.sync_turn)
        return params is None or _has_var_kwargs(params) or keyword in params

    def sync_all(self, user_content: str, assistant_content: str, *, session_id: str = "",
                 messages: Optional[list[dict[str, Any]]] = None,
                 turn_author: Optional[dict[str, Any]] = None) -> None:
        """Sync a completed turn to all providers on the background worker.

        Never inline: a provider's ``sync_turn`` may block for minutes, which kept ``run_conversation``
        open after the user saw the response. The single worker also serializes writes (turn N before N+1).
        ``turn_author`` reaches only providers whose ``sync_turn`` accepts it.

        Everything forwarded is provider egress: turn strings AND the ``messages`` transcript slice
        (tool outputs included) go through ``_redact_for_provider`` so secrets are never archived
        verbatim in a provider's store (#115104).
        """
        providers = list(self._providers)
        clean_user_content = self._strip_skill_scaffolding(user_content) if providers else None
        if not clean_user_content or not self._owner_allows("sync_turn"):
            return
        clean_user_content, assistant_content, redacted_messages, snapshot_user_content = _redact_for_provider(
            clean_user_content, assistant_content, messages, user_content)
        # Capture authors before enqueue; a later turn may reuse the caller dict.
        optional_kwargs = {"messages": redacted_messages, "turn_author": deepcopy(turn_author)}

        snapshot_providers = [p for p in providers
                              if getattr(p, "sync_turn_snapshot_version", 0) == 1]
        snapshot = None
        if snapshot_providers:
            from agent.memory_sync_snapshot import snapshot_completed_turn
            snapshot = snapshot_completed_turn(
                redacted_messages, session_id=session_id, user_content=snapshot_user_content,
                assistant_content=assistant_content)

        def _sync(provider: MemoryProvider) -> None:
            kwargs: dict[str, Any] = {"session_id": session_id}
            for keyword, value in optional_kwargs.items():
                if value is None or not self._provider_sync_accepts(provider, keyword):
                    continue
                if keyword == "messages" and provider in snapshot_providers:
                    kwargs[keyword] = snapshot
                else:
                    # Each provider owns its copy; mutations cannot change other deliveries.
                    kwargs[keyword] = deepcopy(value) if keyword == "turn_author" else value
            self._call_unless_stalled(
                provider, lambda: provider.sync_turn(clean_user_content, assistant_content, **kwargs))

        self._submit_background(
            lambda: self._each_provider("sync_turn failed", _sync, level=logging.WARNING, providers=providers)
        )

    def _submit_background(self, fn, *, kind: str = "write") -> None:
        """Queue ``fn`` on the serialized worker (created lazily; None once shutting down) and track its
        durability class. Runs under the caller's contextvars (``ctx_bound``). If the executor is
        unavailable outside shutdown, run inline — the historical fail-safe."""
        fn = ctx_bound(in_owner_home(self._bound_scope, fn))
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

    def _call_unless_stalled(self, provider: MemoryProvider, call: Callable[[], Any]) -> None:
        """Run a background provider call unless _BACKGROUND_GATE skips it (the provider has stuck calls);
        skipping costs that turn's write/prefetch but stops a wedged backend from pinning one more
        thread per fresh manager. The builtin provider is never gated."""
        if provider.name == "builtin":
            call()
            return
        key = self._registry_key(provider)
        token = object()
        skipped = self._BACKGROUND_GATE.enter(key, token, _BACKGROUND_CALL_STALL_S)
        if skipped:
            stuck, in_flight, skips = skipped
            if skips == 1 or skips % 100 == 0:
                logger.warning(
                    "Memory provider '%s' has %d background calls stuck for over %.0fs (%d in flight); "
                    "skipping its sync/prefetch until it recovers (%d skipped)", provider.name, stuck,
                    _BACKGROUND_CALL_STALL_S, in_flight, skips,
                )
            return
        ok = False
        try:
            call()
            ok = True
        finally:
            self._BACKGROUND_GATE.leave(key, token, ok=ok)

    def _registry_key(self, provider: MemoryProvider) -> tuple[str, str]:
        """Process-wide registry key: the home the providers were bound to (``initialize_all``, which may
        name another profile than the one this manager was built in), else the build-time home."""
        return ((self._bound_scope or {}).get("home_key") or self._home_key, provider.name)

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

    def get_all_tool_schemas(self) -> list[dict[str, Any]]:
        """Collect deduplicated tool schemas from all providers; reserved core tool names are
        skipped because :meth:`add_provider` refuses to route them."""
        from toolsets import _HERMES_CORE_TOOLS

        schemas: list[dict[str, Any]] = []
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

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        """Route a tool call to its provider; returns a JSON string (tool_error on failure)."""
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return tool_error(f"No memory provider handles tool '{tool_name}'")
        scope = caller_scope(self._bound_scope, f"tool {tool_name}")
        if scope is None:
            return tool_error(refused_tool_message(tool_name))
        kwargs = {**kwargs, **scope}
        # A provider written against ``handle_tool_call(tool_name, args)`` gets only what it declares.
        accepted = self._tool_call_keywords(provider)
        if accepted is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        args = _redact_for_provider(args)
        from hermes_cli.observability.shared_metrics_loop import record_provider_memory_call
        try:
            result = provider.handle_tool_call(tool_name, args, **kwargs)
        except Exception as e:
            logger.error("Memory provider '%s' handle_tool_call(%s) failed: %s", provider.name, tool_name, e)
            record_provider_memory_call(provider.name, tool_name, args, raised=True)
            return tool_error(f"Memory tool '{tool_name}' failed: {e}")
        record_provider_memory_call(provider.name, tool_name, args, result)
        return result

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if not self._owner_allows("on_turn_start"):
            return
        message = _redact_for_provider(message)

        def _tick(p: MemoryProvider) -> None:
            # A provider written before the author kwargs declares (turn_number, message) only; it still gets its tick.
            params = _signature_params(p.on_turn_start)
            accepted = kwargs if params is None or _has_var_kwargs(params) else {k: v for k, v in kwargs.items() if k in params}
            p.on_turn_start(turn_number, message, **accepted)

        self._each_provider("on_turn_start failed", _tick)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if not self._owner_allows("on_session_end"):
            return
        messages = _redact_messages_for_egress(messages or [])
        self._each_provider("on_session_end failed", lambda p: p.on_session_end(messages), level=logging.WARNING,
                            exc_info=True)

    def commit_session_boundary_async(self, messages: list[dict[str, Any]], *, new_session_id: str,
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
        if not self._providers or not self._owner_allows("session_boundary"):
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
        if not new_session_id or not self._owner_allows("on_session_switch"):
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

    def on_pre_compress(self, messages: list[dict[str, Any]], *,
                        evidence_messages: Optional[list[dict[str, Any]]] = None, require_checkpoint: bool = False,
                        checkpoint_api_version: int = PRE_COMPRESS_CHECKPOINT_API_VERSION) -> str:
        """Notify providers before compression; return their combined summary-prompt text.

        ``messages`` is the raw v1 transcript; ``evidence_messages`` is the host-normalized list handed
        only to checkpoint (v2+) providers. With ``require_checkpoint`` at least one checkpoint provider
        must succeed — its exception propagates so the caller keeps the uncompressed transcript.
        """
        if not self._owner_allows("on_pre_compress"):
            if require_checkpoint:  # the caller keeps the uncompressed transcript
                raise RuntimeError("Memory pre-compress checkpoint refused; see `hermes logs errors`")
            return ""
        parts = []
        checkpoint_succeeded = False
        messages, evidence_messages = _redact_for_provider(messages or [], evidence_messages)
        for provider in self._providers:
            version = self._checkpoint_api_version(provider)
            if version is None:
                version = _LEGACY_PRE_COMPRESS_API_VERSION
            is_checkpoint_provider = version >= checkpoint_api_version
            use_evidence = is_checkpoint_provider and evidence_messages is not None
            provider_messages = evidence_messages if use_evidence else messages
            kwargs: dict[str, Any] = {}
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
                        metadata: Optional[dict[str, Any]] = None) -> None:
        """Notify external providers when the built-in memory tool writes (skips builtin, the source)."""
        if not self._owner_allows("on_memory_write"):
            return
        content, metadata = _redact_for_provider(content, dict(metadata or {}))

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
    _MIRRORED_MEMORY_ACTIONS = {"add", "replace", "patch", "remove"}

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

    def notify_memory_tool_write(self, tool_result: Any, tool_args: dict[str, Any], *,
                                 build_metadata: Optional[Callable[[], dict[str, Any]]] = None) -> None:
        """Mirror a built-in memory tool call to external providers.

        Gates on a committed write, expands single-op and batched ``operations`` shapes, keeps only
        mutating actions, and forwards ``old_text`` plus provenance from ``build_metadata``.
        ``previous_content`` comes only from the committed store result, never the search
        argument: a partial provider registry cannot safely resolve that argument itself.
        """
        if not self._memory_tool_result_succeeded(tool_result):
            return
        result = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
        target = str(tool_args.get("target") or "memory")
        operations = tool_args.get("operations")
        batched = isinstance(operations, list) and bool(operations)
        for index, op in enumerate(operations if batched else [tool_args], start=1):
            action = str(op.get("action") or "") if isinstance(op, dict) else ""
            if action not in self._MIRRORED_MEMORY_ACTIONS:
                continue
            try:
                metadata = dict(build_metadata() if build_metadata else {})
                metadata.pop("previous_content", None)
                old_text = op.get("old_text")
                if old_text:
                    metadata["old_text"] = str(old_text)
                field = {"replace": "replaced", "patch": "patched", "remove": "removed"}.get(action)
                if field:
                    if batched:
                        entries = result.get(f"{field}_entries", {})
                        previous = entries.get(str(index), entries.get(index)) if isinstance(entries, dict) else None
                    else:
                        previous = result.get(f"{field}_entry")
                        if field == "patched" and previous is None:
                            previous = result.get("patched_entry_before")
                    if isinstance(previous, str) and previous:
                        metadata["previous_content"] = previous
                mirrored_action, content = action, str(op.get("content") or op.get("new_text") or "")
                if action == "patch":
                    # To a provider a patch replaced the whole entry with its edited form; mirror it as
                    # that replace, with the entry the store reports, so providers that know
                    # add/replace/remove handle it unchanged.
                    if batched:
                        after_entries = result.get("patched_entries_after") or {}
                        after = after_entries.get(str(index), after_entries.get(index))
                    else:
                        after = result.get("patched_entry_after")
                    mirrored_action, content = "replace", str(after or "")
                self.on_memory_write(mirrored_action, target, content, metadata=metadata)
            except Exception as e:
                logger.debug("notify_memory_tool_write failed for op %s: %s", action, e)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if not self._owner_allows("on_delegation"):
            return
        task, result = _redact_for_provider(task, result)
        self._each_provider(
            "on_delegation failed",
            lambda p: p.on_delegation(task, result, child_session_id=child_session_id, **kwargs),
        )

    def claim_teardown(self, *, session_end: bool) -> bool:
        """Claim extraction + ``shutdown_all()`` for end-of-session shutdown or soft release. The first
        claimer wins and the loser skips, so the two never overlap and providers (no idempotency
        contract) are never shut down twice. Session-end callers may re-claim (the gateway claims before
        its flush, then ``shutdown_memory_provider`` claims again); a soft release is once-only and also
        loses to a manager another owner is already draining."""
        with self._sync_executor_lock:
            if self._teardown_claim is None and (session_end or not self._shutting_down):
                self._teardown_claim = "session_end" if session_end else "soft_release"
                return True
            return session_end and self._teardown_claim == "session_end"

    def release_soft(self) -> None:
        """Soft release (cache eviction, CLI agent rebuild): stop provider writer threads, which otherwise leak
        one per evicted agent, WITHOUT on_session_end(): sweeps commit first and a pending session end owns
        both. Flush first, since shutdown_all()'s short drain would cancel writes the old worker finished."""
        if self.claim_teardown(session_end=False):
            self.flush_pending(timeout=10)
            self.shutdown_all()

    def shutdown_all(self) -> None:
        """Drain the background executor (bounded), then shut providers down in reverse order."""
        self._drain_sync_executor()
        self._each_provider("shutdown failed", lambda p: p.shutdown(), level=logging.WARNING,
                            providers=self._providers[::-1])

    @property
    def shutdown_drain_state(self) -> dict[str, Any]:
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

    def initialize_for_agent(self, say: Optional[Callable[[str], None]] = None, **kwargs) -> bool:
        """``initialize_all`` for an agent build; True while an external provider is still served. One
        refused as another profile's shared instance is dropped, and ``say`` (the agent's startup
        warning sink) tells the user this profile now runs without it."""
        before = [p for p in self._providers if p.name != "builtin"]
        self.initialize_all(**kwargs)
        for provider in (p for p in before if not any(p is kept for kept in self._providers)):
            message = (f"⚠ Memory provider {provider.name!r} is disabled for this profile: it handed out the "
                       "provider instance another profile already uses. Built-in memory still works; the "
                       "operator can see why with `hermes logs errors`.")
            (say or logger.warning)(message)
        return self._has_external

    def initialize_all(self, session_id: str, **kwargs) -> None:
        """Initialize all providers, injecting ``hermes_home`` so they resolve profile-scoped paths."""
        if "hermes_home" not in kwargs:
            from hermes_constants import get_hermes_home
            kwargs["hermes_home"] = str(get_hermes_home())
        self._bound_scope = bound_scope(kwargs["hermes_home"], kwargs.get("agent_identity"))
        # One instance, one profile: never rebind an instance another profile's manager bound (#140058).
        for provider in [p for p in self._providers if p.name != "builtin"]:
            if not claim_instance(provider, kwargs["hermes_home"]):
                self._drop_provider(provider)
        self._each_provider("initialize failed", lambda p: p.initialize(session_id=session_id, **kwargs),
                            level=logging.WARNING)
