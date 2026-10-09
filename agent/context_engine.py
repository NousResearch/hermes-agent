"""Abstract base class for pluggable context engines.

A context engine controls how conversation context is managed when
approaching the model's token limit. The built-in ContextCompressor
is the default implementation. Third-party engines (e.g. LCM) can
replace it via the plugin system or by being placed in the
``plugins/context_engine/<name>/`` directory.

Selection is config-driven: ``context.engine`` in config.yaml.
Default is ``"compressor"`` (the built-in). Only one engine is active.

The engine is responsible for:
  - Deciding when compaction should fire
  - Performing compaction (summarization, DAG construction, etc.)
  - Optionally exposing tools the agent can call (e.g. lcm_grep)
  - Tracking token usage from API responses

Lifecycle:
  1. Engine is instantiated and registered (plugin register() or default)
  2. on_session_start() called when a conversation begins
  3. update_from_response() called after each API response with usage data
  4. should_compress() checked after each turn
  5. compress() called when should_compress() returns True
  6. on_session_end() called at real session boundaries (CLI exit, /reset,
     gateway session expiry) — NOT per-turn
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text


MEMORY_CONTEXT_MAX_CHARS = 6_000
_MEMORY_CONTEXT_HEAD_CHARS = 4_000
_MEMORY_CONTEXT_TAIL_CHARS = 1_500
_MEMORY_CONTEXT_TRUNCATION_MARKER = "\n...[memory provider context truncated]...\n"


def sanitize_memory_context(memory_context: str) -> str:
    """Prepare provider context for a context-engine/LLM egress boundary."""
    sanitized = redact_sensitive_text(
        memory_context.strip(),
        force=True,
        redact_url_credentials=True,
    )
    if len(sanitized) <= MEMORY_CONTEXT_MAX_CHARS:
        return sanitized
    return (
        sanitized[:_MEMORY_CONTEXT_HEAD_CHARS]
        + _MEMORY_CONTEXT_TRUNCATION_MARKER
        + sanitized[-_MEMORY_CONTEXT_TAIL_CHARS:]
    )


def automatic_compaction_status_message(
    engine: Any,
    *,
    phase: str,
    default_message: str,
    **context: Any,
) -> str | None:
    """Resolve host-visible status for an automatic compaction event.

    Engines can suppress routine automatic status with
    ``emit_automatic_compaction_status = False`` or customize it by defining
    ``get_automatic_compaction_status_message(...)``. Empty strings and
    ``None`` mean "do not emit a lifecycle status".
    """
    if not getattr(engine, "emit_automatic_compaction_status", True):
        return None

    formatter = getattr(engine, "get_automatic_compaction_status_message", None)
    if callable(formatter):
        message = formatter(
            phase=phase,
            default_message=default_message,
            **context,
        )
    else:
        message = default_message

    if message is None:
        return None
    message = str(message).strip()
    return message or None


class ContextEngine(ABC):
    """Base class all context engines must implement."""

    # -- Identity ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'compressor', 'lcm')."""

    # -- Token state (read by run_agent.py for display/logging) ------------
    #
    # Engines MUST maintain these. run_agent.py reads them directly.

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # -- Compaction parameters (read by run_agent.py for preflight) --------
    #
    # These control the preflight compression check.  Subclasses may
    # override via __init__ or property; defaults are sensible for most
    # engines.
    #
    # protect_first_n semantics (since PR #13754): count of non-system head
    # messages always preserved verbatim, IN ADDITION to the system prompt
    # which is always implicitly protected.  Default 3 keeps the
    # historical "system + first 3 non-system messages" head shape.

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # User-visible lifecycle status for automatic host-triggered compaction.
    # Alternative engines that treat compaction as routine background
    # maintenance can set this false to keep successful automatic passes silent;
    # warnings, errors, and explicit manual commands should still surface.
    emit_automatic_compaction_status: bool = True

    # -- Core interface ----------------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response.

        Called after every LLM call with a normalized usage dict. The legacy
        keys ``prompt_tokens``, ``completion_tokens``, and ``total_tokens``
        are always present. Newer hosts also include canonical buckets:
        ``input_tokens``, ``output_tokens``, ``cache_read_tokens``,
        ``cache_write_tokens``, and ``reasoning_tokens``. Engines should
        treat those fields as optional for compatibility with older hosts.
        """

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""

    def should_compress_info(self, prompt_tokens: int = None) -> "tuple[bool, str | None]":
        """Return ``(should_compress, reason)``.

        The base implementation is backward-compatible: engines that only
        implement ``should_compress`` get ``(should_compress(prompt_tokens),
        None)``. Concrete engines with richer block reasons (e.g. a
        summary-LLM cooldown or an anti-thrashing guard) override this to
        surface a human-readable reason so callers can warn the user instead
        of silently skipping compression. Added for the silent-overflow
        warning fix (#62625) so plugin engines don't raise AttributeError.
        """
        return self.should_compress(prompt_tokens), None

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Compact the message list and return the new message list.

        This is the main entry point. The engine receives the full message
        list and returns a (possibly shorter) list that fits within the
        context budget. The implementation is free to summarize, build a
        DAG, or do anything else — as long as the returned list is a valid
        OpenAI-format message sequence.

        Args:
            focus_topic: Optional topic string from manual ``/compress <focus>``.
                Engines that support guided compression should prioritise
                preserving information related to this topic.  Engines that
                don't support it may simply ignore this argument.
            force: Whether a user-requested compression should bypass an
                engine-owned cooldown. Engines without cooldowns may ignore it.
            memory_context: Text returned by memory providers immediately before
                compaction. Summarizing engines should include non-empty text in
                their handoff prompt. Older engines may omit this parameter; the
                host filters unsupported optional arguments by signature.
        """

    # -- Optional: proactive tool-result prune -----------------------------

    def prune_tool_results_only(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Deterministically trim old tool-result payloads without an LLM call.

        Runs on a low, cost-oriented trigger independent of ``should_compress``
        so large-window engines can reclaim re-sent tool output long before full
        compaction would fire. Returns ``(messages, n_pruned)``.

        Default is a safe no-op: the list is returned unchanged with ``0``
        pruned. Engines that don't implement a cheap prune — and any engine that
        predates this hook — inherit this default, so the agent loop's
        post-tool-call prune path never raises ``AttributeError`` on them. The
        built-in ContextCompressor overrides this with the real implementation.
        """
        return messages, 0

    # -- Optional: per-turn context selection (distinct from compression) --

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,
        incoming_message: Dict[str, Any] = None,
        budget_tokens: int = 0,
    ) -> List[Dict[str, Any]]:
        """Optionally choose/replace the context for THIS request, pre-generation.

        Called every turn after the request message list is assembled and
        before it is dispatched to the provider — independent of
        ``should_compress()``. This lets an engine *select* which context
        enters the prompt (retrieval, topic routing, role/branch switching)
        rather than *shrink* context that is already there. The two verbs are
        orthogonal:

          - ``compress()``      : context is too long  -> make it shorter.
          - ``select_context()``: this turn belongs to a different context
                                  -> use that one instead.

        Without this hook, engines that need per-turn access to the message
        list have to force ``should_compress()`` to return ``True`` so that
        ``compress()`` is invoked every turn purely as a callback — which
        conflates selection with compression and degrades behaviour when the
        engine's backend is unavailable. ``select_context()`` removes the need
        for that workaround.

        The returned list is request-only: it replaces the messages sent to
        the provider for this single call and MUST NOT be treated as persisted
        transcript state. The conversation history in the session DB is left
        untouched, so nothing leaks across turns. Return ``None`` to leave the
        request unchanged.

        Unlike the ``pre_llm_call`` plugin hook (which appends to the user
        message and intentionally never rewrites the list, to preserve the
        cache prefix), ``select_context()`` may *replace* the message list.

        Ordering / cache contract: the host runs this hook **before** prompt
        cache-control and **before** every request sanitizer (orphaned-tool
        cleanup, thinking-only/role normalization, whitespace/JSON
        normalization). So (a) whatever the hook returns still passes through
        the same validation as any request — a malformed replacement cannot
        reach the provider — and (b) prompt-cache stability (an AGENTS.md
        invariant) is preserved: the default no-op leaves the request
        byte-identical, so cache behaviour is unchanged for the built-in
        compressor and any non-implementing engine. An engine that *does*
        replace the list changes its own cache prefix by definition; that is
        the engine's concern, and cache-control breakpoints are re-derived on
        the selected list. The hook is evaluated per provider request (so it
        re-runs on retries within a turn), consistent with "select the context
        for THIS request".

        Args:
            request_messages: The assembled request message list (system
                prompt + history + any ephemeral prefill), in OpenAI format.
            conversation_messages: The unmodified persisted conversation
                history, for reference only (do not mutate).
            incoming_message: The current turn's user message, if available.
            budget_tokens: The active model's context length, or 0 if unknown.

        Default returns ``None`` (no-op) — zero impact on the built-in
        compressor or any existing engine.
        """
        return None

    def on_turn_complete(
        self,
        messages: List[Dict[str, Any]],
        usage: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:
        """Observe a finished user turn (post-turn ingestion / observation).

        Called from the standard turn-finalization path once the assistant/tool
        loop completes, with the finalized in-memory transcript snapshot. This
        is the complement to ``select_context()``: selection happens *before*
        the request, while observation happens *after* the turn. It lets an
        engine ingest, index, summarize, or update routing / topic / session
        state from what actually happened — so the next ``select_context()``
        can act on it.

        Coverage: this fires from the normal finalization seam. Some abnormal
        early-return paths in the loop (e.g. a content-policy block or a
        provider terminal failure) persist and return without routing through
        finalization, and therefore do not currently emit this hook. Treat it
        as a best-effort post-turn observation for completed turns, not a
        guaranteed callback for every possible early exit; unifying all
        terminal paths behind one finalization seam is a separate follow-up.

        Together the two hooks remove the need to abuse ``should_compress()`` /
        ``compress()`` as a generic per-turn callback just to observe history,
        and they cover the case where a turn finishes and there may be no next
        request from which to infer the previous turn.

        ``messages`` is a shallow copy and should be treated as read-only:
        return values are ignored and this hook must not rely on transcript
        mutation for persistence. ``kwargs`` may include ``turn_id``,
        ``task_id``, ``api_call_count``, ``interrupted``, ``failed``, and
        ``turn_exit_reason``.

        ``usage`` carries the completed turn's canonical token usage (the same
        dict shape passed to ``update_from_response`` — ``prompt_tokens`` /
        ``completion_tokens`` / ``total_tokens`` plus the canonical
        ``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` /
        ``cache_write_tokens`` / ``reasoning_tokens`` buckets) so an engine can
        weigh how large/expensive the selected context actually was when
        deciding the next ``select_context()``. It is ``None`` on finalized
        turns that never reached a provider response (e.g. interrupt); engines
        must treat it as optional.

        Default is a no-op.
        """
        return None

    # -- Optional: pre-flight check ----------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick rough check before the API call (no real token count yet).

        Default returns False (skip pre-flight). Override if your engine
        can do a cheap estimate.
        """
        return False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """Return True when preflight should trust recent real usage instead.

        Built-in compression uses this to avoid re-compacting from known-noisy
        rough estimates after a compressed request has already fit. Third-party
        engines can ignore it safely.
        """
        return False

    def get_automatic_compaction_status_message(
        self,
        *,
        phase: str,
        default_message: str,
        **context: Any,
    ) -> str | None:
        """Return user-visible status for automatic host-triggered compaction.

        Return ``None`` to suppress successful automatic lifecycle status for
        this compaction event. ``phase`` identifies the host call site (for
        example ``"preflight"`` or ``"compress"``). ``context`` contains
        best-effort fields such as ``approx_tokens`` and ``threshold_tokens``.

        This hook does not control warning/error messages or explicit manual
        commands such as ``/compress``.
        """
        if not self.emit_automatic_compaction_status:
            return None
        return default_message

    # -- Optional: manual /compress preflight ------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick check: is there anything in ``messages`` that can be compacted?

        Used by the gateway ``/compress`` command as a preflight guard —
        returning False lets the gateway report "nothing to compress yet"
        without making an LLM call.

        Default returns True (always attempt).  Engines with a cheap way
        to introspect their own head/tail boundaries should override this
        to return False when the transcript is still entirely protected.
        """
        return True

    # -- Optional: session lifecycle ---------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins.

        Use this to load persisted state (DAG, store) for the session.
        kwargs may include hermes_home, platform, model, etc.
        """

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Called at real session boundaries (CLI exit, /reset, gateway expiry).

        Use this to flush state, close DB connections, etc.
        NOT called per-turn — only when the session truly ends.
        """

    def on_session_reset(self) -> None:
        """Called on /new or /reset. Reset per-session state.

        Default resets compression_count and token tracking.
        """
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    # -- Optional: tools ---------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this engine provides to the agent.

        Default returns empty list (no tools). LCM would return schemas
        for lcm_grep, lcm_describe, lcm_expand here.
        """
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent.

        Only called for tool names returned by get_tool_schemas().
        Must return a JSON string.

        kwargs may include:
          messages: the current in-memory message list (for live ingestion)
        """
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- Optional: status / display ----------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return status dict for display/logging.

        Default returns the standard fields run_agent.py expects.
        """
        # Clamp the -1 "compression just ran, awaiting real usage" sentinel
        # (set by conversation_compression) to 0 so status readers don't see a
        # raw -1 or a negative usage_percent on the transitional turn. Mirrors
        # the CLI/gateway status-bar paths (cli.py, tui_gateway/server.py).
        last_prompt = self.last_prompt_tokens if self.last_prompt_tokens > 0 else 0
        return {
            "last_prompt_tokens": last_prompt,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, last_prompt / self.context_length * 100)
                if self.context_length else 0
            ),
            "compression_count": self.compression_count,
        }

    # -- Optional: model switch support ------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """Called when the user switches models or on fallback activation.

        Default updates context_length and recalculates threshold_tokens
        from threshold_percent. Override if your engine needs more
        (e.g. recalculate DAG budgets, switch summary models).
        """
        self.context_length = context_length
        # Apply per-model threshold overrides if set (longest substring match).
        # Falls back to _config_threshold_percent (the raw config value) when
        # no override matches. Plugin engines that override update_model() can
        # call resolve_model_threshold() for the same logic.
        from agent.context_compressor import resolve_model_threshold
        if not hasattr(self, "_config_threshold_percent"):
            # Snapshot the pre-override percent ONCE so repeated model
            # switches fall back to the engine's configured value, not the
            # previous model's override.
            self._config_threshold_percent = self.threshold_percent
        self._base_threshold_percent = resolve_model_threshold(
            model, getattr(self, "model_thresholds", {}),
            self._config_threshold_percent,
        )
        self.threshold_percent = self._base_threshold_percent
        self.threshold_tokens = int(context_length * self.threshold_percent)
