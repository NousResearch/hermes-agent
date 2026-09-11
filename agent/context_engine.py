"""Abstract base class for pluggable context engines.

A context engine decides when/how conversation context is compacted near the token
limit, tracks usage, and may expose tools. ContextCompressor is the default;
``context.engine`` selects a plugin (``plugins/context_engine/<name>/``); one is active.
Lifecycle: on_session_start() -> per API response update_from_response() -> per turn
should_compress() / compress() -> on_session_end() at real session boundaries only
(CLI exit, /reset, gateway expiry), never per-turn.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text


MEMORY_CONTEXT_MAX_CHARS = 6_000
_MEMORY_CONTEXT_HEAD_CHARS = 4_000
_MEMORY_CONTEXT_TAIL_CHARS = 1_500
_MEMORY_CONTEXT_TRUNCATION_MARKER = "\n...[memory provider context truncated]...\n"


def sanitize_memory_context(memory_context: str) -> str:
    """Prepare provider context for a context-engine/LLM egress boundary."""
    sanitized = redact_sensitive_text(memory_context.strip(), force=True, redact_url_credentials=True)
    if len(sanitized) <= MEMORY_CONTEXT_MAX_CHARS:
        return sanitized
    return sanitized[:_MEMORY_CONTEXT_HEAD_CHARS] + _MEMORY_CONTEXT_TRUNCATION_MARKER + sanitized[-_MEMORY_CONTEXT_TAIL_CHARS:]


def automatic_compaction_status_message(engine: Any, *, phase: str, default_message: str, **context: Any) -> str | None:
    """Host-visible status for an automatic compaction event; ``None`` = emit nothing.

    Engines suppress via ``emit_automatic_compaction_status = False`` or
    customize via ``get_automatic_compaction_status_message(...)``.
    """
    if not getattr(engine, "emit_automatic_compaction_status", True):
        return None
    formatter = getattr(engine, "get_automatic_compaction_status_message", None)
    message = formatter(phase=phase, default_message=default_message, **context) if callable(formatter) else default_message
    if message is None:
        return None
    return str(message).strip() or None


class ContextEngine(ABC):
    """Base class all context engines must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'compressor', 'lcm')."""

    # Token state: engines MUST maintain these; run_agent.py reads them directly.
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0
    # Compaction parameters (read by run_agent.py for preflight). protect_first_n counts
    # non-system head messages kept verbatim IN ADDITION to the always-protected system
    # prompt (3 keeps the historical head shape).
    # These control the preflight compression check. Subclasses may override via __init__ or property;
    # defaults are sensible for most engines. See #13754.
    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6
    # False keeps successful automatic compaction passes silent (routine background
    # maintenance); warnings, errors and manual /compress still surface.
    emit_automatic_compaction_status: bool = True

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage after every LLM call.

        ``prompt_tokens``/``completion_tokens``/``total_tokens`` are always present; the
        canonical buckets (``input_tokens``, ``output_tokens``, ``cache_read_tokens``,
        ``cache_write_tokens``, ``reasoning_tokens``) are optional on older hosts.
        """

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""

    def should_compress_info(self, prompt_tokens: int = None) -> "tuple[bool, str | None]":
        """Return ``(should_compress, reason)``.

        Engines with block reasons (summary-LLM cooldown, anti-thrashing guard) override
        this so callers can warn instead of silently skipping; the default keeps plugin
        engines from raising AttributeError.
        """
        return self.should_compress(prompt_tokens), None

    @abstractmethod
    def compress(
        self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None, force: bool = False, memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Compact ``messages`` into a valid OpenAI-format list that fits the budget.

        ``focus_topic`` comes from manual ``/compress <focus>`` (prioritise that topic);
        ``force`` asks to bypass an engine-owned cooldown; ``memory_context`` is provider
        text for the handoff prompt. Older engines may omit optional parameters — the
        host filters them by signature.
        """

    def prune_tool_results_only(
        self, messages: List[Dict[str, Any]], current_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Deterministically trim old tool-result payloads without an LLM call.

        Runs on a low, cost-oriented trigger independent of ``should_compress`` so
        large-window engines reclaim re-sent tool output long before full compaction.
        Returns ``(messages, n_pruned)``; the default no-op keeps older engines safe.
        """
        return messages, 0

    def select_context(
        self, request_messages: List[Dict[str, Any]], *, conversation_messages: List[Dict[str, Any]] = None,
        incoming_message: Dict[str, Any] = None, budget_tokens: int = 0,
    ) -> List[Dict[str, Any]]:
        """Optionally *select* (replace) the context for THIS request, pre-generation.

        Runs on every provider request (also retries), independent of
        ``should_compress()``: ``compress()`` shrinks over-long context, this swaps in a
        different one (retrieval, topic routing, branch switching). Return ``None`` to
        leave the request unchanged. The returned list is request-only — it MUST NOT be
        treated as persisted transcript state (session DB history is untouched); unlike
        ``pre_llm_call`` it may replace the list. The host runs it before prompt
        cache-control and every request sanitizer, so a malformed replacement never
        reaches the provider and the default no-op keeps the request byte-identical;
        an engine that replaces the list changes its own cache prefix (breakpoints are
        re-derived on the selected list). ``request_messages`` is the assembled request
        (system prompt + history + ephemeral prefill); ``conversation_messages`` is the
        persisted history for reference only (do not mutate); ``budget_tokens`` is the
        model's context length or 0 if unknown.
        """
        return None

    def on_turn_complete(self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None, **kwargs: Any) -> None:
        """Observe a finished turn (complement of ``select_context()``) to index/update
        routing state for the next request.

        Best-effort, not guaranteed: fires from the normal finalization seam only; some
        abnormal early returns (content-policy block, provider terminal failure) skip it.
        ``messages`` is a read-only shallow copy (return value ignored; never rely on
        transcript mutation). ``usage`` has the ``update_from_response`` shape and is
        ``None`` when no provider response was reached (interrupt). ``kwargs`` may include
        ``turn_id``, ``task_id``, ``api_call_count``, ``interrupted``, ``failed``, ``turn_exit_reason``.
        """
        return None

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Cheap rough check before the API call (no real token count yet); default skips."""
        return False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """True when preflight should trust recent real usage over the noisy rough
        estimate (avoids re-compacting after a compressed request already fit)."""
        return False

    def get_automatic_compaction_status_message(
        self, *, phase: str, default_message: str, **context: Any,
    ) -> str | None:
        """User-visible status for automatic compaction, or ``None`` to suppress it.

        ``phase`` is the host call site (``"preflight"`` / ``"compress"``); ``context``
        carries best-effort ``approx_tokens`` / ``threshold_tokens``. Warnings, errors
        and manual ``/compress`` are not governed by this hook.
        """
        return default_message if self.emit_automatic_compaction_status else None

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Preflight guard for gateway ``/compress``: False reports "nothing to
        compress yet" without an LLM call (e.g. transcript entirely protected)."""
        return True

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Session begins: load persisted state. kwargs may include hermes_home, platform, model."""

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Real session boundary (CLI exit, /reset, gateway expiry) — never per-turn."""

    def on_session_reset(self) -> None:
        """/new or /reset: reset per-session state (default: counters and token tracking)."""
        # Reset cross-call calibration state captured under the PREVIOUS model. These fields encode "the
        # provider proved this prompt fit" / "preflight can be deferred" decisions that are only valid for
        # the model that produced them. Carrying them across a switch to a smaller-context model would let
        # should_defer_preflight_to_real_usage() suppress a preflight compression the new model actually
        # needs — the exact oversized-send-after-switch failure in #23767. The new model's first response
        # repopulates them via update_from_response(). Setting last_prompt_tokens to 0 (NOT -1) is
        # deliberate: 0 is the documented "no real usage yet -> use the rough estimate" state, so the post-
        # response should_compress path falls back to estimate_request_tokens_rough rather than skipping
        # compression. -1 is a different sentinel (#36718, "compression just ran, await real usage") and
        # must not be set here.
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Tool schemas this engine exposes to the agent (default: none)."""
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a call to one of this engine's tools; must return a JSON string.
        kwargs may include ``messages`` (live in-memory list)."""
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    def get_status(self) -> Dict[str, Any]:
        """Status dict with the standard fields run_agent.py expects."""
        # Clamp the -1 "compression just ran, awaiting real usage" sentinel to 0 so no
        # reader sees a negative usage_percent on the transitional turn.
        last_prompt = max(self.last_prompt_tokens, 0)
        return {
            "last_prompt_tokens": last_prompt,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": min(100, last_prompt / self.context_length * 100) if self.context_length else 0,
            "compression_count": self.compression_count,
        }

    def update_model(
        self, model: str, context_length: int, base_url: str = "", api_key: str = "",
        provider: str = "", api_mode: str = "",
    ) -> None:
        """Model switch / fallback: recompute threshold_tokens (override for more).

        Per-model threshold override (longest substring match), else the raw config
        percent — snapshotted ONCE so repeated switches fall back to the configured
        value, not the previous model's override.
        """
        self.context_length = context_length
        from agent.context_compressor import resolve_model_threshold
        if not hasattr(self, "_config_threshold_percent"):
            self._config_threshold_percent = self.threshold_percent
        self._base_threshold_percent = resolve_model_threshold(
            model, getattr(self, "model_thresholds", {}), self._config_threshold_percent)
        self.threshold_percent = self._base_threshold_percent
        self.threshold_tokens = int(context_length * self.threshold_percent)
