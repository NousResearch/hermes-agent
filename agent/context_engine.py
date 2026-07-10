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
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
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
        """

    # -- Optional: pre-flight check ----------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick rough check before the API call (no real token count yet).

        Default returns False (skip pre-flight). Override if your engine
        can do a cheap estimate.
        """
        return False

    # -- "Compact on the truth" calibration (P2) ---------------------------
    # Shared concrete implementation for ALL engines (ContextCompressor, LCMEngine,
    # third-party). The rough estimate (estimate_request_tokens_rough, char/3.5)
    # over-counts schema-heavy requests; the provider's real prompt_tokens measures
    # the skew. We scale the rough by the clamped median skew before comparing to
    # threshold, so compaction fires on the provider's real accounting, not the
    # ~21%-inflated guess. State is lazy (engines whose __init__ doesn't set it still
    # work). Pure functions of recorded scalars → no defer-baseline to ratchet.

    _SKEW_FLOOR_DEFAULT = 0.7
    _HARD_FRAC_DEFAULT = 0.95
    _SKEW_HISTORY = 5

    def reset_skew_calibration(self) -> None:
        """Clear per-conversation skew state at a session boundary. The engine is a
        process-global singleton, so a skew learned in one conversation must not leak
        into the next session's first preflight (Greptile #111)."""
        self._recent_skews = []
        self._last_rough_sent = 0
        self.rough_at_last_real = 0

    def seed_skew_calibration(self, ratios: "list[float]") -> None:
        """Seed the skew history from persisted per-session state (restart resume).

        Only applies when the in-memory history is empty (a live history is
        fresher than any persisted snapshot) and only accepts sane ratios
        (0 < r <= 1.0; rough never under-counts). Invalid input is ignored —
        seeding is an optimization, never a correctness requirement.
        """
        if getattr(self, "_recent_skews", None):
            return
        clean = []
        for r in ratios or []:
            try:
                f = float(r)
            except (TypeError, ValueError):
                continue
            if 0.0 < f <= 1.0:
                clean.append(f)
        if clean:
            self._recent_skews = clean[-self._SKEW_HISTORY:]

    def _persist_skew_history(self) -> None:
        """Write the current skew history to the bound session row (best-effort).

        Uses the same session binding as the durable compression-failure
        cooldown (``bind_session_state``). No binding → silently skip.
        """
        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        if not session_db or not session_id:
            return
        writer = getattr(session_db, "record_compression_skew_history", None)
        if writer is None:
            return
        writer(session_id, list(getattr(self, "_recent_skews", []) or []))

    def note_rough_sent(self, rough_tokens: int) -> None:
        """Stash the rough estimate of the request about to be sent so the next
        ``record_skew_from_real``/``update_from_response`` pairs it with the real
        prompt_tokens (the skew denominator). Same message set ⇒ correct ratio."""
        if rough_tokens and rough_tokens > 0:
            self._last_rough_sent = int(rough_tokens)

    def record_skew_from_real(self, real_prompt_tokens: int) -> None:
        """Pair a real provider ``prompt_tokens`` with the stashed rough (atomically,
        from the engine's ``update_from_response``). Records ratio ≤ 1.0 (rough
        over-counts; never scale UP), keeps the last-k for median smoothing.

        T0 (2026-06-27): the stashed rough is CONSUMED (reset to 0) after use, so a
        single ``note_rough_sent`` pairs with exactly ONE real reading. Without this,
        a multi-call turn (one preflight ``note_rough_sent`` + N ``update_from_response``)
        divided the SAME stale rough into N growing reals, polluting the skew median
        the trigger calibrates on. See spec
        ~/.hermes/plans/2026-06-27_skew-telemetry-and-render-harness-SPEC.md.
        """
        last_rough = getattr(self, "_last_rough_sent", 0)
        if real_prompt_tokens and real_prompt_tokens > 0 and last_rough > 0:
            self.rough_at_last_real = last_rough
            ratio = min(1.0, real_prompt_tokens / last_rough)
            hist = getattr(self, "_recent_skews", None)
            if hist is None:
                hist = []
                self._recent_skews = hist
            hist.append(ratio)
            if len(hist) > self._SKEW_HISTORY:
                self._recent_skews = hist[-self._SKEW_HISTORY:]
            # T0: consume the stashed rough so the next real reading without a fresh
            # note_rough_sent records nothing (bounds cross-turn/session mispairing
            # to ≤1 on the process-global singleton).
            self._last_rough_sent = 0
            # Persist the updated history so a process restart can seed the
            # calibration instead of reverting to skew=1.0 (raw rough) on the
            # first post-restart preflight (2026-07-10 false-fire incident).
            # Best-effort: persistence failure must never touch the live turn.
            try:
                self._persist_skew_history()
            except Exception:
                pass
            # T1: skew telemetry — one COMPACTION_SKEW line per FRESH pair, so a skew
            # distribution is buildable from logs. Best-effort: a logging failure or a
            # missing attribute must NEVER propagate into the live turn (INV-2).
            self._emit_skew_telemetry(last_rough, int(real_prompt_tokens), ratio)

    def _emit_skew_telemetry(self, rough: int, real: int, ratio: float) -> None:
        """Best-effort COMPACTION_SKEW telemetry (T1). Never raises into the hot path.

        Emits one ``info`` line per fresh (rough, real) pair and appends the same
        ``task=main`` line to a dedicated append-only sink for the v0.2 floor tune
        (the rotating gateway logs can rotate skew lines out before N accrues).
        """
        try:
            task = getattr(self, "_aux_task", None) or "main"
            model = getattr(self, "model", "") or ""
            provider = getattr(self, "provider", "") or ""
            ctx = getattr(self, "context_length", 0) or 0
            model_str = f"{provider}/{model}" if provider else model
            line = (
                f"COMPACTION_SKEW rough={rough} real={real} ratio={ratio:.3f} "
                f"task={task} model={model_str} ctx={ctx}"
            )
            logger.info(line)
            # Dedicated non-rotating sample sink (main-turn distribution only — the
            # task the trigger uses). Aux tasks don't reach here without their own
            # note_rough_sent (consumed), so this is naturally main-dominated.
            if task == "main":
                self._append_skew_sample(line)
        except Exception:
            # Telemetry must never break a live turn (INV-2).
            pass

    def _append_skew_sample(self, line: str) -> None:
        """Append one skew sample to ~/.hermes/state/skew-samples.log (append-only,
        non-rotating). Best-effort; any failure is swallowed by the caller's guard."""
        import os

        home = os.environ.get("HERMES_HOME") or os.path.join(
            os.path.expanduser("~"), ".hermes"
        )
        # HERMES_HOME may already be a profile dir; the sink is per-process-home,
        # which is the correct scope for a per-process skew distribution.
        state_dir = os.path.join(home, "state")
        os.makedirs(state_dir, exist_ok=True)
        import time as _time

        stamp = _time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.join(state_dir, "skew-samples.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {line}\n")

    def _current_skew(self) -> float:
        """Median of the last-k real/rough ratios, clamped to [floor, 1.0]. Returns
        1.0 (no scaling = pre-P2 behavior) when no real reading has paired yet."""
        hist = getattr(self, "_recent_skews", None)
        if not hist:
            return 1.0
        ordered = sorted(hist)
        mid = len(ordered) // 2
        med = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        floor = getattr(self, "_skew_floor", self._SKEW_FLOOR_DEFAULT)
        return max(floor, min(1.0, med))

    def calibrated_tokens(self, rough_tokens: int) -> int:
        """``round(rough × skew)`` — the rough estimate scaled to the provider's
        measured accounting. Safe default (skew 1.0) ⇒ identical to raw rough."""
        if rough_tokens <= 0:
            return rough_tokens
        return int(round(rough_tokens * self._current_skew()))

    def should_compress_calibrated(self, rough_tokens: int) -> bool:
        """P2 trigger: compact when CALIBRATED rough ≥ threshold, OR when RAW rough
        reaches the window ceiling (skew-independent 413 / dense-paste guard — a
        dense in-turn paste raises raw rough so the ceiling fires even if a stale
        skew would defer). Delegates the actual threshold + anti-thrash to the
        engine's ``should_compress``."""
        ctx_len = getattr(self, "context_length", 0) or 0
        hard_frac = getattr(self, "_hard_frac", self._HARD_FRAC_DEFAULT)
        if ctx_len > 0 and rough_tokens >= int(ctx_len * hard_frac):
            return self.should_compress(rough_tokens)
        return self.should_compress(self.calibrated_tokens(rough_tokens))

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

        Engines may return bare OpenAI function schemas or full
        {"type": "function", "function": ...} tool definitions; the host
        normalizes both. LCM returns schemas for lcm_grep, lcm_describe,
        lcm_expand here.
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
        self.threshold_tokens = int(context_length * self.threshold_percent)
