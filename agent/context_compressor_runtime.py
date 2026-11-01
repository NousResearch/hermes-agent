"""Bind compression summaries to the runtime of their owning session."""
import copy
from typing import Any


_REASONING_CONFIG_UNSET = object()


class CompressionRuntimeMixin:
    def _summary_main_runtime(self) -> dict[str, Any]:
        """Runtime snapshot for the next summary call: the owning session's attempt
        snapshot when it matches, else this compressor's construct/update snapshot.

        A reused compressor can outlive a session effort/model switch. The host scopes a
        fresh runtime (``scoped_runtime_main``) around each compression attempt, and that
        ContextVar also isolates a detached pool worker from a later attempt. The session-id
        match refuses a FOREIGN session's runtime even on the same deployment; standalone
        compressors (no bound session) always use their own snapshot.
        """
        from agent.auxiliary_client import get_scoped_runtime_main

        runtime = get_scoped_runtime_main()
        session_id = getattr(self, "_session_id", "")
        if session_id and runtime.get("session_id") == session_id:
            return runtime
        return {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "api_mode": self.api_mode,
            "reasoning_config": copy.deepcopy(getattr(self, "reasoning_config", None)),
        }


    def update_model(
        self, model: str, context_length: int, base_url: str = "", api_key: Any = "", provider: str = "",
        api_mode: str = "", max_tokens: int | None = None, reasoning_config: Any = _REASONING_CONFIG_UNSET,
    ) -> None:
        """Update model info after a model switch or fallback activation."""
        from agent.context_compressor import _SUMMARY_TOKENS_CEILING

        runtime_changed = (model, provider, base_url, api_mode) != (self.model, self.provider, self.base_url, self.api_mode)
        if reasoning_config is not _REASONING_CONFIG_UNSET:
            self.reasoning_config = copy.deepcopy(reasoning_config)
        elif runtime_changed:
            # An omitted effort cannot carry the previous route's policy across a route
            # change; a same-route context-window recalibration keeps the snapshot.
            self.reasoning_config = None
        self.model, self.base_url, self.api_key, self.provider, self.api_mode = model, base_url, api_key, provider, api_mode
        self.context_length = context_length
        # max_tokens=None means "unspecified": keep the existing output reservation.
        # A switch that genuinely changes the output budget passes the new value explicitly. (#43547)
        if max_tokens is not None:
            self.max_tokens = self._coerce_max_tokens(max_tokens)
        if runtime_changed:
            # The aux ceiling was probed against the previous main runtime (an "auto" aux route follows the
            # main model); the caller re-runs the feasibility probe. A same-runtime recompute (overflow-reported
            # window, grown local window, tier cap) keeps it: the summariser did not change (#114707).
            self._aux_context_ceiling = None
        self._base_threshold_percent, self.threshold_percent, self.threshold_tokens = self._derive_trigger(
            model, context_length, provider)
        self._apply_threshold_tokens_cap()
        # Reset to None so the property recomputes via the mode-aware path (not the legacy formula).
        self._tail_token_budget = None
        _ = self.tail_token_budget  # eager recompute, same timing as before
        self.max_summary_tokens = min(int(context_length * 0.05), _SUMMARY_TOKENS_CEILING)
        # Old usage cannot price a new model. Clear it without arming the post-compaction
        # latch: the next response supplies usage or enables the usage-less fallback.
        self.last_prompt_tokens = self.last_completion_tokens = self.last_total_tokens = 0
        self._reset_real_usage_pairing()
        # Strikes were judged against the previous threshold; void them durably too.
        self._record_ineffective_compression_verdict(0)
        self._prellm_skip_count = 0
        if runtime_changed:
            self._fallback_compression_streak = 0
            self._persist_fallback_compression_streak()
            # Cooldowns are scoped to the failed model/provider; a switch gets an immediate attempt.
            self._clear_compression_failure_cooldown()
            # The overload budget rode those cooldowns; a new runtime restarts it too.
            self._reset_consecutive_overload_aborts()
        self._verify_compaction_cleared_threshold = self._last_compression_made_progress = False
        # Runway was computed against the previous model's trigger; clear the durable copy too.
        self._reset_proactive_prune_rearm()
        self._clear_durable_proactive_prune_rearm()

