"""Pre-LLM summary skips for ContextCompressor: compact deterministically without calling a summary model."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)

# Skip the LLM call when the compressible middle is below this fraction of the
# threshold (and a prior ineffectiveness strike exists); dropping alone suffices.
# See #60451.
_FEASIBILITY_SKIP_MIDDLE_FRACTION = 0.10
_FALLBACK_PROBE_AT_MODEL_CONFIG_KEY = "_fallback_probe_at"


class PreLlmSkipMixin:
    """Decides, before the summary call, whether compress() should skip the summary model."""

    def _set_fallback_probe_at(self, at: float) -> None:
        """Set the summary-model probe deadline, persisting on change only (0 = unarmed) so a rebuilt
        compressor resumes the bench window instead of restarting it (same contract as #100185)."""
        if at == self._fallback_probe_at:
            return
        self._fallback_probe_at = at
        self._durable_write(
            "patch_session_model_config", "summary-model probe deadline", {_FALLBACK_PROBE_AT_MODEL_CONFIG_KEY: at or None},
        )

    def _load_fallback_probe_at(self) -> None:
        self._load_durable(
            "_fallback_probe_at", "get_session_model_config_value", "summary-model probe deadline",
            float, 0.0, _FALLBACK_PROBE_AT_MODEL_CONFIG_KEY, 0.0,
        )

    def _fallback_streak_skip(self, telemetry: dict[str, Any]) -> bool:
        """Pre-LLM skip while two summaries in a row fell back: compact deterministically instead of
        paying for a summary model that keeps failing (#63008). One probe per recovery window; a
        healthy summary resets the streak, another fallback benches it again."""
        if self._fallback_compression_streak < 2:
            self._set_fallback_probe_at(0.0)
            return False
        # Wall clock: the deadline is persisted on the session row so a gateway rebuild resumes the window.
        now = time.time()
        if self._fallback_probe_at and now >= self._fallback_probe_at:
            self._set_fallback_probe_at(0.0)
            if not self.quiet_mode:
                logger.info(
                    "Compression: probing the summary model again after %d fallback summaries in a row",
                    self._fallback_compression_streak,
                )
            return False
        if not self._fallback_probe_at or self._fallback_probe_at - now > self._ANTI_THRASH_RECOVERY_SECONDS:
            self._set_fallback_probe_at(now + self._ANTI_THRASH_RECOVERY_SECONDS)
        self._last_feasibility_skip = True
        telemetry["failure_class"] = "summary_model_benched"
        if not self.quiet_mode:
            logger.warning(
                "Compression: %d fallback summaries in a row — skipping LLM summarization, proceeding with "
                "deterministic message dropping. Next summary-model probe in %.0fs.",
                self._fallback_compression_streak, max(0.0, self._fallback_probe_at - now),
            )
        return True

    def _feasibility_skip(
        self, telemetry: dict[str, Any], turns_to_summarize: list[dict[str, Any]],
        compress_start: int, compress_end: int,
    ) -> bool:
        """Pre-LLM skip after a real-usage ineffectiveness strike (reads the counter, never writes)."""
        if self._ineffective_compression_count < 1:
            return False
        # Reuse the telemetry estimate so log and telemetry agree; None means the regions helper
        # no-op'd (0 is valid).
        middle_tokens = telemetry.get("middle_window_tokens")
        middle_tokens = estimate_messages_tokens_rough(turns_to_summarize) if middle_tokens is None else middle_tokens
        if middle_tokens >= int(self.threshold_tokens * _FEASIBILITY_SKIP_MIDDLE_FRACTION):
            return False
        self._last_feasibility_skip = True
        self._prellm_skip_count += 1
        telemetry["prellm_skip_count"] = self._prellm_skip_count
        if not self.quiet_mode:
            logger.warning(
                "Compression: middle section (%d tokens at indices %d-%d) is below %.0f%% of threshold (%d tokens) — "
                "skipping LLM summarization, proceeding with deterministic message dropping. prellm_skip_count=%d",
                middle_tokens, compress_start, compress_end,
                _FEASIBILITY_SKIP_MIDDLE_FRACTION * 100,
                self.threshold_tokens, self._prellm_skip_count,
            )
        return True
