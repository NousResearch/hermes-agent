"""Proactive feature router — suggests the right Hermes capability at the
right time (PR-B).

The router runs at turn start (before the tool loop) and, when the user's
message matches a registry feature above the confidence threshold, produces
an advisory suggestion string. That string is appended to the current turn's
API copy via the existing ``ext_prefetch_cache`` sidecar — it never touches
the stored conversation, never mutates the system prompt, and never invokes
a tool itself.

Config (config.yaml)::

    proactive_features:
      enabled: false          # master switch (SUGGEST mode; default OFF)
      min_confidence: 0.7     # global floor (registry per-feature is higher)
      rate_limit_turns: 5     # suppress suggestions for N turns after one fires
      auto_apply: false       # OPT-IN: agent may apply suggestion without asking
      features:               # per-feature kill-switch map
        parallel_subtasks: true
        # ...

Safety invariants (SECURITY-BASELINE.md):
  * SUGGEST only by default — auto_apply is opt-in and still runs through
    the normal approval pipeline.
  * Registry entries cannot name unknown capabilities (whitelist at init).
  * No network, no subprocess, no system-prompt mutation.
  * Failures are non-fatal: a broken router must never break a turn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agent.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)


class FeatureRouter:
    """Turn-scoped suggestion engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 registry: Optional[FeatureRegistry] = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.auto_apply = bool(cfg.get("auto_apply", False))
        self.min_confidence = float(cfg.get("min_confidence", 0.7))
        self.rate_limit_turns = max(0, int(cfg.get("rate_limit_turns", 5)))
        self.registry = registry or FeatureRegistry()

        # Per-feature kill switches: {"feature_id": True/False}.
        feature_flags = cfg.get("features", {}) or {}
        self._feature_flags = {
            f.id: bool(feature_flags.get(f.id, True))
            for f in self.registry.features
        }

        # Rate-limit state. Initialize so the FIRST eligible turn is allowed
        # (rate_limit_turns counts turns AFTER a suggestion fires).
        self._turns_since_suggestion = self.rate_limit_turns
        self._last_suggestion: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def on_turn_start(self, user_message: str) -> str:
        """Evaluate a turn and return a suggestion string (or '').

        Called by the turn prologue BEFORE the tool loop. Non-fatal: any
        error produces no suggestion.
        """
        if not self.enabled:
            return ""
        try:
            if self._turns_since_suggestion < self.rate_limit_turns:
                self._turns_since_suggestion += 1
                return ""
            text = user_message if isinstance(user_message, str) else ""
            if not text.strip():
                return ""

            # Respect per-feature kill switches.
            f = self.registry.suggest(text, min_confidence=self.min_confidence)
            if f is None:
                return ""
            if not self._feature_flags.get(f.id, True):
                return ""

            self._turns_since_suggestion = 0
            self._last_suggestion = f.id
            return self.registry.suggest_text(text, min_confidence=self.min_confidence)
        except Exception as e:  # never break a turn
            logger.debug("feature router suggestion failed (non-fatal): %s", e)
            return ""

    def auto_apply_allowed(self) -> bool:
        """True when the user has opted into auto-apply (still approval-gated)."""
        return self.enabled and self.auto_apply
