"""Cost-aware model routing for the Hermes Hub agent (Jarvis Layer I).

Picks the cheapest viable Claude tier for each task and escalates to a stronger
model only when the task looks hard, ambiguous, security-sensitive or
self-modifying. Every decision is logged (bounded, in memory) so thresholds can
be tuned against real workloads, and deep-tier calls are rate-capped so a
looping caller can't run up cost.

The routing table is plain data — edit it here, or override per tier with the
HERMES_HUB_MODEL_FAST / _CORE / _DEEP env vars. Setting HERMES_HUB_MODEL pins
every tier to one model (back-compat with the pre-router single-model setup).
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque

# Tier → model id. Defaults track the current Claude line-up.
_TIER_DEFAULTS = {
    "fast": "claude-haiku-4-5-20251001",
    "core": "claude-sonnet-5",
    "deep": "claude-opus-4-8",
}
_TIER_ENV = {"fast": "HERMES_HUB_MODEL_FAST", "core": "HERMES_HUB_MODEL_CORE",
             "deep": "HERMES_HUB_MODEL_DEEP"}


def env_override(tier: str) -> str | None:
    return os.environ.get(_TIER_ENV[tier]) or None


def effective_tiers(overrides: dict | None = None) -> dict:
    """Resolve each tier with precedence: env var > file override > default."""
    out = {}
    for tier, default in _TIER_DEFAULTS.items():
        out[tier] = env_override(tier) or (overrides or {}).get(tier) or default
    return out


# Back-compat: the env-or-default table (no file overrides), resolved at import.
TIERS = effective_tiers()

# Default tier per task type. Editable — this is the "routing table".
TASK_TIERS = {
    "summarize": "fast",   # extractive-ish, cheap
    "chat": "core",        # default conversation + tool loops
    "briefing": "core",    # moderate synthesis
    "reflection": "deep",  # self-evolution analysis (future Layer G)
    "advisor": "deep",     # scoped escalation guidance (future Phase 5)
}

# A chat turn that matches any of these jumps straight to the deep tier: hard
# reasoning, architecture, security-sensitive, financial judgment, or anything
# that would change the agent's own configuration.
_ESCALATE = re.compile(
    r"\b("
    r"architect(?:ure|ing)?|design\s+(?:a|the)\s+system|trade[- ]?offs?|"
    r"security|vulnerab|exploit|threat\s+model|"
    r"invest|portfolio|tax(?:es)?|legal|lawsuit|contract\s+review|"
    r"self[- ]?(?:modify|evolve|improve)|your\s+(?:own\s+)?(?:prompt|config|permission|routing)|"
    r"prove|derivation|complexity\s+analysis|debug\s+this\s+race"
    r")\b",
    re.IGNORECASE,
)

MAX_LOG = 200


class Router:
    """Stateless-ish routing decisions + a bounded decision log.

    `pin` (from HERMES_HUB_MODEL) forces one model for every tier while still
    recording what tier *would* have been chosen, so cost analysis stays honest.
    """

    def __init__(self, pin: str | None = None, max_deep_per_hour: int = 30,
                 overrides: dict | None = None) -> None:
        self.pin = pin
        self.max_deep_per_hour = max_deep_per_hour
        self._overrides = dict(overrides or {})
        self.tiers = effective_tiers(self._overrides)
        self._deep_calls: deque[float] = deque()
        self._log: deque[dict] = deque(maxlen=MAX_LOG)
        self._lock = threading.Lock()

    def set_overrides(self, overrides: dict | None) -> dict:
        """Adopt file-level per-tier overrides (env still wins). Returns tiers."""
        with self._lock:
            self._overrides = dict(overrides or {})
            self.tiers = effective_tiers(self._overrides)
            return dict(self.tiers)

    def env_locked(self) -> dict:
        """Which tiers are pinned by an env var (file overrides can't move them)."""
        return {tier: env_override(tier) is not None for tier in _TIER_DEFAULTS}

    # -- core decision -------------------------------------------------------
    def classify(self, task_type: str, text: str = "") -> tuple[str, str]:
        """Return (tier, reason) before any cost-cap adjustment."""
        base = TASK_TIERS.get(task_type, "core")
        if task_type == "chat" and text and _ESCALATE.search(text):
            return "deep", "escalate:pattern"
        return base, f"default:{task_type}"

    def route(self, task_type: str, text: str = "") -> dict:
        """Pick a model for this task. Returns a decision dict:
        {model, tier, requested_tier, reason, pinned}.
        """
        tier, reason = self.classify(task_type, text)
        requested = tier
        with self._lock:
            if tier == "deep" and not self._deep_budget_ok_locked():
                tier, reason = "core", reason + "+capped"
            if tier == "deep":
                self._deep_calls.append(time.monotonic())
            decision = {
                "task": task_type,
                "tier": tier,
                "requested_tier": requested,
                "reason": reason,
                "pinned": self.pin is not None,
                "model": self.pin or self.tiers[tier],
                "at": time.time(),
            }
            self._log.append(decision)
        return decision

    # -- deep-tier rate cap --------------------------------------------------
    def _deep_budget_ok_locked(self) -> bool:
        cutoff = time.monotonic() - 3600
        while self._deep_calls and self._deep_calls[0] < cutoff:
            self._deep_calls.popleft()
        return len(self._deep_calls) < self.max_deep_per_hour

    def deep_calls_last_hour(self) -> int:
        with self._lock:
            cutoff = time.monotonic() - 3600
            while self._deep_calls and self._deep_calls[0] < cutoff:
                self._deep_calls.popleft()
            return len(self._deep_calls)

    # -- introspection for the status widget --------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            recent = list(self._log)[-10:]
        return {
            "tiers": dict(self.tiers),
            "overrides": dict(self._overrides),
            "env_locked": self.env_locked(),
            "defaults": dict(_TIER_DEFAULTS),
            "pinned": self.pin,
            "deep_calls_last_hour": self.deep_calls_last_hour(),
            "deep_cap_per_hour": self.max_deep_per_hour,
            "recent": recent,
        }
