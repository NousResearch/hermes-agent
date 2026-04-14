"""
Adaptive Delegation Policy — Issue #9557

Implements bandit-based (Thompson Sampling / UCB / epsilon-greedy) model
selection for delegate_task.  The policy maps task features
(e.g. task_kind, repo_fingerprint, rough_token_estimate, parent_model_tier)
to a recommended dispatch action (local vs. fast-worker vs. strong-worker).

At call time the orchestrator asks the policy "given these features, which
dispatch wins?"  At completion it records (features, dispatch, wall_time_ms,
cost_tokens, succeeded).  The policy updates its posterior so that
similar future tasks get better picks.

Non-goals (per issue #9557):
- NOT replacing the agent's in-context reasoning — policy output is a prior,
  the agent can still override
- NOT a training pipeline — pure bandits over a small discrete action set,
  no GPU
- NOT a new backend — persists via JSONL in the hermes profile directory
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from pathlib import Path
import os

def _get_hermes_home() -> str:
    return os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

DISPATCH_LOCAL = ("local", None)
DISPATCH_FAST = ("delegate", "fast-worker")
DISPATCH_STRONG = ("delegate", "strong-worker")

ALL_ARMS = [DISPATCH_LOCAL, DISPATCH_FAST, DISPATCH_STRONG]
ARM_LABELS = {
    DISPATCH_LOCAL: "local",
    DISPATCH_FAST: "fast-worker",
    DISPATCH_STRONG: "strong-worker",
}


@dataclass
class DelegationOutcome:
    """Single delegation result written to the policy store."""

    bucket: str  # hashed feature bucket, e.g. "typo-fix|abc123|1k|claude-opus"
    arm: tuple[str, str | None]  # e.g. ("delegate", "fast-worker")
    wall_time_ms: int
    cost_tokens: int
    succeeded: bool
    task_kind: str = ""
    parent_model_tier: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ArmState:
    """Thompson Sampling posterior state for one arm in one bucket."""

    successes: int = 0
    failures: int = int(os.environ.get("DELEGATION_POLICY_EPS", "0"))
    pulls: int = 0

    def sample(self) -> float:
        """Sample from Beta(successes+1, failures+1)."""
        return np.random.beta(self.successes + 1, self.failures + 1)

    def update(self, succeeded: bool) -> None:
        self.pulls += 1
        if succeeded:
            self.successes += 1
        else:
            self.failures += 1


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Token estimate buckets
_TOKEN_BUCKETS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]


def _bucket_tokens(rough_estimate: int | None) -> str:
    if rough_estimate is None:
        return "unknown"
    for b in _TOKEN_BUCKETS:
        if rough_estimate <= b:
            return f"lte{b}"
    return f"gt{_TOKEN_BUCKETS[-1]}"


def _tokenise_goal(goal: str) -> str:
    """Coarse keyword-based task kind from goal text."""
    g = goal.lower()
    if any(k in g for k in ("typo", "fix", "bug", "error", "assert")):
        return "typo-fix"
    if any(k in g for k in ("refactor", "rename", "extract", "move", "inline")):
        return "refactor"
    if any(k in g for k in ("test", "spec", "coverage", "pytest", "unittest")):
        return "testing"
    if any(k in g for k in ("read", "find", "search", "grep", "list")):
        return "read-only"
    if any(k in g for k in ("write", "create", "new", "add")):
        return "create"
    if any(k in g for k in ("explain", "why", "what", "how", "document")):
        return "analysis"
    if any(k in g for k in ("deploy", "build", "run", "install", "setup")):
        return "devops"
    return "general"


def _repo_fingerprint(repo_path: str | None) -> str:
    """Cheap repo fingerprint: first 8 chars of a hash of the git remote."""
    if not repo_path:
        return "no-repo"
    try:
        import subprocess

        remote = subprocess.check_output(
            ["git", "-C", repo_path, "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        h = hash(remote.strip().decode("utf-8", errors="replace"))
        return f"{abs(h):08x}"
    except Exception:
        return "no-repo"


def extract_bucket(
    goal: str,
    repo_path: str | None = None,
    rough_token_estimate: int | None = None,
    parent_model_tier: str = "",
) -> str:
    """Map features to a policy bucket string."""
    kind = _tokenise_goal(goal)
    fp = _repo_fingerprint(repo_path)
    tokens = _bucket_tokens(rough_token_estimate)
    tier = parent_model_tier or "unknown"
    return f"{kind}|{fp}|{tokens}|{tier}"


# ---------------------------------------------------------------------------
# DelegationPolicy ABC
# ---------------------------------------------------------------------------

_policy_lock = threading.RLock()
_policy: Optional[DelegationPolicy] = None
_policy_loaded: bool = False


class DelegationPolicy(ABC):
    """Abstract base for all delegation policies."""

    @abstractmethod
    def pick(
        self,
        bucket: str,
        available_arms: list[tuple[str, str | None]] | None = None,
    ) -> tuple[str, str | None]:
        """Return the dispatch (mode, profile) the policy recommends."""
        ...

    @abstractmethod
    def record(self, outcome: DelegationOutcome) -> None:
        """Record a delegation outcome so the policy can learn."""
        ...

    def shutdown(self) -> None:
        """Flush state to disk (called on agent exit)."""
        ...


# ---------------------------------------------------------------------------
# Thompson Sampling
# ---------------------------------------------------------------------------

_POLICY_FILE = "delegation_policy.jsonl"


def _policy_path() -> Path:
    return Path(_get_hermes_home()) / _POLICY_FILE


class ThompsonSamplingPolicy(DelegationPolicy):
    """
    Thompson Sampling over dispatch arms, scoped per feature-bucket.

    Each (bucket, arm) pair maintains a Beta posterior.
    On pick(): sample from each arm's posterior, return argmax.
    On record(): update the sampled arm's posterior with the observed success.

    Thread-safe via _lock.
    """

    def __init__(self) -> None:
        # {bucket: {arm_label: ArmState}}
        self._state: dict[str, dict[str, ArmState]] = {}
        self._total_pulls: int = 0
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load outcomes from JSONL on disk (cold start)."""
        p = _policy_path()
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        outcome = DelegationOutcome(**obj)
                        self._apply_outcome(outcome, persist=False)
                    except Exception:
                        continue
        except Exception:
            pass

    def _apply_outcome(self, outcome: DelegationOutcome, persist: bool = True) -> None:
        """Apply a single outcome to the posterior without acquiring lock."""
        arm_key = ARM_LABELS.get(outcome.arm, str(outcome.arm))
        bucket_state = self._state.setdefault(outcome.bucket, {})
        arm_state = bucket_state.setdefault(arm_key, ArmState())
        arm_state.update(outcome.succeeded)
        self._total_pulls += 1

        if persist:
            try:
                p = _policy_path()
                with open(p, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")
            except Exception:
                pass

    def shutdown(self) -> None:
        """No in-memory state to flush beyond the JSONL append-only log."""
        pass

    # ── core interface ──────────────────────────────────────────────────────

    def pick(
        self,
        bucket: str,
        available_arms: list[tuple[str, str | None]] | None = None,
    ) -> tuple[str, str | None]:
        arms = available_arms if available_arms is not None else ALL_ARMS
        with _policy_lock:
            bucket_state = self._state.get(bucket, {})
            best_arm = DISPATCH_LOCAL
            best_score = -1.0

            for arm in arms:
                arm_key = ARM_LABELS.get(arm, str(arm))
                arm_state = bucket_state.get(arm_key, ArmState())
                score = arm_state.sample()
                if score > best_score:
                    best_score = score
                    best_arm = arm

            return best_arm

    def record(self, outcome: DelegationOutcome) -> None:
        with _policy_lock:
            self._apply_outcome(outcome, persist=True)


# ---------------------------------------------------------------------------
# Epsilon-Greedy (simpler alternative)
# ---------------------------------------------------------------------------

class EpsilonGreedyPolicy(DelegationPolicy):
    """
    Epsilon-greedy: with probability epsilon pick randomly;
    otherwise pick the arm with the highest empirical success rate.

    epsilon: fraction of random exploration (0.0 = pure exploitation)
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self._state: dict[str, dict[str, ArmState]] = {}
        self._total_pulls: int = 0
        self._load()

    def _load(self) -> None:
        p = _policy_path()
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        outcome = DelegationOutcome(**json.loads(line))
                        self._apply_outcome(outcome, persist=False)
                    except Exception:
                        continue
        except Exception:
            pass

    def _apply_outcome(self, outcome: DelegationOutcome, persist: bool = True) -> None:
        arm_key = ARM_LABELS.get(outcome.arm, str(outcome.arm))
        bucket_state = self._state.setdefault(outcome.bucket, {})
        arm_state = bucket_state.setdefault(arm_key, ArmState())
        arm_state.update(outcome.succeeded)
        self._total_pulls += 1
        if persist:
            try:
                with open(_policy_path(), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")
            except Exception:
                pass

    def shutdown(self) -> None:
        pass

    def pick(
        self,
        bucket: str,
        available_arms: list[tuple[str, str | None]] | None = None,
    ) -> tuple[str, str | None]:
        arms = available_arms if available_arms is not None else ALL_ARMS
        if random.random() < self.epsilon:
            return random.choice(arms)

        bucket_state = self._state.get(bucket, {})
        best_arm = DISPATCH_LOCAL
        best_rate = -1.0

        for arm in arms:
            arm_key = ARM_LABELS.get(arm, str(arm))
            arm_state = bucket_state.get(arm_key, ArmState())
            rate = (
                arm_state.successes / arm_state.pulls
                if arm_state.pulls > 0
                else 0.5
            )
            if rate > best_rate:
                best_rate = rate
                best_arm = arm

        return best_arm

    def record(self, outcome: DelegationOutcome) -> None:
        with _policy_lock:
            self._apply_outcome(outcome, persist=True)


# ---------------------------------------------------------------------------
# Static baseline (no-op) — used when policy is disabled
# ---------------------------------------------------------------------------

class StaticPolicy(DelegationPolicy):
    """Always returns the static default, no learning."""

    def pick(
        self,
        bucket: str,
        available_arms: list[tuple[str, str | None]] | None = None,
    ) -> tuple[str, str | None]:
        return DISPATCH_LOCAL

    def record(self, outcome: DelegationOutcome) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Policy factory / global accessor
# ---------------------------------------------------------------------------

_active_policy: DelegationPolicy = StaticPolicy()


def get_policy() -> DelegationPolicy:
    """Return the currently active delegation policy."""
    return _active_policy


def init_policy(
    mode: str = "thompson_sampling",
    epsilon: float = 0.1,
) -> DelegationPolicy:
    """
    Initialise (or reinitialise) the global delegation policy.

    mode: "thompson_sampling" | "epsilon_greedy" | "off"
    epsilon: only used when mode == "epsilon_greedy"
    """
    global _active_policy, _policy_loaded

    with _policy_lock:
        if _policy_loaded:
            _active_policy.shutdown()

        if mode == "thompson_sampling":
            _active_policy = ThompsonSamplingPolicy()
        elif mode == "epsilon_greedy":
            _active_policy = EpsilonGreedyPolicy(epsilon=epsilon)
        else:
            _active_policy = StaticPolicy()

        _policy_loaded = True
        return _active_policy


def record_outcome(
    goal: str,
    arm: tuple[str, str | None],
    succeeded: bool,
    wall_time_ms: int,
    cost_tokens: int,
    repo_path: str | None = None,
    rough_token_estimate: int | None = None,
    parent_model_tier: str = "",
) -> None:
    """Convenience helper to record a delegation outcome."""
    bucket = extract_bucket(goal, repo_path, rough_token_estimate, parent_model_tier)
    outcome = DelegationOutcome(
        bucket=bucket,
        arm=arm,
        wall_time_ms=wall_time_ms,
        cost_tokens=cost_tokens,
        succeeded=succeeded,
        task_kind=_tokenise_goal(goal),
        parent_model_tier=parent_model_tier,
    )
    _active_policy.record(outcome)
