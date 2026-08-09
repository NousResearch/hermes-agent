#!/usr/bin/env python3
"""KV-cache placement policy — the single source of truth for the serving mode.

Implements the placement rule in docs/adr/0012-p12-256k-memory-policy.md:

- Ordinary contexts (<= KV_CACHE_OFFLOAD_THRESHOLD, default 128K = 131072
  tokens) use the fast GPU-resident KV cache. This is the default mode.
- Rare long contexts (> threshold, up to 256K = 262144 tokens) use the
  host-RAM KV-cache offload path.
- The policy is consulted by the serving path *before* KV-cache allocation
  and by every launch wrapper. It is the only place that decides the tier;
  consumers (the GPU fast path, the offload gate, the live check) read its
  output and must not duplicate the decision logic.

Configuration (environment variables, or an optional JSON config file):

    KV_CACHE_MODE                 auto (default) | gpu | offload
                                  auto     -> decide by context length
                                  gpu      -> force GPU-resident (testing)
                                  offload  -> force host-RAM offload (testing)
    KV_CACHE_OFFLOAD_THRESHOLD    int tokens, default 131072 (128K)
    KV_CACHE_MAX_CONTEXT          int tokens, default 262144 (256K)
    KV_CACHE_CONFIG               path to a JSON config file with keys
                                  {"mode", "offload_threshold", "max_context"}

Precedence: defaults < config file < environment variables. Invalid values
raise PolicyConfigError; contexts above KV_CACHE_MAX_CONTEXT raise
ContextOutOfRangeError. A per-request ``override`` (gpu/offload) takes
precedence over every configuration source and is intended for tests and
edge cases.

The module is deliberately stdlib-only so it can run as a pure decision
function in tests, CI, or any launch wrapper without importing torch or
the heavy stack.

Usage (serving path, before KV-cache allocation):

    from scripts.kv_cache_policy import decide_tier, KVCacheTier

    decision = decide_tier(context_len=ctx)
    if decision.tier is KVCacheTier.HOST_RAM_OFFLOAD:
        ...  # allocate KV cache in host RAM
    else:
        ...  # allocate KV cache on the GPU (fast path)
"""

import argparse
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Union

# ---------------------------------------------------------------------------
# Defaults / env names
# ---------------------------------------------------------------------------

DEFAULT_MODE = "auto"
DEFAULT_OFFLOAD_THRESHOLD = 131072  # 128K tokens
DEFAULT_MAX_CONTEXT = 262144        # 256K tokens

ENV_MODE = "KV_CACHE_MODE"
ENV_THRESHOLD = "KV_CACHE_OFFLOAD_THRESHOLD"
ENV_MAX_CONTEXT = "KV_CACHE_MAX_CONTEXT"
ENV_CONFIG_FILE = "KV_CACHE_CONFIG"

VALID_MODES = frozenset({"auto", "gpu", "offload"})
VALID_OVERRIDES = frozenset({"gpu", "offload"})

# Backwards-compatible alias for consumers that knew the old gate constant.
# The threshold now lives here (the single source of truth).
OFFLOAD_THRESHOLD_DEFAULT = DEFAULT_OFFLOAD_THRESHOLD
MAX_CONTEXT_DEFAULT = DEFAULT_MAX_CONTEXT


class KVCacheTier(Enum):
    """Storage tier the policy selects for a request's KV cache."""

    GPU_RESIDENT = "gpu"
    HOST_RAM_OFFLOAD = "offload"


class PolicyConfigError(ValueError):
    """Raised when KV-cache policy configuration is invalid (bad value)."""


class ContextOutOfRangeError(ValueError):
    """Raised when a context length exceeds the supported maximum (256K)."""


@dataclass(frozen=True)
class PlacementDecision:
    """Result of evaluating one request against the KV-cache policy."""

    tier: KVCacheTier
    context_len: int
    threshold: int
    max_context: int
    mode: str
    forced: bool
    reason: str

    @property
    def tier_name(self) -> str:
        return self.tier.value

    def to_dict(self) -> Dict[str, object]:
        return {
            "tier": self.tier_name,
            "context_len": self.context_len,
            "threshold": self.threshold,
            "max_context": self.max_context,
            "mode": self.mode,
            "forced": self.forced,
            "reason": self.reason,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"PlacementDecision(tier={self.tier_name!r}, "
            f"context_len={self.context_len}, threshold={self.threshold}, "
            f"max_context={self.max_context}, mode={self.mode!r}, "
            f"forced={self.forced}, reason={self.reason!r})"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_mode(value: str) -> str:
    if value not in VALID_MODES:
        raise PolicyConfigError(
            f"invalid KV_CACHE_MODE {value!r}; expected one of "
            f"{sorted(VALID_MODES)}"
        )
    return value


def _validate_threshold(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise PolicyConfigError(f"{name} must be an integer, got {value!r}")
    if parsed < 1:
        raise PolicyConfigError(f"{name} must be >= 1, got {parsed}")
    return parsed


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config_file(env: Mapping[str, str]) -> Dict[str, object]:
    """Load optional JSON config file (KV_CACHE_CONFIG). Missing/invalid
    file is a hard config error so a typo'd path never silently falls back
    to defaults."""
    raw_path = (env or os.environ).get(ENV_CONFIG_FILE)
    if not raw_path:
        return {}
    try:
        with open(raw_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PolicyConfigError(
            f"cannot read KV_CACHE_CONFIG file {raw_path!r}: {exc}"
        )
    if not isinstance(data, dict):
        raise PolicyConfigError(
            f"KV_CACHE_CONFIG file {raw_path!r} must contain a JSON object"
        )
    result: Dict[str, object] = {}
    if "mode" in data:
        result["mode"] = data["mode"]
    if "offload_threshold" in data:
        result["offload_threshold"] = data["offload_threshold"]
    if "max_context" in data:
        result["max_context"] = data["max_context"]
    return result


def load_config(env: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """Resolve the effective policy configuration.

    Precedence: built-in defaults < config file < environment variables.
    Raises PolicyConfigError on any invalid value.
    """
    env = env if env is not None else os.environ
    env_map: Mapping[str, str] = env
    config: Dict[str, object] = {
        "mode": DEFAULT_MODE,
        "offload_threshold": DEFAULT_OFFLOAD_THRESHOLD,
        "max_context": DEFAULT_MAX_CONTEXT,
        "source": "defaults",
    }
    file_cfg = _load_config_file(env_map)
    if file_cfg:
        config["source"] = "file"
        for key in ("mode", "offload_threshold", "max_context"):
            if key in file_cfg:
                config[key] = file_cfg[key]

    if env_map.get(ENV_MODE):
        config["mode"] = env_map[ENV_MODE]
        config["source"] = "env"
    if env_map.get(ENV_THRESHOLD):
        config["offload_threshold"] = env_map[ENV_THRESHOLD]
        config["source"] = "env"
    if env_map.get(ENV_MAX_CONTEXT):
        config["max_context"] = env_map[ENV_MAX_CONTEXT]
        config["source"] = "env"

    # Validate everything now, in one place.
    mode = _validate_mode(str(config["mode"]))
    threshold = _validate_threshold(
        str(config["offload_threshold"]), ENV_THRESHOLD
    )
    max_context = _validate_threshold(str(config["max_context"]), ENV_MAX_CONTEXT)
    if max_context < threshold:
        raise PolicyConfigError(
            f"KV_CACHE_MAX_CONTEXT ({max_context}) must be >= "
            f"KV_CACHE_OFFLOAD_THRESHOLD ({threshold})"
        )
    return {
        "mode": mode,
        "offload_threshold": threshold,
        "max_context": max_context,
        "source": str(config["source"]),
    }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def decide_tier(
    context_len: Union[int, str],
    *,
    override: Optional[Union[str, KVCacheTier]] = None,
    env: Optional[Dict[str, str]] = None,
) -> PlacementDecision:
    """Decide the KV-cache storage tier for a request.

    Args:
        context_len: Request context length in tokens. 0 is valid (a prompt
            with no prior context) and stays on the GPU fast path. Negative
            values are a config/validation error.
        override: Optional per-request override: ``"gpu"`` or ``"offload"``
            (or KVCacheTier members). Takes precedence over every
            configuration source. Intended for tests and edge cases.
        env: Optional dict to substitute for os.environ in tests.

    Returns:
        A PlacementDecision with the selected tier and the reason.

    Raises:
        PolicyConfigError: invalid configuration or a negative context.
        ContextOutOfRangeError: context_len > KV_CACHE_MAX_CONTEXT.
    """
    config = load_config(env)
    mode: str = str(config["mode"])
    threshold = int(str(config["offload_threshold"]))
    max_context = int(str(config["max_context"]))

    try:
        ctx = int(context_len)
    except (TypeError, ValueError):
        raise PolicyConfigError(f"context_len must be an integer, got {context_len!r}")
    if ctx < 0:
        raise PolicyConfigError(f"context_len must be >= 0, got {ctx}")

    if ctx > max_context:
        raise ContextOutOfRangeError(
            f"context {ctx} exceeds supported maximum {max_context} (256K)"
        )

    # Per-request override wins over everything.
    forced = mode != "auto"
    if override is not None:
        if isinstance(override, KVCacheTier):
            override_value = override.value
        else:
            override_value = str(override).strip().lower()
        if override_value not in VALID_OVERRIDES:
            raise PolicyConfigError(
                f"invalid override {override_value!r}; expected gpu or offload"
            )
        mode = override_value
        forced = True

    if mode == "gpu":
        return PlacementDecision(
            tier=KVCacheTier.GPU_RESIDENT,
            context_len=ctx,
            threshold=threshold,
            max_context=max_context,
            mode=mode,
            forced=forced,
            reason=f"mode={mode} (forced gpu): KV cache GPU-resident for context {ctx}",
        )
    if mode == "offload":
        return PlacementDecision(
            tier=KVCacheTier.HOST_RAM_OFFLOAD,
            context_len=ctx,
            threshold=threshold,
            max_context=max_context,
            mode=mode,
            forced=forced,
            reason=(
                f"mode={mode} (forced offload): KV cache host-RAM offload "
                f"for context {ctx}"
            ),
        )

    # mode == "auto": the context-length rule.
    if ctx <= threshold:
        return PlacementDecision(
            tier=KVCacheTier.GPU_RESIDENT,
            context_len=ctx,
            threshold=threshold,
            max_context=max_context,
            mode=mode,
            forced=False,
            reason=(
                f"context {ctx} <= threshold {threshold}: "
                "GPU-resident fast path"
            ),
        )
    return PlacementDecision(
        tier=KVCacheTier.HOST_RAM_OFFLOAD,
        context_len=ctx,
        threshold=threshold,
        max_context=max_context,
        mode=mode,
        forced=False,
        reason=(
            f"context {ctx} > threshold {threshold} (<= max {max_context}): "
            "host-RAM KV-cache offload"
        ),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="KV-cache placement policy (P12 serving-mode decision)"
    )
    parser.add_argument(
        "--ctx-size", type=int, required=True,
        help="requested context length in tokens",
    )
    parser.add_argument(
        "--override", choices=sorted(VALID_OVERRIDES), default=None,
        help="force a tier for this request (testing/edge cases)",
    )
    args = parser.parse_args(argv)
    try:
        decision = decide_tier(args.ctx_size, override=args.override)
    except (PolicyConfigError, ContextOutOfRangeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
