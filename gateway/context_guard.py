"""Gateway context guard helpers.

The gateway can run for many turns on chat platforms.  If a session grows too
large, the model/host may compact or fail before the user gets a clean handoff.
These helpers resolve a small, opt-in policy that lets the gateway finish the
current turn, then start the next turn on a fresh session once usage crosses a
configured percentage of the context window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ContextGuardConfig:
    """Resolved gateway context guard policy."""

    enabled: bool = False
    threshold: float = 0.60
    auto_reset_next_turn: bool = True
    append_notice: bool = True

    @property
    def threshold_percent(self) -> int:
        return int(round(self.threshold * 100))


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    # Accept either fraction form (0.60) or percent form (60).
    if parsed > 1:
        parsed = parsed / 100.0
    # Keep enough headroom to finish the current turn, but avoid silly values.
    return max(0.10, min(parsed, 0.95))


def resolve_context_guard_config(
    user_config: Mapping[str, Any] | None,
    platform_key: str,
) -> ContextGuardConfig:
    """Resolve ``gateway.context_guard`` with optional per-platform overrides.

    Example config::

        gateway:
          context_guard:
            enabled: true
            threshold: 0.60
            auto_reset_next_turn: true
            platforms:
              telegram:
                enabled: true
                threshold: 0.55
    """

    cfg = user_config or {}
    gateway_cfg = cfg.get("gateway") if isinstance(cfg, Mapping) else {}
    if not isinstance(gateway_cfg, Mapping):
        gateway_cfg = {}

    guard_cfg = gateway_cfg.get("context_guard")
    if not isinstance(guard_cfg, Mapping):
        # Backward/experimental top-level fallback, harmless if absent.
        guard_cfg = cfg.get("context_guard") if isinstance(cfg, Mapping) else {}
    if not isinstance(guard_cfg, Mapping):
        guard_cfg = {}

    merged: dict[str, Any] = dict(guard_cfg)
    platforms = guard_cfg.get("platforms")
    if isinstance(platforms, Mapping):
        platform_override = platforms.get(platform_key)
        if isinstance(platform_override, Mapping):
            merged.update(platform_override)
    merged.pop("platforms", None)

    return ContextGuardConfig(
        enabled=_truthy(merged.get("enabled"), default=False),
        threshold=_as_float(merged.get("threshold"), default=0.60),
        auto_reset_next_turn=_truthy(
            merged.get("auto_reset_next_turn"), default=True
        ),
        append_notice=_truthy(merged.get("append_notice"), default=True),
    )


def should_guard_context(
    *,
    prompt_tokens: int,
    context_length: int,
    config: ContextGuardConfig,
) -> bool:
    """Return True when a completed turn should trigger a fresh next session."""

    if not config.enabled or context_length <= 0 or prompt_tokens <= 0:
        return False
    return prompt_tokens >= int(context_length * config.threshold)


def build_context_guard_notice(config: ContextGuardConfig) -> str:
    """Short user-facing notice for chat platforms."""

    pct = config.threshold_percent
    if config.auto_reset_next_turn:
        return (
            f"\n\n🏴‍☠️ Context is past ~{pct}%. "
            "I saved this turn and will start fresh on your next message."
        )
    return (
        f"\n\n🏴‍☠️ Context is past ~{pct}%. "
        "Best to start a fresh session soon."
    )
