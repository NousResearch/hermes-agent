"""Photon and AirHermes configuration and feature flag system.

Provides immutable, strictly validated configuration for discrete multi-bubble streaming,
cadence/pacing controls, and native semantic feature flags.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("gateway.airhermes_config")

# Recognized boolean representations
_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})

_DEFAULT_BUBBLE_STREAMING: bool = True
_DEFAULT_BUBBLE_MIN_INTERVAL: float = 0.85
_DEFAULT_BUBBLE_ADAPTIVE_PACING: bool = True
_DEFAULT_ENABLE_TAPBACKS: bool = True
_DEFAULT_ENABLE_POLLS: bool = True
_DEFAULT_ENABLE_CALENDAR: bool = True
_DEFAULT_ENABLE_MAPS: bool = True
_DEFAULT_ENABLE_EFFECTS: bool = True

_MIN_INTERVAL_LOWER_BOUND: float = 0.2
_MIN_INTERVAL_UPPER_BOUND: float = 5.0


def _parse_strict_bool(value: Any, default: bool, setting_name: str) -> bool:
    """Parse a boolean strictly from bool or recognized string variants.

    Falls back to default with a warning if the value cannot be strictly converted.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    # If None or unrecognized, warn and fall back
    if value is not None:
        logger.warning(
            "[AirHermesConfig] Invalid boolean value %r for '%s', falling back to default %s",
            value,
            setting_name,
            default,
        )
    return default


def _parse_strict_float(
    value: Any,
    default: float,
    min_val: float,
    max_val: float,
    setting_name: str,
) -> float:
    """Parse a float strictly, rejecting NaN/Inf and out-of-range numbers with fallback and warning."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            logger.warning(
                "[AirHermesConfig] Non-finite float %r for '%s', falling back to default %s",
                value,
                setting_name,
                default,
            )
            return default
        if not (min_val <= f <= max_val):
            logger.warning(
                "[AirHermesConfig] Float %r for '%s' out of valid range [%s, %s], falling back to default %s",
                f,
                setting_name,
                min_val,
                max_val,
                default,
            )
            return default
        return f
    except (ValueError, TypeError):
        logger.warning(
            "[AirHermesConfig] Invalid numeric value %r for '%s', falling back to default %s",
            value,
            setting_name,
            default,
        )
        return default


def _resolve_env(env: os._Environ, keys: tuple[str, ...]) -> Optional[str]:
    """Retrieve the first matching environment variable value."""
    for k in keys:
        if k in env:
            return env[k]
    return None


@dataclass(frozen=True)
class AirHermesConfig:
    """Immutable configuration snapshot for bubble streaming runtime and turn execution."""

    bubble_streaming: bool = _DEFAULT_BUBBLE_STREAMING
    bubble_min_interval: float = _DEFAULT_BUBBLE_MIN_INTERVAL
    bubble_adaptive_pacing: bool = _DEFAULT_BUBBLE_ADAPTIVE_PACING
    enable_tapbacks: bool = _DEFAULT_ENABLE_TAPBACKS
    enable_polls: bool = _DEFAULT_ENABLE_POLLS
    enable_calendar: bool = _DEFAULT_ENABLE_CALENDAR
    enable_maps: bool = _DEFAULT_ENABLE_MAPS
    enable_effects: bool = _DEFAULT_ENABLE_EFFECTS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bubble_streaming",
            _parse_strict_bool(self.bubble_streaming, _DEFAULT_BUBBLE_STREAMING, "bubble_streaming"),
        )
        object.__setattr__(
            self,
            "bubble_min_interval",
            _parse_strict_float(
                self.bubble_min_interval,
                _DEFAULT_BUBBLE_MIN_INTERVAL,
                _MIN_INTERVAL_LOWER_BOUND,
                _MIN_INTERVAL_UPPER_BOUND,
                "bubble_min_interval",
            ),
        )
        object.__setattr__(
            self,
            "bubble_adaptive_pacing",
            _parse_strict_bool(
                self.bubble_adaptive_pacing,
                _DEFAULT_BUBBLE_ADAPTIVE_PACING,
                "bubble_adaptive_pacing",
            ),
        )
        object.__setattr__(
            self,
            "enable_tapbacks",
            _parse_strict_bool(self.enable_tapbacks, _DEFAULT_ENABLE_TAPBACKS, "enable_tapbacks"),
        )
        object.__setattr__(
            self,
            "enable_polls",
            _parse_strict_bool(self.enable_polls, _DEFAULT_ENABLE_POLLS, "enable_polls"),
        )
        object.__setattr__(
            self,
            "enable_calendar",
            _parse_strict_bool(self.enable_calendar, _DEFAULT_ENABLE_CALENDAR, "enable_calendar"),
        )
        object.__setattr__(
            self,
            "enable_maps",
            _parse_strict_bool(self.enable_maps, _DEFAULT_ENABLE_MAPS, "enable_maps"),
        )
        object.__setattr__(
            self,
            "enable_effects",
            _parse_strict_bool(self.enable_effects, _DEFAULT_ENABLE_EFFECTS, "enable_effects"),
        )

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]] = None) -> AirHermesConfig:
        """Create configuration from dictionary (e.g. YAML extra config), layered with env vars."""
        raw = dict(d or {})
        env = os.environ

        def _get_val(key: str, env_keys: tuple[str, ...]) -> Any:
            env_val = _resolve_env(env, env_keys)
            if env_val is not None:
                return env_val
            if key in raw:
                return raw[key]
            for sub in ("airhermes", "photon"):
                if isinstance(raw.get(sub), dict) and key in raw[sub]:
                    return raw[sub][key]
            return None

        bubble_streaming = _parse_strict_bool(
            _get_val("bubble_streaming", ("AIRHERMES_BUBBLE_STREAMING", "PHOTON_BUBBLE_STREAMING")),
            _DEFAULT_BUBBLE_STREAMING,
            "bubble_streaming",
        )
        bubble_min_interval = _parse_strict_float(
            _get_val(
                "bubble_min_interval",
                ("AIRHERMES_BUBBLE_MIN_INTERVAL", "AIRHERMES_MIN_INTERVAL", "PHOTON_BUBBLE_MIN_INTERVAL", "PHOTON_MIN_INTERVAL"),
            ),
            _DEFAULT_BUBBLE_MIN_INTERVAL,
            _MIN_INTERVAL_LOWER_BOUND,
            _MIN_INTERVAL_UPPER_BOUND,
            "bubble_min_interval",
        )
        bubble_adaptive_pacing = _parse_strict_bool(
            _get_val(
                "bubble_adaptive_pacing",
                ("AIRHERMES_BUBBLE_ADAPTIVE_PACING", "AIRHERMES_ADAPTIVE_PACING", "PHOTON_BUBBLE_ADAPTIVE_PACING", "PHOTON_ADAPTIVE_PACING"),
            ),
            _DEFAULT_BUBBLE_ADAPTIVE_PACING,
            "bubble_adaptive_pacing",
        )
        enable_tapbacks = _parse_strict_bool(
            _get_val("enable_tapbacks", ("AIRHERMES_ENABLE_TAPBACKS", "PHOTON_ENABLE_TAPBACKS")),
            _DEFAULT_ENABLE_TAPBACKS,
            "enable_tapbacks",
        )
        enable_polls = _parse_strict_bool(
            _get_val("enable_polls", ("AIRHERMES_ENABLE_POLLS", "PHOTON_ENABLE_POLLS")),
            _DEFAULT_ENABLE_POLLS,
            "enable_polls",
        )
        enable_calendar = _parse_strict_bool(
            _get_val("enable_calendar", ("AIRHERMES_ENABLE_CALENDAR", "PHOTON_ENABLE_CALENDAR")),
            _DEFAULT_ENABLE_CALENDAR,
            "enable_calendar",
        )
        enable_maps = _parse_strict_bool(
            _get_val("enable_maps", ("AIRHERMES_ENABLE_MAPS", "PHOTON_ENABLE_MAPS")),
            _DEFAULT_ENABLE_MAPS,
            "enable_maps",
        )
        enable_effects = _parse_strict_bool(
            _get_val("enable_effects", ("AIRHERMES_ENABLE_EFFECTS", "PHOTON_ENABLE_EFFECTS")),
            _DEFAULT_ENABLE_EFFECTS,
            "enable_effects",
        )

        return cls(
            bubble_streaming=bubble_streaming,
            bubble_min_interval=bubble_min_interval,
            bubble_adaptive_pacing=bubble_adaptive_pacing,
            enable_tapbacks=enable_tapbacks,
            enable_polls=enable_polls,
            enable_calendar=enable_calendar,
            enable_maps=enable_maps,
            enable_effects=enable_effects,
        )

    @classmethod
    def from_env(cls) -> AirHermesConfig:
        """Create configuration exclusively from environment variables with defaults."""
        return cls.from_dict({})

    @classmethod
    def from_platform_config(cls, platform_config: Any) -> AirHermesConfig:
        """Extract configuration from Hermes PlatformConfig or dictionary."""
        if isinstance(platform_config, AirHermesConfig):
            return platform_config
        if platform_config is None:
            return cls.from_dict({})
        if hasattr(platform_config, "extra") and isinstance(platform_config.extra, dict):
            return cls.from_dict(platform_config.extra)
        if isinstance(platform_config, dict):
            return cls.from_dict(platform_config)
        return cls.from_dict({})


# Upstream alias for platform configuration
PhotonConfig = AirHermesConfig
