"""Strict configuration for the additive Hermes Tag kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError
from .model import ContinuityMode, Sensitivity


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(
            f"{name} contains unknown fields: {', '.join(unknown)}"
        )


def _bool(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float(
    value: Any,
    name: str,
    default: float | None,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return result


def _text(value: Any, name: str, default: str | None = None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ContextConfig:
    enabled: bool = False
    max_chars: int = 12000
    sensitivity_ceiling: Sensitivity = Sensitivity.INTERNAL
    max_facts: int = 64


@dataclass(frozen=True, slots=True)
class ContinuityConfig:
    enabled: bool = True
    mode: ContinuityMode = ContinuityMode.ISOLATED
    max_hops: int = 4


@dataclass(frozen=True, slots=True)
class LeaseConfig:
    ttl_seconds: int = 120
    clock_skew_seconds: int = 5
    signing_secret_ref: str | None = None


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    hourly_tokens: int | None = None
    daily_tokens: int | None = None
    hourly_cost_usd: float | None = None
    daily_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class HermesTagConfig:
    """Process configuration. Defaults are additive and behavior-preserving."""

    enabled: bool = False
    shadow: bool = True
    allow_guests: bool = True
    database_filename: str = "hermes-tag.db"
    context: ContextConfig = field(default_factory=ContextConfig)
    continuity: ContinuityConfig = field(default_factory=ContinuityConfig)
    leases: LeaseConfig = field(default_factory=LeaseConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "HermesTagConfig":
        data = _mapping(raw, "hermes_tag")
        forbidden = {"signing_secret", "hmac_key", "secret"} & set(data)
        if forbidden:
            raise ConfigurationError(
                "Hermes Tag signing material must be resolved by reference, not stored inline"
            )
        _reject_unknown(
            data,
            {
                "enabled",
                "shadow",
                "allow_guests",
                "database_filename",
                "context",
                "continuity",
                "leases",
                "budgets",
            },
            "hermes_tag",
        )

        context_raw = _mapping(data.get("context"), "hermes_tag.context")
        _reject_unknown(
            context_raw,
            {"enabled", "max_chars", "sensitivity_ceiling", "max_facts"},
            "hermes_tag.context",
        )
        sensitivity_value = context_raw.get("sensitivity_ceiling", "internal")
        try:
            sensitivity = Sensitivity.coerce(sensitivity_value)
        except (KeyError, ValueError, TypeError) as exc:
            raise ConfigurationError(
                "hermes_tag.context.sensitivity_ceiling is invalid"
            ) from exc
        context = ContextConfig(
            enabled=_bool(context_raw.get("enabled"), "hermes_tag.context.enabled", False),
            max_chars=_int(
                context_raw.get("max_chars"),
                "hermes_tag.context.max_chars",
                12000,
                256,
                100000,
            ),
            sensitivity_ceiling=sensitivity,
            max_facts=_int(
                context_raw.get("max_facts"),
                "hermes_tag.context.max_facts",
                64,
                1,
                1000,
            ),
        )

        continuity_raw = _mapping(data.get("continuity"), "hermes_tag.continuity")
        _reject_unknown(
            continuity_raw,
            {"enabled", "mode", "max_hops"},
            "hermes_tag.continuity",
        )
        try:
            mode = ContinuityMode(continuity_raw.get("mode", "isolated"))
        except ValueError as exc:
            raise ConfigurationError("hermes_tag.continuity.mode is invalid") from exc
        continuity = ContinuityConfig(
            enabled=_bool(
                continuity_raw.get("enabled"),
                "hermes_tag.continuity.enabled",
                True,
            ),
            mode=mode,
            max_hops=_int(
                continuity_raw.get("max_hops"),
                "hermes_tag.continuity.max_hops",
                4,
                0,
                32,
            ),
        )

        leases_raw = _mapping(data.get("leases"), "hermes_tag.leases")
        if {"signing_secret", "hmac_key", "secret"} & set(leases_raw):
            raise ConfigurationError(
                "hermes_tag.leases may contain signing_secret_ref only"
            )
        _reject_unknown(
            leases_raw,
            {"ttl_seconds", "clock_skew_seconds", "signing_secret_ref"},
            "hermes_tag.leases",
        )
        leases = LeaseConfig(
            ttl_seconds=_int(
                leases_raw.get("ttl_seconds"),
                "hermes_tag.leases.ttl_seconds",
                120,
                5,
                3600,
            ),
            clock_skew_seconds=_int(
                leases_raw.get("clock_skew_seconds"),
                "hermes_tag.leases.clock_skew_seconds",
                5,
                0,
                60,
            ),
            signing_secret_ref=_text(
                leases_raw.get("signing_secret_ref"),
                "hermes_tag.leases.signing_secret_ref",
            ),
        )

        budget_raw = _mapping(data.get("budgets"), "hermes_tag.budgets")
        _reject_unknown(
            budget_raw,
            {
                "hourly_tokens",
                "daily_tokens",
                "hourly_cost_usd",
                "daily_cost_usd",
            },
            "hermes_tag.budgets",
        )
        budgets = BudgetConfig(
            hourly_tokens=(
                _int(
                    budget_raw.get("hourly_tokens"),
                    "hermes_tag.budgets.hourly_tokens",
                    0,
                    1,
                    10**12,
                )
                if budget_raw.get("hourly_tokens") is not None
                else None
            ),
            daily_tokens=(
                _int(
                    budget_raw.get("daily_tokens"),
                    "hermes_tag.budgets.daily_tokens",
                    0,
                    1,
                    10**12,
                )
                if budget_raw.get("daily_tokens") is not None
                else None
            ),
            hourly_cost_usd=_float(
                budget_raw.get("hourly_cost_usd"),
                "hermes_tag.budgets.hourly_cost_usd",
                None,
                0.000001,
                10**9,
            ),
            daily_cost_usd=_float(
                budget_raw.get("daily_cost_usd"),
                "hermes_tag.budgets.daily_cost_usd",
                None,
                0.000001,
                10**9,
            ),
        )

        filename = _text(
            data.get("database_filename"),
            "hermes_tag.database_filename",
            "hermes-tag.db",
        )
        assert filename is not None
        if (
            Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
        ):
            raise ConfigurationError("database_filename must be a plain filename")

        return cls(
            enabled=_bool(data.get("enabled"), "hermes_tag.enabled", False),
            shadow=_bool(data.get("shadow"), "hermes_tag.shadow", True),
            allow_guests=_bool(data.get("allow_guests"), "hermes_tag.allow_guests", True),
            database_filename=filename,
            context=context,
            continuity=continuity,
            leases=leases,
            budgets=budgets,
        )


def profile_state_directory(hermes_home: Path, profile: str) -> Path:
    """Match Hermes' per-profile physical-store layout."""
    root = Path(hermes_home).expanduser().resolve()
    clean_profile = profile.strip()
    if not clean_profile or any(part in clean_profile for part in ("/", "\\", "..")):
        raise ConfigurationError("profile is not a canonical profile name")
    if clean_profile == "default":
        return root
    target = (root / "profiles" / clean_profile).resolve()
    if root != target and root not in target.parents:
        raise ConfigurationError("profile state directory escapes Hermes home")
    return target


def database_path(hermes_home: Path, profile: str, config: HermesTagConfig) -> Path:
    return profile_state_directory(hermes_home, profile) / config.database_filename
