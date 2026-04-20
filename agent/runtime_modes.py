"""Wave 1 runtime-mode schema and built-in defaults.

Runtime modes are a top-level operating-mode concept. They must remain distinct
from route categories, delegation profiles, skills, task contracts, and
archetypes. This module is intentionally schema-only for Wave 1 and does not
implement activation or intent-classification behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


@dataclass(frozen=True)
class RuntimeMode:
    """Canonical Wave 1 runtime-mode definition.

    The schema intentionally models runtime modes as their own concept rather
    than reusing route-category or archetype structures.
    """

    name: str
    description: str
    operating_posture: str
    kind: str = "runtime_mode"


def _require_non_empty_normalized_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must be non-empty")
    if normalized_value != value:
        raise ValueError(f"{field_name} must use canonical normalized form")
    return normalized_value


def validate_runtime_mode(mode: RuntimeMode) -> RuntimeMode:
    if not isinstance(mode, RuntimeMode):
        raise ValueError("runtime mode must be a RuntimeMode instance")

    _require_non_empty_normalized_string(mode.name, "name")
    _require_non_empty_normalized_string(mode.description, "description")
    _require_non_empty_normalized_string(mode.operating_posture, "operating_posture")

    if mode.kind != "runtime_mode":
        raise ValueError("kind must equal runtime_mode")

    return mode


def validate_runtime_modes(
    modes: tuple[RuntimeMode, ...], *, default_name: str
) -> tuple[RuntimeMode, ...]:
    normalized_default_name = _require_non_empty_normalized_string(default_name, "default_name")
    seen_names: set[str] = set()

    for mode in modes:
        validate_runtime_mode(mode)
        if mode.name in seen_names:
            raise ValueError(f"duplicate runtime mode name: {mode.name}")
        seen_names.add(mode.name)

    if normalized_default_name not in seen_names:
        raise ValueError(f"default runtime mode is not registered: {normalized_default_name}")

    return modes


DEFAULT_RUNTIME_MODE_NAME: Final[str] = "default"

BUILTIN_RUNTIME_MODES: Final[tuple[RuntimeMode, ...]] = validate_runtime_modes(
    (
        RuntimeMode(
            name="default",
            description="Standard Hermes operating posture with no specialized runtime bias.",
            operating_posture="balanced_general_operation",
        ),
        RuntimeMode(
            name="ultrawork",
            description="High-focus execution posture reserved as a runtime-mode schema default.",
            operating_posture="high_intensity_execution",
        ),
        RuntimeMode(
            name="ralph",
            description="Bounded Ralph-loop operating posture for structured run-until-complete execution.",
            operating_posture="bounded_progress_execution",
        ),
        RuntimeMode(
            name="interview_planning",
            description="Interview-planning operating posture for structured preparation workflows.",
            operating_posture="structured_preparation",
        ),
        RuntimeMode(
            name="execution_supervisor",
            description="Execution-supervisor operating posture for oversight-oriented sessions.",
            operating_posture="oversight_and_coordination",
        ),
    ),
    default_name=DEFAULT_RUNTIME_MODE_NAME,
)

RUNTIME_MODES_BY_NAME: Final[Mapping[str, RuntimeMode]] = MappingProxyType(
    {mode.name: mode for mode in BUILTIN_RUNTIME_MODES}
)


def list_runtime_modes() -> tuple[RuntimeMode, ...]:
    """Return the built-in Wave 1 runtime modes in declaration order."""

    return BUILTIN_RUNTIME_MODES


def get_default_runtime_mode() -> RuntimeMode:
    """Return the canonical default runtime mode."""

    return RUNTIME_MODES_BY_NAME[DEFAULT_RUNTIME_MODE_NAME]


def resolve_runtime_mode(mode_name: str | None) -> RuntimeMode:
    """Resolve a runtime mode name, falling back to the Wave 1 default.

    Runtime-mode resolution accepts compatibility-safe case/spacing variants so
    runtime posture remains deterministic even when the input arrives from a
    loosely-normalized prompt overlay or delegated child payload.
    """

    if not mode_name:
        return get_default_runtime_mode()

    normalized_name = str(mode_name).strip().lower().replace(" ", "_").replace("-", "_")
    if not normalized_name:
        return get_default_runtime_mode()

    return RUNTIME_MODES_BY_NAME.get(normalized_name, get_default_runtime_mode())


__all__ = [
    "BUILTIN_RUNTIME_MODES",
    "DEFAULT_RUNTIME_MODE_NAME",
    "RUNTIME_MODES_BY_NAME",
    "RuntimeMode",
    "get_default_runtime_mode",
    "list_runtime_modes",
    "resolve_runtime_mode",
    "validate_runtime_mode",
    "validate_runtime_modes",
]
