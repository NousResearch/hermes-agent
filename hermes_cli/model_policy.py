"""Config-driven model policy helpers.

The helpers in this module are deliberately value-free: they compare model
identifiers only and never inspect provider payloads, credentials, prompts,
or environment values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPolicyCheck:
    """Result of checking a model id against configured policy."""

    allowed: bool
    message: str = ""
    required_model: str = ""
    candidate_model: str = ""


def normalize_model_id_for_policy(model_id: Any) -> str:
    """Return the policy comparison form for a model id.

    Provider-prefixed ids are accepted by comparing only the final path
    segment: openai/gpt-5.5 and openrouter/openai/gpt-5.5 normalize
    to gpt-5.5.  The function intentionally does not perform catalog or
    provider lookups.
    """

    value = str(model_id or "").strip().lower()
    if "/" in value:
        value = value.rsplit("/", 1)[-1].strip()
    return value


def _policy_section(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    section = config.get("model_policy")
    return section if isinstance(section, dict) else {}


def fixed_model_from_config(config: dict[str, Any] | None) -> str:
    """Configured fixed model, or an empty string when policy is disabled."""

    fixed = _policy_section(config).get("fixed_model")
    return str(fixed or "").strip()


def forbid_lower_fallback(config: dict[str, Any] | None) -> bool:
    """Whether fallback entries must also match the fixed model."""

    return bool(_policy_section(config).get("forbid_lower_fallback", False))


def check_fixed_model_policy(
    config: dict[str, Any] | None,
    model_id: Any,
    *,
    action: str = "model",
) -> ModelPolicyCheck:
    """Check whether *model_id* satisfies model_policy.fixed_model.

    With no fixed model configured, the policy is inactive and every model is
    allowed.  When active, only the fixed model and provider-prefixed ids
    ending in that fixed model are allowed.
    """

    required = fixed_model_from_config(config)
    if not required:
        return ModelPolicyCheck(allowed=True)

    candidate_display = str(model_id or "").strip()
    candidate_norm = normalize_model_id_for_policy(model_id)
    required_norm = normalize_model_id_for_policy(required)
    if candidate_norm and candidate_norm == required_norm:
        return ModelPolicyCheck(
            allowed=True,
            required_model=required,
            candidate_model=candidate_display,
        )

    shown = candidate_display or "<empty>"
    return ModelPolicyCheck(
        allowed=False,
        message=(
            f"Model policy requires fixed model '{required}'; "
            f"{action} model '{shown}' is not allowed."
        ),
        required_model=required,
        candidate_model=candidate_display,
    )


def check_fallback_model_policy(
    config: dict[str, Any] | None,
    model_id: Any,
) -> ModelPolicyCheck:
    """Check a fallback model only when fallback enforcement is enabled."""

    if not fixed_model_from_config(config) or not forbid_lower_fallback(config):
        return ModelPolicyCheck(allowed=True)
    return check_fixed_model_policy(config, model_id, action="fallback")


def check_config_set_model_policy(
    config: dict[str, Any] | None,
    key: str,
    value: Any,
) -> ModelPolicyCheck:
    """Check hermes config set writes that change the main model id."""

    normalized_key = str(key or "").strip().lower()
    if normalized_key not in {"model", "model.default", "model.model", "model.name"}:
        return ModelPolicyCheck(allowed=True)
    return check_fixed_model_policy(config, value, action="requested")
