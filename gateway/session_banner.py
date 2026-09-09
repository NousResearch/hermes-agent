"""Read-only settings for reset notices; call inside the serving profile scope.

These are session defaults, not a prediction of provider wire clamping, fallback
routing, named-profile workers, or per-task delegation overrides.
"""
from hermes_cli import __version__
from hermes_constants import get_hermes_home, parse_reasoning_effort, profile_name_for_home


def _effort_label(reasoning):
    if not isinstance(reasoning, dict):
        return "default"
    if reasoning.get("enabled") is False:
        return "off"
    return reasoning.get("effort") or "default"


def format_reset_settings(runner) -> str:
    """Describe profile defaults without constructing an agent or resolving credentials."""
    from hermes_cli.config import load_config_readonly
    from tools.approval import _YOLO_MODE_FROZEN
    from tools.approval_context import _get_approval_mode

    config = load_config_readonly()
    delegation = config.get("delegation") or {}
    profile = profile_name_for_home(get_hermes_home()) or "unknown"
    reasoning = runner._load_reasoning_config()
    tier = runner._load_service_tier() or "default (normal)"
    model = delegation.get("model")
    model_label = f"{model} (configured)" if model else "inherited from main"
    raw_effort = delegation.get("reasoning_effort")
    parsed = parse_reasoning_effort(raw_effort)
    effort = "inherited from main"
    if parsed is not None:
        effort = f"{_effort_label(parsed)} (configured)"
    elif raw_effort:
        effort += " (invalid setting)"
    approval = "off (runtime override)" if _YOLO_MODE_FROZEN else _get_approval_mode()
    return "\n".join([
        f"◆ Profile: {profile} · Hermes {__version__}",
        f"◆ Main reasoning: {_effort_label(reasoning)}",
        f"◆ Service tier (requested): {tier}",
        f"◆ Delegation default — model: {model_label}; effort: {effort}",
        f"◆ Tool approval: {approval}",
    ])
