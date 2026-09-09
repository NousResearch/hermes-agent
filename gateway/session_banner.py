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


def session_identifier(session_id, resolve):
    """Return a verified unambiguous prefix, or the full ID if lookup is unavailable."""
    if resolve is not None:
        try:
            for length in range(8, len(session_id)):
                prefix = session_id[:length]
                if not prefix.isdigit() and resolve(prefix) == session_id:
                    return prefix
        except Exception:
            pass
    return session_id


def format_reset_settings(runner, model="") -> str:
    """Describe profile defaults without constructing an agent or resolving credentials."""
    from hermes_cli.config import load_config_readonly
    from tools.approval import _YOLO_MODE_FROZEN
    from tools.approval_context import _get_approval_mode

    config = load_config_readonly()
    delegation = config.get("delegation", {})
    malformed_delegation = not isinstance(delegation, dict)
    if malformed_delegation:
        delegation = {}
    profile = profile_name_for_home(get_hermes_home()) or "unknown"
    reasoning = runner._load_reasoning_config(model) if model is not None else None
    tier = runner._load_service_tier() or "default (normal)"
    # Match delegation's blank-value handling without invoking its credential
    # resolver. A provider override may supply its own model (including saved
    # custom-provider defaults), so absence of delegation.model isn't inheritance.
    delegation_model = str(delegation.get("model") or "").strip()
    delegation_provider = str(delegation.get("provider") or "").strip()
    if delegation_model:
        model_label = f"{delegation_model} (configured)"
    elif delegation_provider:
        model_label = "unknown (provider override)"
    else:
        model_label = "inherited from main"
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
        f"◆ Main reasoning: {_effort_label(reasoning) if model is not None else 'unknown (route unavailable)'}",
        f"◆ Service tier (requested): {tier}",
        "◆ Delegation default: unknown" if malformed_delegation else
        f"◆ Delegation default — model: {model_label}; effort: {effort}",
        f"◆ Tool approval: {approval}",
    ])
