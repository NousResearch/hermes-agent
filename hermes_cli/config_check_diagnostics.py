"""Read-only diagnostics for saved capabilities and semantic policy hazards."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
from typing import Any


def _delivery_policy_diagnostics(config: dict[str, Any]) -> list[str]:
    """Semantic hazards that weaken immutable delegated delivery roles."""
    diagnostics: list[str] = []
    delegation_value = config.get("delegation") or {}
    if not isinstance(delegation_value, dict):
        return ["ERROR: delegation must be a mapping; delivery-role enforcement cannot read this configuration."]
    delegation = delegation_value
    required = delegation.get("require_delivery_role", False)
    auto_approve = delegation.get("subagent_auto_approve", False)
    if not isinstance(required, bool):
        diagnostics.append(
            "ERROR: delegation.require_delivery_role must be true or false; invalid values fail closed at runtime."
        )
    if "role_defaults" in delegation:
        diagnostics.append(
            "ERROR: delegation.role_defaults is not a supported capability override; delivery-role capabilities are "
            "immutable and cannot be broadened in config."
        )
    if auto_approve is True:
        diagnostics.append(
            "WARNING: delegation.subagent_auto_approve automatically approves dangerous commands for subagents; "
            "disable it for untrusted software-delivery workers."
        )

    approvals = config.get("approvals") or {}
    approvals = approvals if isinstance(approvals, dict) else {}
    if any(
        approvals.get(key) == "approve" for key in ("cron_mode", "single_query_mode", "unattended_mode")
    ):
        diagnostics.append(
            "WARNING: an unattended approval mode is 'approve', so dangerous commands can run without a human "
            "approval prompt; use 'deny' for untrusted software-delivery automation."
        )
    allowlist = config.get("command_allowlist") or []

    def _unsafe_permanent_approval(pattern: Any) -> bool:
        if not isinstance(pattern, str):
            return False
        normalized = " ".join(pattern.lower().split())
        if normalized in {"*", "**"}:
            return True
        return bool(
            re.search(r"(?:^|\s)(?:gh|git)\s+\*+(?:\s|$)", normalized)
            or re.search(r"(?:^|\s)gh\s+pr\s+(?:merge|review|close)(?:\s|$)", normalized)
            or re.search(r"(?:^|\s)gh\s+issue\s+close(?:\s|$)", normalized)
            or re.search(r"(?:^|\s)gh\s+api(?:\s|$)", normalized)
            or re.search(r"(?:^|\s)git\s+(?:merge|push\b.*--force(?:-with-lease)?)(?:\s|$)", normalized)
        )

    if isinstance(allowlist, list) and any(
        _unsafe_permanent_approval(pattern) for pattern in allowlist
    ):
        diagnostics.append(
            "WARNING: command_allowlist contains an unsafe permanent approval pattern that bypasses dangerous-command "
            "approval prompts; narrow or remove it."
        )

    agent_cfg = config.get("agent") or {}
    configured = config.get("prefill_messages_file") or (
        agent_cfg.get("prefill_messages_file") if isinstance(agent_cfg, dict) else ""
    )
    if isinstance(configured, str) and configured.strip():
        try:
            from hermes_constants import get_hermes_home

            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = get_hermes_home() / path
            if path.is_file() and path.stat().st_size <= 1_000_000:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                text = json.dumps(payload, ensure_ascii=False).lower()
                role_terms = (
                    "immutable delivery role",
                    "closure controller",
                    "independent reviewer",
                    "must not merge",
                )
                if any(term in text for term in role_terms):
                    diagnostics.append(
                        "WARNING: prefill_messages_file appears to encode software-delivery policy as fabricated "
                        "dialogue; use delegate_task.delivery_role and acceptance_ledger instead."
                    )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return diagnostics


def config_check_diagnostics(config: dict[str, Any], get_env_value: Callable[[str], str | None]) -> list[str]:
    """Report saved selections and delivery-policy hazards without exposing values."""
    from hermes_cli.config import _platform_manifest_env_entries, _platform_plugin_manifests
    from hermes_cli.plugins_discovery import _get_disabled_plugins
    from hermes_cli.toolset_validation import saved_toolset_resolver, validate_platform_toolsets

    diagnostics = validate_platform_toolsets(config.get("platform_toolsets"), saved_toolset_resolver(config))

    # The environment override is the highest-precedence prefill setting. Feed only its path into
    # the semantic scanner; diagnostics never print file contents or secret values.
    prefill_override = get_env_value("HERMES_PREFILL_MESSAGES_FILE")
    if prefill_override:
        config = dict(config)
        config["prefill_messages_file"] = prefill_override

    disabled = _get_disabled_plugins()
    for name, manifest in _platform_plugin_manifests(source="bundled"):
        key = f"platforms/{name}"
        if not {key, str(manifest.get("name"))} & disabled:
            continue
        required = [env for env, _secret, _meta in _platform_manifest_env_entries(manifest, optional=False)]
        if required and all(get_env_value(env) for env in required):
            diagnostics.append(
                f"platform plugin '{key}' is disabled while its required credentials are configured. "
                f"Run `hermes plugins enable {key}` if you want it active."
            )
    diagnostics.extend(_delivery_policy_diagnostics(config))
    return diagnostics


def emit_config_check_diagnostics(diagnostics: list[str], color, colors) -> bool:
    """Print saved-config diagnostics and report whether any are blocking."""

    if not diagnostics:
        return False
    print()
    print(color("  Saved configuration:", colors.BOLD))
    for diagnostic in diagnostics:
        is_error = diagnostic.startswith("ERROR:")
        marker = "✗" if is_error else "⚠"
        print(color(f"    {marker} {diagnostic}", colors.RED if is_error else colors.YELLOW))
    return any(diagnostic.startswith("ERROR:") for diagnostic in diagnostics)
