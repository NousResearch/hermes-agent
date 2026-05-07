"""Enterprise security posture reporting for Hermes.

Hermes accepts an OpenShell-shaped policy surface where it can map policy
intent onto existing controls. The doctor distinguishes host-enforced controls
from policy-compatible controls that need an external sandbox/proxy runtime
such as OpenShell for true kernel or network-gateway enforcement.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


HERMES_POLICY_SCHEMA: dict[str, dict[str, str]] = {
    "filesystem": {
        "write_safe_root": "string",
        "deny_hermes_control_plane": "boolean",
    },
    "filesystem_policy": {
        "read_only": "list[string]",
        "read_write": "list[string]",
        "include_workdir": "boolean",
        "deny_hermes_control_plane": "boolean",
    },
    "landlock": {
        "compatibility": "best_effort|hard_requirement",
    },
    "commands": {
        "deny": "list[string]",
        "allow": "list[string]",
    },
    "environment": {
        "env_passthrough": "list[string]",
    },
    "process": {
        "run_as_user": "string",
        "run_as_group": "string",
        "approvals_mode": "manual|smart|off",
        "cron_mode": "deny|approve",
        "tirith_enabled": "boolean",
        "tirith_fail_open": "boolean",
        "command_denylist": "list[string]",
        "command_allowlist": "list[string]",
    },
    "network": {
        "allow_private_urls": "boolean",
        "website_blocklist_enabled": "boolean",
    },
    "network_policies": {"*": "network_policy_entry"},
    "permissions": {
        "toolsets_allow": "list[string]",
        "toolsets_deny": "list[string]",
        "tools_allow": "list[string]",
        "tools_deny": "list[string]",
        "mcp_reload_confirm": "boolean",
    },
    "gateway": {
        "require_user_allowlist": "boolean",
        "allow_pairing": "boolean",
        "control_commands_bypass_busy_queue": "boolean",
        "approval_commands_inline": "boolean",
    },
    "inference": {
        "redact_secrets": "boolean",
        "managed_provider_routing": "boolean",
    },
}

OPENSHELL_TOP_LEVEL_KEYS = {
    "version",
    "filesystem_policy",
    "landlock",
    "process",
    "network_policies",
}

GATEWAY_ALLOWLIST_ENV_NAMES = (
    "GATEWAY_ALLOWED_USERS",
    "TELEGRAM_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_USERS",
    "DISCORD_ALLOWED_USERS",
    "DISCORD_GROUP_ALLOWED_USERS",
    "SLACK_ALLOWED_USERS",
    "WHATSAPP_ALLOWED_USERS",
    "SIGNAL_ALLOWED_USERS",
    "SIGNAL_GROUP_ALLOWED_USERS",
    "EMAIL_ALLOWED_SENDERS",
    "SMS_ALLOWED_NUMBERS",
)


@dataclass(frozen=True)
class PostureCheck:
    domain: str
    name: str
    status: str
    detail: str
    recommendation: str = ""
    enforcement: str = "hermes"


def _get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _policy_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(_get(config, "security", "policy", default={}))


def _provider_env_blocklist() -> frozenset[str]:
    try:
        from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

        return _HERMES_PROVIDER_ENV_BLOCKLIST
    except Exception:
        return frozenset({
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "GOOGLE_API_KEY",
            "GITHUB_TOKEN",
            "GH_TOKEN",
        })


def _resolve_safe_roots_source(
    config: Mapping[str, Any], env: Mapping[str, str]
) -> tuple[list[str], str]:
    env_root = str(env.get("HERMES_WRITE_SAFE_ROOT", "")).strip()
    if env_root:
        return [env_root], "HERMES_WRITE_SAFE_ROOT"

    policy_cfg = _policy_config(config)
    legacy_filesystem = _as_mapping(policy_cfg.get("filesystem"))
    cfg_root = str(legacy_filesystem.get("write_safe_root") or "").strip()
    if cfg_root:
        return [cfg_root], "security.policy.filesystem.write_safe_root"

    openshell_filesystem = _as_mapping(policy_cfg.get("filesystem_policy"))
    read_write = _as_list(openshell_filesystem.get("read_write"))
    if read_write:
        return read_write, "security.policy.filesystem_policy.read_write"

    return [], "unset"


def _gateway_allowlist_sources(env: Mapping[str, str]) -> list[str]:
    return sorted(
        name for name in GATEWAY_ALLOWLIST_ENV_NAMES if str(env.get(name, "")).strip()
    )


def build_effective_policy(
    config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return Hermes' effective security posture as a serializable dict."""
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    if env is None:
        env = os.environ

    policy_cfg = _policy_config(config)
    filesystem_cfg = _as_mapping(policy_cfg.get("filesystem"))
    filesystem_policy_cfg = _as_mapping(policy_cfg.get("filesystem_policy"))
    landlock_cfg = _as_mapping(policy_cfg.get("landlock"))
    commands_cfg = _as_mapping(policy_cfg.get("commands"))
    permissions_cfg = _as_mapping(policy_cfg.get("permissions"))
    gateway_policy_cfg = _as_mapping(policy_cfg.get("gateway"))
    inference_policy_cfg = _as_mapping(policy_cfg.get("inference"))

    safe_roots, safe_root_source = _resolve_safe_roots_source(config, env)
    env_passthrough = _as_list(_get(config, "terminal", "env_passthrough", default=[]))
    provider_passthrough = sorted(
        name for name in env_passthrough if name in _provider_env_blocklist()
    )
    website_blocklist = _get(config, "security", "website_blocklist", default={})
    if not isinstance(website_blocklist, Mapping):
        website_blocklist = {}

    approvals_mode = str(_get(config, "approvals", "mode", default="manual") or "manual")
    cron_mode = str(_get(config, "approvals", "cron_mode", default="deny") or "deny")
    terminal_backend = str(_get(config, "terminal", "backend", default="local") or "local")
    network_policies = _as_mapping(policy_cfg.get("network_policies"))
    configured_tool_permissions = any(
        _as_list(permissions_cfg.get(key))
        for key in ("toolsets_allow", "toolsets_deny", "tools_allow", "tools_deny")
    )
    allowlist_sources = _gateway_allowlist_sources(env)
    inference_redact = inference_policy_cfg.get("redact_secrets")
    if not isinstance(inference_redact, bool):
        inference_redact = _get(config, "security", "redact_secrets", default=False)

    return {
        "filesystem": {
            "write_safe_root": safe_roots[0] if safe_roots else "",
            "write_safe_root_source": safe_root_source,
            "read_only_paths": _as_list(filesystem_policy_cfg.get("read_only")),
            "read_write_paths": safe_roots,
            "include_workdir": _as_bool(
                filesystem_policy_cfg.get("include_workdir"), False
            ),
            "deny_hermes_control_plane": _as_bool(
                (
                    filesystem_cfg.get("deny_hermes_control_plane")
                    if "deny_hermes_control_plane" in filesystem_cfg
                    else filesystem_policy_cfg.get("deny_hermes_control_plane")
                ),
                True,
            ),
            "deny_home_credentials": True,
            "landlock_compatibility": str(
                landlock_cfg.get("compatibility") or "not_configured"
            ),
            "kernel_filesystem_enforcement": False,
        },
        "environment": {
            "scrub_provider_credentials": True,
            "env_passthrough": env_passthrough,
            "provider_credentials_in_passthrough": provider_passthrough,
        },
        "process": {
            "hardline_blocklist": True,
            "approvals_mode": approvals_mode,
            "cron_mode": cron_mode,
            "run_as_user": str(
                _as_mapping(policy_cfg.get("process")).get("run_as_user") or ""
            ),
            "run_as_group": str(
                _as_mapping(policy_cfg.get("process")).get("run_as_group") or ""
            ),
            "terminal_backend": terminal_backend,
            "tirith_enabled": _as_bool(
                _get(config, "security", "tirith_enabled", default=True), True
            ),
            "tirith_fail_open": _as_bool(
                _get(config, "security", "tirith_fail_open", default=True), True
            ),
            "command_denylist": _as_list(commands_cfg.get("deny"))
            + _as_list(_as_mapping(policy_cfg.get("process")).get("command_denylist")),
            "command_allowlist": _as_list(_get(config, "command_allowlist", default=[]))
            + _as_list(commands_cfg.get("allow"))
            + _as_list(_as_mapping(policy_cfg.get("process")).get("command_allowlist")),
            "kernel_process_enforcement": False,
        },
        "network": {
            "allow_private_urls": _as_bool(
                _get(config, "security", "allow_private_urls", default=False), False
            ),
            "website_blocklist_enabled": _as_bool(
                website_blocklist.get("enabled"), False
            ),
            "website_blocklist_domains": _as_list(website_blocklist.get("domains")),
            "network_policies": network_policies,
            "policy_entry_count": len(network_policies),
            "gateway_proxy_enforced": False,
        },
        "permissions": {
            "toolsets_allow": _as_list(permissions_cfg.get("toolsets_allow")),
            "toolsets_deny": _as_list(permissions_cfg.get("toolsets_deny")),
            "tools_allow": _as_list(permissions_cfg.get("tools_allow")),
            "tools_deny": _as_list(permissions_cfg.get("tools_deny")),
            "configured": configured_tool_permissions,
            "mcp_reload_confirm": _as_bool(
                _get(config, "approvals", "mcp_reload_confirm", default=True), True
            ),
        },
        "gateway": {
            "require_user_allowlist": _as_bool(
                gateway_policy_cfg.get("require_user_allowlist"), True
            ),
            "allow_pairing": _as_bool(gateway_policy_cfg.get("allow_pairing"), True),
            "allowlist_sources": allowlist_sources,
            "control_commands_bypass_busy_queue": True,
            "approval_commands_inline": True,
        },
        "inference": {
            "redact_secrets": _as_bool(inference_redact, False),
            "provider_credentials_stripped_from_tools": True,
            "managed_provider_routing": _as_bool(
                inference_policy_cfg.get("managed_provider_routing"), False
            ),
        },
    }


def evaluate_posture(policy: Mapping[str, Any]) -> list[PostureCheck]:
    checks: list[PostureCheck] = []

    def add(
        domain: str,
        name: str,
        status: str,
        detail: str,
        recommendation: str = "",
        enforcement: str = "hermes",
    ) -> None:
        checks.append(
            PostureCheck(domain, name, status, detail, recommendation, enforcement)
        )

    filesystem = policy.get("filesystem", {})
    read_write_paths = filesystem.get("read_write_paths") or []
    safe_root = str(filesystem.get("write_safe_root") or "")
    add(
        "filesystem",
        "read_write_policy",
        "pass" if read_write_paths else "warn",
        (
            "write operations are constrained to: " + ", ".join(read_write_paths)
            if read_write_paths
            else "write operations are not constrained to policy read_write paths"
        ),
        (
            "Set HERMES_WRITE_SAFE_ROOT, security.policy.filesystem.write_safe_root, "
            "or security.policy.filesystem_policy.read_write."
        ),
        "file-tools",
    )
    if safe_root and len(read_write_paths) > 1:
        add(
            "filesystem",
            "multiple_read_write_paths",
            "pass",
            f"{len(read_write_paths)} write roots are enforced by file tools",
            enforcement="file-tools",
        )
    add(
        "filesystem",
        "control_plane_write_deny",
        "pass" if filesystem.get("deny_hermes_control_plane") else "fail",
        "active Hermes control-plane files are write-denied"
        if filesystem.get("deny_hermes_control_plane")
        else "active Hermes control-plane files are not write-denied",
        "Set security.policy.filesystem.deny_hermes_control_plane: true.",
        "file-tools",
    )
    add(
        "filesystem",
        "home_credential_write_deny",
        "pass",
        "home credential paths are statically write-denied",
        enforcement="file-tools",
    )
    add(
        "filesystem",
        "kernel_filesystem_enforcement",
        "warn",
        "Hermes file tools enforce policy paths, but local shell redirection is not kernel-confined",
        (
            "Run Hermes inside OpenShell, Docker, Singularity, or another "
            "sandbox for mandatory filesystem controls."
        ),
        "external-runtime",
    )

    environment = policy.get("environment", {})
    provider_passthrough = environment.get("provider_credentials_in_passthrough") or []
    add(
        "environment",
        "provider_credential_scrubbing",
        "pass",
        "Hermes-managed provider credentials are scrubbed from local subprocesses by default",
        enforcement="subprocess-env",
    )
    add(
        "environment",
        "provider_env_passthrough",
        "fail" if provider_passthrough else "pass",
        "provider credentials allowed through terminal.env_passthrough: "
        + ", ".join(provider_passthrough)
        if provider_passthrough
        else "terminal.env_passthrough does not include Hermes provider credentials",
        "Remove Hermes provider credentials from terminal.env_passthrough.",
        "subprocess-env",
    )

    process = policy.get("process", {})
    approvals_mode = str(process.get("approvals_mode") or "manual").lower()
    add(
        "process",
        "hardline_blocklist",
        "pass" if process.get("hardline_blocklist") else "fail",
        "unconditional hardline command blocklist is enabled"
        if process.get("hardline_blocklist")
        else "unconditional hardline command blocklist is disabled",
        enforcement="terminal-approval",
    )
    add(
        "process",
        "dangerous_command_approvals",
        "fail" if approvals_mode == "off" else "pass",
        f"approvals.mode is {approvals_mode}",
        "Use approvals.mode: manual or smart for enterprise deployments.",
        "terminal-approval",
    )
    add(
        "process",
        "cron_command_approvals",
        (
            "pass"
            if str(process.get("cron_mode") or "deny").lower() == "deny"
            else "warn"
        ),
        f"approvals.cron_mode is {process.get('cron_mode')}",
        "Use approvals.cron_mode: deny for unattended jobs.",
        "terminal-approval",
    )
    add(
        "process",
        "tirith_scanner",
        "pass" if process.get("tirith_enabled") else "warn",
        "Tirith command scanner is enabled"
        if process.get("tirith_enabled")
        else "Tirith command scanner is disabled",
        "Set security.tirith_enabled: true.",
        "terminal-approval",
    )
    add(
        "process",
        "tirith_fail_closed",
        "warn" if process.get("tirith_fail_open") else "pass",
        "Tirith is configured fail-open"
        if process.get("tirith_fail_open")
        else "Tirith is configured fail-closed",
        "Set security.tirith_fail_open: false for strict enterprise posture.",
        "terminal-approval",
    )
    add(
        "process",
        "kernel_process_enforcement",
        "warn",
        (
            "Hermes approval scanners are active, but local commands do not "
            "run under seccomp or privilege drop"
        ),
        "Use a sandbox backend or OpenShell for syscall filters and non-root process identity.",
        "external-runtime",
    )

    network = policy.get("network", {})
    add(
        "network",
        "private_url_guard",
        "warn" if network.get("allow_private_urls") else "pass",
        "private/internal URLs are allowed"
        if network.get("allow_private_urls")
        else "private/internal URLs are blocked by default",
        "Keep security.allow_private_urls: false unless explicitly required.",
        "browser-tools",
    )
    add(
        "network",
        "website_blocklist",
        "pass" if network.get("website_blocklist_enabled") else "warn",
        "website blocklist is enabled"
        if network.get("website_blocklist_enabled")
        else "website blocklist is disabled",
        "Enable security.website_blocklist for enterprise deny rules.",
        "browser-tools",
    )
    policy_entry_count = int(network.get("policy_entry_count") or 0)
    add(
        "network",
        "network_policy_entries",
        "pass" if policy_entry_count else "warn",
        f"{policy_entry_count} OpenShell-style network policy entries configured",
        "Add security.policy.network_policies entries for destination-specific egress intent.",
        "policy",
    )
    add(
        "network",
        "gateway_proxy_enforcement",
        (
            "fail"
            if policy_entry_count and not network.get("gateway_proxy_enforced")
            else "warn"
        ),
        "Hermes does not provide a CONNECT proxy for terminal egress",
        "Use OpenShell or another egress proxy when network_policies must be mandatory.",
        "external-runtime",
    )

    permissions = policy.get("permissions", {})
    add(
        "permissions",
        "tool_permission_policy",
        "pass" if permissions.get("configured") else "warn",
        "tool allow/deny policy is configured"
        if permissions.get("configured")
        else "no explicit tool allow/deny policy is configured",
        "Use security.policy.permissions to document expected tool/toolset exposure.",
        "tool-loader",
    )
    add(
        "permissions",
        "mcp_reload_confirmation",
        "pass" if permissions.get("mcp_reload_confirm") else "warn",
        "MCP reloads require confirmation"
        if permissions.get("mcp_reload_confirm")
        else "MCP reload confirmation is disabled",
        "Set approvals.mcp_reload_confirm: true.",
        "cli-gateway",
    )

    gateway = policy.get("gateway", {})
    allowlists = gateway.get("allowlist_sources") or []
    require_user_allowlist = bool(gateway.get("require_user_allowlist"))
    add(
        "gateway",
        "user_allowlist",
        "pass" if allowlists or not require_user_allowlist else "warn",
        "gateway allowlists configured via: " + ", ".join(allowlists)
        if allowlists
        else "no gateway user allowlist env vars are configured",
        "Configure platform allowlists or disable require_user_allowlist deliberately.",
        "gateway",
    )
    add(
        "gateway",
        "approval_commands_inline",
        "pass" if gateway.get("approval_commands_inline") else "fail",
        "/approve and /deny bypass busy queues and unblock pending command approvals",
        enforcement="gateway",
    )
    add(
        "gateway",
        "control_commands_bypass_busy_queue",
        "pass" if gateway.get("control_commands_bypass_busy_queue") else "fail",
        "gateway control commands are dispatched before running-agent interruption",
        enforcement="gateway",
    )

    inference = policy.get("inference", {})
    add(
        "inference",
        "secret_redaction",
        "pass" if inference.get("redact_secrets") else "warn",
        "secret redaction is enabled"
        if inference.get("redact_secrets")
        else "secret redaction is disabled",
        "Set security.redact_secrets: true to scrub tool output and logs.",
        "output-redaction",
    )
    add(
        "inference",
        "provider_credentials_stripped",
        "pass" if inference.get("provider_credentials_stripped_from_tools") else "fail",
        "provider credentials are stripped from tool subprocess environments",
        enforcement="subprocess-env",
    )
    add(
        "inference",
        "managed_provider_routing",
        "pass" if inference.get("managed_provider_routing") else "warn",
        "inference is routed through a managed provider gateway"
        if inference.get("managed_provider_routing")
        else "Hermes does not expose an inference.local privacy router",
        (
            "Use OpenShell inference routing or a managed provider gateway "
            "when model credentials must stay outside the agent runtime."
        ),
        "external-runtime",
    )

    return checks


def _load_policy_document(path: str) -> Mapping[str, Any]:
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read policy file: {exc}") from exc
    try:
        if p.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            import yaml

            data = yaml.safe_load(raw)
    except Exception as exc:
        raise ValueError(f"could not parse policy file: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("policy file must contain a mapping")
    return data


def _extract_policy_mapping(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(data.get("security"), Mapping):
        security = data["security"]
        if isinstance(security.get("policy"), Mapping):
            return security["policy"]
    if isinstance(data.get("policy"), Mapping):
        return data["policy"]
    return data


def _validate_expected_type(
    errors: list[str],
    domain: str,
    key: str,
    item: Any,
    expected: str,
) -> None:
    label = f"{domain}.{key}"
    if expected == "boolean" and not isinstance(item, bool):
        errors.append(f"{label} must be a boolean")
    elif expected == "integer" and not isinstance(item, int):
        errors.append(f"{label} must be an integer")
    elif expected == "string" and not isinstance(item, str):
        errors.append(f"{label} must be a string")
    elif expected == "list[string]":
        if not isinstance(item, list) or not all(isinstance(x, str) for x in item):
            errors.append(f"{label} must be a list of strings")
    elif "|" in expected:
        choices = set(expected.split("|"))
        if str(item) not in choices:
            errors.append(f"{label} must be one of: {', '.join(sorted(choices))}")


def _validate_network_policies(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append("network_policies must be a mapping")
        return
    for name, entry in value.items():
        label = f"network_policies.{name}"
        if not isinstance(entry, Mapping):
            errors.append(f"{label} must be a mapping")
            continue
        endpoints = entry.get("endpoints")
        binaries = entry.get("binaries")
        if not isinstance(endpoints, list) or not endpoints:
            errors.append(f"{label}.endpoints must be a non-empty list")
        else:
            for idx, endpoint in enumerate(endpoints):
                endpoint_label = f"{label}.endpoints[{idx}]"
                if not isinstance(endpoint, Mapping):
                    errors.append(f"{endpoint_label} must be a mapping")
                    continue
                if not isinstance(endpoint.get("host"), str):
                    errors.append(f"{endpoint_label}.host must be a string")
                if not isinstance(endpoint.get("port"), int):
                    errors.append(f"{endpoint_label}.port must be an integer")
                protocol = endpoint.get("protocol")
                if protocol is not None and not isinstance(protocol, str):
                    errors.append(f"{endpoint_label}.protocol must be a string")
                enforcement = endpoint.get("enforcement")
                if enforcement is not None and enforcement not in {"audit", "enforce"}:
                    errors.append(f"{endpoint_label}.enforcement must be audit or enforce")
                access = endpoint.get("access")
                if access is not None and access not in {"read-only", "read-write", "full"}:
                    errors.append(
                        f"{endpoint_label}.access must be one of: full, read-only, read-write"
                    )
                for list_key in ("rules", "deny_rules", "allowed_ips"):
                    if list_key in endpoint and not isinstance(endpoint[list_key], list):
                        errors.append(f"{endpoint_label}.{list_key} must be a list")
        if not isinstance(binaries, list) or not binaries:
            errors.append(f"{label}.binaries must be a non-empty list")
        else:
            for idx, binary in enumerate(binaries):
                binary_label = f"{label}.binaries[{idx}]"
                if not isinstance(binary, Mapping):
                    errors.append(f"{binary_label} must be a mapping")
                    continue
                if not isinstance(binary.get("path"), str):
                    errors.append(f"{binary_label}.path must be a string")


def validate_policy_mapping(data: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    policy = _extract_policy_mapping(data)
    errors: list[str] = []
    warnings: list[str] = []
    for domain, value in policy.items():
        if domain == "version":
            if value != 1:
                errors.append("version must be 1")
            continue

        if domain not in HERMES_POLICY_SCHEMA:
            errors.append(f"unsupported policy domain: {domain}")
            continue
        if domain == "network_policies":
            _validate_network_policies(value, errors)
            continue
        if not isinstance(value, Mapping):
            errors.append(f"{domain} must be a mapping")
            continue
        allowed = HERMES_POLICY_SCHEMA[domain]
        for key, item in value.items():
            expected = allowed.get(key)
            if expected is None:
                errors.append(f"unsupported policy key: {domain}.{key}")
                continue
            _validate_expected_type(errors, domain, key, item, expected)
            if domain == "process" and key in {"run_as_user", "run_as_group"}:
                if str(item).strip().lower() in {"root", "0"}:
                    errors.append(f"{domain}.{key} must not be root or 0")
    openshell_specific_keys = OPENSHELL_TOP_LEVEL_KEYS - {"process"}
    if any(key in policy for key in openshell_specific_keys):
        if policy.get("version") != 1:
            errors.append("OpenShell-shaped policies must include version: 1")
    if not errors and not policy:
        warnings.append("policy file did not contain any supported policy domains")
    return errors, warnings


def _print_policy_text(policy: Mapping[str, Any]) -> None:
    print("Hermes Security Policy (effective)")
    for domain in (
        "filesystem",
        "environment",
        "process",
        "network",
        "permissions",
        "gateway",
        "inference",
    ):
        print(f"\n{domain}:")
        values = policy.get(domain, {})
        if isinstance(values, Mapping):
            for key, value in values.items():
                display = "<unset>" if value == "" else value
                print(f"  {key}: {display}")


def _print_checks_text(checks: list[PostureCheck]) -> None:
    print("Hermes Security Doctor")
    for check in checks:
        label = check.status.upper()
        print(
            f"{label:<4} {check.domain}.{check.name} "
            f"[{check.enforcement}]: {check.detail}"
        )
        if check.status != "pass" and check.recommendation:
            print(f"     {check.recommendation}")


def security_command(args: Any) -> None:
    action = getattr(args, "security_action", None) or "doctor"
    if action == "doctor":
        policy = build_effective_policy()
        checks = evaluate_posture(policy)
        if getattr(args, "json", False):
            print(json.dumps([asdict(check) for check in checks], indent=2))
        else:
            _print_checks_text(checks)
        code = (
            1
            if getattr(args, "strict", False)
            and any(c.status != "pass" for c in checks)
            else 0
        )
        raise SystemExit(code)

    if action == "policy":
        policy_action = getattr(args, "policy_action", None) or "show"
        if policy_action == "show":
            policy = build_effective_policy()
            if getattr(args, "json", False):
                print(json.dumps(policy, indent=2, sort_keys=True))
            else:
                _print_policy_text(policy)
            raise SystemExit(0)
        if policy_action == "validate":
            try:
                data = _load_policy_document(args.file)
                errors, warnings = validate_policy_mapping(data)
            except ValueError as exc:
                errors, warnings = [str(exc)], []
            if getattr(args, "json", False):
                print(json.dumps({"errors": errors, "warnings": warnings}, indent=2))
            else:
                if errors:
                    print("Policy invalid:")
                    for error in errors:
                        print(f"  - {error}")
                else:
                    print("Policy valid.")
                for warning in warnings:
                    print(f"Warning: {warning}")
            raise SystemExit(1 if errors else 0)

    print("Unknown security command", file=sys.stderr)
    raise SystemExit(2)
