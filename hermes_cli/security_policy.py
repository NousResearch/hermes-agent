"""Enterprise security posture reporting for Hermes.

This is intentionally a thin OpenShell-inspired policy surface over controls
Hermes already has: filesystem write guards, environment scrubbing, command
approvals, URL guard settings, and redaction. It does not claim network proxy
or kernel-level enforcement that Hermes does not currently provide.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_POLICY_SCHEMA: dict[str, dict[str, str]] = {
    "filesystem": {
        "write_safe_root": "string",
        "deny_hermes_control_plane": "boolean",
    },
    "environment": {
        "env_passthrough": "list[string]",
    },
    "process": {
        "approvals_mode": "manual|smart|off",
        "tirith_enabled": "boolean",
        "tirith_fail_open": "boolean",
    },
    "network": {
        "allow_private_urls": "boolean",
        "website_blocklist_enabled": "boolean",
    },
    "inference": {
        "redact_secrets": "boolean",
    },
}


@dataclass(frozen=True)
class PostureCheck:
    domain: str
    name: str
    status: str
    detail: str
    recommendation: str = ""


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


def _resolve_safe_root_source(
    config: Mapping[str, Any], env: Mapping[str, str]
) -> tuple[str, str]:
    env_root = str(env.get("HERMES_WRITE_SAFE_ROOT", "")).strip()
    if env_root:
        return env_root, "HERMES_WRITE_SAFE_ROOT"
    cfg_root = str(
        _get(config, "security", "policy", "filesystem", "write_safe_root", default="")
        or ""
    ).strip()
    if cfg_root:
        return cfg_root, "security.policy.filesystem.write_safe_root"
    return "", "unset"


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

    safe_root, safe_root_source = _resolve_safe_root_source(config, env)
    env_passthrough = _as_list(_get(config, "terminal", "env_passthrough", default=[]))
    provider_passthrough = sorted(
        name for name in env_passthrough if name in _provider_env_blocklist()
    )
    website_blocklist = _get(config, "security", "website_blocklist", default={})
    if not isinstance(website_blocklist, Mapping):
        website_blocklist = {}

    approvals_mode = str(_get(config, "approvals", "mode", default="manual") or "manual")

    return {
        "filesystem": {
            "write_safe_root": safe_root,
            "write_safe_root_source": safe_root_source,
            "deny_hermes_control_plane": _as_bool(
                _get(
                    config,
                    "security",
                    "policy",
                    "filesystem",
                    "deny_hermes_control_plane",
                    default=True,
                ),
                True,
            ),
            "deny_home_credentials": True,
        },
        "environment": {
            "scrub_provider_credentials": True,
            "env_passthrough": env_passthrough,
            "provider_credentials_in_passthrough": provider_passthrough,
        },
        "process": {
            "hardline_blocklist": True,
            "approvals_mode": approvals_mode,
            "tirith_enabled": _as_bool(
                _get(config, "security", "tirith_enabled", default=True), True
            ),
            "tirith_fail_open": _as_bool(
                _get(config, "security", "tirith_fail_open", default=True), True
            ),
        },
        "network": {
            "allow_private_urls": _as_bool(
                _get(config, "security", "allow_private_urls", default=False), False
            ),
            "website_blocklist_enabled": _as_bool(
                website_blocklist.get("enabled"), False
            ),
            "website_blocklist_domains": _as_list(website_blocklist.get("domains")),
        },
        "inference": {
            "redact_secrets": _as_bool(
                _get(config, "security", "redact_secrets", default=False), False
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
    ) -> None:
        checks.append(PostureCheck(domain, name, status, detail, recommendation))

    filesystem = policy.get("filesystem", {})
    safe_root = str(filesystem.get("write_safe_root") or "")
    add(
        "filesystem",
        "write_safe_root",
        "pass" if safe_root else "warn",
        (
            f"write-safe root is {safe_root!r}"
            if safe_root
            else "write-safe root is not set"
        ),
        (
            "Set security.policy.filesystem.write_safe_root or "
            "HERMES_WRITE_SAFE_ROOT for workspace-only writes."
        ),
    )
    add(
        "filesystem",
        "control_plane_write_deny",
        "pass" if filesystem.get("deny_hermes_control_plane") else "fail",
        "active Hermes control-plane files are write-denied"
        if filesystem.get("deny_hermes_control_plane")
        else "active Hermes control-plane files are not write-denied",
        "Set security.policy.filesystem.deny_hermes_control_plane: true.",
    )
    add(
        "filesystem",
        "home_credential_write_deny",
        "pass",
        "home credential paths are statically write-denied",
    )

    environment = policy.get("environment", {})
    provider_passthrough = environment.get("provider_credentials_in_passthrough") or []
    add(
        "environment",
        "provider_credential_scrubbing",
        "pass",
        "Hermes-managed provider credentials are scrubbed from local subprocesses by default",
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
    )
    add(
        "process",
        "dangerous_command_approvals",
        "fail" if approvals_mode == "off" else "pass",
        f"approvals.mode is {approvals_mode}",
        "Use approvals.mode: manual or smart for enterprise deployments.",
    )
    add(
        "process",
        "tirith_scanner",
        "pass" if process.get("tirith_enabled") else "warn",
        "Tirith command scanner is enabled"
        if process.get("tirith_enabled")
        else "Tirith command scanner is disabled",
        "Set security.tirith_enabled: true.",
    )
    add(
        "process",
        "tirith_fail_closed",
        "warn" if process.get("tirith_fail_open") else "pass",
        "Tirith is configured fail-open"
        if process.get("tirith_fail_open")
        else "Tirith is configured fail-closed",
        "Set security.tirith_fail_open: false for strict enterprise posture.",
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
    )
    add(
        "network",
        "website_blocklist",
        "pass" if network.get("website_blocklist_enabled") else "warn",
        "website blocklist is enabled"
        if network.get("website_blocklist_enabled")
        else "website blocklist is disabled",
        "Enable security.website_blocklist for enterprise deny rules.",
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


def validate_policy_mapping(data: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    policy = _extract_policy_mapping(data)
    errors: list[str] = []
    warnings: list[str] = []
    for domain, value in policy.items():
        if domain not in SUPPORTED_POLICY_SCHEMA:
            errors.append(f"unsupported policy domain: {domain}")
            continue
        if not isinstance(value, Mapping):
            errors.append(f"{domain} must be a mapping")
            continue
        allowed = SUPPORTED_POLICY_SCHEMA[domain]
        for key, item in value.items():
            expected = allowed.get(key)
            if expected is None:
                errors.append(f"unsupported policy key: {domain}.{key}")
                continue
            if expected == "boolean" and not isinstance(item, bool):
                errors.append(f"{domain}.{key} must be a boolean")
            elif expected == "string" and not isinstance(item, str):
                errors.append(f"{domain}.{key} must be a string")
            elif expected == "list[string]":
                if not isinstance(item, list) or not all(
                    isinstance(x, str) for x in item
                ):
                    errors.append(f"{domain}.{key} must be a list of strings")
            elif "|" in expected:
                choices = set(expected.split("|"))
                if str(item) not in choices:
                    errors.append(
                        f"{domain}.{key} must be one of: {', '.join(sorted(choices))}"
                    )
    if not errors and not policy:
        warnings.append("policy file did not contain any supported policy domains")
    return errors, warnings


def _print_policy_text(policy: Mapping[str, Any]) -> None:
    print("Hermes Security Policy (effective)")
    for domain in ("filesystem", "environment", "process", "network", "inference"):
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
        print(f"{label:<4} {check.domain}.{check.name}: {check.detail}")
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
