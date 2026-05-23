"""Read-only Hermes control-plane inventory.

The control inventory is intentionally observational: it reads local manifests,
configuration shape, and lightweight service status, but it does not execute
registered tools, mutate config, or print secret values.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is a repo dependency.
    yaml = None

from hermes_constants import get_hermes_home
from hermes_cli.config import get_env_value, read_raw_config
from hermes_cli.docker_security import (
    analyze_docker_command,
    analyze_docker_terminal_config,
    finding_notes,
    summarize_findings,
)
from hermes_cli.security_policy import (
    RiskDecision,
    classify_command,
    classify_tool_action,
)


SCHEMA_VERSION = 1
DEFAULT_PLATFORMS = ("cli", "cron", "telegram", "gateway", "api-server")
LAUNCHD_LABELS = (
    "ai.hermes.gateway",
    "com.agent1.hermes.gateway",
    "ai.hermes.health-guardian",
)
OPERATOR_SCRIPT_NAMES = (
    "hermes-gateway.sh",
    "hermes-env.sh",
    "agent-health-guardian.sh",
    "hermes-command-center.sh",
)
SECRET_NAME_RE = re.compile(
    r"(secret|token|api[_-]?key|password|passwd|private[_-]?key|credential|auth)",
    re.IGNORECASE,
)
SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"\b\d{7,}:[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
)
SECRET_ENV_NAME_PATTERN = (
    r"(?:[A-Za-z_][A-Za-z0-9_]*(?:key|token|secret|password|passwd|auth|credential)[A-Za-z0-9_]*"
    r"|key|token|secret|password|passwd|auth|credential)"
)
ENV_ASSIGNMENT_RE = re.compile(
    rf"\b({SECRET_ENV_NAME_PATTERN})=(\"[^\"]*\"|'[^']*'|[^ \t\n\r;&|]+)",
    re.IGNORECASE,
)
SECRET_OPTION_RE = re.compile(
    r"(--?[A-Za-z0-9_-]*(?:secret|token|api[-_]?key|password|passwd|auth|credential)[A-Za-z0-9_-]*(?:=|\s+))(\"[^\"]*\"|'[^']*'|[^ \t\n\r;&|]+)",
    re.IGNORECASE,
)
BEARER_RE = re.compile(r"(?i)\b(Bearer\s+)([A-Za-z0-9._\-]{12,})")
URL_PASSWORD_RE = re.compile(r"([a-z][a-z0-9+.-]*://[^/\s:@]+:)([^@\s/]+)(@)")
COMMAND_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = str(path)
    try:
        home = str(Path.home())
        if text == home:
            return "~"
        if text.startswith(home + os.sep):
            return "~" + text[len(home):]
    except Exception:
        pass
    return text


def redact_text(value: Any) -> str:
    """Return a string with secret-looking values removed."""
    text = "" if value is None else str(value)
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub("<redacted>", text)
    text = ENV_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=<redacted>", text)
    text = SECRET_OPTION_RE.sub(lambda m: f"{m.group(1)}<redacted>", text)
    text = BEARER_RE.sub(lambda m: f"{m.group(1)}<redacted>", text)
    text = URL_PASSWORD_RE.sub(lambda m: f"{m.group(1)}<redacted>{m.group(3)}", text)
    return text


def _credential_names(names: list[Any]) -> list[dict[str, Any]]:
    result = []
    for raw in sorted({str(name) for name in names if str(name).strip()}):
        result.append({"name": raw, "present": bool(get_env_value(raw))})
    return result


def _credential_requirement_names(raw: Any) -> list[str]:
    """Normalize manifest/config credential declarations to env var names."""
    names: list[str] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict) and value.get("name"):
                names.append(str(value["name"]))
            else:
                names.append(str(key))
        return [name for name in names if name.strip()]
    for entry in _as_list(raw):
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("env") or entry.get("key")
            if name:
                names.append(str(name))
        elif entry is not None:
            names.append(str(entry))
    return [name for name in names if name.strip()]


def _binary_requirement(binary: str | None) -> dict[str, Any] | None:
    if not binary:
        return None
    binary = redact_text(binary)
    path = shutil.which(binary)
    return {"name": binary, "present": bool(path), "path": _display_path(path) if path else None}


def _command_binary(command: Any) -> str | None:
    """Return the executable token from a shell-like command, without env assignments."""
    if not command:
        return None
    text = str(command)
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()
    if not parts:
        return None

    index = 0
    if parts and parts[0] == "env":
        index = 1
    while index < len(parts):
        token = parts[index]
        if COMMAND_ENV_ASSIGNMENT_RE.match(token):
            index += 1
            continue
        if parts[0] == "env" and token in {"-i", "-0"}:
            index += 1
            continue
        if parts[0] == "env" and token in {"-u", "--unset"}:
            index += 2
            continue
        if token.startswith("-"):
            return None
        if "=" in token:
            return None
        return redact_text(token)
    return None


def _risk_decision_for_name_and_text(
    name: str,
    text: str = "",
    *,
    command_like: bool = False,
) -> RiskDecision:
    if command_like:
        return classify_command(text or name, description=name)
    return classify_tool_action(name, description=text)


def _risk_for_name_and_text(name: str, text: str = "", *, command_like: bool = False) -> str:
    return _risk_decision_for_name_and_text(
        name,
        text,
        command_like=command_like,
    ).risk_tier


def _approval_for_risk(risk: str) -> str:
    return {
        "R0": "allow",
        "R1": "allow",
        "R2": "confirm",
        "R3": "confirm",
        "R4": "typed_confirm",
        "R5": "deny",
    }.get(risk, "confirm")


def _item(
    *,
    item_id: str,
    layer: str,
    name: str,
    status: str,
    observed_state: dict[str, Any] | None = None,
    enabled_for: list[str] | None = None,
    entrypoint: str | None = None,
    owner_file: str | None = None,
    tools: list[str] | None = None,
    requires: dict[str, Any] | None = None,
    cost_class: str = "unknown",
    risk_class: str | None = None,
    risk_category: str | None = None,
    approval_policy: str | None = None,
    health_probe: dict[str, Any] | None = None,
    safe_next_actions: list[str] | None = None,
    notes: list[str] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    decision = _risk_decision_for_name_and_text(name, entrypoint or "")
    risk = risk_class or decision.risk_tier
    category = risk_category or decision.risk_class
    policy = approval_policy or (
        decision.approval_policy if risk_class is None else _approval_for_risk(risk)
    )
    return {
        "id": item_id,
        "layer": layer,
        "name": name,
        "status": status,
        "observed_state": observed_state or {},
        "policy_overlay": {
            "managed_by": "hermes-control-plane",
            "enforcement_mode": "observe",
            "risk_category": category,
            "risk_tier": risk,
            "typed_confirmation_required": policy == "typed_confirm",
        },
        "enabled_for": sorted(enabled_for or []),
        "entrypoint": redact_text(entrypoint) if entrypoint else None,
        "owner_file": _display_path(owner_file) if owner_file else None,
        "tools": sorted({str(tool) for tool in (tools or [])}),
        "requires": requires or {"binaries": [], "credentials": [], "services": []},
        "cost_class": cost_class,
        "risk_class": risk,
        "risk_category": category,
        "approval_policy": policy,
        "health_probe": health_probe or {"kind": "none"},
        "last_observed": now or _utc_now(),
        "safe_next_actions": safe_next_actions or [],
        "notes": notes or [],
    }


def _enabled_platform_map(config: dict[str, Any]) -> dict[str, set[str]]:
    platform_toolsets = config.get("platform_toolsets") if isinstance(config, dict) else {}
    if not isinstance(platform_toolsets, dict):
        platform_toolsets = {}

    platforms = set(DEFAULT_PLATFORMS) | {str(name) for name in platform_toolsets}
    result: dict[str, set[str]] = {}
    for platform_name in sorted(platforms):
        raw_names = platform_toolsets.get(platform_name)
        if raw_names is None:
            if platform_name == "api-server":
                raw_names = ["hermes-api-server"]
            elif platform_name == "gateway":
                raw_names = ["hermes-gateway"]
            else:
                raw_names = [f"hermes-{platform_name}"]
        result[platform_name] = {str(name) for name in _as_list(raw_names)}
    return result


def _enabled_for_toolset(toolset_name: str, enabled_map: dict[str, set[str]]) -> list[str]:
    from toolsets import resolve_toolset

    enabled_for = []
    toolset_tools: set[str] = set()
    try:
        toolset_tools = set(resolve_toolset(toolset_name))
    except Exception:
        toolset_tools = set()

    for platform_name, configured in enabled_map.items():
        if toolset_name in configured:
            enabled_for.append(platform_name)
            continue
        for configured_name in configured:
            try:
                configured_tools = set(resolve_toolset(configured_name))
            except Exception:
                configured_tools = set()
            if toolset_tools and toolset_tools.issubset(configured_tools):
                enabled_for.append(platform_name)
                break
    return enabled_for


def _collect_toolsets(
    config: dict[str, Any],
    enabled_map: dict[str, set[str]],
    requirements: dict[str, bool],
    now: str,
) -> list[dict[str, Any]]:
    from toolsets import TOOLSETS, resolve_toolset

    items = []
    for name, definition in sorted(TOOLSETS.items()):
        enabled_for = _enabled_for_toolset(name, enabled_map)
        available = requirements.get(name, True)
        status = "disabled" if not enabled_for else ("enabled" if available else "gated")
        tools = []
        try:
            tools = resolve_toolset(name)
        except Exception:
            tools = list(definition.get("tools", []))
        decision = _risk_decision_for_name_and_text(name, " ".join(tools))
        items.append(_item(
            item_id=f"toolset.{name}",
            layer="toolset",
            name=name,
            status=status,
            observed_state={
                "source": "toolsets.TOOLSETS",
                "confidence": "high",
                "description": definition.get("description", ""),
                "includes": [str(item) for item in _as_list(definition.get("includes"))],
                "runtime_requirements_ok": available,
            },
            enabled_for=enabled_for,
            tools=tools,
            risk_class=decision.risk_tier,
            risk_category=decision.risk_class,
            approval_policy=decision.approval_policy,
            health_probe={"kind": "registry_requirements", "target": name},
            safe_next_actions=[
                f"Use `hermes tools list --platform {enabled_for[0]}` to inspect live enablement."
            ] if enabled_for else ["Enable through `hermes tools` only if needed."],
            now=now,
        ))

    fallback = config.get("fallback") if isinstance(config, dict) else {}
    providers = []
    if isinstance(fallback, dict):
        providers = [str(provider) for provider in _as_list(fallback.get("providers"))]
    items.append(_item(
        item_id="config.fallback.providers",
        layer="config",
        name="fallback.providers",
        status="enabled" if providers else "gated",
        observed_state={
            "source": "~/.hermes/config.yaml",
            "confidence": "high",
            "provider_count": len(providers),
        },
        enabled_for=["cli", "gateway"] if providers else [],
        cost_class="metered" if providers else "unknown",
        risk_class="R1",
        risk_category="read_only",
        health_probe={"kind": "config_presence", "target": "fallback.providers"},
        safe_next_actions=[
            "Run `hermes fallback list` to verify provider order without exposing keys."
        ],
        notes=[] if providers else ["No fallback providers configured in raw config."],
        now=now,
    ))
    return items


def _collect_tools(
    enabled_map: dict[str, set[str]],
    requirements: dict[str, bool],
    now: str,
) -> list[dict[str, Any]]:
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    tool_to_toolset = registry.get_tool_to_toolset_map()
    items = []
    for name in registry.get_all_tool_names():
        entry = registry.get_entry(name)
        toolset = tool_to_toolset.get(name) or (entry.toolset if entry else "")
        enabled_for = _enabled_for_toolset(toolset, enabled_map) if toolset else []
        req_env = [str(env) for env in getattr(entry, "requires_env", []) or []]
        runtime_ok = requirements.get(toolset, True)
        missing_creds = [req for req in req_env if not get_env_value(req)]
        status = "disabled" if not enabled_for else ("gated" if not runtime_ok else "enabled")
        owner_file = None
        handler = getattr(entry, "handler", None)
        if handler is not None:
            try:
                module = __import__(handler.__module__, fromlist=["__file__"])
                owner_file = getattr(module, "__file__", None)
            except Exception:
                owner_file = None
        decision = classify_tool_action(name, description=toolset, toolset=toolset)
        items.append(_item(
            item_id=f"tool.{name}",
            layer="tool",
            name=name,
            status=status,
            observed_state={
                "source": "tools.registry",
                "confidence": "high",
                "toolset": toolset,
                "description": getattr(entry, "description", "") or "",
                "async_handler": bool(getattr(entry, "is_async", False)),
                "runtime_requirements_ok": runtime_ok,
            },
            enabled_for=enabled_for,
            owner_file=owner_file,
            requires={"binaries": [], "credentials": _credential_names(req_env), "services": []},
            risk_class=decision.risk_tier,
            risk_category=decision.risk_class,
            approval_policy=decision.approval_policy,
            health_probe={"kind": "toolset_requirement", "target": toolset or name},
            safe_next_actions=[
                "Inspect with `hermes tools list --platform cli` before changing enablement."
            ],
            notes=(
                [f"Missing required credential: {name}" for name in missing_creds]
                if not runtime_ok
                else [f"Credential candidate not present: {name}" for name in missing_creds]
            ),
            now=now,
        ))
    return items


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif yaml is not None:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            data = {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _iter_manifest_paths(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    manifest_names = {"plugin.yaml", "plugin.yml", "manifest.yaml", "manifest.yml", "plugin.json"}
    paths: list[Path] = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.name in manifest_names:
            if ".git" in path.parts:
                continue
            paths.append(path)
    return paths


def _manifest_key(base: Path, manifest_path: Path, manifest: dict[str, Any]) -> str:
    try:
        rel_parent = manifest_path.parent.relative_to(base)
        key = str(rel_parent)
    except Exception:
        key = manifest_path.parent.name
    explicit = manifest.get("name") or manifest.get("id")
    if explicit and "/" not in key and key not in {".", ""}:
        return str(explicit)
    return key if key not in {".", ""} else str(explicit or manifest_path.parent.name)


def _collect_plugins(config: dict[str, Any], repo_root: Path, hermes_home: Path, now: str) -> list[dict[str, Any]]:
    plugin_cfg = config.get("plugins") if isinstance(config, dict) else {}
    if not isinstance(plugin_cfg, dict):
        plugin_cfg = {}
    enabled = {str(name) for name in _as_list(plugin_cfg.get("enabled"))}
    disabled = {str(name) for name in _as_list(plugin_cfg.get("disabled"))}
    bases = ((repo_root / "plugins", "bundled"), (hermes_home / "plugins", "user"))
    items = []
    seen: set[tuple[str, str]] = set()
    for base, source in bases:
        for manifest_path in _iter_manifest_paths(base):
            manifest = _read_manifest(manifest_path)
            key = _manifest_key(base, manifest_path, manifest)
            dedupe_key = (source, key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            raw_req = manifest.get("requires_env") or manifest.get("required_env") or manifest.get("env")
            req_env = _credential_requirement_names(raw_req)
            missing_env = [name for name in req_env if not get_env_value(name)]
            is_enabled = key in enabled or manifest.get("enabled_by_default") is True
            if key in disabled:
                status = "disabled"
            elif not is_enabled:
                status = "gated"
            elif missing_env:
                status = "gated"
            else:
                status = "enabled"
            tools = [str(tool) for tool in _as_list(manifest.get("tools"))]
            toolsets = [str(ts) for ts in _as_list(manifest.get("toolsets"))]
            decision = _risk_decision_for_name_and_text(key, " ".join(tools + toolsets))
            items.append(_item(
                item_id=f"plugin.{source}.{key.replace('/', '.')}",
                layer="plugin",
                name=key,
                status=status,
                observed_state={
                    "source": source,
                    "confidence": "medium",
                    "manifest": _display_path(manifest_path),
                    "version": str(manifest.get("version", "")),
                    "toolsets": toolsets,
                },
                enabled_for=["plugin-manager"] if is_enabled and key not in disabled else [],
                owner_file=str(manifest_path),
                tools=tools,
                requires={"binaries": [], "credentials": _credential_names(req_env), "services": []},
                risk_class=decision.risk_tier,
                risk_category=decision.risk_class,
                approval_policy=decision.approval_policy,
                health_probe={"kind": "manifest_presence", "target": key},
                safe_next_actions=[
                    f"Use `hermes plugins list` before enabling or disabling {key}."
                ],
                notes=[f"Missing credential: {name}" for name in missing_env],
                now=now,
            ))
    return items


def _collect_mcp_servers(config: dict[str, Any], now: str) -> list[dict[str, Any]]:
    servers = config.get("mcp_servers") if isinstance(config, dict) else {}
    if not isinstance(servers, dict):
        return []
    items = []
    for name, server_cfg in sorted(servers.items(), key=lambda pair: str(pair[0])):
        if not isinstance(server_cfg, dict):
            server_cfg = {}
        enabled = server_cfg.get("enabled", True)
        enabled_bool = enabled if isinstance(enabled, bool) else str(enabled).lower() not in {"0", "false", "no", "off"}
        command = server_cfg.get("command") or server_cfg.get("url") or server_cfg.get("transport") or ""
        env_cfg = server_cfg.get("env") if isinstance(server_cfg.get("env"), dict) else {}
        binary = None
        if server_cfg.get("command"):
            binary = _command_binary(server_cfg.get("command"))
        binary_req = _binary_requirement(binary)
        missing_binary = binary_req is not None and not binary_req["present"]
        status = "disabled" if not enabled_bool else ("gated" if missing_binary else "enabled")
        decision = _risk_decision_for_name_and_text(str(name), str(command), command_like=True)
        docker_findings = analyze_docker_command(str(command))
        observed_state = {
            "source": "~/.hermes/config.yaml",
            "confidence": "high",
            "transport": str(server_cfg.get("transport", "")),
            "env_vars": sorted(str(key) for key in env_cfg),
        }
        if docker_findings:
            observed_state["docker_security"] = {
                **summarize_findings(docker_findings),
                "findings": [finding.to_dict() for finding in docker_findings],
            }
        notes = ["Configured command binary is missing."] if missing_binary else []
        notes.extend(finding_notes(docker_findings))
        items.append(_item(
            item_id=f"mcp.{name}",
            layer="mcp",
            name=str(name),
            status=status,
            observed_state=observed_state,
            enabled_for=["mcp"] if enabled_bool else [],
            entrypoint=command,
            requires={
                "binaries": [binary_req] if binary_req else [],
                "credentials": _credential_names([key for key in env_cfg if SECRET_NAME_RE.search(str(key))]),
                "services": [],
            },
            risk_class=decision.risk_tier,
            risk_category=decision.risk_class,
            approval_policy=decision.approval_policy,
            health_probe={"kind": "binary_presence", "target": binary} if binary else {"kind": "config_presence", "target": str(name)},
            safe_next_actions=["Run a targeted MCP smoke test before enabling this server for agents."],
            notes=notes,
            now=now,
        ))
    return items


def _collect_container_backend(config: dict[str, Any], now: str) -> list[dict[str, Any]]:
    terminal = config.get("terminal") if isinstance(config, dict) else {}
    if not isinstance(terminal, dict):
        terminal = {}
    backend = str(terminal.get("backend", "local") or "local").strip().lower()
    docker_keys_present = sorted(str(key) for key in terminal if str(key).startswith("docker_"))
    docker_findings = analyze_docker_terminal_config(config)
    forward_env_names = [str(name).strip() for name in _as_list(terminal.get("docker_forward_env")) if str(name).strip()]
    docker_env = terminal.get("docker_env")
    docker_env_keys = sorted(str(key) for key in docker_env) if isinstance(docker_env, dict) else []
    sensitive_names = [
        name
        for name in forward_env_names + docker_env_keys
        if SECRET_NAME_RE.search(name) or name.upper() in {"DOCKER_CONFIG", "DOCKER_HOST", "SSH_AUTH_SOCK"}
    ]
    review = {
        **summarize_findings(docker_findings),
        "findings": [finding.to_dict() for finding in docker_findings],
    }
    typed_required = bool(review["typed_confirmation_required"])
    if backend == "docker":
        status = "degraded" if typed_required else "enabled"
    elif docker_keys_present:
        status = "gated"
    else:
        status = "disabled"
    notes = finding_notes(docker_findings)
    if backend != "docker" and docker_keys_present:
        notes.append("Docker settings are present but terminal.backend is not docker.")
    if not docker_keys_present and backend != "docker":
        notes.append("Docker backend is not enabled in terminal config.")
    risk = "R4" if typed_required else ("R3" if backend == "docker" or docker_keys_present else "R1")
    category = "credential_sensitive" if backend == "docker" or docker_findings else "read_only"
    policy = "typed_confirm" if typed_required else _approval_for_risk(risk)
    return [_item(
        item_id="container_backend.docker",
        layer="container_backend",
        name="Docker terminal backend",
        status=status,
        observed_state={
            "source": "~/.hermes/config.yaml",
            "confidence": "high",
            "backend": backend,
            "docker_config_keys_present": docker_keys_present,
            "docker_forward_env_names": sorted(set(forward_env_names)),
            "docker_env_keys": docker_env_keys,
            "docker_volume_count": len(_as_list(terminal.get("docker_volumes"))),
            "docker_extra_arg_count": len(_as_list(terminal.get("docker_extra_args"))),
            "docker_mount_cwd_to_workspace": str(terminal.get("docker_mount_cwd_to_workspace", "")).lower()
            in {"true", "1", "yes", "on"},
            "docker_security": review,
        },
        enabled_for=["terminal"] if backend == "docker" else [],
        requires={"binaries": [_binary_requirement("docker")], "credentials": _credential_names(sensitive_names), "services": []},
        risk_class=risk,
        risk_category=category,
        approval_policy=policy,
        health_probe={"kind": "binary_presence", "target": "docker"} if backend == "docker" else {"kind": "config_presence", "target": "terminal.backend"},
        safe_next_actions=[
            "Keep Docker credential forwarding per-job and explicit.",
            "Avoid mounting Docker socket, host home, or credential directories into containers.",
        ],
        notes=notes,
        now=now,
    )]


def _collect_quick_commands(config: dict[str, Any], now: str) -> list[dict[str, Any]]:
    commands = config.get("quick_commands") if isinstance(config, dict) else {}
    if not isinstance(commands, dict):
        return []
    items = []
    for name, value in sorted(commands.items(), key=lambda pair: str(pair[0])):
        if isinstance(value, dict):
            command_text = value.get("command") or value.get("prompt") or ""
            description = str(value.get("description", ""))
        else:
            command_text = str(value)
            description = ""
        preview = redact_text(command_text).replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        decision = _risk_decision_for_name_and_text(
            str(name),
            command_text,
            command_like=True,
        )
        docker_findings = analyze_docker_command(command_text)
        observed_state = {
            "source": "~/.hermes/config.yaml",
            "confidence": "high",
            "description": redact_text(description),
            "command_preview": preview,
        }
        if docker_findings:
            observed_state["docker_security"] = {
                **summarize_findings(docker_findings),
                "findings": [finding.to_dict() for finding in docker_findings],
            }
        items.append(_item(
            item_id=f"quick_command.{name}",
            layer="quick_command",
            name=str(name),
            status="enabled",
            observed_state=observed_state,
            enabled_for=["cli"],
            risk_class=decision.risk_tier,
            risk_category=decision.risk_class,
            approval_policy=decision.approval_policy,
            health_probe={"kind": "none"},
            safe_next_actions=["Review command contents locally before running shortcuts with side effects."],
            notes=finding_notes(docker_findings),
            now=now,
        ))
    return items


def _collect_cron_jobs(hermes_home: Path, now: str) -> list[dict[str, Any]]:
    jobs_path = hermes_home / "cron" / "jobs.json"
    if not jobs_path.is_file():
        return [_item(
            item_id="cron.jobs",
            layer="cron",
            name="cron.jobs",
            status="missing",
            observed_state={"source": _display_path(jobs_path), "confidence": "high"},
            risk_class="R3",
            risk_category="external_side_effect",
            health_probe={"kind": "file_presence", "target": _display_path(jobs_path)},
            safe_next_actions=["Run `hermes cron list` after configuring scheduled work."],
            now=now,
        )]
    try:
        data = json.loads(jobs_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    raw_jobs = data.get("jobs") if isinstance(data, dict) else []
    if not isinstance(raw_jobs, list):
        raw_jobs = []
    items = []
    for index, job in enumerate(raw_jobs):
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("id") or f"job-{index}")
        enabled = bool(job.get("enabled", not job.get("paused", False)))
        status = "enabled" if enabled else "disabled"
        items.append(_item(
            item_id=f"cron.{job_id}",
            layer="cron",
            name=job_id,
            status=status,
            observed_state={
                "source": _display_path(jobs_path),
                "confidence": "high",
                "schedule": str(job.get("schedule", "")),
                "mode": str(job.get("mode", "")),
                "platform": str(job.get("platform", "")),
                "delivery": str(job.get("delivery", "")),
                "last_run_at": str(job.get("last_run_at", "")),
                "next_run_at": str(job.get("next_run_at", "")),
            },
            enabled_for=["cron"] if enabled else [],
            risk_class="R3",
            risk_category="external_side_effect",
            health_probe={"kind": "cron_metadata", "target": job_id},
            safe_next_actions=[f"Use `hermes cron run {job_id}` only for explicit smoke tests."],
            now=now,
        ))
    return items


def _parse_launchctl_output(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, pattern in {
        "pid": r"\bpid\s*=\s*(\d+)",
        "last_exit_status": r"\blast exit code\s*=\s*([^\n]+)",
        "program": r"\bprogram\s*=\s*([^\n]+)",
    }.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed[key] = redact_text(match.group(1).strip())
    return parsed


def _collect_launchd(include_runtime: bool, now: str) -> list[dict[str, Any]]:
    items = []
    launchctl = shutil.which("launchctl")
    uid = os.getuid() if hasattr(os, "getuid") else None
    for label in LAUNCHD_LABELS:
        observed = {
            "source": "launchctl",
            "confidence": "low" if not include_runtime else "medium",
            "label": label,
        }
        status = "unknown"
        notes = []
        if not include_runtime:
            notes.append("Runtime probing disabled.")
        elif platform.system() != "Darwin" or not launchctl or uid is None:
            status = "unknown"
            notes.append("launchctl is unavailable on this platform.")
        else:
            try:
                proc = subprocess.run(
                    [launchctl, "print", f"gui/{uid}/{label}"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if proc.returncode == 0:
                    parsed = _parse_launchctl_output(proc.stdout)
                    observed.update(parsed)
                    status = "healthy" if parsed.get("pid") else "enabled"
                else:
                    status = "missing"
            except subprocess.TimeoutExpired:
                status = "degraded"
                notes.append("launchctl probe timed out.")
            except Exception as exc:
                status = "unknown"
                notes.append(f"launchctl probe failed: {type(exc).__name__}")
        items.append(_item(
            item_id=f"launchd.{label}",
            layer="launchd",
            name=label,
            status=status,
            observed_state=observed,
            enabled_for=["macos-launchd"] if status in {"healthy", "enabled"} else [],
            risk_class="R3",
            risk_category="credential_sensitive",
            health_probe={"kind": "launchctl_print", "target": label},
            safe_next_actions=["Use `hermes gateway status` before restarting or replacing launchd services."],
            notes=notes,
            now=now,
        ))
    return items


def _collect_operator_scripts(operator_scripts_dir: Path, now: str) -> list[dict[str, Any]]:
    items = []
    candidate_paths = [operator_scripts_dir / name for name in OPERATOR_SCRIPT_NAMES]
    if operator_scripts_dir.is_dir():
        for path in sorted(operator_scripts_dir.glob("hermes*.sh")):
            if path not in candidate_paths:
                candidate_paths.append(path)
    for path in candidate_paths:
        exists = path.is_file()
        executable = exists and os.access(path, os.X_OK)
        status = "enabled" if executable else ("degraded" if exists else "missing")
        items.append(_item(
            item_id=f"operator_script.{path.name}",
            layer="operator_script",
            name=path.name,
            status=status,
            observed_state={
                "source": "filesystem",
                "confidence": "high",
                "path": _display_path(path),
                "executable": executable,
            },
            enabled_for=["operator"] if exists else [],
            entrypoint=str(path),
            owner_file=str(path) if exists else None,
            risk_class=_risk_for_name_and_text(path.name, str(path)),
            risk_category=_risk_decision_for_name_and_text(path.name, str(path)).risk_class,
            health_probe={"kind": "file_presence", "target": _display_path(path)},
            safe_next_actions=["Inspect script contents before changing wrapper behavior."],
            now=now,
        ))
    return items


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer = Counter(str(item.get("layer")) for item in items)
    by_status = Counter(str(item.get("status")) for item in items)
    gated = [
        {"id": item["id"], "name": item["name"], "layer": item["layer"]}
        for item in items
        if item.get("status") in {"gated", "missing", "degraded"}
    ][:50]
    high_risk_enabled = [
        {
            "id": item["id"],
            "risk_class": item["risk_class"],
            "risk_category": item.get("risk_category"),
            "approval_policy": item["approval_policy"],
        }
        for item in items
        if item.get("status") in {"enabled", "healthy"} and item.get("risk_class") in {"R4", "R5"}
    ][:50]
    return {
        "total_items": len(items),
        "by_layer": dict(sorted(by_layer.items())),
        "by_status": dict(sorted(by_status.items())),
        "gated_or_missing": gated,
        "high_risk_enabled": high_risk_enabled,
        "warnings": [
            "Inventory is redacted and observational; use existing Hermes commands for live actions.",
            "Credential fields report presence only, never values.",
        ],
    }


def _secret_scan_inventory(inventory: dict[str, Any]) -> list[str]:
    serialized = json.dumps(inventory, sort_keys=True, ensure_ascii=False)
    findings = []
    for pattern in SECRET_VALUE_PATTERNS:
        if pattern.search(serialized):
            findings.append(pattern.pattern)
    for name, pattern, value_group in (
        ("env_assignment", ENV_ASSIGNMENT_RE, 2),
        ("secret_option", SECRET_OPTION_RE, 2),
        ("bearer_token", BEARER_RE, 2),
        ("url_password", URL_PASSWORD_RE, 2),
    ):
        for match in pattern.finditer(serialized):
            value = match.group(value_group).strip("\"'")
            if value and value != "<redacted>":
                findings.append(name)
                break
    return findings


def build_inventory(
    *,
    config: dict[str, Any] | None = None,
    hermes_home: str | Path | None = None,
    repo_root: str | Path | None = None,
    operator_scripts_dir: str | Path | None = None,
    include_runtime: bool = True,
    probe_tool_requirements: bool = True,
) -> dict[str, Any]:
    """Build a redacted, read-only inventory for Hermes control-plane review."""
    now = _utc_now()
    raw_config = config if config is not None else read_raw_config()
    if not isinstance(raw_config, dict):
        raw_config = {}
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    operator_dir = (
        Path(operator_scripts_dir)
        if operator_scripts_dir is not None
        else Path.home() / "Operator" / "scripts"
    )
    enabled_map = _enabled_platform_map(raw_config)

    requirements: dict[str, bool] = {}
    if probe_tool_requirements:
        try:
            from tools.registry import discover_builtin_tools, registry

            discover_builtin_tools()
            requirements = registry.check_toolset_requirements()
        except Exception:
            requirements = {}

    items: list[dict[str, Any]] = []
    items.extend(_collect_toolsets(raw_config, enabled_map, requirements, now))
    items.extend(_collect_tools(enabled_map, requirements, now))
    items.extend(_collect_plugins(raw_config, repo, home, now))
    items.extend(_collect_container_backend(raw_config, now))
    items.extend(_collect_mcp_servers(raw_config, now))
    items.extend(_collect_quick_commands(raw_config, now))
    items.extend(_collect_cron_jobs(home, now))
    items.extend(_collect_launchd(include_runtime, now))
    items.extend(_collect_operator_scripts(operator_dir, now))

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now,
        "owner": "hermes-control-plane",
        "redacted": True,
        "scope": {
            "repo_root": _display_path(repo),
            "hermes_home": _display_path(home),
            "operator_scripts_dir": _display_path(operator_dir),
            "runtime_probe": include_runtime,
            "tool_requirement_probe": probe_tool_requirements,
        },
        "summary": _summary(items),
        "items": sorted(items, key=lambda item: (item["layer"], item["name"], item["id"])),
    }
    leaks = _secret_scan_inventory(inventory)
    if leaks:
        raise RuntimeError(f"control inventory redaction failed for {len(leaks)} pattern(s)")
    return inventory


def format_markdown(inventory: dict[str, Any]) -> str:
    summary = inventory.get("summary", {})
    lines = [
        "# Hermes Control Inventory",
        "",
        f"- Generated: {inventory.get('generated_at')}",
        f"- Schema version: {inventory.get('schema_version')}",
        f"- Redacted: {inventory.get('redacted')}",
        f"- Total items: {summary.get('total_items', 0)}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted((summary.get("by_status") or {}).items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Layer Counts", ""])
    for layer, count in sorted((summary.get("by_layer") or {}).items()):
        lines.append(f"- {layer}: {count}")
    lines.extend(["", "## Gated Or Missing", ""])
    gated = summary.get("gated_or_missing") or []
    if gated:
        for entry in gated:
            lines.append(f"- {entry.get('id')} ({entry.get('layer')})")
    else:
        lines.append("- None")
    lines.extend(["", "## High Risk Enabled", ""])
    high_risk = summary.get("high_risk_enabled") or []
    if high_risk:
        for entry in high_risk:
            lines.append(
                f"- {entry.get('id')}: {entry.get('risk_class')} "
                f"{entry.get('risk_category')} / {entry.get('approval_policy')}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Items", ""])
    for item in inventory.get("items", []):
        enabled_for = ",".join(item.get("enabled_for") or []) or "-"
        lines.append(
            f"- {item.get('id')}: {item.get('status')} "
            f"[{item.get('risk_class')} {item.get('risk_category')}/{item.get('approval_policy')}] "
            f"enabled_for={enabled_for}"
        )
    return "\n".join(lines) + "\n"


def control_command(args: argparse.Namespace) -> int:
    action = getattr(args, "control_action", None) or "inventory"
    if action != "inventory":
        raise SystemExit(f"Unknown control action: {action}")
    inventory = build_inventory(
        include_runtime=not bool(getattr(args, "no_runtime", False)),
        probe_tool_requirements=not bool(getattr(args, "no_tool_probe", False)),
    )
    output_format = getattr(args, "format", None) or "json"
    if output_format == "markdown":
        print(format_markdown(inventory), end="")
    else:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


def register_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="control_action")
    inv = sub.add_parser(
        "inventory",
        help="Print a redacted read-only inventory of Hermes tools, plugins, and runtime controls.",
    )
    fmt = inv.add_mutually_exclusive_group()
    fmt.add_argument("--json", dest="format", action="store_const", const="json", default="json")
    fmt.add_argument("--markdown", dest="format", action="store_const", const="markdown")
    inv.add_argument(
        "--redact",
        action="store_true",
        default=True,
        help="Always enabled; credential values are never printed.",
    )
    inv.add_argument(
        "--no-runtime",
        action="store_true",
        help="Skip launchd/runtime probes and report static inventory only.",
    )
    inv.add_argument(
        "--no-tool-probe",
        action="store_true",
        help="Skip registry requirement probes for faster static output.",
    )
    parser.set_defaults(func=control_command, control_action="inventory", format="json")
