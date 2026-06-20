"""AgentCyber status and setup helpers.

This module is deliberately read-mostly and secret-safe: status output reports
presence/booleans for sensitive fields, never credential values.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

_AGENTCYBER_TOOLS = ("threat_intel", "extract_iocs", "vuln_triage", "ir_incident", "network_scan")
_LIVE_USB_TOOLS = ("live_usb",)


def _redacted_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": str(runtime.get("provider") or ""),
        "model": str(runtime.get("model") or ""),
        "base_url_present": bool(str(runtime.get("base_url") or "").strip()),
        "api_key_present": bool(str(runtime.get("api_key") or "").strip()),
        "api_key_env": str(runtime.get("api_key_env") or ""),
        "api_mode": str(runtime.get("api_mode") or "chat_completions"),
        "context_length": runtime.get("context_length"),
    }


def _ollama_tags_url(base_url: str) -> str:
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunparse((parsed.scheme or "http", parsed.netloc, f"{path}/api/tags", "", "", ""))


def check_local_runtime_health(runtime: dict[str, Any], *, timeout: float = 3.0) -> dict[str, Any]:
    provider = str(runtime.get("provider") or "").strip().lower()
    base_url = str(runtime.get("base_url") or "").strip()
    model = str(runtime.get("model") or "").strip()
    if not base_url:
        return {"ok": False, "reason": "missing base_url"}
    if provider not in {"ollama", "custom", "local", "lmstudio"}:
        return {"ok": None, "reason": "provider is not directly health-checked"}

    url = _ollama_tags_url(base_url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - status path reports type safely
        return {"ok": False, "url": url, "reason": f"{type(exc).__name__}: {exc}"}

    models = [str(entry.get("name") or "") for entry in payload.get("models", []) if isinstance(entry, dict)]
    model_present = (not model) or any(name == model or name.split(":", 1)[0] == model for name in models)
    return {
        "ok": bool(model_present),
        "url": url,
        "model_present": bool(model_present),
        "models_count": len(models),
    }


def _git_relation() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {
            "head": _run(["git", "rev-parse", "HEAD"]),
            "branch": _run(["git", "branch", "--show-current"]),
            "dirty": bool(_run(["git", "status", "--short"])),
            "behind_origin_main": int(_run(["git", "rev-list", "--count", "HEAD..origin/main"])),
            "behind_upstream_main": int(_run(["git", "rev-list", "--count", "HEAD..upstream/main"])),
            "ahead_upstream_main": int(_run(["git", "rev-list", "--count", "upstream/main..HEAD"])),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def build_status_report(
    config: dict[str, Any],
    *,
    platform: str = "cli",
    health_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = check_local_runtime_health,
) -> dict[str, Any]:
    from agent.cyber_policy import load_asset_registry
    from hermes_cli.tools_config import _checklist_toolset_keys, _get_platform_tools
    from tools.registry import discover_builtin_tools, registry

    cyber_cfg = config.get("agent_cyber", {}) if isinstance(config, dict) else {}
    if not isinstance(cyber_cfg, dict):
        cyber_cfg = {}
    routing = cyber_cfg.get("routing", {}) if isinstance(cyber_cfg.get("routing"), dict) else {}
    runtime = routing.get("local_open_weight", {}) if isinstance(routing.get("local_open_weight"), dict) else {}

    registry_obj = load_asset_registry(config)
    discover_builtin_tools()
    registered_tools = sorted(name for name in (*_AGENTCYBER_TOOLS, *_LIVE_USB_TOOLS) if name in registry._tools)
    enabled_toolsets = sorted(_get_platform_tools(config, platform, include_default_mcp_servers=False))
    configurable = sorted(_checklist_toolset_keys(platform))
    health = health_fn(runtime) if health_fn else {"ok": None, "reason": "skipped"}

    return {
        "agent_cyber": {
            "routing_enabled": bool(routing.get("enabled", True)),
            "require_local_for_sensitive": bool(routing.get("require_local_for_sensitive", True)),
            "allow_hosted_override": bool(routing.get("allow_hosted_override", True)),
            "local_open_weight": _redacted_runtime(runtime),
        },
        "local_runtime_health": health,
        "assets": {
            "count": len(registry_obj.assets),
            "source": registry_obj.source,
            "builtin_enabled": bool(cyber_cfg.get("include_builtin_bc_assets", True)),
        },
        "toolsets": {
            "platform": platform,
            "cyber_visible": "cyber" in configurable,
            "live_usb_visible": "live_usb" in configurable,
            "cyber_enabled": "cyber" in enabled_toolsets,
            "live_usb_enabled": "live_usb" in enabled_toolsets,
            "registered_tools": registered_tools,
        },
        "git": _git_relation(),
    }


def apply_agentcyber_setup(
    config: dict[str, Any],
    *,
    platform: str = "cli",
    provider: str = "ollama",
    model: str = "qwen3-coder:30b",
    base_url: str = "http://192.168.1.120:11434/v1",
    api_mode: str = "chat_completions",
    enable_live_usb: bool = False,
) -> dict[str, Any]:
    cfg = deepcopy(config) if isinstance(config, dict) else {}
    cyber_cfg = cfg.setdefault("agent_cyber", {})
    routing = cyber_cfg.setdefault("routing", {})
    routing["enabled"] = True
    routing["require_local_for_sensitive"] = True
    routing.setdefault("allow_hosted_override", True)
    routing.setdefault("allow_hosted_open_weight", False)
    routing["local_open_weight"] = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": "",
        "api_mode": api_mode,
        "context_length": 131072,
    }
    cyber_cfg.setdefault("include_builtin_bc_assets", True)
    cyber_cfg.setdefault("asset_registry", {"file": "", "assets": []})
    cyber_cfg.setdefault("execution_gates", {"enabled": True})

    platform_toolsets = cfg.setdefault("platform_toolsets", {})
    current = platform_toolsets.setdefault(platform, ["hermes-cli"])
    if not isinstance(current, list):
        current = ["hermes-cli"]
        platform_toolsets[platform] = current
    for key in ("cyber", "live_usb") if enable_live_usb else ("cyber",):
        if key not in current:
            current.append(key)
    return cfg


def _format_status(report: dict[str, Any]) -> str:
    runtime = report["agent_cyber"]["local_open_weight"]
    health = report["local_runtime_health"]
    toolsets = report["toolsets"]
    assets = report["assets"]
    git = report.get("git", {})
    lines = [
        "AgentCyber status",
        f"  routing enabled: {report['agent_cyber']['routing_enabled']}",
        f"  require local for sensitive: {report['agent_cyber']['require_local_for_sensitive']}",
        f"  local runtime: provider={runtime['provider'] or '-'} model={runtime['model'] or '-'} base_url_present={runtime['base_url_present']}",
        f"  local health: {health.get('ok')} ({health.get('reason', 'checked')})",
        f"  assets: {assets['count']} from {assets['source']}",
        f"  toolsets[{toolsets['platform']}]: cyber visible={toolsets['cyber_visible']} enabled={toolsets['cyber_enabled']}; live_usb visible={toolsets['live_usb_visible']} enabled={toolsets['live_usb_enabled']}",
        f"  registered cyber tools: {', '.join(toolsets['registered_tools']) or '-'}",
    ]
    if git.get("available", True):
        lines.append(
            f"  git: branch={git.get('branch') or '-'} dirty={git.get('dirty')} behind_origin={git.get('behind_origin_main')} behind_upstream={git.get('behind_upstream_main')}"
        )
    return "\n".join(lines)


def _approval_to_dict(approval) -> dict[str, Any]:  # noqa: ANN001
    return {
        "approval_id": approval.approval_id,
        "created_at": approval.created_at,
        "expires_at": approval.expires_at,
        "operator": approval.operator,
        "reason": approval.reason,
        "gate": approval.gate,
        "tool_name": approval.tool_name,
        "asset_matches": list(approval.asset_matches),
        "revoked": approval.revoked,
        "redacted_args": approval.redacted_args or {},
    }


def _breakglass_store(path: str):
    from agent.cyber_breakglass import BreakGlassStore

    return BreakGlassStore(Path(path).expanduser()) if path else BreakGlassStore()


def _create_breakglass_approval(args) -> tuple[object, bool]:  # noqa: ANN001
    from agent.cyber_policy import classify_tool_gate, load_asset_registry

    from hermes_cli.config import load_config

    try:
        function_args = json.loads(getattr(args, "args_json", "") or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--args-json must be valid JSON: {exc}") from exc
    if not isinstance(function_args, dict):
        raise SystemExit("--args-json must decode to an object")

    tool_name = getattr(args, "tool", "")
    gate, _reason, candidates = classify_tool_gate(tool_name, function_args)
    if gate != "S5":
        raise SystemExit(f"break-glass approvals are only for S5 actions; classified as {gate}")
    registry = load_asset_registry(load_config())
    matched = registry.matching_assets(candidates, gate="S3") if candidates else []
    asset_matches = tuple(asset.name for asset in matched)
    if not asset_matches:
        raise SystemExit("S5 approval requires at least one matching authorized asset")

    store = _breakglass_store(getattr(args, "store", ""))
    approval = store.create(
        tool_name=tool_name,
        function_args=function_args,
        gate=gate,
        asset_matches=asset_matches,
        operator=getattr(args, "operator", ""),
        reason=getattr(args, "reason", ""),
        ttl_minutes=int(getattr(args, "ttl_minutes", 15)),
    ) if getattr(args, "apply", False) else None
    if approval is None:
        from agent.cyber_breakglass import BreakGlassApproval, fingerprint_tool_call, redacted_args, utc_now, _iso
        from datetime import timedelta

        now = utc_now()
        approval = BreakGlassApproval(
            approval_id="DRY-RUN",
            created_at=_iso(now),
            expires_at=_iso(now + timedelta(minutes=max(1, int(getattr(args, "ttl_minutes", 15))))),
            operator=getattr(args, "operator", ""),
            reason=getattr(args, "reason", ""),
            gate=gate,
            tool_name=tool_name,
            args_fingerprint=fingerprint_tool_call(tool_name, function_args),
            asset_matches=asset_matches,
            redacted_args=redacted_args(function_args),
        )
    return approval, bool(getattr(args, "apply", False))


def _handle_breakglass(args) -> int:  # noqa: ANN001
    action = getattr(args, "breakglass_action", None)
    if action in {"create", "request"}:
        approval, applied = _create_breakglass_approval(args)
        payload = _approval_to_dict(approval)
        payload["dry_run"] = not applied
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"approval_id: {approval.approval_id}")
            print(f"gate: {approval.gate}")
            print(f"tool: {approval.tool_name}")
            print(f"asset_matches: {', '.join(approval.asset_matches)}")
            print(f"expires_at: {approval.expires_at}")
            print("status: written" if applied else "status: dry-run; re-run with --apply to write")
        return 0

    if action == "list":
        store = _breakglass_store(getattr(args, "store", ""))
        payload = [_approval_to_dict(approval) for approval in store.list()]
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            if not payload:
                print("No break-glass approvals found.")
            for item in payload:
                print(f"{item['approval_id']} {item['gate']} {item['tool_name']} expires={item['expires_at']} revoked={item['revoked']}")
        return 0

    if action == "revoke":
        store = _breakglass_store(getattr(args, "store", ""))
        try:
            approval = store.revoke(getattr(args, "approval_id", ""))
        except KeyError as exc:
            raise SystemExit(f"unknown break-glass approval: {getattr(args, 'approval_id', '')}") from exc
        payload = _approval_to_dict(approval)
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"revoked: {approval.approval_id}")
        return 0

    print("Usage: hermes agentcyber breakglass {create,list,revoke}")
    return 2


def agentcyber_command(args) -> int:  # noqa: ANN001
    from hermes_cli.config import load_config, save_config

    action = getattr(args, "agentcyber_action", None) or "status"
    if action == "status":
        report = build_status_report(load_config(), platform=getattr(args, "platform", "cli"))
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(_format_status(report))
        return 0

    if action == "breakglass":
        return _handle_breakglass(args)

    if action == "setup":
        new_config = apply_agentcyber_setup(
            load_config(),
            platform=getattr(args, "platform", "cli"),
            provider=getattr(args, "provider", "ollama"),
            model=getattr(args, "model", "qwen3-coder:30b"),
            base_url=getattr(args, "base_url", "http://192.168.1.120:11434/v1"),
            api_mode=getattr(args, "api_mode", "chat_completions"),
            enable_live_usb=bool(getattr(args, "enable_live_usb", False)),
        )
        if getattr(args, "apply", False):
            save_config(new_config)
            print("AgentCyber config updated. Start a new session or /reset for tool changes to apply.")
        else:
            print(json.dumps(build_status_report(new_config, platform=getattr(args, "platform", "cli"), health_fn=None), indent=2, sort_keys=True))
            print("\nDry run only. Re-run with --apply to write config.yaml.")
        return 0

    print("Usage: hermes agentcyber {status,setup}")
    return 2
