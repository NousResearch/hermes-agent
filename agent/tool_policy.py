"""Unified tool policy evaluation and signed audit logging."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hmac
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from hermes_constants import get_hermes_home
from tools.threat_patterns import scan_for_threats


POLICY_MODES = {"personal_trusted", "confirm_dangerous", "read_only", "project_sandbox", "team_governed"}
MUTATING_TOOLS = {
    "terminal", "execute_code", "write_file", "patch", "browser_click",
    "browser_type", "browser_press", "browser_scroll", "browser_navigate",
    "send_message", "cronjob", "memory", "skill_manage", "process",
}
SHELL_TOOLS = {"terminal", "execute_code", "process"}
FILE_TOOLS = {"read_file", "write_file", "patch", "search_files"}
NETWORK_TOOLS = {"web_search", "web_extract", "browser_navigate"}
BROWSER_TOOLS = {name for name in MUTATING_TOOLS if name.startswith("browser_")}
SHELL_INJECTION_RE = re.compile(r"(\|\s*(?:sh|bash)|;\s*(?:rm|curl|wget)|`[^`]+`|\$\([^)]*\))")
SECRET_EXFIL_RE = re.compile(r"(\.env\b|auth\.json\b|api[_-]?key|token|secret|/\.ssh/|\.netrc\b)", re.I)


@dataclass(frozen=True)
class ToolPolicyDecision:
    action: str
    mode: str
    risk: str
    reason: str
    categories: tuple[str, ...]
    preview: str

    @property
    def allows_execution(self) -> bool:
        return self.action != "block"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "mode": self.mode,
            "risk": self.risk,
            "reason": self.reason,
            "categories": list(self.categories),
            "preview": self.preview,
        }


def load_policy_config(config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    if isinstance(config, Mapping) and any(key in config for key in ("mode", "allow", "deny")):
        return config
    security = config.get("security") if isinstance(config, Mapping) else {}
    policy = security.get("tool_policy") if isinstance(security, Mapping) else {}
    return policy if isinstance(policy, Mapping) else {}


def evaluate_tool_policy(
    tool_name: str,
    args: Mapping[str, Any] | None,
    *,
    toolset: str | None = None,
    cwd: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> ToolPolicyDecision:
    policy = load_policy_config(config)
    mode = str(policy.get("mode") or "confirm_dangerous").strip() or "confirm_dangerous"
    if mode not in POLICY_MODES:
        mode = "confirm_dangerous"
    args = args or {}
    categories = _categories(tool_name, toolset)
    risk, reason = _risk(tool_name, args, categories)

    deny_reason = _rule_match(policy.get("deny"), tool_name, args, toolset=toolset)
    if deny_reason:
        return _decision("block", mode, "high", deny_reason, categories, tool_name)
    allow_reason = _rule_match(policy.get("allow"), tool_name, args, toolset=toolset)
    if allow_reason:
        return _decision("allow", mode, risk, f"allow rule matched: {allow_reason}", categories, tool_name)

    if mode == "personal_trusted":
        return _decision("allow", mode, risk, reason or "trusted mode", categories, tool_name)
    if mode == "read_only" and tool_name in MUTATING_TOOLS:
        return _decision("block", mode, "high", "read-only policy blocks mutating tools", categories, tool_name)
    if mode == "project_sandbox" and _path_outside_cwd(args, cwd):
        return _decision("block", mode, "high", "project sandbox blocks paths outside cwd", categories, tool_name)
    if mode == "team_governed" and risk == "high":
        return _decision("block", mode, risk, reason or "team-governed policy blocks high-risk action", categories, tool_name)
    return _decision("allow", mode, risk, reason or "policy allows execution", categories, tool_name)


def preview_tool_policy(tool_name: str, args: Mapping[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
    return evaluate_tool_policy(tool_name, args, **kwargs).to_dict()


def audit_tool_policy(
    tool_name: str,
    args: Mapping[str, Any] | None,
    decision: ToolPolicyDecision,
    *,
    session_id: str = "",
    tool_call_id: str = "",
) -> None:
    if decision.risk != "high" and decision.action != "block":
        return
    path = get_hermes_home() / "logs" / "tool_policy_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "tool_name": tool_name,
        "args_preview": _redacted_args(args or {}),
        "decision": decision.to_dict(),
        "session_id": session_id,
        "tool_call_id": tool_call_id,
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    signature = hmac.new(_audit_key(), payload.encode("utf-8"), "sha256").hexdigest()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"record": record, "signature": signature}, sort_keys=True) + "\n")


def _audit_key() -> bytes:
    key_path = get_hermes_home() / "logs" / ".tool_policy_audit.key"
    if key_path.exists():
        return key_path.read_bytes()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    key_path.write_bytes(key)
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    return key


def _decision(action: str, mode: str, risk: str, reason: str, categories: tuple[str, ...], tool_name: str) -> ToolPolicyDecision:
    return ToolPolicyDecision(action=action, mode=mode, risk=risk, reason=reason, categories=categories, preview=f"{action.upper()} {tool_name}: {reason}")


def _categories(tool_name: str, toolset: Optional[str]) -> tuple[str, ...]:
    categories = []
    if tool_name in SHELL_TOOLS:
        categories.append("shell")
    if tool_name in FILE_TOOLS:
        categories.append("file")
    if tool_name in NETWORK_TOOLS:
        categories.append("network")
    if tool_name in BROWSER_TOOLS:
        categories.append("browser")
    if tool_name.startswith("mcp_") or (toolset or "").startswith("mcp-"):
        categories.append("mcp")
    if toolset and toolset.startswith("plugin:"):
        categories.append("plugin")
    if SECRET_EXFIL_RE.search(tool_name):
        categories.append("credential")
    return tuple(categories or ["tool"])


def _risk(tool_name: str, args: Mapping[str, Any], categories: tuple[str, ...]) -> tuple[str, str]:
    text = json.dumps(args, default=str, ensure_ascii=False)
    threats = scan_for_threats(text, scope="strict")
    if threats:
        return "high", f"prompt-injection/exfiltration pattern: {', '.join(threats)}"
    if "shell" in categories and SHELL_INJECTION_RE.search(text):
        return "high", "shell injection pattern detected"
    if SECRET_EXFIL_RE.search(text):
        return "high", "secret or credential access pattern detected"
    if tool_name in MUTATING_TOOLS:
        return "medium", "mutating tool"
    return "low", "read-only or low-risk tool"


def _path_outside_cwd(args: Mapping[str, Any], cwd: str | None) -> bool:
    if not cwd:
        return False
    root = Path(cwd).expanduser().resolve()
    for key in ("path", "file_path", "target_path", "output_path"):
        value = args.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            return True
    return False


def _rule_match(rules: Any, tool_name: str, args: Mapping[str, Any], *, toolset: str | None) -> str:
    if not isinstance(rules, Mapping):
        return ""
    if _glob_any(tool_name, rules.get("commands")):
        return "command"
    if toolset and _glob_any(toolset, rules.get("toolsets")):
        return "toolset"
    for value in _arg_paths(args):
        if _glob_any(value, rules.get("paths")):
            return "path"
    domain = _domain_from_args(args)
    if domain and _glob_any(domain, rules.get("domains")):
        return "domain"
    provider = str(args.get("provider") or "")
    if provider and _glob_any(provider, rules.get("providers")):
        return "provider"
    return ""


def _glob_any(value: str, patterns: Any) -> bool:
    if not isinstance(patterns, list):
        return False
    return any(fnmatch.fnmatch(value, str(pattern)) for pattern in patterns)


def _arg_paths(args: Mapping[str, Any]) -> list[str]:
    return [str(args[key]) for key in ("path", "file_path", "target_path", "output_path") if isinstance(args.get(key), str)]


def _domain_from_args(args: Mapping[str, Any]) -> str:
    for key in ("url", "uri", "target_url"):
        value = args.get(key)
        if isinstance(value, str):
            return urlparse(value).hostname or ""
    return ""


def _redacted_args(args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: ("[REDACTED]" if re.search(r"(key|token|secret|password)", str(key), re.I) else value)
        for key, value in args.items()
    }
