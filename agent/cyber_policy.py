"""AgentCyber asset registry and S0-S5 execution gate helpers.

This module is intentionally deterministic and side-effect-light: it loads the
operator's authorized asset registry from config/file/env, classifies a proposed
tool call into an AgentCyber execution gate, and returns allow/block metadata.
It does not execute tools itself.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hermes_constants import get_hermes_home


_GATE_ORDER = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
_READ_ONLY_TOOLS = frozenset({
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
    "session_search",
    "vision_analyze",
    "browser_snapshot",
    "browser_get_images",
    "browser_console",
    "mcp_azure_manager_list_subscriptions",
    "mcp_azure_manager_list_resource_groups",
    "mcp_azure_manager_list_azure_resources",
    "mcp_azure_manager_search_resources",
    "mcp_azure_manager_get_vm_status",
    "mcp_azure_manager_list_vms",
    "mcp_azure_manager_list_storage_accounts",
    "mcp_azure_manager_get_cost_summary",
})
_MUTATING_TOOLS = frozenset({
    "write_file",
    "patch",
    "skill_manage",
    "terminal",
    "execute_code",
    "browser_click",
    "browser_type",
    "browser_press",
    "send_message",
    "cronjob",
})
_SECURITY_TOOLS = frozenset({
    "network_scan",
    "threat_intel",
    "vuln_triage",
    "extract_iocs",
    "ir_incident",
})
_READ_ONLY_SECURITY_TOOLS = frozenset({
    "extract_iocs",
    "threat_intel",
    "vuln_triage",
})
_LOCAL_INCIDENT_TOOLS = frozenset({"ir_incident"})
_DESTRUCTIVE_RE = re.compile(
    r"\b(rm\s+-rf|mkfs|dd\s+if=|shred|wipe|format\s+disk|factory\s+reset|"
    r"delete\s+all|rotate\s+credentials?|password\s+reset|disable\s+account|"
    r"iptables\s+-F|ufw\s+disable|firewall\s+reset|reset\s+the\s+firewall)\b",
    re.IGNORECASE,
)
_RECON_RE = re.compile(r"\b(nmap|masscan|zmap|rustscan|nikto|sqlmap|ffuf|gobuster|scan)\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?(?![\d.])")
_DOMAIN_RE = re.compile(r"\b(?=.{4,253}\b)(?:[a-zA-Z0-9-]{1,63}\.)+[a-zA-Z]{2,63}\b")
_VM_ID_RE = re.compile(r"\bvm\s*#?\s*(\d{1,5})\b", re.IGNORECASE)
_LOCAL_FILE_DOMAIN_SUFFIXES = frozenset({
    "bash",
    "cfg",
    "css",
    "csv",
    "ini",
    "js",
    "json",
    "jsx",
    "lock",
    "log",
    "md",
    "py",
    "ps1",
    "rst",
    "sh",
    "sql",
    "toml",
    "ts",
    "tsx",
    "txt",
    "xml",
    "yaml",
    "yml",
    "zsh",
})


@dataclass(frozen=True)
class AuthorizedAsset:
    name: str
    identifiers: tuple[str, ...]
    tags: tuple[str, ...] = ()
    allowed_gates: tuple[str, ...] = ("S0", "S1", "S2", "S3")


@dataclass(frozen=True)
class AssetRegistry:
    assets: tuple[AuthorizedAsset, ...]
    source: str

    def matches_any(self, candidates: list[str], *, gate: str = "S2") -> bool:
        return bool(self.matching_assets(candidates, gate=gate))

    def matching_assets(self, candidates: list[str], *, gate: str = "S2") -> list[AuthorizedAsset]:
        gate_level = _GATE_ORDER.get(gate, 0)
        matches: list[AuthorizedAsset] = []
        for asset in self.assets:
            max_gate = max((_GATE_ORDER.get(g, -1) for g in asset.allowed_gates), default=-1)
            if max_gate < gate_level:
                continue
            if any(_identifier_matches(identifier, candidate) for identifier in asset.identifiers for candidate in candidates):
                matches.append(asset)
        return matches


@dataclass(frozen=True)
class ExecutionGateDecision:
    gate: str
    allowed: bool
    reason: str
    asset_matches: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    approval_id: str = ""

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "gate": self.gate,
            "allowed": self.allowed,
            "reason": self.reason,
            "asset_matches": list(self.asset_matches),
            "candidates": list(self.candidates),
        }
        if self.approval_id:
            metadata["breakglass_approval_id"] = self.approval_id
        return metadata


def builtin_bc_asset_registry() -> AssetRegistry:
    """Return the built-in BC-owned/lab registry for AgentCyber forks.

    These are identifiers, not credentials. Operators can override/extend them
    with ``agent_cyber.asset_registry`` in config or a registry file.
    """

    return AssetRegistry(
        assets=(
            AuthorizedAsset(
                name="breaking-circuits-public",
                identifiers=(
                    "breakingcircuits.com",
                    "*.breakingcircuits.com",
                    "bde.it.com",
                    "*.bde.it.com",
                    "beforedisaster.org",
                    "*.beforedisaster.org",
                ),
                tags=("bc-owned", "public"),
                allowed_gates=("S0", "S1", "S2", "S3"),
            ),
            AuthorizedAsset(
                name="bc-lab-lan",
                identifiers=("192.168.1.0/24",),
                tags=("bc-owned", "lab", "lan"),
                allowed_gates=("S0", "S1", "S2", "S3"),
            ),
            AuthorizedAsset(
                name="bc-lab-key-hosts",
                identifiers=(
                    "192.168.1.115",
                    "192.168.1.120",
                    "192.168.1.121",
                    "192.168.1.122",
                    "192.168.1.137",
                    "vm 112",
                    "vm112",
                ),
                tags=("bc-owned", "lab"),
                allowed_gates=("S0", "S1", "S2", "S3"),
            ),
        ),
        source="builtin:breaking-circuits",
    )


def load_asset_registry(config: dict[str, Any] | None = None) -> AssetRegistry:
    """Load the authorized asset registry from config/file/env plus BC defaults."""

    config = config if isinstance(config, dict) else _load_config_quietly()
    cyber_cfg = config.get("agent_cyber", {}) if isinstance(config, dict) else {}
    if not isinstance(cyber_cfg, dict):
        cyber_cfg = {}
    registry_cfg = cyber_cfg.get("asset_registry", {})
    assets: list[AuthorizedAsset] = []
    sources: list[str] = []

    include_builtin = bool(cyber_cfg.get("include_builtin_bc_assets", True))
    if include_builtin:
        builtin = builtin_bc_asset_registry()
        assets.extend(builtin.assets)
        sources.append(builtin.source)

    file_path = ""
    inline_assets: Any = None
    if isinstance(registry_cfg, str):
        file_path = registry_cfg
    elif isinstance(registry_cfg, dict):
        file_path = str(registry_cfg.get("file") or registry_cfg.get("path") or "").strip()
        inline_assets = registry_cfg.get("assets")
    elif isinstance(registry_cfg, list):
        inline_assets = registry_cfg

    env_path = os.getenv("HERMES_AGENTCYBER_ASSET_REGISTRY", "").strip()
    if env_path:
        file_path = env_path

    if file_path:
        path = _expand_path(file_path)
        loaded = _load_registry_file(path)
        if loaded:
            assets.extend(loaded)
            sources.append(str(path))

    if inline_assets:
        loaded_inline = _parse_assets(inline_assets)
        if loaded_inline:
            assets.extend(loaded_inline)
            sources.append("config:agent_cyber.asset_registry.assets")

    return AssetRegistry(assets=tuple(_dedupe_assets(assets)), source=", ".join(sources) or "empty")


def evaluate_execution_gate(
    tool_name: str,
    function_args: dict[str, Any] | None,
    *,
    route: str | None = None,
    registry: AssetRegistry | None = None,
    config: dict[str, Any] | None = None,
) -> ExecutionGateDecision:
    """Classify a tool call into S0-S5 and decide whether it may execute."""

    function_args = function_args or {}
    config = config if isinstance(config, dict) else _load_config_quietly()
    cyber_cfg = config.get("agent_cyber", {}) if isinstance(config, dict) else {}
    if not isinstance(cyber_cfg, dict):
        cyber_cfg = {}
    gates_cfg = cyber_cfg.get("execution_gates", {}) if isinstance(cyber_cfg.get("execution_gates", {}), dict) else {}
    enabled = bool(gates_cfg.get("enabled", True))
    if not enabled:
        return ExecutionGateDecision("S0", True, "AgentCyber execution gates disabled")

    registry = registry or load_asset_registry(config)
    gate, reason, candidates = classify_tool_gate(tool_name, function_args, route=route)

    if gate in {"S0", "S1"}:
        return ExecutionGateDecision(gate, True, reason, candidates=tuple(candidates))

    if gate == "S5":
        matched = registry.matching_assets(candidates, gate="S3") if candidates else []
        asset_matches = tuple(asset.name for asset in matched)
        from agent.cyber_breakglass import BreakGlassStore, extract_approval_id, validate_approval

        store_path = gates_cfg.get("breakglass_store") or gates_cfg.get("breakglass_store_path")
        store = BreakGlassStore(Path(str(store_path)).expanduser()) if store_path else None
        approval_check = validate_approval(
            approval_id=extract_approval_id(function_args),
            tool_name=tool_name,
            function_args=function_args,
            gate=gate,
            asset_matches=asset_matches,
            store=store,
        )
        if approval_check.allowed and approval_check.approval is not None:
            return ExecutionGateDecision(
                gate,
                True,
                approval_check.reason,
                asset_matches=asset_matches,
                candidates=tuple(candidates),
                approval_id=approval_check.approval.approval_id,
            )
        return ExecutionGateDecision(
            gate,
            False,
            f"S5 destructive/external-high-risk action requires valid explicit human approval outside autonomous tool flow: {approval_check.reason}",
            asset_matches=asset_matches,
            candidates=tuple(candidates),
            approval_id=approval_check.approval.approval_id if approval_check.approval else "",
        )

    matched = registry.matching_assets(candidates, gate=gate) if candidates else []
    if matched or (not candidates and gate == "S3"):
        return ExecutionGateDecision(
            gate,
            True,
            reason if candidates else f"{reason} (local-only)",
            asset_matches=tuple(asset.name for asset in matched) if matched else ("local-system",),
            candidates=tuple(candidates),
        )

    # S2+ with no clear authorized asset stays blocked. Operators can extend the
    # registry or rephrase with an in-scope identifier.
    return ExecutionGateDecision(
        gate,
        False,
        f"{gate} action needs a matching authorized asset registry entry before tool execution",
        candidates=tuple(candidates),
    )


def classify_tool_gate(tool_name: str, function_args: dict[str, Any], *, route: str | None = None) -> tuple[str, str, list[str]]:
    name = (tool_name or "").strip()
    arg_text = json.dumps(function_args or {}, ensure_ascii=False, sort_keys=True)
    candidates = _extract_candidates(arg_text)
    route = (route or "").strip().lower()

    if name in _READ_ONLY_TOOLS:
        return "S1", "read-only information retrieval", candidates
    if name in _READ_ONLY_SECURITY_TOOLS:
        return "S1", "read-only cyber analysis/intel helper", candidates
    if name in _LOCAL_INCIDENT_TOOLS:
        return "S3", "local incident-response state mutation", candidates
    if _DESTRUCTIVE_RE.search(arg_text) or route == "destructive_high_risk":
        return "S5", "destructive or high-impact operation", candidates
    if name == "network_scan" or _RECON_RE.search(arg_text):
        return "S2", "controlled reconnaissance/scanning", candidates
    if route in {"credentials_sensitive", "ir_breakglass"}:
        return "S3", "credential-sensitive or incident-recovery action", candidates
    if route in {"cyber_lab", "malware_re"} and name in _MUTATING_TOOLS:
        return "S3", "cyber lab action with mutation/execution capability", candidates
    if name in _MUTATING_TOOLS:
        return "S3", "local mutation or command execution", candidates
    if name in _SECURITY_TOOLS:
        return "S2", "cybersecurity tool action", candidates
    return "S1", "default low-risk tool call", candidates


def gate_tool_call_for_agent(agent: Any, function_name: str, function_args: dict[str, Any] | None) -> ExecutionGateDecision:
    metadata = getattr(agent, "_current_cyber_route_metadata", None)
    route = metadata.get("route") if isinstance(metadata, dict) else None
    config = getattr(agent, "_agentcyber_config", None)
    if config is None:
        config = _load_config_quietly()
        agent._agentcyber_config = config
    registry = getattr(agent, "_agentcyber_asset_registry", None)
    if registry is None:
        registry = load_asset_registry(config)
        agent._agentcyber_asset_registry = registry
    decision = evaluate_execution_gate(function_name, function_args, route=route, registry=registry, config=config)
    agent._current_agentcyber_execution_gate = decision.to_metadata()
    return decision


def tool_block_message(decision: ExecutionGateDecision) -> str:
    return json.dumps(
        {
            "error": "AgentCyber execution gate blocked tool call",
            "agentcyber_gate": decision.to_metadata(),
            "next_steps": [
                "Add the target to agent_cyber.asset_registry if it is BC-owned or lab-authorized.",
                "Use read-only triage first for unknown third-party assets.",
                "For destructive S5 actions, require explicit human approval/break-glass outside autonomous execution.",
            ],
        },
        ensure_ascii=False,
    )


def _extract_candidates(text: str) -> list[str]:
    found: list[str] = []
    for match in _IPV4_RE.findall(text or ""):
        try:
            ipaddress.ip_network(match, strict=False)
            found.append(match)
        except ValueError:
            pass
    for match in _DOMAIN_RE.findall(text or ""):
        candidate = match.lower()
        if _looks_like_local_file_candidate(candidate):
            continue
        found.append(candidate)
    for match in _VM_ID_RE.findall(text or ""):
        vm_id = str(match).lstrip("0") or "0"
        found.append(f"vm {vm_id}")
        found.append(f"vm{vm_id}")
    # URL parser pass catches hosts where regex context is awkward.
    for token in re.findall(r"https?://[^\s'\"<>]+", text or ""):
        parsed = urlparse(token)
        if parsed.hostname:
            found.append(parsed.hostname.lower())
    return list(dict.fromkeys(found))


def _looks_like_local_file_candidate(candidate: str) -> bool:
    suffix = str(candidate or "").rsplit(".", 1)[-1].lower()
    return suffix in _LOCAL_FILE_DOMAIN_SUFFIXES


def _identifier_matches(identifier: str, candidate: str) -> bool:
    ident = str(identifier or "").strip().lower()
    cand = str(candidate or "").strip().lower()
    if not ident or not cand:
        return False
    cand_host = urlparse(cand).hostname or cand
    try:
        net = ipaddress.ip_network(ident, strict=False)
        if "/" in cand_host:
            cand_net = ipaddress.ip_network(cand_host, strict=False)
            return (
                cand_net.version == net.version
                and int(cand_net.network_address) >= int(net.network_address)
                and int(cand_net.broadcast_address) <= int(net.broadcast_address)
            )
        cand_ip = ipaddress.ip_address(cand_host)
        return cand_ip.version == net.version and cand_ip in net
    except ValueError:
        pass
    if "*" in ident:
        return fnmatch.fnmatch(cand_host, ident)
    return cand_host == ident or cand_host.endswith("." + ident)


def _parse_assets(raw_assets: Any) -> list[AuthorizedAsset]:
    if not isinstance(raw_assets, list):
        return []
    parsed: list[AuthorizedAsset] = []
    for idx, item in enumerate(raw_assets):
        if not isinstance(item, dict):
            continue
        identifiers = item.get("identifiers") or item.get("hosts") or item.get("targets") or []
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        identifiers = tuple(str(v).strip() for v in identifiers if str(v).strip())
        if not identifiers:
            continue
        gates = item.get("allowed_gates") or item.get("gates") or ("S0", "S1", "S2", "S3")
        if isinstance(gates, str):
            gates = [g.strip() for g in gates.split(",")]
        tags = item.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        parsed.append(
            AuthorizedAsset(
                name=str(item.get("name") or f"asset-{idx}").strip(),
                identifiers=identifiers,
                tags=tuple(str(t).strip() for t in tags if str(t).strip()),
                allowed_gates=tuple(g for g in gates if g in _GATE_ORDER) or ("S0", "S1"),
            )
        )
    return parsed


def _load_registry_file(path: Path) -> list[AuthorizedAsset]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            import yaml
            data = yaml.safe_load(raw) or {}
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("assets", [])
    return _parse_assets(data)


def _expand_path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = get_hermes_home() / path
    return path


def _dedupe_assets(assets: list[AuthorizedAsset]) -> list[AuthorizedAsset]:
    seen = set()
    result = []
    for asset in assets:
        key = (asset.name, asset.identifiers)
        if key in seen:
            continue
        seen.add(key)
        result.append(asset)
    return result


def _load_config_quietly() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}
