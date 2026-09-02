"""Redacted runtime capability registry for Hermes profiles."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import yaml

from hermes_cli.profiles import get_profile_dir, list_profiles as _list_profile_infos, normalize_profile_name, profile_exists

CapabilityKind = Literal["tool", "toolset", "mcp", "composio"]
RiskClass = Literal["READ", "PREPARE", "CONSEQUENTIAL_WRITE", "CREDENTIAL_EXPORT"]

# Toolkits Composio can satisfy for the Executive Capability Bus.  Phase 1
# keeps this deliberately explicit and redacted: the registry records that a
# profile can route a capability through its configured Composio MCP server;
# Composio itself remains the credential boundary and no account/token values
# are exposed here.
_COMPOSIO_TOOLKIT_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "vercel": ("mcp:vercel", "composio:vercel", "toolkit:vercel"),
}


@dataclass
class WorkloadInfo:
    running_count: int = 0
    max_concurrency: Optional[int] = None
    queued_count: int = 0

    @property
    def capacity_remaining(self) -> Optional[int]:
        if self.max_concurrency is None:
            return None
        return max(0, int(self.max_concurrency) - int(self.running_count))

    @property
    def saturated(self) -> bool:
        return self.capacity_remaining == 0 if self.max_concurrency is not None else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "running_count": self.running_count,
            "max_concurrency": self.max_concurrency,
            "queued_count": self.queued_count,
            "capacity_remaining": self.capacity_remaining,
            "saturated": self.saturated,
        }


@dataclass
class ProfileCapability:
    profile: str
    capability: str
    kind: CapabilityKind
    configured: bool = False
    enabled: bool = False
    tool_exists: bool = False
    credential_present: bool = False
    credential_usable: Optional[bool] = None
    credential_check: str = "not_tested"
    gateway_running: bool = False
    worker_available: bool = False
    risk_class: RiskClass = "READ"
    source: str = ""
    notes: list[str] = field(default_factory=list)
    workload: WorkloadInfo = field(default_factory=WorkloadInfo)
    rank_score: float = 0.0
    rank_reasons: list[str] = field(default_factory=list)

    @property
    def executable(self) -> bool:
        if self.kind == "mcp":
            return self.configured and self.enabled and self.worker_available
        if self.kind == "composio":
            return self.configured and self.enabled and self.worker_available and self.credential_present
        return self.enabled and self.worker_available

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workload"] = self.workload.to_dict()
        data["executable"] = self.executable
        return data


@dataclass
class CapabilityQueryResult:
    capability: str
    profiles: list[ProfileCapability]
    generated_at: float
    recommendation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "profiles": [p.to_dict() for p in self.profiles],
            "generated_at": self.generated_at,
            "recommendation": self.recommendation,
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _boolish(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def list_profiles(include_default: bool = True) -> list[str]:
    names = [p.name for p in _list_profile_infos()]
    # Some non-interactive/import-only contexts do not return every on-disk
    # profile through the CLI helper. Fall back to the profile directory so
    # cross-profile capability discovery can still see worker profiles such as
    # the VPS `director` profile.
    try:
        base = Path.home() / ".hermes" / "profiles"
        if base.exists():
            names.extend(p.name for p in base.iterdir() if p.is_dir())
    except Exception:
        pass
    if include_default and "default" not in names:
        names.insert(0, "default")
    if not include_default:
        names = [n for n in names if n != "default"]
    return sorted(set(names), key=lambda n: (n != "default", n))


def profile_home(profile: str) -> Path:
    return get_profile_dir(normalize_profile_name(profile))


def _credential_present(home: Path, server_name: str, cfg: dict[str, Any]) -> bool:
    token_dir = home / "mcp-tokens"
    for suffix in (".json", ".client.json", ".meta.json"):
        if (token_dir / f"{server_name}{suffix}").exists():
            return True
    headers = cfg.get("headers")
    if isinstance(headers, dict) and headers:
        return True
    env_keys = cfg.get("env") or cfg.get("env_vars")
    if isinstance(env_keys, dict):
        return any(bool(v) for v in env_keys.values())
    if isinstance(env_keys, list):
        return any(bool(os.getenv(str(v))) for v in env_keys)
    return False


def collect_workload(*, profiles: Optional[Iterable[str]] = None, max_concurrency: Optional[int] = None) -> dict[str, WorkloadInfo]:
    wanted = {normalize_profile_name(p) for p in profiles} if profiles else None
    info: dict[str, WorkloadInfo] = {}
    if wanted:
        for p in wanted:
            info[p] = WorkloadInfo(max_concurrency=max_concurrency)
    try:
        from hermes_cli import kanban_db as kb
        with kb.connect_closing() as conn:
            rows = conn.execute(
                "SELECT assignee, status, COUNT(*) AS n FROM tasks "
                "WHERE assignee IS NOT NULL AND status IN ('running','ready') "
                "GROUP BY assignee, status"
            ).fetchall()
            for row in rows:
                assignee = normalize_profile_name(row["assignee"])
                if wanted and assignee not in wanted:
                    continue
                item = info.setdefault(assignee, WorkloadInfo(max_concurrency=max_concurrency))
                if row["status"] == "running":
                    item.running_count = int(row["n"])
                elif row["status"] == "ready":
                    item.queued_count = int(row["n"])
    except Exception:
        pass
    return info


def inspect_profile_capabilities(
    profile: str,
    *,
    test_credentials: bool = False,
    include_disabled: bool = True,
    workload: Optional[WorkloadInfo] = None,
) -> list[ProfileCapability]:
    profile = normalize_profile_name(profile)
    home = profile_home(profile)
    cfg = _read_yaml(home / "config.yaml")
    worker = profile_exists(profile)
    gateway = False
    try:
        for p in _list_profile_infos():
            if p.name == profile:
                gateway = bool(p.gateway_running)
                break
    except Exception:
        gateway = False
    wl = workload or WorkloadInfo()
    capabilities: list[ProfileCapability] = []

    raw_toolsets: list[Any] = []
    top_toolsets = cfg.get("toolsets") or cfg.get("tools", {}).get("toolsets") or []
    if isinstance(top_toolsets, str):
        raw_toolsets.append(top_toolsets)
    elif isinstance(top_toolsets, list):
        raw_toolsets.extend(top_toolsets)

    # Tool enablement is platform-scoped in normal gateway setups
    # (`platform_toolsets.discord`, `platform_toolsets.cli`, etc.). The
    # capability bus needs to advertise a profile as capable when the toolset
    # is enabled on any platform, not only when it appears in the legacy
    # top-level `toolsets` list.
    platform_toolsets = cfg.get("platform_toolsets") or {}
    if isinstance(platform_toolsets, dict):
        for values in platform_toolsets.values():
            if isinstance(values, str):
                raw_toolsets.append(values)
            elif isinstance(values, list):
                raw_toolsets.extend(values)

    for ts in sorted({str(t) for t in raw_toolsets if t}):
        capabilities.append(ProfileCapability(
            profile=profile,
            capability=f"toolset:{ts}",
            kind="toolset",
            configured=True,
            enabled=True,
            tool_exists=True,
            gateway_running=gateway,
            worker_available=worker,
            source=str(home / "config.yaml"),
            workload=wl,
        ))
        try:
            from toolsets import resolve_toolset
            for tool in resolve_toolset(ts):
                capabilities.append(ProfileCapability(
                    profile=profile,
                    capability=f"tool:{tool}",
                    kind="tool",
                    configured=True,
                    enabled=True,
                    tool_exists=True,
                    gateway_running=gateway,
                    worker_available=worker,
                    source=f"toolset:{ts}",
                    workload=wl,
                ))
        except Exception:
            pass

    mcp_servers = cfg.get("mcp_servers") or {}
    if isinstance(mcp_servers, dict):
        for name, server_cfg in sorted(mcp_servers.items()):
            if not isinstance(server_cfg, dict):
                server_cfg = {}
            enabled = _boolish(server_cfg.get("enabled"), default=True)
            if not enabled and not include_disabled:
                continue
            credential_present = _credential_present(home, str(name), server_cfg)
            credential_check = "not_tested"
            usable: Optional[bool] = None
            if not enabled:
                credential_check = "disabled"
            elif not test_credentials:
                credential_check = "not_tested"
            cap = ProfileCapability(
                profile=profile,
                capability=f"mcp:{name}",
                kind="mcp",
                configured=True,
                enabled=enabled,
                tool_exists=enabled,
                credential_present=credential_present,
                credential_usable=usable,
                credential_check=credential_check,
                gateway_running=gateway,
                worker_available=worker,
                source=str(home / "config.yaml"),
                workload=wl,
            )
            if not enabled:
                cap.notes.append("MCP server is configured but disabled.")
            if credential_present:
                cap.notes.append("Credential material appears present; values redacted.")
            capabilities.append(cap)

    composio_cfg = mcp_servers.get("composio") if isinstance(mcp_servers, dict) else None
    if isinstance(composio_cfg, dict):
        composio_enabled = _boolish(composio_cfg.get("enabled"), default=True)
        if composio_enabled or include_disabled:
            for toolkit, aliases in sorted(_COMPOSIO_TOOLKIT_CAPABILITIES.items()):
                for alias in aliases:
                    cap = ProfileCapability(
                        profile=profile,
                        capability=alias,
                        kind="composio",
                        configured=True,
                        enabled=composio_enabled,
                        tool_exists=composio_enabled,
                        credential_present=composio_enabled,
                        credential_usable=None,
                        credential_check="composio_connection_assumed" if composio_enabled else "disabled",
                        gateway_running=gateway,
                        worker_available=worker,
                        source=f"mcp:composio/toolkit:{toolkit}",
                        workload=wl,
                    )
                    cap.notes.append(
                        f"Routed through the profile's Composio MCP server for toolkit:{toolkit}; "
                        "Composio owns credential isolation and token values remain hidden."
                    )
                    if not composio_enabled:
                        cap.notes.append("Composio MCP server is configured but disabled.")
                    capabilities.append(cap)
    return capabilities


def _score(cap: ProfileCapability) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if cap.executable:
        score += 100; reasons.append("executable")
    if cap.enabled:
        score += 25; reasons.append("enabled")
    if cap.credential_present:
        score += 20; reasons.append("credential_present")
    if cap.credential_usable is True:
        score += 15; reasons.append("credential_usable")
    elif cap.credential_usable is False:
        score -= 50; reasons.append("credential_failed")
    if cap.worker_available:
        score += 10; reasons.append("worker_available")
    if cap.gateway_running:
        score += 5; reasons.append("gateway_running")
    if cap.workload.max_concurrency is not None:
        if cap.workload.saturated:
            score -= 40; reasons.append("profile_saturated")
        else:
            score += 5; reasons.append("capacity_available")
    # Workload is intentionally heavy enough to beat small domain hints; an
    # idle capable peer should outrank a busy preferred owner in Phase 1.
    score -= cap.workload.running_count * 20
    score -= cap.workload.queued_count * 10
    if cap.workload.running_count:
        reasons.append(f"running_count={cap.workload.running_count}")
    if cap.workload.queued_count:
        reasons.append(f"queued_count={cap.workload.queued_count}")
    if cap.profile == "cto" and cap.capability.startswith(("mcp:github", "tool:terminal")):
        score += 3; reasons.append("domain_hint_cto")
    if cap.profile == "default" and cap.source.startswith("mcp:composio/toolkit:vercel"):
        score += 8; reasons.append("domain_hint_default_composio_vercel")
    if cap.profile == "cto" and cap.capability.startswith("mcp:vercel") and cap.kind != "composio":
        score += 3; reasons.append("domain_hint_cto_native_vercel")
    return score, reasons


def build_capability_registry(
    *,
    profiles: Optional[list[str]] = None,
    test_credentials: bool = False,
    include_disabled: bool = True,
    max_concurrency: Optional[int] = None,
) -> dict[str, list[ProfileCapability]]:
    names = profiles or list_profiles()
    workload = collect_workload(profiles=names, max_concurrency=max_concurrency)
    registry: dict[str, list[ProfileCapability]] = {}
    for name in names:
        wl = workload.get(normalize_profile_name(name), WorkloadInfo(max_concurrency=max_concurrency))
        for cap in inspect_profile_capabilities(
            name,
            test_credentials=test_credentials,
            include_disabled=include_disabled,
            workload=wl,
        ):
            cap.rank_score, cap.rank_reasons = _score(cap)
            registry.setdefault(cap.capability, []).append(cap)
    for caps in registry.values():
        caps.sort(key=lambda c: c.rank_score, reverse=True)
    return registry


def find_capability(
    capability: str,
    *,
    requester_profile: Optional[str] = None,
    risk: RiskClass = "READ",
    test_credentials: bool = False,
    include_disabled: bool = False,
    profiles: Optional[list[str]] = None,
    max_concurrency: Optional[int] = None,
) -> CapabilityQueryResult:
    registry = build_capability_registry(
        profiles=profiles,
        test_credentials=test_credentials,
        include_disabled=include_disabled,
        max_concurrency=max_concurrency,
    )
    caps = list(registry.get(capability, []))
    if requester_profile:
        requester = normalize_profile_name(requester_profile)
        # Prefer another executor when scores tie; requester can still win if it is the only/best owner.
        for cap in caps:
            if cap.profile == requester:
                cap.rank_score -= 1
                cap.rank_reasons.append("requester_profile_deprioritized")
        caps.sort(key=lambda c: c.rank_score, reverse=True)
    executable = [c for c in caps if c.executable and not c.workload.saturated]
    best = executable[0] if executable else (caps[0] if caps else None)
    recommendation = {
        "status": "executable" if executable else ("configured_not_executable" if caps else "not_found"),
        "best_profile": best.profile if best else None,
        "reason": ("best workload-aware executor selected" if executable else ("capability found but not currently executable" if caps else "no profile advertises this capability")),
    }
    return CapabilityQueryResult(capability=capability, profiles=caps, generated_at=time.time(), recommendation=recommendation)


def to_json(result: CapabilityQueryResult | dict[str, list[ProfileCapability]]) -> str:
    if isinstance(result, CapabilityQueryResult):
        data = result.to_dict()
    else:
        data = {k: [c.to_dict() for c in v] for k, v in result.items()}
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
