"""
plugins/sbl — System Boundary Layer

Thin awareness layer at tool-call level. Every write_file/patch/terminal
to system paths (/etc/, /opt/, /usr/) is checked against a dynamically-built
service map. After each write, new paths are learned and persisted.

Architecture:
  pre_tool_call        → classify path → snapshot → dependency lookup
  transform_tool_result → learn new paths
  on_session_start     → single log at session start

No agent memory required. Tool call = runtime, not LLM context.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# FHS classification
SYSTEM_PREFIXES = [
    "/etc/", "/opt/", "/usr/", "/var/lib/", "/var/log/",
    "/var/www/", "/boot/", "/bin/", "/sbin/", "/lib/",
    "/lib64/", "/snap/", "/usr/local/etc/",
]
USER_PREFIXES = ["/home/", "/tmp/", "/root/", "/var/tmp/"]

WRITE_TOOLS = {"write_file", "patch", "terminal"}

TERMINAL_WRITE_PATTERNS = [
    re.compile(r"(?:^|\s)(?:sudo\s+)?(?:echo|cat|printf)\s+.*?(?:>>|>)\s*(\S+)"),
    re.compile(r"(?:^|\s)(?:sudo\s+)?(?:cp|mv)\s+.*\s+(\S+)"),
    re.compile(r"(?:^|\s)(?:sudo\s+)?rm\s+(?:.*\s+)?(\S+)"),
    re.compile(r"(?:^|\s)(?:sudo\s+)?sed\s+-i\s+.*?\s+(\S+)"),
    re.compile(r"(?:^|\s)(?:sudo\s+)?systemctl\s+(?:\S+\s+)?(\S+)"),
]

_KNOWN_CONFIG_PATTERNS = {
    "nginx": ["/etc/nginx/"],
    "xray": ["/usr/local/etc/xray/", "/etc/xray/"],
    "stalwart": ["/opt/stalwart/"],
    "fail2ban": ["/etc/fail2ban/"],
    "ssh": ["/etc/ssh/"],
    "certbot": ["/etc/letsencrypt/"],
    "docker": ["/etc/docker/"],
}
_RUNTIME_SERVICES = {"certbot", "cron", "systemd", "user"}


@dataclass
class ServiceMap:
    services: dict = field(default_factory=dict)
    file_owners: dict = field(default_factory=dict)
    port_owners: dict = field(default_factory=dict)


# Global state
_snapshot_taken: bool = False
_service_map: ServiceMap = ServiceMap()
_change_log: list[dict] = []
_SNAPSHOT_DIR: Optional[Path] = None


# ── FHS Classifier ─────────────────────────────────────────────────────────

def _classify_path(path: str) -> str:
    if not path:
        return "UNKNOWN"
    path_str = str(path).rstrip("/")
    for prefix in SYSTEM_PREFIXES:
        if path_str.startswith(prefix):
            return "SYSTEM"
    for prefix in USER_PREFIXES:
        if path_str.startswith(prefix):
            return "USER"
    return "UNKNOWN"


def _normalize_to_path(tool_name: str, args: dict) -> tuple[str, str]:
    path = ""
    if tool_name == "write_file":
        path = str(args.get("path", ""))
    elif tool_name == "patch":
        path = str(args.get("path", ""))
    elif tool_name == "terminal":
        return _classify_terminal_cmd(str(args.get("command", "")))
    return path, _classify_path(path)


def _classify_terminal_cmd(cmd: str) -> tuple[str, str]:
    if not cmd:
        return "", "UNKNOWN"
    for pattern in TERMINAL_WRITE_PATTERNS:
        m = pattern.search(cmd)
        if m:
            path = m.group(1)
            if not path.startswith("/"):
                if path in _KNOWN_CONFIG_PATTERNS or path in (
                    "nginx", "xray", "stalwart", "fail2ban",
                    "certbot", "docker", "ssh", "networking",
                    "postfix", "dovecot"):
                    return path, "SYSTEM"
                return path, "SYSTEM"
            return path, _classify_path(path)
    if "systemctl" in cmd:
        return "system-command", "SYSTEM"
    if "nginx -t" in cmd:
        return "nginx-config-test", "SYSTEM"
    return "", "UNKNOWN"


# ── Snapshot ───────────────────────────────────────────────────────────────

def _ensure_snapshot_dir() -> Path:
    global _SNAPSHOT_DIR
    if _SNAPSHOT_DIR is None:
        for candidate in [
            Path("/opt/hermes-victim-data/sbl-snapshot"),
            Path("/tmp/sbl-snapshot"),
            Path.home() / ".hermes" / "sbl-snapshot",
        ]:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                _SNAPSHOT_DIR = candidate
                break
            except (OSError, PermissionError):
                continue
    return _SNAPSHOT_DIR


def _take_snapshot() -> ServiceMap:
    """Dynamic audit: systemctl + ss + /proc + heuristics. No hardcoded services."""
    global _snapshot_taken, _service_map
    snap_dir = _ensure_snapshot_dir()
    sm = ServiceMap()

    # Source 1: systemctl — running units
    try:
        result = subprocess.run(
            ["systemctl", "list-units", "--state=running", "--no-legend", "--no-pager"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            (snap_dir / "services.txt").write_text(result.stdout)
            for line in result.stdout.strip().split("\n"):
                parts = line.split(None, 4)
                if not parts:
                    continue
                name = parts[0]
                srv = name.replace(".service", "").split("@")[0]
                if not srv or srv in _RUNTIME_SERVICES:
                    continue
                entry = {"ports": [], "configs": [], "type": "systemd"}
                try:
                    cat = subprocess.run(
                        ["systemctl", "cat", name],
                        capture_output=True, text=True, timeout=5,
                    )
                    if cat.returncode == 0:
                        (snap_dir / f"unit-{srv}.txt").write_text(cat.stdout)
                        for cl in cat.stdout.split("\n"):
                            cl = cl.strip()
                            if cl.startswith("ExecStart="):
                                parts = cl.split("=", 1)[1].split(None, 1)
                                if parts:
                                    entry["exec"] = parts[0]
                            elif cl.startswith("EnvironmentFile="):
                                ef = cl.split("=", 1)[1].strip("-")
                                entry.setdefault("configs", []).append(ef)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                for known_srv, configs in _KNOWN_CONFIG_PATTERNS.items():
                    if known_srv in srv or srv in known_srv:
                        entry["configs"] = list(set(entry.get("configs", []) + configs))
                        break
                sm.services[srv] = entry
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Source 2: ss -tlnp — ports + process names
    ports_raw = ""
    try:
        result = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            ports_raw = result.stdout
            (snap_dir / "ports.txt").write_text(ports_raw)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    port_processes = []
    for line in ports_raw.split("\n"):
        if "LISTEN" not in line:
            continue
        parts = line.split()
        port = ""
        for p in parts:
            if ":" in p:
                candidate = p.rsplit(":", 1)[-1]
                if candidate.isdigit():
                    port = candidate
                    break
        proc_name = ""
        pid = ""
        for p in parts:
            m = re.search(r'"([^"]+)"', p)
            if m:
                proc_name = m.group(1)
            pm = re.search(r'pid=(\d+)', p)
            if pm:
                pid = pm.group(1)
        if port and proc_name:
            port_processes.append((port, proc_name, pid))

    # Populate port_owners from collected ports
    for port, proc_name, _ in port_processes:
        sm.port_owners.setdefault(port, set()).add(proc_name)

    # Source 3: /proc — open files per PID
    pid_to_svc = {pid: pn for _, pn, pid in port_processes if pid}
    for pid, proc_name in pid_to_svc.items():
        proc_fd = Path(f"/proc/{pid}/fd")
        if not proc_fd.is_dir():
            continue
        try:
            exe_path = str(Path(f"/proc/{pid}/exe").resolve()) if Path(f"/proc/{pid}/exe").exists() else ""
        except (OSError, PermissionError):
            exe_path = ""
        try:
            for fd in proc_fd.iterdir():
                try:
                    link = os.readlink(str(fd))
                    if link.startswith("/etc/") or link.startswith("/opt/") or link.startswith("/usr/local/etc/"):
                        if proc_name not in sm.services:
                            sm.services[proc_name] = {"ports": [], "configs": [], "type": "daemon"}
                        if exe_path:
                            sm.services[proc_name]["exec"] = exe_path
                        if link not in sm.services[proc_name].setdefault("configs", []):
                            sm.services[proc_name]["configs"].append(link)
                except (OSError, PermissionError):
                    continue
        except (OSError, PermissionError):
            pass

    # Cross-reference: ports → services
    for port, proc_name, pid in port_processes:
        if proc_name not in sm.services:
            sm.services[proc_name] = {"ports": [], "configs": [], "type": "daemon"}
        if port not in sm.services[proc_name].setdefault("ports", []):
            sm.services[proc_name]["ports"].append(port)

    # Build file_owners
    for srv_name, srv_info in sm.services.items():
        for cfg in srv_info.get("configs", []):
            prefix = cfg if cfg.endswith("/") else str(Path(cfg).parent) + "/"
            sm.file_owners[cfg] = list(set(sm.file_owners.get(cfg, []) + [srv_name]))
        for known_srv, configs in _KNOWN_CONFIG_PATTERNS.items():
            if known_srv in srv_name or srv_name in known_srv:
                for cfg_prefix in configs:
                    existing = sm.file_owners.setdefault(cfg_prefix, [])
                    if srv_name not in existing:
                        existing.append(srv_name)

    # Cross-service: /etc/letsencrypt/ → certbot + nginx
    if "certbot" in sm.services and "nginx" in sm.services:
        for path in list(sm.file_owners.keys()):
            if "letsencrypt" in path or "ssl" in path or "cert" in path:
                existing = sm.file_owners.setdefault(path, [])
                for svc in ("certbot", "nginx"):
                    if svc not in existing:
                        existing.append(svc)

    # Load learned changes
    learned_file = snap_dir / "learned_deps.json"
    if learned_file.exists():
        try:
            learned = json.loads(learned_file.read_text())
            for path, services in learned.get("file_owners", {}).items():
                existing = sm.file_owners.setdefault(path, [])
                for s in services:
                    if s not in existing:
                        existing.append(s)
            for name, info in learned.get("services", {}).items():
                if name not in sm.services:
                    sm.services[name] = info
        except Exception:
            pass

    # Defaults
    if "/etc/hosts" not in sm.file_owners:
        sm.file_owners["/etc/hosts"] = ["networking"]
    if "/etc/ssh/sshd_config" not in sm.file_owners:
        sm.file_owners["/etc/ssh/sshd_config"] = ["ssh"]

    (snap_dir / "service_map.json").write_text(json.dumps({
        "services": sm.services,
        "file_owners": sm.file_owners,
        "timestamp": datetime.now().isoformat(),
    }, indent=2, default=str))

    _snapshot_taken = True
    _service_map = sm
    return sm


def _has_snapshot() -> bool:
    global _snapshot_taken
    if _snapshot_taken:
        return True
    snap_dir = _ensure_snapshot_dir()
    map_file = snap_dir / "service_map.json"
    if map_file.exists():
        try:
            data = json.loads(map_file.read_text())
            _service_map.services = data.get("services", {})
            _service_map.file_owners = data.get("file_owners", {})
            _snapshot_taken = True
            return True
        except Exception:
            return False
    return False


# ── Dependency Lookup ──────────────────────────────────────────────────────

def _lookup_dependencies(path: str) -> list[dict]:
    if not path:
        return []
    deps = []
    for cfg_path, services in _service_map.file_owners.items():
        if path.startswith(cfg_path) or cfg_path in path or path in cfg_path:
            for svc in services:
                svc_info = _service_map.services.get(svc, {})
                deps.append({
                    "service": svc,
                    "through": cfg_path,
                    "ports": svc_info.get("ports", []),
                    "type": svc_info.get("type", "unknown"),
                })
    seen = set()
    unique = []
    for d in deps:
        if d["service"] not in seen:
            seen.add(d["service"])
            unique.append(d)
    return unique


def _format_deps(deps: list[dict]) -> str:
    if not deps:
        return ""
    parts = []
    for d in deps:
        extras = []
        if d["ports"]:
            extras.append(f"port{'s' if len(d['ports']) != 1 else ''}: {', '.join(d['ports'])}")
        tag = f" ({'; '.join(extras)})" if extras else ""
        parts.append(f"  • {d['service']} [{d['type']}]{tag} — via {d['through']}")
    return "\n".join(parts)


# ── Learning ───────────────────────────────────────────────────────────────

def _learn_change(tool_name: str, args: dict) -> None:
    if tool_name not in WRITE_TOOLS or not isinstance(args, dict):
        return
    path, classification = _normalize_to_path(tool_name, args)
    if classification != "SYSTEM" or not path:
        return
    if path not in _service_map.file_owners:
        affected = _lookup_dependencies(path)
        svc_names = [d["service"] for d in affected] if affected else ["custom"]
        _service_map.file_owners[path] = svc_names
        logger.info("SBL learned new dependency: %s → %s", path, svc_names)
    entry = {"tool": tool_name, "path": path, "timestamp": datetime.now().isoformat()}
    _change_log.append(entry)
    snap_dir = _ensure_snapshot_dir()
    learned_file = snap_dir / "learned_deps.json"
    try:
        existing = json.loads(learned_file.read_text()) if learned_file.exists() else {}
    except Exception:
        existing = {}
    existing.setdefault("file_owners", {}).update({path: _service_map.file_owners[path]})
    existing["services"] = _service_map.services
    learned_file.write_text(json.dumps(existing, indent=2, default=str))


# ── Hooks ─────────────────────────────────────────────────────────────────

def _on_pre_tool_call(
    tool_name: str = "",
    args: Optional[dict] = None,
    **kwargs,
) -> Optional[str]:
    """Pre-write: classify path, snapshot, check dependencies.

    Returns None → pass through (USER, SYSTEM without deps, non-write tools).
    Returns str → info for SYSTEM with deps, or block for UNKNOWN.
    """
    if tool_name not in WRITE_TOOLS or not isinstance(args, dict):
        return None
    path, classification = _normalize_to_path(tool_name, args)

    if classification == "USER":
        return None
    if classification == "UNKNOWN":
        return (
            f"[SBL] Unclassified path: '{path}' — blocked.\n"
            f"  Use known paths under /etc/, /opt/, /usr/, or user paths under /home/, /tmp/"
        )

    # SYSTEM: ensure snapshot
    if not _has_snapshot():
        try:
            _take_snapshot()
        except Exception as e:
            logger.error("[SBL] snapshot failed: %s", e)
            return None

    deps = _lookup_dependencies(path)
    if deps:
        dep_str = _format_deps(deps)
        logger.info("[SBL] %s affects %d services:\n%s", path, len(deps), dep_str)
        return f"[SBL] Writing to {path} affects running services:\n{dep_str}"
    return None


def _on_transform_tool_result(
    tool_name: str = "",
    args: Optional[dict] = None,
    result: Any = None,
    **kwargs,
) -> Optional[str]:
    """Post-write: learn new paths. No force re-snapshot."""
    if tool_name not in WRITE_TOOLS or not isinstance(args, dict):
        return None
    if isinstance(result, dict) and result.get("error"):
        return None
    _learn_change(tool_name, args)
    return None


def _on_session_start(**kwargs) -> None:
    """Full audit on first session on a new system.
    
    Two-phase audit:
      1. Quick snapshot (systemctl + ss + /proc) — always, <1s
      2. Deep audit (fd + rg + cert cross-ref) — only on first run
    
    Deep audit runs synchronously on first session. The user sees
    a complete infrastructure map before any write happens.
    
    Returns None (hook), but persists to disk so future restarts
    load via _has_snapshot().
    """
    if _has_snapshot():
        logger.info("[SBL] Snapshot loaded: %d services, %d configs",
                    len(_service_map.services), len(_service_map.file_owners))
        return
    
    # Phase 1: Quick snapshot (always)
    logger.info("[SBL] First run on new system — starting full audit...")
    try:
        sm = _take_snapshot()
        logger.info("[SBL] Snapshot: %d services, %d configs",
                    len(sm.services), len(sm.file_owners))
    except Exception as e:
        logger.error("[SBL] Snapshot failed: %s", e)
    
    # Phase 2: Deep audit (fd + rg + cert) — only on first run
    try:
        from plugins.sbl.deep_audit import _audit, format_summary
        data = _audit()
        summary = format_summary(data)
        logger.info("[SBL] Deep audit complete — %d services, %d configs, %d cert users",
                    len(data['services']), data['configs_total'], len(data['cert_users']))
        # Сохраняем deep audit результаты в service_map
        for svc, info in data['services'].items():
            if svc not in _service_map.services:
                _service_map.services[svc] = {}
            if info.get('ports'):
                _service_map.services[svc]['ports'] = info['ports']
            if info.get('cross'):
                _service_map.services[svc]['cross'] = info['cross']
        # Персист
        learned = {'services': _service_map.services, 'file_owners': _service_map.file_owners,
                   'deep_audit': {'timestamp': datetime.now().isoformat(), 'services': list(data['services'].keys()),
                                  'cert_users': data['cert_users'], 'cert_domains': data['cert_domains']}}
        snap_dir = _ensure_snapshot_dir()
        (snap_dir / 'learned_deps.json').write_text(json.dumps(learned, indent=2, default=str))
        # Возвращаем сводку через logger — она попадёт в контекст агента
        logger.info("[SBL] === DEEP AUDIT SUMMARY ===\n%s", summary)
    except ImportError as e:
        logger.warning("[SBL] Deep audit unavailable (fd/rg not installed?): %s", e)
    except Exception as e:
        logger.warning("[SBL] Deep audit failed: %s", e)


# ── SBL Commands ───────────────────────────────────────────────────────────

def _handle_sbl_snapshot(cmd_args: str = "") -> str:
    global _snapshot_taken, _service_map
    parts = cmd_args.strip().split(None, 1)
    subcmd = parts[0] if parts else "status"

    if subcmd == "snapshot":
        sm = _take_snapshot()
        return (
            f"SBL Snapshot updated:\n"
            f"  • {len(sm.services)} services\n"
            f"  • {len(sm.file_owners)} config dependencies\n"
            f"  • {len(_change_log)} learned changes"
        )

    if subcmd == "deps":
        if not _has_snapshot():
            return "SBL: No snapshot. Run /sbl snapshot first"
        filter_path = parts[1] if len(parts) > 1 else ""
        if filter_path:
            deps = _lookup_dependencies(filter_path)
            if deps:
                return f"Dependencies for {filter_path}:\n{_format_deps(deps)}"
            return f"No known dependencies for {filter_path}"
        lines = ["SBL Full Dependency Map:"]
        for fpath, services in sorted(_service_map.file_owners.items()):
            lines.append(f"  {fpath} → {', '.join(services)}")
        return "\n".join(lines)

    if subcmd == "deep-audit":
        try:
            from plugins.sbl.deep_audit import _audit, format_summary
            data = _audit()
            return format_summary(data)
        except ImportError as e:
            return f"SBL Deep Audit unavailable: {e}. Install fd and rg: apt install fd-find ripgrep"
        except Exception as e:
            return f"SBL Deep Audit failed: {e}"

    if subcmd == "changes":
        if not _change_log:
            return "SBL: No changes recorded."
        lines = [f"SBL Change Log ({len(_change_log)} entries):"]
        for c in _change_log[-10:]:
            lines.append(f"  [{c['timestamp']}] {c['tool']} → {c['path']}")
        return "\n".join(lines)

    if subcmd == "reset":
        _snapshot_taken = False
        _service_map = ServiceMap()
        _change_log.clear()
        snap_dir = _ensure_snapshot_dir()
        for f in snap_dir.glob("*"):
            f.unlink()
        return "SBL reset."

    # status (default)
    if not _has_snapshot():
        return "SBL: No snapshot. First SYSTEM write will auto-snapshot."
    sm = _service_map
    # Проверяем, был ли deep audit
    snap_dir = _ensure_snapshot_dir()
    learned_file = snap_dir / "learned_deps.json"
    deep_audit_info = ""
    if learned_file.exists():
        try:
            learned = json.loads(learned_file.read_text())
            da = learned.get("deep_audit", {})
            if da.get("services"):
                deep_audit_info = (
                    f"\n  Deep audit: {len(da['services'])} active services"
                    f"\n  Cert users: {len(da.get('cert_users', []))}"
                )
                if da.get("cert_domains"):
                    deep_audit_info += f"\n  SSL domains: {len(da['cert_domains'])}"
        except Exception:
            pass
    lines = [f"SBL Status: {len(sm.services)} services (snapshot), {len(sm.file_owners)} configs"]
    if deep_audit_info:
        lines.append(deep_audit_info)
    lines.append(f"  Changes applied: {len(_change_log)}")
    if _change_log:
        lines.append(f"  Recent: {_change_log[-1]['path']}")
    return "\n".join(lines)


# ── Registration ───────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Register SBL hooks: pre_tool_call + transform_tool_result + on_session_start."""
    try:
        ctx.register_hook("pre_tool_call", _on_pre_tool_call)
        ctx.register_hook("transform_tool_result", _on_transform_tool_result)
        ctx.register_hook("on_session_start", _on_session_start)
        ctx.register_command(
            "sbl",
            handler=_handle_sbl_snapshot,
            description="System Boundary Layer — snapshot, status, deps, changes",
        )
        logger.info("[SBL] Registered: 3 hooks + /sbl command")
    except Exception as e:
        logger.critical("[SBL] Registration FAILED: %s", e)
        # Minimal fallback: /sbl command only
        try:
            ctx.register_command(
                "sbl",
                handler=_handle_sbl_snapshot,
                description="SBL (hooks FAILED — only /sbl available)",
            )
        except Exception:
            logger.critical("[SBL] Complete registration failure")
