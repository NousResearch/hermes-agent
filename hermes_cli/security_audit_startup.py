"""Startup security posture audit (warn-on-load, never blocks).

Surfaces dangerous host / deployment posture at process start. Checks are advisory and
independently fail-safe: they return human-readable strings, any internal error yields no finding,
and nothing here ever raises or blocks startup.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hermes.security_audit")

# The audit runs once per process even if both the CLI and gateway startup paths call it.
_AUDIT_RAN = False


def _is_root() -> bool:
    """True when the process runs as POSIX uid 0. Always False on Windows."""
    getuid = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    try:
        return getuid is not None and getuid() == 0
    except Exception:
        return False


def _running_as_root() -> Optional[str]:
    return None if not _is_root() else (
        "Running as ROOT. The agent's terminal/file tools execute with full root privileges — a single "
        "prompt-injection or exposed endpoint is a full host compromise. Run Hermes as an unprivileged user "
        "(or in a sandboxed terminal backend / container with a non-root user).")


def _iter_sshd_config_lines() -> list[str]:
    """Non-comment lines from sshd_config + its drop-in directory."""
    lines: list[str] = []
    paths: list[Path] = [Path("/etc/ssh/sshd_config")]
    try:
        paths.extend(sorted(Path("/etc/ssh/sshd_config.d").glob("*.conf")))
    except Exception:
        pass
    for p in paths:
        try:
            raw_lines = p.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except Exception:
            continue
        lines.extend(s for s in map(str.strip, raw_lines) if s and not s.startswith("#"))
    return lines


def _sshd_config_ports(lines: list[str]) -> set[int]:
    """Ports sshd is told to listen on (the daemon defaults to 22 without a Port directive)."""
    matches = (re.match(r"(?i)^Port\s+(\d{1,5})\b", line) for line in lines)
    return {int(m.group(1)) for m in matches if m} or {22}


def _listening_tcp_ports(
    paths: tuple[Path, ...] = (Path("/proc/net/tcp"), Path("/proc/net/tcp6")),
) -> Optional[set[int]]:
    """Local ports in LISTEN state across the given /proc/net tables, or None when none is readable
    (macOS/Windows have no /proc, so the caller falls back to keeping the warning).
    """
    ports: set[int] = set()
    readable = False
    for path in paths:
        try:
            rows = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue
        readable = True
        for row in rows[1:]:  # first line is the column header
            fields = row.split()
            if len(fields) < 4 or fields[3] != "0A":  # TCP_LISTEN
                continue
            try:
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
            except (IndexError, ValueError):
                continue
    return ports if readable else None


def _sshd_process_running(proc_root: Path = Path("/proc")) -> Optional[bool]:
    """True when a process named sshd exists, False when none does, None when /proc is unlistable."""
    try:
        pid_dirs = [p for p in proc_root.iterdir() if p.name.isdigit()]
    except OSError:
        return None
    if not pid_dirs:
        return None
    for pid_dir in pid_dirs:
        try:
            if (pid_dir / "comm").read_text(encoding="utf-8-sig", errors="replace").strip() == "sshd":
                return True
        except OSError:
            continue
    return False


def _sshd_serving(lines: list[str]) -> Optional[bool]:
    """True when an SSH daemon can accept connections on this host, False when it definitively
    cannot (no listener on any configured Port and no sshd process), None when that cannot be
    determined — no /proc to inspect, or tables that cannot be read. Never guesses "not serving";
    the macOS Remote-Login lane is a separate check (PR #78073).
    """
    listening = _listening_tcp_ports()
    if listening is None:
        return None
    if _sshd_config_ports(lines) & listening:
        return True
    # A daemon started with an explicit -p (or moved listener) still serves, so the process wins.
    return _sshd_process_running()


def _ssh_password_auth_enabled() -> Optional[str]:
    """Warn when sshd has password auth enabled — the classic brute-force surface, which pairs
    badly with a root-capable agent box. None when there is no sshd config to read, and None for a
    stock config (no directive) on a host where no daemon is serving — there is nothing to brute-force.
    """
    lines = _iter_sshd_config_lines()
    if not lines:
        return None
    # Last directive wins in sshd_config. Default (no directive) is "yes".
    directives = [m.group(1).lower() for m in map(re.compile(r"(?i)^PasswordAuthentication\s+(\w+)").match, lines) if m]
    if directives and directives[-1] == "no":
        return None
    if not directives and _sshd_serving(lines) is False:
        return None  # an explicit opt-in still warns: it bites the moment sshd is started
    qualifier = "" if directives else " (default — no explicit directive)"
    return (f"SSH password authentication is ENABLED{qualifier}. Password auth is brute-forceable and dangerous "
            "on an internet-facing box. Set 'PasswordAuthentication no' in sshd_config and use key-based auth.")


def _in_container() -> bool:
    """Best-effort container detection (Docker / Podman / generic OCI)."""
    if os.path.exists("/.dockerenv"):
        return True
    if os.environ.get("HERMES_DESKTOP_CHILD_PID"):
        return False  # desktop child, not a server container
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return any(tok in cgroup for tok in ("docker", "containerd", "kubepods", "libpod"))


def _path_is_mounted(path: Path) -> bool:
    """True if *path* sits on a persistent mount per /proc/mounts, i.e. the most specific
    mountpoint at or above it is NOT the ephemeral container root overlay/tmpfs.
    """
    try:
        target = path.resolve()
    except Exception:
        target = path
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return True  # can't tell — fail safe (no warning)
    # (mountpoint, fstype) entries at or above target; the longest wins, first one on ties.
    covering = [
        (Path(parts[1]), parts[2])
        for parts in (line.split() for line in mounts)
        if len(parts) >= 3 and (Path(parts[1]) == target or Path(parts[1]) in target.parents)
    ]
    if not covering:
        return True
    best_fstype = max(covering, key=lambda entry: len(str(entry[0])))[1]
    return best_fstype not in ("overlay", "tmpfs", "aufs")


def _container_no_volume_mount(hermes_home: Optional[Path]) -> Optional[str]:
    if not _in_container():
        return None
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    if _path_is_mounted(hermes_home):  # any error propagates to run_security_audit (= no finding)
        return None
    return (f"Running in a container but the data dir ({hermes_home}) is NOT on a persistent volume mount — "
            "sessions, memory, skills, and API keys are ephemeral and lost on container restart. Mount a host "
            "volume over the HERMES_HOME data directory.")


def _network_listener_without_auth(config: Optional[dict]) -> list[str]:
    """Warn about a network-accessible API server with no API_SERVER_KEY. Read-only against
    config + env; overlaps the hard fail-closed guards but surfaces the posture at startup.
    """
    # Any error (incl. an unimportable gateway package) propagates to run_security_audit.
    from gateway.platforms.base import is_network_accessible

    plats = (config or {}).get("platforms") or {}
    api = plats.get("api_server") if isinstance(plats, dict) else None
    if not (isinstance(api, dict) and api.get("enabled")):
        return []
    extra = api.get("extra") or {}
    host = extra.get("host") or os.environ.get("API_SERVER_HOST", "127.0.0.1")
    key = extra.get("key") or os.environ.get("API_SERVER_KEY", "")
    if not is_network_accessible(str(host)) or str(key).strip():
        return []
    return [f"OpenAI-compatible API server is network-accessible ({host}) with NO API_SERVER_KEY. It dispatches "
            "terminal-capable agent work — an unauthenticated network endpoint is remote code execution. "
            "Set a strong API_SERVER_KEY."]


def run_security_audit(*, hermes_home: Optional[Path] = None, config: Optional[dict] = None) -> list[str]:
    """Run all checks and return human-readable warning strings. Pure (no logging); a check that
    raises simply contributes no finding.
    """
    findings: list[str] = []
    for check in (_running_as_root, _ssh_password_auth_enabled, lambda: _container_no_volume_mount(hermes_home),
                  lambda: _network_listener_without_auth(config)):
        try:
            r = check()
        except Exception:
            continue
        if isinstance(r, list):
            findings.extend(r)
        elif r:
            findings.append(r)
    return findings


def log_startup_security_warnings(
    *, hermes_home: Optional[Path] = None, config: Optional[dict] = None, force: bool = False
) -> list[str]:
    """Run the audit once per process (``force=True`` re-runs) and log each finding as a warning."""
    global _AUDIT_RAN
    if _AUDIT_RAN and not force:
        return []
    _AUDIT_RAN = True
    findings = run_security_audit(hermes_home=hermes_home, config=config)
    if findings:
        logger.warning("Security posture audit found %d issue(s) — review your deployment:", len(findings))
        for i, f in enumerate(findings, 1):
            logger.warning("  [security %d/%d] %s", i, len(findings), f)
    return findings
