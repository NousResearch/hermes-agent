"""Read-only pre-spawn mount/writeability evidence for kanban worker launches.

Minimal passive instrumentation at the dispatcher/worker boundary to
distinguish a transient ext4 RO remount (errors=remount-ro) from
path-local visibility / namespace effects.

Captures, without privilege escalation and without mutating state:
  - /proc/mounts (full, and filtered line for /)
  - /proc/self/mountinfo (full, and filtered line for mountpoint /)
  - mount(8) line for /
  - filesystem mount options including errors=remount-ro
  - os.access + stat probes for profile root / state.db / logs / agent.log / auth.lock
  - open-append writability probe (bounded write-probe: opens existing file with "ab"
    without writing then closes; does bump atime and requires W, not pure read-only,
    but does not mutate content)
  - bounded kernel/journal tail if already readable (dmesg, journalctl -k, /var/log/kern.log)
  - disk usage (df -h /) and inode usage if available

All helpers are best-effort and never raise — unknown/unreadable fields
become {"error": "..."} so evidence is still durable.

Two entry points:
  - capture_prespawn_evidence(...) -> dict  (pure, returns evidence)
  - write_prespawn_evidence_file(log_dir, task_id, evidence) -> Path|None
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _read_text_file(path: str, limit_bytes: int = 65536) -> Dict[str, Any]:
    try:
        p = Path(path)
        if not p.is_file():
            return {"path": path, "exists": False}
        data = p.read_bytes()[:limit_bytes].decode("utf-8", errors="replace")
        return {"path": path, "exists": True, "content": data.strip(), "truncated": p.stat().st_size > limit_bytes}
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "exists": None, "error": f"{type(exc).__name__}: {exc}"}


def _filter_mount_line(mounts_text: str, mountpoint: str = "/") -> Optional[str]:
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == mountpoint:
            return line
    return None


def _filter_mountinfo_line(mountinfo_text: str, mountpoint: str = "/") -> Optional[str]:
    # mountinfo format: ... - fstype source options
    # mountpoint is field 5 (0-indexed 4)
    for line in mountinfo_text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[4] == mountpoint:
            return line
    return None


def _run_cmd(cmd: list[str], timeout: float = 2.0, max_bytes: int = 8192) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        out = (proc.stdout or "")[:max_bytes]
        err = (proc.stderr or "")[:1024]
        return {"cmd": " ".join(cmd), "returncode": proc.returncode, "stdout": out.strip(), "stderr": err.strip()}
    except FileNotFoundError as exc:
        return {"cmd": " ".join(cmd), "error": f"not found: {exc}"}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": " ".join(cmd), "error": f"{type(exc).__name__}: {exc}"}


def _probe_path(path: Path) -> Dict[str, Any]:
    d: Dict[str, Any] = {"path": str(path)}
    try:
        d["exists"] = path.exists()
    except Exception as exc:  # noqa: BLE001
        d["exists_error"] = f"{type(exc).__name__}: {exc}"
    try:
        d["is_dir"] = path.is_dir()
        d["is_file"] = path.is_file()
    except Exception:
        pass
    try:
        st = path.stat()
        d["stat"] = {"mode_octal": oct(st.st_mode), "uid": st.st_uid, "gid": st.st_gid, "size": st.st_size}
    except FileNotFoundError:
        d["stat"] = None
    except Exception as exc:  # noqa: BLE001
        d["stat_error"] = f"{type(exc).__name__}: {exc}"
    # lsattr / getfacl best-effort (may not exist)
    # Don't fail evidence if not installed
    try:
        d["access_R"] = os.access(str(path), os.R_OK)
        d["access_W"] = os.access(str(path), os.W_OK)
        d["access_X"] = os.access(str(path), os.X_OK)
        d["access_RW"] = os.access(str(path), os.R_OK | os.W_OK)
    except Exception as exc:  # noqa: BLE001
        d["access_error"] = f"{type(exc).__name__}: {exc}"
    # open-append writability probe for files (bounded write-probe, not pure
    # read-only: opens with "ab" without writing then closes; requires W,
    # bumps atime, but does not mutate content)
    if d.get("exists") and path.is_file():
        try:
            with open(path, "ab"):
                pass
            d["open_append_probe"] = "PASS"
        except OSError as exc:
            d["open_append_probe"] = f"FAIL errno={exc.errno} {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            d["open_append_probe"] = f"FAIL {type(exc).__name__}: {exc}"
    elif path.parent.exists():
        # For non-existent file, probe parent dir writeability via temp file creation without leaving artifact:
        # Use os.access on parent; don't actually create file (keeps read-only intent)
        try:
            d["parent_access_W"] = os.access(str(path.parent), os.W_OK)
        except Exception:
            pass
    return d


def capture_prespawn_evidence(
    task_id: str,
    profile: str,
    workspace: str,
    board: Optional[str] = None,
    hermes_home: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture read-only evidence snapshot for a kanban worker launch.

    Note: the open-append probe inside _probe_path is a bounded
    writability probe (open "ab" without writing), not pure read-only;
    see module docstring for writability-probe semantics.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    evidence: Dict[str, Any] = {
        "schema_version": 1,
        "captured_at": ts,
        "capture_side": "dispatcher_prespawn",
        "task_id": task_id,
        "profile": profile,
        "workspace": workspace,
        "board": board,
        "pid": os.getpid(),
        "hostname": _run_cmd(["hostname"], timeout=1.0).get("stdout") or (os.uname().nodename if hasattr(os, "uname") else None),
    }

    # 1. mounts
    mounts = _read_text_file("/proc/mounts", limit_bytes=16384)
    evidence["proc_mounts"] = mounts
    if mounts.get("content"):
        evidence["proc_mounts_root_line"] = _filter_mount_line(mounts["content"], "/")
    mounts2 = _read_text_file("/proc/self/mountinfo", limit_bytes=16384)
    evidence["proc_self_mountinfo"] = mounts2
    if mounts2.get("content"):
        evidence["proc_self_mountinfo_root_line"] = _filter_mountinfo_line(mounts2["content"], "/")
    evidence["mount_cmd"] = _run_cmd(["mount"], timeout=2.0)
    # explicit grep for root line
    if evidence["mount_cmd"].get("stdout"):
        for ln in evidence["mount_cmd"]["stdout"].splitlines():
            if " on / " in ln:
                evidence["mount_cmd_root_line"] = ln
                break

    # 2. filesystem usage (df)
    evidence["df_h_root"] = _run_cmd(["df", "-h", "/"], timeout=2.0)
    evidence["df_i_root"] = _run_cmd(["df", "-i", "/"], timeout=2.0)

    # 3. profile path probes
    # Resolve hermes_home for profile
    resolved_home: Optional[str] = hermes_home
    if not resolved_home:
        try:
            from hermes_cli.profiles import resolve_profile_env  # type: ignore

            resolved_home = resolve_profile_env(profile)
        except Exception:
            try:
                from hermes_constants import get_hermes_home  # type: ignore

                resolved_home = str(get_hermes_home())
            except Exception:
                resolved_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    evidence["resolved_hermes_home"] = resolved_home
    if resolved_home:
        hp = Path(resolved_home)
        evidence["profile_home_probe"] = _probe_path(hp)
        evidence["state_db_probe"] = _probe_path(hp / "state.db")
        evidence["state_db_wal_probe"] = _probe_path(hp / "state.db-wal")
        evidence["state_db_shm_probe"] = _probe_path(hp / "state.db-shm")
        evidence["logs_dir_probe"] = _probe_path(hp / "logs")
        evidence["agent_log_probe"] = _probe_path(hp / "logs" / "agent.log")
        evidence["errors_log_probe"] = _probe_path(hp / "logs" / "errors.log")
        evidence["auth_lock_probe"] = _probe_path(hp / "auth.lock")
        evidence["sessions_dir_probe"] = _probe_path(hp / "sessions")
        evidence["board_probe"] = _probe_path(hp / "kanban" / "boards" / (board or "default")) if board and board != "default" else _probe_path(hp / "kanban" / "workspaces" / task_id) if task_id else None
        # also probe the resolved workspace path directly
        if workspace:
            evidence["workspace_probe"] = _probe_path(Path(workspace))
        # kanban db probe
        try:
            from hermes_cli.kanban_db import kanban_db_path  # type: ignore

            db_path = kanban_db_path(board=board)
            evidence["kanban_db_probe"] = _probe_path(Path(db_path))
            evidence["kanban_db_path"] = str(db_path)
        except Exception as exc:  # noqa: BLE001
            evidence["kanban_db_probe_error"] = f"{type(exc).__name__}: {exc}"

    # 4. bounded kernel/journal evidence (no privilege escalation, read-only)
    # Only run if readable; never escalate.
    evidence["dmesg_tail"] = _run_cmd(["dmesg", "--ctime", "--level", "err,warn"], timeout=2.0)
    # Truncate dmesg to last 50 lines to bound size
    if evidence["dmesg_tail"].get("stdout"):
        lines = evidence["dmesg_tail"]["stdout"].splitlines()
        if len(lines) > 50:
            evidence["dmesg_tail"]["stdout"] = "\n".join(lines[-50:])
            evidence["dmesg_tail"]["truncated_to_last_50"] = True

    evidence["journalctl_k_tail"] = _run_cmd(["journalctl", "-k", "--no-pager", "-n", "50", "--priority", "0..4"], timeout=2.0)
    if evidence["journalctl_k_tail"].get("stdout"):
        # journalctl may require privilege; capture error instead
        pass

    # /var/log/kern.log tail if readable
    kern_log = _read_text_file("/var/log/kern.log", limit_bytes=8192)
    if kern_log.get("content"):
        lines = kern_log["content"].splitlines()
        if len(lines) > 30:
            kern_log["content"] = "\n".join(lines[-30:])
            kern_log["truncated_to_last_30"] = True
    evidence["var_log_kern_tail"] = kern_log

    # /proc/meminfo pressure hint (no write)
    evidence["proc_meminfo_head"] = _read_text_file("/proc/meminfo", limit_bytes=4096)

    # errors=remount-ro detection summary
    mount_opts = ""
    if evidence.get("proc_mounts_root_line"):
        mount_opts = evidence["proc_mounts_root_line"] or ""
    elif mounts.get("content"):
        mount_opts = mounts["content"] or ""
    evidence["errors_remount_ro_present"] = "errors=remount-ro" in mount_opts if mount_opts else False
    evidence["is_root_rw"] = " rw," in mount_opts or " (rw" in mount_opts if mount_opts else None
    # Explicit rw vs ro parse from root line
    root_line = evidence.get("proc_mounts_root_line") or ""
    evidence["root_mount_rw"] = " rw," in root_line if root_line else None
    evidence["root_mount_ro"] = " ro," in root_line if root_line else None

    return evidence


def write_prespawn_evidence_file(log_dir: Path, task_id: str, evidence: Dict[str, Any], suffix: str = "prespawn") -> Optional[Path]:
    """Write evidence JSON to log_dir/<task_id>.<suffix>.json, bounded and never raises."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / f"{task_id}.{suffix}.json"
        # Also write a human-readable .log alongside
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=False), encoding="utf-8")
        tmp.rename(out)
        return out
    except Exception:
        return None
