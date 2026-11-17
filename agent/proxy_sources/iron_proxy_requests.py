"""Per-session outbound request log read from iron-proxy's per-request records.

iron-proxy v0.50 writes one JSON line per request to its stderr (``msg: "request"``, an ``audit``
group with host/method/path/remote_addr/action/status_code/duration_ms), which Hermes appends to
``<HERMES_HOME>/proxy/iron-proxy.log``. This module turns those lines into
``{method, host, path, status, ms}`` rows and nothing else: headers never appear in the record, and
the transform annotations, tunnel traces and optional body capture are never read. The path is
iron-proxy's ``r.URL.Path`` (no query string); any ``?`` that does appear is cut and the rest runs
through forced secret redaction, because a token can live in a path segment.

Attribution. Proxy tokens are minted per provider and shared by every sandbox, so the token cannot
name a session. The finest real key in the record is ``remote_addr``: the sandbox container's bridge
IP. The Docker backend records ``{ip, task_id, container}`` in ``sandboxes.jsonl`` when it starts an
egress-routed container; a request belongs to the newest registration of its IP made before it.
With ``container_persistent: false`` the task id is the session id, so the log is per session; with
a persistent (profile-shared) container every session of the profile shares one IP and the rows
are that container's.

Pure observation: nothing here allows, denies or alters traffic.
"""

from __future__ import annotations

import json
import os
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent.proxy_sources.iron_proxy import _O_NOFOLLOW, _proxy_state_dir, _proxy_state_dir_ro
from agent.redact import redact_sensitive_text

_LOG_TAIL_BYTES = 4 * 1024 * 1024
_REGISTRY_NAME = "sandboxes.jsonl"
_DEFAULT_LIMIT = 200


@dataclass(frozen=True)
class EgressRequest:
    """One outbound request as the dashboard and CLI show it; the field set is the privacy contract."""
    ts: float
    method: str
    host: str
    path: str
    status: int
    ms: float
    action: str


@dataclass(frozen=True)
class _Record:
    request: EgressRequest
    remote_ip: str


def _safe_path(raw: object) -> str:
    path = str(raw or "").split("?", 1)[0].split("#", 1)[0]
    return redact_sensitive_text(path, force=True) if path else ""


def _remote_ip(addr: str) -> str:
    host = addr.rsplit(":", 1)[0] if addr.count(":") == 1 or addr.startswith("[") else addr
    return host.strip("[]")


def parse_request_line(line: str) -> Optional[_Record]:
    """A per-request record from one log line, or None for daemon diagnostics and noise.

    Allowed CONNECTs are dropped: the MITM'd request inside the tunnel logs its own record with the
    real method and path. A rejected CONNECT (host off the allowlist) is the only trace, so it stays.
    """
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    audit = entry.get("audit") if isinstance(entry, dict) and entry.get("msg") == "request" else None
    if not isinstance(audit, dict):
        return None
    method, action = str(audit.get("method") or ""), str(audit.get("action") or "")
    if method == "CONNECT" and action == "allow":
        return None
    try:
        ts = datetime.fromisoformat(str(entry.get("time"))).timestamp()
    except ValueError:
        return None
    request = EgressRequest(
        ts=ts, method=method, host=str(audit.get("host") or ""), path=_safe_path(audit.get("path")),
        status=int(audit.get("status_code") or 0), ms=float(audit.get("duration_ms") or 0.0), action=action)
    return _Record(request, _remote_ip(str(audit.get("remote_addr") or "")))


def _registry_path() -> Path:
    return _proxy_state_dir_ro() / _REGISTRY_NAME


def record_sandbox(ip: str, task_id: str, container_id: str) -> None:
    """Remember which task an egress-routed sandbox IP belongs to from now on (0o600, append-only)."""
    line = json.dumps({"ts": time.time(), "ip": ip, "task_id": task_id, "container": container_id[:12]})
    fd = os.open(str(_proxy_state_dir() / _REGISTRY_NAME), os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW, 0o600)
    try:
        os.write(fd, (line + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _read_lines(path: Path, tail_bytes: int) -> list[str]:
    try:
        with path.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - tail_bytes))
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[1:] if len(data) == tail_bytes and lines else lines


def _registrations() -> dict[str, tuple[list[float], list[str]]]:
    """ip -> (ascending registration times, task ids) for an O(log n) lookup per record."""
    out: dict[str, tuple[list[float], list[str]]] = {}
    for line in _read_lines(_registry_path(), _LOG_TAIL_BYTES):
        try:
            row = json.loads(line)
            ip, ts, task = str(row["ip"]), float(row["ts"]), str(row["task_id"])
        except (ValueError, KeyError, TypeError):
            continue
        times, tasks = out.setdefault(ip, ([], []))
        times.append(ts)
        tasks.append(task)
    return out


def _owner(regs: dict[str, tuple[list[float], list[str]]], record: _Record) -> Optional[str]:
    times, tasks = regs.get(record.remote_ip, ([], []))
    i = bisect_right(times, record.request.ts)
    return tasks[i - 1] if i else None


def _task_matches(task_id: Optional[str], session_id: str) -> bool:
    # Routed profiles qualify the container key as ``<scope>:<session_id>``.
    return task_id is not None and (task_id == session_id or task_id.endswith(":" + session_id))


def egress_enabled() -> bool:
    from hermes_cli.config import load_config

    return bool((load_config().get("proxy") or {}).get("enabled"))


def read_requests(*, session_id: Optional[str] = None, limit: int = _DEFAULT_LIMIT) -> dict:
    """``{"enabled", "requests"}`` newest first. Egress off -> ``enabled: False`` and no rows, whatever
    an old log still holds. ``session_id`` keeps only rows attributed to that session's sandbox."""
    if not egress_enabled():
        return {"enabled": False, "requests": []}
    records = [r for line in _read_lines(_proxy_state_dir_ro() / "iron-proxy.log", _LOG_TAIL_BYTES)
               if (r := parse_request_line(line)) is not None]
    if session_id:
        regs = _registrations()
        records = [r for r in records if _task_matches(_owner(regs, r), session_id)]
    rows = [asdict(r.request) for r in reversed(records[-max(1, limit):])]
    return {"enabled": True, "requests": rows}


__all__ = ["EgressRequest", "egress_enabled", "parse_request_line", "read_requests", "record_sandbox"]
