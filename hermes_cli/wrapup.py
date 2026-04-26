"""Safe wrap-up and resume helpers for Hermes CLI sessions.

The helper is intentionally conservative: it only acts on live CLI runtime
records that expose a confident ``session_id``. Ambiguous process-only matches
are skipped so the workflow never closes an unrelated terminal by accident.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from hermes_constants import get_hermes_home
from hermes_state import SessionDB

WRAP_UP_DIRNAME = "wrap-up"
RUNTIME_DIR = "runtime/cli_sessions"
ACTIVE_STATUSES = {"active", "running", "tool", "responding"}
CLOSED_STATUSES = {"closed_gracefully", "closed_after_timeout", "closed_hermes_only"}
SKIPPED_STATUSES = {"skipped_no_confident_session_id", "skipped_missing_session"}


@dataclass
class RuntimeRecord:
    pid: int
    session_id: str | None
    status: str
    cwd: str | None = None
    tty: str | None = None
    cmdline: list[str] | None = None
    path: Path | None = None
    ppid: int | None = None
    pgid: int | None = None
    sid: int | None = None
    updated_at: float | None = None

    def to_manifest(self) -> dict[str, Any]:
        data = asdict(self)
        if self.path is not None:
            data["path"] = str(self.path)
        return data


def _runtime_dir(hermes_home: Path | None = None) -> Path:
    return (hermes_home or get_hermes_home()) / RUNTIME_DIR


def _now() -> float:
    return time.time()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_runtime_record(
    *,
    session_id: str,
    status: str,
    hermes_home: Path | None = None,
    cwd: Path | str | None = None,
    pid: int | None = None,
    tty: str | None = None,
) -> RuntimeRecord:
    """Write/update the runtime heartbeat for the current CLI session."""
    pid = pid or os.getpid()
    cwd_text = str(cwd if cwd is not None else Path.cwd())
    tty_text = tty
    if tty_text is None:
        try:
            tty_text = os.ttyname(sys.stdin.fileno())
        except Exception:
            tty_text = None

    record = RuntimeRecord(
        pid=pid,
        ppid=os.getppid() if pid == os.getpid() else None,
        pgid=_safe_getpgid(pid),
        sid=_safe_getsid(pid),
        session_id=session_id,
        status=status,
        cwd=cwd_text,
        tty=tty_text,
        cmdline=_read_cmdline(pid),
        updated_at=_now(),
    )
    path = _runtime_dir(hermes_home) / f"{pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record.path = path
    _atomic_write_json(path, record.to_manifest())
    return record


def remove_runtime_record(pid: int | None = None, hermes_home: Path | None = None) -> None:
    path = _runtime_dir(hermes_home) / f"{pid or os.getpid()}.json"
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def load_runtime_records(
    hermes_home: Path | None = None,
    *,
    pid_alive: Callable[[int], bool] | None = None,
    stale_after_seconds: int = 86400,
) -> list[RuntimeRecord]:
    """Load non-stale runtime records whose PIDs are still alive."""
    pid_alive = pid_alive or is_pid_alive
    records: list[RuntimeRecord] = []
    now = _now()
    root = _runtime_dir(hermes_home)
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            pid = int(data["pid"])
        except Exception:
            continue
        if not pid_alive(pid):
            _safe_unlink(path)
            continue
        updated_at = data.get("updated_at")
        if isinstance(updated_at, (int, float)) and now - float(updated_at) > stale_after_seconds:
            _safe_unlink(path)
            continue
        records.append(
            RuntimeRecord(
                pid=pid,
                ppid=data.get("ppid"),
                pgid=data.get("pgid"),
                sid=data.get("sid"),
                session_id=data.get("session_id") or None,
                status=data.get("status") or "unknown",
                cwd=data.get("cwd"),
                tty=data.get("tty"),
                cmdline=data.get("cmdline") or None,
                path=path,
                updated_at=updated_at,
            )
        )
    return records


def wrap_up(
    *,
    hermes_home: Path | None = None,
    timeout_seconds: int = 600,
    close_windows: bool = True,
) -> dict[str, Any]:
    """Discover live CLI runtime records and wrap them up."""
    hermes_home = hermes_home or get_hermes_home()
    session_db = SessionDB(db_path=hermes_home / "state.db")
    try:
        return wrap_up_records(
            sorted(load_runtime_records(hermes_home), key=lambda record: record.pid in _ancestor_pids()),
            session_db=session_db,
            hermes_home=hermes_home,
            timeout_seconds=timeout_seconds,
            close_session=lambda record: close_cli_session(record, close_window=close_windows),
            wait_for_idle=lambda record, timeout_seconds: wait_for_idle_record(
                record, hermes_home=hermes_home, timeout_seconds=timeout_seconds
            ),
            controller_pids=_ancestor_pids(),
        )
    finally:
        session_db.close()


def wrap_up_records(
    records: Iterable[RuntimeRecord],
    *,
    session_db: SessionDB,
    hermes_home: Path,
    run_id: str | None = None,
    run_dir: Path | None = None,
    timeout_seconds: int = 600,
    close_session: Callable[[RuntimeRecord], str],
    wait_for_idle: Callable[[RuntimeRecord, int], RuntimeRecord],
    controller_pids: set[int] | None = None,
) -> dict[str, Any]:
    """Export, drain, close, and manifest a set of runtime records."""
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_dir or hermes_home / WRAP_UP_DIRNAME / "runs" / run_id
    exports_dir = run_dir / "exports"
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hermes_home": str(hermes_home),
        "timeout_seconds": timeout_seconds,
        "sessions": [],
    }

    controller_pids = controller_pids or set()

    for original in records:
        record = original
        is_controller = record.pid in controller_pids
        entry = record.to_manifest()
        if is_controller:
            entry["controller_session"] = True
        session_id = record.session_id
        if not session_id:
            entry.update({"close_status": "skipped_no_confident_session_id"})
            manifest["sessions"].append(entry)
            continue

        resolved_id = session_db.resolve_resume_session_id(session_id)
        session = session_db.get_session(resolved_id)
        if not session or session.get("source") != "cli":
            entry.update({"session_id": resolved_id, "close_status": "skipped_missing_session"})
            manifest["sessions"].append(entry)
            continue

        if (record.status or "").lower() in ACTIVE_STATUSES and not is_controller:
            record = wait_for_idle(record, timeout_seconds)
            entry.update(record.to_manifest())

        exports_dir.mkdir(parents=True, exist_ok=True)
        export_path = exports_dir / f"{resolved_id}.json"
        exported = session_db.export_session(resolved_id)
        if exported is not None:
            _atomic_write_json(export_path, exported)
            entry["export_path"] = str(export_path)

        close_status = close_session(record)
        if (record.status or "").lower() in ACTIVE_STATUSES and close_status == "closed_gracefully":
            close_status = "closed_after_timeout"
        entry.update(
            {
                "session_id": resolved_id,
                "source": session.get("source"),
                "title": session.get("title"),
                "message_count": session.get("message_count"),
                "close_status": close_status,
                "resume_command": f"hermes --resume {resolved_id}",
            }
        )
        manifest["sessions"].append(entry)

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    latest_path = hermes_home / WRAP_UP_DIRNAME / "latest.json"
    _atomic_write_json(manifest_path, manifest)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(latest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["latest_path"] = str(latest_path)
    return manifest


def wait_for_idle_record(record: RuntimeRecord, *, hermes_home: Path, timeout_seconds: int) -> RuntimeRecord:
    deadline = _now() + max(0, timeout_seconds)
    current = record
    while _now() < deadline:
        latest = _find_record_by_pid(record.pid, hermes_home)
        if latest is None:
            return current
        current = latest
        if (current.status or "").lower() not in ACTIVE_STATUSES:
            return current
        time.sleep(2)
    return current


def close_cli_session(record: RuntimeRecord, *, close_window: bool = True) -> str:
    """Close a Hermes CLI process, attempting graceful termination first.

    Closing the actual Windows Terminal tab is best-effort and intentionally not
    broad: if we cannot prove the window identity, we only close Hermes itself.
    """
    if not is_pid_alive(record.pid):
        return "closed_gracefully"

    try:
        os.kill(record.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "closed_gracefully"
    except Exception as exc:
        return f"failed:{exc}"

    deadline = _now() + 5
    while _now() < deadline:
        if not is_pid_alive(record.pid):
            return "closed_gracefully"
        time.sleep(0.2)

    try:
        os.kill(record.pid, signal.SIGKILL)
    except ProcessLookupError:
        return "closed_gracefully"
    except Exception as exc:
        return f"failed:{exc}"
    return "closed_hermes_only"


def resume_commands_from_manifest(manifest: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for entry in manifest.get("sessions", []):
        if entry.get("close_status") in CLOSED_STATUSES and entry.get("session_id"):
            session_id = str(entry["session_id"])
            commands.append(f"hermes --resume {shlex.quote(session_id)}")
    return commands


def continue_sessions(
    *,
    hermes_home: Path | None = None,
    launcher: Callable[[str], bool] | None = None,
    max_auto_open: int = 5,
) -> dict[str, Any]:
    hermes_home = hermes_home or get_hermes_home()
    latest_path = hermes_home / WRAP_UP_DIRNAME / "latest.json"
    manifest = json.loads(latest_path.read_text())
    commands = resume_commands_from_manifest(manifest)
    launcher = launcher or launch_resume_command
    auto_commands = commands[:max_auto_open]
    manual_commands = commands[max_auto_open:]
    launched: list[str] = []
    failed: list[str] = []
    for command in auto_commands:
        if launcher(command):
            launched.append(command)
        else:
            failed.append(command)
            manual_commands.insert(0, command)
    return {"launched": launched, "manual": manual_commands, "failed": failed, "manifest": str(latest_path)}


def launch_resume_command(command: str) -> bool:
    """Open a resume command in a new Windows Terminal tab/window from WSL.

    Falls back to False if Windows Terminal is unavailable; the caller prints
    the command for manual execution.
    """
    if not _is_wsl():
        return False
    try:
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", "wt.exe", "wsl.exe", "-e", "bash", "-lc", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _find_record_by_pid(pid: int, hermes_home: Path) -> RuntimeRecord | None:
    for record in load_runtime_records(hermes_home):
        if record.pid == pid:
            return record
    return None


def _ancestor_pids(pid: int | None = None) -> set[int]:
    """Return Linux ancestor PIDs for the current helper process."""
    pid = pid or os.getpid()
    ancestors: set[int] = set()
    current = pid
    for _ in range(64):
        stat_path = Path("/proc") / str(current) / "stat"
        try:
            text = stat_path.read_text()
            ppid = int(text.rsplit(")", 1)[1].split()[1])
        except Exception:
            break
        if ppid <= 1 or ppid in ancestors:
            break
        ancestors.add(ppid)
        current = ppid
    return ancestors


def _read_cmdline(pid: int) -> list[str] | None:
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    parts = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    return parts or None


def _safe_getpgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except Exception:
        return None


def _safe_getsid(pid: int) -> int | None:
    try:
        return os.getsid(pid)
    except Exception:
        return None


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")
    tmp.replace(path)


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except Exception:
        return False


def _print_wrap_summary(manifest: dict[str, Any]) -> None:
    print(f"wrap-up run: {manifest['run_id']}")
    print(f"manifest: {manifest.get('manifest_path') or manifest.get('latest_path')}")
    for entry in manifest.get("sessions", []):
        sid = entry.get("session_id") or "<unknown>"
        print(f"- {sid}: {entry.get('close_status')}")


def _print_continue_summary(result: dict[str, Any]) -> None:
    print(f"manifest: {result['manifest']}")
    for command in result["launched"]:
        print(f"opened: {command}")
    if result["manual"]:
        print("manual resume commands:")
        for command in result["manual"]:
            print(f"  {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hermes_cli.wrapup")
    sub = parser.add_subparsers(dest="command", required=True)
    wrap_parser = sub.add_parser("wrap-up")
    wrap_parser.add_argument("--timeout", type=int, default=600)
    wrap_parser.add_argument("--no-close-windows", action="store_true")
    continue_parser = sub.add_parser("continue-sessions")
    continue_parser.add_argument("--max-auto-open", type=int, default=5)
    args = parser.parse_args(argv)

    if args.command == "wrap-up":
        manifest = wrap_up(timeout_seconds=args.timeout, close_windows=not args.no_close_windows)
        _print_wrap_summary(manifest)
        return 0
    if args.command == "continue-sessions":
        result = continue_sessions(max_auto_open=args.max_auto_open)
        _print_continue_summary(result)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
