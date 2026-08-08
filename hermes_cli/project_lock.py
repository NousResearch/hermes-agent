"""Windows project-directory lock diagnostics and fail-closed release checks."""

from __future__ import annotations

import ctypes
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


_HANDLE_ENUM_UNAVAILABLE = "HANDLE_ENUMERATION_UNAVAILABLE_REQUIRES_ADMIN"


def _same_or_child(candidate: str | None, root: Path) -> bool:
    if not candidate:
        return False
    try:
        return os.path.commonpath((os.path.normcase(os.path.abspath(candidate)), os.path.normcase(str(root)))) == os.path.normcase(str(root))
    except (OSError, ValueError):
        return False


def _rename_probe_windows(path: Path) -> dict[str, Any]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    create_file.restype = ctypes.c_void_p
    handle = create_file(str(path), 0x00010000, 0x1 | 0x2 | 0x4, None, 3, 0x02000000, None)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        winerror = ctypes.get_last_error()
        return {"ok": False, "winerror": winerror, "message": ctypes.FormatError(winerror).strip()}
    kernel32.CloseHandle(ctypes.c_void_p(handle))
    return {"ok": True, "winerror": None, "message": "DELETE_SHARE_PROBE_PASS"}


def rename_probe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "winerror": None, "message": "PATH_NOT_FOUND"}
    if os.name == "nt":
        return _rename_probe_windows(path)
    return {"ok": True, "winerror": None, "message": "WINDOWS_ONLY_PROBE_NOT_REQUIRED"}


def _rename_probe_without_current_cwd(path: Path) -> dict[str, Any]:
    """Probe without letting this short-lived diagnostic own the target CWD."""

    original_cwd = Path.cwd().resolve(strict=False)
    if not _same_or_child(str(original_cwd), path):
        return rename_probe(path)

    from hermes_constants import get_hermes_home

    candidates: list[Path] = []
    try:
        candidates.append(Path(get_hermes_home()).expanduser().resolve(strict=False))
    except (OSError, RuntimeError):
        pass
    candidates.extend((Path(__file__).resolve().parent, Path.home().resolve(strict=False)))
    stable_cwd = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_dir() and not _same_or_child(str(candidate), path)
        ),
        None,
    )
    if stable_cwd is None:
        return {
            "ok": False,
            "winerror": None,
            "message": "SAFE_DIAGNOSTIC_CWD_UNAVAILABLE",
        }

    probe: dict[str, Any] | None = None
    try:
        os.chdir(stable_cwd)
        probe = rename_probe(path)
    except OSError as exc:
        probe = {
            "ok": False,
            "winerror": getattr(exc, "winerror", None),
            "message": f"SAFE_DIAGNOSTIC_CWD_FAILED: {exc}",
        }
    finally:
        try:
            os.chdir(original_cwd)
        except OSError as exc:
            probe = {
                "ok": False,
                "winerror": getattr(exc, "winerror", None),
                "message": f"DIAGNOSTIC_CWD_RESTORE_FAILED: {exc}",
            }
    assert probe is not None
    return probe


def _process_role(row: dict[str, Any], hermes_pids: set[int]) -> str:
    if row["pid"] not in hermes_pids:
        return "child"
    command = [str(part) for part in row.get("cmdline") or []]
    executable = str(row.get("exe") or "")
    if "\\venv\\scripts\\hermes.exe" in executable.casefold():
        return "cli-launcher"
    process_type = next((part.split("=", 1)[1] for part in command if part.startswith("--type=")), None)
    if process_type:
        return f"desktop-{process_type}"
    return "desktop-main"


def _process_snapshot(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    try:
        import psutil
    except ImportError:
        return [], [], "PROCESS_ENUMERATION_UNAVAILABLE_PSUTIL_MISSING"

    all_rows: list[dict[str, Any]] = []
    unavailable = False
    for proc in psutil.process_iter(["pid", "ppid", "name", "exe", "cmdline", "cwd"]):
        try:
            info = proc.info
            all_rows.append({
                "pid": info.get("pid"), "ppid": info.get("ppid"),
                "name": info.get("name") or "", "exe": info.get("exe"),
                "cmdline": info.get("cmdline") or [], "cwd": info.get("cwd"),
            })
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            unavailable = True

    hermes_pids = {row["pid"] for row in all_rows if str(row["name"]).casefold() == "hermes.exe"}
    descendants = set(hermes_pids)
    changed = True
    while changed:
        changed = False
        for row in all_rows:
            if row["ppid"] in descendants and row["pid"] not in descendants:
                descendants.add(row["pid"])
                changed = True
    hermes = [dict(row, role=_process_role(row, hermes_pids)) for row in all_rows if row["pid"] in descendants]
    holders = [row for row in all_rows if _same_or_child(row.get("cwd"), path)]
    if os.name == "nt":
        try:
            unavailable = unavailable or not bool(ctypes.windll.shell32.IsUserAnAdmin())
        except OSError:
            unavailable = True
    marker = _HANDLE_ENUM_UNAVAILABLE if os.name == "nt" and unavailable else None
    return hermes, holders, marker


def diagnose(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=False)
    hermes, holders, marker = _process_snapshot(path)
    current_cwd = Path.cwd().resolve(strict=False)
    holders = [row for row in holders if row.get("pid") != os.getpid()]
    probe = _rename_probe_without_current_cwd(path)
    hermes_pids = {row.get("pid") for row in hermes}
    hermes_holders = [row for row in holders if row.get("pid") in hermes_pids]
    action = "NONE"
    if not probe["ok"]:
        if not path.exists():
            action = "CHECK_PATH"
        elif hermes_holders:
            action = "RUN_RELEASE_PATH_OR_CLOSE_MATCHING_HERMES_SESSION"
        elif holders:
            action = "CLOSE_OR_CHDIR_LISTED_PROCESS"
        else:
            action = "LOCK_OWNER_UNKNOWN_RESTART_HERMES_IF_STILL_LOCKED"
    return {
        "path": str(path),
        "path_exists": path.exists(),
        "is_current_process_cwd": _same_or_child(str(current_cwd), path),
        "hermes_processes": hermes,
        "child_processes": [row for row in hermes if row.get("role") != "desktop-main"],
        "known_open_handles": holders,
        "lock_owner_if_available": holders or None,
        "handle_enumeration": marker or "BEST_EFFORT_SAME_USER_PROCESS_CWD",
        "rename_probe": probe,
        "winerror": probe.get("winerror"),
        "recommended_action": action,
    }


def _release_current_process_cwd(path: Path) -> bool:
    if not _same_or_child(str(Path.cwd().resolve(strict=False)), path):
        return False

    from hermes_constants import get_hermes_home

    stable_cwd = Path(get_hermes_home()).expanduser().resolve(strict=False)
    if _same_or_child(str(stable_cwd), path):
        return False
    try:
        stable_cwd.mkdir(parents=True, exist_ok=True)
        os.chdir(stable_cwd)
        return True
    except OSError:
        return False


def release_path(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=False)
    current_cwd_released = _release_current_process_cwd(path)
    before = diagnose(str(path))
    if before["rename_probe"]["ok"]:
        action = "CURRENT_PROCESS_CWD_RELEASED" if current_cwd_released else "NO_ACTIVE_LOCK"
        return {"released": True, "action": action, "diagnostic": before}
    response = _request_desktop_release(path)
    after = diagnose(str(path))
    if response and response.get("released") and after["rename_probe"]["ok"]:
        return {"released": True, "action": "GRACEFUL_DESKTOP_RELEASE", "desktop": response, "diagnostic": after}
    return {
        "released": False,
        "action": "GRACEFUL_DESKTOP_RELEASE_FAILED_RESTART_REQUIRED",
        "desktop": response,
        "diagnostic": after,
    }


def _request_desktop_release(path: Path, timeout: float = 5.0) -> dict[str, Any] | None:
    from hermes_constants import get_hermes_home

    control = Path(get_hermes_home()) / "cache" / "project-control"
    control.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    request = control / f"request-{request_id}.json"
    response = control / f"response-{request_id}.json"
    temporary = control / f".{request.name}.tmp"
    temporary.write_text(json.dumps({"id": request_id, "path": str(path)}), encoding="utf-8")
    os.replace(temporary, request)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                return json.loads(response.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                time.sleep(0.05)
        return None
    finally:
        for item in (request, response, temporary):
            try:
                item.unlink()
            except FileNotFoundError:
                pass
