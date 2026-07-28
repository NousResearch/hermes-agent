"""Native Windows trust boundary for Desktop SSH backend lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON = 1024 * 1024
_MAX_LOG = 512 * 1024
_OPEN_REPARSE_POINT = 0x00200000
_DELETE_ON_CLOSE = 0x04000000
_MOVE_REPLACE_EXISTING = 0x00000001
_MOVE_WRITE_THROUGH = 0x00000008


def _win32() -> tuple[Any, ...]:
    if sys.platform != "win32":
        raise RuntimeError("Windows SSH runtime is only available on Windows")
    import ntsecuritycon
    import pywintypes
    import win32api
    import win32con
    import win32file
    import win32process
    import win32security
    return ntsecuritycon, pywintypes, win32api, win32con, win32file, win32process, win32security


def _ownership(value: str) -> str:
    if not _HEX32.fullmatch(value or ""):
        raise ValueError("invalid ownership ID")
    return value


def _nonce(value: str) -> str:
    if not _HEX16.fullmatch(value or ""):
        raise ValueError("invalid spawn nonce")
    return value


def _root() -> Path:
    return get_hermes_home() / "desktop-ssh"


def _directory(ownership_id: str) -> Path:
    return _root() / _ownership(ownership_id)


def _token_path(ownership_id: str, spawn_nonce: str) -> Path:
    return _directory(ownership_id) / f"{_nonce(spawn_nonce)}.token"


def _log_path(ownership_id: str, spawn_nonce: str) -> Path:
    return _directory(ownership_id) / f"{_nonce(spawn_nonce)}.log"


def _lock_path(ownership_id: str) -> Path:
    return _directory(ownership_id) / "backend.lock.json"


def _current_sid():
    _, _, win32api, win32con, _, _, win32security = _win32()
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    return win32security.GetTokenInformation(token, win32security.TokenUser)[0]


def _system_sid():
    return _win32()[6].ConvertStringSidToSid("S-1-5-18")


def _security_attributes():
    ntsecuritycon, _, _, _, _, _, win32security = _win32()
    owner = _current_sid()
    acl = win32security.ACL()
    for sid in (owner, _system_sid()):
        acl.AddAccessAllowedAceEx(win32security.ACL_REVISION, 0, ntsecuritycon.FILE_ALL_ACCESS, sid)
    descriptor = win32security.SECURITY_DESCRIPTOR()
    descriptor.SetSecurityDescriptorOwner(owner, False)
    descriptor.SetSecurityDescriptorDacl(True, acl, False)
    # Protect the DACL so inheritable parent ACEs (%LOCALAPPDATA% grants) are not merged in.
    descriptor.SetSecurityDescriptorControl(win32security.SE_DACL_PROTECTED, win32security.SE_DACL_PROTECTED)
    attributes = win32security.SECURITY_ATTRIBUTES()
    attributes.SECURITY_DESCRIPTOR = descriptor
    return attributes


def _allowed_sids():
    win32security = _win32()[6]
    return {win32security.ConvertSidToStringSid(_current_sid()),
            win32security.ConvertSidToStringSid(_system_sid())}


def _verify_security(handle) -> None:
    _, _, _, _, _, _, win32security = _win32()
    info = win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
    descriptor = win32security.GetSecurityInfo(handle, win32security.SE_FILE_OBJECT, info)
    owner = descriptor.GetSecurityDescriptorOwner()
    owner_value = win32security.ConvertSidToStringSid(owner)
    if owner_value not in _allowed_sids():
        raise OSError("Windows SSH runtime object has the wrong owner")
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None:
        raise OSError("Windows SSH runtime object has a null DACL")
    allowed = _allowed_sids()
    allow_types = {
        win32security.ACCESS_ALLOWED_ACE_TYPE,
        win32security.ACCESS_ALLOWED_OBJECT_ACE_TYPE,
        getattr(win32security, "ACCESS_ALLOWED_CALLBACK_ACE_TYPE", 9),
        getattr(win32security, "ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE", 11),
    }
    for index in range(dacl.GetAceCount()):
        ace = dacl.GetAce(index)
        ace_type = ace[0][0]
        mask = ace[1]
        sid = ace[-1]
        if ace_type in allow_types and mask:
            if win32security.ConvertSidToStringSid(sid) not in allowed:
                raise OSError("Windows SSH runtime object has a permissive DACL")


def _open(path: Path, access: int, creation: int, flags: int, share: int = 0):
    _, _, _, _, win32file, _, _ = _win32()
    handle = win32file.CreateFile(str(path), access, share, _security_attributes(), creation, flags, None)
    try:
        actual = win32file.GetFinalPathNameByHandle(handle, 0)
        if actual.startswith("\\\\?\\"):
            actual = actual[4:]
        expected = os.path.abspath(str(path))
        if os.path.normcase(actual) != os.path.normcase(expected):
            raise OSError("Windows SSH runtime handle escaped its expected path")
        attributes = win32file.GetFileInformationByHandle(handle)[0]
        if attributes & 0x400:
            raise OSError("Windows SSH runtime path contains a reparse point")
        _verify_security(handle)
        return handle
    except BaseException:
        win32file.CloseHandle(handle)
        raise


def _ensure_directory(path: Path) -> None:
    _, pywintypes, _, win32con, win32file, _, _ = _win32()
    if path.parent != path:
        _ensure_directory(path.parent) if path.parent not in (Path(path.anchor), path) and not path.parent.exists() else None
    if not path.exists():
        try:
            win32file.CreateDirectory(str(path), _security_attributes())
        except pywintypes.error as exc:
            if exc.winerror != 183:
                raise
    handle = _open(path, win32con.GENERIC_READ | win32con.READ_CONTROL, win32con.OPEN_EXISTING,
                   win32con.FILE_FLAG_BACKUP_SEMANTICS | _OPEN_REPARSE_POINT,
                   win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE)
    win32file.CloseHandle(handle)


def _ensure_scope(ownership_id: str) -> Path:
    root = _root()
    _ensure_directory(root)
    directory = _directory(ownership_id)
    _ensure_directory(directory)
    return directory


def upload_token(ownership_id: str, spawn_nonce: str, token: bytes) -> dict[str, Any]:
    _, _, _, win32con, win32file, _, _ = _win32()
    if len(token) != 64 or not _HEX64.fullmatch(token.decode("ascii", errors="ignore")):
        raise ValueError("invalid session token")
    _ensure_scope(ownership_id)
    path = _token_path(ownership_id, spawn_nonce)
    try:
        handle = _open(path, win32con.GENERIC_WRITE | win32con.READ_CONTROL, win32con.CREATE_NEW,
                       win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT)
        try:
            win32file.WriteFile(handle, token)
            win32file.FlushFileBuffers(handle)
        finally:
            win32file.CloseHandle(handle)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"path": str(path)}


def read_token(path_value: str) -> str:
    _, _, _, win32con, win32file, _, _ = _win32()
    path = Path(path_value)
    root = _root()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SystemExit("--ssh-session-token-file must be under the desktop-ssh directory") from exc
    if len(relative.parts) != 2 or not _HEX32.fullmatch(relative.parts[0]) or not re.fullmatch(r"[0-9a-f]{16}\.token", relative.parts[1]):
        raise SystemExit("--ssh-session-token-file has an invalid runtime path")
    flags = win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT | _DELETE_ON_CLOSE
    try:
        handle = _open(path, win32con.GENERIC_READ | win32con.READ_CONTROL | win32con.DELETE,
                       win32con.OPEN_EXISTING, flags)
    except Exception as exc:
        raise SystemExit("--ssh-session-token-file is not accessible") from exc
    try:
        _, data = win32file.ReadFile(handle, 65)
    finally:
        win32file.CloseHandle(handle)
    token = data.decode("ascii", errors="ignore")
    if len(token) != 64 or not _HEX64.fullmatch(token):
        raise SystemExit("--ssh-session-token-file contains an invalid token")
    return token


def _read_json_stdin() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(_MAX_JSON + 1)
    if len(raw) > _MAX_JSON:
        raise ValueError("runtime payload is too large")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("runtime payload must be an object")
    return parsed


def read_lock(ownership_id: str) -> dict[str, Any] | None:
    _, pywintypes, _, win32con, win32file, _, _ = _win32()
    _ensure_scope(ownership_id)
    path = _lock_path(ownership_id)
    try:
        handle = _open(path, win32con.GENERIC_READ | win32con.READ_CONTROL, win32con.OPEN_EXISTING,
                       win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT,
                       win32con.FILE_SHARE_READ)
    except pywintypes.error as exc:
        if exc.winerror in (2, 3):
            return None
        raise
    try:
        _, data = win32file.ReadFile(handle, _MAX_JSON + 1)
    finally:
        win32file.CloseHandle(handle)
    if len(data) > _MAX_JSON:
        return None
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def write_lock(ownership_id: str, payload: dict[str, Any]) -> None:
    _, _, _, win32con, win32file, _, _ = _win32()
    directory = _ensure_scope(ownership_id)
    data = json.dumps(payload, separators=(",", ":")).encode()
    if len(data) > _MAX_JSON:
        raise ValueError("lock payload is too large")
    temporary = directory / f".{os.urandom(8).hex()}.lock.tmp"
    handle = _open(temporary, win32con.GENERIC_WRITE | win32con.READ_CONTROL, win32con.CREATE_NEW,
                   win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT)
    try:
        win32file.WriteFile(handle, data)
        win32file.FlushFileBuffers(handle)
    finally:
        win32file.CloseHandle(handle)
    win32file.MoveFileEx(str(temporary), str(_lock_path(ownership_id)),
                         _MOVE_REPLACE_EXISTING | _MOVE_WRITE_THROUGH)


def remove_artifact(path: Path) -> bool:
    _, pywintypes, _, win32con, win32file, _, _ = _win32()
    try:
        handle = _open(path, win32con.DELETE | win32con.READ_CONTROL, win32con.OPEN_EXISTING,
                       win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT | _DELETE_ON_CLOSE)
    except pywintypes.error as exc:
        if exc.winerror in (2, 3):
            return False
        raise
    win32file.CloseHandle(handle)
    return True


def process_state(pid: int, creation_time_ns: int, hermes_path: str, spawn_nonce: str) -> dict[str, Any]:
    import psutil
    _nonce(spawn_nonce)
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess as exc:
        return {"alive": False, "owned": False, "indeterminate": False, "reason": type(exc).__name__}
    try:
        actual_creation = int(process.create_time() * 1_000_000_000)
        argv = process.cmdline()
    except psutil.NoSuchProcess as exc:
        return {"alive": False, "owned": False, "indeterminate": False, "reason": type(exc).__name__}
    except psutil.AccessDenied as exc:
        return {"alive": True, "owned": False, "indeterminate": True, "reason": type(exc).__name__}
    if actual_creation != creation_time_ns:
        return {"alive": False, "owned": False, "indeterminate": False, "reason": "creation-time",
                "actualCreationTimeNs": str(actual_creation), "expectedCreationTimeNs": str(creation_time_ns)}
    if not argv:
        return {"alive": True, "owned": False, "indeterminate": True, "reason": "argv-unavailable"}
    expected = os.path.normcase(os.path.abspath(hermes_path))
    arg0 = os.path.normcase(os.path.abspath(argv[0]))
    # argv[0] is either the hermes exe directly, or (normal case) the base Python
    # interpreter -- its exact path varies by venv/uv layout, so match on "a python
    # running our module". We launch via `-c` bootstrap, so it shows as
    # `-c <bootstrap that runs hermes_cli.main>`; also accept a plain `-m` launch.
    # Identity is anchored by the unforgeable creation-time + secret owner-nonce below.
    is_python = os.path.basename(arg0).startswith("python")
    launches_module = (
        argv[1:3] == ["-m", "hermes_cli.main"]
        or (len(argv) > 2 and argv[1] == "-c" and "hermes_cli.main" in argv[2])
    )
    executable_match = arg0 == expected or (is_python and launches_module)
    try:
        serve = argv.index("serve")
        owner = argv.index("--ssh-owner-nonce", serve + 1)
        owned = executable_match and "--isolated" in argv[serve + 1:] and argv[owner + 1] == spawn_nonce
    except (ValueError, IndexError):
        owned = False
    return {"alive": process.is_running(), "owned": owned, "indeterminate": False,
            "creationTimeNs": str(actual_creation), "reason": "owned" if owned else "argv",
            "argv": argv[:20], "expectedExecutable": expected}


def terminate_owned(pid: int, creation_time_ns: int, hermes_path: str, spawn_nonce: str) -> bool:
    state = process_state(pid, creation_time_ns, hermes_path, spawn_nonce)
    if not state["alive"] or not state["owned"]:
        return False
    import psutil
    process = psutil.Process(pid)
    if int(process.create_time() * 1_000_000_000) != creation_time_ns:
        return False
    process.terminate()
    try:
        process.wait(5)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(5)
    return True


def _resolve_direct_interpreter(python_entry: str) -> tuple[str, list[str]]:
    """Resolve the venv launcher to the base interpreter it would exec, plus the
    sys.path to reproduce. On Windows a venv Scripts\\python.exe is a launcher stub
    that spawns the real interpreter as a CHILD (two PIDs); spawning the base
    interpreter directly with the launcher's sys.path injected yields ONE process
    that both owns the port and is the process we lock.

    Also resolves hermes_cli's own location and prepends its parent, because the
    launcher finds the module via cwd / an editable-install path hook that a bare
    PYTHONPATH does not reproduce."""
    query = (
        "import sys,json,os,importlib.util as u;"
        "s=u.find_spec('hermes_cli');"
        "root=os.path.dirname(os.path.dirname(s.origin)) if s and s.origin else '';"
        "print(json.dumps({'base':getattr(sys,'_base_executable','') or sys.executable,"
        "'path':[p for p in sys.path if p],'root':root}))"
    )
    out = subprocess.run([python_entry, "-c", query], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    if out.returncode != 0:
        raise ValueError("could not resolve the base Python interpreter")
    info = json.loads(out.stdout.strip().splitlines()[-1])
    base = info["base"]
    if not base or not os.path.isfile(base):
        raise ValueError("base Python interpreter was not found")
    # drop editable-install finder markers ('__editable__.*.finder.__path_hook__')
    # that PYTHONPATH cannot reproduce; keep only real filesystem entries.
    py_path = [p for p in info.get("path", []) if os.path.exists(p)]
    # the hermes_cli package parent must be explicit -- the launcher resolves it via
    # cwd or an editable path hook, neither of which survives a bare PYTHONPATH spawn.
    root = info.get("root") or ""
    if root and os.path.isdir(root) and root not in py_path:
        py_path.insert(0, root)
    if not root or not os.path.isdir(root):
        raise ValueError("could not locate the hermes_cli package")
    return base, py_path


def spawn_backend(payload: dict[str, Any]) -> dict[str, Any]:
    ownership_id = _ownership(str(payload["ownershipId"]))
    spawn_nonce = _nonce(str(payload["spawnNonce"]))
    configured_path = str(payload["hermesPath"])
    if not os.path.isabs(configured_path):
        raise ValueError("Hermes path must be absolute")
    hermes_path = os.path.abspath(configured_path)
    token_path = str(_token_path(ownership_id, spawn_nonce))
    log_path = _log_path(ownership_id, spawn_nonce)
    profile = str(payload.get("profile") or "")
    if len(profile) > 256 or any(ch in profile for ch in "\x00\r\n"):
        raise ValueError("invalid profile")
    venv_dir = os.path.dirname(hermes_path)
    python_entry = os.path.join(venv_dir, "python.exe")
    if not os.path.isfile(python_entry):
        raise ValueError("Hermes Python runtime was not found")
    base_python, sys_path = _resolve_direct_interpreter(python_entry)
    # Seed sys.path IN-PROCESS via a -c bootstrap rather than exporting PYTHONPATH:
    # PYTHONPATH would be inherited by every subprocess the running backend spawns
    # (terminal tool, user scripts), shadowing their imports. This keeps the path
    # scoped to the backend process alone.
    bootstrap = (
        "import sys,runpy;"
        f"sys.path[:0]={sys_path!r};"
        "runpy.run_module('hermes_cli.main',run_name='__main__',alter_sys=True)"
    )
    args = [base_python, "-c", bootstrap]
    if profile:
        args.extend(["--profile", profile])
    args.extend(["serve", "--isolated", "--host", "127.0.0.1", "--port", "0",
                 "--ssh-session-token-file", token_path, "--ssh-owner-nonce", spawn_nonce])
    # VIRTUAL_ENV preserves venv identity; PYTHONPATH is deliberately NOT set (see above).
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = os.path.dirname(venv_dir)
    env.pop("PYTHONPATH", None)
    _ensure_scope(ownership_id)
    creationflags = 0x00000008 | 0x00000200 | 0x01000000
    _, _, _, win32con, win32file, _, _ = _win32()
    log_handle = _open(log_path, win32con.GENERIC_WRITE | win32con.READ_CONTROL,
                       win32con.CREATE_NEW, win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT,
                       win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE)
    import msvcrt
    log_fd = msvcrt.open_osfhandle(int(log_handle), os.O_WRONLY)
    with os.fdopen(log_fd, "wb", buffering=0) as log_stream:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log_stream, stderr=log_stream,
                                   close_fds=True, creationflags=creationflags, env=env)
    creation_time_ns = int(__import__("psutil").Process(process.pid).create_time() * 1_000_000_000)
    return {"pid": process.pid, "creationTimeNs": str(creation_time_ns),
            "logPath": str(log_path), "tokenPath": token_path}


def inspect_hermes(hermes_path: str) -> dict[str, Any]:
    path = os.path.abspath(hermes_path)
    if not os.path.isabs(hermes_path) or not os.path.isfile(path):
        raise ValueError("Hermes path is not an executable file")
    version = subprocess.run([path, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
    help_result = subprocess.run([path, "serve", "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
    help_text = help_result.stdout + help_result.stderr
    return {
        "path": path,
        "version": (version.stdout + version.stderr).splitlines()[0] if version.returncode == 0 else "",
        "supported": "--ssh-session-token-file" in help_text and "--ssh-owner-nonce" in help_text,
    }


def dispatch(argv: list[str]) -> Any:
    if not argv:
        raise ValueError("missing operation")
    operation = argv[0]
    if operation == "probe":
        import platform
        return {"os": "Windows", "arch": platform.machine(), "hermesHome": str(get_hermes_home()), "python": sys.executable}
    if operation == "upload-token" and len(argv) == 3:
        return upload_token(argv[1], argv[2], sys.stdin.buffer.read(65))
    if operation == "read-lock" and len(argv) == 2:
        return read_lock(argv[1])
    if operation == "write-lock" and len(argv) == 2:
        write_lock(argv[1], _read_json_stdin()); return {"ok": True}
    if operation == "remove-lock" and len(argv) == 2:
        return {"removed": remove_artifact(_lock_path(argv[1]))}
    if operation == "remove-token" and len(argv) == 3:
        return {"removed": remove_artifact(_token_path(argv[1], argv[2]))}
    if operation == "read-log" and len(argv) == 3:
        path = _log_path(argv[1], argv[2])
        _, pywintypes, _, win32con, win32file, _, _ = _win32()
        try:
            handle = _open(path, win32con.GENERIC_READ | win32con.READ_CONTROL,
                           win32con.OPEN_EXISTING, win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT,
                           win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE)
        except pywintypes.error as exc:
            if exc.winerror in (2, 3):
                return {"content": ""}
            raise
        try:
            _, data = win32file.ReadFile(handle, _MAX_LOG)
        finally:
            win32file.CloseHandle(handle)
        return {"content": data.decode(errors="replace")}
    if operation == "remove-log" and len(argv) == 3:
        return {"removed": remove_artifact(_log_path(argv[1], argv[2]))}
    if operation == "spawn":
        return spawn_backend(_read_json_stdin())
    if operation == "inspect" and len(argv) == 2:
        return inspect_hermes(argv[1])
    if operation == "process-state" and len(argv) == 5:
        return process_state(int(argv[1]), int(argv[2]), argv[3], argv[4])
    if operation == "terminate" and len(argv) == 5:
        return {"terminated": terminate_owned(int(argv[1]), int(argv[2]), argv[3], argv[4])}
    raise ValueError("invalid operation")


def main() -> None:
    try:
        print(json.dumps(dispatch(sys.argv[1:]), separators=(",", ":")))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
