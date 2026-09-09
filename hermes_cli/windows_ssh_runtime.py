"""Native Windows trust boundary for Desktop SSH backend lifecycle."""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hermes_constants import get_default_hermes_root

_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON = 1024 * 1024
_MAX_LOG = 512 * 1024
_OPEN_REPARSE_POINT = 0x00200000
_DELETE_ON_CLOSE = 0x04000000
_MOVE_REPLACE_EXISTING = 0x00000001
_MOVE_WRITE_THROUGH = 0x00000008


def _win32() -> Any:
    """Namespace of the pywin32 modules (ntsecuritycon, pywintypes, win32api, win32con, win32file,
    win32security); import is deferred so the module imports on non-Windows hosts."""
    if sys.platform != "win32":
        raise RuntimeError("Windows SSH runtime is only available on Windows")
    names = ("ntsecuritycon", "pywintypes", "win32api", "win32con", "win32file", "win32security")
    return SimpleNamespace(**{name: importlib.import_module(name) for name in names})


def _check(pattern: re.Pattern, value: str, message: str) -> str:
    if not pattern.fullmatch(value or ""):
        raise ValueError(message)
    return value


def _ownership(value: str) -> str:
    return _check(_HEX32, value, "invalid ownership ID")


def _nonce(value: str) -> str:
    return _check(_HEX16, value, "invalid spawn nonce")


def _root() -> Path:
    # The helper uploads the token before the child applies `--profile`; read_token() runs after
    # profile activation. Anchor both to the machine root so a named profile (or custom
    # HERMES_HOME) cannot move the reader away from the helper's token.
    return get_default_hermes_root() / "desktop-ssh"


def _directory(ownership_id: str) -> Path:
    return _root() / _ownership(ownership_id)


def _token_path(ownership_id: str, spawn_nonce: str) -> Path:
    return _directory(ownership_id) / f"{_nonce(spawn_nonce)}.token"


def _log_path(ownership_id: str, spawn_nonce: str) -> Path:
    return _directory(ownership_id) / f"{_nonce(spawn_nonce)}.log"


def _lock_path(ownership_id: str) -> Path:
    return _directory(ownership_id) / "backend.lock.json"


def _current_sid():
    w = _win32()
    token = w.win32security.OpenProcessToken(w.win32api.GetCurrentProcess(), w.win32con.TOKEN_QUERY)
    return w.win32security.GetTokenInformation(token, w.win32security.TokenUser)[0]


def _system_sid():
    return _win32().win32security.ConvertStringSidToSid("S-1-5-18")


def _sid_str(sid) -> str:
    return _win32().win32security.ConvertSidToStringSid(sid)


def _security_attributes():
    w = _win32()
    ntsecuritycon, win32security = w.ntsecuritycon, w.win32security
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
    return {_sid_str(_current_sid()), _sid_str(_system_sid())}


def _verify_security(handle) -> None:
    win32security = _win32().win32security
    info = win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
    descriptor = win32security.GetSecurityInfo(handle, win32security.SE_FILE_OBJECT, info)
    allowed = _allowed_sids()
    if _sid_str(descriptor.GetSecurityDescriptorOwner()) not in allowed:
        raise OSError("Windows SSH runtime object has the wrong owner")
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None:
        raise OSError("Windows SSH runtime object has a null DACL")
    allow_types = {
        win32security.ACCESS_ALLOWED_ACE_TYPE,
        win32security.ACCESS_ALLOWED_OBJECT_ACE_TYPE,
        getattr(win32security, "ACCESS_ALLOWED_CALLBACK_ACE_TYPE", 9),
        getattr(win32security, "ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE", 11)}
    for index in range(dacl.GetAceCount()):
        ace = dacl.GetAce(index)
        if ace[0][0] in allow_types and ace[1] and _sid_str(ace[-1]) not in allowed:
            raise OSError("Windows SSH runtime object has a permissive DACL")


def _open(path: Path, access: int, creation: int, flags: int, share: int = 0):
    win32file = _win32().win32file
    handle = win32file.CreateFile(str(path), access, share, _security_attributes(), creation, flags, None)
    try:
        actual = win32file.GetFinalPathNameByHandle(handle, 0).removeprefix("\\\\?\\")
        if os.path.normcase(actual) != os.path.normcase(os.path.abspath(str(path))):
            raise OSError("Windows SSH runtime handle escaped its expected path")
        if win32file.GetFileInformationByHandle(handle)[0] & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise OSError("Windows SSH runtime path contains a reparse point")
        _verify_security(handle)
        return handle
    except BaseException:
        win32file.CloseHandle(handle)
        raise


def _open_existing(path: Path, access: int, extra_flags: int = 0, share: int = 0):
    """``_open`` an existing file (FILE_ATTRIBUTE_NORMAL | reparse guard); None when it is missing."""
    w = _win32()
    try:
        return _open(path, access, w.win32con.OPEN_EXISTING,
                     w.win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT | extra_flags, share)
    except w.pywintypes.error as exc:
        if exc.winerror in (2, 3):
            return None
        raise


def _read_shared(path: Path, limit: int, share: int) -> bytes | None:
    """Read up to ``limit`` bytes of ``path`` opened read-only with ``share``; None when missing."""
    w = _win32()
    handle = _open_existing(path, w.win32con.GENERIC_READ | w.win32con.READ_CONTROL, share=share)
    if handle is None:
        return None
    try:
        return w.win32file.ReadFile(handle, limit)[1]
    finally:
        w.win32file.CloseHandle(handle)


def _write_new(path: Path, data: bytes, share: int = 0) -> None:
    """Create ``path`` (CREATE_NEW), write ``data`` and flush."""
    w = _win32()
    win32con, win32file = w.win32con, w.win32file
    handle = _open(path, win32con.GENERIC_WRITE | win32con.READ_CONTROL, win32con.CREATE_NEW,
                   win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT, share)
    try:
        win32file.WriteFile(handle, data)
        win32file.FlushFileBuffers(handle)
    finally:
        win32file.CloseHandle(handle)


def _ensure_directory(path: Path) -> None:
    w = _win32()
    pywintypes, win32con, win32file = w.pywintypes, w.win32con, w.win32file
    if path.parent not in (Path(path.anchor), path) and not path.parent.exists():
        _ensure_directory(path.parent)
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
    _ensure_directory(_root())
    directory = _directory(ownership_id)
    _ensure_directory(directory)
    return directory


def upload_token(ownership_id: str, spawn_nonce: str, token: bytes) -> dict[str, Any]:
    if len(token) != 64 or not _HEX64.fullmatch(token.decode("ascii", errors="ignore")):
        raise ValueError("invalid session token")
    _ensure_scope(ownership_id)
    path = _token_path(ownership_id, spawn_nonce)
    try:
        _write_new(path, token)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"path": str(path)}


def read_token(path_value: str) -> str:
    w = _win32()
    win32con, win32file = w.win32con, w.win32file
    path = Path(path_value)
    try:
        relative = path.relative_to(_root())
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
    win32con = _win32().win32con
    _ensure_scope(ownership_id)
    data = _read_shared(_lock_path(ownership_id), _MAX_JSON + 1, win32con.FILE_SHARE_READ)
    if data is None or len(data) > _MAX_JSON:
        return None
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def write_lock(ownership_id: str, payload: dict[str, Any]) -> None:
    win32file = _win32().win32file
    directory = _ensure_scope(ownership_id)
    data = json.dumps(payload, separators=(",", ":")).encode()
    if len(data) > _MAX_JSON:
        raise ValueError("lock payload is too large")
    temporary = directory / f".{os.urandom(8).hex()}.lock.tmp"
    _write_new(temporary, data)
    win32file.MoveFileEx(str(temporary), str(_lock_path(ownership_id)),
                         _MOVE_REPLACE_EXISTING | _MOVE_WRITE_THROUGH)


def remove_artifact(path: Path) -> bool:
    w = _win32()
    handle = _open_existing(path, w.win32con.DELETE | w.win32con.READ_CONTROL, _DELETE_ON_CLOSE)
    if handle is None:
        return False
    w.win32file.CloseHandle(handle)
    return True


def process_state(pid: int, creation_time_ns: int, hermes_path: str, spawn_nonce: str) -> dict[str, Any]:
    import psutil
    _nonce(spawn_nonce)
    try:
        process = psutil.Process(pid)
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
    # argv[0] is the hermes exe or (normally) the base Python, whose path varies by venv/uv
    # layout — so match "a python running our module" (`-c` bootstrap or plain `-m`). Identity
    # is anchored by the unforgeable creation-time + secret owner-nonce below.
    is_python = os.path.basename(arg0).startswith("python")
    launches_module = (
        argv[1:3] == ["-m", "hermes_cli.main"]
        or (len(argv) > 2 and argv[1] == "-c" and "hermes_cli.main" in argv[2]))
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
    """Resolve the venv launcher to (base interpreter, sys.path to reproduce).

    On Windows a venv Scripts\\python.exe is a stub that spawns the real interpreter as a CHILD
    (two PIDs); spawning the base interpreter with the launcher's sys.path injected yields ONE
    process that both owns the port and is the one we lock. hermes_cli's parent is prepended
    because the launcher finds it via cwd / an editable-install hook a bare PYTHONPATH lacks."""
    query = (
        "import sys,json,os,importlib.util as u;"
        "s=u.find_spec('hermes_cli');"
        "root=os.path.dirname(os.path.dirname(s.origin)) if s and s.origin else '';"
        "print(json.dumps({'base':getattr(sys,'_base_executable','') or sys.executable,"
        "'path':[p for p in sys.path if p],'root':root}))")
    out = subprocess.run([python_entry, "-c", query], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    if out.returncode != 0:
        raise ValueError("could not resolve the base Python interpreter")
    info = json.loads(out.stdout.strip().splitlines()[-1])
    base = info["base"]
    if not base or not os.path.isfile(base):
        raise ValueError("base Python interpreter was not found")
    # keep only real filesystem entries (drops '__editable__.*' finder markers)
    py_path = [p for p in info.get("path", []) if os.path.exists(p)]
    root = info.get("root") or ""
    if not root or not os.path.isdir(root):
        raise ValueError("could not locate the hermes_cli package")
    if root not in py_path:
        py_path.insert(0, root)
    return base, py_path


def spawn_backend(payload: dict[str, Any]) -> dict[str, Any]:
    ownership_id = _ownership(str(payload["ownershipId"]))
    spawn_nonce = _nonce(str(payload["spawnNonce"]))
    configured_path = str(payload["hermesPath"])
    if not os.path.isabs(configured_path):
        raise ValueError("Hermes path must be absolute")
    hermes_path = os.path.abspath(configured_path)
    token_path = str(_token_path(ownership_id, spawn_nonce))
    profile = str(payload.get("profile") or "")
    if len(profile) > 256 or any(ch in profile for ch in "\x00\r\n"):
        raise ValueError("invalid profile")
    venv_dir = os.path.dirname(hermes_path)
    python_entry = os.path.join(venv_dir, "python.exe")
    if not os.path.isfile(python_entry):
        raise ValueError("Hermes Python runtime was not found")
    base_python, sys_path = _resolve_direct_interpreter(python_entry)
    # Seed sys.path IN-PROCESS via -c rather than PYTHONPATH, which every subprocess the backend
    # spawns (terminal tool, user scripts) would inherit, shadowing their imports.
    bootstrap = (
        "import sys,runpy;"
        f"sys.path[:0]={sys_path!r};"
        "runpy.run_module('hermes_cli.main',run_name='__main__',alter_sys=True)")
    args = [base_python, "-c", bootstrap]
    if profile:
        args.extend(["--profile", profile])
    args.extend(["serve", "--isolated", "--host", "127.0.0.1", "--port", "0",
                 "--ssh-session-token-file", token_path, "--ssh-owner-nonce", spawn_nonce])
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = os.path.dirname(venv_dir)
    env.pop("PYTHONPATH", None)
    _ensure_scope(ownership_id)
    log_path = _log_path(ownership_id, spawn_nonce)
    win32con = _win32().win32con
    log_handle = _open(log_path, win32con.GENERIC_WRITE | win32con.READ_CONTROL,
                       win32con.CREATE_NEW, win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT,
                       win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE)
    import msvcrt
    log_fd = msvcrt.open_osfhandle(int(log_handle), os.O_WRONLY)
    with os.fdopen(log_fd, "wb", buffering=0) as log_stream:
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log_stream, stderr=log_stream,
                                   close_fds=True, creationflags=0x00000008 | 0x00000200 | 0x01000000, env=env)
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
        "supported": "--ssh-session-token-file" in help_text and "--ssh-owner-nonce" in help_text}


def _probe(*_: str) -> dict[str, Any]:
    import platform
    return {"os": "Windows", "arch": platform.machine(), "hermesHome": str(get_default_hermes_root()), "python": sys.executable}


def _read_log(ownership_id: str, spawn_nonce: str) -> dict[str, Any]:
    win32con = _win32().win32con
    data = _read_shared(_log_path(ownership_id, spawn_nonce), _MAX_LOG,
                        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE)
    return {"content": "" if data is None else data.decode(errors="replace")}


def _write_lock_op(ownership_id: str) -> dict[str, Any]:
    write_lock(ownership_id, _read_json_stdin())
    return {"ok": True}


# operation -> (argument count or None for "any", handler(*args)).
_OPERATIONS: dict[str, tuple[int | None, Any]] = {
    "probe": (None, _probe),
    "upload-token": (2, lambda o, n: upload_token(o, n, sys.stdin.buffer.read(65))),
    "read-lock": (1, read_lock),
    "write-lock": (1, _write_lock_op),
    "remove-lock": (1, lambda o: {"removed": remove_artifact(_lock_path(o))}),
    "remove-token": (2, lambda o, n: {"removed": remove_artifact(_token_path(o, n))}),
    "read-log": (2, _read_log),
    "remove-log": (2, lambda o, n: {"removed": remove_artifact(_log_path(o, n))}),
    "spawn": (None, lambda *_: spawn_backend(_read_json_stdin())),
    "inspect": (1, inspect_hermes),
    "process-state": (4, lambda p, c, h, n: process_state(int(p), int(c), h, n)),
    "terminate": (4, lambda p, c, h, n: {"terminated": terminate_owned(int(p), int(c), h, n)})}


def dispatch(argv: list[str]) -> Any:
    if not argv:
        raise ValueError("missing operation")
    operation, args = argv[0], argv[1:]
    entry = _OPERATIONS.get(operation)
    if entry is None or (entry[0] is not None and len(args) != entry[0]):
        raise ValueError("invalid operation")
    return entry[1](*args)


def main() -> None:
    try:
        print(json.dumps(dispatch(sys.argv[1:]), separators=(",", ":")))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
