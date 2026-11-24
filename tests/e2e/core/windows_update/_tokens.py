"""Standard-user tokens for the Windows install/update journeys (the runner session is elevated).

Two uses:

* :func:`standard_user_access` asks the kernel (AccessCheck) whether a standard-user token may
  open a path with a given access mask, without starting anything.
* :class:`StandardUserProcess` starts a real process under that token, so a journey can run
  ``hermes gateway run``, ``hermes gateway status`` or a turn exactly as a non-elevated user's
  console, login item or Desktop would.

The token is the runner's own token passed through ``CreateRestrictedToken`` with
BUILTIN\\Administrators deny-only, every privilege but SeChangeNotify dropped (``LUA_TOKEN``) and
Medium integrity: the shape of UAC's filtered token. It never depends on the host's SAFER policy
(Blacksmith's image hands back a SAFER token with Administrators still enabled).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from ctypes import wintypes
from pathlib import Path

_ADMINISTRATORS = "S-1-5-32-544"
_SYSTEM = "S-1-5-18"
_MEDIUM_IL = "S-1-16-8192"
FILE_READ_EXECUTE = 0x1200A9  # FILE_GENERIC_READ | FILE_GENERIC_EXECUTE
FILE_READ_WRITE = 0x12019F  # FILE_GENERIC_READ | FILE_GENERIC_WRITE
FILE_ALL_ACCESS = 0x1F01FF

_handle = wintypes.HANDLE


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _GenericMapping(ctypes.Structure):
    _fields_ = [("GenericRead", wintypes.DWORD), ("GenericWrite", wintypes.DWORD),
                ("GenericExecute", wintypes.DWORD), ("GenericAll", wintypes.DWORD)]


def _dlls():
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = _handle
    kernel32.CloseHandle.argtypes = [_handle]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    advapi32.OpenProcessToken.argtypes = [_handle, wintypes.DWORD, ctypes.POINTER(_handle)]
    advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    advapi32.CreateRestrictedToken.argtypes = [
        _handle, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_SidAndAttributes), wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(_handle)]
    advapi32.SetTokenInformation.argtypes = [_handle, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    advapi32.GetTokenInformation.argtypes = [_handle, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                             ctypes.POINTER(wintypes.DWORD)]
    advapi32.DuplicateToken.argtypes = [_handle, ctypes.c_int, ctypes.POINTER(_handle)]
    advapi32.GetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                                          wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi32.AccessCheck.argtypes = [ctypes.c_void_p, _handle, wintypes.DWORD, ctypes.POINTER(_GenericMapping),
                                     ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.BOOL)]
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    advapi32.GetSecurityDescriptorDacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                                                   ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
    return advapi32, kernel32


def _check(ok: int, what: str) -> None:
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(err, f"{what}: WinError {err} {ctypes.FormatError(err).strip()}")


class _Sid:
    """A SID from its string form, freed on exit."""

    def __init__(self, advapi32, kernel32, text: str) -> None:
        self._kernel32 = kernel32
        self.ptr = ctypes.c_void_p()
        _check(advapi32.ConvertStringSidToSidW(text, ctypes.byref(self.ptr)), f"ConvertStringSidToSidW({text})")

    def __enter__(self) -> _Sid:
        return self

    def __exit__(self, *exc) -> None:
        self._kernel32.LocalFree(self.ptr)


def current_user_sid() -> str:
    """String SID of this process's token user (the account a standard-user token keeps)."""
    advapi32, kernel32 = _dlls()
    token = _handle()
    _check(advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)), "OpenProcessToken")
    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))  # TokenUser
        buf = ctypes.create_string_buffer(needed.value)
        _check(advapi32.GetTokenInformation(token, 1, buf, needed, ctypes.byref(needed)), "GetTokenInformation")
        sid = ctypes.cast(buf, ctypes.POINTER(_SidAndAttributes)).contents.Sid
        text = wintypes.LPWSTR()
        _check(advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)), "ConvertSidToStringSidW")
        try:
            return str(text.value)
        finally:
            kernel32.LocalFree(text)
    finally:
        kernel32.CloseHandle(token)


def standard_user_token() -> _handle:
    """A primary token for this user with the UAC filtered-token shape. The caller closes it."""
    advapi32, kernel32 = _dlls()
    own = _handle()
    # TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ADJUST_DEFAULT
    _check(advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0001 | 0x0002 | 0x0008 | 0x0080,
                                     ctypes.byref(own)), "OpenProcessToken")
    out = _handle()
    try:
        with _Sid(advapi32, kernel32, _ADMINISTRATORS) as admins:
            disable = (_SidAndAttributes * 1)(_SidAndAttributes(admins.ptr, 0))
            # DISABLE_MAX_PRIVILEGE (0x1) | LUA_TOKEN (0x4)
            _check(advapi32.CreateRestrictedToken(own, 0x1 | 0x4, 1, disable, 0, None, 0, None, ctypes.byref(out)),
                   "CreateRestrictedToken")
    finally:
        kernel32.CloseHandle(own)
    try:
        with _Sid(advapi32, kernel32, _MEDIUM_IL) as medium:
            label = _SidAndAttributes(medium.ptr, 0x20)  # SE_GROUP_INTEGRITY
            size = ctypes.sizeof(label) + advapi32.GetLengthSid(medium.ptr)
            _check(advapi32.SetTokenInformation(out, 25, ctypes.byref(label), size),  # TokenIntegrityLevel
                   "SetTokenInformation(TokenIntegrityLevel)")
        # An elevated admin token's default owner is Administrators, now deny-only: new objects must
        # be owned by the user, as a filtered token's are.
        user_sid = current_user_sid()
        with _Sid(advapi32, kernel32, user_sid) as user:
            owner = ctypes.c_void_p(user.ptr.value)
            _check(advapi32.SetTokenInformation(out, 4, ctypes.byref(owner), ctypes.sizeof(owner)),  # TokenOwner
                   "SetTokenInformation(TokenOwner)")
        # Likewise its default DACL (Administrators + SYSTEM): the processes, threads and unnamed
        # objects this token creates would be unopenable by the user's other processes. A filtered
        # token's default DACL grants the user and SYSTEM.
        sd = ctypes.c_void_p()
        _check(advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"D:(A;;GA;;;{user_sid})(A;;GA;;;SY)", 1, ctypes.byref(sd), None),
            "ConvertStringSecurityDescriptorToSecurityDescriptorW")
        try:
            present, dacl, defaulted = wintypes.BOOL(), ctypes.c_void_p(), wintypes.BOOL()
            _check(advapi32.GetSecurityDescriptorDacl(sd, ctypes.byref(present), ctypes.byref(dacl),
                                                      ctypes.byref(defaulted)), "GetSecurityDescriptorDacl")
            _check(advapi32.SetTokenInformation(out, 6, ctypes.byref(dacl), ctypes.sizeof(dacl)),  # TokenDefaultDacl
                   "SetTokenInformation(TokenDefaultDacl)")
        finally:
            kernel32.LocalFree(sd)
    except BaseException:
        kernel32.CloseHandle(out)
        raise
    return out


def standard_user_access(paths: list[Path], access: int = FILE_READ_EXECUTE) -> dict[Path, str]:
    """``{path: ""}`` when a standard-user token is granted ``access`` on ``path``, else why not.

    The kernel's own decision (AccessCheck) against :func:`standard_user_token`.
    """
    advapi32, kernel32 = _dlls()
    primary = standard_user_token()
    imp = _handle()
    try:
        _check(advapi32.DuplicateToken(primary, 2, ctypes.byref(imp)), "DuplicateToken")  # SecurityImpersonation
        mapping = _GenericMapping(0x120089, 0x120116, 0x1200A0, 0x1F01FF)
        verdicts: dict[Path, str] = {}
        for path in paths:
            needed = wintypes.DWORD()
            info = 0x1 | 0x2 | 0x4 | 0x10  # OWNER | GROUP | DACL | LABEL
            advapi32.GetFileSecurityW(str(path), info, None, 0, ctypes.byref(needed))
            sd = ctypes.create_string_buffer(max(needed.value, 1))
            try:
                _check(advapi32.GetFileSecurityW(str(path), info, sd, needed, ctypes.byref(needed)),
                       "GetFileSecurityW")
                privs, privs_len = ctypes.create_string_buffer(1024), wintypes.DWORD(1024)
                granted, status = wintypes.DWORD(), wintypes.BOOL()
                _check(advapi32.AccessCheck(sd, imp, access, ctypes.byref(mapping), privs,
                                            ctypes.byref(privs_len), ctypes.byref(granted), ctypes.byref(status)),
                       "AccessCheck")
                verdicts[path] = "" if status.value else f"access 0x{access:x} denied to a standard-user token"
            except OSError as exc:
                verdicts[path] = str(exc)
        return verdicts
    finally:
        for h in (imp, primary):
            if h:
                kernel32.CloseHandle(h)


def harness_control(scratch: Path) -> str:
    """The token reads System32 but not an Administrators+SYSTEM-only file: it really is restricted."""
    admin_only = scratch / "admin-only.txt"
    admin_only.write_text("x", encoding="utf-8")
    subprocess.run(["icacls", str(admin_only), "/inheritance:r", "/grant:r", f"*{_ADMINISTRATORS}:F",
                    f"*{_SYSTEM}:F"], capture_output=True, check=True, timeout=60)
    cmd = Path(r"C:\Windows\System32\cmd.exe")
    verdicts = standard_user_access([cmd, admin_only])
    assert verdicts[cmd] == "", f"harness: the standard-user token cannot even run cmd.exe: {verdicts[cmd]}"
    assert verdicts[admin_only], "harness: the standard-user token still reads an Administrators-only file"
    return "ok"


def grant_user_profile_acl(profile: Path) -> None:
    """Give a fake profile the DACL Windows gives a real one: the user, SYSTEM and Administrators
    full control, inherited by everything below.

    Without it a profile the elevated harness made under ``C:\\Users`` inherits only
    ``Users:(RX)`` plus CREATOR OWNER (= Administrators for an elevated creator), so no
    non-elevated process of the same user could write its own home: a harness artefact, never a
    user's machine."""
    profile.mkdir(parents=True, exist_ok=True)
    aces = [f"*{sid}:(OI)(CI)F" for sid in (current_user_sid(), _SYSTEM, _ADMINISTRATORS)]
    subprocess.run(["icacls", str(profile), "/inheritance:r", "/grant:r", *aces, "/Q"],
                   capture_output=True, check=True, timeout=300)
    # What the harness already staged below (the seeded home) inherits it, as a real profile's does.
    if any(profile.iterdir()):
        subprocess.run(["icacls", str(profile / "*"), "/reset", "/T", "/C", "/Q"],
                       capture_output=True, check=True, timeout=300)


def hand_to_user(path: Path) -> None:
    """Make a harness dir (serve.git, the gitconfig, install.ps1) the user's own, as a user's clone
    or download would be: owner = the user (git refuses a repo it judges another owner's) plus an
    inherited full-control grant."""
    sid = current_user_sid()
    for args in (["/setowner", f"*{sid}"], ["/grant", f"*{sid}:(OI)(CI)F"]):
        subprocess.run(["icacls", str(path), *args, "/T", "/C", "/Q"], capture_output=True, check=True, timeout=300)


# -- starting a process under the token -------------------------------------------------------


class _StartupInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", _handle), ("hStdOutput", _handle),
                ("hStdError", _handle)]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", _handle), ("hThread", _handle), ("dwProcessId", wintypes.DWORD),
                ("dwThreadId", wintypes.DWORD)]


_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_STARTF_USESTDHANDLES = 0x00000100
_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_INFINITE = 0xFFFFFFFF
_STILL_ACTIVE = 259


class StandardUserProcess:
    """A ``subprocess.Popen``-shaped handle on a process started under :func:`standard_user_token`.

    stdout and stderr go to ``log`` (an inheritable handle the elevated parent opened, so the
    child needs no access to the transcript dir); stdin is NUL. Only those two handles are
    inherited (PROC_THREAD_ATTRIBUTE_HANDLE_LIST), never the pytest process's pipes.
    """

    def __init__(self, argv: list[str], *, cwd: Path, env: dict[str, str], log: Path,
                 creationflags: int = 0, append: bool = False) -> None:
        import msvcrt

        advapi32, kernel32 = _dlls()
        self._kernel32 = kernel32
        kernel32.InitializeProcThreadAttributeList.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                                               ctypes.POINTER(ctypes.c_size_t)]
        kernel32.UpdateProcThreadAttribute.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t,
                                                       ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                                       ctypes.c_void_p]
        kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
        advapi32.CreateProcessAsUserW.argtypes = [
            _handle, wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL,
            wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p,
            ctypes.POINTER(_ProcessInformation)]
        kernel32.WaitForSingleObject.argtypes = [_handle, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [_handle, ctypes.POINTER(wintypes.DWORD)]

        self.args = argv
        self.returncode: int | None = None
        out = open(log, "ab" if append else "wb")
        nul = open(os.devnull, "rb")
        token = standard_user_token()
        attrs = None
        try:
            out_h = msvcrt.get_osfhandle(out.fileno())
            in_h = msvcrt.get_osfhandle(nul.fileno())
            os.set_handle_inheritable(out_h, True)
            os.set_handle_inheritable(in_h, True)
            handles = (_handle * 2)(out_h, in_h)
            size = ctypes.c_size_t()
            kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
            attrs = ctypes.create_string_buffer(size.value)
            _check(kernel32.InitializeProcThreadAttributeList(attrs, 1, 0, ctypes.byref(size)),
                   "InitializeProcThreadAttributeList")
            _check(kernel32.UpdateProcThreadAttribute(attrs, 0, _PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles,
                                                      ctypes.sizeof(handles), None, None),
                   "UpdateProcThreadAttribute")
            si = _StartupInfoEx()
            si.StartupInfo.cb = ctypes.sizeof(_StartupInfoEx)
            si.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
            si.StartupInfo.hStdInput = in_h
            si.StartupInfo.hStdOutput = out_h
            si.StartupInfo.hStdError = out_h
            si.lpAttributeList = ctypes.cast(attrs, ctypes.c_void_p)
            block = "".join(f"{k}={v}\0" for k, v in sorted(env.items(), key=lambda kv: kv[0].upper())) + "\0"
            env_buf = ctypes.create_unicode_buffer(block, len(block))
            cmdline = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
            pi = _ProcessInformation()
            flags = creationflags | _EXTENDED_STARTUPINFO_PRESENT | _CREATE_UNICODE_ENVIRONMENT
            _check(advapi32.CreateProcessAsUserW(token, None, cmdline, None, None, True, flags, env_buf,
                                                 str(cwd), ctypes.byref(si), ctypes.byref(pi)),
                   f"CreateProcessAsUserW({argv[0]})")
            kernel32.CloseHandle(pi.hThread)
            self._process = pi.hProcess
            self.pid = int(pi.dwProcessId)
        finally:
            if attrs is not None:
                kernel32.DeleteProcThreadAttributeList(attrs)
            kernel32.CloseHandle(token)
            out.close()
            nul.close()

    def poll(self) -> int | None:
        if self.returncode is None and self._kernel32.WaitForSingleObject(self._process, 0) == 0:
            code = wintypes.DWORD()
            self._kernel32.GetExitCodeProcess(self._process, ctypes.byref(code))
            self.returncode = int(code.value)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        ms = _INFINITE if timeout is None else max(0, int(timeout * 1000))
        if self._kernel32.WaitForSingleObject(self._process, ms) != 0:
            raise subprocess.TimeoutExpired(self.args, timeout or 0)
        return self.poll()  # type: ignore[return-value]

    def __del__(self) -> None:
        try:
            self._kernel32.CloseHandle(self._process)
        except Exception:
            pass


def probe_token_elevation(cwd: Path, env: dict[str, str], log: Path) -> str:
    """What a process under the token says about itself (``whoami /groups``), as
    ``elevated=<bool> admin=<bool>``: elevated = High/System integrity, admin = Administrators
    enabled (not deny-only).

    The harness control for every non-elevated cell: a child that is still elevated (or still
    holds an enabled Administrators group) would make the cell prove nothing."""
    import csv
    import io

    whoami = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "whoami.exe"
    proc = StandardUserProcess([str(whoami), "/groups", "/fo", "csv", "/nh"], cwd=cwd, env=env, log=log)
    rc = proc.wait(120)
    raw = log.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    assert rc == 0, f"harness: whoami under the standard-user token exited {rc}: {text}"
    rows = list(csv.reader(io.StringIO(text)))
    sids = {row[2]: row[3] for row in rows if len(row) >= 4}
    elevated = "S-1-16-12288" in sids or "S-1-16-16384" in sids
    admin = _ADMINISTRATORS in sids and "deny only" not in sids[_ADMINISTRATORS].lower()
    verdict = f"elevated={elevated} admin={admin}"
    assert "S-1-16-8192" in sids, f"harness: the standard-user token is not Medium integrity: {text}"
    assert verdict == "elevated=False admin=False", f"harness: the standard-user token is not standard: {text}"
    return verdict
