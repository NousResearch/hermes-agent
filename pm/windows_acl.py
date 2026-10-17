"""Windows ACL guards for the managed tool store (``%LOCALAPPDATA%\\hermes\\tools``).

An elevated ``hermes update``/installer publishes tool entries with the elevated
token, and some of those entries land with a DACL that grants the ordinary user
nothing at all (#126860). The store is then poisoned: every non-elevated
``hermes update`` dies with a raw ``[WinError 5]`` on the verify/replace rename,
and the affected tools are unusable at runtime for that user. Nothing on the
product path detected or repaired the state, so it persisted forever.

Two defenses, both no-ops off Windows and cheap no-ops for healthy stores:

* :func:`ensure_user_readable` — publish-time probe. An elevated publisher
  checks its own output the way the OS would for a standard-user token
  (``AccessCheck`` against a ``SAFER_LEVELID_NORMALUSER`` token — the same
  probe ``tests/e2e/core/windows_update/test_paths_and_acls.py`` asserts with)
  and repairs the tree's ACL before the entry is recorded as installed.
* :func:`wrap_access_denied` — translates an ``errno 5`` ``OSError`` from a
  publish/verify/replace step into an :class:`InstallError` that names the
  entry, the likely cause (a previous elevated session published it
  admin-owned) and the exact recovery, instead of surfacing a bare WinError 5.

The repair restores inheritance from the store root first (``icacls /reset``),
which yields the profile-inherited user ACEs a healthy store has; if the probe
still denies a standard-user token it falls back to granting the invoking
account (whose SID an elevated process still knows from its own token) full
control over the tree.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

LOG = logging.getLogger(__name__)

_ICACLS_TIMEOUT = 180


def _is_windows() -> bool:
    return os.name == "nt"


def is_elevated() -> bool:
    """Whether this process runs with an elevated (Administrator) token."""
    if not _is_windows():
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        LOG.debug("windows_acl: elevation probe failed", exc_info=True)
        return False


def standard_user_denied(path: Path) -> bool:
    """True when a standard-user token may NOT read+traverse *path*.

    The kernel's own decision (``AccessCheck``) for the
    ``SAFER_LEVELID_NORMALUSER`` token a logon task gets on a UAC machine.
    An unreadable security descriptor counts as denied — a standard user
    could not read it either. Probe infrastructure failures return False so
    a broken probe never prompts an ACL rewrite.
    """
    if not _is_windows():
        return False
    try:
        return _access_check_denied(Path(path))
    except Exception:
        LOG.debug("windows_acl: probe failed on %s", path, exc_info=True)
        return False


def _access_check_denied(path: Path) -> bool:
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class GENERIC_MAPPING(ctypes.Structure):
        _fields_ = [(name, wintypes.DWORD) for name in
                    ("GenericRead", "GenericWrite", "GenericExecute", "GenericAll")]

    handle = wintypes.HANDLE
    advapi32.SaferCreateLevel.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                          ctypes.POINTER(handle), ctypes.c_void_p]
    advapi32.SaferComputeTokenFromLevel.argtypes = [handle, handle, ctypes.POINTER(handle),
                                                    wintypes.DWORD, ctypes.c_void_p]
    advapi32.SaferCloseLevel.argtypes = [handle]
    advapi32.DuplicateToken.argtypes = [handle, ctypes.c_int, ctypes.POINTER(handle)]
    advapi32.GetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                                          wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi32.AccessCheck.argtypes = [ctypes.c_void_p, handle, wintypes.DWORD,
                                     ctypes.POINTER(GENERIC_MAPPING), ctypes.c_void_p,
                                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                     ctypes.POINTER(wintypes.BOOL)]
    kernel32.CloseHandle.argtypes = [handle]

    def check(ok: int, what: str) -> None:
        if not ok:
            raise OSError(ctypes.get_last_error(), f"{what} failed")

    level, primary, imp = handle(), handle(), handle()
    # SAFER_SCOPEID_USER=2, SAFER_LEVELID_NORMALUSER=0x20000, SAFER_LEVEL_OPEN=1
    check(advapi32.SaferCreateLevel(2, 0x20000, 1, ctypes.byref(level), None), "SaferCreateLevel")
    try:
        check(advapi32.SaferComputeTokenFromLevel(level, None, ctypes.byref(primary), 0, None),
              "SaferComputeTokenFromLevel")
    finally:
        advapi32.SaferCloseLevel(level)
    try:
        check(advapi32.DuplicateToken(primary, 2, ctypes.byref(imp)), "DuplicateToken")  # SecurityImpersonation
        mapping = GENERIC_MAPPING(0x120089, 0x120116, 0x1200A0, 0x1F01FF)
        needed = wintypes.DWORD()
        # OWNER | GROUP | DACL
        if not advapi32.GetFileSecurityW(str(path), 0x7, None, 0, ctypes.byref(needed)):
            err = ctypes.get_last_error()
            if err == 5:  # the SD itself is hidden from ordinary callers: denied
                return True
            raise OSError(err, "GetFileSecurityW (size) failed")
        sd = ctypes.create_string_buffer(max(needed.value, 1))
        check(advapi32.GetFileSecurityW(str(path), 0x7, sd, needed, ctypes.byref(needed)),
              "GetFileSecurityW")
        privs, privs_len = ctypes.create_string_buffer(1024), wintypes.DWORD(1024)
        granted, status = wintypes.DWORD(), wintypes.BOOL()
        check(advapi32.AccessCheck(sd, imp, 0x1200A9, ctypes.byref(mapping), privs,
                                   ctypes.byref(privs_len), ctypes.byref(granted),
                                   ctypes.byref(status)),  # FILE_GENERIC_READ | FILE_GENERIC_EXECUTE
              "AccessCheck")
        return not status.value
    finally:
        for h in (imp, primary):
            if h:
                kernel32.CloseHandle(h)


def _account_sid() -> str | None:
    """The invoking account's SID — an elevated process still runs AS the user."""
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
        advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                 wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.HANDLE)]
        TOKEN_QUERY = 0x8
        TokenUser = 1
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_QUERY,
                                         ctypes.byref(token)):
            return None
        try:
            needed = wintypes.DWORD()
            advapi32.GetTokenInformation(token, TokenUser, None, 0, ctypes.byref(needed))
            buffer = ctypes.create_string_buffer(needed.value)
            if not advapi32.GetTokenInformation(token, TokenUser, buffer, needed.value,
                                                ctypes.byref(needed)):
                return None
            sid = ctypes.cast(buffer, ctypes.c_void_p).value  # SID_AND_ATTRIBUTES: SID first
            string = wintypes.LPWSTR()
            if not kernel32.ConvertSidToStringSidW(sid, ctypes.byref(string)):
                return None
            return string.value
        finally:
            kernel32.CloseHandle(token)
    except Exception:
        LOG.debug("windows_acl: account SID lookup failed", exc_info=True)
        return None


def _icacls(args: list[str]) -> bool:
    try:
        proc = subprocess.run(["icacls", *args], capture_output=True, text=True,
                              timeout=_ICACLS_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        LOG.debug("windows_acl: icacls %s failed to run", args[:1], exc_info=True)
        return False
    if proc.returncode != 0:
        LOG.warning("windows_acl: icacls %s exited %s: %s",
                    args[:1], proc.returncode, (proc.stderr or proc.stdout or "").strip()[:300])
    return proc.returncode == 0


def repair_user_acl(root: Path) -> bool:
    """Make *root* (recursively) readable+traversable by a standard-user token.

    Restores inheritance from the parent first — the healthy shape — then, if
    the probe still denies access, grants the invoking account explicit full
    control (an elevated process runs as that account, so its own token still
    carries the SID). Returns True when the final probe passes.
    """
    root = Path(root)
    _icacls([str(root), "/reset", "/t"])
    if not standard_user_denied(root):
        return True
    sid = _account_sid()
    if sid:
        _icacls([str(root), "/grant", f"*{sid}:(OI)(CI)F", "/t"])
    return not standard_user_denied(root)


def ensure_user_readable(package_name: str, entry: Path, store_root: Path) -> None:
    """Publish-time guard: an elevated publisher must not ship a user-unreadable entry.

    Probes the freshly published *entry* (and the store root it inherits from);
    when a standard-user token is denied and this process is elevated — i.e.
    this very session created the poisoned ACL — the tree is repaired. A still
    denied entry raises so the install fails loudly instead of poisoning the
    store silently (#126860). Healthy non-elevated stores pay one probe.
    """
    entry = Path(entry)
    store_root = Path(store_root)
    if not _is_windows() or not is_elevated():
        return
    poisoned = [p for p in (entry, store_root) if standard_user_denied(p)]
    if not poisoned:
        return
    LOG.warning("windows_acl: elevated session published a user-unreadable tree; repairing %s",
                [str(p) for p in poisoned])
    from pm.package import InstallError

    for path in poisoned:
        if not repair_user_acl(path):
            raise InstallError(
                package_name,
                f"published {path} is not accessible to a standard user and the ACL repair failed",
                f'from an elevated console run: icacls "{path}" /reset /t — then retry',
            )


def is_access_denied(err: OSError) -> bool:
    """Whether *err* is ERROR_ACCESS_DENIED (WinError 5 / EACCES)."""
    return getattr(err, "winerror", None) == 5 or err.errno == 13


def access_denied_error(package_name: str, doing: str, path: Path | str, err: OSError):
    """Translate an ERROR_ACCESS_DENIED *err* into an actionable InstallError.

    The verify/replace steps surface bare ``[WinError 5]`` when the entry on
    disk was published admin-owned by an earlier elevated session; name the
    entry and the recovery instead (#126860)."""
    from pm.package import InstallError

    return InstallError(
        package_name,
        f"access denied while {doing} {path} ({err}) — a previous elevated session may have "
        "published this entry admin-owned",
        'from an elevated console run: icacls "<store entry>" /reset /t (or delete the entry), '
        "then retry; an elevated `hermes update` now repairs such entries automatically",
    )


def wrap_access_denied(package_name: str, doing: str, path: Path | str, err: OSError):
    """Re-raise *err* as an actionable InstallError when it is ERROR_ACCESS_DENIED.

    Non-access-denied errors pass through untouched."""
    if not is_access_denied(err):
        raise err
    return access_denied_error(package_name, doing, path, err)
