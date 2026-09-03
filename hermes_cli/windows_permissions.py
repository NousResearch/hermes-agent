"""Owner-only filesystem permissions for native Windows state files.

POSIX ``chmod(0600/0700)`` does not express a Windows trust boundary: CPython
only toggles the read-only attribute and ``stat().st_mode`` continues to report
the inherited ACL.  Hermes stores credentials and scheduler state below its
home, so the Windows equivalent must be an explicit, protected DACL.

The policy intentionally mirrors ``windows_ssh_runtime``: the current user and
LocalSystem receive full control; inherited and unrelated allow ACEs are
removed.  Directories propagate the same policy to new children.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _win32_modules():
    if sys.platform != "win32":
        raise RuntimeError("Windows permissions are only available on Windows")

    import ntsecuritycon
    import win32api
    import win32con
    import win32security

    return ntsecuritycon, win32api, win32con, win32security


def _current_sid():
    _, win32api, win32con, win32security = _win32_modules()
    token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
    )
    return win32security.GetTokenInformation(token, win32security.TokenUser)[0]


def _system_sid():
    return _win32_modules()[3].ConvertStringSidToSid("S-1-5-18")


def restrict_path_to_current_user(path: Path | str) -> None:
    """Replace ``path``'s DACL with current-user + LocalSystem full control.

    This function is deliberately strict and raises on failure.  Callers that
    historically treated permission tightening as best-effort may catch and
    log the error, while tests and security-sensitive creation paths can verify
    the boundary directly.
    """

    target = Path(path)
    ntsecuritycon, _, _, win32security = _win32_modules()
    owner = _current_sid()
    acl = win32security.ACL()
    inheritance = 0
    if target.is_dir():
        inheritance = (
            win32security.OBJECT_INHERIT_ACE
            | win32security.CONTAINER_INHERIT_ACE
        )
    for sid in (owner, _system_sid()):
        acl.AddAccessAllowedAceEx(
            win32security.ACL_REVISION,
            inheritance,
            ntsecuritycon.FILE_ALL_ACCESS,
            sid,
        )

    security_info = (
        win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION
    )
    win32security.SetNamedSecurityInfo(
        str(target),
        win32security.SE_FILE_OBJECT,
        security_info,
        None,
        None,
        acl,
        None,
    )


def path_is_restricted_to_current_user(path: Path | str) -> bool:
    """Return whether ``path`` has the protected Hermes Windows DACL."""

    _, _, _, win32security = _win32_modules()
    descriptor = win32security.GetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION,
    )
    control, _revision = descriptor.GetSecurityDescriptorControl()
    if not control & win32security.SE_DACL_PROTECTED:
        return False
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None:
        return False

    allowed = {
        win32security.ConvertSidToStringSid(_current_sid()),
        win32security.ConvertSidToStringSid(_system_sid()),
    }
    allow_types = {
        win32security.ACCESS_ALLOWED_ACE_TYPE,
        win32security.ACCESS_ALLOWED_OBJECT_ACE_TYPE,
        getattr(win32security, "ACCESS_ALLOWED_CALLBACK_ACE_TYPE", 9),
        getattr(win32security, "ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE", 11),
    }
    observed_allowed: set[str] = set()
    for index in range(dacl.GetAceCount()):
        ace = dacl.GetAce(index)
        ace_type = ace[0][0]
        mask = ace[1]
        sid = win32security.ConvertSidToStringSid(ace[-1])
        if ace_type in allow_types and mask:
            if sid not in allowed:
                return False
            observed_allowed.add(sid)
    return observed_allowed == allowed
