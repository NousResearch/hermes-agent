"""A Windows user profile path with non-ASCII characters and a space, and tool ACLs.

Failure class: paths and ACLs.

* ``C:\\Users\\Jörg Ñúñez`` is an ordinary Windows account. ``install.ps1`` must get
  through the uv ``python-deps`` stage there (#124526), and ``hermes`` must then work and
  update from that profile.
* After ``hermes update`` the managed toolchain under ``%LOCALAPPDATA%\\hermes\\tools``
  must still be executable by a non-elevated process: a logon Scheduled Task or Startup
  entry runs with a standard-user token even when the updating session was elevated
  (#122935). The runner session is elevated, so each tool's DACL is checked (kernel
  AccessCheck) against a Basic User (SAFER_LEVELID_NORMALUSER) token, the token
  ``runas /trustlevel:0x20000`` uses.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    Journey,
    fail_with,
    new_machine,
    one_shot_turn,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

PERSON = "Jörg Ñúñez"  # the profile is "Jörg Ñúñez hermes-e2e-<id>"
_TOOL_NAMES = {"python.exe", "node.exe", "git.exe", "uv.exe", "rg.exe"}


_FILE_READ_EXECUTE = 0x1200A9  # FILE_GENERIC_READ | FILE_GENERIC_EXECUTE


def standard_user_access(paths: list[Path]) -> dict[Path, str]:
    """``{path: ""}`` when a Basic User token may read+execute ``path``, else why not.

    The kernel's own decision (AccessCheck) for the SAFER_LEVELID_NORMALUSER token that
    ``runas /trustlevel:0x20000`` uses: what a logon-time task gets on a UAC machine.
    """
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class GENERIC_MAPPING(ctypes.Structure):
        _fields_ = [("GenericRead", wintypes.DWORD), ("GenericWrite", wintypes.DWORD),
                    ("GenericExecute", wintypes.DWORD), ("GenericAll", wintypes.DWORD)]

    handle = wintypes.HANDLE
    advapi32.SaferCreateLevel.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                          ctypes.POINTER(handle), ctypes.c_void_p]
    advapi32.SaferComputeTokenFromLevel.argtypes = [handle, handle, ctypes.POINTER(handle),
                                                    wintypes.DWORD, ctypes.c_void_p]
    advapi32.SaferCloseLevel.argtypes = [handle]
    advapi32.DuplicateToken.argtypes = [handle, ctypes.c_int, ctypes.POINTER(handle)]
    advapi32.GetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                                          wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi32.AccessCheck.argtypes = [ctypes.c_void_p, handle, wintypes.DWORD, ctypes.POINTER(GENERIC_MAPPING),
                                     ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.BOOL)]
    kernel32.CloseHandle.argtypes = [handle]

    def check(ok: int, what: str) -> None:
        if not ok:
            err = ctypes.get_last_error()
            raise OSError(err, f"{what}: WinError {err} {ctypes.FormatError(err).strip()}")

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
        verdicts: dict[Path, str] = {}
        for path in paths:
            needed = wintypes.DWORD()
            info = 0x1 | 0x2 | 0x4  # OWNER | GROUP | DACL
            advapi32.GetFileSecurityW(str(path), info, None, 0, ctypes.byref(needed))
            sd = ctypes.create_string_buffer(max(needed.value, 1))
            try:
                check(advapi32.GetFileSecurityW(str(path), info, sd, needed, ctypes.byref(needed)),
                      "GetFileSecurityW")
                privs, privs_len = ctypes.create_string_buffer(1024), wintypes.DWORD(1024)
                granted, status = wintypes.DWORD(), wintypes.BOOL()
                check(advapi32.AccessCheck(sd, imp, _FILE_READ_EXECUTE, ctypes.byref(mapping), privs,
                                           ctypes.byref(privs_len), ctypes.byref(granted), ctypes.byref(status)),
                      "AccessCheck")
                verdicts[path] = "" if status.value else "access denied to a standard-user token"
            except OSError as exc:
                verdicts[path] = str(exc)
        return verdicts
    finally:
        for h in (imp, primary):
            if h:
                kernel32.CloseHandle(h)


def _standard_user_token_is_really_restricted(scratch: Path) -> str:
    """Harness control: the token reads System32 but not an Administrators+SYSTEM-only file."""
    admin_only = scratch / "admin-only.txt"
    admin_only.write_text("x", encoding="utf-8")
    subprocess.run(["icacls", str(admin_only), "/inheritance:r", "/grant:r", "*S-1-5-32-544:F", "*S-1-5-18:F"],
                   capture_output=True, check=True, timeout=60)
    cmd = Path(r"C:\Windows\System32\cmd.exe")
    verdicts = standard_user_access([cmd, admin_only])
    assert verdicts[cmd] == "", f"harness: the standard-user token cannot even run cmd.exe: {verdicts[cmd]}"
    assert verdicts[admin_only], "harness: the standard-user token still reads an Administrators-only file"
    return "ok"


def _managed_tools(hermes_home: Path) -> list[Path]:
    tools = hermes_home / "tools"
    found = []
    for pattern in ("*/*.exe", "*/*/*.exe", "*/*/*/*.exe"):
        found += [p for p in tools.glob(pattern) if p.name.lower() in _TOOL_NAMES]
    return sorted(set(found))


def _tool_access(hermes_home: Path) -> dict[Path, str]:
    return standard_user_access(_managed_tools(hermes_home))


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("paths"), srv.base_url, label="paths",
                              person=PERSON, system_git=True)
        j = Journey(machine)
        try:
            install = j.step("install", machine.install)
            if j.ok("install") and install.returncode == 0:
                j.step("version", lambda: machine.hermes("--version"))
                j.step("turn", lambda: one_shot_turn(machine, srv, "turn-unicode-profile"))
                machine.advance()
                j.step("update", machine.update)
                j.step("control", lambda: _standard_user_token_is_really_restricted(machine.root))
                j.step("tool_access", lambda: _tool_access(machine.hermes_home))
            yield j
        finally:
            machine.teardown()


def test_install_from_non_ascii_profile_with_spaces(journey: Journey) -> None:
    m, run = journey.machine, journey["install"]
    first_error = next((ln.strip() for ln in run.stdout.splitlines() if "[X]" in ln or "✗" in ln), "<none>")
    assert run.returncode == 0, fail_with(
        m, f"install.ps1 failed for a profile path with non-ASCII characters and spaces: {first_error}", run)


def test_hermes_works_and_updates_from_that_profile(journey: Journey) -> None:
    m = journey.machine
    version, turn, update = journey["version"], journey["turn"], journey["update"]
    assert version.returncode == 0 and "Aro Agent v" in version.stdout, fail_with(
        m, "hermes --version fails from a non-ASCII profile", version)
    assert turn.ok, fail_with(
        m, f"a turn fails from a non-ASCII profile (reply printed={turn.reply_id in turn.run.stdout}, "
           f"prompt reached provider={turn.reached_wire})", turn.run)
    assert update.returncode == 0 and m.installed_head() == m.next, fail_with(
        m, f"hermes update from a non-ASCII profile exited {update.returncode}, checkout at {m.installed_head()}",
        update)


def test_managed_tools_stay_executable_for_standard_user(journey: Journey) -> None:
    m, verdicts = journey.machine, journey["tool_access"]
    journey["control"]
    journey["update"]
    assert any(exe.name.lower() == "python.exe" for exe in verdicts), fail_with(
        m, f"no managed python.exe under {m.hermes_home / 'tools'}: {[str(e) for e in verdicts]}")
    broken = [f"{exe.relative_to(m.hermes_home)}: {why}" for exe, why in verdicts.items() if why]
    assert not broken, fail_with(
        m, f"managed tools are not executable by a non-elevated process after update: {'; '.join(broken)}")
