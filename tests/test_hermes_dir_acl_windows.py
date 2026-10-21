"""#139845 / #134889 W8 + W2 on real Windows ACLs.

The reporters' dirs were created by an elevated process: a DACL of SYSTEM + Administrators only, no
entry for the user. Their own non-elevated token holds Administrators as deny-only, so it gets
``WinError 5`` inside. Each probe runs in a child that builds that filtered token itself
(``CreateRestrictedToken``: Administrators deny-only, every privilege but SeChangeNotify removed, after
SeBackup/SeRestore are removed from the process token outright), impersonates it, and then touches the
dir the way production does. Each test first proves that child is refused where the reporters were
and skips with the runner's identity when it is not (the arm64 hosted runner once let it in).

The parent needs an elevated token (the hosted runner has one) to stand in for the elevated
install; ``make_user_dir`` only grants when the creating process is elevated.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

_REPO = Path(__file__).resolve().parent.parent

_DROP_PRIVILEGES = textwrap.dedent(r"""
    import ctypes, ctypes.wintypes as w, json, os, sys
    from pathlib import Path

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", w.DWORD), ("HighPart", w.LONG)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", w.DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", w.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    kernel.GetCurrentProcess.restype = w.HANDLE
    advapi.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
    advapi.LookupPrivilegeValueW.argtypes = [w.LPCWSTR, w.LPCWSTR, ctypes.POINTER(LUID)]
    advapi.AdjustTokenPrivileges.argtypes = [w.HANDLE, w.BOOL, ctypes.POINTER(TOKEN_PRIVILEGES),
                                             w.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    token = w.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0020 | 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    for name in ("SeBackupPrivilege", "SeRestorePrivilege"):
        luid = LUID()
        if not advapi.LookupPrivilegeValueW(None, name, ctypes.byref(luid)):
            raise ctypes.WinError(ctypes.get_last_error())
        tp = TOKEN_PRIVILEGES(1, (LUID_AND_ATTRIBUTES * 1)(LUID_AND_ATTRIBUTES(luid, 0x00000004)))
        if not advapi.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None):
            raise ctypes.WinError(ctypes.get_last_error())
""")

# SeBackup/SeRestore are REMOVED from the process token first (as in
# tests/tools/test_bot_live_delivery_windows_acl.py): CPython opens with FILE_FLAG_BACKUP_SEMANTICS, so an
# elevated runner token that kept either would bypass every ACE even under the restricted token.
_FILTERED_TOKEN = _DROP_PRIVILEGES + textwrap.dedent(r"""
    import subprocess

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", w.DWORD)]

    kernel.GetCurrentProcess.restype = w.HANDLE
    advapi.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
    advapi.ConvertStringSidToSidW.argtypes = [w.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    advapi.CreateRestrictedToken.argtypes = [w.HANDLE, w.DWORD, w.DWORD, ctypes.POINTER(SID_AND_ATTRIBUTES),
                                             w.DWORD, ctypes.c_void_p, w.DWORD, ctypes.c_void_p,
                                             ctypes.POINTER(w.HANDLE)]
    advapi.ImpersonateLoggedOnUser.argtypes = [w.HANDLE]
    token = w.HANDLE()
    # TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_IMPERSONATE
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0002 | 0x0008 | 0x0001 | 0x0004,
                                   ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    admins = ctypes.c_void_p()
    if not advapi.ConvertStringSidToSidW("S-1-5-32-544", ctypes.byref(admins)):
        raise ctypes.WinError(ctypes.get_last_error())
    disable = (SID_AND_ATTRIBUTES * 1)(SID_AND_ATTRIBUTES(admins, 0))
    filtered = w.HANDLE()
    # DISABLE_MAX_PRIVILEGE: what a UAC-filtered token keeps, i.e. no backup/restore bypass.
    if not advapi.CreateRestrictedToken(token, 0x1, 1, disable, 0, None, 0, None, ctypes.byref(filtered)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not advapi.ImpersonateLoggedOnUser(filtered):
        raise ctypes.WinError(ctypes.get_last_error())

    def usable(directory):
        try:
            fd = os.open(Path(directory) / "probe.lock", os.O_CREAT | os.O_WRONLY, 0o600)
        except PermissionError:
            return False
        os.close(fd)
        return True

    def runner_identity(directory):
        who = subprocess.run(["whoami", "/user", "/groups", "/priv"], capture_output=True, text=True,
                             check=False).stdout
        acl = subprocess.run(["icacls", str(directory)], capture_output=True, text=True, check=False).stdout
        return who + "\n" + acl
""")

_REPORTERS_DACL = ["/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"]


def _elevated() -> bool:
    sys.path.insert(0, str(_REPO))
    from hermes_dir_acl import process_is_elevated
    return process_is_elevated()


def _child(code: str, *args: str, prelude: str = _FILTERED_TOKEN) -> dict:
    child = subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code), *args], cwd=_REPO,
                           capture_output=True, text=True, check=False, timeout=180)
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout.strip().splitlines()[-1])


@pytest.fixture()
def reporters_parent(tmp_path):
    if not _elevated():
        pytest.skip("needs an elevated token to stand in for the elevated install")
    parent = tmp_path / "hermes"
    parent.mkdir()
    subprocess.run(["icacls", str(parent), *_REPORTERS_DACL], check=True, capture_output=True)
    yield parent
    subprocess.run(["icacls", str(parent), "/reset", "/t", "/c", "/q"], check=False, capture_output=True)


def _require_faithful_probe(reporters_parent: Path, reporters_dir: Path) -> None:
    """Skip (never xfail) when this runner cannot stand in for the reporters' machine.

    Two preconditions, each proved by the same filtered child the fix is checked with: it must be
    refused by an Everyone:(DENY)(F) ACE (no privilege bypass left at all), and by the reporters'
    SYSTEM + Administrators DACL on *reporters_dir* (no other identity on the runner's token that
    DACL still grants). Only a failed precondition skips; the fix's own assertion always runs here.
    """
    denied = reporters_parent.parent / "everyone-denied"
    denied.mkdir()
    subprocess.run(["icacls", str(denied), "/deny", "*S-1-1-0:(F)"], check=True, capture_output=True)
    try:
        probe = _child("""
            plain = usable(sys.argv[2])
            print(json.dumps({"denied": usable(sys.argv[1]), "plain": plain,
                              "identity": runner_identity(sys.argv[2]) if plain else ""}))
        """, str(denied), str(reporters_dir))
    finally:
        subprocess.run(["icacls", str(denied), "/remove:d", "*S-1-1-0"], check=False, capture_output=True)
    if probe["denied"]:
        pytest.skip("precondition failed: the filtered child still opens a dir an Everyone:(DENY)(F) ACE "
                    "refuses, so it bypasses ACLs on this runner and cannot judge the fix")
    if probe["plain"]:
        pytest.skip("precondition failed: this runner's filtered token is still granted by the reporters' "
                    "SYSTEM + Administrators DACL, so a dir without the fix is not unusable here:\n"
                    + probe["identity"])


def test_dir_made_while_elevated_stays_usable_by_the_filtered_token(reporters_parent):
    from hermes_dir_acl import make_user_dir

    plain = reporters_parent / "plain"
    plain.mkdir()  # inherits the reporters' SYSTEM + Administrators DACL
    _require_faithful_probe(reporters_parent, plain)
    granted = make_user_dir(reporters_parent / "gateway-locks", 0o700)
    result = _child("""
        print(json.dumps({"plain": usable(sys.argv[1]), "granted": usable(sys.argv[2])}))
    """, str(plain), str(granted))
    assert result["plain"] is False, "the filtered token could use the reporters' DACL: the probe is not faithful"
    assert result["granted"] is True


def test_elevated_rerun_repairs_a_dir_an_earlier_elevated_run_left(reporters_parent):
    from hermes_dir_acl import make_user_dir

    left = reporters_parent / "state"
    left.mkdir()
    _require_faithful_probe(reporters_parent, left)
    make_user_dir(left, parents=True, exist_ok=True)  # how every production caller re-enters
    make_user_dir(left, parents=True, exist_ok=True)  # idempotent: no second ACE
    after = _child("print(json.dumps({'usable': usable(sys.argv[1])}))", str(left))
    acl = subprocess.run(["icacls", str(left)], capture_output=True, text=True, check=True).stdout
    assert after["usable"] is True, acl


def test_elevated_repair_also_reaches_an_old_lock_already_inside_the_dir(reporters_parent):
    """Review #140490: the pywin32 write does not propagate, so a token lock an earlier elevated run
    left INSIDE the dir kept the reporters' DACL after the dir itself was repaired."""
    from hermes_dir_acl import make_user_dir

    left = reporters_parent / "gateway-locks"
    left.mkdir()
    old = left / "telegram-bot-token-x.lock"
    old.write_text("{}")  # inherits SYSTEM + Administrators only
    _require_faithful_probe(reporters_parent, left)
    make_user_dir(left, parents=True, exist_ok=True)
    after = _child("""
        try:
            with open(sys.argv[1], "r+", encoding="utf-8") as handle:
                handle.write("{}")
            ok = True
        except PermissionError:
            ok = False
        print(json.dumps({"old_lock_writable": ok}))
    """, str(old))
    acl = subprocess.run(["icacls", str(old)], capture_output=True, text=True, check=True).stdout
    assert after["old_lock_writable"] is True, acl


def test_elevated_update_step_makes_a_stranded_home_usable(reporters_parent, monkeypatch):
    """alt-glitch on #140490: the owed ``windows_home_acl`` step had unit tests only. Here the real
    step runs from this elevated runner over a home whose nested dirs and files carry the reporters'
    Administrators-only DACL, with real grants; the filtered token must use every one afterwards.
    (The full elevated ``hermes update`` run stays a workflow_dispatch / live retest.)"""
    from hermes_cli import update_windows_acl

    nested = reporters_parent / "sessions" / "2026"
    nested.mkdir(parents=True)
    old = reporters_parent / "gateway-locks"
    old.mkdir()
    (old / "telegram-bot-token-x.lock").write_text("{}")
    _require_faithful_probe(reporters_parent, nested)
    monkeypatch.setenv("HERMES_HOME", str(reporters_parent))
    update_windows_acl.repair_windows_home_acl()
    after = _child("""
        print(json.dumps({"nested": usable(sys.argv[1]), "locks": usable(sys.argv[2]),
                          "home": usable(sys.argv[3])}))
    """, str(nested), str(old), str(reporters_parent))
    assert after == {"nested": True, "locks": True, "home": True}, after


def test_token_lock_on_an_unusable_lock_dir_parks_with_one_named_warning(tmp_path):
    """W2: the gateway meets a ``gateway-locks`` it may not create in.

    Same shape as ``tests/tools/test_bot_live_delivery_windows_acl.py`` (proven on the hosted
    runner): an Everyone DENY ACE that no identity escapes, and a child with SeBackup/SeRestore
    removed so ``os.open`` has no bypass. Needs no elevation."""
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    subprocess.run(["icacls", str(lock_dir), "/deny", "*S-1-1-0:(F)"], check=True, capture_output=True)
    try:
        _w2(lock_dir)
    finally:
        subprocess.run(["icacls", str(lock_dir), "/remove:d", "*S-1-1-0"], check=False, capture_output=True)


def _w2(lock_dir):
    result = _child(r"""
        import logging
        from unittest.mock import MagicMock
        os.environ["HERMES_GATEWAY_LOCK_DIR"] = sys.argv[1]
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append((record.levelname, record.getMessage()))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.DEBUG)
        from gateway.platforms.base import BasePlatformAdapter

        class Stub(BasePlatformAdapter):
            platform = MagicMock(value="telegram")
            async def connect(self, *, is_reconnect=False): return True
            async def disconnect(self): pass
            async def send(self, *a, **k): pass
            async def get_chat_info(self, chat_id): return {}

        codes = []
        for _ in range(3):  # the reconnect watcher re-enters; one WARNING, never a raise
            a = Stub.__new__(Stub)
            a._running, a._fatal_error_handler, a._status_write_logged = True, None, None
            a._fatal_error_code = a._fatal_error_message = None
            a._fatal_error_retryable = True
            a._platform_lock_scope = a._platform_lock_identity = None
            a._platform_lock_takeover_allowed = a._platform_lock_takeover_attempted = False
            a._write_runtime_status_safe = lambda *x, **k: None
            ok = a._acquire_platform_lock("telegram-bot-token", "tok", "Telegram bot token")
            codes.append([ok, a._fatal_error_code, a._fatal_error_retryable, a._fatal_error_message])
        print(json.dumps({"codes": codes, "warnings": [m for lvl, m in records
                                                      if lvl == "WARNING" and "not usable" in m]}))
    """, str(lock_dir), prelude=_DROP_PRIVILEGES)
    for ok, code, retryable, message in result["codes"]:
        assert (ok, code, retryable) == (False, "lock_dir_unusable", True)
        assert "icacls" in message and "/reset" in message and str(lock_dir) in message
    assert len(result["warnings"]) == 1, result


# ── P2-5: an elevated grant never follows a junction out of the Hermes home ──────────────────────


@pytest.fixture()
def home_with_junction_out(tmp_path, monkeypatch):
    """A Hermes home holding a directory junction (what same-user medium-integrity code can plant)
    to a directory outside it, standing in for ``C:\\Program Files\\<app>``."""
    if not _elevated():
        pytest.skip("needs an elevated token: only an elevated run grants")
    import _winapi

    home = tmp_path / "hermes"
    (home / "cache").mkdir(parents=True)
    outside = tmp_path / "Program Files" / "App"
    (outside / "bin").mkdir(parents=True)
    (outside / "bin" / "app.exe").write_text("x")
    subprocess.run(["icacls", str(outside), *_REPORTERS_DACL], check=True, capture_output=True)
    junction = home / "cache" / "images"
    _winapi.CreateJunction(str(outside), str(junction))
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home, outside, junction
    junction.rmdir()  # removes the junction itself, not the target
    subprocess.run(["icacls", str(outside), "/reset", "/t", "/c", "/q"], check=False, capture_output=True)


def _dacls(paths):
    """icacls text of each of *paths*, path-normalized for comparison."""
    return [subprocess.run(["icacls", str(path)], capture_output=True, text=True, check=True).stdout
            .replace(str(path), "<p>") for path in paths]


def test_elevated_grants_never_reach_a_junction_target_outside_the_home(home_with_junction_out):
    import hermes_dir_acl
    from hermes_cli import update_windows_acl

    home, outside, junction = home_with_junction_out
    assert hermes_dir_acl.is_reparse_point(junction)
    watched = [outside, *sorted(outside.rglob("*"))]
    before = _dacls(watched)
    hermes_dir_acl._granted.clear()
    hermes_dir_acl.make_user_dir(junction, parents=True, exist_ok=True)  # the existing-leaf "repair" case
    hermes_dir_acl.make_user_dir(junction / "thumbs", parents=True, exist_ok=True)  # through the junction
    assert hermes_dir_acl.grant_user_access(junction) is False
    granted, skipped, failed = hermes_dir_acl.grant_tree_user_access(home)
    update_windows_acl.repair_windows_home_acl()  # the post-update repair that used icacls /T
    assert junction in skipped and not failed and granted >= 2
    assert _dacls(watched) == before, "an elevated grant followed the junction out of the home"
    # The real entries in the home still got the grant.
    whoami = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True)
    sid = whoami.stdout.strip().rsplit(",", 1)[-1].strip().strip('"')
    home_acl = subprocess.run(["icacls", str(home / "cache")], capture_output=True, text=True, check=True).stdout
    user = subprocess.run(["whoami"], capture_output=True, text=True, check=True).stdout.strip()
    assert sid in home_acl or user.lower() in home_acl.lower(), home_acl
