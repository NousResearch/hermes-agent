"""The external-holder census behind profile delete/rename (PR #93508 review).

POSIX shares the state.db holder scan's descriptor census (``hermes_state_holders``): a deleted
file still open under the profile, a holder that reached it through another path, and a
Hermes process whose descriptors cannot be read all keep the profile in place. Windows asks
Restart Manager about every file in the profile at once; when that fails, its psutil sweep
reads open files of same-user processes only: fetching another user's (system) process
handles faulted inside psutil on Windows + Python 3.14 and killed the operation with no
Python exception.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import profile_lifecycle, profiles
from hermes_constants import named_profile_is_deleted
import hermes_state_holders


class _Denied(Exception):
    pass


def _fake_psutil(procs: list["_Proc"], me: str) -> types.ModuleType:
    module = types.ModuleType("psutil")
    module.NoSuchProcess = module.ZombieProcess = module.AccessDenied = _Denied

    def process_iter(attrs):
        # psutil populates ``info`` by calling every requested attribute up front.
        for proc in procs:
            proc.info = {name: getattr(proc, name)() for name in attrs}
            yield proc

    module.process_iter = process_iter
    module.Process = lambda _pid: types.SimpleNamespace(username=lambda: me)
    return module


class _Proc:
    def __init__(self, pid: int, user: str | None, paths: list[Path]):
        self._pid, self._user, self._paths = pid, user, paths
        self.open_files_calls = 0

    def pid(self):
        return self._pid

    def username(self):
        return self._user

    def open_files(self):
        self.open_files_calls += 1
        return [types.SimpleNamespace(path=str(path)) for path in self._paths]


@pytest.mark.platforms("windows")
def test_census_never_reads_handles_of_processes_it_cannot_prove_same_user(tmp_path, monkeypatch):
    profile = tmp_path / "profiles" / "alpha"
    profile.mkdir(parents=True)
    held = profile / "state.db"
    system = _Proc(4, "NT AUTHORITY\\SYSTEM", [held])
    unreadable_owner = _Proc(5, None, [held])
    sibling = _Proc(6, "me", [held])
    stranger = _Proc(7, "me", [tmp_path / "elsewhere.txt"])
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil([system, unreadable_owner, sibling, stranger], "me"))

    def restart_manager_unavailable(_root):
        raise OSError("rstrtmgr unavailable")

    # The per-process scan is POSIX's census and Windows' fallback.
    monkeypatch.setattr(profile_lifecycle, "_windows_profile_holders", restart_manager_unavailable)

    assert profile_lifecycle.external_profile_file_holders(profile) == [6]
    assert system.open_files_calls == 0
    assert unreadable_owner.open_files_calls == 0

    # A retry narrowed to the first census's holders reads no other process's handles.
    stranger.open_files_calls = 0
    assert profile_lifecycle.external_profile_file_holders(profile, [6]) == [6]
    assert stranger.open_files_calls == 0


def test_release_is_confirmed_by_a_fresh_census(monkeypatch):
    """A holder that exits can leave a child it spawned holding the profile; that child is
    only visible to a new census, so re-checking the old holders alone must not release."""
    parent, child = 7, 8
    live = {parent, child}
    visible = {parent}  # the child is not a candidate while its parent lives

    def census(_profile, candidates=None):
        if candidates is None:
            return sorted(live & visible)
        held = sorted(live & set(candidates))
        live.discard(parent)  # the parent exits after its first re-check
        visible.add(child)
        return held

    monkeypatch.setattr(profile_lifecycle, "external_profile_file_holders", census)
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_DB_RELEASE_TIMEOUT_SECONDS", 0.5)

    assert profile_lifecycle.wait_for_external_profile_file_release("profile") == [child]


# Holds the file named on its first stdin line until stdin closes. ``undumpable`` makes
# /proc/<pid>/fd root-owned, as for ``sshd: <user>``, ``gpg-agent`` or a Chromium sandbox.
_HOLDER = """
import sys
held = open(sys.stdin.readline().strip(), "ab")
if "undumpable" in sys.argv:
    import ctypes
    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE
print("ready", flush=True)
sys.stdin.read()
"""


@contextmanager
def _holding(path: Path, script: Path, *args: str):
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_HOLDER)
    proc = subprocess.Popen([sys.executable, str(script), *args], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    try:
        proc.stdin.write(f"{path}\n")
        proc.stdin.flush()
        assert proc.stdout.readline().strip() == "ready"
        yield proc
    finally:
        proc.stdin.close()
        proc.wait(30)


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    root = tmp_path / "isolated-hermes"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # No service manager, live backend or multiplexer is part of this census test.
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_bot_desktop", "_notify_multiplexer"):
        monkeypatch.setattr(profiles, name, lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.profiles_process_stop._profile_bound_backend_pids", lambda *a, **k: [])
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_DB_RELEASE_TIMEOUT_SECONDS", 0)
    return profiles.create_profile("worker", no_alias=True, no_skills=True)


@pytest.mark.platforms("linux")
def test_holder_of_a_deleted_file_under_the_profile_blocks_deletion(profile_home, tmp_path):
    """/proc spells it ``<path> (deleted)``: the process still writes into this generation."""
    rotated = profile_home / "logs" / "rotated.log"
    rotated.parent.mkdir(parents=True, exist_ok=True)
    with _holding(rotated, tmp_path / "bin" / "holder.py") as holder:
        rotated.unlink()
        assert holder.pid in profile_lifecycle.external_profile_file_holders(profile_home)
        with pytest.raises(RuntimeError, match=str(holder.pid)):
            profiles.delete_profile("worker", yes=True)
        assert profile_home.is_dir() and not named_profile_is_deleted(profile_home)

    assert profiles.delete_profile("worker", yes=True) == profile_home
    assert not profile_home.exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("census_through_link", [True, False], ids=["census-via-link", "holder-via-link"])
def test_holder_through_a_symlinked_profile_path_is_found(tmp_path, census_through_link):
    real = tmp_path / "real" / "profiles" / "alpha"
    real.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "real", target_is_directory=True)
    aliased = link / "profiles" / "alpha"
    census_root, opened = (aliased, real) if census_through_link else (real, aliased)

    with _holding(opened / "state.db", tmp_path / "bin" / "holder.py") as holder:
        assert holder.pid in profile_lifecycle.external_profile_file_holders(census_root)
    assert holder.pid not in profile_lifecycle.external_profile_file_holders(census_root)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("hermes", [
    True,
    # Every desktop or SSH login keeps such processes alive (``sshd: <user>``, ``gpg-agent``,
    # Chromium's sandbox); counting them would make deletion impossible. Root reads them all.
    pytest.param(False, marks=pytest.mark.skipif(os.geteuid() == 0, reason="root reads every fd table")),
], ids=["hermes", "not-hermes"])
def test_uninspectable_process_counts_only_when_it_is_hermes(tmp_path, hermes):
    """A same-user process whose descriptors cannot be read may hold any file in the profile."""
    profile = tmp_path / "profiles" / "alpha"
    profile.mkdir(parents=True)
    script = tmp_path / "bin" / ("hermes_cli/main.py" if hermes else "holder.py")

    with _holding(profile / "state.db", script, "--hermes-home", str(profile), "undumpable") as holder:
        assert (holder.pid in profile_lifecycle.external_profile_file_holders(profile)) is hermes


@pytest.mark.platforms("linux")
def test_holder_reaching_the_profile_through_a_bind_mount_is_found_by_identity(tmp_path, monkeypatch):
    """A Docker sandbox run as the host user sees ``<home>/sandboxes/<task>`` as ``/workspace``,
    and that is how /proc spells its descriptors; the open file's identity still names ours."""
    profile = tmp_path / "profiles" / "alpha"
    held = profile / "sandboxes" / "task" / "out.log"
    held.parent.mkdir(parents=True)
    projected_os = types.SimpleNamespace(**vars(os))
    monkeypatch.setattr(hermes_state_holders, "os", projected_os)

    def readlink(path):
        target = os.readlink(path)
        return "/workspace/out.log" if target == str(held) else target

    projected_os.readlink = readlink

    with _holding(held, tmp_path / "bin" / "holder.py") as holder:
        assert holder.pid in profile_lifecycle.external_profile_file_holders(profile)
