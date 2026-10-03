"""The external-holder census behind profile delete/rename (PR #93508 review).

The psutil scan reads open files of same-user processes only: fetching another user's
(system) process handles faulted inside psutil on Windows + Python 3.14 and killed the
operation with no Python exception. Windows instead asks Restart Manager about every file
in the profile at once.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from hermes_cli import profile_lifecycle


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


def test_restart_manager_census_registers_every_profile_file(tmp_path, monkeypatch):
    """Windows asks about the whole tree in one query; nested files count and we are not a holder."""
    import ctypes
    import os

    root = tmp_path / "profiles" / "alpha"
    files = [root / "state.db", root / "logs" / "mcp-stderr.log", root / "browser-profile" / "chrome" / "Cookies"]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    registered = []

    class Fn:
        def __init__(self, impl):
            self.impl = impl

        def __call__(self, *args):
            return self.impl(*args)

    def start(session, _flags, _key):
        session._obj.value = 1
        return 0

    def register(_session, count, names, *_rest):
        registered.extend(names[index] for index in range(count))
        return 0

    def get_list(_session, needed, count, apps, _reasons):
        needed._obj.value = 2
        if apps is None:
            return 234  # ERROR_MORE_DATA: sizing call
        apps[0].process.pid, apps[1].process.pid = os.getpid(), 4242
        count._obj.value = 2
        return 0

    class Api:
        RmStartSession, RmRegisterResources, RmGetList = Fn(start), Fn(register), Fn(get_list)
        RmEndSession = Fn(lambda _session: 0)

    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: Api(), raising=False)

    assert profile_lifecycle._windows_profile_holders(root) == [4242]
    assert sorted(registered) == sorted(str(path) for path in files)


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
