"""A POSIX desktop swap preserves live sessions and treats uncertain process scans as unsafe."""

from pathlib import Path
import sys
import types

import pytest

from hermes_cli import main_desktop, main_desktop_processes
from tests.hermes_cli.test_gui_command import _make_desktop_tree, _packaged_exe_rel


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("has_live", [True, False], ids=["live-and-previous", "previous-only"])
@pytest.mark.parametrize(
    ("running", "message"),
    [([2468], "still running (pid 2468)"), (None, "couldn't check whether hermes desktop is running")],
)
def test_posix_swap_keeps_previous_bundle_when_a_process_uses_it(
    tmp_path, monkeypatch, capsys, has_live, running, message,
):
    """an in-use or uncertain rollback bundle must survive cleanup and crash recovery."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    live_root = main_desktop._desktop_unpacked_root(live_exe, desktop_dir / "release")
    previous = live_root.with_name(live_root.name + main_desktop._DESKTOP_PREVIOUS_SUFFIX)
    if has_live:
        previous.mkdir(parents=True)
    else:
        live_root.rename(previous)
    previous_file = previous / "still-needed-by-live-app"
    previous_file.write_text("keep", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")

    monkeypatch.setattr(
        main_desktop_processes,
        "processes_running_from",
        lambda tree: running if tree == previous else [],
    )

    assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None
    if has_live:
        assert live_exe.read_text(encoding="utf-8") == "old"
    else:
        assert not live_root.exists()
        assert (previous / live_exe.relative_to(live_root)).read_text(encoding="utf-8") == "old"
    assert previous_file.read_text(encoding="utf-8") == "keep"
    assert not staging.exists()
    assert message in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_posix_swap_rolls_back_if_desktop_launch_races_the_precheck(tmp_path, monkeypatch, capsys):
    """a desktop appearing during the rename must get the original bundle back."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")

    checked = []

    def process_scan(tree):
        checked.append(tree)
        return [] if len(checked) == 1 else [9876]

    monkeypatch.setattr(main_desktop_processes, "processes_running_from", process_scan)

    assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None
    assert live_exe.read_text(encoding="utf-8") == "old"
    assert not (live_exe.parent.parent / (live_exe.parent.name + ".previous")).exists()
    assert not staging.exists()
    assert "still running (pid 9876)" in capsys.readouterr().out


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    ("running", "message"),
    [([4321], "still running (pid 4321)"), (None, "couldn't check whether hermes desktop is running")],
)
def test_posix_swap_leaves_a_running_desktop_untouched(tmp_path, monkeypatch, capsys, running, message):
    """an unattended swap must not kill the desktop or replace files under it."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")

    monkeypatch.setattr(main_desktop_processes, "processes_running_from", lambda _tree: running)
    monkeypatch.setattr(
        main_desktop,
        "_stop_desktop_processes_locking_build",
        lambda *args, **kwargs: pytest.fail("a live POSIX desktop must not be terminated"),
    )

    assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None
    assert live_exe.read_text(encoding="utf-8") == "old"
    assert not (live_exe.parent.parent / (live_exe.parent.name + ".previous")).exists()
    assert not staging.exists()
    assert message in capsys.readouterr().out


def test_desktop_process_scan_matches_only_processes_from_the_bundle(tmp_path, monkeypatch):
    """the swap guard should match the live bundle, not other running apps."""
    tree = tmp_path / "release" / "Hermes.app"
    exe = tree / "Contents" / "MacOS" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.touch()

    class _FakeProc:
        def __init__(self, pid, path):
            self.info = {"pid": pid, "exe": path}

    class _FakePsutil:
        @staticmethod
        def process_iter(attrs):
            assert attrs == ["pid", "exe", "name", "cmdline"]
            return [
                _FakeProc(41, str(exe)),
                _FakeProc(42, "/usr/bin/other"),
                types.SimpleNamespace(info={"pid": 43, "exe": None, "name": "Hermes Helper", "cmdline": []}),
                types.SimpleNamespace(info={"pid": 44, "exe": None, "name": None, "cmdline": None}),
            ]

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)

    assert main_desktop_processes.processes_running_from(tree) is None

    class _ReadablePsutil:
        @staticmethod
        def process_iter(attrs):
            assert attrs == ["pid", "exe", "name", "cmdline"]
            return [_FakeProc(41, str(exe)), _FakeProc(42, "/usr/bin/other")]

    monkeypatch.setitem(sys.modules, "psutil", _ReadablePsutil)
    assert main_desktop_processes.processes_running_from(tree) == [41]

    class _UnknownProcessPsutil:
        @staticmethod
        def process_iter(_attrs):
            return [types.SimpleNamespace(info={"pid": 43, "exe": None, "name": None, "cmdline": None})]

    monkeypatch.setitem(sys.modules, "psutil", _UnknownProcessPsutil)
    assert main_desktop_processes.processes_running_from(tree) is None

    class _DeniedProc:
        @property
        def info(self):
            raise PermissionError("process details are hidden")

    class _DeniedPsutil:
        @staticmethod
        def process_iter(_attrs):
            return [_DeniedProc()]

    monkeypatch.setitem(sys.modules, "psutil", _DeniedPsutil)
    assert main_desktop_processes.processes_running_from(tree) is None


def test_desktop_process_scan_fails_closed_when_psutil_cannot_enumerate(monkeypatch):
    class _FakePsutil:
        @staticmethod
        def process_iter(_attrs):
            raise OSError("process table unavailable")

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)

    assert main_desktop_processes.processes_running_from(Path("release")) is None
