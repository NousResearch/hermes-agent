"""Regression tests for the s4-w1a extraction of the update-recovery cluster.

The update-recovery breadcrumb helpers moved from ``hermes_cli/main.py`` to
``hermes_cli/update_recovery.py`` (main.py decomposition, mechanical move,
wave-1 shard s4 cluster c3). This suite pins the extraction contract:

1. ``hermes_cli.main`` re-exports every moved name, so callers and test
   monkeypatches on ``hermes_cli.main.<name>`` keep resolving to the SAME
   function object as before the move.
2. References to helpers whose canonical home is ``hermes_cli.main`` (and
   moved-but-test-patched siblings) are routed through a lazy ``_m()`` main
   reference, so monkeypatching ``hermes_cli.main.<name>`` still reaches this
   code path (``PROJECT_ROOT``, ``_is_windows``, ``_venv_scripts_dir``,
   ``_hermes_exe_shims``, ``_default_venv_install_target``,
   ``_repair_venv_via_import_probes``, ``_lazy_refresh_repair_specs``,
   ``_LAZY_REFRESH_REPAIR_PACKAGES``, ...).
"""

from __future__ import annotations

import sys

import pytest

import hermes_cli.main as main_mod
from hermes_cli import update_recovery as ur

MOVED = [
    "_update_marker_path",
    "_lazy_refresh_marker_path",
    "_pytest_owns_live_checkout",
    "_clear_marker_file",
    "_clear_update_incomplete_marker",
    "_clear_lazy_refresh_incomplete_marker",
    "_recover_from_interrupted_install",
    "_recover_lazy_refresh_marker_locked",
    "_recover_core_update_marker_locked",
    "_windows_running_hermes_launcher_locked",
]


@pytest.mark.parametrize("name", MOVED)
def test_main_reexports_same_object(name):
    """Re-export seam: hermes_cli.main.<name> IS the moved function object."""
    assert getattr(main_mod, name) is getattr(ur, name)


def test_marker_paths_follow_main_project_root(monkeypatch, tmp_path):
    """_m() routing: PROJECT_ROOT patches on hermes_cli.main reach the module."""
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    assert ur._update_marker_path() == tmp_path / ".update-incomplete"
    assert ur._lazy_refresh_marker_path() == tmp_path / ".lazy-refresh-incomplete"
    # re-exported aliases resolve identically
    assert main_mod._update_marker_path() == tmp_path / ".update-incomplete"


def test_clear_marker_file_never_raises(tmp_path):
    ur._clear_marker_file(tmp_path / "missing", label="x")  # FileNotFoundError path
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    ur._clear_marker_file(blocker, label="x")  # OSError path (IsADirectoryError)
    ur._clear_update_incomplete_marker()  # no markers anywhere -> no-ops
    ur._clear_lazy_refresh_incomplete_marker()


def test_pytest_guard_predicate_sandboxed(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert ur._pytest_owns_live_checkout(tmp_path) is False
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    assert ur._pytest_owns_live_checkout(tmp_path) is False  # not the live checkout


def test_windows_launcher_locked_sees_main_patches(monkeypatch, tmp_path):
    """_m() routing: patches on hermes_cli.main reach the moved code path."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"")

    monkeypatch.setattr(main_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(main_mod, "_venv_scripts_dir", lambda: tmp_path)
    monkeypatch.setattr(main_mod, "_hermes_exe_shims", lambda d: [shim])

    class FakeProc:
        def __init__(self, exe_path):
            self._exe = exe_path

        def exe(self):
            return self._exe

        def parents(self):
            return [FakeProc(str(shim))]

    monkeypatch.setattr("psutil.Process", lambda: FakeProc(sys.executable))
    assert ur._windows_running_hermes_launcher_locked() is True

    monkeypatch.setattr(main_mod, "_is_windows", lambda: False)
    assert ur._windows_running_hermes_launcher_locked() is False


def test_lazy_refresh_recovery_routes_through_main(monkeypatch, tmp_path, capsys):
    """Recovery uses main's patched helpers via _m() and clears the marker."""
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    marker = tmp_path / ".lazy-refresh-incomplete"
    marker.write_text("started=1\npid=1\n")

    monkeypatch.setattr(
        main_mod, "_default_venv_install_target", lambda: (["uv", "pip"], {})
    )
    monkeypatch.setattr(
        main_mod, "_repair_venv_via_import_probes", lambda *a, **k: "healthy"
    )

    ur._recover_lazy_refresh_marker_locked()

    assert not marker.exists()
    assert "confirmed" in capsys.readouterr().out


def test_lazy_refresh_manual_command_uses_main_repair_tables(
    monkeypatch, tmp_path, capsys
):
    """Non-healthy status reaches _m()._lazy_refresh_repair_specs/_LAZY_REFRESH_REPAIR_PACKAGES."""
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    marker = tmp_path / ".lazy-refresh-incomplete"
    marker.write_text("started=1\npid=1\n")

    monkeypatch.setattr(
        main_mod, "_default_venv_install_target", lambda: (["uv", "pip"], {})
    )
    monkeypatch.setattr(
        main_mod, "_repair_venv_via_import_probes", lambda *a, **k: "broken"
    )
    seen = []
    monkeypatch.setattr(
        main_mod,
        "_lazy_refresh_repair_specs",
        lambda pkgs: seen.append(pkgs) or ["pkg==1.0"],
    )

    ur._recover_lazy_refresh_marker_locked()

    assert seen and seen[0] == sorted(
        set(main_mod._LAZY_REFRESH_REPAIR_PACKAGES.values())
    )
    assert marker.exists()  # indeterminate/broken keeps the marker for next launch
    assert "Recover manually" in capsys.readouterr().out
