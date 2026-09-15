"""``hermes doctor``'s stale-backend check: flag processes older than the code on disk.

A Desktop backend reuses across reconnects and keeps the modules it imported at startup, so one
that predates an in-place update fails mid-turn with ``cannot import name ...`` long after update
printed success. The update now recycles them; the doctor check covers the hand-updated checkout
and drift nobody restarted.
"""

from __future__ import annotations

import os

from hermes_cli import dashboard_procs, doctor
from hermes_cli import doctor_platform


def _touch(path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_newest_source_mtime_walks_core_packages_and_skips_heavy_dirs(tmp_path):
    _touch(tmp_path / "gateway" / "session.py", 100.0)
    _touch(tmp_path / "tui_gateway" / "server.py", 300.0)
    _touch(tmp_path / "hermes_constants.py", 200.0)
    # Ignored by design: caches, vendored bundles and build output never carry runtime code.
    _touch(tmp_path / "gateway" / "__pycache__" / "session.py", 999.0)
    _touch(tmp_path / "tools" / "node_modules" / "x.py", 999.0)
    _touch(tmp_path / "agent" / "release" / "x.py", 999.0)

    assert doctor_platform._newest_source_mtime(tmp_path) == 300.0


def test_newest_source_mtime_is_none_when_nothing_is_scannable(tmp_path):
    assert doctor_platform._newest_source_mtime(tmp_path / "missing") is None


def test_stale_backend_check_reports_pids_and_a_restart_action(monkeypatch):
    monkeypatch.setattr(doctor, "PROJECT_ROOT", doctor.PROJECT_ROOT)  # keep the real root
    monkeypatch.setattr(doctor_platform, "_newest_source_mtime", lambda root: 1000.0)
    monkeypatch.setattr(dashboard_procs, "stale_desktop_backend_pids", lambda reference: [4242, 4243])

    finding = doctor_platform._check_stale_backends(False)

    assert finding.issues == []
    assert len(finding.manual_issues) == 1
    assert "4242, 4243" in finding.manual_issues[0]


def test_stale_backend_check_is_silent_when_every_backend_is_current(monkeypatch):
    monkeypatch.setattr(doctor_platform, "_newest_source_mtime", lambda root: 1000.0)
    monkeypatch.setattr(dashboard_procs, "stale_desktop_backend_pids", lambda reference: [])

    finding = doctor_platform._check_stale_backends(False)

    assert finding.issues == [] and finding.manual_issues == []


def test_stale_backend_check_never_raises_on_a_probe_failure(monkeypatch):
    def _boom(reference):
        raise OSError("no process table")

    monkeypatch.setattr(doctor_platform, "_newest_source_mtime", lambda root: 1000.0)
    monkeypatch.setattr(dashboard_procs, "stale_desktop_backend_pids", _boom)

    finding = doctor_platform._check_stale_backends(False)

    assert finding.issues == [] and finding.manual_issues == []
