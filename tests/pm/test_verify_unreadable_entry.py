"""An entry this user cannot list is reported as unverifiable, never a crash.

On Windows an entry published from an elevated session can carry a DACL the
standard user cannot read: is_file() on its contents returns False and
iterdir() raises PermissionError. Verification diagnoses used to list the
entry unguarded, so `hermes pm doctor` died on the first such entry and never
reported the packages after it.
"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

import pm.paths as paths
from pm import get_package
from pm.lock import Facts
from pm.package import _missing_reason
from pm.store import current_target
from tests.pm._fixtures import make_tar, served  # noqa: F401
from tests.pm.test_pm_core import _pin, pm_env  # noqa: F401


def _deny(monkeypatch, entry: Path) -> None:
    """Make `entry` behave like a tree this user is not allowed to read."""
    real_iterdir, real_is_file = Path.iterdir, Path.is_file

    def iterdir(self):
        if self == entry:
            raise PermissionError(errno.EACCES, "Access is denied", str(entry))
        return real_iterdir(self)

    def is_file(self, *args, **kwargs):
        if entry in self.parents:
            return False
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(Path, "is_file", is_file)


def test_unreadable_entry_is_unverifiable_not_missing(tmp_path, monkeypatch):
    entry = tmp_path / "tool-1.0"
    (entry / "bin").mkdir(parents=True)
    (entry / "bin" / "tool").write_bytes(b"x")
    _deny(monkeypatch, entry)

    reason = _missing_reason(entry / "bin" / "tool", entry)

    assert reason.startswith("cannot verify bin/tool:")
    assert "not readable by this user" in reason
    assert "Access is denied" in reason
    assert "missing under" not in reason


def test_absent_entry_is_still_reported_missing(tmp_path):
    entry = tmp_path / "tool-1.0"

    assert _missing_reason(entry / "bin" / "tool", entry) == (
        f"bin/tool missing under {entry}; store entry does not exist"
    )


def test_chromium_marker_check_survives_unreadable_entry(tmp_path, monkeypatch):
    entry = tmp_path / "chromium-1"
    entry.mkdir()
    (entry / "INSTALLATION_COMPLETE").write_text("", encoding="utf-8")
    _deny(monkeypatch, entry)

    reason = get_package("chromium").verify(entry, current_target())

    assert reason.startswith("cannot verify INSTALLATION_COMPLETE:")
    assert "not readable by this user" in reason


def test_doctor_reports_unreadable_entry_and_keeps_walking(pm_env, monkeypatch, capsys):
    from pm.cli import cmd_doctor
    from pm.install import ensure

    lockfile_path, _runtime, docroot, _base_url = pm_env
    _, digest = make_tar(docroot, "deptool-1.0.tar.gz", {"bin/faketool": "#!d"})
    _pin(lockfile_path, "deptool", "1.0", digest)
    ensure("deptool", base_env={})
    ensure("faketool", base_env={})
    # deptool walks first; before the fix its unreadable entry aborted doctor
    # with PermissionError and faketool was never reported.
    entry = paths.store_root() / Facts(paths.facts_path()).get("deptool")["entry"]
    _deny(monkeypatch, entry)

    assert cmd_doctor(None) == 1

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("✗ deptool: installed but failed verification: cannot verify bin/faketool:")
    assert "not readable by this user" in lines[0]
    assert lines[1] == "✓ faketool 1.0"


@pytest.mark.parametrize("stage", ["verify", "digest"])
def test_doctor_reports_os_error_per_entry(pm_env, monkeypatch, capsys, stage):
    from pm.cli import cmd_doctor
    from pm.install import ensure

    ensure("faketool", base_env={})

    def denied(*_args, **_kwargs):
        raise PermissionError(errno.EACCES, "Access is denied")

    if stage == "verify":
        monkeypatch.setattr(type(get_package("faketool")), "verify", denied)
    else:
        # cmd_doctor imports tree_digest at call time, so this is the one it binds.
        monkeypatch.setattr("pm.store.tree_digest", denied)

    assert cmd_doctor(None) == 1
    out = capsys.readouterr().out
    assert "✗ faketool: could not be verified:" in out
    assert "Access is denied" in out
