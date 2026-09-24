"""macOS Full Disk Access onboarding guidance (issue #52010 follow-up).

One FDA grant silences every per-folder TCC prompt permanently. Doctor reports
the state and prints the one-switch setup; ``hermes setup`` surfaces the same
tip at onboarding. The probe reads the FDA-gated TCC db directory, which
returns EPERM (no dialog) without the grant; any other error is indeterminate
and must stay silent. The probe is gated on the real host, so the macOS
branches run on the macOS lane and the off-macOS silence runs on Linux.
"""

import contextlib
import io
import os
import pathlib

import pytest

from hermes_cli import doctor_platform
from hermes_cli.setup_quick import _print_macos_fda_tip


def _capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture
def tcc_dir(home):
    tcc = home / "Library" / "Application Support" / "com.apple.TCC"
    tcc.mkdir(parents=True)
    return tcc


@pytest.fixture
def fda_denied(tcc_dir, monkeypatch):
    """The TCC dir exists but listing it fails with EPERM, exactly as without the grant."""
    real_listdir = os.listdir

    def _listdir(path="."):
        if pathlib.Path(path) == tcc_dir:
            raise PermissionError(13, "Operation not permitted", str(path))
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", _listdir)
    return tcc_dir


@pytest.mark.linux_only
def test_doctor_and_setup_are_silent_off_macos(tcc_dir):
    assert _capture(doctor_platform.check_macos_full_disk_access) == ""
    assert _capture(_print_macos_fda_tip) == ""


@pytest.mark.macos_only
def test_doctor_reports_granted_without_guidance(tcc_dir):
    out = _capture(doctor_platform.check_macos_full_disk_access)
    assert "Full Disk Access granted" in out
    assert "Privacy_AllFiles" not in out


@pytest.mark.macos_only
def test_doctor_prints_one_switch_guidance_when_denied(fda_denied):
    out = _capture(doctor_platform.check_macos_full_disk_access)
    assert "Privacy_AllFiles" in out
    assert "granted" not in out


@pytest.mark.macos_only
def test_doctor_is_silent_when_probe_is_indeterminate(home):
    """Missing TCC dir (FileNotFoundError, not EPERM) must not nag."""
    assert _capture(doctor_platform.check_macos_full_disk_access) == ""


@pytest.mark.macos_only
def test_setup_tip_is_silent_when_already_granted(tcc_dir):
    assert _capture(_print_macos_fda_tip) == ""


@pytest.mark.macos_only
def test_setup_tip_printed_when_denied(fda_denied):
    out = _capture(_print_macos_fda_tip)
    assert "Privacy_AllFiles" in out
