"""macOS Full Disk Access onboarding guidance (issue #52010 follow-up).

One FDA grant silences every per-folder TCC prompt permanently. Doctor
reports the state and prints the one-switch setup; `hermes setup` surfaces
the same tip at onboarding. The probe must never itself trigger a prompt —
it reads the FDA-gated TCC db directory, which returns EPERM (no dialog)
without the grant.
"""

import io
import contextlib

import hermes_cli.doctor as doctor_mod
from hermes_cli.setup import _print_macos_fda_tip


def _capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


class TestDoctorFdaCheck:
    def test_silent_on_non_macos(self, monkeypatch):
        monkeypatch.setattr(doctor_mod.sys, "platform", "linux")
        out = _capture(doctor_mod.check_macos_full_disk_access)
        assert out == ""

    def test_granted_reports_ok(self, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        tcc = tmp_path / "Library" / "Application Support" / "com.apple.TCC"
        tcc.mkdir(parents=True)
        monkeypatch.setattr(doctor_mod.Path, "home", classmethod(lambda cls: tmp_path))
        out = _capture(doctor_mod.check_macos_full_disk_access)
        assert "Full Disk Access granted" in out
        assert "Privacy_AllFiles" not in out

    def test_denied_prints_one_switch_guidance(self, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        tcc = tmp_path / "Library" / "Application Support" / "com.apple.TCC"
        tcc.mkdir(parents=True)
        monkeypatch.setattr(doctor_mod.Path, "home", classmethod(lambda cls: tmp_path))

        def _eperm(path):
            raise PermissionError(13, "Operation not permitted", str(path))

        monkeypatch.setattr(doctor_mod.os, "listdir", _eperm)
        out = _capture(doctor_mod.check_macos_full_disk_access)
        assert "Full Disk Access" in out
        assert "Privacy_AllFiles" in out
        assert "System Settings" in out

    def test_indeterminate_probe_is_silent(self, monkeypatch, tmp_path):
        """Missing TCC dir (weird install) must not nag."""
        monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
        monkeypatch.setattr(doctor_mod.Path, "home", classmethod(lambda cls: tmp_path))
        # tmp_path has no Library/Application Support/com.apple.TCC →
        # FileNotFoundError (an OSError that is not PermissionError).
        out = _capture(doctor_mod.check_macos_full_disk_access)
        assert out == ""


class TestSetupFdaTip:
    def test_silent_on_non_macos(self, monkeypatch):
        import hermes_cli.setup as setup_mod

        monkeypatch.setattr(setup_mod.sys, "platform", "linux")
        out = _capture(_print_macos_fda_tip)
        assert out == ""

    def test_silent_when_already_granted(self, monkeypatch, tmp_path):
        import hermes_cli.setup as setup_mod

        monkeypatch.setattr(setup_mod.sys, "platform", "darwin")
        tcc = tmp_path / "Library" / "Application Support" / "com.apple.TCC"
        tcc.mkdir(parents=True)
        monkeypatch.setattr(setup_mod.Path, "home", classmethod(lambda cls: tmp_path))
        out = _capture(_print_macos_fda_tip)
        assert out == ""

    def test_tip_printed_when_denied(self, monkeypatch, tmp_path):
        import hermes_cli.setup as setup_mod

        monkeypatch.setattr(setup_mod.sys, "platform", "darwin")
        tcc = tmp_path / "Library" / "Application Support" / "com.apple.TCC"
        tcc.mkdir(parents=True)
        monkeypatch.setattr(setup_mod.Path, "home", classmethod(lambda cls: tmp_path))

        def _eperm(path):
            raise PermissionError(13, "Operation not permitted", str(path))

        monkeypatch.setattr(setup_mod.os, "listdir", _eperm)
        out = _capture(_print_macos_fda_tip)
        assert "Full Disk Access" in out
        assert "Privacy_AllFiles" in out
        assert "survives every Hermes update" in out
