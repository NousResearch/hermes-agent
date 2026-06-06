"""Regression tests for Hermes update/Desktop build contract diagnostics."""

import json
import subprocess

from hermes_cli import main as hm


def test_desktop_backend_stamp_mismatch_reads_install_stamp(monkeypatch, tmp_path):
    project_root = tmp_path / "hermes-agent"
    stamp_path = project_root / "apps" / "desktop" / "build" / "install-stamp.json"
    stamp_path.parent.mkdir(parents=True)
    stamp_path.write_text(
        json.dumps({"schemaVersion": 1, "commit": "oldcommit1234567890"}),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        assert cmd == ["git", "rev-parse", "HEAD"]
        return subprocess.CompletedProcess(cmd, 0, stdout="newcommitabcdef123456\n", stderr="")

    monkeypatch.setattr(hm.subprocess, "run", fake_run)
    monkeypatch.setattr(hm, "_desktop_stamp_path", lambda: tmp_path / "missing-desktop-build-stamp.json")

    assert hm._desktop_backend_stamp_mismatch(project_root) == (
        "oldcommit1234567890",
        "newcommitabcdef123456",
    )


def test_desktop_backend_stamp_mismatch_accepts_matching_prefix(monkeypatch, tmp_path):
    project_root = tmp_path / "hermes-agent"
    stamp_path = project_root / "apps" / "desktop" / "build" / "install-stamp.json"
    stamp_path.parent.mkdir(parents=True)
    stamp_path.write_text(
        json.dumps({"schemaVersion": 1, "commit": "150687447"}),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="150687447bc9e01a028c3dedf9589406cc321a4f\n",
            stderr="",
        )

    monkeypatch.setattr(hm.subprocess, "run", fake_run)
    monkeypatch.setattr(hm, "_desktop_stamp_path", lambda: tmp_path / "missing-desktop-build-stamp.json")

    assert hm._desktop_backend_stamp_mismatch(project_root) is None


def test_print_desktop_repair_instruction_shows_explicit_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(hm, "_desktop_backend_stamp_mismatch", lambda _root: ("oldcommit123456", "newcommitabcdef"))

    hm._print_desktop_repair_instruction(tmp_path, reason="Post-update Desktop/backend contract check failed.")

    out = capsys.readouterr().out
    assert "Post-update Desktop/backend contract check failed." in out
    assert "Desktop build commit: oldcommit123" in out
    assert "Backend/source commit: newcommitabc" in out
    assert "Repair command: hermes desktop --build-only" in out


def test_windows_application_control_error_detection():
    exc = OSError("[WinError 4551] Application control policy blocked this file.")
    exc.winerror = 4551

    assert hm._is_windows_application_control_error(exc)
    assert hm._is_windows_application_control_error(
        OSError("[WinError 4551] Application control policy blocked this file.")
    )
    assert not hm._is_windows_application_control_error(OSError("permission denied"))
