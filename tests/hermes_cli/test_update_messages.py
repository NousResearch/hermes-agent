"""User-facing wording contracts for ``hermes update`` failure/partial output.

Invariants only (what the user must be able to read and act on), never whole-string snapshots.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import update_cmd
import hermes_cli.update_receipt as ur


class TestFleetMatrixVerdict:
    """cli-09: a stale/down fleet must end with an explicit "not complete" verdict, because
    ``✓ Update complete!`` was already printed before the restart phase."""

    def test_stale_rows_print_counted_verdict_with_commands(self, capsys):
        fleet = [
            {"profile": "default", "pid": 1234, "code_sha": "abc12345ff", "state": "stale"},
            {"profile": "work", "pid": 99, "code_sha": None, "state": "down"},
            {"profile": "ok", "pid": 7, "code_sha": "a" * 40, "state": "current"},
        ]
        assert ur.print_fleet_version_matrix(fleet) is True
        out = capsys.readouterr().out
        verdict = [line for line in out.splitlines() if "Update not complete" in line]
        assert len(verdict) == 1
        assert verdict[0].startswith("✗")
        assert "2 gateway" in out
        assert "hermes gateway restart" in out
        assert "hermes -p <profile> gateway restart" in out
        assert "hermes gateway status" in out

    def test_unknown_rows_print_no_verdict(self, capsys):
        fleet = [{"profile": "default", "pid": 1, "code_sha": None, "state": "unknown"}]
        assert ur.print_fleet_version_matrix(fleet) is False
        assert "Update not complete" not in capsys.readouterr().out


class TestCalledProcessErrorMessage:
    """cli-10: a dependency-install failure leads with a plain sentence, not the exception repr."""

    @pytest.fixture
    def no_exit(self, monkeypatch):
        monkeypatch.setattr(update_cmd, "_finalize_receipt", lambda *_a, **_k: None)
        monkeypatch.setattr(update_cmd.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    def test_dep_install_failure_lead_and_next_step(self, capsys, no_exit, monkeypatch):
        monkeypatch.setattr(update_cmd._m(), "_is_windows", lambda: False)
        exc = subprocess.CalledProcessError(1, ["uv", "pip", "install", "-e", ".[all]"], stderr="error: disk full")
        with pytest.raises(SystemExit) as info:
            update_cmd._handle_update_called_process_error(exc, object(), False, False)
        assert info.value.code == 1
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert lines[0].startswith("✗ Python dependency install failed")
        assert "code update itself succeeded" in lines[0]
        assert "returned non-zero exit status" not in lines[0]
        assert "disk full" in out
        assert "hermes update" in out
        assert "Re-downloading" not in out

    def test_non_dep_failure_moves_repr_to_details_line(self, capsys, no_exit, monkeypatch):
        monkeypatch.setattr(update_cmd._m(), "_is_windows", lambda: False)
        exc = subprocess.CalledProcessError(128, ["git", "pull"], stderr="fatal: not a git repository")
        with pytest.raises(SystemExit):
            update_cmd._handle_update_called_process_error(exc, object(), False, False)
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert lines[0].startswith("✗ Git update failed")
        assert "returned non-zero exit status" not in lines[0]
        assert any(line.strip().startswith("Details:") and "returned non-zero exit status 128" in line
                   for line in lines)
