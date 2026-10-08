"""Behavior contracts for the selected PostgreSQL manifest runner (HX006)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import validate_selected_postgresql_manifest as manifest


def test_partition_runs_bounded_canonical_runner(monkeypatch, capsys):
    def fake_run(command, **kwargs):
        assert command == [
            str(manifest.RUNNER),
            "-o",
            "addopts=",
            "tests/example.py",
            "-v",
            "--tb=short",
        ]
        assert kwargs["cwd"] == manifest.ROOT
        assert kwargs["env"]["HERMES_TEST_WORKERS"] == "1"
        assert kwargs["env"]["HERMES_TEST_FILE_RETRIES"] == "0"
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["text"] is True
        assert 60 < kwargs["timeout"] < float("inf")
        return subprocess.CompletedProcess(
            command, 0, "=== Summary: 1 files, 7 tests passed, 0 failed in 1s ===\n"
        )

    monkeypatch.setattr(manifest.subprocess, "run", fake_run)
    assert manifest.run_partition("sample", ("tests/example.py",)) == ("sample", 1, 7)
    assert "=== Summary:" in capsys.readouterr().out


def test_partition_nonzero_exit_still_fails(monkeypatch):
    def fake_run(command, **kwargs):
        assert kwargs["timeout"] > 60
        return subprocess.CompletedProcess(
            command, 2, "=== Summary: 1 files, 7 tests passed, 0 failed in 1s ===\n"
        )

    monkeypatch.setattr(manifest.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=r"sample failed \(exit=2\)"):
        manifest.run_partition("sample", ("tests/example.py",))


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_timeout_hides_partial_output_and_cause_in_both_modes(
    monkeypatch, capsys, mode
):
    def fake_run(command, **kwargs):
        assert 60 < kwargs["timeout"] < float("inf")
        raise subprocess.TimeoutExpired(
            ["private-argv", *command],
            kwargs["timeout"],
            output=b"private-output",
            stderr=b"private-stderr",
        )

    monkeypatch.setattr(manifest.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [str(Path(manifest.__file__)), "--mode", mode])
    with pytest.raises(RuntimeError, match="timed out") as caught:
        manifest.main()
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "private" not in str(caught.value)
    assert "MANIFEST_OK" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_modes_preserve_summary_counts(monkeypatch, capsys, mode):
    def fake_run(command, **kwargs):
        paths = command[3:-2]
        assert kwargs["timeout"] > 60
        return subprocess.CompletedProcess(
            command,
            0,
            f"=== Summary: {len(paths)} files, {len(paths) * 2} tests passed, 0 failed in 1s ===\n",
        )

    monkeypatch.setattr(manifest.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [str(Path(manifest.__file__)), "--mode", mode])
    assert manifest.main() == 0
    assert (
        f"MANIFEST_OK mode={mode} files={len(manifest.MANIFEST)} tests={len(manifest.MANIFEST) * 2}"
        in capsys.readouterr().out
    )
