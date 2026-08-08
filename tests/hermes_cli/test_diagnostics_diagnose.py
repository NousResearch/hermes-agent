"""Tests for ``hermes debug diagnose`` (U4 / KTD6).

Covers the three failure modes the plan names for the WPR helper (absent,
stuck, failing), plus the sanitization rule the process tree shares with the
desktop bundle manifest: basename only, never a command line.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import pytest

from hermes_cli.diagnostics_diagnose import (
    CommandResult,
    basename_only,
    collect_process_tree,
    find_wpr,
    run_bounded,
    run_diagnose,
    run_wpr_trace,
)

MARKER = "HERMES-DIAGNOSE-CANARY-4b21"


# ── fakes ────────────────────────────────────────────────────────────


class FakeProc:
    """A child process whose behaviour each test dictates."""

    def __init__(self, *, output="", returncode=0, hang=False):
        self._output = output
        self.returncode = returncode
        self._hang = hang
        self.killed = False
        self.communicate_calls = 0

    def communicate(self, timeout=None):
        self.communicate_calls += 1
        # A hung child stays hung until it is killed; after the kill it drains.
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired(cmd="wpr", timeout=timeout or 0)
        return self._output, ""

    def kill(self):
        self.killed = True


def make_popen(procs):
    """Return a Popen stand-in handing out *procs* in order, recording argv."""
    calls: list[list[str]] = []
    remaining = list(procs)

    def popen(cmd, **_kwargs):
        calls.append(list(cmd))
        return remaining.pop(0) if remaining else FakeProc()

    popen.calls = calls  # type: ignore[attr-defined]
    return popen


def ns(**kwargs):
    return argparse.Namespace(**kwargs)


# ── basename / sanitization ──────────────────────────────────────────


def test_basename_only_strips_paths_and_argv():
    assert basename_only(r"C:\Program Files\Hermes\Hermes.exe") == "Hermes.exe"
    assert basename_only("/opt/hermes/bin/hermes") == "hermes"
    assert basename_only(f"/opt/hermes/bin/hermes --token={MARKER}") == "hermes"
    assert basename_only("") == "unknown"
    assert basename_only(None) == "unknown"


def test_process_tree_keeps_only_pid_ppid_basename():
    def source():
        return [
            {
                "pid": 100,
                "ppid": 1,
                "name": "hermes.exe",
                "exe": r"C:\hermes\hermes.exe",
                # Fields psutil could return that must never survive.
                "cmdline": ["hermes", f"--api-key={MARKER}"],
                "environ": {"HERMES_TOKEN": MARKER},
            },
            {"pid": 101, "ppid": 100, "name": "python.exe", "exe": r"C:\py\python.exe"},
            {"pid": 1, "ppid": 0, "name": "explorer.exe", "exe": r"C:\Windows\explorer.exe"},
            {"pid": 900, "ppid": 1, "name": "chrome.exe", "exe": r"C:\chrome\chrome.exe"},
        ]

    tree = collect_process_tree(iter_processes=source)

    # The hermes process, its parent (ancestor) and its child; not chrome.
    assert tree == [
        {"pid": 1, "ppid": 0, "name": "explorer.exe"},
        {"pid": 100, "ppid": 1, "name": "hermes.exe"},
        {"pid": 101, "ppid": 100, "name": "python.exe"},
    ]
    assert MARKER not in json.dumps(tree)
    for entry in tree:
        assert sorted(entry) == ["name", "pid", "ppid"]


def test_process_tree_survives_a_cyclic_parent_map():
    def source():
        return [
            {"pid": 5, "ppid": 6, "name": "hermes"},
            {"pid": 6, "ppid": 5, "name": "shell"},
        ]

    assert [entry["pid"] for entry in collect_process_tree(iter_processes=source)] == [5, 6]


def test_process_tree_is_empty_when_the_source_explodes():
    def source():
        raise RuntimeError("psutil unavailable")

    assert collect_process_tree(iter_processes=source) == []


# ── wpr availability ─────────────────────────────────────────────────


def test_find_wpr_returns_none_when_absent(monkeypatch):
    monkeypatch.delenv("SystemRoot", raising=False)
    monkeypatch.delenv("SYSTEMROOT", raising=False)
    assert find_wpr(which=lambda _name: None) is None


def test_find_wpr_prefers_path():
    assert find_wpr(which=lambda _name: "/usr/bin/wpr") == "/usr/bin/wpr"


def test_wpr_absent_is_skipped_not_failed(tmp_path, monkeypatch):
    monkeypatch.delenv("SystemRoot", raising=False)
    monkeypatch.delenv("SYSTEMROOT", raising=False)
    monkeypatch.setattr("hermes_cli.diagnostics_diagnose.shutil.which", lambda _n: None)

    result = run_wpr_trace(tmp_path / "unsafe-to-share", seconds=0)

    assert result["status"] == "skipped"
    assert "wpr.exe" in result["reason"]
    # Nothing was created for a trace that never ran.
    assert not (tmp_path / "unsafe-to-share").exists()


# ── wpr happy path + hard timeout ────────────────────────────────────


def test_wpr_records_and_stops(tmp_path):
    popen = make_popen([FakeProc(), FakeProc()])
    slept: list[float] = []

    result = run_wpr_trace(
        tmp_path / "unsafe",
        profile="GeneralProfile",
        seconds=7,
        wpr_path="wpr.exe",
        popen=popen,
        sleep=slept.append,
    )

    assert result["status"] == "recorded"
    assert result["seconds"] == 7
    assert result["file"] == "trace.etl"
    assert slept == [7]
    assert popen.calls[0][:3] == ["wpr.exe", "-start", "GeneralProfile"]
    assert popen.calls[1][:2] == ["wpr.exe", "-stop"]
    assert popen.calls[1][2].endswith("trace.etl")


def test_stuck_wpr_start_is_killed_at_the_hard_timeout(tmp_path):
    stuck = FakeProc(hang=True)
    popen = make_popen([stuck])

    result = run_wpr_trace(
        tmp_path / "unsafe",
        seconds=0,
        timeout_s=0.01,
        wpr_path="wpr.exe",
        popen=popen,
        sleep=lambda _s: None,
    )

    assert stuck.killed is True
    assert result["status"] == "timeout"
    assert result["phase"] == "start"
    # A start that never returned must not be followed by a stop.
    assert len(popen.calls) == 1


def test_stuck_wpr_stop_is_killed_and_the_session_cancelled(tmp_path):
    stuck_stop = FakeProc(hang=True)
    popen = make_popen([FakeProc(), stuck_stop, FakeProc()])

    result = run_wpr_trace(
        tmp_path / "unsafe",
        seconds=0,
        timeout_s=0.01,
        wpr_path="wpr.exe",
        popen=popen,
        sleep=lambda _s: None,
    )

    assert stuck_stop.killed is True
    assert result["status"] == "timeout"
    assert result["phase"] == "stop"
    # A killed -stop leaves a live system-wide session; it must be cancelled.
    assert popen.calls[-1] == ["wpr.exe", "-cancel"]


def test_wpr_start_failure_reports_the_first_output_line(tmp_path):
    popen = make_popen([FakeProc(returncode=5, output="Error: elevated privileges required.\nmore\n")])

    result = run_wpr_trace(
        tmp_path / "unsafe", seconds=0, wpr_path="wpr.exe", popen=popen, sleep=lambda _s: None
    )

    assert result["status"] == "failed"
    assert result["phase"] == "start"
    assert result["reason"] == "Error: elevated privileges required."


def test_run_bounded_reports_a_clean_exit():
    popen = make_popen([FakeProc(output="fine", returncode=0)])
    result = run_bounded(["wpr", "-status"], 5, popen=popen)

    assert isinstance(result, CommandResult)
    assert result.ok is True
    assert result.timed_out is False
    assert result.output == "fine"


# ── command ──────────────────────────────────────────────────────────


def test_diagnose_writes_a_report_without_wpr(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.diagnostics_diagnose.collect_process_tree",
        lambda **_k: [{"pid": 100, "ppid": 1, "name": "hermes.exe"}],
    )

    out = tmp_path / "report"
    code = run_diagnose(ns(out=str(out), wpr=False))

    assert code == 0
    report = json.loads((out / "diagnose.json").read_text(encoding="utf-8"))
    assert report["process_tree"] == [{"pid": 100, "ppid": 1, "name": "hermes.exe"}]
    assert report["wpr"]["status"] == "skipped"
    # No opt-in means no unsafe directory at all.
    assert not (out / "unsafe-to-share").exists()
    assert "hermes.exe" in capsys.readouterr().out


def test_diagnose_labels_the_wpr_output_unsafe_to_share(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.diagnostics_diagnose.collect_process_tree", lambda **_k: [])
    monkeypatch.setattr(
        "hermes_cli.diagnostics_diagnose.run_wpr_trace",
        lambda _dir, **_k: {"status": "recorded", "profile": "GeneralProfile", "seconds": 30, "file": "trace.etl"},
    )

    out = tmp_path / "report"
    run_diagnose(ns(out=str(out), wpr=True, wpr_profile="GeneralProfile", wpr_seconds=30, timeout=300))

    readme = (out / "unsafe-to-share" / "README.txt").read_text(encoding="utf-8")
    assert "DO NOT SHARE" in readme
    assert "COMMAND LINES" in readme

    printed = capsys.readouterr().out
    assert "UNSANITIZED" in printed


def test_diagnose_succeeds_when_wpr_is_unavailable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.diagnostics_diagnose.collect_process_tree", lambda **_k: [])
    monkeypatch.setattr(
        "hermes_cli.diagnostics_diagnose.run_wpr_trace",
        lambda _dir, **_k: {"status": "skipped", "reason": "wpr.exe not found"},
    )

    out = tmp_path / "report"
    code = run_diagnose(ns(out=str(out), wpr=True, wpr_profile=None, wpr_seconds=None, timeout=None))

    assert code == 0
    report = json.loads((out / "diagnose.json").read_text(encoding="utf-8"))
    assert report["wpr"]["status"] == "skipped"
    assert "skipped" in capsys.readouterr().out


@pytest.mark.parametrize("status,phase", [("failed", "start"), ("timeout", "stop")])
def test_diagnose_still_writes_the_report_when_wpr_fails(tmp_path, monkeypatch, status, phase):
    monkeypatch.setattr(
        "hermes_cli.diagnostics_diagnose.collect_process_tree",
        lambda **_k: [{"pid": 7, "ppid": 1, "name": "hermes"}],
    )
    monkeypatch.setattr(
        "hermes_cli.diagnostics_diagnose.run_wpr_trace",
        lambda _dir, **_k: {"status": status, "phase": phase, "reason": "boom"},
    )

    out = tmp_path / "report"
    assert run_diagnose(ns(out=str(out), wpr=True, wpr_profile=None, wpr_seconds=None, timeout=None)) == 0

    report = json.loads((out / "diagnose.json").read_text(encoding="utf-8"))
    assert report["wpr"]["status"] == status
    assert report["process_tree"]


def test_debug_diagnose_parser_is_registered():
    from hermes_cli.subcommands.debug import build_debug_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    handler = object()
    build_debug_parser(subparsers, cmd_debug=handler)

    parsed = parser.parse_args(["debug", "diagnose", "--wpr", "--wpr-seconds", "5"])

    assert parsed.func is handler
    assert parsed.debug_command == "diagnose"
    assert parsed.wpr is True
    assert parsed.wpr_seconds == 5
    # Default is opt-OUT: a bare diagnose records no ETL.
    assert parser.parse_args(["debug", "diagnose"]).wpr is False
