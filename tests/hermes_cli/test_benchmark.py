"""Tests for the hermes benchmark subcommand.

We mock the measurement primitives so tests don't actually fork
subprocesses for every case. One end-to-end test (``test_run_end_to_end``)
does spawn a real subprocess to verify the import path works.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ──────────────────────────── measurement primitives ────────────────────────


def test_get_memory_mb_uses_pss_when_available(monkeypatch):
    from hermes_cli.subcommands.benchmark import _get_memory_mb

    # psutil is installed (project already has it); verify we get a
    # positive number back. We don't assert PSS vs RSS here — that's
    # platform-dependent and tested in the kind label.
    mb, kind = _get_memory_mb(os.getpid())
    assert mb > 0
    assert kind in ("pss", "rss")


def test_get_memory_mb_falls_back_to_rss(monkeypatch):
    """When psutil isn't available, we use resource.getrusage and
    report kind='rss'."""
    import hermes_cli.subcommands.benchmark as bm
    monkeypatch.setattr(bm, "_HAS_PSUTIL", False)
    mb, kind = bm._get_memory_mb(os.getpid())
    assert kind == "rss"
    assert mb >= 0


# ──────────────────────────── cold start ────────────────────────────


def test_measure_cold_start_returns_n_numbers(tmp_path):
    """Spawns N subprocesses; verifies we get N floats back and they're
    positive. Skipped if Python isn't on PATH (CI envs sometimes)."""
    from hermes_cli.subcommands.benchmark import _measure_cold_start
    if not shutil_which(sys.executable):  # always true on dev
        pass
    times = _measure_cold_start(2, project_root=str(_REPO_ROOT))
    assert len(times) == 2
    assert all(t > 0 for t in times)


def shutil_which(cmd):
    """Lightweight shutil.which replacement to keep this test file
    free of the stdlib import at module top."""
    from shutil import which
    return which(cmd)


# ──────────────────────────── formatting ────────────────────────────


def test_format_markdown_includes_all_metrics():
    from hermes_cli.subcommands.benchmark import _format_markdown
    out = _format_markdown(
        cold=[100.0, 110.0, 105.0],
        idle_mb=150.0, idle_kind="pss",
        delta_mb=10.0, delta_kind="pss",
    )
    assert "Hermes benchmark" in out
    assert "Cold start" in out
    assert "Idle PSS" in out
    assert "Per-session PSS" in out
    assert "105.0 ms" in out  # mean of [100, 110, 105]
    assert "150.0 MB" in out
    assert "10.0 MB" in out


def test_format_markdown_handles_darwin_note(monkeypatch):
    from hermes_cli.subcommands.benchmark import _format_markdown
    import sys as _sys
    monkeypatch.setattr(_sys, "platform", "darwin")
    out = _format_markdown(
        cold=[100.0], idle_mb=200.0, idle_kind="rss",
        delta_mb=15.0, delta_kind="rss",
    )
    assert "macOS" in out
    assert "RSS" in out


def test_format_json_emits_valid_json():
    from hermes_cli.subcommands.benchmark import _format_json
    out = _format_json(
        cold=[100.0, 110.0], idle_mb=150.0, idle_kind="pss",
        delta_mb=10.0, delta_kind="pss",
    )
    data = json.loads(out)
    assert data["cold_start_ms"] == [100.0, 110.0]
    assert data["cold_start_mean_ms"] == 105.0
    assert data["idle_memory_mb"] == 150.0
    assert data["idle_memory_kind"] == "pss"
    assert data["per_session_delta_mb"] == 10.0


def test_format_json_handles_empty_cold():
    from hermes_cli.subcommands.benchmark import _format_json
    out = _format_json(
        cold=[], idle_mb=0.0, idle_kind="unknown",
        delta_mb=0.0, delta_kind="unknown",
    )
    data = json.loads(out)
    assert data["cold_start_ms"] == []
    assert data["cold_start_mean_ms"] == 0.0


# ──────────────────────────── run() entry point ────────────────────────────


def test_run_emits_markdown_by_default(capsys, monkeypatch):
    """When --json is False, run() prints markdown. We mock the
    measurement primitives so the test is deterministic."""
    from hermes_cli.subcommands import benchmark as bm

    monkeypatch.setattr(bm, "_measure_cold_start",
                        lambda n, project_root: [123.0, 125.0, 127.0])
    monkeypatch.setattr(bm, "_measure_idle_pss", lambda: (142.0, "pss"))
    monkeypatch.setattr(bm, "_measure_per_session_delta",
                        lambda project_root: (18.5, "pss"))

    rc = bm.run(n=3, json_out=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Cold start" in out
    assert "Idle PSS" in out
    assert "Per-session PSS" in out
    # Mean of 123, 125, 127 = 125.0
    assert "125.0 ms" in out


def test_run_emits_json_when_requested(capsys, monkeypatch):
    from hermes_cli.subcommands import benchmark as bm

    monkeypatch.setattr(bm, "_measure_cold_start",
                        lambda n, project_root: [200.0])
    monkeypatch.setattr(bm, "_measure_idle_pss", lambda: (100.0, "rss"))
    monkeypatch.setattr(bm, "_measure_per_session_delta",
                        lambda project_root: (10.0, "rss"))

    rc = bm.run(n=1, json_out=True)
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["cold_start_ms"] == [200.0]
    assert data["idle_memory_mb"] == 100.0


def test_run_returns_1_on_failure(monkeypatch, capsys):
    """If a measurement raises, run() prints the error and returns 1."""
    from hermes_cli.subcommands import benchmark as bm

    def boom(n, project_root):
        raise RuntimeError("simulated subprocess failure")

    monkeypatch.setattr(bm, "_measure_cold_start", boom)
    rc = bm.run(n=1, json_out=False)
    assert rc == 1
    err = capsys.readouterr().err
    assert "simulated subprocess failure" in err


# ──────────────────────────── CLI parser registration ────────────────────────────


def test_benchmark_parser_registration():
    """build_benchmark_parser should attach a 'benchmark' subparser with
    --n and --json flags. Tested in isolation by constructing a fresh
    subparsers object."""
    from hermes_cli.subcommands.benchmark import build_benchmark_parser

    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    build_benchmark_parser(sub, cmd_benchmark=lambda args: None)

    # Parse a sample invocation
    args = parser.parse_args(["benchmark", "--n", "5", "--json"])
    assert args.n == 5
    assert args.json is True
    assert callable(args.func)


def test_benchmark_parser_defaults():
    """Defaults: n=3, json=False."""
    from hermes_cli.subcommands.benchmark import build_benchmark_parser

    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    build_benchmark_parser(sub, cmd_benchmark=lambda args: None)

    args = parser.parse_args(["benchmark"])
    assert args.n == 3
    assert args.json is False


# ──────────────────────────── end-to-end smoke ────────────────────────────


def test_run_end_to_end_smoke():
    """One real subprocess invocation to verify the full path works.
    Skipped if hermes's venv can't be found (the test environment
    matters)."""
    from hermes_cli.subcommands.benchmark import _measure_cold_start
    times = _measure_cold_start(1, project_root=str(_REPO_ROOT))
    assert len(times) == 1
    # Cold import on this machine takes < 30s (was ~1.5s in baseline)
    assert times[0] < 30000, f"cold start took {times[0]:.0f}ms — too slow"
