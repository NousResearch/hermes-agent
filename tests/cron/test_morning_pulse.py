"""Tests for scripts/morning_pulse.py — the consolidated fleet health report.

Covers the output contract (Sahil standard): colour-coded compact message,
verdict header, MEDIA tag last, dark HTML with attention-first sections.
Runs the script in-process with monkeypatched check functions so the test
never depends on live system state.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path.home() / ".hermes" / "scripts" / "morning_pulse.py"


def load_module():
    spec = importlib.util.spec_from_file_location("morning_pulse", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mp(tmp_path, monkeypatch):
    """Module with reports dir + clock pinned to a temp HERMES_HOME."""
    mod = load_module()
    monkeypatch.setattr(mod, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(mod, "CHECKS", [])
    return mod


def _run_main(monkeypatch, mp, checks):
    """Patch the 8 check_* functions to record fixed results, run main()."""
    def mk(fam, st, det):
        return lambda: checks.append((fam, st, det))
    fixed = [
        ("memory", "ok", "RAM 16% · swap 0%"),
        ("disk", "ok", "/ 61% · /home 32% free"),
        ("gateway", "ok", "up (2 processes)"),
        ("mcp", "ok", "all MCP servers healthy"),
        ("discord", "ok", "bots responding"),
        ("backup", "ok", "3h old (kensei-x.tar.gz)"),
        ("cron", "warn", "1 job(s) errored: some-job"),
        ("config", "crit", "~/.hermes git broken"),
    ]
    for name in ("check_memory", "check_disk", "check_gateway", "check_backup",
                 "check_cron", "check_config_drift"):
        monkeypatch.setattr(mp, name, lambda: None)
    # patch run_checker invocations by faking subprocess results
    class R:
        stdout = ""
        stderr = ""
        returncode = 0
    monkeypatch.setattr(mp.subprocess, "run",
                        lambda *a, **k: R(), raising=True)
    monkeypatch.setattr(mp.sys, "stdout", open("/dev/null", "w"))
    captured = {}

    def fake_write(s):
        captured["out"] = captured.get("out", "") + s

    class Std:
        def write(self, s):
            fake_write(s)
        def flush(self):
            pass
    monkeypatch.setattr(mp.sys, "stdout", Std())
    # seed CHECKS directly (main() calls check fns; they're no-ops now)
    for fam, st, det in fixed:
        mp.CHECKS.append((fam, st, det))
    rc = mp.main()
    return rc, captured.get("out", "")


def test_output_contract_all_sections(mp, monkeypatch):
    rc, out = _run_main(monkeypatch, mp, mp.CHECKS)
    assert rc == 0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    # header with verdict
    assert lines[0].startswith("🌅 Morning Pulse — ")
    assert "1 CRITICAL" in lines[0]
    # warn+crit lines present with correct icons
    assert any(l.startswith("🟡 cron:") for l in lines)
    assert any(l.startswith("🔴 config:") for l in lines)
    # pointer line + MEDIA last
    assert lines[-2] == "📊 Full report attached"
    assert lines[-1].startswith("MEDIA:/")
    # compact: ≤12 lines total
    assert len(lines) <= 12


def test_html_report_written_and_dark(mp, monkeypatch):
    rc, out = _run_main(monkeypatch, mp, mp.CHECKS)
    media_line = [l for l in out.splitlines() if l.startswith("MEDIA:")][0]
    path = Path(media_line[len("MEDIA:"):])
    html = path.read_text()
    assert path.stat().st_size > 0
    assert "color-scheme: dark" in html
    assert "#ef4444" in html and "#f59e0b" in html and "#22c55e" in html
    assert "Needs attention" in html
    assert "All checks" in html
    # crit item surfaces in the attention section before the full list
    attn = html.split("Needs attention")[1].split("All checks")[0]
    assert "git broken" in attn


def test_all_green_is_compact(mp, monkeypatch):
    for name in ("check_memory", "check_disk", "check_gateway", "check_backup",
                 "check_cron", "check_config_drift"):
        monkeypatch.setattr(mp, name, lambda: None)

    class R:
        stdout = ""
        stderr = ""
        returncode = 0
    monkeypatch.setattr(mp.subprocess, "run", lambda *a, **k: R())

    class Std:
        def __init__(self):
            self.buf = ""
        def write(self, s):
            self.buf += s
        def flush(self):
            pass
    std = Std()
    monkeypatch.setattr(mp.sys, "stdout", std)
    mp.CHECKS.clear()
    for fam in ("memory", "disk", "gateway", "mcp", "discord", "backup",
                "cron", "config"):
        mp.CHECKS.append((fam, "ok", "fine"))
    rc = mp.main()
    lines = [l for l in std.buf.splitlines() if l.strip()]
    assert "ALL GREEN" in lines[0]
    assert sum(1 for l in lines if l.startswith("🟢")) == 1  # single grouped line
    assert len(lines) <= 5
