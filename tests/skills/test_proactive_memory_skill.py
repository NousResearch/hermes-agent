"""proactive-memory: memory_bank.py's relevance gate returns only the entries whose trigger is in
the proposed action (selective beats always-inject), supersede/cap keep the bank compact, and
ungrounded input is rejected. Runs the real script as a subprocess, offline."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "optional-skills/autonomous-ai-agents/proactive-memory/scripts/memory_bank.py")


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, timeout=30)


def _seed(bank, entries):
    for kind, text, trigger in entries:
        proc = _run("record", bank, "--kind", kind, "--text", text, "--trigger", trigger)
        assert proc.returncode == 0, proc.stderr


@pytest.fixture
def bank(tmp_path):
    path = tmp_path / "nested" / "bank.jsonl"
    _seed(path, [
        ("procedural", "uv sync fails behind the proxy unless UV_NATIVE_TLS=1", "uv sync,uv pip"),
        ("status", "section 4 (DPIA) still open", "dpia,section 4"),
        ("knowledge", "NHS toolkit requires Node v18", "node,npm"),
    ])
    return path


def test_gate_returns_only_action_relevant_entries(bank):
    relevant = json.loads(_run("check", bank, "--action", "run uv sync to install deps").stdout)
    unrelated = json.loads(_run("check", bank, "--action", "git status").stdout)
    total = json.loads(_run("list", bank).stdout)["count"]

    assert [r["id"] for r in relevant["reminders"]] == ["m1"]
    assert relevant["reminders"][0]["matched_triggers"] == ["uv sync"]
    # The gate is a strict subset of the bank, and an unrelated action surfaces nothing —
    # the contract that makes selective intervention different from re-injecting everything.
    assert 0 < relevant["count"] < total
    assert unrelated["count"] == 0


def test_procedural_reminder_leads_regardless_of_insertion_order(tmp_path):
    path = tmp_path / "order.jsonl"
    _seed(path, [("knowledge", "k", "deploy"), ("status", "s", "deploy"),
                 ("procedural", "p", "deploy")])
    kinds = [r["kind"] for r in json.loads(_run("check", path, "--action", "deploy now").stdout)["reminders"]]
    assert kinds == ["procedural", "status", "knowledge"]


def test_supersede_retires_the_old_entry(bank):
    superseded = _run("record", bank, "--kind", "status",
                      "--text", "section 4 (DPIA) complete", "--trigger", "dpia,section 4",
                      "--supersedes", "m2")
    assert superseded.returncode == 0, superseded.stderr
    reminders = json.loads(_run("check", bank, "--action", "open section 4").stdout)["reminders"]
    texts = [r["text"] for r in reminders]
    assert "section 4 (DPIA) complete" in texts
    assert "section 4 (DPIA) still open" not in texts


def test_bank_stays_capped_per_kind(tmp_path):
    path = tmp_path / "cap.jsonl"
    for i in range(10):
        assert _run("record", path, "--kind", "procedural", "--text", f"avoid mistake {i}",
                    "--trigger", f"cmd{i}", "--cap", "8").returncode == 0
    active = json.loads(_run("list", path, "--kind", "procedural").stdout)
    assert active["count"] == 8
    # Newest wins: the two oldest (cmd0, cmd1) are retired, cmd9 survives.
    surviving = {t for e in active["entries"] for t in e["triggers"]}
    assert "cmd0" not in surviving and "cmd9" in surviving


@pytest.mark.parametrize("args", [
    ("record", "--kind", "status", "--text", "   ", "--trigger", "x"),
    ("record", "--kind", "status", "--text", "grounded?", "--trigger", " , ,"),
    ("record", "--kind", "status", "--text", "t", "--trigger", "x", "--supersedes", "m99"),
    ("check", "--action", "   "),
])
def test_invalid_input_is_rejected_without_writing(tmp_path, args):
    path = tmp_path / "rej.jsonl"
    mode, rest = args[0], args[1:]
    proc = _run(mode, path, *rest)
    assert proc.returncode == 2 and proc.stderr.startswith("memory_bank:")
    assert not path.exists()
