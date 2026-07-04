import importlib.util
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relay_call = load_module("relay_call", "relay-call.py")
gate_run = load_module("gate_run", "gate-run.py")


def test_classify_exit0_long_stdout_mentions_login_is_ok():
    stdout = "หน้า login ต้องแสดงข้อความ please login และ credential invalid ให้ผู้ใช้เห็นอย่างถูกต้อง"

    assert relay_call.classify(0, stdout, "") == "ok"


def test_classify_exit0_short_stdout_with_stderr_login_is_auth():
    assert relay_call.classify(0, "", "please login") == "auth"


def test_classify_exit0_long_stdout_with_stderr_not_found_is_ok():
    stdout = "Codex completed the requested work and produced a normal detailed task summary."

    assert relay_call.classify(0, stdout, "rm: no such file or directory") == "ok"


def test_classify_nonzero_quota_not_found_and_crash():
    assert relay_call.classify(1, "", "429 rate limit") == "quota"
    assert relay_call.classify(127, "", "") == "not_found"
    assert relay_call.classify(1, "plain build log", "") == "crash"


def test_classify_nonzero_command_not_found_stays_not_found():
    assert relay_call.classify(1, "", "sh: foo: command not found") == "not_found"


def test_bump_counter_counts_expires_and_reads_legacy_number(tmp_path):
    assert relay_call.bump_counter(tmp_path, ".session-calls", session_hours=12) == 1
    assert relay_call.bump_counter(tmp_path, ".session-calls", session_hours=12) == 2

    old_started = time.time() - (13 * 3600)
    counter = tmp_path / ".hermes" / "ai-relay" / ".session-calls"
    counter.write_text(json.dumps({"count": 7, "started": old_started}), encoding="utf-8")
    assert relay_call.bump_counter(tmp_path, ".session-calls", session_hours=12) == 1

    legacy = tmp_path / ".hermes" / "ai-relay" / ".session-fable-calls"
    legacy.write_text("4", encoding="utf-8")
    assert relay_call.bump_counter(tmp_path, ".session-fable-calls", session_hours=12) == 5


def test_is_tool_missing_matches_only_gate_tools():
    py = "/repo/.venv/bin/python"

    assert gate_run.is_tool_missing("No module named pytest", [py, "-m", "pytest", "-q"]) is True
    assert (
        gate_run.is_tool_missing(
            "ModuleNotFoundError: No module named 'myapp'",
            [py, "-m", "pytest", "-q"],
        )
        is False
    )
    assert gate_run.is_tool_missing("npm: command not found", ["npm", "run", "test"]) is True
    assert (
        gate_run.is_tool_missing(
            "ENOENT: no such file or directory, open 'x.json'",
            ["npm", "run", "test"],
        )
        is False
    )


def test_repo_python_prefers_dot_venv_when_both_exist(tmp_path):
    dot_py = tmp_path / ".venv" / "bin" / "python"
    dot_py.parent.mkdir(parents=True)
    dot_py.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(dot_py, 0o755)

    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(py, 0o755)

    assert gate_run.repo_python(tmp_path) == str(dot_py)


def test_detect_gate_prefers_repo_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(py, 0o755)

    cmd, label = gate_run.detect_gate(tmp_path)

    assert cmd == [str(py), "-m", "pytest", "-q"]
    assert label == "pytest -q"
