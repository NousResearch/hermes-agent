"""Behavioral tests for the MiniMax Code CLI skill wrapper."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "mcode-cli"
    / "scripts"
    / "run_mcode.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mcode_skill_wrapper", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _success_result(answer: str = "done") -> dict:
    return {
        "schemaVersion": 1,
        "type": "exec.result",
        "sessionId": "session-1",
        "turnId": "turn-1",
        "status": "succeeded",
        "answer": answer,
        "durationMs": 12,
    }


def test_build_command_keeps_prompt_out_of_argv(tmp_path):
    wrapper = _load_module()

    command = wrapper.build_command(
        mcode="mcode",
        cwd=tmp_path,
        permission="smart",
        model="minimax/MiniMax-M2.5",
        session="session-1",
        continue_session=False,
        timeout="2m",
        max_steps=8,
    )

    assert command == [
        "mcode",
        "exec",
        "--input",
        "-",
        "--input-format",
        "json",
        "--output-format",
        "json",
        "--cwd",
        str(tmp_path),
        "--permission",
        "smart",
        "--model",
        "minimax/MiniMax-M2.5",
        "--session",
        "session-1",
        "--timeout",
        "2m",
        "--max-steps",
        "8",
    ]


def test_run_passes_prompt_over_stdin_and_emits_valid_result(tmp_path):
    wrapper = _load_module()
    captured = {}

    def fake_runner(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command, 0, json.dumps(_success_result("fixed")), ""
        )

    stdout: list[str] = []
    stderr: list[str] = []
    exit_code = wrapper.run_mcode(
        prompt="修复失败的测试；不要执行 $(touch should-not-run)",
        cwd=tmp_path,
        runner=fake_runner,
        which=lambda _command: "/tools/mcode.cmd",
        stdout=stdout.append,
        stderr=stderr.append,
    )

    assert exit_code == 0
    assert captured["command"][0] == "/tools/mcode.cmd"
    assert captured["command"][-2:] == ["--permission", "smart"]
    assert json.loads(captured["input"]) == {
        "prompt": "修复失败的测试；不要执行 $(touch should-not-run)"
    }
    assert captured["text"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "strict"
    assert captured["capture_output"] is True
    assert json.loads(stdout[0]) == _success_result("fixed")
    assert stderr == []


@pytest.mark.parametrize(
    ("status", "returncode"),
    (("timeout", 6), ("limit_exceeded", 7), ("cancelled", 130)),
)
def test_run_accepts_all_exec_result_terminal_statuses(tmp_path, status, returncode):
    wrapper = _load_module()
    result = {**_success_result(), "status": status, "answer": None}

    def fake_runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, returncode, json.dumps(result), "")

    stdout: list[str] = []
    stderr: list[str] = []
    exit_code = wrapper.run_mcode(
        prompt="Keep working",
        cwd=tmp_path,
        runner=fake_runner,
        which=lambda _command: "/tools/mcode",
        stdout=stdout.append,
        stderr=stderr.append,
    )

    assert exit_code == returncode
    assert json.loads(stdout[0]) == result
    assert stderr == []


def test_run_rejects_exec_result_without_duration(tmp_path):
    wrapper = _load_module()
    result = _success_result()
    result.pop("durationMs")

    def fake_runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    stdout: list[str] = []
    stderr: list[str] = []
    exit_code = wrapper.run_mcode(
        prompt="Keep working",
        cwd=tmp_path,
        runner=fake_runner,
        which=lambda _command: "/tools/mcode",
        stdout=stdout.append,
        stderr=stderr.append,
    )

    assert exit_code == 1
    assert stdout == []
    assert stderr == ["mcode returned invalid ExecResultV1 JSON.\n"]


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_run_rejects_non_standard_json_constants(tmp_path, constant):
    wrapper = _load_module()
    raw_result = json.dumps(_success_result()).replace("12", constant, 1)

    def fake_runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, raw_result, "")

    stdout: list[str] = []
    stderr: list[str] = []
    exit_code = wrapper.run_mcode(
        prompt="Keep working",
        cwd=tmp_path,
        runner=fake_runner,
        which=lambda _command: "/tools/mcode",
        stdout=stdout.append,
        stderr=stderr.append,
    )

    assert exit_code == 1
    assert stdout == []
    assert stderr == ["mcode returned invalid ExecResultV1 JSON.\n"]


def test_run_preserves_mcode_failure_and_diagnostics(tmp_path):
    wrapper = _load_module()
    failed = {
        **_success_result(),
        "status": "failed",
        "answer": None,
        "error": {"category": "runtime", "message": "provider unavailable"},
    }

    def fake_runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 4, json.dumps(failed), "provider unavailable\n"
        )

    stdout: list[str] = []
    stderr: list[str] = []
    exit_code = wrapper.run_mcode(
        prompt="Fix it",
        cwd=tmp_path,
        runner=fake_runner,
        which=lambda _command: "/tools/mcode",
        stdout=stdout.append,
        stderr=stderr.append,
    )

    assert exit_code == 4
    assert json.loads(stdout[0]) == failed
    assert stderr == ["provider unavailable\n"]


def test_run_rejects_non_contract_stdout(tmp_path):
    wrapper = _load_module()

    def fake_runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, "not-json\n", "")

    stdout: list[str] = []
    stderr: list[str] = []
    exit_code = wrapper.run_mcode(
        prompt="Fix it",
        cwd=tmp_path,
        runner=fake_runner,
        which=lambda _command: "/tools/mcode",
        stdout=stdout.append,
        stderr=stderr.append,
    )

    assert exit_code == 1
    assert stdout == []
    assert stderr == ["mcode returned invalid ExecResultV1 JSON.\n"]


def test_run_reports_missing_mcode_binary(tmp_path):
    wrapper = _load_module()
    runner_called = False

    def fake_runner(_command, **_kwargs):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner must not start when mcode cannot be resolved")

    stderr: list[str] = []
    exit_code = wrapper.run_mcode(
        prompt="Fix it",
        cwd=tmp_path,
        runner=fake_runner,
        which=lambda _command: None,
        stdout=lambda _value: None,
        stderr=stderr.append,
    )

    assert exit_code == 127
    assert runner_called is False
    assert stderr == ["mcode was not found on PATH. Install @minimax-ai/code first.\n"]


def test_main_reads_prompt_file_and_resolves_workspace(tmp_path, monkeypatch):
    wrapper = _load_module()
    prompt_file = tmp_path / "task.md"
    prompt_file.write_text("Fix the focused test.\n", encoding="utf-8")
    captured = {}

    def fake_run_mcode(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(wrapper, "run_mcode", fake_run_mcode)

    exit_code = wrapper.main([
        "--cwd",
        str(tmp_path),
        "--prompt-file",
        str(prompt_file),
        "--permission",
        "off",
        "--max-steps",
        "3",
    ])

    assert exit_code == 0
    assert captured["prompt"] == "Fix the focused test.\n"
    assert captured["cwd"] == tmp_path.resolve()
    assert captured["permission"] == "off"
    assert captured["max_steps"] == 3
