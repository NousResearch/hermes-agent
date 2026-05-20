from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import provider_validate as pv


def test_parse_session_id_from_stdout_or_stderr():
    assert pv.parse_session_id("session_id: 20260520_abc", "") == "20260520_abc"
    assert pv.parse_session_id("", "noise\nsession_id: sid-123\n") == "sid-123"
    assert pv.parse_session_id("no session", "still none") is None


def test_extract_tool_calls_supports_assistant_and_tool_rows():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": "{}"}},
                {"name": "search_files"},
            ],
        },
        {"role": "tool", "tool_name": "read_file", "content": "ok"},
    ]

    assert pv.extract_tool_calls(messages) == [
        "read_file",
        "search_files",
        "read_file",
    ]


def test_score_visible_reasoning_only_checks_user_visible_text(tmp_path: Path):
    case = pv.ValidationCase(
        case_id="visible_reasoning_leak",
        prompt="prompt",
        expected_text="VISIBLE_REASONING_CLEAN",
        expect_no_tools=True,
    )
    messages = [
        {
            "role": "assistant",
            "content": "VISIBLE_REASONING_CLEAN",
            "reasoning_content": "<think>internal diagnostic reasoning is allowed here</think>",
        }
    ]

    result = pv.score_case(
        case,
        returncode=0,
        stdout="session_id: sid\nVISIBLE_REASONING_CLEAN\n",
        stderr="",
        session_id="sid",
        messages=messages,
        elapsed_s=0.1,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        session_path=tmp_path / "session.json",
    )

    assert result.ok
    assert result.checks["visible_reasoning_clean"] is True


def test_score_fails_visible_think_tags(tmp_path: Path):
    case = pv.ValidationCase(
        case_id="visible_reasoning_leak",
        prompt="prompt",
        expected_text="VISIBLE_REASONING_CLEAN",
    )
    messages = [
        {
            "role": "assistant",
            "content": "<think>scratchpad</think> VISIBLE_REASONING_CLEAN",
        }
    ]

    result = pv.score_case(
        case,
        returncode=0,
        stdout="session_id: sid\n",
        stderr="",
        session_id="sid",
        messages=messages,
        elapsed_s=0.1,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        session_path=None,
    )

    assert not result.ok
    assert "visible_reasoning_clean" in result.failure_reasons


def test_build_chat_command_uses_real_hermes_quiet_session_receipts():
    cmd = pv.build_chat_command(
        provider="custom:local",
        model="test-model",
        toolsets="file",
        source="provider-validation:case",
        prompt="hello",
        hermes_executable="/bin/hermes",
    )

    assert cmd == [
        "/bin/hermes",
        "chat",
        "-Q",
        "--ignore-rules",
        "--source",
        "provider-validation:case",
        "--toolsets",
        "file",
        "--provider",
        "custom:local",
        "--model",
        "test-model",
        "-q",
        "hello",
    ]


def test_run_validation_writes_receipts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    case = pv.ValidationCase(
        case_id="read_file_real_tool",
        prompt="read fixture",
        expected_text="alpha-271",
        required_tools=("read_file",),
    )
    monkeypatch.setattr(pv, "get_suite_cases", lambda suite, fixture_dir: [case])
    monkeypatch.setattr(
        pv,
        "load_session_messages",
        lambda session_id: [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file"}}],
            },
            {"role": "assistant", "content": "alpha-271"},
        ],
    )

    seen_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen_cmds.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="session_id: test-session\nalpha-271\n",
            stderr="",
        )

    monkeypatch.setattr(pv.subprocess, "run", fake_run)

    args = SimpleNamespace(
        out=str(tmp_path),
        suite="agent-readiness",
        provider="custom:local",
        model="model-a",
        toolsets="file",
        timeout=5.0,
        hermes_executable="/bin/hermes",
    )

    assert pv.run_validation(args) == 0
    assert seen_cmds[0][:3] == ["/bin/hermes", "chat", "-Q"]
    assert (tmp_path / "results.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "raw" / "read_file_real_tool.stdout").read_text(
        encoding="utf-8"
    ).startswith("session_id: test-session")
    assert (tmp_path / "raw" / "read_file_real_tool.session.json").is_file()
