from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import provider_validate as pv


def test_parse_session_id_from_stdout_or_stderr():
    assert pv.parse_session_id("session_id: 20260520_abc", "") == "20260520_abc"
    assert pv.parse_session_id("", "noise\nsession_id: sid-123\n") == "sid-123"
    assert pv.parse_session_id("no session", "still none") is None


def test_extract_tool_calls_preserves_order_arguments_results_and_status():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/missing.txt"}',
                    },
                },
                {
                    "id": "call-2",
                    "function": {
                        "name": "search_files",
                        "arguments": {"path": "/tmp/tree"},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "tool_name": "read_file",
            "content": "Error: file missing",
            "effect_disposition": "error",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "tool_name": "search_files",
            "content": "found needle",
            "effect_disposition": "success",
        },
    ]

    receipts = pv.extract_tool_calls(messages)

    assert [receipt["name"] for receipt in receipts] == ["read_file", "search_files"]
    assert receipts[0]["arguments"] == {"path": "/tmp/missing.txt"}
    assert receipts[0]["result"] == "Error: file missing"
    assert receipts[0]["status"] == "error"
    assert receipts[1]["arguments"] == {"path": "/tmp/tree"}
    assert receipts[1]["result"] == "found needle"
    assert receipts[1]["status"] == "success"


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
    session_path = tmp_path / "session.json"
    session_path.write_text("[]", encoding="utf-8")

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
        session_path=session_path,
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
    assert "session_receipt_loaded" in result.failure_reasons


def test_stdout_alone_cannot_pass_without_loaded_session_receipt(tmp_path: Path):
    case = pv.ValidationCase(
        case_id="stdout_only",
        prompt="prompt",
        expected_text="PRINTED_OK",
    )

    result = pv.score_case(
        case,
        returncode=0,
        stdout="session_id: sid\nPRINTED_OK\n",
        stderr="",
        session_id="sid",
        messages=[],
        elapsed_s=0.1,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        session_path=None,
    )

    assert not result.ok
    assert result.final_text == ""
    assert result.checks["session_receipt_loaded"] is False
    assert result.checks["expected_text_found"] is False


def test_failed_read_recovery_requires_expected_paths_and_order(tmp_path: Path):
    missing = tmp_path / "missing.txt"
    recovered = tmp_path / "recovered.txt"
    case = pv.ValidationCase(
        case_id="recovery",
        prompt="prompt",
        expected_text="RECOVERY_OK",
        expected_tool_sequence=(
            ("read_file", str(missing), "error"),
            ("read_file", str(recovered), "success"),
        ),
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": str(missing)},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "tool_name": "read_file",
            "content": "not found",
            "status": "error",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-2",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": str(recovered)},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "tool_name": "read_file",
            "content": "recovered",
            "status": "success",
        },
        {"role": "assistant", "content": "RECOVERY_OK"},
    ]
    session_path = tmp_path / "session.json"
    session_path.write_text("[]", encoding="utf-8")

    result = pv.score_case(
        case,
        returncode=0,
        stdout="",
        stderr="",
        session_id="sid",
        messages=messages,
        elapsed_s=0.1,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        session_path=session_path,
    )

    assert result.ok
    assert result.checks["expected_tool_sequence"] is True

    reversed_result = pv.score_case(
        case,
        returncode=0,
        stdout="",
        stderr="",
        session_id="sid",
        messages=[messages[2], messages[3], messages[0], messages[1], messages[4]],
        elapsed_s=0.1,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        session_path=session_path,
    )
    assert not reversed_result.checks["expected_tool_sequence"]


def test_forbidden_artifact_must_be_absent(tmp_path: Path):
    artifact = tmp_path / "should_not_exist.txt"
    case = pv.ValidationCase(
        case_id="abstention",
        prompt="prompt",
        expected_text="BLOCKED",
        forbidden_artifacts=(str(artifact),),
    )
    session_path = tmp_path / "session.json"
    session_path.write_text("[]", encoding="utf-8")
    kwargs = dict(
        returncode=0,
        stdout="",
        stderr="",
        session_id="sid",
        messages=[{"role": "assistant", "content": "BLOCKED"}],
        elapsed_s=0.1,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        session_path=session_path,
    )

    assert pv.score_case(case, **kwargs).ok
    artifact.write_text("unsafe", encoding="utf-8")
    result = pv.score_case(case, **kwargs)
    assert not result.checks["forbidden_artifacts_absent"]


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
    assert "--ignore-rules" not in cmd


def _validation_args(out: Path) -> SimpleNamespace:
    return SimpleNamespace(
        out=str(out),
        suite="agent-readiness",
        provider="custom:local",
        model="model-a",
        toolsets="file",
        timeout=5.0,
        hermes_executable="/bin/hermes",
    )


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
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"/tmp/read_marker.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "tool_name": "read_file",
                "content": "alpha-271",
                "status": "success",
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

    assert pv.run_validation(_validation_args(tmp_path)) == 0
    assert seen_cmds[0][:3] == ["/bin/hermes", "chat", "-Q"]
    assert "--ignore-rules" not in seen_cmds[0]
    assert (tmp_path / "results.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "raw" / "read_file_real_tool.stdout").read_text(
        encoding="utf-8"
    ).startswith("session_id: test-session")
    assert (tmp_path / "raw" / "read_file_real_tool.session.json").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "SCREEN-PASS"
    receipt = summary["results"][0]["tool_calls"][0]
    assert receipt["arguments"] == {"path": "/tmp/read_marker.txt"}
    assert receipt["result"] == "alpha-271"
    assert receipt["status"] == "success"


def test_run_validation_preserves_timeout_evidence_and_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    case = pv.ValidationCase("timeout", "prompt", "PRINTED_OK")
    monkeypatch.setattr(pv, "get_suite_cases", lambda suite, fixture_dir: [case])

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd, kwargs["timeout"], output="partial stdout", stderr="partial stderr"
        )

    monkeypatch.setattr(pv.subprocess, "run", fake_run)

    assert pv.run_validation(_validation_args(tmp_path)) == 1
    assert "partial stdout" in (tmp_path / "raw" / "timeout.stdout").read_text()
    stderr = (tmp_path / "raw" / "timeout.stderr").read_text()
    assert "partial stderr" in stderr
    assert "Timed out" in stderr
    assert (tmp_path / "raw" / "timeout.session-error.txt").is_file()
    result = json.loads((tmp_path / "summary.json").read_text())["results"][0]
    assert result["timed_out"] is True
    assert result["ok"] is False
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "GATE-FAILED"


def test_run_validation_rejects_printed_output_with_invalid_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    case = pv.ValidationCase("invalid_session", "prompt", "PRINTED_OK")
    monkeypatch.setattr(pv, "get_suite_cases", lambda suite, fixture_dir: [case])
    monkeypatch.setattr(
        pv.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="PRINTED_OK\n",
            stderr="provider completed without a session receipt\n",
        ),
    )

    assert pv.run_validation(_validation_args(tmp_path)) == 1
    assert (tmp_path / "raw" / "invalid_session.stdout").read_text().startswith(
        "PRINTED_OK"
    )
    assert (
        "session_id not found"
        in (tmp_path / "raw" / "invalid_session.session-error.txt").read_text()
    )
    result = json.loads((tmp_path / "summary.json").read_text())["results"][0]
    assert result["final_text"] == ""
    assert result["ok"] is False
    assert result["checks"]["session_receipt_loaded"] is False
