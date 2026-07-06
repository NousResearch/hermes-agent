from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_compact_finalization_prompt_uses_packet_fields_without_raw_history():
    from hermes_cli.finalization_mode import build_compact_finalization_prompt

    prompt = build_compact_finalization_prompt(
        {
            "session_id": "root-session",
            "latest_session_id": "child-session",
            "task_contract": "Finish Task 5 compact finalization mode.",
            "verified_artifacts": [
                "tests/hermes_cli/test_closeout_resume_state.py passed",
                "hermes_cli/finalization_mode.py reviewed",
            ],
            "tests_run": [
                "python -m pytest tests/hermes_cli/test_closeout_resume_state.py -q"
            ],
            "blockers": ["queued steer is waiting behind an active model request"],
            "next_safe_action": "Return a compact final or blocked answer in the Hermes app.",
            "required_model": "gpt-5.5",
            "raw_logs": "RAW_LOG_SENTINEL should never appear",
            "queued_prompt_text": "QUEUED_PROMPT_SENTINEL should never appear",
            "transcript_excerpt": "TRANSCRIPT_SENTINEL should never appear",
        }
    )

    assert "root-session" in prompt
    assert "child-session" in prompt
    assert "Finish Task 5 compact finalization mode." in prompt
    assert "tests/hermes_cli/test_closeout_resume_state.py passed" in prompt
    assert "queued steer is waiting behind an active model request" in prompt
    assert "Return a compact final or blocked answer in the Hermes app." in prompt
    assert "gpt-5.5" in prompt
    assert "chat --query-file" in prompt
    assert "Do not replay broad history" in prompt
    assert "RAW_LOG_SENTINEL" not in prompt
    assert "QUEUED_PROMPT_SENTINEL" not in prompt
    assert "TRANSCRIPT_SENTINEL" not in prompt
    assert len(prompt) < 5000


def test_compact_finalization_prompt_carries_contract_checklist_guidance():
    from hermes_cli.finalization_mode import build_compact_finalization_prompt

    prompt = build_compact_finalization_prompt(
        {
            "session_id": "root-session",
            "task_contract": "Requirements:\n1. Update closeout classifier.\n2. Run focused tests.",
            "contract_checklist": [
                {
                    "requirement": "Update closeout classifier",
                    "status": "done",
                    "evidence": "hermes_cli/closeout_state.py",
                    "residual_risk": "none",
                    "next_action": "",
                }
            ],
        }
    )

    assert "Requirement checklist" in prompt
    assert "Update closeout classifier" in prompt
    assert "status: done" in prompt
    assert "evidence: hermes_cli/closeout_state.py" in prompt
    assert "partial, blocked, or not_started" in prompt
    assert "Simple tasks" in prompt


def test_write_closeout_state_extends_existing_packet_with_finalization_fields(hermes_home):
    from hermes_cli.closeout_state import write_closeout_state
    from hermes_cli.closure_artifacts import read_closure_artifact

    path = write_closeout_state(
        session_id="root-session",
        latest_session_id="child-session",
        task_id="task-5",
        task_contract="Produce final app answer from compact state.",
        verified_artifacts=["focused pytest green"],
        tests_run=["pytest closeout/finalization -q"],
        blockers=["post-main CI not checked"],
        next_safe_action="Give blocked answer with exact next verification.",
        required_model="gpt-5.5",
        final_response="CI not checked.",
        remaining_closeout_tasks=["watch post-main CI"],
    )

    data = read_closure_artifact(path)

    assert data["task_contract"] == "Produce final app answer from compact state."
    assert data["verified_artifacts"] == ["focused pytest green"]
    assert data["blockers"] == ["post-main CI not checked"]
    assert data["next_safe_action"] == "Give blocked answer with exact next verification."
    assert data["required_model"] == "gpt-5.5"
    assert "compact_finalization_prompt" in data
    assert "Produce final app answer from compact state." in data["compact_finalization_prompt"]


def test_runtime_recovery_finalization_prompt_writes_file(hermes_home, tmp_path, capsys):
    from hermes_cli.closeout_state import write_closeout_state
    from hermes_cli.runtime_cli import _cmd_recovery_finalization_prompt

    write_closeout_state(
        session_id="root-session",
        latest_session_id="child-session",
        task_id="task-5",
        task_contract="Finish from compact packet.",
        verified_artifacts=["green focused tests"],
        next_safe_action="Answer in Hermes app.",
        required_model="gpt-5.5",
    )
    out = tmp_path / "finalization.prompt.txt"

    rc = _cmd_recovery_finalization_prompt(
        SimpleNamespace(session="root-session", task_id=None, out=str(out))
    )
    stdout = capsys.readouterr().out

    assert rc == 0
    assert str(out) in stdout
    content = out.read_text(encoding="utf-8")
    assert "Finish from compact packet." in content
    assert "child-session" in content
    assert "gpt-5.5" in content


def test_runtime_parser_registers_recovery_finalization_prompt():
    from hermes_cli import runtime_cli

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    runtime_cli.build_parser(subparsers)

    args = parser.parse_args(
        [
            "runtime",
            "recovery",
            "finalization-prompt",
            "--session",
            "root-session",
            "--out",
            "prompt.txt",
        ]
    )

    assert args.func is runtime_cli._cmd_recovery_finalization_prompt
    assert args.session == "root-session"
    assert args.out == "prompt.txt"
