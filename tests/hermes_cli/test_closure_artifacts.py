from __future__ import annotations

import json
import argparse
from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_max_iteration_exit_writes_closure_artifact(hermes_home):
    from hermes_cli.closure_artifacts import (
        latest_closure_artifact,
        read_closure_artifact,
        write_closure_artifact,
    )

    path = write_closure_artifact(
        session_id="session-a",
        task_id="t_card",
        status="max_iterations_reached",
        last_completed_step="patched one test",
        changed_files=["hermes_cli/closure_artifacts.py"],
        tests_run=["pytest tests/hermes_cli/test_closure_artifacts.py"],
        test_results={"focused": "failed before implementation"},
        failing_tests=["test_resume_uses_compact_artifact"],
        remaining_checklist=["wire CLI command"],
        exact_resume_prompt="Resume from the compact closure packet only.",
        active_session_lease_released=True,
    )

    data = read_closure_artifact(path)

    assert path.exists()
    assert data["schema_version"] == 1
    assert data["session_id"] == "session-a"
    assert data["task_id"] == "t_card"
    assert data["status"] == "max_iterations_reached"
    assert data["last_completed_step"] == "patched one test"
    assert data["changed_files"] == ["hermes_cli/closure_artifacts.py"]
    assert data["tests_run"] == ["pytest tests/hermes_cli/test_closure_artifacts.py"]
    assert data["test_results"]["focused"] == "failed before implementation"
    assert data["failing_tests"] == ["test_resume_uses_compact_artifact"]
    assert data["remaining_checklist"] == ["wire CLI command"]
    assert data["active_session_lease_released"] is True
    assert data["artifact_path"] == str(path)

    latest = latest_closure_artifact(session_id="session-a")
    assert latest is not None
    assert latest["artifact_path"] == str(path)


def test_closure_artifact_excludes_secrets(hermes_home):
    from hermes_cli.closure_artifacts import read_closure_artifact, write_closure_artifact

    path = write_closure_artifact(
        session_id="session-secret",
        task_id="t_secret",
        status="max_iterations_reached",
        last_completed_step="provider returned sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        changed_files=["app.py"],
        tests_run=["TOKEN=sk-proj-abcdefghijklmnopqrstuvwxyz123456 pytest"],
        test_results={"raw": "api_key=sk-proj-abcdefghijklmnopqrstuvwxyz123456"},
        failing_tests=[],
        remaining_checklist=["remove token=sk-proj-abcdefghijklmnopqrstuvwxyz123456"],
        exact_resume_prompt="Do not print sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        active_session_lease_released=False,
    )

    text = path.read_text(encoding="utf-8")
    data = read_closure_artifact(path)

    assert "sk-proj-" not in text
    assert "[REDACTED_SECRET]" in json.dumps(data)


def test_closure_artifact_paths_are_unique_for_same_millisecond(hermes_home):
    from hermes_cli.closure_artifacts import read_closure_artifact, write_closure_artifact

    first = write_closure_artifact(
        session_id="same-ms",
        task_id="task-a",
        status="max_iterations_reached",
        remaining_checklist=["first"],
        created_at=1234.567,
    )
    second = write_closure_artifact(
        session_id="same-ms",
        task_id="task-a",
        status="max_iterations_reached",
        remaining_checklist=["second"],
        created_at=1234.567,
    )

    assert first != second
    assert first.exists()
    assert second.exists()
    assert read_closure_artifact(first)["remaining_checklist"] == ["first"]
    assert read_closure_artifact(second)["remaining_checklist"] == ["second"]


def test_resume_uses_compact_artifact(hermes_home):
    from hermes_cli.closure_artifacts import (
        build_resume_prompt_from_artifact,
        read_closure_artifact,
        write_closure_artifact,
    )

    path = write_closure_artifact(
        session_id="session-resume",
        task_id="t_resume",
        status="max_iterations_reached",
        last_completed_step="created closure artifact",
        changed_files=["a.py"],
        tests_run=["pytest -q"],
        test_results={"pytest -q": "1 failed"},
        failing_tests=["test_x"],
        remaining_checklist=["fix test_x"],
        exact_resume_prompt="Fix test_x from a.py with the packet below.",
        active_session_lease_released=True,
    )

    prompt = build_resume_prompt_from_artifact(read_closure_artifact(path))

    assert "COMPACT MAX-ITERATION CLOSURE PACKET" in prompt
    assert "Do not load or replay the bloated parent history" in prompt
    assert "Fix test_x from a.py with the packet below." in prompt
    assert "t_resume" in prompt
    assert str(path) in prompt


def test_closure_cli_latest_and_command_registry(hermes_home, capsys):
    from hermes_cli.closure_artifacts import write_closure_artifact
    from hermes_cli.closure_cli import _cmd_closure_latest, build_closure_parser
    from hermes_cli.commands import resolve_command

    write_closure_artifact(
        session_id="session-cli",
        task_id="t_cli",
        status="max_iterations_reached",
        last_completed_step="step",
        changed_files=[],
        tests_run=[],
        test_results={},
        failing_tests=[],
        remaining_checklist=[],
        exact_resume_prompt="Resume exactly.",
        active_session_lease_released=True,
    )

    rc = _cmd_closure_latest(
        SimpleNamespace(session_id="session-cli", task_id=None, json=True, resume_prompt=False)
    )

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["session_id"] == "session-cli"
    assert resolve_command("closure") is not None

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_closure_parser(subparsers)
    args = parser.parse_args(["closure", "resume", "--session-id", "session-cli"])
    assert args.func(args) == 0
    resume_out = capsys.readouterr().out
    assert "COMPACT MAX-ITERATION CLOSURE PACKET" in resume_out
    assert "Do not load or replay the bloated parent history" in resume_out


def test_turn_finalizer_max_iteration_writes_closure_artifact(hermes_home):
    from agent.turn_finalizer import finalize_turn
    from hermes_cli.closure_artifacts import latest_closure_artifact

    class Budget:
        remaining = 0
        used = 1
        max_total = 1

    class Compressor:
        last_prompt_tokens = 0

    class Agent:
        max_iterations = 1
        iteration_budget = Budget()
        quiet_mode = True
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        session_id = "finalizer-session"
        session_input_tokens = 0
        session_output_tokens = 0
        session_cache_read_tokens = 0
        session_cache_write_tokens = 0
        session_reasoning_tokens = 0
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_estimated_cost_usd = 0
        session_cost_status = "unknown"
        session_cost_source = "test"
        context_compressor = Compressor()
        _turn_failed_file_mutations = {}
        _tool_guardrail_halt_decision = None
        _response_was_previewed = False
        _interrupt_message = None
        _skill_nudge_interval = 0
        _iters_since_skill = 0
        valid_tool_names = set()

        def _emit_status(self, *_args, **_kwargs):
            return None

        def _safe_print(self, *_args, **_kwargs):
            return None

        def _handle_max_iterations(self, _messages, _api_call_count):
            return "summary after budget"

        def _save_trajectory(self, *_args, **_kwargs):
            return None

        def _cleanup_task_resources(self, *_args, **_kwargs):
            return None

        def _drop_trailing_empty_response_scaffolding(self, *_args, **_kwargs):
            return None

        def _persist_session(self, *_args, **_kwargs):
            return None

        def _file_mutation_verifier_enabled(self):
            return False

        def _turn_completion_explainer_enabled(self):
            return False

        def _drain_pending_steer(self):
            return None

        def _sync_external_memory_for_turn(self, **_kwargs):
            return None

        def _spawn_background_review(self, **_kwargs):
            return None

        def clear_interrupt(self):
            return None

    result = finalize_turn(
        Agent(),
        final_response=None,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "do work"}],
        conversation_history=[],
        effective_task_id="default",
        turn_id="turn-1",
        user_message="do work",
        original_user_message="do work",
        _should_review_memory=False,
        _turn_exit_reason="loop_exhausted",
    )

    artifact = latest_closure_artifact(session_id="finalizer-session")

    assert result["closure_artifact"] == artifact["artifact_path"]
    assert artifact["status"] == "max_iterations_reached"
    assert artifact["last_completed_step"] == "summary after budget"
    assert artifact["active_session_lease_released"] is False
    assert artifact["latest_session_id"] == "finalizer-session"
    assert "Do not perform live provider" in artifact["safe_bounded_resume_prompt"]
    assert artifact["closeout_status"] == "complete_candidate"
