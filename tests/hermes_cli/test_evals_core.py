from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import yaml


def test_validate_manifest_accepts_minimal_candidate() -> None:
    from hermes_cli.evals_core import validate_manifest

    manifest = {
        "schema_version": 1,
        "id": "source-reading-001",
        "status": "candidate",
        "instruction": "Read the supplied source before answering.",
        "source": {
            "kind": "session_trace",
            "digest": "sha256:" + ("a" * 64),
            "message_count": 2,
            "sanitized": True,
        },
        "environment": {"allowed_tools": ["web", "session_search"]},
        "success": {
            "deterministic": [],
            "judged": ["Every factual claim is grounded in the supplied source."],
        },
        "forbidden": ["Use session history as proof of current source contents."],
        "skills": ["x-link-resolution-fallback"],
    }

    result = validate_manifest(manifest)

    assert result.errors == ()
    assert result.ready is False
    assert "status must be 'approved'" in result.warnings


def test_build_candidate_from_trace_extracts_failure_signals_and_redacts() -> None:
    from hermes_cli.evals_core import build_candidate_from_trace, validate_manifest

    session = {
        "id": "session-123",
        "source": "telegram",
        "model": "gpt-test",
        "cwd": "/tmp/project",
        "tool_call_count": 2,
        "estimated_cost_usd": 0.25,
    }
    messages = [
        {
            "id": 10,
            "role": "user",
            "content": (
                "Deploy /tmp/project using token sk-abcdefghijklmnopqrstuvwxyz. "
                "Contact me at ric@example.com or 630-555-1212. "
                "Fetch https://alice:hunter2@example.com/callback?api_key=topsecret123."
            ),
        },
        {
            "id": 11,
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [
                {"function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {
            "id": 12,
            "role": "tool",
            "tool_name": "terminal",
            "content": '{"success": false, "error": "connection refused"}',
        },
        {
            "id": 13,
            "role": "user",
            "content": "No, you did not verify the deployment.",
        },
        {
            "id": 14,
            "role": "tool",
            "tool_name": "terminal",
            "content": (
                "[OUT-OF-BAND USER MESSAGE — a direct message from the user, "
                "delivered mid-turn; not tool output]\nStop and verify first.\n"
                "[/OUT-OF-BAND USER MESSAGE]"
            ),
        },
    ]

    manifest = build_candidate_from_trace(session, messages)

    assert manifest["id"].startswith("trace-")
    assert "session-123" not in manifest["id"]
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in manifest["instruction"]
    assert "$WORKSPACE" in manifest["instruction"]
    assert "[email]" in manifest["instruction"]
    assert "[phone]" in manifest["instruction"]
    assert "hunter2" not in manifest["instruction"]
    assert "topsecret123" not in manifest["instruction"]
    assert manifest["source"]["message_count"] == 5
    assert manifest["source"]["digest"].startswith("sha256:")
    assert "session_id" not in manifest["source"]
    assert manifest["source"]["sanitized"] is True
    assert manifest["environment"]["allowed_tools"] == ["terminal"]
    assert {signal["kind"] for signal in manifest["signals"]} == {
        "tool_failure",
        "user_correction",
        "midturn_user_steering",
    }
    assert all("excerpt" not in signal for signal in manifest["signals"])
    assert validate_manifest(manifest).errors == ()


def test_cmd_evals_mine_writes_sanitized_candidate(tmp_path, monkeypatch, capsys) -> None:
    from hermes_cli import evals_cmd

    class FakeDB:
        def resolve_session_id(self, value: str) -> str | None:
            return "session-123" if value == "session" else None

        def get_session(self, session_id: str):
            return {
                "id": session_id,
                "source": "telegram",
                "model": "gpt-test",
                "tool_call_count": 1,
            }

        def get_messages(self, session_id: str):
            return [
                {"id": 1, "role": "user", "content": "Check API_KEY=super-secret-value"},
                {
                    "id": 2,
                    "role": "tool",
                    "tool_name": "terminal",
                    "content": '{"exit_code": 1, "output": "failed"}',
                },
            ]

        def close(self) -> None:
            pass

    output = tmp_path / "candidate.yaml"
    monkeypatch.setattr(evals_cmd, "_open_session_db", FakeDB)
    args = SimpleNamespace(
        evals_action="mine",
        session_id="session",
        output=output,
        force=False,
    )

    rc = evals_cmd.cmd_evals(args)

    assert rc == 0
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    rendered = output.read_text(encoding="utf-8")
    assert payload["source"]["digest"].startswith("sha256:")
    assert "session-123" not in rendered
    assert "super-secret-value" not in rendered
    assert "Review required" in capsys.readouterr().out


def test_cmd_evals_validate_ready_rejects_candidate(tmp_path, capsys) -> None:
    from hermes_cli.evals_cmd import cmd_evals

    path = tmp_path / "candidate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "candidate-001",
                "status": "candidate",
                "instruction": "Do the work.",
                "source": {"kind": "manual", "sanitized": True},
                "environment": {"allowed_tools": []},
                "success": {"deterministic": [], "judged": ["It worked."]},
                "forbidden": [],
                "skills": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(evals_action="validate", path=path, ready=True)

    rc = cmd_evals(args)

    assert rc == 1
    assert "NOT READY" in capsys.readouterr().out


def test_build_evals_parser_wires_mine_and_validate() -> None:
    from hermes_cli.subcommands.evals import build_evals_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    sentinel = object()
    build_evals_parser(subparsers, cmd_evals=sentinel)

    mined = parser.parse_args(
        ["evals", "mine", "abc123", "--output", "candidate.yaml"]
    )
    validated = parser.parse_args(["evals", "validate", "corpus", "--ready"])
    scored = parser.parse_args(["evals", "score", "task.yaml", "run.json"])

    assert mined.func is sentinel
    assert mined.session_id == "abc123"
    assert mined.output.name == "candidate.yaml"
    assert validated.func is sentinel
    assert validated.ready is True
    assert scored.func is sentinel
    assert scored.task.name == "task.yaml"
    assert scored.run.name == "run.json"


def test_score_run_artifact_applies_deterministic_checks() -> None:
    from hermes_cli.evals_core import score_run_artifact

    manifest = {
        "schema_version": 1,
        "id": "deploy-check-001",
        "status": "approved",
        "instruction": "Deploy and verify the service.",
        "source": {"kind": "manual", "sanitized": True},
        "environment": {"allowed_tools": ["terminal"]},
        "success": {
            "deterministic": [
                {"type": "tool_called", "name": "terminal"},
                {"type": "tool_succeeded", "name": "terminal"},
                {"type": "final_response_contains", "value": "verified"},
                {"type": "final_response_excludes", "value": "probably"},
            ],
            "judged": [],
        },
        "forbidden": [],
        "skills": [],
    }
    run = {
        "task_id": "deploy-check-001",
        "final_response": "Deployment verified with a live health check.",
        "tool_calls": [
            {"name": "terminal", "result": {"exit_code": 0, "output": "healthy"}}
        ],
    }

    result = score_run_artifact(manifest, run)

    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["deterministic"] == {"passed": 5, "total": 5}
    assert all(check["passed"] for check in result["checks"])


def test_score_run_artifact_requires_separate_judge_when_criteria_exist() -> None:
    from hermes_cli.evals_core import score_run_artifact

    manifest = {
        "schema_version": 1,
        "id": "research-001",
        "status": "approved",
        "instruction": "Research the claim.",
        "source": {"kind": "manual", "sanitized": True},
        "environment": {"allowed_tools": ["web_search"]},
        "success": {
            "deterministic": [{"type": "tool_called", "name": "web_search"}],
            "judged": ["The synthesis is faithful to the sources."],
        },
        "forbidden": [],
        "skills": [],
    }
    run = {
        "task_id": "research-001",
        "final_response": "A sourced answer.",
        "tool_calls": [{"name": "web_search", "result": {"success": True}}],
    }

    result = score_run_artifact(manifest, run)

    assert result["status"] == "needs_judge"
    assert result["passed"] is None
    assert result["judge_criteria"] == ["The synthesis is faithful to the sources."]


def test_cmd_evals_score_writes_machine_readable_result(tmp_path, capsys) -> None:
    from hermes_cli.evals_cmd import cmd_evals

    task = tmp_path / "task.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "score-001",
                "status": "approved",
                "instruction": "Verify it.",
                "source": {"kind": "manual", "sanitized": True},
                "environment": {"allowed_tools": ["terminal"]},
                "success": {
                    "deterministic": [{"type": "tool_succeeded", "name": "terminal"}],
                    "judged": [],
                },
                "forbidden": [],
                "skills": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "task_id": "score-001",
                "final_response": "Verified.",
                "tool_calls": [{"name": "terminal", "result": {"exit_code": 0}}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    args = SimpleNamespace(
        evals_action="score",
        task=task,
        run=run,
        output=output,
        force=False,
    )

    rc = cmd_evals(args)

    assert rc == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert "PASSED score-001" in capsys.readouterr().out


def test_score_rejects_unapproved_candidate() -> None:
    import pytest

    from hermes_cli.evals_core import score_run_artifact

    manifest = {
        "schema_version": 1,
        "id": "candidate-001",
        "status": "candidate",
        "instruction": "Do it.",
        "source": {"kind": "manual", "sanitized": True},
        "environment": {"allowed_tools": []},
        "success": {"deterministic": [], "judged": ["It worked."]},
        "forbidden": [],
        "skills": [],
    }

    with pytest.raises(ValueError, match="not ready"):
        score_run_artifact(
            manifest,
            {"task_id": "candidate-001", "final_response": "Done", "tool_calls": []},
        )


def test_validate_requires_allowed_tools_and_safe_session_provenance() -> None:
    from hermes_cli.evals_core import validate_manifest

    manifest = {
        "schema_version": 1,
        "id": "trace-001",
        "status": "candidate",
        "instruction": "Do it.",
        "source": {
            "kind": "session_trace",
            "digest": "not-a-digest",
            "message_count": True,
            "sanitized": True,
        },
        "environment": {},
        "success": {"deterministic": [], "judged": ["It worked."]},
        "forbidden": [],
        "skills": [],
    }

    result = validate_manifest(manifest)

    assert "source.digest must be a sha256 digest for session_trace tasks" in result.errors
    assert "source.message_count must be a positive integer" in result.errors
    assert "environment.allowed_tools must be a list of non-empty strings" in result.errors


def test_candidate_recovers_skill_instruction_and_has_stable_opaque_id() -> None:
    from hermes_cli.evals_core import build_candidate_from_trace, validate_manifest

    expanded = (
        '[IMPORTANT: The user has invoked the "code-review" skill. '
        "The full skill content is loaded below.]\n"
        "SECRET SKILL BODY THAT MUST NOT BECOME THE TASK\n"
        "The user has provided the following instruction alongside the skill invocation: "
        "Review the authentication patch."
    )
    messages = [
        {"id": 1, "role": "user", "content": expanded},
        {"id": 2, "role": "assistant", "content": "Reviewed."},
    ]

    first = build_candidate_from_trace({"id": "raw-session-one"}, messages)
    second = build_candidate_from_trace({"id": "raw-session-two"}, messages)

    assert first["instruction"] == "Review the authentication patch."
    assert "SECRET SKILL BODY" not in first["instruction"]
    assert first["id"] == second["id"]
    assert "raw-session" not in first["id"]
    assert first["source"]["digest"] == second["source"]["digest"]

    first["future_field"] = True
    assert validate_manifest(first).errors == (
        "unknown top-level fields: future_field",
    )


def test_manifest_loader_rejects_duplicate_yaml_keys(tmp_path) -> None:
    import pytest

    from hermes_cli.evals_cmd import load_manifest

    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "schema_version: 1\nstatus: candidate\nstatus: approved\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key 'status'"):
        load_manifest(path)


def test_score_enforces_allowed_tools_and_surfaces_forbidden_rules() -> None:
    from hermes_cli.evals_core import score_run_artifact

    manifest = {
        "schema_version": 1,
        "id": "policy-001",
        "status": "approved",
        "instruction": "Use web search only.",
        "source": {"kind": "manual", "sanitized": True},
        "environment": {"allowed_tools": ["web_search"]},
        "success": {
            "deterministic": [{"type": "final_response_contains", "value": "done"}],
            "judged": [],
        },
        "forbidden": ["Do not modify local files."],
        "skills": [],
    }

    failed = score_run_artifact(
        manifest,
        {
            "task_id": "policy-001",
            "final_response": "Done.",
            "tool_calls": [{"name": "terminal", "result": {"exit_code": 0}}],
        },
    )
    assert failed["status"] == "failed"
    assert failed["checks"][0]["passed"] is False
    assert "terminal" in failed["checks"][0]["detail"]

    needs_judge = score_run_artifact(
        manifest,
        {
            "task_id": "policy-001",
            "final_response": "Done.",
            "tool_calls": [{"name": "web_search", "result": {"success": True}}],
        },
    )
    assert needs_judge["status"] == "needs_judge"
    assert needs_judge["judge_criteria"] == [
        "Verify forbidden behavior did not occur: Do not modify local files."
    ]
