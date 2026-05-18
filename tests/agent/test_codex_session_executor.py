from __future__ import annotations

import json

from agent.transports.codex_app_server_session import TurnResult


class FakeCodexSession:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.calls = []
        FakeCodexSession.instances.append(self)

    def run_turn(self, user_input, **kwargs):
        self.calls.append((user_input, kwargs))
        return TurnResult(
            final_text=json.dumps(
                {
                    "summary": "patched the focused tests",
                    "changed_files": ["tools/codex_session_tool.py"],
                    "commands_run": ["pytest tests/agent/test_codex_session_executor.py -q"],
                    "tests_run": ["pytest tests/agent/test_codex_session_executor.py -q"],
                    "diff": "diff --git a/tools/codex_session_tool.py b/tools/codex_session_tool.py",
                }
            ),
            tool_iterations=2,
            turn_id="turn-1",
            thread_id="thread-1",
        )

    def close(self):
        self.closed = True


def test_codex_session_executor_returns_structured_evidence_not_final_answer():
    from agent.executors.codex_session import run_codex_session

    evidence = run_codex_session(
        task="Fix the failing focused test",
        cwd="/tmp/project",
        session_factory=FakeCodexSession,
        turn_timeout=12,
    )

    assert evidence["success"] is True
    assert evidence["summary"] == "patched the focused tests"
    assert evidence["changed_files"] == ["tools/codex_session_tool.py"]
    assert evidence["commands_run"] == ["pytest tests/agent/test_codex_session_executor.py -q"]
    assert evidence["tests_run"] == ["pytest tests/agent/test_codex_session_executor.py -q"]
    assert evidence["diff"].startswith("diff --git")
    assert evidence["user_facing_final"] is False
    assert evidence["requires_hermes_verification"] is True
    assert evidence["codex"] == {"thread_id": "thread-1", "turn_id": "turn-1", "tool_iterations": 2}

    fake = FakeCodexSession.instances[-1]
    assert fake.kwargs["cwd"] == "/tmp/project"
    assert len(fake.calls) == 1
    prompt, kwargs = fake.calls[0]
    assert "bounded Codex executor under Hermes Agent" in prompt
    assert "Hermes/Yuuka remains the user-facing agent and verifier" in prompt
    assert "Do not write a final response to the user" in prompt
    assert "success must be false if you cannot verify the requested condition" in prompt
    assert "Task:\nFix the failing focused test" in prompt
    assert kwargs == {"turn_timeout": 12}
    assert fake.closed is True


def test_codex_session_executor_uses_hermes_bounded_write_profile_args():
    from agent.executors.codex_session import run_codex_session

    run_codex_session(
        task="Edit only the task worktree marker file",
        cwd="/tmp/task-worktree",
        session_factory=FakeCodexSession,
        permission_profile="hermes-worktree-write",
    )

    fake = FakeCodexSession.instances[-1]
    assert fake.kwargs["permission_profile"] == "hermes-worktree-write"
    assert fake.kwargs["app_server_extra_args"] == [
        "-c",
        'sandbox_mode="danger-full-access"',
        "-c",
        'approval_policy="never"',
    ]


def test_codex_session_executor_keeps_default_profile_without_app_server_overrides():
    from agent.executors.codex_session import run_codex_session

    run_codex_session(
        task="Inspect only",
        session_factory=FakeCodexSession,
    )

    fake = FakeCodexSession.instances[-1]
    assert fake.kwargs["permission_profile"] is None
    assert fake.kwargs["app_server_extra_args"] is None


def test_codex_session_executor_wraps_plain_text_as_unverified_summary():
    from agent.executors.codex_session import run_codex_session

    class PlainTextSession(FakeCodexSession):
        def run_turn(self, user_input, **kwargs):
            return TurnResult(final_text="I changed files and ran tests.")

    evidence = run_codex_session(task="Do work", session_factory=PlainTextSession)

    assert evidence["success"] is True
    assert evidence["summary"] == "I changed files and ran tests."
    assert evidence["changed_files"] == []
    assert evidence["commands_run"] == []
    assert evidence["tests_run"] == []
    assert evidence["requires_hermes_verification"] is True
    assert evidence["raw_codex_final_text"] == "I changed files and ran tests."


def test_codex_session_executor_honors_codex_reported_failure():
    from agent.executors.codex_session import run_codex_session

    class FailedEvidenceSession(FakeCodexSession):
        def run_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text=json.dumps(
                    {
                        "success": False,
                        "summary": "sandbox command failed before verification",
                        "commands_run": ["test -e tools/codex_session_tool.py"],
                        "error": "sandbox command failed before verification",
                    }
                )
            )

    evidence = run_codex_session(task="Verify file", session_factory=FailedEvidenceSession)

    assert evidence["success"] is False
    assert evidence["summary"] == "sandbox command failed before verification"
    assert evidence["commands_run"] == ["test -e tools/codex_session_tool.py"]
    assert evidence["error"] == "sandbox command failed before verification"
    assert evidence["requires_hermes_verification"] is True


def test_codex_session_executor_rejects_contradictory_unable_to_verify_success():
    from agent.executors.codex_session import run_codex_session

    class ContradictoryEvidenceSession(FakeCodexSession):
        def run_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text=json.dumps(
                    {
                        "success": True,
                        "summary": "Unable to verify file existence because sandboxed read-only commands failed.",
                        "changed_files": [],
                        "commands_run": ["test -e tools/codex_session_tool.py"],
                        "tests_run": [],
                        "diff": "",
                        "error": None,
                    }
                )
            )

    evidence = run_codex_session(task="Verify file", session_factory=ContradictoryEvidenceSession)

    assert evidence["success"] is False
    assert evidence["error"] == "codex_evidence_reports_unverified_result"
    assert evidence["ambiguous"] is True
    assert evidence["changed_files"] == []
    assert evidence["requires_hermes_verification"] is True


def test_codex_session_executor_reports_errors_as_evidence():
    from agent.executors.codex_session import run_codex_session

    class ErrorSession(FakeCodexSession):
        def run_turn(self, user_input, **kwargs):
            return TurnResult(error="codex app-server startup failed", should_retire=True)

    evidence = run_codex_session(task="Do work", session_factory=ErrorSession)

    assert evidence["success"] is False
    assert evidence["error"] == "codex app-server startup failed"
    assert evidence["should_retire"] is True
    assert evidence["changed_files"] == []
    assert evidence["user_facing_final"] is False
    assert evidence["requires_hermes_verification"] is True
