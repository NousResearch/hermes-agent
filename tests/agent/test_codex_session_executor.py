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
    assert "Task:\nFix the failing focused test" in prompt
    assert kwargs == {"turn_timeout": 12}
    assert fake.closed is True


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
