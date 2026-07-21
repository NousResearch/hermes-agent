"""Background-review integration for autonomous skill validation/refinement."""

from __future__ import annotations

import json
from pathlib import Path

from agent.background_skill_lifecycle import (
    collect_successful_skill_mutations,
    run_background_skill_lifecycles,
)
from agent import background_review
from tools.skill_lifecycle_orchestrator import TestExecutionResult as ExecutionResult
from tools.skill_validation import validation_allows_discovery


def _review_messages(name: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call-{name}",
                    "function": {
                        "name": "skill_manage",
                        "arguments": json.dumps({"action": "write_file", "name": name}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call-{name}",
            "content": json.dumps({"success": True, "message": "written"}),
        },
    ]


def _skill(home: Path, name: str = "demo-skill") -> Path:
    skill_dir = home / "skills" / name
    (skill_dir / "tests").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo.\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    (skill_dir / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert True\n",
        encoding="utf-8",
    )
    return skill_dir


def test_collects_only_successful_package_mutations() -> None:
    messages = _review_messages("demo-skill")
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-remember",
                        "function": {
                            "name": "skill_manage",
                            "arguments": json.dumps(
                                {"action": "remember", "name": "ignored-memory"}
                            ),
                        },
                    },
                    {
                        "id": "call-failed",
                        "function": {
                            "name": "skill_manage",
                            "arguments": json.dumps(
                                {"action": "patch", "name": "failed-skill"}
                            ),
                        },
                    },
                    {
                        "id": "call-staged",
                        "function": {
                            "name": "skill_manage",
                            "arguments": json.dumps(
                                {"action": "patch", "name": "staged-skill"}
                            ),
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-remember",
                "content": json.dumps({"success": True}),
            },
            {
                "role": "tool",
                "tool_call_id": "call-failed",
                "content": json.dumps({"success": False}),
            },
            {
                "role": "tool",
                "tool_call_id": "call-staged",
                "content": json.dumps({"success": True, "staged": True}),
            },
        ]
    )

    assert collect_successful_skill_mutations(messages) == ["demo-skill"]


def test_collector_excludes_mutations_in_inherited_history() -> None:
    inherited = _review_messages("old-skill")
    current = inherited + _review_messages("new-skill")

    assert collect_successful_skill_mutations(
        current, prior_messages=inherited
    ) == ["new-skill"]


def test_background_lifecycle_refines_failed_skill_then_registers(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill_dir = _skill(tmp_path)
    executions = iter(
        [
            ExecutionResult(1, "assertion failed", "test"),
            ExecutionResult(0, "1 passed", "test"),
        ]
    )

    class ReviewAgent:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.histories: list[list[dict]] = []

        def run_conversation(self, user_message: str, **kwargs) -> None:
            self.prompts.append(user_message)
            self.histories.append(kwargs["conversation_history"])
            (skill_dir / "scripts").mkdir(exist_ok=True)
            (skill_dir / "scripts" / "fix.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )

    agent = ReviewAgent()
    results = run_background_skill_lifecycles(
        agent,
        _review_messages("demo-skill"),
        execute=lambda _request: next(executions),
        max_refinements=2,
    )

    assert results["demo-skill"].status == "passed"
    assert results["demo-skill"].refinement_attempts == 1
    assert len(agent.prompts) == 1
    assert "assertion failed" in agent.prompts[0]
    assert agent.histories[0] == _review_messages("demo-skill")
    assert validation_allows_discovery(skill_dir) is True


def test_background_lifecycle_fails_closed_without_isolated_executor(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill_dir = _skill(tmp_path)

    results = run_background_skill_lifecycles(
        object(),
        _review_messages("demo-skill"),
        execute=None,
    )

    assert results == {}
    assert validation_allows_discovery(skill_dir) is False


def test_malformed_skill_does_not_abort_later_lifecycles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    bad_dir = _skill(tmp_path, "bad-skill")
    good_dir = _skill(tmp_path, "good-skill")
    try:
        (bad_dir / "scripts").mkdir()
        (bad_dir / "scripts" / "unsafe-link").symlink_to(tmp_path / "outside")
    except OSError:
        return

    messages = _review_messages("bad-skill") + _review_messages("good-skill")
    results = run_background_skill_lifecycles(
        object(),
        messages,
        execute=lambda _request: ExecutionResult(0, "1 passed", "test"),
    )

    assert results["bad-skill"].status == "error"
    assert results["bad-skill"].registered is False
    assert results["good-skill"].status == "passed"
    assert results["good-skill"].registered is True
    assert validation_allows_discovery(bad_dir) is False
    assert validation_allows_discovery(good_dir) is True


def test_background_review_dispatches_autonomous_lifecycle(monkeypatch) -> None:
    executor = object()
    captured = {}

    monkeypatch.setattr(
        "tools.skill_test_sandbox.BubblewrapTestExecutor.discover",
        lambda: executor,
    )

    def run(agent, messages, *, execute, max_refinements, prior_messages):
        captured.update(
            agent=agent,
            messages=messages,
            execute=execute,
            max_refinements=max_refinements,
            prior_messages=prior_messages,
        )
        return {"demo-skill": object()}

    monkeypatch.setattr(
        "agent.background_skill_lifecycle.run_background_skill_lifecycles", run
    )
    agent = object()
    messages = _review_messages("demo-skill")
    prior_messages = [{"role": "user", "content": "prior"}]

    result = background_review._run_autonomous_skill_lifecycle(
        agent, messages, prior_messages=prior_messages
    )

    assert result.keys() == {"demo-skill"}
    assert captured["agent"] is agent
    assert captured["messages"] is messages
    assert captured["execute"] is executor
    assert captured["max_refinements"] == 2
    assert captured["prior_messages"] is prior_messages


def test_background_review_lifecycle_failure_does_not_abort_review(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.skill_test_sandbox.BubblewrapTestExecutor.discover",
        lambda: (_ for _ in ()).throw(RuntimeError("sandbox probe failed")),
    )

    assert background_review._run_autonomous_skill_lifecycle(object(), []) == {}


def test_background_review_prompts_request_tests_for_code_backed_skills() -> None:
    expected = "isolated lifecycle runner"
    assert expected in background_review._SKILL_REVIEW_PROMPT
    assert expected in background_review._COMBINED_REVIEW_PROMPT
