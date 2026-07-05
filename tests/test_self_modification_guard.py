"""Tests for the runtime self-modification write gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.runtime_evidence import clear_runtime_evidence, current_runtime_evidence, seed_turn_runtime_evidence
from model_tools import handle_function_call


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _blocked_call(function_name: str, args: dict, user_task: str):
    with patch("hermes_cli.plugins.has_hook", return_value=False), patch(
        "model_tools.registry.dispatch",
        side_effect=AssertionError("dispatch should not run for blocked self-modification"),
    ):
        return json.loads(
            handle_function_call(
                function_name,
                args,
                task_id="task-1",
                user_task=user_task,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
            )
        )


class TestSelfModificationGuard:
    def test_explicit_self_improvement_without_target_files_is_blocked(self):
        target = _repo_root() / "skills/productivity/notion/SKILL.md"
        result = _blocked_call(
            "write_file",
            {"path": str(target), "content": "patched"},
            "Explicit self-improvement request: update Hermes guardrails.",
        )
        assert "error" in result
        assert "self-modification" in result["error"].lower()
        assert "explicit user instruction" in result["error"].lower()

    def test_explicit_self_improvement_request_with_named_target_files_can_pass(self):
        target = _repo_root() / "skills/autonomous-ai-agents/hermes-agent/SKILL.md"

        def _dispatch(*_args, **_kwargs):
            evidence = current_runtime_evidence()
            assert evidence is not None
            assert evidence["latest_user_request"].startswith("Explicit self-improvement request")
            assert evidence["pre_edit_git_status"] == "M tests/test_self_modification_guard.py"
            return '{"success": true, "message": "updated", "path": "SKILL.md"}'

        with patch("hermes_cli.plugins.has_hook", return_value=False), patch(
            "model_tools.registry.dispatch",
            side_effect=_dispatch,
        ) as dispatch, patch("agent.self_modification_guard._capture_git_status", side_effect=["M tests/test_self_modification_guard.py", "M tests/test_self_modification_guard.py"]):
            result = json.loads(
                handle_function_call(
                    "write_file",
                    {"path": str(target), "content": "patched"},
                    task_id="task-2",
                    user_task=f"Explicit self-improvement request: edit the target file {target} and validate it.",
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                )
            )
        assert result["success"] is True
        assert "self_modification_report" in result
        report = result["self_modification_report"]
        assert report["files_changed"] == [str(target)]
        assert "git status --short (pre-edit)" in report["validation_tests_run"]
        assert report["commit_happened"] is False
        assert report["push_happened"] is False
        dispatch.assert_called_once()

    def test_protected_write_without_pre_edit_git_status_is_blocked(self):
        target = _repo_root() / "skills/autonomous-ai-agents/hermes-agent/SKILL.md"
        with patch("hermes_cli.plugins.has_hook", return_value=False), patch(
            "agent.self_modification_guard._capture_git_status",
            return_value=None,
        ):
            result = _blocked_call(
                "write_file",
                {"path": str(target), "content": "patched"},
                f"Explicit self-improvement request: edit the target file {target} and validate it.",
            )
        assert "error" in result
        assert "pre-edit git status" in result["error"].lower()

    def test_protected_write_uses_target_repo_root_instead_of_cwd(self, tmp_path, monkeypatch):
        cwd_repo = tmp_path / "cwd-repo"
        target_repo = tmp_path / "target-repo"
        nested_target = target_repo / "skills" / "productivity" / "notion" / "SKILL.md"

        for repo, marker_name in ((cwd_repo, "cwd-only.txt"), (target_repo, "target-only.txt")):
            repo.mkdir(parents=True)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / marker_name).write_text("marker\n", encoding="utf-8")

        (cwd_repo / "cwd-only.txt").write_text("marker\n", encoding="utf-8")
        (target_repo / "target-only.txt").write_text("marker\n", encoding="utf-8")
        nested_target.parent.mkdir(parents=True, exist_ok=True)
        nested_target.write_text("patched\n", encoding="utf-8")
        monkeypatch.chdir(cwd_repo)

        def _dispatch(*_args, **_kwargs):
            evidence = current_runtime_evidence()
            assert evidence is not None
            assert evidence["audit_repo_root"] == str(target_repo)
            assert "target-only.txt" in evidence["pre_edit_git_status"]
            assert "cwd-only.txt" not in evidence["pre_edit_git_status"]
            return '{"success": true, "message": "updated", "path": "SKILL.md"}'

        with patch("hermes_cli.plugins.has_hook", return_value=False), patch(
            "model_tools.registry.dispatch",
            side_effect=_dispatch,
        ):
            result = json.loads(
                handle_function_call(
                    "write_file",
                    {"path": str(nested_target), "content": "patched"},
                    task_id="task-root-resolution",
                    user_task=f"Explicit self-improvement request: edit the target file {nested_target} and validate it.",
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                )
            )

        report = result["self_modification_report"]
        assert "target-only.txt" in report["pre_edit_git_status"]
        assert "cwd-only.txt" not in report["pre_edit_git_status"]
        assert "target-only.txt" in report["final_git_status"]
        assert report["files_changed"] == [str(nested_target)]

    def test_runtime_evidence_is_shared_across_nested_calls(self):
        clear_runtime_evidence()
        seed_turn_runtime_evidence(latest_user_request="Explicit self-improvement request: update Hermes guardrails.")

        def _nested_update() -> None:
            evidence = current_runtime_evidence()
            assert evidence is not None
            assert evidence["latest_user_request"] == "Explicit self-improvement request: update Hermes guardrails."
            evidence["failing_file"] = "docs/runtime-guardrails-roadmap.md"
            evidence["failing_line"] = 103

        _nested_update()

        evidence = current_runtime_evidence()
        assert evidence is not None
        assert evidence["failing_file"] == "docs/runtime-guardrails-roadmap.md"
        assert evidence["failing_line"] == 103
        clear_runtime_evidence()

    def test_after_protected_write_includes_validation_diff_and_status_summary(self):
        target = _repo_root() / "skills/autonomous-ai-agents/hermes-agent/SKILL.md"
        with patch("hermes_cli.plugins.has_hook", return_value=False), patch(
            "model_tools.registry.dispatch",
            return_value='{"success": true, "message": "updated", "path": "SKILL.md"}',
        ), patch("agent.self_modification_guard._capture_git_status", side_effect=[" M pre.txt", " M post.txt"]), patch("agent.self_modification_guard._capture_git_diff_summary", return_value=" pre.txt | 1 +"):
            result = json.loads(
                handle_function_call(
                    "write_file",
                    {"path": str(target), "content": "patched"},
                    task_id="task-3",
                    user_task=f"Explicit self-improvement request: edit the target file {target} and validate it.",
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                )
            )
        report = result["self_modification_report"]
        assert report["files_changed"] == [str(target)]
        assert report["validation_tests_run"]
        assert report["final_diff_summary"] == "pre.txt | 1 +"
        assert report["final_git_status"] == "M post.txt"
        assert "pre-edit" in report["validation_summary"].lower() or report["validation_summary"]

    @pytest.mark.parametrize(
        ("function_name", "args", "user_task"),
        [
            (
                "terminal",
                {"command": f"printf hi > {_repo_root() / 'docs/hermes-guardrails.md'}"},
                "Please improve the launch notes.",
            ),
            (
                "terminal",
                {"command": f"printf hi >> {_repo_root() / 'docs/hermes-guardrails.md'}"},
                "Please improve the launch notes.",
            ),
            (
                "terminal",
                {"command": f"printf hi | tee {_repo_root() / 'docs/hermes-guardrails.md'}"},
                "Please improve the launch notes.",
            ),
            (
                "terminal",
                {"command": f"sed -i 's/a/b/' {_repo_root() / 'docs/hermes-guardrails.md'}"},
                "Please improve the launch notes.",
            ),
            (
                "execute_code",
                {"code": "with open(r'" + str(_repo_root() / 'docs/hermes-guardrails.md') + "', 'w') as f:\n    f.write('patched')"},
                "Please improve the launch notes.",
            ),
            (
                "terminal",
                {"command": f"cp source.txt {_repo_root() / 'docs/hermes-guardrails.md'}"},
                "Please improve the launch notes.",
            ),
            (
                "terminal",
                {"command": f"mv source.txt {_repo_root() / 'docs/hermes-guardrails.md'}"},
                "Please improve the launch notes.",
            ),
        ],
    )
    def test_shell_write_variants_are_blocked_when_unrequested(self, function_name, args, user_task):
        result = _blocked_call(function_name, args, user_task)
        assert "error" in result
        assert "blocked self-modification" in result["error"].lower()

    def test_skill_manage_is_blocked_for_unrequested_skill_writes(self):
        result = _blocked_call(
            "skill_manage",
            {"action": "update", "name": "notion", "category": "productivity"},
            "Please improve the launch notes.",
        )
        assert "error" in result
        assert "skill_manage" in result["error"].lower()
        assert "protected path" in result["error"].lower()

    def test_notion_healthcheck_cannot_mutate_notion_skill_files(self):
        target = _repo_root() / "skills/productivity/notion/SKILL.md"
        result = _blocked_call(
            "write_file",
            {"path": str(target), "content": "healthcheck mutation"},
            "Run a Notion healthcheck.",
        )
        assert "error" in result
        assert "notion" in str(target).lower()
        assert "blocked self-modification" in result["error"].lower()
