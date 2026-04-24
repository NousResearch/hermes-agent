from __future__ import annotations

from hermes_cli.command_templates import build_command_invocation
from hermes_cli.commands import SUBCOMMANDS, resolve_command


def test_refactor_parses_scope_strategy_and_contract_fields(tmp_path):
    invocation = build_command_invocation(
        "refactor",
        raw_args="--scope module --strategy extract-then-adapt simplify parser orchestration",
        session_id="session-123",
        cwd=str(tmp_path),
    )

    assert invocation.task_contract["task"] == "simplify parser orchestration"
    assert invocation.task_contract["context"]["refactor"]["scope"] == "module"
    assert invocation.task_contract["context"]["refactor"]["strategy"] == "extract-then-adapt"
    assert invocation.task_contract["context"]["blast_radius"] == "module-local changes plus direct imports and callers"
    assert invocation.task_contract["context"]["acceptance_tests"] == [
        "run focused tests covering the touched module and its direct callers",
        "verify renamed or extracted symbols still resolve at call sites",
        "confirm externally observable behavior remains unchanged unless explicitly requested",
    ]
    assert "code-intel" in invocation.task_contract["required_tools"]
    assert invocation.orchestration_hints["refactor"]["scope"] == "module"
    assert invocation.orchestration_hints["refactor"]["strategy"] == "extract-then-adapt"
    assert invocation.orchestration_hints["tool_preferences"]["prefer"] == ["code-intel"]
    assert invocation.named_workflow is not None


def test_refactor_keeps_natural_language_prompt_backwards_compatible(tmp_path):
    invocation = build_command_invocation(
        "refactor",
        raw_args="clean up the parser without changing behavior",
        cwd=str(tmp_path),
    )

    assert invocation.task_contract["task"] == "clean up the parser without changing behavior"
    assert invocation.task_contract["context"]["refactor"]["scope"] == "file"
    assert invocation.task_contract["context"]["refactor"]["strategy"] == "safe-mechanical"
    assert invocation.orchestration_hints["request"] == "clean up the parser without changing behavior"


def test_refactor_repo_scope_requires_explicit_approval(tmp_path):
    invocation = build_command_invocation(
        "refactor",
        raw_args="--scope repo --strategy rename-then-adapt normalize public API names",
        cwd=str(tmp_path),
    )

    assert invocation.task_contract["context"]["refactor"]["scope"] == "repo"
    assert invocation.task_contract["context"]["refactor"]["repo_wide_approved"] is False
    assert invocation.task_contract["context"]["refactor"]["approval_required"] is True
    assert invocation.task_contract["expected_outcome"] == "Refuse repo-wide refactor execution until explicit repo-wide approval is supplied."
    assert invocation.orchestration_hints["refactor"]["status"] == "blocked_pending_repo_approval"
    assert invocation.named_workflow is None


def test_refactor_repo_scope_runs_when_explicitly_approved(tmp_path):
    invocation = build_command_invocation(
        "refactor",
        raw_args="--scope repo --approve-repo-wide update import paths consistently",
        cwd=str(tmp_path),
    )

    assert invocation.task_contract["context"]["refactor"]["scope"] == "repo"
    assert invocation.task_contract["context"]["refactor"]["repo_wide_approved"] is True
    assert invocation.task_contract["context"]["blast_radius"] == "repo-wide coordinated changes across multiple modules and call paths"
    assert invocation.orchestration_hints["refactor"]["status"] == "approved"
    assert invocation.named_workflow is not None


def test_refactor_command_registry_metadata_is_exposed():
    command = resolve_command("refactor")

    assert command is not None
    assert "--scope" in command.args_hint
    assert "--strategy" in command.args_hint
    assert "/refactor" in SUBCOMMANDS
    assert SUBCOMMANDS["/refactor"] == ["--scope", "--strategy", "--approve-repo-wide"]
