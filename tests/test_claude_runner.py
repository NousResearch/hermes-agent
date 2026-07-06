from __future__ import annotations

from pathlib import Path
import logging

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

import hermes.claude_runner as claude_runner
from hermes.claude_runner import (
    ANTHROPIC_API_KEY_ENV,
    BuildApprovalRequiredError,
    ClaudeRunnerConfig,
    ClaudeStageRunner,
    DesignRunStore,
    DesignSpec,
    DesignStageParseError,
    RunStatus,
)


def test_claude_runner_config_reads_anthropic_key_from_environment(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "test-api-key")

    config = ClaudeRunnerConfig(
        model_name="claude-sonnet-4-5",
        allowed_tools=["Read", "Write"],
        permission_mode="acceptEdits",
        cwd=Path("/tmp/project"),
        max_turns=3,
    )

    assert config.model_name == "claude-sonnet-4-5"
    assert config.allowed_tools == ["Read", "Write"]
    assert config.permission_mode == "acceptEdits"
    assert config.cwd == Path("/tmp/project")
    assert config.max_turns == 3
    assert config.anthropic_api_key == "test-api-key"


@pytest.mark.asyncio
async def test_run_design_stage_returns_design_spec_from_fenced_json(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=claude_runner.__name__)
    calls: list[dict[str, object]] = []

    async def fake_query(*, prompt, options):
        calls.append({"prompt": prompt, "options": options})
        yield AssistantMessage(
            content=[
                TextBlock(
                    '```json\n'
                    '{"summary":"Build it","files_to_change":["app.py"],'
                    '"steps":["Write code"],"risks":["Low"],'
                    '"test_plan":["Run pytest"]}'
                    '\n```'
                )
            ],
            model="opus",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session-1",
            total_cost_usd=0.42,
            usage={"input_tokens": 10, "output_tokens": 20},
        )

    monkeypatch.setattr(claude_runner, "query", fake_query)

    spec = await ClaudeStageRunner().run_design_stage("Need a thing")

    assert spec == DesignSpec(
        summary="Build it",
        files_to_change=["app.py"],
        steps=["Write code"],
        risks=["Low"],
        test_plan=["Run pytest"],
    )
    assert len(calls) == 1
    assert calls[0]["prompt"] == "Need a thing"
    options = calls[0]["options"]
    assert options.setting_sources == []
    assert options.model == "opus"
    assert options.allowed_tools == ["Read", "Grep", "Glob"]
    assert options.permission_mode == "dontAsk"
    assert options.max_turns == 15
    assert "single JSON object" in options.system_prompt
    assert "summary" in options.system_prompt
    assert "files_to_change" in options.system_prompt
    assert "steps" in options.system_prompt
    assert "risks" in options.system_prompt
    assert "test_plan" in options.system_prompt
    assert "Claude design stage usage" in caplog.text
    assert "0.42" in caplog.text


@pytest.mark.asyncio
async def test_run_design_stage_retries_once_with_corrective_prompt(monkeypatch):
    prompts: list[str] = []

    async def fake_query(*, prompt, options):
        prompts.append(prompt)
        if len(prompts) == 1:
            yield AssistantMessage(content=[TextBlock("not json")], model="opus")
        else:
            yield AssistantMessage(
                content=[
                    TextBlock(
                        '{"summary":"Fixed","files_to_change":[],'
                        '"steps":[],"risks":[],"test_plan":[]}'
                    )
                ],
                model="opus",
            )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=f"session-{len(prompts)}",
        )

    monkeypatch.setattr(claude_runner, "query", fake_query)

    spec = await ClaudeStageRunner().run_design_stage("Need valid output")

    assert spec.summary == "Fixed"
    assert len(prompts) == 2
    assert "Your previous response could not be parsed as the required JSON" in prompts[1]
    assert "Need valid output" in prompts[1]


@pytest.mark.asyncio
async def test_run_design_stage_raises_typed_error_after_retry_parse_failure(monkeypatch):
    async def fake_query(*, prompt, options):
        yield AssistantMessage(content=[TextBlock("still not json")], model="opus")

    monkeypatch.setattr(claude_runner, "query", fake_query)

    with pytest.raises(DesignStageParseError):
        await ClaudeStageRunner().run_design_stage("Need valid output")


def _sample_design_spec() -> DesignSpec:
    return DesignSpec(
        summary="Implement approval gate",
        files_to_change=["hermes/claude_runner.py"],
        steps=["Persist run", "Wait for approval"],
        risks=["Stale approval"],
        test_plan=["pytest tests/test_claude_runner.py"],
    )


def test_design_run_store_persists_pending_approval_record(tmp_path):
    store = DesignRunStore(tmp_path)

    record = store.create_pending_run("Need an approval gate", _sample_design_spec())
    loaded = store.get(record.run_id)

    assert loaded.run_id == record.run_id
    assert loaded.requirements == "Need an approval gate"
    assert loaded.design_spec == _sample_design_spec()
    assert loaded.status == RunStatus.PENDING_APPROVAL
    assert loaded.approver_identity is None
    assert loaded.created_at
    assert loaded.updated_at


def test_design_run_store_approve_reject_and_reject_reason(tmp_path):
    store = DesignRunStore(tmp_path)
    approved = store.create_pending_run("Approve me", _sample_design_spec())
    rejected = store.create_pending_run("Reject me", _sample_design_spec())

    approved = store.approve(approved.run_id, approver_identity="telegram:123/Mike")
    rejected = store.reject(
        rejected.run_id,
        approver_identity="telegram:123/Mike",
        reason="Scope is too broad",
    )

    assert approved.status == RunStatus.APPROVED
    assert approved.approver_identity == "telegram:123/Mike"
    assert approved.approved_at is not None
    assert rejected.status == RunStatus.REJECTED
    assert rejected.approver_identity == "telegram:123/Mike"
    assert rejected.rejection_reason == "Scope is too broad"
    assert rejected.rejected_at is not None


def test_build_stage_refuses_unapproved_runs_and_marks_approved_run_building(tmp_path):
    store = DesignRunStore(tmp_path)
    runner = ClaudeStageRunner(run_store=store)
    record = store.create_pending_run("Need build", _sample_design_spec())

    with pytest.raises(BuildApprovalRequiredError):
        runner.start_build_stage(record.run_id)

    store.approve(record.run_id, approver_identity="telegram:123/Mike")
    building = runner.start_build_stage(record.run_id)

    assert building.status == RunStatus.BUILDING
    assert store.get(record.run_id).status == RunStatus.BUILDING


@pytest.mark.asyncio
async def test_run_design_stage_creates_pending_record_when_requested(monkeypatch, tmp_path):
    async def fake_collect(self, prompt):
        return '{"summary":"Ready","files_to_change":[],"steps":[],"risks":[],"test_plan":[]}'

    monkeypatch.setattr(ClaudeStageRunner, "_collect_design_stage_text", fake_collect)
    store = DesignRunStore(tmp_path)
    runner = ClaudeStageRunner(run_store=store)

    record = await runner.run_design_stage_with_approval("Need persisted design")

    assert record.requirements == "Need persisted design"
    assert record.design_spec.summary == "Ready"
    assert record.status == RunStatus.PENDING_APPROVAL
    assert store.get(record.run_id).status == RunStatus.PENDING_APPROVAL


@pytest.mark.asyncio
async def test_run_build_stage_uses_approved_spec_options_cwd_and_streams_progress(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    progress: list[str] = []

    async def fake_query(*, prompt, options):
        calls.append({"prompt": prompt, "options": options})
        yield AssistantMessage(content=[TextBlock("Editing files now")], model="sonnet")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="build-session-1",
            total_cost_usd=0.12,
            usage={"input_tokens": 5, "output_tokens": 7},
        )

    monkeypatch.setattr(claude_runner, "query", fake_query)
    monkeypatch.setattr(claude_runner, "_target_repo_path_from_config", lambda: tmp_path)
    monkeypatch.setattr(claude_runner, "_git_diff", lambda cwd: "diff --git a/app.py b/app.py\n")

    result = await ClaudeStageRunner().run_build_stage(
        _sample_design_spec(),
        progress_callback=progress.append,
    )

    assert result.diff == "diff --git a/app.py b/app.py\n"
    assert "Editing files now" in progress
    assert "diff --git" in progress[-1]
    assert len(calls) == 1
    options = calls[0]["options"]
    assert options.setting_sources == []
    assert options.model == "sonnet"
    assert options.allowed_tools == ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]
    assert options.permission_mode == "acceptEdits"
    assert Path(options.cwd) == tmp_path
    assert "Bash tool is allowed only for running tests and linters" in options.system_prompt
    assert "Do not commit" in options.system_prompt
    assert "Implement approval gate" in calls[0]["prompt"]
    assert "files_to_change" in calls[0]["prompt"]


def test_target_repo_path_from_config_requires_explicit_path(monkeypatch):
    monkeypatch.setattr(claude_runner, "_load_hermes_config", lambda: {})

    with pytest.raises(claude_runner.BuildConfigError):
        claude_runner._target_repo_path_from_config()


def test_target_repo_path_from_config_reads_claude_runner_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        claude_runner,
        "_load_hermes_config",
        lambda: {"claude_runner": {"target_repo_path": str(tmp_path)}},
    )

    assert claude_runner._target_repo_path_from_config() == tmp_path


@pytest.mark.asyncio
async def test_run_review_stage_uses_read_only_options_parses_json_and_posts_verdict(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    progress: list[str] = []

    async def fake_query(*, prompt, options):
        calls.append({"prompt": prompt, "options": options})
        yield AssistantMessage(
            content=[TextBlock('{"verdict":"fail","findings":["Missing regression test"]}')],
            model="sonnet",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="review-session-1",
            total_cost_usd=0.05,
            usage={"input_tokens": 3, "output_tokens": 4},
        )

    monkeypatch.setattr(claude_runner, "query", fake_query)
    monkeypatch.setattr(claude_runner, "_target_repo_path_from_config", lambda: tmp_path)

    diff = "diff --git a/app.py b/app.py\n"
    result = await ClaudeStageRunner().run_review_stage(
        _sample_design_spec(),
        diff,
        progress_callback=progress.append,
    )

    assert result.verdict == "fail"
    assert result.findings == ["Missing regression test"]
    assert len(calls) == 1
    options = calls[0]["options"]
    assert options.setting_sources == []
    assert options.model == "sonnet"
    assert options.allowed_tools == ["Read", "Glob", "Grep"]
    assert options.permission_mode == "dontAsk"
    assert Path(options.cwd) == tmp_path
    assert "verify the diff implements the spec" in options.system_prompt
    assert "security issues" in options.system_prompt
    assert "missing tests" in options.system_prompt
    assert "verdict" in options.system_prompt
    assert "findings" in options.system_prompt
    assert "diff --git" in calls[0]["prompt"]
    assert "Implement approval gate" in calls[0]["prompt"]
    assert any("Review verdict: fail" in message for message in progress)
    assert any("diff --git" in message for message in progress)


@pytest.mark.asyncio
async def test_run_review_stage_strips_fenced_json(monkeypatch, tmp_path):
    async def fake_query(*, prompt, options):
        yield AssistantMessage(
            content=[TextBlock('```json\n{"verdict":"pass","findings":[]}\n```')],
            model="sonnet",
        )

    monkeypatch.setattr(claude_runner, "query", fake_query)
    monkeypatch.setattr(claude_runner, "_target_repo_path_from_config", lambda: tmp_path)

    result = await ClaudeStageRunner().run_review_stage(_sample_design_spec(), "diff")

    assert result.verdict == "pass"
    assert result.findings == []
