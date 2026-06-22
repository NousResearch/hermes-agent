"""Tests for the native /ooo gateway router core."""

from __future__ import annotations

import asyncio
from typing import Any

from gateway.ouroboros_native import (
    OOO_SUBCOMMANDS,
    OooNativeContext,
    handle_ooo_native,
    parse_ooo_command,
)
from gateway.ouroboros_state import OooStateContext, OooStateStore


class FakeMcpCaller:
    def __init__(self, responses: dict[str, dict[str, Any]] | None = None, raises: Exception | None = None):
        self.responses = responses or {}
        self.raises = raises
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def __call__(self, tool_name: str, args: dict[str, Any] | None = None, timeout: float = 45.0):
        call_args = dict(args or {})
        self.calls.append((tool_name, call_args, timeout))
        if self.raises is not None:
            raise self.raises
        return dict(self.responses.get(tool_name, {"success": True, "tool": tool_name}))


def _state_context() -> OooStateContext:
    return OooStateContext(
        platform="discord",
        guild_id="guild-1",
        channel_id="channel-1",
        thread_id="thread-1",
        user_id="user-1",
        profile="default",
    )


def _native_ctx(
    tmp_path,
    fake: FakeMcpCaller,
    *,
    cwd: str | None = "/repo",
    allow_mutating_side_effects: bool = False,
):
    state_context = _state_context()
    store = OooStateStore(tmp_path / "ooo-state.json")
    return OooNativeContext(
        cwd=cwd,
        state_context=state_context,
        state_store=store,
        mcp_caller=fake,
        allow_mutating_side_effects=allow_mutating_side_effects,
        idempotency_key="idem-1",
    ), store, state_context


def _run(raw: str, ctx: OooNativeContext):
    return asyncio.run(handle_ooo_native(raw, ctx))


def test_subcommands_include_registered_discord_commands_and_router_job():
    expected = {
        "help",
        "interview",
        "seed",
        "run",
        "evaluate",
        "status",
        "pm",
        "qa",
        "unstuck",
        "evolve",
        "ralph",
        "auto",
        "cancel",
        "resume-session",
        "setup",
        "config",
        "brownfield",
        "publish",
        "welcome",
        "tutorial",
        "update",
    }

    assert set(OOO_SUBCOMMANDS) == expected
    assert parse_ooo_command("job status job-1").name == "job"


def test_cli_registry_reuses_native_ooo_subcommands():
    from hermes_cli.commands import COMMAND_REGISTRY

    command = next(cmd for cmd in COMMAND_REGISTRY if cmd.name == "ooo")

    assert command.subcommands == OOO_SUBCOMMANDS


def test_parse_aliases_and_normalization():
    assert parse_ooo_command("").name == "help"
    assert parse_ooo_command("h").name == "help"
    assert parse_ooo_command("?").name == "help"
    assert parse_ooo_command("init build a service").name == "interview"
    assert parse_ooo_command("resume session-1").name == "resume-session"

    parsed = parse_ooo_command("/ooo RUN --skip_qa")

    assert parsed.name == "run"
    assert parsed.tokens == ["--skip_qa"]
    assert parsed.args_text == "--skip_qa"


def test_malformed_quotes_return_usage_without_mcp_call(tmp_path):
    fake = FakeMcpCaller()
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    response = _run('interview "unterminated', ctx)

    assert fake.calls == []
    assert "Traceback" not in response.text
    assert "따옴표" in response.text or "quote" in response.text.lower()
    assert "사용" in response.text or "usage" in response.text.lower()


def test_help_is_native_and_does_not_call_mcp(tmp_path):
    fake = FakeMcpCaller()
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    response = _run("help", ctx)

    assert fake.calls == []
    assert response.used_tool is None
    assert "/ooo" in response.text
    assert "interview" in response.text


def test_interview_start_calls_mcp_includes_cwd_and_stores_interview_session(tmp_path):
    fake = FakeMcpCaller({"ouroboros_interview": {"success": True, "session_id": "iv-1"}})
    ctx, store, scope = _native_ctx(tmp_path, fake)

    response = _run("/ooo interview build native router", ctx)

    assert fake.calls == [
        (
            "ouroboros_interview",
            {"initial_context": "build native router", "cwd": "/repo"},
            45.0,
        )
    ]
    assert response.used_tool == "ouroboros_interview"
    assert "iv-1" in response.text
    loaded = store.load(scope)
    assert loaded.interview_session_id == "iv-1"
    assert loaded.last_session_id == "iv-1"


def test_interview_response_includes_safe_question_payload_summary(tmp_path):
    fake = FakeMcpCaller(
        {
            "ouroboros_interview": {
                "success": True,
                "session_id": "iv-1",
                "question": "\x1b[31mWhat problem should we solve next?\x1b[0m",
            }
        }
    )
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    response = _run("interview build native router", ctx)

    assert "What problem should we solve next?" in response.text
    assert "\x1b[" not in response.text


def test_interview_answer_uses_recent_interview_session(tmp_path):
    fake = FakeMcpCaller({"ouroboros_interview": {"success": True, "session_id": "iv-2"}})
    ctx, store, scope = _native_ctx(tmp_path, fake)
    store.update(scope, interview_session_id="iv-1")

    response = _run("interview answer clarified requirements", ctx)

    assert fake.calls == [
        (
            "ouroboros_interview",
            {"session_id": "iv-1", "answer": "clarified requirements"},
            45.0,
        )
    ]
    assert response.state_updates == {"last_session_id": "iv-2", "interview_session_id": "iv-2"}


def test_pm_start_and_generate_route_and_store_pm_session(tmp_path):
    fake = FakeMcpCaller({"ouroboros_pm_interview": {"success": True, "session_id": "pm-1"}})
    ctx, store, scope = _native_ctx(tmp_path, fake)

    start_response = _run("pm product discovery", ctx)
    generate_response = _run("pm generate", ctx)

    assert fake.calls == [
        (
            "ouroboros_pm_interview",
            {"initial_context": "product discovery", "cwd": "/repo"},
            45.0,
        ),
        (
            "ouroboros_pm_interview",
            {"session_id": "pm-1", "action": "generate"},
            45.0,
        ),
    ]
    assert "pm-1" in start_response.text
    assert generate_response.used_tool == "ouroboros_pm_interview"
    loaded = store.load(scope)
    assert loaded.pm_session_id == "pm-1"
    assert loaded.last_session_id == "pm-1"


def test_seed_uses_recent_interview_session_and_force_flag(tmp_path):
    fake = FakeMcpCaller({"ouroboros_generate_seed": {"success": True, "seed_id": "seed-1"}})
    ctx, store, scope = _native_ctx(tmp_path, fake)
    store.update(scope, interview_session_id="iv-1", pm_session_id="pm-1")

    response = _run("seed --force", ctx)

    assert fake.calls == [
        ("ouroboros_generate_seed", {"session_id": "iv-1", "force": True}, 45.0)
    ]
    assert response.used_tool == "ouroboros_generate_seed"
    assert store.load(scope).last_seed_id == "seed-1"


def test_run_uses_start_execute_seed_passes_idempotency_and_stores_job(tmp_path):
    fake = FakeMcpCaller({"ouroboros_start_execute_seed": {"success": True, "job_id": "job-1"}})
    ctx, store, scope = _native_ctx(tmp_path, fake)

    response = _run(
        "run --seed-path seed.yaml --session sess-1 --max-iterations 3 --skip-qa",
        ctx,
    )

    assert fake.calls == [
        (
            "ouroboros_start_execute_seed",
            {
                "seed_path": "seed.yaml",
                "session_id": "sess-1",
                "cwd": "/repo",
                "max_iterations": 3,
                "skip_qa": True,
                "idempotency_key": "idem-1",
            },
            45.0,
        )
    ]
    assert "/ooo status --job job-1" in response.text
    assert store.load(scope).last_job_id == "job-1"


def test_evaluate_positional_and_flag_paths_use_start_evaluate(tmp_path):
    fake = FakeMcpCaller({"ouroboros_start_evaluate": {"success": True, "job_id": "eval-job"}})
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    positional = _run("evaluate sess-1 artifact.txt --consensus", ctx)
    flagged = _run("evaluate --session sess-2 --artifact report.md", ctx)

    assert fake.calls == [
        (
            "ouroboros_start_evaluate",
            {
                "session_id": "sess-1",
                "artifact": "artifact.txt",
                "trigger_consensus": True,
                "working_dir": "/repo",
            },
            45.0,
        ),
        (
            "ouroboros_start_evaluate",
            {"session_id": "sess-2", "artifact": "report.md", "working_dir": "/repo"},
            45.0,
        ),
    ]
    assert positional.used_tool == "ouroboros_start_evaluate"
    assert "/ooo status --job eval-job" in positional.text
    assert flagged.used_tool == "ouroboros_start_evaluate"


def test_qa_uses_default_quality_bar_when_omitted(tmp_path):
    fake = FakeMcpCaller({"ouroboros_qa": {"success": True, "score": 0.9}})
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    response = _run("qa --artifact answer.md", ctx)

    assert fake.calls == [
        (
            "ouroboros_qa",
            {
                "artifact": "answer.md",
                "quality_bar": "General correctness, completeness, and actionable quality.",
                "artifact_type": "code",
            },
            45.0,
        )
    ]
    assert "quality_bar" in response.payload or "Quality bar" in response.text


def test_qa_response_includes_safe_score_and_verdict_summary(tmp_path):
    fake = FakeMcpCaller(
        {"ouroboros_qa": {"success": True, "score": 0.91, "verdict": "pass", "passed": True}}
    )
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    response = _run("qa --artifact answer.md", ctx)

    assert "score=0.91" in response.text
    assert "verdict=pass" in response.text
    assert "passed=True" in response.text


def test_status_and_job_commands_route_with_recent_job_fallback(tmp_path):
    fake = FakeMcpCaller(
        {
            "ouroboros_job_status": {"success": True},
            "ouroboros_job_result": {"success": True, "job_id": "job-2"},
        }
    )
    ctx, store, scope = _native_ctx(tmp_path, fake)
    store.update(scope, last_job_id="recent-job", last_session_id="sess-1")

    status = _run("status --job job-1", ctx)
    recent = _run("job status", ctx)
    result = _run("job result job-2", ctx)

    assert fake.calls == [
        ("ouroboros_job_status", {"job_id": "job-1"}, 45.0),
        ("ouroboros_job_status", {"job_id": "recent-job"}, 45.0),
        ("ouroboros_job_result", {"job_id": "job-2"}, 45.0),
    ]
    assert status.used_tool == "ouroboros_job_status"
    assert "최근" in recent.text or "recent" in recent.text.lower()
    assert result.used_tool == "ouroboros_job_result"


def test_job_wait_status_session_and_unstuck_routes(tmp_path):
    fake = FakeMcpCaller(
        {
            "ouroboros_job_wait": {"success": True, "status": "running"},
            "ouroboros_session_status": {"success": True, "status": "active"},
            "ouroboros_lateral_think": {"success": True, "summary": "try a smaller slice"},
        }
    )
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    wait = _run("job wait job-1 --timeout 7 --cursor 4 --view summary --stream progress --wait-for raw", ctx)
    session = _run("status --session sess-1", ctx)
    unstuck = _run(
        'unstuck --problem "stuck debugging" --persona hacker --approach "retrying" --stagnation-pattern loop',
        ctx,
    )

    assert fake.calls == [
        (
            "ouroboros_job_wait",
            {
                "job_id": "job-1",
                "cursor": 4,
                "timeout_seconds": 7,
                "view": "summary",
                "stream": "progress",
                "wait_for": "raw",
            },
            45.0,
        ),
        ("ouroboros_session_status", {"session_id": "sess-1"}, 45.0),
        (
            "ouroboros_lateral_think",
            {
                "problem_context": "stuck debugging",
                "current_approach": "retrying",
                "persona": "hacker",
                "stagnation_pattern": "loop",
            },
            45.0,
        ),
    ]
    assert wait.used_tool == "ouroboros_job_wait"
    assert session.used_tool == "ouroboros_session_status"
    assert unstuck.used_tool == "ouroboros_lateral_think"


def test_cancel_requires_explicit_id_then_routes_job_and_execution(tmp_path):
    fake = FakeMcpCaller(
        {
            "ouroboros_cancel_job": {"success": True},
            "ouroboros_cancel_execution": {"success": True},
        }
    )
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    blocked = _run("cancel", ctx)
    by_job = _run("cancel --job job-1", ctx)
    by_execution = _run("cancel --execution exec-1", ctx)

    assert fake.calls == [
        ("ouroboros_cancel_job", {"job_id": "job-1"}, 45.0),
        ("ouroboros_cancel_execution", {"execution_id": "exec-1"}, 45.0),
    ]
    assert "명시" in blocked.text or "explicit" in blocked.text.lower()
    assert by_job.used_tool == "ouroboros_cancel_job"
    assert by_execution.used_tool == "ouroboros_cancel_execution"


def test_evolve_ralph_and_auto_route_to_start_tools(tmp_path):
    fake = FakeMcpCaller(
        {
            "ouroboros_start_evolve_step": {"success": True, "job_id": "evolve-job"},
            "ouroboros_start_ralph": {"success": True, "job_id": "ralph-job"},
            "ouroboros_start_auto": {"success": True, "auto_session_id": "auto-1", "job_id": "auto-job"},
        }
    )
    ctx, store, scope = _native_ctx(tmp_path, fake)

    evolve = _run("evolve line-1 --skip-qa", ctx)
    ralph = _run("ralph line-2 --max-generations 2", ctx)
    auto = _run("auto build the product --complete-product", ctx)

    assert fake.calls == [
        (
            "ouroboros_start_evolve_step",
            {"lineage_id": "line-1", "skip_qa": True, "project_dir": "/repo"},
            45.0,
        ),
        (
            "ouroboros_start_ralph",
            {"lineage_id": "line-2", "max_generations": 2, "project_dir": "/repo"},
            45.0,
        ),
        (
            "ouroboros_start_auto",
            {"goal": "build the product", "cwd": "/repo", "complete_product": True},
            45.0,
        ),
    ]
    assert evolve.used_tool == "ouroboros_start_evolve_step"
    assert ralph.used_tool == "ouroboros_start_ralph"
    assert store.load(scope).auto_session_id == "auto-1"
    assert store.load(scope).last_job_id == "auto-job"
    assert "/ooo status --job auto-job" in auto.text


def test_evolve_max_generations_is_usage_error_and_does_not_call_mcp(tmp_path):
    fake = FakeMcpCaller({"ouroboros_start_evolve_step": {"success": True, "job_id": "evolve-job"}})
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    response = _run("evolve line-1 --max-generations 3", ctx)

    assert fake.calls == []
    assert response.used_tool is None
    assert "ralph" in response.text.lower()


def test_brownfield_default_allows_only_read_queries_and_blocks_mutating_forms(tmp_path):
    fake = FakeMcpCaller({"ouroboros_brownfield": {"success": True, "repos": []}})
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    query = _run("brownfield query --limit 5 --offset 2", ctx)
    list_alias = _run("brownfield list", ctx)
    default_only_action = _run("brownfield default-only", ctx)
    default_only_flag = _run("brownfield --default-only", ctx)
    scan = _run("brownfield scan --root /tmp/repos", ctx)
    default = _run("brownfield default", ctx)
    path = _run("brownfield query --path /tmp/repo", ctx)

    assert fake.calls == [
        ("ouroboros_brownfield", {"action": "query", "offset": 2, "limit": 5}, 45.0),
        ("ouroboros_brownfield", {"action": "query"}, 45.0),
        ("ouroboros_brownfield", {"action": "query", "default_only": True}, 45.0),
        ("ouroboros_brownfield", {"action": "query", "default_only": True}, 45.0),
    ]
    assert query.used_tool == "ouroboros_brownfield"
    assert list_alias.used_tool == "ouroboros_brownfield"
    assert default_only_action.used_tool == "ouroboros_brownfield"
    assert default_only_flag.used_tool == "ouroboros_brownfield"
    for blocked in (scan, default, path):
        assert blocked.used_tool is None
        assert blocked.payload and blocked.payload.get("blocked") is True
        assert "중단" in blocked.text or "stop" in blocked.text.lower()


def test_brownfield_scan_can_route_when_mutating_side_effects_are_allowed(tmp_path):
    fake = FakeMcpCaller({"ouroboros_brownfield": {"success": True, "repos": []}})
    ctx, _store, _scope = _native_ctx(tmp_path, fake, allow_mutating_side_effects=True)

    response = _run("brownfield scan --root /tmp/repos", ctx)

    assert fake.calls == [
        ("ouroboros_brownfield", {"action": "scan", "scan_root": "/tmp/repos"}, 45.0)
    ]
    assert response.used_tool == "ouroboros_brownfield"


def test_stop_gated_and_static_commands_do_not_call_mcp(tmp_path):
    fake = FakeMcpCaller()
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    stop_texts = [_run(command, ctx).text for command in ("setup", "config", "update", "publish")]
    static_texts = [_run(command, ctx).text for command in ("welcome", "tutorial", "resume-session")]

    assert fake.calls == []
    assert all("중단" in text or "stop" in text.lower() or "승인" in text for text in stop_texts)
    assert all(text for text in static_texts)


def test_mcp_exception_returns_safe_error_without_raw_traceback(tmp_path):
    fake = FakeMcpCaller(raises=RuntimeError("boom from fake"))
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    response = _run("interview hello", ctx)

    assert fake.calls[0][0] == "ouroboros_interview"
    assert response.used_tool == "ouroboros_interview"
    assert "boom from fake" in response.text
    assert "Traceback" not in response.text


def test_mcp_exception_redacts_secret_like_text(tmp_path):
    raw_key = "sk-abc...3456"
    raw_credential = ("tok" + "en=") + "super-secret-value"
    fake = FakeMcpCaller(raises=RuntimeError(f"failed with {raw_key} and {raw_credential}"))
    ctx, _store, _scope = _native_ctx(tmp_path, fake)

    response = _run("interview hello", ctx)
    assert raw_key not in response.text
    assert raw_credential not in response.text
    assert "Traceback" not in response.text
    assert response.payload is not None
    assert raw_key not in response.payload.get("error", "")
    assert raw_credential not in response.payload.get("error", "")


def test_mcp_call_runs_through_asyncio_to_thread(tmp_path, monkeypatch):
    fake = FakeMcpCaller({"ouroboros_interview": {"success": True, "session_id": "iv-thread"}})
    ctx, _store, _scope = _native_ctx(tmp_path, fake)
    to_thread_calls = []

    async def fake_to_thread(func, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr("gateway.ouroboros_native.asyncio.to_thread", fake_to_thread)

    response = _run("interview hello", ctx)

    assert response.used_tool == "ouroboros_interview"
    assert len(to_thread_calls) == 1
    assert to_thread_calls[0][1] == (
        "ouroboros_interview",
        {"initial_context": "hello", "cwd": "/repo"},
        45.0,
    )
