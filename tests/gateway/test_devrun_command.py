import asyncio
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


def _make_event(text="/devrun fix test", repo=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
    )
    if repo is not None:
        text = f"/devrun repo={repo} task=fix test"
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    runner._reply_anchor_for_event = lambda event: event.message_id
    return runner


class _StubAdapter(BasePlatformAdapter):
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def test_parse_devrun_args_plain():
    from gateway.devrun import parse_devrun_args

    parsed = parse_devrun_args("fix failing test")
    assert parsed.repo is None
    assert parsed.task == "fix failing test"


def test_parse_devrun_args_repo_and_task():
    from gateway.devrun import parse_devrun_args

    parsed = parse_devrun_args("repo=/tmp/project task=fix failing test")
    assert parsed.repo == "/tmp/project"
    assert parsed.task == "fix failing test"


def test_parse_devrun_args_board_repo_and_task():
    from gateway.devrun import parse_devrun_args

    parsed = parse_devrun_args("board=yug65-devflow repo=/tmp/project task=fix failing test")
    assert parsed.board == "yug65-devflow"
    assert parsed.repo == "/tmp/project"
    assert parsed.task == "fix failing test"


def test_resolve_kanban_board_requires_existing_board(tmp_path, monkeypatch):
    from gateway.devrun import resolve_kanban_board
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    with pytest.raises(ValueError, match="does not exist"):
        resolve_kanban_board("missing-board")

    kb.create_board("project-board")

    assert resolve_kanban_board("project-board") == "project-board"
    assert resolve_kanban_board(None) is None


def test_classify_risk():
    from gateway.devrun import HIGH_RISK, LOW_RISK, MEDIUM_RISK, classify_risk

    assert classify_risk("fix typo") == LOW_RISK
    assert classify_risk("refactor auth integration") == MEDIUM_RISK
    assert classify_risk("我想做一个登录功能") == MEDIUM_RISK
    assert classify_risk("restart gateway and edit .env") == HIGH_RISK


def test_read_only_task_mode_and_prompts():
    from gateway.devrun import DevRunJob, build_execution_prompt, build_review_packet, is_read_only_task

    job = DevRunJob(
        job_id="dev_ro",
        raw_task="只读评估 README，不要修改文件",
        task="只读评估 README，不要修改文件",
        repo="/tmp/project",
        risk="medium",
        status="reviewing",
    )

    assert is_read_only_task(job.task)
    review_packet = build_review_packet(job)
    prompt = build_execution_prompt(job)

    assert "Mode: read_only" in review_packet
    assert "no file writes" in review_packet
    assert "make low-risk scoped edits" not in review_packet
    assert "Mode: read_only" in prompt
    assert "Do not write, edit, rename, delete" in prompt
    assert "Low-risk code edits may be applied" not in prompt


def test_commands_are_registered():
    from hermes_cli.commands import resolve_command

    assert resolve_command("devrun").name == "devrun"
    assert resolve_command("devreview").name == "devreview"
    assert resolve_command("devflow").name == "devflow"
    assert resolve_command("devstatus").name == "devstatus"
    assert resolve_command("devcancel").name == "devcancel"


def test_collect_repo_changes(tmp_path):
    from gateway.devrun import collect_repo_changes

    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    changed = tmp_path / "changed.txt"
    changed.write_text("hello", encoding="utf-8")

    files, summary = collect_repo_changes(tmp_path)

    assert "changed.txt" in files
    assert summary == "git status --porcelain=v1"


def test_sidecar_output_text_reads_worker_deliverable(tmp_path):
    from gateway.devrun import _sidecar_output_text, extract_review_verdict

    deliverable = tmp_path / "deliverable.md"
    deliverable.write_text("Verdict: PASS\n\nLooks bounded.", encoding="utf-8")
    stdout = json.dumps({"ok": True, "outputPath": str(deliverable)})

    text = _sidecar_output_text(stdout)

    assert "Verdict: PASS" in text
    assert extract_review_verdict(text) == "PASS"


def test_build_devreview_packet_is_review_only(tmp_path):
    from gateway.devrun import DevRunJob, build_devreview_packet

    (tmp_path / "README.md").write_text("# Demo\nsafe line\napi_key = should-not-leak\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"scripts":{"test":"echo ok"}}', encoding="utf-8")
    job = DevRunJob(
        job_id="dev_review_packet",
        raw_task="review the feature",
        task="review the feature",
        repo=str(tmp_path),
        risk="medium",
        status="reviewing",
    )

    packet = build_devreview_packet(job)

    assert "# DevReview Packet" in packet
    assert "review-only" in packet
    assert "Do not modify files" in packet
    assert "Requirements" in packet
    assert "Architecture" in packet
    assert "Testing" in packet
    assert "Security" in packet
    assert "Final acceptance" in packet
    assert "README.md" in packet
    assert "package.json" in packet
    assert "safe line" in packet
    assert "should-not-leak" not in packet
    assert "[REDACTED possible secret/config line]" in packet


def test_run_devreview_council_combines_worker_reports(tmp_path, monkeypatch):
    from gateway.devrun import DevRunJob, run_devreview_council

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    job = DevRunJob(
        job_id="dev_review_council",
        raw_task="review the feature",
        task="review the feature",
        repo=str(tmp_path),
        risk="medium",
        status="reviewing",
    )

    def fake_worker(job, worker, label, packet_path, timeout):
        assert packet_path.exists()
        return label, f"{worker} report ok"

    with patch("gateway.devrun.run_devreview_worker", side_effect=fake_worker):
        report = run_devreview_council(job, timeout_per_worker=1)

    assert "# DevReview Council Report" in report
    assert "product report ok" in report
    assert "architecture report ok" in report
    assert "testplan report ok" in report
    assert "review report ok" in report


def test_run_devflow_preflight_uses_single_review_worker(tmp_path, monkeypatch):
    from gateway.devrun import DevRunJob, run_devflow_preflight

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    job = DevRunJob(
        job_id="dev_flow_preflight",
        raw_task="review the feature",
        task="review the feature",
        repo=str(tmp_path),
        risk="medium",
        status="reviewing",
    )

    calls = []

    def fake_worker(job, worker, label, packet_path, timeout):
        calls.append((worker, label, timeout))
        return label, "quick report ok"

    with patch("gateway.devrun.run_devreview_worker", side_effect=fake_worker):
        report = run_devflow_preflight(job, timeout=7)

    assert len(calls) == 1
    assert calls[0][0] == "review"
    assert calls[0][2] == 7
    assert "# DevFlow Preflight Report" in report
    assert "quick report ok" in report


def test_devreview_worker_uses_aux_fallback_on_quota_error(tmp_path, monkeypatch):
    from gateway.devrun import DevRunJob, run_devreview_worker

    packet = tmp_path / "packet.md"
    packet.write_text("# Packet\nread-only review", encoding="utf-8")
    job = DevRunJob(
        job_id="dev_review_fallback",
        raw_task="review safely",
        task="review safely",
        repo=str(tmp_path),
        risk="high",
        status="reviewing",
    )

    failed = subprocess.CompletedProcess(
        args=["node"],
        returncode=1,
        stdout=json.dumps({
            "ok": False,
            "error": "OpenAI chat request failed (HTTP 403): 用户额度不足, 剩余额度: 0",
        }),
        stderr="",
    )
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Verdict: NEEDS_CHANGES\n\nFallback review ok."

    with patch("gateway.devrun.find_worker_command", return_value=["node", "worker.mjs"]), patch(
        "gateway.devrun.subprocess.run", return_value=failed
    ), patch("agent.auxiliary_client.call_llm", return_value=response) as call_llm:
        label, report = run_devreview_worker(job, "review", "快速门禁", packet, timeout=7)

    assert label == "快速门禁"
    assert "Fallback reviewer used" in report
    assert "Fallback review ok" in report
    call_llm.assert_called_once()
    assert call_llm.call_args.kwargs["task"] == "devreview_fallback"


def test_devreview_worker_returns_blocked_when_aux_fallback_also_unavailable(tmp_path):
    from gateway.devrun import DevRunJob, run_devreview_worker

    packet = tmp_path / "packet.md"
    packet.write_text("# Packet\nread-only review", encoding="utf-8")
    job = DevRunJob(
        job_id="dev_review_fallback_blocked",
        raw_task="review safely",
        task="review safely",
        repo=str(tmp_path),
        risk="high",
        status="reviewing",
    )

    failed = subprocess.CompletedProcess(
        args=["node"],
        returncode=1,
        stdout=json.dumps({
            "ok": False,
            "error": "OpenAI chat request failed (HTTP 403): 用户额度不足, 剩余额度: 0",
        }),
        stderr="",
    )

    with patch("gateway.devrun.find_worker_command", return_value=["node", "worker.mjs"]), patch(
        "gateway.devrun.subprocess.run", return_value=failed
    ), patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("empty fallback")):
        label, report = run_devreview_worker(job, "review", "快速门禁", packet, timeout=7)

    assert label == "快速门禁"
    assert report.startswith("WORKER_FAILED:")
    assert "Verdict: BLOCKED" in report
    assert "DevReview reviewer model channel unavailable" in report
    assert "用户额度不足" in report


def test_devreview_worker_does_not_aux_fallback_on_review_content_error(tmp_path):
    from gateway.devrun import DevRunJob, run_devreview_worker

    packet = tmp_path / "packet.md"
    packet.write_text("# Packet\nread-only review", encoding="utf-8")
    job = DevRunJob(
        job_id="dev_review_no_fallback",
        raw_task="review safely",
        task="review safely",
        repo=str(tmp_path),
        risk="high",
        status="reviewing",
    )

    failed = subprocess.CompletedProcess(
        args=["node"],
        returncode=1,
        stdout=json.dumps({"ok": False, "error": "review packet had insufficient evidence"}),
        stderr="",
    )

    with patch("gateway.devrun.find_worker_command", return_value=["node", "worker.mjs"]), patch(
        "gateway.devrun.subprocess.run", return_value=failed
    ), patch("agent.auxiliary_client.call_llm") as call_llm:
        _label, report = run_devreview_worker(job, "review", "快速门禁", packet, timeout=7)

    assert report.startswith("WORKER_FAILED:")
    assert "insufficient evidence" in report
    call_llm.assert_not_called()


def test_devflow_jobs_render_with_devflow_label(tmp_path, monkeypatch):
    from gateway.devrun import DevRunJob, render_jobs, summarize_job

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    job = DevRunJob(
        job_id="dev_flow_label",
        raw_task="complex project",
        task="complex project",
        repo=str(tmp_path),
        risk="medium",
        status="reviewing",
        review_summary="DevFlow route: Kanban triage. DevReview council queued.",
    )

    summary = summarize_job(job)
    listing = render_jobs([job])

    assert summary.startswith("✦ **DevFlow Command Center**")
    assert "状态  预审中" in summary
    assert listing.startswith("DevFlow dev_flow_label")


def test_estimate_devflow_budget_mentions_minimax_calls():
    from gateway.devrun import (
        _extract_devflow_budget,
        estimate_devflow_budget,
        format_devflow_budget_audit,
        format_devflow_budget_estimate,
    )

    estimate = estimate_devflow_budget(
        "complex full end-to-end project",
        "medium",
        "kanban",
        sidecar_available=True,
    )
    text = format_devflow_budget_estimate(estimate)

    assert estimate.agent_count_min >= 2
    assert estimate.minimax_calls_max > estimate.minimax_calls_min
    assert "Estimated agents:" in text
    assert "Estimated MiniMax-M3 calls:" in text

    audit = format_devflow_budget_audit(estimate)
    assert "DevFlow Budget Gate Evidence" in audit
    assert "Estimated agents: 2-" in audit
    assert "Estimated MiniMax-M3 calls: 3-" in audit
    assert "pre-spend estimate and mobile approval gate" in audit
    assert _extract_devflow_budget(audit)[0].startswith("2-")
    assert _extract_devflow_budget(audit)[1].startswith("3-")


def test_devreview_report_failed_when_all_workers_fail():
    from gateway.devrun import devreview_report_failed

    assert devreview_report_failed("- Worker status: all workers failed\n\nWORKER_FAILED: quota unavailable")
    assert not devreview_report_failed("- Worker status: 1 worker(s) failed\n\npartial report")


def test_extract_worker_error_reads_receipt_event(tmp_path):
    from gateway.devrun import _extract_worker_error

    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "events": {
                    "events": [
                        {
                            "payload": {
                                "error_message": "OpenAI chat request failed (HTTP 403): 用户额度不足, 剩余额度: ＄0.000000"
                            }
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    meta = json.dumps({"receiptPath": str(receipt)})

    assert "用户额度不足" in _extract_worker_error(meta)


def test_collect_new_repo_changes_excludes_baseline(tmp_path):
    from gateway.devrun import DevRunJob, collect_new_repo_changes, record_repo_baseline

    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    existing = tmp_path / "existing.txt"
    existing.write_text("before", encoding="utf-8")
    job = DevRunJob(
        job_id="dev_delta",
        raw_task="fix",
        task="fix",
        repo=str(tmp_path),
        risk="low",
        status="running",
    )

    record_repo_baseline(job)
    new_file = tmp_path / "new.txt"
    new_file.write_text("after", encoding="utf-8")
    files, summary = collect_new_repo_changes(job)

    assert "existing.txt" not in files
    assert "new.txt" in files
    assert "git status --porcelain=v1" in summary


def test_extract_review_verdict_prefers_verdict_section():
    from gateway.devrun import extract_review_verdict

    text = """
    Use verdict exactly as one of: PASS, NEEDS_CHANGES, or BLOCKED.

    ## 1. Verdict

    **NEEDS_CHANGES**
    """

    assert extract_review_verdict(text) == "NEEDS_CHANGES"


def test_extract_review_verdict_handles_explicit_verdict_line():
    from gateway.devrun import extract_review_verdict

    text = """
    ## 1. 结论

    **Verdict: NEEDS_CHANGES**

    This is not BLOCKED.
    """

    assert extract_review_verdict(text) == "NEEDS_CHANGES"


def test_extract_review_verdict_treats_pass_with_risk_as_pass():
    from gateway.devrun import extract_review_verdict

    assert extract_review_verdict("## Verdict\n\nPASS_WITH_RISK") == "PASS"


def test_repo_path_must_be_inside_allowed_root(tmp_path, monkeypatch):
    from gateway.devrun import resolve_repo_path

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    project = allowed / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(allowed))

    assert resolve_repo_path(str(project)) == project.resolve()
    with pytest.raises(ValueError, match="outside DevRun allowed roots"):
        resolve_repo_path(str(outside))


def test_repo_path_rejects_sensitive_home_dir(tmp_path, monkeypatch):
    from gateway.devrun import resolve_repo_path

    home = tmp_path / "home"
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(home))

    with pytest.raises(ValueError, match="sensitive home directory"):
        resolve_repo_path(str(ssh_dir))


def test_job_files_are_private(tmp_path, monkeypatch):
    from gateway.devrun import DevRunJob, save_job

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    job = DevRunJob(
        job_id="dev_perm",
        raw_task="fix",
        task="fix",
        repo=str(tmp_path),
        risk="low",
        status="queued",
    )

    save_job(job)
    job_path = tmp_path / ".hermes" / "devrun" / "jobs" / "dev_perm.json"

    assert job_path.exists()
    if hasattr(job_path, "stat"):
        assert (job_path.stat().st_mode & 0o777) == 0o600


@pytest.mark.asyncio
async def test_handle_devrun_starts_background_job(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    created = []

    def capture_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        created.append(task)
        return task

    with patch("gateway.slash_commands.asyncio.create_task", side_effect=capture_task):
        event = _make_event(repo=tmp_path)
        result = await runner._handle_devrun_command(event)

    assert "DevRun job started:" in result
    assert "Risk: low" in result
    assert str(tmp_path) in result
    assert len(created) == 1


@pytest.mark.asyncio
async def test_handle_devreview_starts_review_job(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    created = []

    def capture_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        created.append(task)
        return task

    with (
        patch("gateway.devrun.find_sidecar_command", return_value=["node", "worker.mjs", "--worker", "review"]),
        patch("gateway.slash_commands.asyncio.create_task", side_effect=capture_task),
    ):
        event = _make_event(f"/devreview repo={tmp_path} task=review architecture")
        result = await runner._handle_devreview_command(event)

    assert "DevReview job started:" in result
    assert "Status: reviewing" in result
    assert "Lenses: requirements, architecture, testing, security, final acceptance" in result
    assert len(created) == 1


@pytest.mark.asyncio
async def test_handle_devflow_low_risk_routes_to_devrun(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    created = []

    def capture_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        created.append(task)
        return task

    with patch("gateway.slash_commands.asyncio.create_task", side_effect=capture_task):
        event = _make_event(f"/devflow repo={tmp_path} task=fix typo")
        result = await runner._handle_devflow_command(event)

    assert "🧭 **DevFlow 已接管**" in result
    assert "低风险小任务" in result
    assert "风险：低风险" in result
    assert len(created) == 1


@pytest.mark.asyncio
async def test_handle_devflow_complex_requires_budget_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    created = []

    def capture_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        created.append(task)
        return task

    with (
        patch("gateway.devrun.find_sidecar_command", return_value=["node", "worker.mjs", "--worker", "review"]),
        patch("gateway.slash_commands.asyncio.create_task", side_effect=capture_task),
    ):
        event = _make_event(f"/devflow repo={tmp_path} task=complex project: implement multi-module development")
        result = await runner._handle_devflow_command(event)

    assert "✦ **DevFlow Command Center**" in result
    assert "智能体  " in result
    assert "M3 调用" in result
    assert "备用指令" in result
    assert len(created) == 0

    from gateway.devrun import list_jobs
    from tools import slash_confirm

    job = list_jobs(limit=1)[0]
    assert job.status == "awaiting_approval"
    slash_confirm.clear(build_session_key(event.source))


@pytest.mark.asyncio
async def test_handle_devflow_plain_language_uses_recent_repo_and_budget_gate(tmp_path, monkeypatch):
    from gateway.devrun import DevRunJob, list_jobs, save_job

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    previous = DevRunJob(
        job_id="dev_recent_repo",
        raw_task="previous",
        task="previous task",
        repo=str(tmp_path),
        risk="low",
        status="done",
    )
    save_job(previous)

    created = []

    def capture_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        created.append(task)
        return task

    with (
        patch("gateway.devrun.find_sidecar_command", return_value=["node", "worker.mjs", "--worker", "review"]),
        patch("gateway.slash_commands.asyncio.create_task", side_effect=capture_task),
    ):
        event = _make_event("/devflow 我想做一个登录功能")
        result = await runner._handle_devflow_command(event)

    assert "✦ **DevFlow Command Center**" in result
    assert f"`{tmp_path.resolve()}`" in result
    assert "我想做一个登录功能" in result
    assert "M3 调用" in result
    assert created == []

    job = list_jobs(limit=1)[0]
    assert job.status == "awaiting_approval"
    assert job.risk == "medium"
    assert job.repo == str(tmp_path.resolve())

    from tools import slash_confirm

    slash_confirm.clear(build_session_key(event.source))


@pytest.mark.asyncio
async def test_handle_devflow_plain_language_without_recent_repo_asks_for_project(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    event = _make_event("/devflow 我想做一个登录功能")
    result = await runner._handle_devflow_command(event)

    assert "我还不知道要在哪个项目里做" in result
    assert "/devflow repo=/Users/yuxiansheng/dev/你的项目 task=我想做一个登录功能" in result
    assert "/devflow 我想做一个登录功能" in result

    from gateway.devrun import list_jobs

    assert list_jobs(limit=1) == []


@pytest.mark.asyncio
async def test_devflow_budget_confirmation_starts_kanban_route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    created = []

    def capture_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        created.append(task)
        return task

    with (
        patch("gateway.devrun.find_sidecar_command", return_value=["node", "worker.mjs", "--worker", "review"]),
        patch("asyncio.create_task", side_effect=capture_task),
    ):
        event = _make_event(f"/devflow repo={tmp_path} task=complex project: implement multi-module development")
        result = await runner._handle_devflow_command(event)

        assert "M3 调用" in result
        assert created == []

        from tools import slash_confirm

        resolved = await slash_confirm.resolve(build_session_key(event.source), "1", "once")

    assert "✦ **DevFlow Command Center**" in resolved
    assert "状态  已启动" in resolved
    assert "M3 调用" in resolved
    assert len(created) == 1
    slash_confirm.clear(build_session_key(event.source))


@pytest.mark.asyncio
async def test_devflow_budget_cancel_renders_chinese_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    with patch("gateway.devrun.find_sidecar_command", return_value=["node", "worker.mjs", "--worker", "review"]):
        event = _make_event(f"/devflow repo={tmp_path} task=我想做一个登录功能")
        result = await runner._handle_devflow_command(event)

        assert "✦ **DevFlow Command Center**" in result

        from tools import slash_confirm

        resolved = await slash_confirm.resolve(build_session_key(event.source), "1", "cancel")

    assert "✦ **DevFlow Command Center**" in resolved
    assert "状态  已暂停" in resolved
    assert "预算  未消耗" in resolved
    assert "Status:" not in resolved
    assert "Risk:" not in resolved
    assert "Review summary:" not in resolved
    assert "Error:" not in resolved
    slash_confirm.clear(build_session_key(event.source))


@pytest.mark.asyncio
async def test_devflow_background_creates_kanban_triage_card(tmp_path, monkeypatch):
    from gateway.devrun import DevRunJob, save_job

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import kanban_db as kb

    kb.create_board("devflow-board")

    runner = _make_runner()
    runner._send_devrun_status_update = AsyncMock()

    job = DevRunJob(
        job_id="dev_flow_kanban",
        raw_task="复杂项目",
        task="复杂项目：实现多模块开发",
        repo=str(tmp_path),
        risk="medium",
        status="reviewing",
        board="devflow-board",
    )
    save_job(job)

    async def fake_executor(func, loaded_job):
        return "# DevReview Council Report\n- Worker status: all workers completed\n\nPASS"

    def fake_run_slash(text):
        import shlex

        tokens = shlex.split(text)
        assert "create" in text
        assert "DevFlow Budget Gate Evidence" in text
        assert "Estimated MiniMax-M3 calls:" in text
        assert "Current live verification scope:" in text
        assert "authoritative evidence for this run" in text
        assert "preflight questions to verify" in text
        assert "Historical closure or evidence files only count" in text
        assert tokens[:3] == ["--board", "devflow-board", "create"]
        assert "--triage" in text
        assert "--goal" in tokens
        assert tokens[tokens.index("--goal-max-turns") + 1] == "3"
        assert "dir:" in text
        return "Created t_abc123  (todo, assignee=-)"

    runner._run_in_executor_with_context = fake_executor

    class FakeConn:
        def close(self):
            pass

    connected_boards = []

    def fake_connect(*, board=None):
        connected_boards.append(board)
        return FakeConn()

    with (
        patch("gateway.devrun.find_sidecar_command", return_value=["node", "worker.mjs", "--worker", "review"]),
        patch("hermes_cli.kanban.run_slash", side_effect=fake_run_slash),
        patch("hermes_cli.kanban_db.connect", side_effect=fake_connect),
        patch("hermes_cli.kanban_db.add_notify_sub") as add_notify_sub,
    ):
        await runner._run_devflow_background_task(
            job_id="dev_flow_kanban",
            source=_make_event(repo=tmp_path).source,
            event_message_id="m1",
        )

    from gateway.devrun import load_job

    done = load_job("dev_flow_kanban")
    assert done.status == "done"
    assert done.review_verdict == "DEVFLOW"
    assert "DevFlow Budget Gate Evidence" in done.review_summary
    assert "Estimated agents:" in done.review_summary
    assert "Estimated MiniMax-M3 calls:" in done.review_summary
    assert "Created t_abc123" in done.test_results
    assert "Subscribed to Kanban updates for t_abc123." in done.test_results
    assert done.board == "devflow-board"
    assert connected_boards == ["devflow-board"]
    add_notify_sub.assert_called_once()
    assert add_notify_sub.call_args.kwargs["task_id"] == "t_abc123"
    assert add_notify_sub.call_args.kwargs["platform"] == "telegram"
    assert add_notify_sub.call_args.kwargs["chat_id"] == "67890"


@pytest.mark.asyncio
async def test_high_risk_devrun_waits_for_mobile_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    created = []

    def capture_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        created.append(task)
        return task

    with patch("gateway.slash_commands.asyncio.create_task", side_effect=capture_task):
        event = _make_event(f"/devrun repo={tmp_path} task=restart gateway")
        result = await runner._handle_devrun_command(event)

    from gateway.devrun import list_jobs
    from tools import slash_confirm

    job = list_jobs(limit=1)[0]
    assert "High-risk DevRun needs mobile approval" in result
    assert job.status == "awaiting_approval"
    assert job.risk == "high"
    assert created == []

    slash_confirm.clear(build_session_key(event.source))


@pytest.mark.asyncio
async def test_devstatus_and_devcancel(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_DEVRUN_ALLOWED_ROOTS", str(tmp_path))
    runner = _make_runner()

    with patch("gateway.slash_commands.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
        started = await runner._handle_devrun_command(_make_event(repo=tmp_path))

    job_id = next(line.split(":", 1)[1].strip() for line in started.splitlines() if line.startswith("DevRun job started:"))

    status = await runner._handle_devstatus_command(_make_event(f"/devstatus {job_id}"))
    assert job_id in status
    assert "Status:" in status

    cancelled = await runner._handle_devcancel_command(_make_event(f"/devcancel {job_id}"))
    assert "DevRun cancel requested." in cancelled
    assert "Status: blocked" in cancelled


@pytest.mark.asyncio
async def test_review_pass_does_not_execute_cancelled_job(tmp_path, monkeypatch):
    from gateway.devrun import DevRunJob, save_job

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    runner = _make_runner()
    runner._send_devrun_status_update = AsyncMock()

    job = DevRunJob(
        job_id="dev_cancel_race",
        raw_task="refactor api",
        task="refactor api",
        repo=str(tmp_path),
        risk="medium",
        status="reviewing",
    )
    save_job(job)

    async def fake_executor(func, loaded_job):
        loaded_job.cancelled = True
        loaded_job.status = "blocked"
        save_job(loaded_job)
        return "PASS", "PASS"

    runner._run_in_executor_with_context = fake_executor
    runner._run_devrun_background_task = MagicMock()

    await runner._run_devrun_review_then_execute(
        job_id="dev_cancel_race",
        source=_make_event(repo=tmp_path).source,
        event_message_id="m1",
    )

    runner._run_devrun_background_task.assert_not_called()


@pytest.mark.asyncio
async def test_devrun_bypasses_active_session_guard():
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.TELEGRAM)
    adapter._busy_text_mode = ""
    adapter.sent_responses = []

    async def handler(event):
        return f"handled:{event.get_command()}"

    async def send_retry(chat_id, content, **kwargs):
        adapter.sent_responses.append(content)

    adapter._message_handler = handler
    adapter._send_with_retry = send_retry

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()

    event = MessageEvent(
        text="/devrun repo=/tmp task=fix typo",
        message_type=MessageType.TEXT,
        source=source,
    )
    await adapter.handle_message(event)

    assert session_key not in adapter._pending_messages
    assert any("handled:devrun" in response for response in adapter.sent_responses)


@pytest.mark.asyncio
async def test_devreview_bypasses_active_session_guard():
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.TELEGRAM)
    adapter._busy_text_mode = ""
    adapter.sent_responses = []

    async def handler(event):
        return f"handled:{event.get_command()}"

    async def send_retry(chat_id, content, **kwargs):
        adapter.sent_responses.append(content)

    adapter._message_handler = handler
    adapter._send_with_retry = send_retry

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()

    event = MessageEvent(
        text="/devreview repo=/tmp task=review architecture",
        message_type=MessageType.TEXT,
        source=source,
    )
    await adapter.handle_message(event)

    assert session_key not in adapter._pending_messages
    assert any("handled:devreview" in response for response in adapter.sent_responses)
