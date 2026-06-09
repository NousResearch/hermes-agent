import asyncio

from pathlib import Path

import pytest

import gateway.managed_dev_workflow as mdw
from gateway.managed_dev_workflow import (
    DEV_CHANNEL_ID,
    DEV_CHANNEL_MENTION,
    AmbiguousBindingError,
    WfDevCommand,
    WfDevError,
    WorkflowBinding,
    WorkflowBindingStore,
    build_execution_prompt,
    build_planning_request,
    classify_turn,
    collect_plan,
    dispatch_pending_notifications,
    format_list_embed,
    format_notification_message,
    format_status_text,
    handle_wf_dev_command,
    is_managed_dev_workflow_enabled,
    is_wf_dev_event,
    normalize_auto_skills,
    parse_wf_dev_command,
    parse_wf_dev_event,
    write_task_files,
)


def _payload(**overrides):
    base = {
        "task_id": "task-1",
        "event_id": 7,
        "event_type": "COMPLETED_DETECTED",
        "status": "COMPLETED",
        "title": "웹 계산기",
        "blocker_id": None,
        "summary": "기능 구현 완료, 테스트 통과",
        "raw_excerpt": "All 12 tests passed",
        "requires_user_response": False,
        "discord_channel_id": DEV_CHANNEL_ID,
        "log_file": "/tmp/task-1.log",
        "created_at": "2026-06-08T10:00:00",
        "recommended_user_message": "완료 메시지 템플릿으로 요약을 보내.",
    }
    base.update(overrides)
    return base


def test_normalize_auto_skills_and_enablement():
    assert normalize_auto_skills(None) == []
    assert normalize_auto_skills("managed-dev-workflow") == ["managed-dev-workflow"]
    assert normalize_auto_skills(["a", "managed-dev-workflow"]) == ["a", "managed-dev-workflow"]
    assert is_managed_dev_workflow_enabled(["a", "managed-dev-workflow"])
    assert not is_managed_dev_workflow_enabled(["a", "b"])


def test_classify_turn_routes_expected_actions():
    assert classify_turn("승인", "WAITING_APPROVAL") == "approve_start"
    assert classify_turn("수정: 타입스크립트로", "WAITING_APPROVAL") == "revise_plan"
    assert classify_turn("아무거나 답", "WAITING_INPUT") == "reply"
    assert classify_turn("상태", "RUNNING") == "status"
    assert classify_turn("중단", "RUNNING") == "stop"
    assert classify_turn("새 계산기 만들자", None) == "plan"


def test_write_task_files_creates_plan_and_prompt(tmp_path: Path):
    files = write_task_files(
        tmp_path,
        "task-123",
        "# plan\n- do thing",
        user_message="타입스크립트로 진행",
    )
    plan_path = Path(files["plan_file"])
    prompt_path = Path(files["prompt_file"])

    assert plan_path.exists()
    assert prompt_path.exists()
    assert plan_path.read_text(encoding="utf-8") == "# plan\n- do thing\n"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "Task ID: task-123" in prompt_text
    assert "Approved plan:" in prompt_text
    assert "타입스크립트로 진행" in prompt_text


def test_build_execution_prompt_contains_guardrails():
    prompt = build_execution_prompt("task-1", "- implement feature", "vite + typescript")
    assert "Implement the approved plan below exactly" in prompt
    assert "If you need a choice" in prompt
    assert "vite + typescript" in prompt


def test_build_planning_request_includes_revision_context():
    prompt = build_planning_request("수정: 타입스크립트로 개발", "## 개발 계획\n기존 초안")
    assert "사용자 요청:" in prompt
    assert "수정: 타입스크립트로 개발" in prompt
    assert "기존 계획 초안:" in prompt
    assert "최신 수정 요청을 반영" in prompt


def test_collect_plan_reads_plan_file_path_from_stdout(tmp_path: Path, monkeypatch):
    plan_path = tmp_path / "sample-plan.md"
    plan_path.write_text(
        "## 개발 계획\n\n### 1. 현재 이해한 내용\n- a\n\n### 2. 코드 구조 확인 결과\n- b\n\n### 3. 구현 계획\n1. c\n2. d\n3. e\n\n### 4. 변경 예상 범위\n- f\n\n### 5. 위험 요소\n- g\n\n### 6. 확인 필요한 사항\n- h\n",
        encoding="utf-8",
    )

    class DummyProc:
        returncode = 0
        stdout = str(plan_path)
        stderr = ""

    monkeypatch.setattr("gateway.managed_dev_workflow.subprocess.run", lambda *a, **k: DummyProc())
    result = collect_plan(tmp_path, "task-1", "웹 계산기 계획만")
    assert result["plan_text"].startswith("## 개발 계획")
    assert result["request_file"].endswith("task-1-plan-request.md")


def test_format_notification_prepends_mention_for_dev_channel():
    msg = format_notification_message(_payload())
    # First line must begin with the required dev-channel mention.
    assert msg.splitlines()[0].startswith(DEV_CHANNEL_MENTION)
    # Structured, human-facing content — not a raw JSON dump.
    assert "{" not in msg and "event_id" not in msg
    assert "웹 계산기" in msg
    assert "기능 구현 완료" in msg


def test_format_notification_no_mention_for_other_channel():
    msg = format_notification_message(_payload(discord_channel_id="999"))
    assert DEV_CHANNEL_MENTION not in msg


def test_format_notification_distinct_templates_per_event_type():
    completed = format_notification_message(_payload(event_type="COMPLETED_DETECTED", status="COMPLETED"))
    failed = format_notification_message(_payload(event_type="FAILED_DETECTED", status="FAILED"))
    blocked = format_notification_message(_payload(event_type="BLOCKED_DETECTED", status="BLOCKED"))
    waiting = format_notification_message(
        _payload(event_type="WAITING_INPUT_DETECTED", status="WAITING_INPUT", requires_user_response=True)
    )
    # Each event type renders a recognizably different headline.
    headlines = {completed.splitlines()[0], failed.splitlines()[0], blocked.splitlines()[0], waiting.splitlines()[0]}
    assert len(headlines) == 4
    # Waiting/blocked prompt the user to respond.
    assert "답장" in waiting or "답장" in blocked


def test_dispatch_acks_after_successful_send():
    sent = []
    acked = []

    async def fake_send(channel_id, text):
        sent.append((channel_id, text))
        return True

    def fake_pending(repo_dir, task_id):
        return [_payload(task_id=task_id, event_id=7)]

    def fake_ack(repo_dir, task_id, event_id):
        acked.append((task_id, event_id))
        return {"acked": True}

    summary = asyncio.run(
        dispatch_pending_notifications(
            "/repo",
            fake_send,
            task_ids=["task-1"],
            _list_pending=fake_pending,
            _ack=fake_ack,
        )
    )
    assert len(sent) == 1
    assert acked == [("task-1", 7)]
    assert summary["sent"] == 1
    assert summary["acked"] == 1


def test_dispatch_does_not_ack_when_send_fails():
    acked = []

    async def failing_send(channel_id, text):
        return False

    def fake_pending(repo_dir, task_id):
        return [_payload(task_id=task_id, event_id=9)]

    def fake_ack(repo_dir, task_id, event_id):
        acked.append((task_id, event_id))
        return {"acked": True}

    summary = asyncio.run(
        dispatch_pending_notifications(
            "/repo",
            failing_send,
            task_ids=["task-1"],
            _list_pending=fake_pending,
            _ack=fake_ack,
        )
    )
    # Send failed → notification stays pending, no ack.
    assert acked == []
    assert summary["failed"] == 1
    assert summary["acked"] == 0


def test_dispatch_does_not_ack_when_send_raises():
    acked = []

    async def raising_send(channel_id, text):
        raise RuntimeError("discord down")

    def fake_pending(repo_dir, task_id):
        return [_payload(task_id=task_id, event_id=11)]

    def fake_ack(repo_dir, task_id, event_id):
        acked.append((task_id, event_id))
        return {"acked": True}

    summary = asyncio.run(
        dispatch_pending_notifications(
            "/repo",
            raising_send,
            task_ids=["task-1"],
            _list_pending=fake_pending,
            _ack=fake_ack,
        )
    )
    assert acked == []
    assert summary["failed"] == 1


def test_dispatch_isolates_per_task_failures():
    sent = []

    async def fake_send(channel_id, text):
        sent.append(channel_id)
        return True

    def fake_pending(repo_dir, task_id):
        if task_id == "bad":
            raise RuntimeError("supervisor exploded")
        return [_payload(task_id=task_id, event_id=1)]

    def fake_ack(repo_dir, task_id, event_id):
        return {"acked": True}

    summary = asyncio.run(
        dispatch_pending_notifications(
            "/repo",
            fake_send,
            task_ids=["bad", "good"],
            _list_pending=fake_pending,
            _ack=fake_ack,
        )
    )
    # One task blew up but the other still delivered.
    assert summary["sent"] == 1
    assert len(sent) == 1


def test_dispatch_skips_notifications_without_channel():
    async def fake_send(channel_id, text):
        raise AssertionError("should not send without a channel")

    def fake_pending(repo_dir, task_id):
        return [_payload(task_id=task_id, event_id=3, discord_channel_id=None)]

    summary = asyncio.run(
        dispatch_pending_notifications(
            "/repo",
            fake_send,
            task_ids=["task-1"],
            _list_pending=fake_pending,
            _ack=lambda *a: {"acked": True},
        )
    )
    assert summary["skipped"] == 1
    assert summary["sent"] == 0


# ===========================================================================
# Deterministic /wf-dev control plane
# ===========================================================================


def _store(tmp_path) -> WorkflowBindingStore:
    return WorkflowBindingStore(tmp_path / "bindings.db")


def _seed(store, task_id="dev-1", ref="DEV-1", *, thread_id="th-1",
          channel_id="ch-1", status="WAITING_APPROVAL", is_active=True,
          title="Router", plan_version=1):
    store.save(WorkflowBinding(
        task_id=task_id, ref=ref, thread_id=thread_id, channel_id=channel_id,
        status=status, is_active=is_active, title=title, plan_version=plan_version,
    ))
    return store.get(task_id)


# ---- parsing / normalization ----------------------------------------------


def test_parse_wf_dev_command_defaults_and_coercion():
    cmd = parse_wf_dev_command({"subcommand": "list", "page": "3", "auto_start": "no"})
    assert cmd.subcommand == "list"
    assert cmd.page == 3
    assert cmd.auto_start is False
    # Omitted optionals default deterministically.
    assert cmd.scope == "thread"
    assert cmd.status_filter == "active"
    assert cmd.task_id is None


def test_parse_wf_dev_command_strips_and_nulls_empty():
    cmd = parse_wf_dev_command({
        "subcommand": "PLAN", "request": "  build it  ", "task_id": "   ",
        "channel_id": "100", "thread_id": "200",
    })
    assert cmd.subcommand == "plan"  # lowercased
    assert cmd.request == "build it"
    assert cmd.task_id is None  # whitespace-only → None
    assert cmd.channel_id == "100"
    assert cmd.thread_id == "200"


def test_parse_wf_dev_command_rejects_unknown_subcommand():
    with pytest.raises(WfDevError):
        parse_wf_dev_command({"subcommand": "frobnicate"})


def test_parse_wf_dev_command_rejects_missing_subcommand():
    with pytest.raises(WfDevError):
        parse_wf_dev_command({"request": "no sub"})


def test_is_wf_dev_event_and_parse_event():
    class _Evt:
        metadata = {"wf_dev": {"subcommand": "status", "task_id": "dev-9"}}

    assert is_wf_dev_event(_Evt())
    cmd = parse_wf_dev_event(_Evt())
    assert cmd.subcommand == "status"
    assert cmd.task_id == "dev-9"


def test_is_wf_dev_event_false_for_plain_event():
    class _Plain:
        metadata = None

    class _NoAttr:
        pass

    assert not is_wf_dev_event(_Plain())
    assert not is_wf_dev_event(_NoAttr())


# ---- binding persistence ---------------------------------------------------


def test_binding_store_save_get_roundtrip(tmp_path):
    store = _store(tmp_path)
    b = _seed(store)
    got = store.get("dev-1")
    assert got.ref == "DEV-1"
    assert got.thread_id == "th-1"
    assert got.is_active is True
    assert got.created_at and got.updated_at


def test_binding_store_next_ref_monotonic(tmp_path):
    store = _store(tmp_path)
    assert store.next_ref() == "DEV-1"
    _seed(store, task_id="a", ref="DEV-1")
    _seed(store, task_id="b", ref="DEV-2")
    assert store.next_ref() == "DEV-3"


def test_binding_resolve_active_prefers_thread(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="thread-task", ref="DEV-1", thread_id="th-9", channel_id="ch-9")
    _seed(store, task_id="chan-task", ref="DEV-2", thread_id="th-other", channel_id="ch-9")
    # Thread match wins even though both share channel ch-9.
    got = store.resolve_active(thread_id="th-9", channel_id="ch-9")
    assert got.task_id == "thread-task"


def test_binding_resolve_active_none_when_unbound(tmp_path):
    store = _store(tmp_path)
    assert store.resolve_active(thread_id="nope", channel_id="nope") is None


def test_binding_resolve_active_ambiguous_raises(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="a", ref="DEV-1", thread_id=None, channel_id="ch-x")
    _seed(store, task_id="b", ref="DEV-2", thread_id=None, channel_id="ch-x")
    with pytest.raises(AmbiguousBindingError):
        store.resolve_active(channel_id="ch-x")


def test_binding_deactivate_and_status_update(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    store.update_status("dev-1", "RUNNING")
    assert store.get("dev-1").status == "RUNNING"
    store.deactivate("dev-1")
    got = store.get("dev-1")
    assert got.is_active is False
    # Deactivated rows drop out of active resolution.
    assert store.resolve_active(thread_id="th-1") is None


def test_binding_list_recent_filters(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="a", ref="DEV-1", status="RUNNING", is_active=True)
    _seed(store, task_id="b", ref="DEV-2", status="COMPLETED", is_active=False)
    active = store.list_recent(thread_id="th-1", scope="thread", status_filter="active")
    assert [b.task_id for b in active] == ["a"]
    allrows = store.list_recent(thread_id="th-1", scope="thread", status_filter="all")
    assert {b.task_id for b in allrows} == {"a", "b"}
    completed = store.list_recent(thread_id="th-1", scope="thread", status_filter="completed")
    assert [b.task_id for b in completed] == ["b"]


# ---- router: help / list ---------------------------------------------------


def _handle(cmd, tmp_path, store=None, **kw):
    store = store or _store(tmp_path)
    return handle_wf_dev_command(cmd, repo_dir=tmp_path, store=store, **kw), store


def test_handle_help(tmp_path):
    resp, _ = _handle(WfDevCommand(subcommand="help"), tmp_path)
    assert resp.ok
    assert "/wf-dev commands" in resp.text


def test_handle_list_embed_shows_ref_and_task_id(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-20260609-001", ref="DEV-142", status="RUNNING", title="Discord router")
    resp, _ = _handle(WfDevCommand(subcommand="list", thread_id="th-1"), tmp_path, store=store)
    assert resp.ok
    assert resp.embed is not None
    desc = resp.embed["description"]
    assert "DEV-142" in desc
    assert "dev-20260609-001" in desc
    assert "Ref" in desc and "Task ID" in desc
    # Code-block pseudo-table.
    assert desc.startswith("```") and desc.rstrip().endswith("```")


def test_format_list_embed_empty_is_safe():
    embed = format_list_embed([], scope="thread", page=1)
    assert "none" in embed["description"]
    assert embed["title"]


# ---- router: status --------------------------------------------------------


def test_handle_status_resolves_active_binding_without_task_id(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="RUNNING")

    def fake_status(repo, tid):
        assert tid == "dev-1"
        return {"status": "RUNNING", "latest_events": [{"event_type": "progress"}]}

    resp, _ = _handle(WfDevCommand(subcommand="status", thread_id="th-1"), tmp_path,
                      store=store, status_fn=fake_status)
    assert resp.ok
    assert "DEV-1" in resp.text and "dev-1" in resp.text
    assert "RUNNING" in resp.text


def test_handle_status_no_binding_fails_closed(tmp_path):
    resp, _ = _handle(WfDevCommand(subcommand="status", thread_id="empty"), tmp_path)
    assert resp.ok is False
    assert resp.ephemeral is True
    assert "No active workflow" in resp.text


def test_handle_status_terminal_deactivates_binding(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="RUNNING")

    def fake_status(repo, tid):
        return {"status": "COMPLETED", "latest_events": []}

    resp, store = _handle(WfDevCommand(subcommand="status", thread_id="th-1"), tmp_path,
                          store=store, status_fn=fake_status)
    assert resp.ok
    assert store.get("dev-1").is_active is False


# ---- router: approve / start (state gating) --------------------------------


def test_handle_approve_auto_start_happy_path(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="WAITING_APPROVAL")
    calls = {}

    def fake_status(repo, tid):
        return {"status": "WAITING_APPROVAL", "latest_events": []}

    def fake_approve_start(repo, tid, *, discord_channel_id=None):
        calls["approve_start"] = tid
        return {"approve": {"status": "APPROVED"}, "start": {"status": "RUNNING"}}

    resp, store = _handle(WfDevCommand(subcommand="approve", thread_id="th-1"), tmp_path,
                          store=store, status_fn=fake_status,
                          approve_and_start_fn=fake_approve_start)
    assert resp.ok
    assert calls["approve_start"] == "dev-1"
    assert "RUNNING" in resp.text
    assert store.get("dev-1").status == "RUNNING"


def test_handle_approve_rejected_when_not_waiting_approval(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="RUNNING")

    def fake_status(repo, tid):
        return {"status": "RUNNING", "latest_events": []}

    resp, _ = _handle(WfDevCommand(subcommand="approve", thread_id="th-1"), tmp_path,
                      store=store, status_fn=fake_status)
    assert resp.ok is False
    assert "approve rejected" in resp.text


def test_handle_start_requires_approved(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="WAITING_APPROVAL")

    def fake_status(repo, tid):
        return {"status": "WAITING_APPROVAL", "latest_events": []}

    resp, _ = _handle(WfDevCommand(subcommand="start", thread_id="th-1"), tmp_path,
                      store=store, status_fn=fake_status)
    assert resp.ok is False
    assert "start rejected" in resp.text


def test_handle_start_happy_path(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="APPROVED")

    def fake_status(repo, tid):
        return {"status": "APPROVED", "latest_events": []}

    def fake_start(repo, tid, *, discord_channel_id=None):
        return {"status": "RUNNING"}

    resp, store = _handle(WfDevCommand(subcommand="start", thread_id="th-1"), tmp_path,
                          store=store, status_fn=fake_status, start_fn=fake_start)
    assert resp.ok
    assert store.get("dev-1").status == "RUNNING"


# ---- router: reply (state gating) ------------------------------------------


def test_handle_reply_only_in_waiting_or_blocked(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="RUNNING")

    def fake_status(repo, tid):
        return {"status": "RUNNING", "latest_events": []}

    resp, _ = _handle(WfDevCommand(subcommand="reply", message="go", thread_id="th-1"),
                      tmp_path, store=store, status_fn=fake_status)
    assert resp.ok is False
    assert "reply rejected" in resp.text


def test_handle_reply_forwards_when_waiting_input(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="WAITING_INPUT")
    forwarded = {}

    def fake_status(repo, tid):
        return {"status": "WAITING_INPUT", "latest_events": []}

    def fake_reply(repo, tid, message):
        forwarded["msg"] = message
        return {"status": "RUNNING"}

    resp, _ = _handle(WfDevCommand(subcommand="reply", message="keep filename", thread_id="th-1"),
                      tmp_path, store=store, status_fn=fake_status, reply_fn=fake_reply)
    assert resp.ok
    assert forwarded["msg"] == "keep filename"
    assert "current_status: RUNNING" in resp.text


def test_handle_reply_requires_message(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="WAITING_INPUT")
    resp, _ = _handle(WfDevCommand(subcommand="reply", thread_id="th-1"), tmp_path, store=store)
    assert resp.ok is False
    assert "message" in resp.text


# ---- router: explicit task_id wins over thread binding ---------------------


def test_explicit_task_id_overrides_thread_binding(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="thread-active", ref="DEV-1", status="RUNNING", thread_id="th-1")
    _seed(store, task_id="other", ref="DEV-2", status="STOPPED", thread_id="th-other")

    seen = {}

    def fake_status(repo, tid):
        seen["tid"] = tid
        return {"status": "RUNNING", "latest_events": []}

    resp, _ = _handle(WfDevCommand(subcommand="status", task_id="other", thread_id="th-1"),
                      tmp_path, store=store, status_fn=fake_status)
    assert seen["tid"] == "other"
    assert "DEV-2" in resp.text


# ---- router: stop / summary ------------------------------------------------


def test_handle_stop_deactivates(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="RUNNING")

    def fake_stop(repo, tid):
        return {"status": "STOPPED"}

    resp, store = _handle(WfDevCommand(subcommand="stop", thread_id="th-1"), tmp_path,
                          store=store, stop_fn=fake_stop)
    assert resp.ok
    assert "STOPPED" in resp.text
    assert store.get("dev-1").is_active is False


def test_handle_summary(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="COMPLETED")

    def fake_summary(repo, tid):
        return {"summary": {"status": "COMPLETED", "final_summary": "all done",
                            "verification_summary": "12 passed"}}

    resp, store = _handle(WfDevCommand(subcommand="summary", thread_id="th-1"), tmp_path,
                          store=store, summary_fn=fake_summary)
    assert resp.ok
    assert "DEV-1" in resp.text and "dev-1" in resp.text
    assert "all done" in resp.text
    assert store.get("dev-1").is_active is False


# ---- router: plan / revise -------------------------------------------------


def test_handle_plan_creates_binding_with_ref_and_task_id(tmp_path):
    store = _store(tmp_path)

    def fake_collect(repo, tid, request):
        return {"plan_text": "## plan", "request_file": "req.md"}

    def fake_save(repo, tid, plan_text, **kw):
        return {"status": "WAITING_APPROVAL", "plan_version": 1}

    resp, store = _handle(
        WfDevCommand(subcommand="plan", request="build router",
                     thread_id="th-1", channel_id="ch-1", session_id="dev-sess"),
        tmp_path, store=store, collect_plan_fn=fake_collect, save_plan_fn=fake_save,
    )
    assert resp.ok
    assert resp.ref == "DEV-1"
    assert resp.task_id == "dev-sess"
    assert "WAITING_APPROVAL" in resp.text
    binding = store.get("dev-sess")
    assert binding.is_active and binding.ref == "DEV-1"


def test_handle_plan_blocks_second_active_in_thread(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="existing", ref="DEV-1", status="RUNNING", thread_id="th-1")
    resp, _ = _handle(
        WfDevCommand(subcommand="plan", request="another", thread_id="th-1", channel_id="ch-1"),
        tmp_path, store=store,
    )
    assert resp.ok is False
    assert "already has an active workflow" in resp.text


def test_handle_revise_rejected_while_running(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="RUNNING")

    def fake_status(repo, tid):
        return {"status": "RUNNING", "latest_events": []}

    resp, _ = _handle(WfDevCommand(subcommand="revise", request="change it", thread_id="th-1"),
                      tmp_path, store=store, status_fn=fake_status)
    assert resp.ok is False
    assert "revise rejected" in resp.text


def test_handle_revise_bumps_plan_version(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="WAITING_APPROVAL", plan_version=1)

    def fake_status(repo, tid):
        return {"status": "WAITING_APPROVAL", "latest_events": []}

    def fake_collect(repo, tid, request):
        return {"plan_text": "## plan v2", "request_file": "req.md"}

    def fake_save(repo, tid, plan_text, **kw):
        return {"status": "WAITING_APPROVAL", "plan_version": 2}

    resp, store = _handle(
        WfDevCommand(subcommand="revise", request="use /wf-dev", thread_id="th-1"),
        tmp_path, store=store, status_fn=fake_status,
        collect_plan_fn=fake_collect, save_plan_fn=fake_save,
    )
    assert resp.ok
    assert "plan_version: 2" in resp.text
    assert store.get("dev-1").plan_version == 2


# ---- fail-closed guarantee -------------------------------------------------


def test_handle_never_raises_on_supervisor_error(tmp_path):
    store = _store(tmp_path)
    _seed(store, task_id="dev-1", ref="DEV-1", status="WAITING_INPUT")

    def boom_status(repo, tid):
        return {"status": "WAITING_INPUT", "latest_events": []}

    def boom_reply(repo, tid, message):
        raise RuntimeError("supervisor exploded")

    resp, _ = _handle(WfDevCommand(subcommand="reply", message="x", thread_id="th-1"),
                      tmp_path, store=store, status_fn=boom_status, reply_fn=boom_reply)
    # Fail closed: explicit error response, never a raised exception.
    assert resp.ok is False
    assert resp.ephemeral is True
    assert "실패" in resp.text


def test_format_status_text_always_includes_ref_and_task_id():
    text = format_status_text(ref="DEV-7", task_id="dev-7", status="RUNNING")
    assert "ref: DEV-7" in text
    assert "task_id: dev-7" in text
