"""Tests for the durable gateway background job store."""

from __future__ import annotations

import time
from pathlib import Path

from gateway.config import Platform
from gateway.session import SessionSource


def _make_source(
    *,
    platform: Platform = Platform.QQ_NAPCAT,
    chat_id: str = "726109087",
    chat_type: str = "group",
    user_id: str = "179033731",
    thread_id: str | None = None,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name="發發發",
        thread_id=thread_id,
    )


def _make_store(tmp_path: Path):
    from gateway.background_jobs import BackgroundJobStore

    return BackgroundJobStore(db_path=tmp_path / "background_jobs.db")


def test_create_and_list_jobs_by_scope(tmp_path):
    from gateway.background_jobs import background_job_chat_key, background_job_scope_key

    store = _make_store(tmp_path)
    source = _make_source()
    session_key = "qq_napcat:group:726109087:179033731"
    task_id = "bg_100001_abcd12"

    store.create_job(
        task_id=task_id,
        prompt="继续把线上问题查清楚",
        source=source,
        session_key=session_key,
        job_kind="auto",
        worker_name="马嘎",
        preloaded_skills=["qq-intel"],
        conversation_history=[{"role": "user", "content": "继续"}],
        context_prompt="context",
        admin_user_ids=["179033731"],
        is_admin_user=True,
    )

    job = store.get_job(task_id)
    assert job is not None
    assert job["task_id"] == task_id
    assert job["status"] == "queued"
    assert job["chat_key"] == background_job_chat_key(source)
    assert job["scope_key"] == background_job_scope_key(source, session_key=session_key)
    assert job["source"]["chat_id"] == "726109087"
    assert job["preloaded_skills"] == ["qq-intel"]

    active = store.list_jobs(
        chat_key=background_job_chat_key(source),
        scope_key=background_job_scope_key(source, session_key=session_key),
        active_only=True,
    )
    assert [item["task_id"] for item in active] == [task_id]

    store.mark_job_completed(task_id, raw_response="done")
    assert store.list_jobs(
        chat_key=background_job_chat_key(source),
        scope_key=background_job_scope_key(source, session_key=session_key),
        active_only=True,
    ) == []


def test_list_jobs_without_scope_returns_all_jobs(tmp_path):
    store = _make_store(tmp_path)
    source_a = _make_source(chat_id="726109087", user_id="179033731")
    source_b = _make_source(chat_id="726109087", user_id="888888")

    store.create_job(
        task_id="bg_scope_a",
        prompt="任务A",
        source=source_a,
        session_key="qq_napcat:group:726109087:179033731",
    )
    store.create_job(
        task_id="bg_scope_b",
        prompt="任务B",
        source=source_b,
        session_key="qq_napcat:group:726109087:888888",
    )

    jobs = store.list_jobs()

    assert [job["task_id"] for job in jobs] == ["bg_scope_a", "bg_scope_b"]


def test_delivery_claim_release_and_complete(tmp_path):
    store = _make_store(tmp_path)
    source = _make_source(chat_type="dm", chat_id="179033731")
    task_id = "bg_100002_abcd12"

    store.create_job(
        task_id=task_id,
        prompt="汇报",
        source=source,
        session_key="qq_napcat:dm:179033731",
    )
    store.mark_job_completed(task_id, raw_response="汇报完成")

    claimed = store.claim_delivery_jobs(claimer="poller-a", limit=10, lease_seconds=60)
    assert [item["task_id"] for item in claimed] == [task_id]
    assert store.claim_delivery_jobs(claimer="poller-b", limit=10, lease_seconds=60) == []

    store.release_delivery_claim(task_id, error="adapter unavailable")
    claimed_again = store.claim_delivery_jobs(claimer="poller-b", limit=10, lease_seconds=60)
    assert [item["task_id"] for item in claimed_again] == [task_id]

    store.mark_job_delivered(task_id)
    assert store.claim_delivery_jobs(claimer="poller-c", limit=10, lease_seconds=60) == []


def test_explicit_cancel_wins_over_late_completion(tmp_path):
    store = _make_store(tmp_path)
    source = _make_source(chat_type="dm", chat_id="179033731")
    task_id = "bg_100003_abcd12"

    store.create_job(
        task_id=task_id,
        prompt="执行危险操作",
        source=source,
        session_key="qq_napcat:dm:179033731",
    )
    store.mark_job_running(task_id)
    store.mark_job_cancelled(task_id, reason="stop requested")
    store.mark_job_completed(task_id, raw_response="should be ignored")

    job = store.get_job(task_id)
    assert job is not None
    assert job["status"] == "cancelled"
    assert not job["raw_response"]
    assert job["error"] == "stop requested"


def test_external_approval_request_round_trip(tmp_path):
    store = _make_store(tmp_path)
    source = _make_source(chat_type="dm", chat_id="179033731")
    task_id = "bg_100004_abcd12"

    store.create_job(
        task_id=task_id,
        prompt="改配置",
        source=source,
        session_key="qq_napcat:dm:179033731",
    )

    request_id = store.create_approval_request(
        task_id=task_id,
        session_key="qq_napcat:dm:179033731",
        source=source,
        approval_data={
            "command": "systemctl restart hermes-gateway.service",
            "description": "stop/disable system service",
            "prompt_title": "Dangerous command requires approval",
            "approver_name": "董事长",
            "allow_persistence": False,
            "pattern_key": "stop/disable system service",
            "pattern_keys": ["stop/disable system service"],
        },
    )

    claimed = store.claim_approval_notifications(claimer="poller-a", limit=10, lease_seconds=60)
    assert len(claimed) == 1
    assert claimed[0]["request_id"] == request_id
    assert claimed[0]["task_id"] == task_id

    assert store.resolve_approval_requests(
        session_key="qq_napcat:dm:179033731",
        choice="once",
        resolve_all=False,
    ) == 1
    assert store.wait_for_approval_resolution(
        request_id,
        timeout_seconds=0.2,
        poll_interval_seconds=0.01,
    ) == "once"


def test_touch_heartbeat_updates_job_liveness(tmp_path):
    store = _make_store(tmp_path)
    source = _make_source(chat_type="dm", chat_id="179033731")
    task_id = "bg_100005_abcd12"

    store.create_job(
        task_id=task_id,
        prompt="继续执行",
        source=source,
        session_key="qq_napcat:dm:179033731",
    )
    store.mark_job_running(task_id)

    before = store.get_job(task_id)
    store.touch_job_heartbeat(task_id)
    after = store.get_job(task_id)

    assert before is not None and after is not None
    assert after["last_heartbeat_at"] is not None
    assert after["last_heartbeat_at"] >= before["updated_at"]
    assert after["heartbeat_count"] == 1


def test_recover_stale_running_subprocess_job_when_pid_is_gone(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    source = _make_source(chat_type="dm", chat_id="179033731")
    task_id = "bg_100006_abcd12"

    store.create_job(
        task_id=task_id,
        prompt="跑很久的任务",
        source=source,
        session_key="qq_napcat:dm:179033731",
    )
    store.mark_job_running(task_id)
    store.update_job_launcher(
        task_id,
        {
            "launcher_type": "subprocess",
            "launcher_pid": 999999,
        },
    )

    monkeypatch.setattr("gateway.background_jobs.os.kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    recovered = store.recover_stale_jobs(
        now_ts=time.time() + 600,
        queued_grace_seconds=30,
        heartbeat_stale_seconds=30,
    )

    assert [item["task_id"] for item in recovered] == [task_id]
    job = store.get_job(task_id)
    assert job is not None
    assert job["status"] == "failed"
    assert "heartbeat" in str(job["error"] or "") or "worker" in str(job["error"] or "")
    assert job["recovered_at"] is not None


def test_recover_stale_queued_systemd_job_surfaces_unit_failure_summary(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    source = _make_source(chat_type="dm", chat_id="179033731")
    task_id = "bg_100006_systemd"

    store.create_job(
        task_id=task_id,
        prompt="后台继续处理",
        source=source,
        session_key="qq_napcat:dm:179033731",
    )
    store.update_job_launcher(
        task_id,
        {
            "launcher_type": "systemd-run",
            "launcher_unit": "hermes-bg-test",
            "launcher_scope": "system",
        },
    )

    monkeypatch.setattr(
        "gateway.background_jobs._systemd_unit_failure_summary",
        lambda unit, scope="": "ImportError: cannot import name 'reset_external_approval_backend'",
    )

    recovered = store.recover_stale_jobs(
        now_ts=time.time() + 600,
        queued_grace_seconds=30,
        heartbeat_stale_seconds=30,
    )

    assert [item["task_id"] for item in recovered] == [task_id]
    job = store.get_job(task_id)
    assert job is not None
    assert job["status"] == "failed"
    assert "failed before heartbeat" in str(job["error"] or "")
    assert "ImportError" in str(job["error"] or "")
    assert job["recovered_at"] is not None


def test_count_pending_approval_requests_for_session(tmp_path):
    store = _make_store(tmp_path)
    source = _make_source(chat_type="dm", chat_id="179033731")
    session_key = "qq_napcat:dm:179033731"

    store.create_job(
        task_id="bg_100007_abcd12",
        prompt="审批1",
        source=source,
        session_key=session_key,
    )
    store.create_job(
        task_id="bg_100008_abcd12",
        prompt="审批2",
        source=source,
        session_key=session_key,
    )
    for task_id in ("bg_100007_abcd12", "bg_100008_abcd12"):
        store.create_approval_request(
            task_id=task_id,
            session_key=session_key,
            source=source,
            approval_data={
                "command": "systemctl restart hermes-gateway.service",
                "description": "stop/disable system service",
                "prompt_title": "Dangerous command requires approval",
                "approver_name": "董事长",
                "allow_persistence": False,
                "pattern_key": "stop/disable system service",
                "pattern_keys": ["stop/disable system service"],
            },
        )

    assert store.count_pending_approval_requests(session_key) == 2
