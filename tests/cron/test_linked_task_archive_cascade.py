"""Issue #106744 — archive/stop must pause only cron jobs stamped with linked_task_id.

Detection: after archive_task(T), any job with linked_task_id == T that is still
enabled or claimable via claim_job_for_fire is a failure. Unlinked jobs (including
those sharing origin.chat_id) stay enabled. A pause_linked / archive receipt lists
only jobs that were actually paused.
"""

from __future__ import annotations

import json

import pytest

from cron.jobs import (
    claim_job_for_fire,
    create_job,
    get_job,
    load_jobs,
    save_jobs,
)
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools.cronjob_tools import cronjob


ORIGIN = {"platform": "telegram", "chat_id": "chat-106744"}


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@pytest.fixture()
def kanban_conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kbc.connect(db_path=tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _stamp_linked_task(job_id: str, task_id: str) -> None:
    """Persist linked_task_id without going through create_job (pre-fix has no param)."""
    jobs = load_jobs()
    for job in jobs:
        if job.get("id") == job_id:
            job["linked_task_id"] = task_id
            break
    else:
        raise AssertionError(f"job {job_id} missing from store")
    save_jobs(jobs)


def _archive_pause_receipt():
    return getattr(kb.archive_task, "last_pause_receipt", {"paused": []})


def test_archive_pauses_only_jobs_stamped_to_that_task(tmp_cron_dir, kanban_conn):
    tid = kb.create_task(kanban_conn, title="initiative-106744")
    try:
        linked = create_job(
            prompt="Kanban monitor for initiative",
            schedule="every 1m",
            name="linked-monitor",
            origin=ORIGIN,
            linked_task_id=tid,
        )
    except TypeError:
        linked = create_job(
            prompt="Kanban monitor for initiative",
            schedule="every 1m",
            name="linked-monitor",
            origin=ORIGIN,
        )
    unlinked = create_job(
        prompt="Unrelated same-chat reminder",
        schedule="every 1m",
        name="unlinked-same-chat",
        origin=ORIGIN,
    )
    if get_job(linked["id"]).get("linked_task_id") != tid:
        _stamp_linked_task(linked["id"], tid)

    assert kb.archive_task(kanban_conn, tid)

    stored_linked = get_job(linked["id"])
    stored_unlinked = get_job(unlinked["id"])
    assert stored_linked["enabled"] is False
    assert claim_job_for_fire(linked["id"]) is False
    assert stored_unlinked["enabled"] is True
    assert stored_unlinked.get("linked_task_id") in (None, "", False)

    receipt = _archive_pause_receipt()
    paused_ids = list(receipt.get("paused") or [])
    assert paused_ids == [linked["id"]]
    assert unlinked["id"] not in paused_ids


def test_pause_linked_receipt_lists_paused_and_skips_unlinked(tmp_cron_dir):
    tid = "task-106744-pause-linked"
    try:
        created = json.loads(
            cronjob(
                action="create",
                prompt="Linked kanban monitor",
                schedule="every 1m",
                name="linked-j",
                linked_task_id=tid,
            )
        )
    except TypeError:
        created = {"success": False}
    if not created.get("success"):
        created = json.loads(
            cronjob(action="create", prompt="Linked kanban monitor", schedule="every 1m", name="linked-j")
        )
        assert created.get("success") is True
        _stamp_linked_task(created["job_id"], tid)
    elif get_job(created["job_id"]).get("linked_task_id") != tid:
        _stamp_linked_task(created["job_id"], tid)

    unlinked = json.loads(
        cronjob(action="create", prompt="Unlinked same-chat job", schedule="every 1m", name="unlinked-k")
    )
    assert unlinked.get("success") is True

    payload = json.loads(cronjob(action="pause_linked", task_id=tid))
    paused_ids = list(payload.get("paused") or [])
    assert paused_ids == [created["job_id"]]
    assert payload.get("skipped_unlinked", 0) >= 1
    assert payload.get("errors", []) == []
    assert get_job(created["job_id"])["enabled"] is False
    assert get_job(unlinked["job_id"])["enabled"] is True
    assert "linked_task_id" not in get_job(unlinked["job_id"])


def test_stop_leaves_unlinked_same_chat_cron_enabled(tmp_cron_dir, monkeypatch):
    """CONTROL: /stop / session interrupt must not pause unlinked cron by origin.chat_id."""
    job = create_job(
        prompt="Unlinked same-chat monitor",
        schedule="every 1m",
        name="unlinked-control",
        origin=ORIGIN,
    )
    monkeypatch.setattr("tools.process_registry.process_registry.kill_all", lambda: 0)

    def _interrupt_all(**_kw):
        return 0

    monkeypatch.setattr("tools.async_delegation.interrupt_all", _interrupt_all, raising=False)

    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    CLICommandsMixin._handle_stop_command(object())

    stored = get_job(job["id"])
    assert stored["enabled"] is True
    assert claim_job_for_fire(job["id"]) is not False


def test_create_omits_blank_linked_task_id(tmp_cron_dir):
    job = create_job(prompt="legacy unstamped", schedule="every 1h", linked_task_id="  ")
    assert "linked_task_id" not in job
    assert "linked_task_id" not in get_job(job["id"])


def test_archive_succeeds_when_cron_cascade_raises(tmp_cron_dir, kanban_conn, monkeypatch):
    """Fail-open: cron store / lock failure must not block archive_task success."""
    tid = kb.create_task(kanban_conn, title="fail-open-archive")

    def _boom(*_a, **_k):
        raise RuntimeError("cron lock failed")

    monkeypatch.setattr("cron.jobs.pause_jobs_for_task", _boom)
    assert kb.archive_task(kanban_conn, tid) is True
    assert kb.get_task(kanban_conn, tid).status == "archived"
    receipt = _archive_pause_receipt()
    assert receipt["paused"] == []
    assert receipt["errors"]
