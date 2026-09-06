
from hermes_cli import kanban_db_connect, kanban_db_dispatch, kanban_db_notify
"""Canonical delivery-review events cross the real DB/notifier boundary."""

import asyncio
import json

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


HEAD_A = "a" * 40
HEAD_B = "b" * 40


class FakeDiscordTransport:
    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(
            {"chat_id": chat_id, "text": text, "metadata": dict(metadata or {})}
        )
        return SendResult(success=True, message_id=f"discord-{len(self.sent)}")

    async def handle_message(self, event):
        self.handled.append(event)


class RejectFirstDiscordTransport(FakeDiscordTransport):
    async def send(self, chat_id, text, metadata=None):
        self.sent.append(
            {"chat_id": chat_id, "text": text, "metadata": dict(metadata or {})}
        )
        if len(self.sent) == 1:
            return SendResult(
                success=False,
                error="temporary Discord rejection",
                raw_response={"delivery_rejected": True},
            )
        return SendResult(success=True, message_id=f"discord-{len(self.sent)}")


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.DISCORD: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


async def _one_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _notify(monkeypatch, adapter):
    asyncio.run(_one_tick(monkeypatch, _runner(adapter)))


@pytest.fixture
def board(tmp_path, monkeypatch):
    db = tmp_path / "delivery-review.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(kanban_db_dispatch, "_memory_pressure_level", lambda: "normal")
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _: True)
    kanban_db_connect.init_db()
    with kanban_db_connect.connect(db) as conn:
        implementation = kb.create_task(
            conn, title="Repair delivery notifier", assignee="integrator"
        )
        review = kb.create_task(
            conn,
            title="Independent exact-head review",
            assignee="reviewer",
            parents=[implementation],
        )
        for task_id in (implementation, review):
            kanban_db_notify.add_notify_sub(
                conn,
                task_id=task_id,
                platform="discord",
                chat_id="origin-channel",
                thread_id="origin-thread",
                chat_type="thread",
                delivery_mode="notify",
            )
    return db, implementation, review


def _candidate(review_task, head, *, ci="success"):
    return {
        "review_requirement": {
            "required": True,
            "owner": "orchestrator",
            "review_task_id": review_task,
        },
        "delivery_review": {
            "head": head,
            "evidence": {
                "artifact": "repo:wt/notifier-integrated",
                "checks": [{"command": "focused gateway tests", "result": "passed"}],
                "ci_head": head,
                "ci": ci,
                "draft": False,
                "proof_head": head,
                "proof": "passed",
            },
        },
    }


def _submit(conn, implementation, review_task, head, *, ci="success"):
    claimed = kb.claim_task(conn, implementation, claimer="remote:integrator")
    assert claimed is not None
    assert kb.complete_task(
        conn,
        implementation,
        summary=f"Candidate {head[:8]} ready for exact-head review",
        metadata=_candidate(review_task, head, ci=ci),
        expected_run_id=claimed.current_run_id,
    )


def _dispatch(conn, *, max_spawn=0):
    return kanban_db_dispatch.dispatch_once(
        conn,
        spawn_fn=lambda *args: None,
        reconcile_orphans=False,
        max_spawn=max_spawn,
    )


def _review(conn, review_task, head, verdict, findings=""):
    task = kb.get_task(conn, review_task)
    assert task.status == "running"
    assert kb.complete_task(
        conn,
        review_task,
        summary=f"{verdict} exact candidate {head[:8]}",
        metadata={
            "delivery_review": {
                "head": head,
                "verdict": verdict,
                "findings": findings,
            }
        },
        expected_run_id=task.current_run_id,
    )


def _events(conn, task_id, kind):
    return [event for event in kb.list_events(conn, task_id) if event.kind == kind]


def test_canonical_review_generations_render_once_through_durable_discord_ledger(
    board, monkeypatch
):
    db, implementation, review_task = board
    adapter = FakeDiscordTransport()

    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_A)
        assert not _dispatch(conn).spawned
        phase_a = _events(conn, implementation, "delivery_phase_completed")[-1]
        assert phase_a.payload["transition"] == (
            f"{implementation}:{review_task}:{HEAD_A}"
        )
        assert phase_a.payload["acceptance"] == "pending"
    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 1
    assert "PHASE COMPLETE" in adapter.sent[-1]["text"]
    assert "Ready: no" in adapter.sent[-1]["text"]
    assert implementation in adapter.sent[-1]["text"]
    assert review_task in adapter.sent[-1]["text"]
    assert HEAD_A in adapter.sent[-1]["text"]

    with kanban_db_connect.connect(db) as conn:
        assert [item[0] for item in _dispatch(conn, max_spawn=1).spawned] == [
            review_task
        ]
        findings = "Ordering breaks replay: preserve event 17 before event 18."
        _review(conn, review_task, HEAD_A, "BLOCK", findings)
        assert not _dispatch(conn).spawned
        changes = _events(conn, implementation, "delivery_changes_requested")[-1]
        assert changes.payload["transition"] == phase_a.payload["transition"]
        assert changes.payload["status"] == "ready"
        assert changes.payload["findings"] == findings
        assert changes.payload["review_run"]
        reviewer_completion = _events(conn, review_task, "completed")[-1]
        assert reviewer_completion.payload["completion_kind"] == "delivery_review_result"
        assert reviewer_completion.payload["ready"] is False
        # Persist the legacy/partial shape that omitted the advisory `ready`
        # flag. Classification must stay tied to completion_kind, never to a
        # missing boolean or the review prose.
        completion_payload = dict(reviewer_completion.payload)
        completion_payload.pop("ready")
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE id = ?",
            (json.dumps(completion_payload, sort_keys=True), reviewer_completion.id),
        )
    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 3
    notices = [item["text"] for item in adapter.sent[-2:]]
    reviewer_notice = next(text for text in notices if "REVIEW RESULT RECORDED" in text)
    changes_notice = next(text for text in notices if "CHANGES REQUESTED" in text)
    assert "REVIEW RESULT RECORDED" in reviewer_notice
    assert "Ready: no" in reviewer_notice
    assert "review_approved" not in reviewer_notice
    assert implementation in reviewer_notice
    assert "CHANGES REQUESTED" in changes_notice
    assert "Ready: no" in changes_notice
    assert findings in changes_notice
    assert "@integrator" in changes_notice
    assert HEAD_A in changes_notice

    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_B)
        assert not _dispatch(conn).spawned
    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 4
    assert "PHASE COMPLETE" in adapter.sent[-1]["text"]
    assert HEAD_B in adapter.sent[-1]["text"]

    with kanban_db_connect.connect(db) as conn:
        assert [item[0] for item in _dispatch(conn, max_spawn=1).spawned] == [
            review_task
        ]
        _review(conn, review_task, HEAD_B, "PASS")
        assert not _dispatch(conn).spawned
        accepted = _events(conn, implementation, "delivery_accepted")[-1]
        assert accepted.payload["transition"] == (
            f"{implementation}:{review_task}:{HEAD_B}"
        )
        assert accepted.payload["acceptance"] == "ready"
        assert accepted.payload["review_run"]
    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 6
    notices = [item["text"] for item in adapter.sent[-2:]]
    reviewer_notice = next(text for text in notices if "REVIEW RESULT RECORDED" in text)
    accepted_notice = next(text for text in notices if "DELIVERY ACCEPTED" in text)
    assert "REVIEW RESULT RECORDED" in reviewer_notice
    assert "Ready: no" in reviewer_notice
    assert "DELIVERY ACCEPTED" in accepted_notice
    assert "Ready: yes" in accepted_notice
    assert HEAD_B in accepted_notice
    assert review_task in accepted_notice

    # Two fresh watcher instances model restart/replay. The SQLite ledger has
    # acknowledged each event, so neither restart duplicates a notice.
    _notify(monkeypatch, adapter)
    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 6


def test_canonical_review_hold_renders_exact_reason_without_handoff_duplicate(
    board, monkeypatch
):
    db, implementation, review_task = board
    adapter = FakeDiscordTransport()
    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_A, ci="failure")
        assert not _dispatch(conn).spawned
        assert [item[0] for item in _dispatch(conn, max_spawn=1).spawned] == [
            review_task
        ]
        _review(conn, review_task, HEAD_A, "PASS")
        assert not _dispatch(conn).spawned
        hold = _events(conn, implementation, "delivery_review_hold")[-1]
        assert hold.payload["transition"].startswith(
            f"{implementation}:{review_task}:{HEAD_A}:"
        )
        assert hold.payload["reason"] == (
            "review passed; CI, nonDraft or proof gate missing/failed/stale"
        )
    _notify(monkeypatch, adapter)

    texts = [item["text"] for item in adapter.sent]
    assert len(texts) == 3
    assert sum("PHASE COMPLETE" in text for text in texts) == 1
    assert sum("REVIEW RESULT RECORDED" in text for text in texts) == 1
    assert sum("DELIVERY HOLD" in text for text in texts) == 1
    hold_notice = next(text for text in texts if "DELIVERY HOLD" in text)
    assert "Ready: no" in hold_notice
    assert hold.payload["reason"] in hold_notice
    assert "@orchestrator" in hold_notice
    assert implementation in hold_notice
    assert review_task in hold_notice
    assert HEAD_A in hold_notice


def test_generation_handoff_before_controller_tick_names_existing_review_task(
    board, monkeypatch
):
    db, implementation, review_task = board
    adapter = FakeDiscordTransport()
    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_A)
        handoff = _events(conn, implementation, "review_handoff_required")[-1]
        assert handoff.payload["review_task_id"] == review_task

    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 1
    notice = adapter.sent[0]["text"]
    assert "REVIEW HANDOFF" in notice
    assert "Ready: no" in notice
    assert review_task in notice
    assert "canonical review task is missing" not in notice
    assert "@orchestrator" in notice


def test_old_generation_canonical_events_do_not_swallow_new_generation_handoff(
    board, monkeypatch
):
    db, implementation, review_task = board
    adapter = FakeDiscordTransport()

    # Hold all notification polling while generation A reaches BLOCK, then
    # submit generation B without giving the delivery controller another tick.
    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_A)
        assert not _dispatch(conn).spawned
        assert [item[0] for item in _dispatch(conn, max_spawn=1).spawned] == [
            review_task
        ]
        _review(conn, review_task, HEAD_A, "BLOCK", "Generation A is stale.")
        assert not _dispatch(conn).spawned
        _submit(conn, implementation, review_task, HEAD_B)

        handoff_b = _events(conn, implementation, "review_handoff_required")[-1]
        phase_a = _events(conn, implementation, "delivery_phase_completed")[-1]
        assert handoff_b.run_id is not None
        assert phase_a.run_id is not None
        assert handoff_b.run_id != phase_a.run_id

    _notify(monkeypatch, adapter)

    texts = [item["text"] for item in adapter.sent]
    assert sum("PHASE COMPLETE" in text for text in texts) == 1
    assert sum("CHANGES REQUESTED" in text for text in texts) == 1
    handoffs = [text for text in texts if "REVIEW HANDOFF" in text]
    assert len(handoffs) == 1
    assert HEAD_B[:8] in handoffs[0]
    assert review_task in handoffs[0]


def test_failed_canonical_phase_after_precursor_coalescing_retries_once_after_restart(
    board, monkeypatch
):
    db, implementation, review_task = board
    rejected = RejectFirstDiscordTransport()

    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_A)
        assert not _dispatch(conn).spawned
        handoff = _events(conn, implementation, "review_handoff_required")[-1]
        phase = _events(conn, implementation, "delivery_phase_completed")[-1]

    # The matching handoff precursor is coalesced, but a rejected canonical
    # phase remains pending in the existing durable delivery ledger.
    _notify(monkeypatch, rejected)
    assert len(rejected.sent) == 1
    assert "PHASE COMPLETE" in rejected.sent[0]["text"]
    with kanban_db_connect.connect(db) as conn:
        cursor = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs WHERE task_id = ?",
            (implementation,),
        ).fetchone()["last_event_id"]
        pending = conn.execute(
            "SELECT state, attempt_count FROM kanban_notify_deliveries "
            "WHERE task_id = ? AND event_id = ?",
            (implementation, phase.id),
        ).fetchone()
        assert cursor == handoff.id
        assert dict(pending) == {"state": "pending", "attempt_count": 1}

    restarted = FakeDiscordTransport()
    _notify(monkeypatch, restarted)
    assert len(restarted.sent) == 1
    assert "PHASE COMPLETE" in restarted.sent[0]["text"]
    assert HEAD_A in restarted.sent[0]["text"]
    with kanban_db_connect.connect(db) as conn:
        cursor = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs WHERE task_id = ?",
            (implementation,),
        ).fetchone()["last_event_id"]
        assert cursor == phase.id
        assert conn.execute(
            "SELECT 1 FROM kanban_notify_deliveries "
            "WHERE task_id = ? AND event_id = ?",
            (implementation, phase.id),
        ).fetchone() is None

    # A second restart sees the successful acknowledgement and sends nothing.
    _notify(monkeypatch, restarted)
    assert len(restarted.sent) == 1


def test_precursor_ack_failure_releases_no_send_claim_for_fresh_watchers(
    board, monkeypatch
):
    db, implementation, review_task = board
    adapter = FakeDiscordTransport()

    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_A)
        assert not _dispatch(conn).spawned
        handoff = _events(conn, implementation, "review_handoff_required")[-1]
        phase = _events(conn, implementation, "delivery_phase_completed")[-1]

    faulted = _runner(adapter)
    real_ack = faulted._kanban_ack_delivery
    failed_once = False

    def fail_precursor_ack_once(sub, event_id, *args):
        nonlocal failed_once
        if event_id == handoff.id and not failed_once:
            failed_once = True
            raise RuntimeError("fault before precursor acknowledgement transaction")
        return real_ack(sub, event_id, *args)

    faulted._kanban_ack_delivery = fail_precursor_ack_once
    asyncio.run(_one_tick(monkeypatch, faulted))
    assert adapter.sent == []
    with kanban_db_connect.connect(db) as conn:
        row = conn.execute(
            "SELECT state, claim_token, claim_owner FROM kanban_notify_deliveries "
            "WHERE task_id = ? AND event_id = ?",
            (implementation, handoff.id),
        ).fetchone()
        assert dict(row) == {
            "state": "pending",
            "claim_token": None,
            "claim_owner": None,
        }

    # Fresh watcher objects in the same process used to see the live owner of
    # the stranded `sending` row forever. The first recovery tick coalesces the
    # precursor and emits its canonical successor; the second proves replay is
    # settled rather than duplicated.
    _notify(monkeypatch, adapter)
    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 1
    assert "PHASE COMPLETE" in adapter.sent[0]["text"]
    with kanban_db_connect.connect(db) as conn:
        assert conn.execute(
            "SELECT 1 FROM kanban_notify_deliveries "
            "WHERE task_id = ? AND event_id IN (?, ?)",
            (implementation, handoff.id, phase.id),
        ).fetchall() == []


def test_precursor_post_commit_ack_exception_does_not_resurrect_row(
    board, monkeypatch
):
    db, implementation, review_task = board
    adapter = FakeDiscordTransport()

    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_A)
        assert not _dispatch(conn).spawned
        handoff = _events(conn, implementation, "review_handoff_required")[-1]

    faulted = _runner(adapter)
    real_ack = faulted._kanban_ack_delivery
    failed_once = False

    def commit_then_fail_once(sub, event_id, *args):
        nonlocal failed_once
        acknowledged = real_ack(sub, event_id, *args)
        if event_id == handoff.id and not failed_once:
            failed_once = True
            assert acknowledged
            raise RuntimeError("response lost after precursor ack commit")
        return acknowledged

    faulted._kanban_ack_delivery = commit_then_fail_once
    asyncio.run(_one_tick(monkeypatch, faulted))
    assert adapter.sent == []
    with kanban_db_connect.connect(db) as conn:
        assert conn.execute(
            "SELECT 1 FROM kanban_notify_deliveries "
            "WHERE task_id = ? AND event_id = ?",
            (implementation, handoff.id),
        ).fetchone() is None

    _notify(monkeypatch, adapter)
    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 1
    assert "PHASE COMPLETE" in adapter.sent[0]["text"]


def test_precursor_ack_recovery_is_fenced_from_overlapping_successor_claim(
    board, monkeypatch
):
    db, implementation, review_task = board
    adapter = FakeDiscordTransport()

    with kanban_db_connect.connect(db) as conn:
        _submit(conn, implementation, review_task, HEAD_A)
        assert not _dispatch(conn).spawned
        handoff = _events(conn, implementation, "review_handoff_required")[-1]

    faulted = _runner(adapter)
    replacement_claim = None

    def replace_claim_then_fail(sub, event_id, claim_token, message_id, board_slug):
        nonlocal replacement_claim
        assert event_id == handoff.id
        assert faulted._kanban_retry_delivery(
            sub, event_id, claim_token, "simulated owner handoff", board_slug,
        )
        replacement_claim = faulted._kanban_begin_delivery(
            sub, event_id, board_slug,
        )
        assert replacement_claim and replacement_claim != claim_token
        raise RuntimeError("stale precursor owner lost acknowledgement response")

    faulted._kanban_ack_delivery = replace_claim_then_fail
    asyncio.run(_one_tick(monkeypatch, faulted))
    assert adapter.sent == []
    with kanban_db_connect.connect(db) as conn:
        row = conn.execute(
            "SELECT state, claim_token FROM kanban_notify_deliveries "
            "WHERE task_id = ? AND event_id = ?",
            (implementation, handoff.id),
        ).fetchone()
        assert dict(row) == {
            "state": "sending",
            "claim_token": replacement_claim,
        }

    # The stale catch path must not reset the successor's fenced ownership.
    assert faulted._kanban_ack_delivery is replace_claim_then_fail
    assert GatewayRunner._kanban_ack_delivery(
        faulted, {"task_id": implementation, "platform": "discord",
                  "chat_id": "origin-channel", "thread_id": "origin-thread"},
        handoff.id, replacement_claim, None, None,
    )
    _notify(monkeypatch, adapter)
    _notify(monkeypatch, adapter)
    assert len(adapter.sent) == 1
    assert "PHASE COMPLETE" in adapter.sent[0]["text"]
