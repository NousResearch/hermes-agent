"""Store-level durable event busy-rerun queue.

fire_claim (durable lease) and the scheduler in-memory running guard are
separate authorities. Event admission is one atomic profile-scoped store
operation: embed a batch in a new claim, or merge into one pending rerun
batch, with bounded receipts. Event reruns must not consume scheduled
occurrences.
"""

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _contexts(job):
    pending = job.get("pending_event_batch") or {}
    events = pending.get("events") if isinstance(pending, dict) else pending
    if not isinstance(events, list):
        events = []
    claim = job.get("fire_claim") if isinstance(job.get("fire_claim"), dict) else {}
    claimed = (claim.get("event_batch") or {}).get("events") or []
    return list(events), list(claimed)


class TestEventAdmissionQueue:
    def test_duplicate_delivery_does_not_add_another_run(self, temp_home):
        from cron.jobs import admit_job_event, claim_job_for_fire, create_job, get_job

        job = create_job(prompt="x", schedule="every 5m", name="q")
        assert claim_job_for_fire(job["id"], manual=True) is True
        first = admit_job_event(job["id"], delivery_id="d1", context="alpha")
        again = admit_job_event(job["id"], delivery_id="d1", context="alpha-retry")
        assert first["status"] == "queued"
        assert again["status"] == "duplicate"
        pending, _claimed = _contexts(get_job(job["id"]))
        assert [event["delivery_id"] for event in pending] == ["d1"]
        assert pending[0]["context"] == "alpha"

    def test_distinct_busy_events_batch_in_order(self, temp_home):
        from cron.jobs import admit_job_event, claim_job_for_fire, create_job, get_job

        job = create_job(prompt="x", schedule="every 5m", name="q")
        assert claim_job_for_fire(job["id"], manual=True) is True
        assert admit_job_event(job["id"], delivery_id="d1", context="one")["status"] == "queued"
        assert admit_job_event(job["id"], delivery_id="d2", context="two")["status"] == "queued"
        pending, _claimed = _contexts(get_job(job["id"]))
        assert [event["delivery_id"] for event in pending] == ["d1", "d2"]
        assert [event["context"] for event in pending] == ["one", "two"]

    def test_count_overflow_is_unavailable_without_receipt(self, temp_home):
        from cron.jobs import (
            EVENT_BATCH_MAX_EVENTS,
            admit_job_event,
            claim_job_for_fire,
            create_job,
            get_job,
        )

        job = create_job(prompt="x", schedule="every 5m", name="q")
        assert claim_job_for_fire(job["id"], manual=True) is True
        for i in range(EVENT_BATCH_MAX_EVENTS):
            result = admit_job_event(job["id"], delivery_id=f"d{i}", context=f"c{i}")
            assert result["status"] == "queued"
        overflow_id = "overflow-count"
        result = admit_job_event(job["id"], delivery_id=overflow_id, context="too-many")
        assert result["status"] == "unavailable"
        refreshed = get_job(job["id"])
        pending, _claimed = _contexts(refreshed)
        assert len(pending) == EVENT_BATCH_MAX_EVENTS
        assert overflow_id not in (refreshed.get("event_delivery_receipts") or {})

    def test_byte_overflow_is_unavailable_without_receipt(self, temp_home):
        from cron.jobs import (
            EVENT_BATCH_MAX_BYTES,
            admit_job_event,
            claim_job_for_fire,
            create_job,
            get_job,
        )

        job = create_job(prompt="x", schedule="every 5m", name="q")
        assert claim_job_for_fire(job["id"], manual=True) is True
        huge = "x" * (EVENT_BATCH_MAX_BYTES + 1)
        result = admit_job_event(job["id"], delivery_id="huge", context=huge)
        assert result["status"] == "unavailable"
        refreshed = get_job(job["id"])
        pending, _claimed = _contexts(refreshed)
        assert pending == []
        assert "huge" not in (refreshed.get("event_delivery_receipts") or {})

    def test_write_failure_is_unavailable_without_receipt(self, temp_home, monkeypatch):
        from cron import jobs as jobs_mod
        from cron.jobs import admit_job_event, create_job, get_job

        job = create_job(prompt="x", schedule="every 5m", name="q")

        def _boom(*_a, **_k):
            raise OSError("disk")

        monkeypatch.setattr(jobs_mod, "save_jobs", _boom)
        result = admit_job_event(job["id"], delivery_id="d-fail", context="ctx")
        assert result["status"] == "unavailable"
        refreshed = get_job(job["id"])
        assert "d-fail" not in (refreshed.get("event_delivery_receipts") or {})
        assert not (refreshed.get("pending_event_batch") or {}).get("events")

    def test_event_claim_does_not_shift_next_run_at(self, temp_home):
        from cron.jobs import admit_job_event, create_job, get_job

        job = create_job(prompt="x", schedule="every 5m", name="q")
        before = get_job(job["id"])["next_run_at"]
        result = admit_job_event(job["id"], delivery_id="d1", context="wake")
        assert result["status"] == "claimed"
        after = get_job(job["id"])["next_run_at"]
        assert after == before
        _pending, claimed = _contexts(get_job(job["id"]))
        assert claimed[0]["context"] == "wake"


class TestEventCompletionFollowUp:
    def test_owner_fenced_completion_arms_one_followup_without_clearing_pending(
        self, temp_home
    ):
        """A finishing event run with pending intent must stay due once, keep
        the pending batch, and leave the scheduled next_run_at untouched."""
        from cron.jobs import (
            admit_job_event,
            create_job,
            get_due_jobs,
            get_job,
            mark_job_run,
        )

        job = create_job(prompt="x", schedule="every 5m", name="q")
        scheduled_next = get_job(job["id"])["next_run_at"]
        first = admit_job_event(job["id"], delivery_id="d1", context="running")
        assert first["status"] == "claimed"
        owner = get_job(job["id"])["fire_claim"]["by"]
        queued = admit_job_event(job["id"], delivery_id="d2", context="follow")
        assert queued["status"] == "queued"

        assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True
        refreshed = get_job(job["id"])
        pending, claimed = _contexts(refreshed)
        assert claimed == []
        assert [event["delivery_id"] for event in pending] == ["d2"]
        assert refreshed.get("event_rerun_due") is True
        assert refreshed["next_run_at"] == scheduled_next
        assert refreshed.get("fire_claim") in (None, {})

        due_ids = [item["id"] for item in get_due_jobs()]
        assert job["id"] in due_ids
        still = get_job(job["id"])
        assert [event["delivery_id"] for event in _contexts(still)[0]] == ["d2"]
        assert still["next_run_at"] == scheduled_next

    def test_next_claim_promotes_pending_batch_and_clears_it(self, temp_home):
        from cron.jobs import (
            admit_job_event,
            claim_job_for_fire,
            create_job,
            get_job,
            mark_job_run,
        )

        job = create_job(prompt="x", schedule="every 5m", name="q")
        scheduled_next = get_job(job["id"])["next_run_at"]
        first = admit_job_event(job["id"], delivery_id="d1", context="running")
        owner = first["job"]["fire_claim"]["by"]
        assert admit_job_event(job["id"], delivery_id="d2", context="follow")["status"] == "queued"
        assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True

        promoted = claim_job_for_fire(job["id"], return_job=True)
        assert isinstance(promoted, dict)
        pending, claimed = _contexts(promoted)
        assert pending == []
        assert [event["delivery_id"] for event in claimed] == ["d2"]
        assert claimed[0]["context"] == "follow"
        assert promoted.get("event_rerun_due") in (None, False)
        assert promoted.get("_scheduled_instant") in (None, False)
        refreshed = get_job(job["id"])
        assert refreshed["next_run_at"] == scheduled_next
        assert not (refreshed.get("pending_event_batch") or {}).get("events")


class TestEventClaimRestartRecovery:
    def test_stale_event_bearing_claim_moves_back_to_pending(self, temp_home):
        from datetime import timedelta

        from cron.jobs import (
            admit_job_event,
            create_job,
            get_due_jobs,
            get_job,
            load_jobs,
            save_jobs,
        )
        from cron import jobs as jobs_mod

        job = create_job(prompt="x", schedule="every 5m", name="q")
        scheduled_next = get_job(job["id"])["next_run_at"]
        result = admit_job_event(job["id"], delivery_id="d-stale", context="survives restart")
        assert result["status"] == "claimed"

        records = load_jobs()
        for record in records:
            if record["id"] == job["id"]:
                record["fire_claim"]["at"] = (
                    jobs_mod._hermes_now() - timedelta(seconds=jobs_mod.FIRE_CLAIM_TTL_SECONDS + 5)
                ).isoformat()
        save_jobs(records)

        due_ids = [item["id"] for item in get_due_jobs()]
        assert job["id"] in due_ids
        refreshed = get_job(job["id"])
        pending, claimed = _contexts(refreshed)
        assert claimed == []
        assert [event["delivery_id"] for event in pending] == ["d-stale"]
        assert pending[0]["context"] == "survives restart"
        assert refreshed.get("event_rerun_due") is True
        assert refreshed.get("fire_claim") in (None, {})
        assert refreshed["next_run_at"] == scheduled_next


class TestEventLifecycle:
    def test_pause_refuses_admission_and_keeps_pending_until_resume(self, temp_home):
        from cron.jobs import (
            admit_job_event,
            claim_job_for_fire,
            create_job,
            get_due_jobs,
            get_job,
            pause_job,
            resume_job,
        )

        job = create_job(prompt="x", schedule="every 5m", name="q")
        assert claim_job_for_fire(job["id"], manual=True) is True
        owner = get_job(job["id"])["fire_claim"]["by"]
        assert admit_job_event(job["id"], delivery_id="d-pause", context="held")["status"] == "queued"
        pause_job(job["id"])
        refused = admit_job_event(job["id"], delivery_id="d-new", context="nope")
        assert refused["status"] == "unavailable"
        from cron.jobs import mark_job_run

        assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True
        assert job["id"] not in [item["id"] for item in get_due_jobs()]
        pending, _claimed = _contexts(get_job(job["id"]))
        assert [event["delivery_id"] for event in pending] == ["d-pause"]
        resume_job(job["id"])
        refreshed = get_job(job["id"])
        assert refreshed.get("event_rerun_due") is True
        assert [event["delivery_id"] for event in _contexts(refreshed)[0]] == ["d-pause"]
        assert job["id"] in [item["id"] for item in get_due_jobs()]

    def test_delete_drops_pending_and_refuses_admission(self, temp_home):
        from cron.jobs import (
            admit_job_event,
            claim_job_for_fire,
            create_job,
            get_job,
            remove_job,
        )

        job = create_job(prompt="x", schedule="every 5m", name="q")
        assert claim_job_for_fire(job["id"], manual=True) is True
        assert admit_job_event(job["id"], delivery_id="d-del", context="gone")["status"] == "queued"
        assert remove_job(job["id"]) is True
        assert get_job(job["id"]) is None
        assert admit_job_event(job["id"], delivery_id="d-after", context="x")["status"] == "unavailable"

    def test_event_run_does_not_consume_repeat_budget(self, temp_home):
        from cron.jobs import admit_job_event, create_job, get_job, mark_job_run

        job = create_job(prompt="x", schedule="every 5m", name="q", repeat=3)
        before = (get_job(job["id"]).get("repeat") or {}).get("completed", 0)
        result = admit_job_event(job["id"], delivery_id="d-rep", context="wake")
        owner = result["job"]["fire_claim"]["by"]
        assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True
        after = (get_job(job["id"]).get("repeat") or {}).get("completed", 0)
        assert after == before

    def test_finite_oneshot_followup_runs_then_completes(self, temp_home):
        from cron.jobs import (
            admit_job_event,
            claim_dispatch,
            claim_job_for_fire,
            create_job,
            get_job,
            is_job_runnable,
            mark_job_run,
        )

        job = create_job(prompt="x", schedule="in 30m", name="once")
        assert job["schedule"]["kind"] == "once"
        assert claim_dispatch(job["id"]) is True
        assert claim_job_for_fire(job["id"], manual=True) is True
        owner = get_job(job["id"])["fire_claim"]["by"]
        assert admit_job_event(job["id"], delivery_id="d-os", context="after")["status"] == "queued"
        assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True
        mid = get_job(job["id"])
        assert is_job_runnable(mid)
        assert mid.get("state") != "completed"
        assert [event["delivery_id"] for event in _contexts(mid)[0]] == ["d-os"]
        promoted = claim_job_for_fire(job["id"], return_job=True)
        assert isinstance(promoted, dict)
        event_owner = promoted["fire_claim"]["by"]
        assert mark_job_run(job["id"], True, expected_fire_owner=event_owner) is True
        done = get_job(job["id"])
        assert done["state"] == "completed"
        assert not is_job_runnable(done)

    def test_profile_stores_are_isolated(self, temp_home):
        from pathlib import Path

        from cron.jobs import admit_job_event, claim_job_for_fire, create_job, get_job, use_cron_store

        home_a = Path(temp_home) / "a"
        home_b = Path(temp_home) / "b"
        home_a.mkdir()
        home_b.mkdir()
        with use_cron_store(home_a):
            job_a = create_job(prompt="a", schedule="every 5m", name="shared")
            assert claim_job_for_fire(job_a["id"], manual=True) is True
            assert admit_job_event(job_a["id"], delivery_id="d-a", context="alpha")["status"] == "queued"
        with use_cron_store(home_b):
            job_b = create_job(prompt="b", schedule="every 5m", name="shared")
            pending_b, _ = _contexts(get_job(job_b["id"]))
            assert pending_b == []
            assert admit_job_event(job_b["id"], delivery_id="d-a", context="beta")["status"] == "claimed"
        with use_cron_store(home_a):
            pending_a, _ = _contexts(get_job(job_a["id"]))
            assert [event["context"] for event in pending_a] == ["alpha"]


class TestEventSchedulerWiring:
    def test_run_one_job_uses_event_batch_and_skips_oneshot_dispatch(self, temp_home, monkeypatch):
        import cron.scheduler as sched

        captured = {}

        def fake_run_job(job, *, extra_prompt=None, **_kw):
            captured["extra_prompt"] = extra_prompt
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_k: True)
        monkeypatch.setattr(sched, "heartbeat_fire_claim", lambda *_a, **_k: True)
        monkeypatch.setattr(
            sched,
            "claim_dispatch",
            lambda _jid: (_ for _ in ()).throw(AssertionError("event rerun consumed claim_dispatch")),
        )
        job = {
            "id": "evt-1",
            "name": "t",
            "prompt": "base",
            "schedule": {"kind": "once"},
            "repeat": {"times": 1, "completed": 1},
            "fire_claim": {
                "by": "owner",
                "event_batch": {
                    "events": [{"delivery_id": "d1", "context": "wake ctx", "accepted_at": "t"}]
                },
            },
        }
        assert sched.run_one_job(job) is True
        assert captured.get("extra_prompt") and "wake ctx" in captured["extra_prompt"]
