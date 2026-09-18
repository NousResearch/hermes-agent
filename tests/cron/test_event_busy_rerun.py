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

        promoted = claim_job_for_fire(job["id"], return_job=True, event_rerun=True)
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
        # Production order: fire claim, then claim_dispatch inside run_one_job.
        assert claim_job_for_fire(job["id"], manual=True) is True
        assert claim_dispatch(job["id"]) is True
        owner = get_job(job["id"])["fire_claim"]["by"]
        assert admit_job_event(job["id"], delivery_id="d-os", context="after")["status"] == "queued"
        assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True
        mid = get_job(job["id"])
        assert is_job_runnable(mid)
        assert mid.get("state") != "completed"
        assert [event["delivery_id"] for event in _contexts(mid)[0]] == ["d-os"]
        promoted = claim_job_for_fire(job["id"], return_job=True, event_rerun=True)
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


class TestReviewRepros:
    def _arm_pending(self, context="ctx-ONE", delivery_id="d-pending"):
        from cron.jobs import admit_job_event, claim_job_for_fire, create_job, get_job, mark_job_run

        job = create_job(prompt="x", schedule="every 5m", name="q", repeat=5)
        first = admit_job_event(job["id"], delivery_id="d-run", context="running")
        owner = first["job"]["fire_claim"]["by"]
        assert admit_job_event(job["id"], delivery_id=delivery_id, context=context)["status"] == "queued"
        assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True
        refreshed = get_job(job["id"])
        assert refreshed.get("event_rerun_due") is True
        return refreshed, context

    def test_manual_run_with_prompt_does_not_consume_pending_batch(self, temp_home, monkeypatch):
        """Review F2: a manual/forced claim must not promote pending events under
        an unrelated operator prompt. Pending intent survives for its own event run."""
        from cron.jobs import claim_job_for_fire, get_job
        from tools.cronjob_tools import _execute_job_now
        import cron.scheduler as sched

        job, context = self._arm_pending()
        prompts = []

        def _fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", _fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)

        result = _execute_job_now(job, extra_prompt="operator says hi")
        assert result.get("claimed") is True
        joined = "\n".join(str(p) for p in prompts if p)
        assert "operator says hi" in joined
        assert context not in joined
        after = get_job(job["id"])
        pending, claimed = _contexts(after)
        assert [event["context"] for event in pending] == [context]
        assert claimed == []
        assert after.get("event_rerun_due") is True

        forced = claim_job_for_fire(job["id"], force=True, return_job=True)
        assert isinstance(forced, dict)
        assert not (forced.get("fire_claim") or {}).get("event_batch")
        pending_after_force, _ = _contexts(get_job(job["id"]))
        assert [event["context"] for event in pending_after_force] == [context]

        provider = claim_job_for_fire(job["id"], return_job=True)
        assert provider is False or (
            isinstance(provider, dict) and not (provider.get("fire_claim") or {}).get("event_batch")
        )
        still = get_job(job["id"])
        pending_still, _ = _contexts(still)
        assert [event["context"] for event in pending_still] == [context]
        assert still.get("event_rerun_due") is True

    def test_promoted_event_run_does_not_consume_due_scheduled_slot(self, temp_home, monkeypatch):
        """Review F3: when a scheduled slot is also due, the event follow-up must
        not stamp/skip that occurrence. The scheduled run remains due with its
        own repeat accounting."""
        from cron.jobs import get_job, load_jobs, save_jobs
        from cron import jobs as jobs_mod
        import cron.scheduler as sched

        job, context = self._arm_pending()
        due_slot = jobs_mod._hermes_now().isoformat()
        records = load_jobs()
        for record in records:
            if record["id"] == job["id"]:
                record["next_run_at"] = due_slot
        save_jobs(records)
        scheduled_next = get_job(job["id"])["next_run_at"]

        prompts = []

        def _fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", _fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)

        tick1 = sched.tick(verbose=False, sync=True)
        assert tick1 >= 1
        assert any(context in str(p) for p in prompts if p), f"event context missing: {prompts!r}"
        after1 = get_job(job["id"])
        assert after1["next_run_at"] == scheduled_next
        assert (after1.get("repeat") or {}).get("completed", 0) == 0
        assert not (after1.get("pending_event_batch") or {}).get("events")

        prompts.clear()
        tick2 = sched.tick(verbose=False, sync=True)
        assert tick2 >= 1, "scheduled occurrence was skipped after the event rerun"
        after2 = get_job(job["id"])
        assert after2["next_run_at"] != scheduled_next
        assert (after2.get("repeat") or {}).get("completed", 0) == 1
        assert not any(context in str(p) for p in prompts if p)

    def test_trigger_job_with_pending_runs_manual_then_event_once(self, temp_home, monkeypatch):
        """Review-r2 N1: dashboard/API trigger_job(prompt) while pending is armed
        must yield exactly two occurrence-free runs — operator prompt, then
        event batch — with no mixing, no phantom scheduled fire, and no
        repeat.completed bump."""
        from datetime import datetime

        from cron.jobs import get_job, trigger_job
        from cron import jobs as jobs_mod
        import cron.scheduler as sched

        job, context = self._arm_pending()
        operator = "operator-prompt"
        trigger_job(job["id"], extra_prompt=operator)
        triggered = get_job(job["id"])
        assert triggered.get("manual_run_at") == triggered.get("next_run_at")
        assert triggered.get("manual_run_prompt") == operator

        prompts = []

        def _fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", _fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)

        tick1 = sched.tick(verbose=False, sync=True)
        assert tick1 >= 1
        assert len(prompts) == 1
        assert prompts[0] == operator
        assert context not in str(prompts[0])
        after1 = get_job(job["id"])
        assert (after1.get("repeat") or {}).get("completed", 0) == 0
        assert [event["context"] for event in _contexts(after1)[0]] == [context]
        assert after1.get("event_rerun_due") is True
        assert after1.get("manual_run_at") in (None, False)
        assert after1.get("next_run_at") != triggered["next_run_at"]

        prompts.clear()
        tick2 = sched.tick(verbose=False, sync=True)
        assert tick2 >= 1
        assert len(prompts) == 1
        assert context in str(prompts[0])
        assert operator not in str(prompts[0])
        after2 = get_job(job["id"])
        assert (after2.get("repeat") or {}).get("completed", 0) == 0
        assert not (after2.get("pending_event_batch") or {}).get("events")
        assert after2.get("event_rerun_due") in (None, False)
        assert after2["next_run_at"] == after1["next_run_at"]
        assert datetime.fromisoformat(after2["next_run_at"]) > jobs_mod._hermes_now()

        prompts.clear()
        tick3 = sched.tick(verbose=False, sync=True)
        assert tick3 == 0
        assert prompts == []
        assert (get_job(job["id"]).get("repeat") or {}).get("completed", 0) == 0

    def test_event_rerun_claim_without_pending_fails_closed(self, temp_home):
        """Review-r2 N2: event_rerun=True with nothing to promote must not claim."""
        from cron.jobs import claim_job_for_fire, create_job, get_job

        job = create_job(prompt="x", schedule="every 5m", name="q", repeat=5)
        before = get_job(job["id"])
        claimed = claim_job_for_fire(job["id"], event_rerun=True, return_job=True)
        assert claimed is False
        after = get_job(job["id"])
        assert after["next_run_at"] == before["next_run_at"]
        assert (after.get("repeat") or {}).get("completed", 0) == (
            before.get("repeat") or {}
        ).get("completed", 0)
        assert not after.get("fire_claim")

    def test_stale_event_rerun_snapshot_does_not_run_after_batch_drains(
        self, temp_home, monkeypatch
    ):
        """Review-r2 N2 race: due-scan marked _event_rerun, another claimant
        drained the batch, then the queued snapshot must not fall through to
        a scheduled-style claim."""
        from cron.executions import create_execution
        from cron.jobs import claim_job_for_fire, get_due_jobs, get_job, mark_job_run
        import cron.scheduler as sched

        job, _context = self._arm_pending()
        due = next(item for item in get_due_jobs() if item["id"] == job["id"])
        assert due.get("_event_rerun") is True
        promoted = claim_job_for_fire(job["id"], event_rerun=True, return_job=True)
        assert isinstance(promoted, dict)
        owner = promoted["fire_claim"]["by"]
        assert mark_job_run(job["id"], True, expected_fire_owner=owner) is True
        drained = get_job(job["id"])
        next_before = drained["next_run_at"]
        completed_before = (drained.get("repeat") or {}).get("completed", 0)
        assert not (drained.get("pending_event_batch") or {}).get("events")

        ran = []

        def _fake_run_job(_job, *, extra_prompt=None, **_kw):
            ran.append(extra_prompt)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", _fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)

        due["execution_id"] = create_execution(job["id"], source="builtin")["id"]
        sched._process_due_job(due, adapters=None, loop=None, verbose=False)
        assert ran == []
        after = get_job(job["id"])
        assert after["next_run_at"] == next_before
        assert (after.get("repeat") or {}).get("completed", 0) == completed_before
        assert claim_job_for_fire(job["id"], event_rerun=True, return_job=True) is False


class TestReviewR3InflightTrigger:
    """Review-r3 R1: trigger_job while a real fire_claim is live must not
    become a phantom scheduled occurrence of the manual instant."""

    def _trigger_during_inflight(self, monkeypatch, *, mode, with_pending):
        import threading

        from cron import jobs as jobs_mod
        from cron.jobs import (
            admit_job_event,
            create_job,
            get_job,
            load_jobs,
            save_jobs,
            trigger_job,
        )
        from tools.cronjob_tools import _run_claimed_job
        import cron.scheduler as sched

        operator = "op-during-run"
        pending_ctx = "ctx-ONE"
        inflight_ctx = "running"
        started = threading.Event()
        release = threading.Event()
        prompts = []
        thread_errors = []

        def fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            started.set()
            assert release.wait(15)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)

        job = create_job(prompt="x", schedule="every 5m", name="q", repeat=5)
        job_id = job["id"]

        if mode == "event":
            claimed = admit_job_event(
                job_id, delivery_id="d-run", context=inflight_ctx
            )
            assert claimed["status"] == "claimed"
            snap = claimed["job"]

            def _run_event():
                try:
                    _run_claimed_job(snap, extra_prompt=inflight_ctx)
                except Exception as exc:
                    thread_errors.append(exc)
                    started.set()

            worker = threading.Thread(target=_run_event)
        else:
            records = load_jobs()
            for record in records:
                if record["id"] == job_id:
                    record["next_run_at"] = jobs_mod._hermes_now().isoformat()
            save_jobs(records)

            def _run_scheduled():
                try:
                    sched.tick(verbose=False, sync=True)
                except Exception as exc:
                    thread_errors.append(exc)
                    started.set()

            worker = threading.Thread(target=_run_scheduled)

        worker.start()
        assert started.wait(15), f"in-flight run never started ({mode}); errors={thread_errors!r}"
        assert not thread_errors

        if with_pending:
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"

        mid = get_job(job_id)
        assert mid.get("fire_claim")
        triggered = trigger_job(job_id, extra_prompt=operator)
        assert triggered is not None
        manual_at = triggered.get("manual_run_at")
        assert manual_at
        assert triggered.get("manual_run_prompt") == operator
        assert triggered.get("fire_claim")

        inflight_prompts = list(prompts)
        release.set()
        worker.join(30)
        assert not worker.is_alive()
        assert not thread_errors
        after = get_job(job_id)
        assert after.get("fire_claim") in (None, {})
        assert after.get("manual_run_prompt") == operator
        assert after.get("manual_run_at") == after.get("next_run_at") == manual_at

        ticks = []
        for _ in range(3):
            prompts.clear()
            n = sched.tick(verbose=False, sync=True)
            ticks.append((n, list(prompts)))
        final = get_job(job_id)
        return {
            "job_id": job_id,
            "operator": operator,
            "pending_ctx": pending_ctx,
            "inflight_ctx": inflight_ctx,
            "inflight_prompts": inflight_prompts,
            "after": after,
            "ticks": ticks,
            "final": final,
            "manual_at": manual_at,
        }

    def _assert_no_phantom(self, result, *, scheduled_counted):
        from datetime import datetime

        from cron import jobs as jobs_mod

        operator = result["operator"]
        pending_ctx = result["pending_ctx"]
        ticks = result["ticks"]
        final = result["final"]
        after = result["after"]
        manual_at = result["manual_at"]
        completed_after = (after.get("repeat") or {}).get("completed", 0)
        completed_final = (final.get("repeat") or {}).get("completed", 0)

        if scheduled_counted:
            assert completed_after == 1
        else:
            assert completed_after == 0

        assert ticks[0][0] >= 1
        assert ticks[0][1] == [operator]
        pending_events = (after.get("pending_event_batch") or {}).get("events") or []
        if pending_events:
            assert ticks[1][0] >= 1
            assert len(ticks[1][1]) == 1
            assert pending_ctx in str(ticks[1][1][0])
            assert operator not in str(ticks[1][1][0])
            assert ticks[2][0] == 0
            assert ticks[2][1] == []
            assert completed_final == completed_after
            assert not (final.get("pending_event_batch") or {}).get("events")
        else:
            assert ticks[1][0] == 0
            assert ticks[1][1] == []
            assert ticks[2][0] == 0

        bare = [tick for tick in ticks if tick[0] >= 1 and tick[1] == [None]]
        assert bare == []
        dispatch = final.get("last_dispatch") or {}
        assert dispatch.get("scheduled_at") != manual_at
        assert datetime.fromisoformat(final["next_run_at"]) > jobs_mod._hermes_now()

    def test_trigger_during_event_run_with_pending(self, temp_home, monkeypatch):
        result = self._trigger_during_inflight(
            monkeypatch, mode="event", with_pending=True
        )
        assert result["inflight_ctx"] in str(result["inflight_prompts"])
        assert result["operator"] not in "".join(str(p) for p in result["inflight_prompts"])
        self._assert_no_phantom(result, scheduled_counted=False)

    def test_trigger_during_event_run_without_pending(self, temp_home, monkeypatch):
        result = self._trigger_during_inflight(
            monkeypatch, mode="event", with_pending=False
        )
        assert result["inflight_ctx"] in str(result["inflight_prompts"])
        self._assert_no_phantom(result, scheduled_counted=False)

    def test_trigger_during_scheduled_run_with_pending(self, temp_home, monkeypatch):
        result = self._trigger_during_inflight(
            monkeypatch, mode="sched", with_pending=True
        )
        assert result["inflight_prompts"] == [None]
        self._assert_no_phantom(result, scheduled_counted=True)


class TestReviewR4InflightManual:
    """Review-r4 R4-A: consumed_manual is a flag, not the consumed stamp identity.

    A run-now issued while a stamp-consuming manual run is in flight must leave
    the newer stamp for the next tick. With pending events the second prompt
    must run, then the batch; without pending the second run-now is honoured
    rather than dropped. The leftover instant must not fire as a bare
    scheduled occurrence.
    """

    def _second_trigger_during_manual(self, monkeypatch, *, pending_when):
        import threading

        from cron.executions import list_executions
        from cron.jobs import (
            admit_job_event,
            create_job,
            get_job,
            mark_job_run,
            trigger_job,
        )
        import cron.scheduler as sched

        op_one = "op-ONE"
        op_two = "op-TWO"
        pending_ctx = "ctx-ONE"
        started = threading.Event()
        release = threading.Event()
        prompts = []
        thread_errors = []

        def fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            started.set()
            assert release.wait(15)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)

        job = create_job(prompt="x", schedule="every 5m", name="q", repeat=5)
        job_id = job["id"]

        if pending_when == "before":
            first = admit_job_event(job_id, delivery_id="d-run", context="running")
            owner = first["job"]["fire_claim"]["by"]
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"
            assert mark_job_run(job_id, True, expected_fire_owner=owner) is True
            armed = get_job(job_id)
            assert armed.get("event_rerun_due") is True

        trigger_job(job_id, extra_prompt=op_one)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        worker = threading.Thread(target=_run_tick)
        worker.start()
        assert started.wait(15), f"manual run never started; errors={thread_errors!r}"
        assert not thread_errors
        inflight_prompts = list(prompts)
        mid = get_job(job_id)
        assert mid.get("fire_claim")

        if pending_when == "mid":
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"

        second = trigger_job(job_id, extra_prompt=op_two)
        assert second is not None
        leftover_at = second.get("manual_run_at")
        assert leftover_at
        assert second.get("manual_run_prompt") == op_two

        release.set()
        worker.join(30)
        assert not worker.is_alive()
        assert not thread_errors
        after = get_job(job_id)
        assert after.get("fire_claim") in (None, {})
        ticks = []
        for _ in range(3):
            prompts.clear()
            n = sched.tick(verbose=False, sync=True)
            ticks.append((n, list(prompts)))
        final = get_job(job_id)
        ledger = list_executions(job_id=job_id)
        return {
            "job_id": job_id,
            "op_one": op_one,
            "op_two": op_two,
            "pending_ctx": pending_ctx,
            "pending_when": pending_when,
            "inflight_prompts": inflight_prompts,
            "after": after,
            "ticks": ticks,
            "final": final,
            "leftover_at": leftover_at,
            "ledger": ledger,
        }

    def _assert_second_prompt_honoured(self, result, *, with_pending):
        from datetime import datetime

        from cron import jobs as jobs_mod

        leftover_at = result["leftover_at"]
        after = result["after"]
        ticks = result["ticks"]
        final = result["final"]
        op_two = result["op_two"]
        pending_ctx = result["pending_ctx"]

        assert result["inflight_prompts"] == [result["op_one"]]
        assert after.get("manual_run_prompt") == op_two
        assert after.get("manual_run_at") == leftover_at
        assert after.get("next_run_at") == leftover_at

        assert ticks[0][0] >= 1
        assert ticks[0][1] == [op_two]
        if with_pending:
            assert (after.get("repeat") or {}).get("completed", 0) == 0
            assert ticks[1][0] >= 1
            assert len(ticks[1][1]) == 1
            assert pending_ctx in str(ticks[1][1][0])
            assert op_two not in str(ticks[1][1][0])
            assert ticks[2][0] == 0
            assert ticks[2][1] == []
            assert (final.get("repeat") or {}).get("completed", 0) == 0
            assert not (final.get("pending_event_batch") or {}).get("events")
        else:
            assert ticks[1][0] == 0
            assert ticks[1][1] == []
            assert ticks[2][0] == 0
            assert ticks[2][1] == []
            assert (final.get("repeat") or {}).get("completed", 0) == 2

        bare = [tick for tick in ticks if tick[0] >= 1 and tick[1] == [None]]
        assert bare == [], f"leftover instant fired as a bare occurrence: {ticks!r}"
        dispatch = final.get("last_dispatch") or {}
        assert dispatch.get("scheduled_at") != leftover_at
        assert all(row.get("scheduled_instant") != leftover_at for row in result["ledger"])
        assert datetime.fromisoformat(final["next_run_at"]) > jobs_mod._hermes_now()

    @pytest.mark.parametrize("pending_when", ["before", "mid"])
    def test_second_run_now_during_manual_with_pending(
        self, temp_home, monkeypatch, pending_when
    ):
        result = self._second_trigger_during_manual(
            monkeypatch, pending_when=pending_when
        )
        self._assert_second_prompt_honoured(result, with_pending=True)

    def test_second_run_now_during_manual_without_pending(self, temp_home, monkeypatch):
        result = self._second_trigger_during_manual(monkeypatch, pending_when=None)
        self._assert_second_prompt_honoured(result, with_pending=False)


class TestReviewR4OneshotLeftover:
    """Review-r4 R4-B: leftover restore must not resurrect a finite one-shot
    this completion just retired. The record stays inspectable; a 202-accepted
    pending batch still drains via pending-driven re-enable.
    """

    def _trigger_during_inflight_oneshot(self, monkeypatch, *, with_pending):
        import threading

        from cron import jobs as jobs_mod
        from cron.jobs import (
            admit_job_event,
            create_job,
            get_job,
            load_jobs,
            save_jobs,
            trigger_job,
        )
        import cron.scheduler as sched

        operator = "op-during-run"
        pending_ctx = "ctx-ONE"
        started = threading.Event()
        release = threading.Event()
        prompts = []
        thread_errors = []

        def fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            started.set()
            assert release.wait(15)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)

        job = create_job(prompt="x", schedule="in 30m", name="once")
        job_id = job["id"]
        records = load_jobs()
        for record in records:
            if record["id"] == job_id:
                record["next_run_at"] = jobs_mod._hermes_now().isoformat()
        save_jobs(records)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        worker = threading.Thread(target=_run_tick)
        worker.start()
        assert started.wait(15), f"oneshot run never started; errors={thread_errors!r}"
        assert not thread_errors

        if with_pending:
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"

        triggered = trigger_job(job_id, extra_prompt=operator)
        assert triggered is not None
        assert triggered.get("manual_run_at")
        assert triggered.get("fire_claim")

        release.set()
        worker.join(30)
        assert not worker.is_alive()
        assert not thread_errors
        after = get_job(job_id)
        ticks = []
        for _ in range(3):
            prompts.clear()
            n = sched.tick(verbose=False, sync=True)
            ticks.append((n, list(prompts)))
        final = get_job(job_id)
        return {
            "job_id": job_id,
            "operator": operator,
            "pending_ctx": pending_ctx,
            "after": after,
            "ticks": ticks,
            "final": final,
        }

    def test_trigger_during_inflight_oneshot_without_pending(
        self, temp_home, monkeypatch, caplog
    ):
        import logging

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            result = self._trigger_during_inflight_oneshot(
                monkeypatch, with_pending=False
            )
        final = result["final"]
        assert final is not None, "retired one-shot record was deleted"
        assert final.get("id") == result["job_id"]
        assert final.get("state") == "completed"
        assert final.get("enabled") is False
        assert all(tick[0] == 0 and tick[1] == [] for tick in result["ticks"])
        misleading = [
            rec
            for rec in caplog.records
            if "re-armed without a budget reset" in rec.getMessage()
            or "WITHOUT firing" in rec.getMessage()
        ]
        assert misleading == []

    def test_trigger_during_inflight_oneshot_with_pending(
        self, temp_home, monkeypatch, caplog
    ):
        import logging

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            result = self._trigger_during_inflight_oneshot(
                monkeypatch, with_pending=True
            )
        ticks = result["ticks"]
        final = result["final"]
        pending_ctx = result["pending_ctx"]
        assert final is not None, "retired one-shot record was deleted"
        assert final.get("id") == result["job_id"]
        assert ticks[0][0] >= 1
        assert len(ticks[0][1]) == 1
        assert pending_ctx in str(ticks[0][1][0])
        assert result["operator"] not in str(ticks[0][1][0])
        assert ticks[1][0] == 0
        assert ticks[2][0] == 0
        assert not (final.get("pending_event_batch") or {}).get("events")
        assert final.get("state") == "completed"
        assert final.get("enabled") is False
        misleading = [
            rec
            for rec in caplog.records
            if "re-armed without a budget reset" in rec.getMessage()
            or "WITHOUT firing" in rec.getMessage()
        ]
        assert misleading == []


def _misleading_oneshot_removal(caplog):
    return [
        rec
        for rec in caplog.records
        if "re-armed without a budget reset" in rec.getMessage()
        or "WITHOUT firing" in rec.getMessage()
    ]


class TestReviewR5OneshotBudget:
    """Review-r5: a budget-exhausted one-shot must not carry a live manual
    stamp into the next tick. ``manual_due`` would skip the event drain and
    ``_oneshot_dispatch_limit_reached`` would delete the record and the
    202-accepted batch with a misleading re-arm warning.
    """

    def _block_run_job(self, monkeypatch):
        import threading

        import cron.scheduler as sched

        started = threading.Event()
        release = threading.Event()
        prompts = []
        thread_errors = []

        def fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            started.set()
            assert release.wait(15)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)
        return started, release, prompts, thread_errors, sched

    def _drain_ticks(self, sched, prompts, n=3):
        ticks = []
        for _ in range(n):
            prompts.clear()
            count = sched.tick(verbose=False, sync=True)
            ticks.append((count, list(prompts)))
        return ticks

    def test_second_run_now_during_manual_oneshot_with_pending(
        self, temp_home, monkeypatch, caplog
    ):
        """R5-A: second run-now during an in-flight *manual* one-shot plus a
        202-queued event. Predecessor dropped the leftover prompt but kept
        the record and drained the batch; leftover restore at this head
        deletes both.
        """
        import logging
        import threading

        from cron.jobs import admit_job_event, create_job, get_job, trigger_job

        started, release, prompts, thread_errors, sched = self._block_run_job(
            monkeypatch
        )
        op_one = "op-ONE"
        op_two = "op-TWO"
        pending_ctx = "ctx-ONE"

        job = create_job(prompt="x", schedule="in 30m", name="once")
        job_id = job["id"]
        trigger_job(job_id, extra_prompt=op_one)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            worker = threading.Thread(target=_run_tick)
            worker.start()
            assert started.wait(15), (
                f"manual oneshot never started; errors={thread_errors!r}"
            )
            assert not thread_errors
            inflight_prompts = list(prompts)
            inflight = get_job(job_id)
            assert inflight.get("fire_claim")
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"
            second = trigger_job(job_id, extra_prompt=op_two)
            assert second is not None
            leftover_at = second.get("manual_run_at")
            assert leftover_at
            assert second.get("manual_run_prompt") == op_two

            release.set()
            worker.join(30)
            assert not worker.is_alive()
            assert not thread_errors
            after = get_job(job_id)
            ticks = self._drain_ticks(sched, prompts)
            final = get_job(job_id)

        assert inflight_prompts == [op_one]
        assert after is not None, "one-shot record was deleted after in-flight completion"
        assert after.get("id") == job_id
        assert [event["context"] for event in _contexts(after)[0]] == [pending_ctx]
        # Occurrence-free completion left completed == times; restoring the
        # leftover stamp would make the next tick's manual_due hit the
        # dispatch-limit guard. Drop it (R4-B / predecessor parity).
        assert after.get("manual_run_prompt") in (None, False)
        assert after.get("manual_run_at") in (None, False)
        assert ticks[0][0] >= 1
        assert len(ticks[0][1]) == 1
        assert pending_ctx in str(ticks[0][1][0])
        assert op_two not in str(ticks[0][1][0])
        assert ticks[1][0] == 0
        assert ticks[1][1] == []
        assert ticks[2][0] == 0
        assert ticks[2][1] == []
        assert final is not None, "retired one-shot record was deleted"
        assert final.get("id") == job_id
        assert final.get("state") == "completed"
        assert final.get("enabled") is False
        assert not (final.get("pending_event_batch") or {}).get("events")
        assert _misleading_oneshot_removal(caplog) == []

    def test_run_now_on_pending_reenabled_retired_oneshot(
        self, temp_home, monkeypatch, caplog
    ):
        """R5-B: run-now on a one-shot that pending re-enable left
        ``enabled=True, state=scheduled`` after retirement. ``trigger_job``
        must not stamp a live manual due that the next tick deletes; the
        accepted batch must still drain.
        """
        import logging
        import threading

        from cron import jobs as jobs_mod
        from cron.jobs import (
            admit_job_event,
            create_job,
            get_job,
            load_jobs,
            save_jobs,
            trigger_job,
        )

        started, release, prompts, thread_errors, sched = self._block_run_job(
            monkeypatch
        )
        pending_ctx = "ctx-ONE"
        operator = "op-late"

        job = create_job(prompt="x", schedule="in 30m", name="once")
        job_id = job["id"]
        records = load_jobs()
        for record in records:
            if record["id"] == job_id:
                record["next_run_at"] = jobs_mod._hermes_now().isoformat()
        save_jobs(records)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            worker = threading.Thread(target=_run_tick)
            worker.start()
            assert started.wait(15), (
                f"oneshot run never started; errors={thread_errors!r}"
            )
            assert not thread_errors
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"
            release.set()
            worker.join(30)
            assert not worker.is_alive()
            assert not thread_errors
            after = get_job(job_id)
            assert after is not None, "one-shot record was deleted after completion"
            assert after.get("state") == "scheduled"
            assert after.get("enabled") is True
            assert [event["context"] for event in _contexts(after)[0]] == [pending_ctx]
            assert (after.get("repeat") or {}).get("completed") == (
                after.get("repeat") or {}
            ).get("times")

            with pytest.raises(ValueError, match="terminal"):
                trigger_job(job_id, extra_prompt=operator)
            refused = get_job(job_id)
            assert refused is not None
            assert refused.get("manual_run_at") in (None, False)
            assert refused.get("manual_run_prompt") in (None, False)
            assert [event["context"] for event in _contexts(refused)[0]] == [
                pending_ctx
            ]

            ticks = self._drain_ticks(sched, prompts)
            final = get_job(job_id)

        assert ticks[0][0] >= 1
        assert len(ticks[0][1]) == 1
        assert pending_ctx in str(ticks[0][1][0])
        assert operator not in str(ticks[0][1][0])
        assert ticks[1][0] == 0
        assert ticks[2][0] == 0
        assert final is not None, "retired one-shot record was deleted"
        assert final.get("id") == job_id
        assert final.get("state") == "completed"
        assert final.get("enabled") is False
        assert not (final.get("pending_event_batch") or {}).get("events")
        assert _misleading_oneshot_removal(caplog) == []


class TestReviewR6OneshotBudget:
    """Review-r6: a budget-exhausted one-shot kept non-terminal so a 202
    batch can drain must not be retired by ``cronjob(action='run')`` /
    ``claim_job_for_fire(manual=True)``, and occurrence-free completion
    must not leave the consumed stamp as a past ``next_run_at``.
    """

    def _block_run_job(self, monkeypatch):
        import threading

        import cron.scheduler as sched

        started = threading.Event()
        release = threading.Event()
        prompts = []
        thread_errors = []

        def fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            started.set()
            assert release.wait(15)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)
        return started, release, prompts, thread_errors, sched

    def _drain_ticks(self, sched, prompts, n=3):
        ticks = []
        for _ in range(n):
            prompts.clear()
            count = sched.tick(verbose=False, sync=True)
            ticks.append((count, list(prompts)))
        return ticks

    def _assert_direct_run_refused_and_batch_drains(
        self, job_id, pending_ctx, operator, sched, prompts, caplog,
    ):
        from cron.jobs import get_job
        from tools.cronjob_tools import _execute_job_now

        after = get_job(job_id)
        assert after is not None, "one-shot record was deleted after completion"
        assert after.get("state") == "scheduled"
        assert after.get("enabled") is True
        assert [event["context"] for event in _contexts(after)[0]] == [pending_ctx]
        assert (after.get("repeat") or {}).get("completed") == (
            after.get("repeat") or {}
        ).get("times")
        # R6-B: consumed stamp is not an occurrence on an exhausted one-shot.
        assert after.get("next_run_at") in (None, False)

        result = _execute_job_now(after, extra_prompt=operator)
        refused = get_job(job_id)
        assert result.get("claimed") is False
        assert result.get("success") is False
        assert "terminal" in (result.get("error") or "")
        assert "resume" in (result.get("error") or "")
        assert refused is not None
        assert refused.get("state") == "scheduled"
        assert refused.get("enabled") is True
        assert not refused.get("fire_claim")
        assert refused.get("event_rerun_due") is True
        assert [event["context"] for event in _contexts(refused)[0]] == [
            pending_ctx
        ]
        assert refused.get("manual_run_at") in (None, False)
        assert refused.get("manual_run_prompt") in (None, False)

        ticks = self._drain_ticks(sched, prompts)
        final = get_job(job_id)

        assert ticks[0][0] >= 1
        assert len(ticks[0][1]) == 1
        assert pending_ctx in str(ticks[0][1][0])
        assert operator not in str(ticks[0][1][0])
        assert ticks[1][0] == 0
        assert ticks[2][0] == 0
        assert final is not None, "retired one-shot record was deleted"
        assert final.get("id") == job_id
        assert final.get("state") == "completed"
        assert final.get("enabled") is False
        assert not (final.get("pending_event_batch") or {}).get("events")
        assert not final.get("fire_claim")
        assert _misleading_oneshot_removal(caplog) == []

    def test_direct_run_on_pending_reenabled_retired_oneshot(
        self, temp_home, monkeypatch, caplog
    ):
        """R6-A / R5-B shape: ``cronjob(action='run')`` after pending
        re-enable left a budget-exhausted one-shot scheduled. Predecessor
        ``claim_dispatch`` retain marked it completed and stranded the
        202 batch with a dangling fire_claim.
        """
        import logging
        import threading

        from cron import jobs as jobs_mod
        from cron.jobs import admit_job_event, create_job, load_jobs, save_jobs

        started, release, prompts, thread_errors, sched = self._block_run_job(
            monkeypatch
        )
        pending_ctx = "ctx-ONE"
        operator = "op-late"

        job = create_job(prompt="x", schedule="in 30m", name="once")
        job_id = job["id"]
        records = load_jobs()
        for record in records:
            if record["id"] == job_id:
                record["next_run_at"] = jobs_mod._hermes_now().isoformat()
        save_jobs(records)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            worker = threading.Thread(target=_run_tick)
            worker.start()
            assert started.wait(15), (
                f"oneshot run never started; errors={thread_errors!r}"
            )
            assert not thread_errors
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"
            release.set()
            worker.join(30)
            assert not worker.is_alive()
            assert not thread_errors
            self._assert_direct_run_refused_and_batch_drains(
                job_id, pending_ctx, operator, sched, prompts, caplog,
            )

    def test_direct_run_on_exhausted_manual_oneshot_after_leftover_drop(
        self, temp_home, monkeypatch, caplog
    ):
        """R6-A / R5-A post-completion shape: leftover run-now was dropped,
        record kept scheduled with the 202 batch. Direct run must refuse
        the same way and leave the drain intact.
        """
        import logging
        import threading

        from cron.jobs import admit_job_event, create_job, trigger_job

        started, release, prompts, thread_errors, sched = self._block_run_job(
            monkeypatch
        )
        op_one = "op-ONE"
        op_two = "op-TWO"
        pending_ctx = "ctx-ONE"
        operator = "op-late"

        job = create_job(prompt="x", schedule="in 30m", name="once")
        job_id = job["id"]
        trigger_job(job_id, extra_prompt=op_one)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            worker = threading.Thread(target=_run_tick)
            worker.start()
            assert started.wait(15), (
                f"manual oneshot never started; errors={thread_errors!r}"
            )
            assert not thread_errors
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"
            second = trigger_job(job_id, extra_prompt=op_two)
            assert second is not None
            release.set()
            worker.join(30)
            assert not worker.is_alive()
            assert not thread_errors
            self._assert_direct_run_refused_and_batch_drains(
                job_id, pending_ctx, operator, sched, prompts, caplog,
            )

    def test_foreign_tick_during_drain_does_not_delete_exhausted_oneshot(
        self, temp_home, monkeypatch, caplog
    ):
        """R6-B: occurrence-free completion of a manual one-shot with a
        queued event must not leave ``next_run_at`` at the consumed past
        stamp. A second scheduler process (running-set check patched
        False) must not hit ``_oneshot_dispatch_limit_reached`` and delete
        the record under the live drain run.
        """
        import logging
        import threading

        from cron import jobs as jobs_mod
        from cron.jobs import admit_job_event, create_job, get_job, trigger_job

        started, release, prompts, thread_errors, sched = self._block_run_job(
            monkeypatch
        )
        op_one = "op-ONE"
        pending_ctx = "ctx-ONE"

        job = create_job(prompt="x", schedule="in 30m", name="once")
        job_id = job["id"]
        trigger_job(job_id, extra_prompt=op_one)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            worker = threading.Thread(target=_run_tick)
            worker.start()
            assert started.wait(15), (
                f"manual oneshot never started; errors={thread_errors!r}"
            )
            assert not thread_errors
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"
            release.set()
            worker.join(30)
            assert not worker.is_alive()
            assert not thread_errors

            after = get_job(job_id)
            assert after is not None, "one-shot record was deleted after completion"
            assert after.get("state") == "scheduled"
            assert after.get("enabled") is True
            assert [event["context"] for event in _contexts(after)[0]] == [
                pending_ctx
            ]
            assert (after.get("repeat") or {}).get("completed") == (
                after.get("repeat") or {}
            ).get("times")
            # Consumed stamp is not an occurrence; a past next_run_at would
            # look due to a foreign tick during the drain run.
            assert after.get("next_run_at") in (None, False)
            assert after.get("manual_run_at") in (None, False)

            started.clear()
            release.clear()
            prompts.clear()
            drain_worker = threading.Thread(target=_run_tick)
            drain_worker.start()
            assert started.wait(15), (
                f"drain run never started; errors={thread_errors!r}"
            )
            assert not thread_errors
            inflight = get_job(job_id)
            assert inflight is not None
            assert inflight.get("fire_claim")
            assert not (inflight.get("pending_event_batch") or {}).get("events")

            monkeypatch.setattr(
                jobs_mod, "_job_running_in_this_process", lambda _jid: False
            )
            jobs_mod.get_due_jobs()
            still = get_job(job_id)
            assert still is not None, (
                "foreign tick deleted the record under the live drain run"
            )
            assert still.get("id") == job_id
            assert still.get("fire_claim")
            assert _misleading_oneshot_removal(caplog) == []

            release.set()
            drain_worker.join(30)
            assert not drain_worker.is_alive()
            assert not thread_errors

            ticks = self._drain_ticks(sched, prompts)
            final = get_job(job_id)

        assert ticks[0][0] == 0
        assert ticks[1][0] == 0
        assert final is not None, "retired one-shot record was deleted"
        assert final.get("id") == job_id
        assert final.get("state") == "completed"
        assert final.get("enabled") is False
        assert not (final.get("pending_event_batch") or {}).get("events")
        assert not final.get("fire_claim")
        assert _misleading_oneshot_removal(caplog) == []


class TestReviewR7OneshotBudget:
    """Review-r7: the r5/r6 exhausted-one-shot refusal must be atomic with
    the stamp write, and occurrence-free completion on a once job must not
    leave the consumed stamp as a live due regardless of repeat budget.
    """

    def _block_run_job(self, monkeypatch):
        import threading

        import cron.scheduler as sched

        started = threading.Event()
        release = threading.Event()
        prompts = []
        thread_errors = []

        def fake_run_job(_job, *, extra_prompt=None, **_kw):
            prompts.append(extra_prompt)
            started.set()
            assert release.wait(15)
            return True, "out", "final", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_k: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_k: None)
        return started, release, prompts, thread_errors, sched

    def _drain_ticks(self, sched, prompts, n=3):
        ticks = []
        for _ in range(n):
            prompts.clear()
            count = sched.tick(verbose=False, sync=True)
            ticks.append((count, list(prompts)))
        return ticks

    def test_trigger_job_stamp_after_inflight_completion_does_not_delete_batch(
        self, temp_home, monkeypatch, caplog
    ):
        """R7-A: ``trigger_job`` evaluates the exhausted-one-shot refusal
        under ``resolve_job_ref`` and writes the stamp under ``update_job``.
        Landing ``mark_job_run`` between those lock scopes stamps a
        pending-re-enabled exhausted one-shot; the next tick's
        ``manual_due`` hits ``_oneshot_dispatch_limit_reached`` and deletes
        the record plus the 202 batch.
        """
        import logging
        import threading

        from cron import jobs as jobs_mod
        from cron.jobs import (
            admit_job_event,
            create_job,
            get_job,
            load_jobs,
            save_jobs,
            trigger_job,
        )

        started, release, prompts, thread_errors, sched = self._block_run_job(
            monkeypatch
        )
        pending_ctx = "ctx-ONE"
        operator = "op-late"

        job = create_job(prompt="x", schedule="in 30m", name="once")
        job_id = job["id"]
        records = load_jobs()
        for record in records:
            if record["id"] == job_id:
                record["next_run_at"] = jobs_mod._hermes_now().isoformat()
        save_jobs(records)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        real_update_job = jobs_mod.update_job
        real_with_job = jobs_mod._with_job
        inject_gate = threading.Event()
        injected = threading.Event()
        trigger_ident = threading.get_ident()

        def _inject_completion():
            if (
                inject_gate.is_set()
                and not injected.is_set()
                and threading.get_ident() == trigger_ident
            ):
                injected.set()
                release.set()
                worker.join(30)
                assert not worker.is_alive()
                assert not thread_errors

        def _update_job_inject(*args, **kwargs):
            _inject_completion()
            return real_update_job(*args, **kwargs)

        def _with_job_inject(*args, **kwargs):
            _inject_completion()
            return real_with_job(*args, **kwargs)

        monkeypatch.setattr(jobs_mod, "update_job", _update_job_inject)
        monkeypatch.setattr(jobs_mod, "_with_job", _with_job_inject)

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            worker = threading.Thread(target=_run_tick)
            worker.start()
            assert started.wait(15), (
                f"oneshot run never started; errors={thread_errors!r}"
            )
            assert not thread_errors
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"
            inflight = get_job(job_id)
            assert inflight.get("fire_claim") or inflight.get("run_claim")

            inject_gate.set()
            try:
                trigger_job(job_id, extra_prompt=operator)
            except ValueError as exc:
                assert "terminal" in str(exc)
                assert "resume" in str(exc)
            finally:
                if not release.is_set():
                    release.set()
                if worker.is_alive():
                    worker.join(30)
                assert not worker.is_alive()
                assert not thread_errors

            after = get_job(job_id)
            assert after is not None, (
                "one-shot record was deleted after the raced completion"
            )
            assert after.get("id") == job_id
            assert after.get("state") == "scheduled"
            assert after.get("enabled") is True
            assert [event["context"] for event in _contexts(after)[0]] == [
                pending_ctx
            ]
            assert (after.get("repeat") or {}).get("completed") == (
                after.get("repeat") or {}
            ).get("times")
            # Check+write must refuse or drop the stamp once the record is
            # exhausted and re-enabled for drain. A live manual due here is
            # the R5-B deletion by another door.
            assert after.get("manual_run_at") in (None, False) or after.get(
                "manual_run_at"
            ) != after.get("next_run_at")
            assert after.get("manual_run_prompt") in (None, False)
            assert after.get("event_rerun_due") is True

            ticks = self._drain_ticks(sched, prompts)
            final = get_job(job_id)

        assert ticks[0][0] >= 1
        assert len(ticks[0][1]) == 1
        assert pending_ctx in str(ticks[0][1][0])
        assert operator not in str(ticks[0][1][0])
        assert ticks[1][0] == 0
        assert ticks[2][0] == 0
        assert final is not None, "retired one-shot record was deleted"
        assert final.get("id") == job_id
        assert final.get("state") == "completed"
        assert final.get("enabled") is False
        assert not (final.get("pending_event_batch") or {}).get("events")
        assert not final.get("fire_claim")
        assert _misleading_oneshot_removal(caplog) == []

    @pytest.mark.parametrize("repeat", [2, "forever"], ids=["repeat-2", "repeat-forever"])
    def test_occurrence_free_oneshot_non_unit_repeat_does_not_phantom_run(
        self, temp_home, monkeypatch, caplog, repeat
    ):
        """R7-B: occurrence-free completion of a once job with ``repeat=2``
        or ``repeat=forever`` must not keep the consumed past stamp as
        ``next_run_at``. That stamp is not a live scheduled due; the tick
        after drain would fire a phantom bare run.
        """
        import logging
        import threading

        from cron.jobs import admit_job_event, create_job, get_job, trigger_job, update_job

        started, release, prompts, thread_errors, sched = self._block_run_job(
            monkeypatch
        )
        op_one = "op-ONE"
        pending_ctx = "ctx-ONE"

        job = create_job(prompt="x", schedule="in 30m", name="once", repeat=2)
        job_id = job["id"]
        if repeat == "forever":
            updated = update_job(job_id, {"repeat": "forever"})
            assert (updated.get("repeat") or {}).get("times") is None
        else:
            assert (get_job(job_id).get("repeat") or {}).get("times") == 2
        trigger_job(job_id, extra_prompt=op_one)

        def _run_tick():
            try:
                sched.tick(verbose=False, sync=True)
            except Exception as exc:
                thread_errors.append(exc)
                started.set()

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            worker = threading.Thread(target=_run_tick)
            worker.start()
            assert started.wait(15), (
                f"manual oneshot never started; errors={thread_errors!r}"
            )
            assert not thread_errors
            queued = admit_job_event(job_id, delivery_id="d-p", context=pending_ctx)
            assert queued["status"] == "queued"
            release.set()
            worker.join(30)
            assert not worker.is_alive()
            assert not thread_errors

            after = get_job(job_id)
            assert after is not None, "one-shot record was deleted after completion"
            assert after.get("state") == "scheduled"
            assert after.get("enabled") is True
            assert [event["context"] for event in _contexts(after)[0]] == [
                pending_ctx
            ]
            # Consumed stamp is not an occurrence, even when the repeat
            # budget is not the default once-only times=1.
            assert after.get("next_run_at") in (None, False)
            assert after.get("manual_run_at") in (None, False)

            ticks = self._drain_ticks(sched, prompts, n=4)
            final = get_job(job_id)

        assert ticks[0][0] >= 1
        assert len(ticks[0][1]) == 1
        assert pending_ctx in str(ticks[0][1][0])
        assert op_one not in str(ticks[0][1][0])
        bare = [tick for tick in ticks if tick[0] >= 1 and tick[1] == [None]]
        assert bare == [], f"phantom bare run after drain: {ticks!r}"
        assert ticks[1][0] == 0
        assert ticks[2][0] == 0
        assert ticks[3][0] == 0
        assert final is not None, "retired one-shot record was deleted"
        assert final.get("id") == job_id
        assert final.get("state") == "completed"
        assert final.get("enabled") is False
        assert not (final.get("pending_event_batch") or {}).get("events")
        assert not final.get("fire_claim")
        assert _misleading_oneshot_removal(caplog) == []
