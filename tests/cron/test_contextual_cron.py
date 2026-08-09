from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import json
from typing import Any, cast

import pytest


def _point_store(monkeypatch, tmp_path):
    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    return jobs


def _bind_session(
    key: str,
    *,
    session_id: str = "sid",
    routing_revision: int = 0,
    route_instance_id: str = "route-instance-a",
    route_principal=None,
):
    from gateway.session_context import set_session_vars

    return set_session_vars(
        platform="telegram",
        chat_id="42",
        chat_type="dm",
        user_id="42",
        session_key=key,
        session_id=session_id,
        route_instance_id=route_instance_id,
        route_principal=route_principal,
        routing_revision=routing_revision,
    )


def _physical_binding(session_key: str = "stable") -> dict[str, Any]:
    return {
        "context_binding": {
            "session_key": session_key,
            "session_id": "session-1",
            "routing_revision": 0,
        }
    }


def test_public_job_record_redacts_contextual_binding_and_execution_internals():
    from cron.jobs import public_job_record

    secret_values = {
        "session_key": "telegram:dm:42:42",
        "admitted_session_id": "session-private",
        "routing_revision": 19,
        "agent_result_json": '{"message":"private model result"}',
        "delivery_claim_owner": "scheduler-private",
    }
    public = public_job_record(
        {
            "id": "job-public",
            "name": "safe name",
            "prompt": "safe prompt",
            "schedule": {"kind": "interval", "minutes": 5},
            "session_target": "current",
            "session_key": secret_values["session_key"],
            "context_binding": {
                "session_key": secret_values["session_key"],
                "chat_id": "42",
                "user_id": "42",
            },
            "origin": {"platform": "telegram", "chat_id": "42"},
            "provider_snapshot": "legacy-provider",
            "model_snapshot": "legacy-model",
            "run_claim": {"execution_id": "exec-private"},
            "_pending_accounting_execution_ids": ["exec-private"],
            "latest_execution": {
                "id": "exec-public",
                "status": "completed",
                "outcome": "notify",
                "phase": "terminal",
                "started_at": "2026-07-31T00:00:00Z",
                **secret_values,
            },
        }
    )

    assert public["session_target"] == "current"
    assert public["latest_execution"] == {
        "id": "exec-public",
        "status": "completed",
        "outcome": "notify",
        "phase": "terminal",
        "started_at": "2026-07-31T00:00:00Z",
    }
    assert "origin" not in public
    assert public["provider_snapshot"] == "legacy-provider"
    assert public["model_snapshot"] == "legacy-model"
    encoded = json.dumps(public, sort_keys=True)
    for field in (
        "session_key",
        "context_binding",
        "admitted_session_id",
        "routing_revision",
        "agent_result_json",
        "delivery_claim_owner",
    ):
        assert field not in encoded
    for value in (
        "telegram:dm:42:42",
        "session-private",
        "private model result",
        "exec-private",
    ):
        assert value not in encoded


def test_public_job_record_drops_malformed_contextual_execution_payload():
    from cron.jobs import public_job_record

    public = public_job_record(
        {
            "id": "bad-current",
            "session_target": "current",
            "latest_execution": "ROUTE-SECRET-CANARY",
        }
    )

    assert public["id"] == "bad-current"
    assert "latest_execution" not in public


def test_contextual_authority_marker_and_degraded_replace_are_linearized(
    monkeypatch, tmp_path
):
    import contextlib
    import threading
    import time

    import cron.jobs as jobs

    jobs = _point_store(monkeypatch, tmp_path)
    ordinary = {"id": "ordinary", "session_target": "isolated"}
    contextual = {"id": "contextual", "session_target": "current"}
    jobs.save_jobs([ordinary])

    original_replace = jobs.atomic_replace
    writer_at_replace = threading.Event()
    release_writer = threading.Event()
    errors = []

    def blocked_replace(source, destination):
        if threading.current_thread().name == "degraded-writer":
            writer_at_replace.set()
            assert release_writer.wait(timeout=5)
        return original_replace(source, destination)

    @contextlib.contextmanager
    def fake_jobs_lock():
        previous = getattr(jobs._jobs_lock_state, "cross_process_acquired", False)
        acquired = threading.current_thread().name != "degraded-writer"
        jobs._jobs_lock_state.cross_process_acquired = acquired
        try:
            yield acquired
        finally:
            jobs._jobs_lock_state.cross_process_acquired = previous

    monkeypatch.setattr(jobs, "atomic_replace", blocked_replace)
    monkeypatch.setattr(jobs, "_jobs_lock", fake_jobs_lock)

    def degraded_write():
        try:
            jobs.save_jobs([ordinary])
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def first_contextual_write():
        try:
            jobs.save_jobs([ordinary, contextual])
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    writer = threading.Thread(target=degraded_write, name="degraded-writer")
    creator = threading.Thread(target=first_contextual_write, name="creator")
    writer.start()
    assert writer_at_replace.wait(timeout=5)
    creator.start()
    time.sleep(0.05)
    assert creator.is_alive(), "marker publication must wait for the staged replace"
    release_writer.set()
    writer.join(timeout=5)
    creator.join(timeout=5)

    assert not writer.is_alive()
    assert not creator.is_alive()
    assert errors == []
    assert {job["id"] for job in jobs.load_jobs()} == {"ordinary", "contextual"}
    assert jobs._contextual_jobs_authority_marker().exists()
    assert not jobs._contextual_jobs_authority_pending_marker().exists()


def test_missing_contextual_update_does_not_publish_authority(monkeypatch, tmp_path):
    import contextlib

    import cron.jobs as jobs

    jobs = _point_store(monkeypatch, tmp_path)
    ordinary = {"id": "ordinary", "session_target": "isolated", "name": "before"}
    jobs.save_jobs([ordinary])

    assert jobs.update_job("missing", {"session_target": "current"}) is None
    assert not jobs._contextual_jobs_authority_marker().exists()

    @contextlib.contextmanager
    def degraded_jobs_lock():
        previous = getattr(jobs._jobs_lock_state, "cross_process_acquired", False)
        jobs._jobs_lock_state.cross_process_acquired = False
        try:
            yield False
        finally:
            jobs._jobs_lock_state.cross_process_acquired = previous

    monkeypatch.setattr(jobs, "_jobs_lock", degraded_jobs_lock)
    jobs.save_jobs([{**ordinary, "name": "after"}])
    assert jobs.load_jobs()[0]["name"] == "after"


def test_rejected_contextual_mutations_do_not_publish_authority(monkeypatch, tmp_path):
    import cron.jobs as jobs

    jobs = _point_store(monkeypatch, tmp_path)
    _bind_session("telegram:dm:42:42")

    with pytest.raises(ValueError):
        jobs.create_job(
            prompt="never",
            schedule="2000-01-01T00:00:00+00:00",
            name="rejected-create",
            session_target="current",
        )
    assert not jobs._contextual_jobs_authority_marker().exists()
    assert jobs.load_jobs() == []

    ordinary = jobs.create_job(
        prompt="ordinary",
        schedule="0 9 * * *",
        name="ordinary",
    )
    with pytest.raises(ValueError):
        jobs.update_job(
            ordinary["id"],
            {
                "session_target": "current",
                "schedule": "2000-01-01T00:00:00+00:00",
            },
        )
    assert not jobs._contextual_jobs_authority_marker().exists()
    stored = jobs.get_job(ordinary["id"])
    assert stored is not None
    assert stored.get("session_target", "isolated") == "isolated"


def test_failed_authority_publication_rolls_back_marker(monkeypatch, tmp_path):
    import contextlib

    import cron.jobs as jobs

    jobs = _point_store(monkeypatch, tmp_path)
    ordinary = {"id": "ordinary", "session_target": "isolated", "name": "before"}
    contextual = {"id": "contextual", "session_target": "current"}
    jobs.save_jobs([ordinary])

    def partially_publish_then_fail():
        marker = jobs._contextual_jobs_authority_marker()
        marker.touch()
        raise OSError("injected marker fsync failure")

    monkeypatch.setattr(
        jobs,
        "_publish_contextual_jobs_authority_unlocked",
        partially_publish_then_fail,
    )
    with pytest.raises(OSError, match="marker fsync failure"):
        jobs.save_jobs([ordinary, contextual])

    assert not jobs._contextual_jobs_authority_marker().exists()
    assert [job["id"] for job in jobs.load_jobs()] == ["ordinary"]

    @contextlib.contextmanager
    def degraded_jobs_lock():
        previous = getattr(jobs._jobs_lock_state, "cross_process_acquired", False)
        jobs._jobs_lock_state.cross_process_acquired = False
        try:
            yield False
        finally:
            jobs._jobs_lock_state.cross_process_acquired = previous

    monkeypatch.setattr(jobs, "_jobs_lock", degraded_jobs_lock)
    jobs.save_jobs([{**ordinary, "name": "after"}])
    assert jobs.load_jobs()[0]["name"] == "after"


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="requires POSIX fork")
def test_crash_before_first_contextual_commit_does_not_publish_false_authority(
    monkeypatch, tmp_path
):
    import contextlib
    import os

    import cron.jobs as jobs

    jobs = _point_store(monkeypatch, tmp_path)
    ordinary = {"id": "ordinary", "session_target": "isolated", "name": "before"}
    contextual = {"id": "contextual", "session_target": "current"}
    jobs.save_jobs([ordinary])

    original_replace = jobs.atomic_replace

    def crash_before_replace(_source, _destination):
        os._exit(73)

    monkeypatch.setattr(jobs, "atomic_replace", crash_before_replace)
    child = os.fork()  # windows-footgun: ok - test is skip-gated on os.fork
    if child == 0:  # pragma: no cover - child exits without returning to pytest
        jobs.save_jobs([ordinary, contextual])
        os._exit(74)

    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 73
    monkeypatch.setattr(jobs, "atomic_replace", original_replace)

    # The durable jobs payload never crossed the replacement boundary, so a
    # restart must not treat contextual authority as successfully committed.
    assert [job["id"] for job in jobs.load_jobs()] == ["ordinary"]
    assert jobs._contextual_jobs_authority_pending_marker().exists()

    @contextlib.contextmanager
    def degraded_jobs_lock():
        previous = getattr(jobs._jobs_lock_state, "cross_process_acquired", False)
        jobs._jobs_lock_state.cross_process_acquired = False
        try:
            yield False
        finally:
            jobs._jobs_lock_state.cross_process_acquired = previous

    monkeypatch.setattr(jobs, "_jobs_lock", degraded_jobs_lock)
    jobs.save_jobs([{**ordinary, "name": "after"}])

    assert jobs.load_jobs()[0]["name"] == "after"
    assert not jobs._contextual_jobs_authority_marker().exists()
    assert not jobs._contextual_jobs_authority_pending_marker().exists()


def test_successful_contextual_commit_keeps_permanent_authority(
    monkeypatch, tmp_path
):
    import contextlib

    import cron.jobs as jobs

    jobs = _point_store(monkeypatch, tmp_path)
    ordinary = {"id": "ordinary", "session_target": "isolated", "name": "before"}
    contextual = {"id": "contextual", "session_target": "current"}

    jobs.save_jobs([ordinary, contextual])
    assert jobs._contextual_jobs_authority_marker().exists()
    assert not jobs._contextual_jobs_authority_pending_marker().exists()

    # Removing the last contextual job does not revoke the profile's authority.
    jobs.save_jobs([ordinary])

    @contextlib.contextmanager
    def degraded_jobs_lock():
        previous = getattr(jobs._jobs_lock_state, "cross_process_acquired", False)
        jobs._jobs_lock_state.cross_process_acquired = False
        try:
            yield False
        finally:
            jobs._jobs_lock_state.cross_process_acquired = previous

    monkeypatch.setattr(jobs, "_jobs_lock", degraded_jobs_lock)
    with pytest.raises(RuntimeError, match="degraded jobs.json write"):
        jobs.save_jobs([{**ordinary, "name": "must-not-commit"}])

    assert jobs.load_jobs()[0]["name"] == "before"
    assert jobs._contextual_jobs_authority_marker().exists()
    assert not jobs._contextual_jobs_authority_pending_marker().exists()


def test_public_job_record_preserves_legacy_isolated_api_shape():
    from cron.jobs import public_job_record

    legacy = {
        "id": "legacy",
        "origin": {"platform": "telegram", "chat_id": "42"},
        "provider_snapshot": "provider",
        "model_snapshot": "model",
        "run_claim": {"at": "then", "by": "machine"},
        "fire_claim": {"at": "then"},
        "latest_execution": {
            "id": "legacy-exec",
            "job_id": "legacy",
            "source": "cron",
            "claimed_at": "then",
            "error": "legacy error",
            "owner_pid": 123,
            "custom_future_execution": {"kept": True},
        },
        "custom_future_field": {"kept": True},
    }
    public = public_job_record(legacy)

    assert public == legacy


def test_legacy_job_defaults_to_isolated_without_session_key(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    jobs.save_jobs(
        [
            {
                "id": "legacy",
                "prompt": "standalone",
                "schedule": {"kind": "interval", "minutes": 5},
                "enabled": True,
            }
        ]
    )

    loaded = jobs.get_job("legacy")

    assert loaded["session_target"] == "isolated"
    assert "session_key" not in loaded


def test_degraded_jobs_lock_makes_mixed_contextual_store_read_only(
    monkeypatch, tmp_path
):
    jobs = _point_store(monkeypatch, tmp_path)
    due_at = (jobs._hermes_now() - timedelta(minutes=1)).isoformat()
    jobs.save_jobs(
        [
            {
                "id": "ordinary-due",
                "prompt": "ordinary",
                "schedule": {"kind": "interval", "minutes": 5},
                "next_run_at": due_at,
                "enabled": True,
            },
            {
                "id": "contextual-due",
                "prompt": "contextual",
                "schedule": {"kind": "once", "run_at": due_at},
                "next_run_at": due_at,
                "enabled": True,
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
                **_physical_binding("telegram:dm:42:42"),
            },
        ]
    )
    before = jobs.JOBS_FILE.read_bytes()

    @contextmanager
    def degraded_lock():
        yield False

    monkeypatch.setattr(jobs, "_jobs_lock", degraded_lock)

    assert jobs.get_due_jobs() == []
    assert jobs.JOBS_FILE.read_bytes() == before


def test_degraded_jobs_lock_preserves_ordinary_only_due_scan(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    due_at = (jobs._hermes_now() - timedelta(seconds=1)).isoformat()
    jobs.save_jobs(
        [
            {
                "id": "ordinary-due",
                "prompt": "ordinary",
                "schedule": {"kind": "interval", "minutes": 5},
                "next_run_at": due_at,
                "enabled": True,
            }
        ]
    )

    @contextmanager
    def degraded_lock():
        yield False

    monkeypatch.setattr(jobs, "_jobs_lock", degraded_lock)

    assert [job["id"] for job in jobs.get_due_jobs()] == ["ordinary-due"]


def test_degraded_jobs_lock_rejects_ordinary_writer_for_mixed_contextual_store(
    monkeypatch, tmp_path
):
    jobs = _point_store(monkeypatch, tmp_path)
    due_at = (jobs._hermes_now() - timedelta(minutes=1)).isoformat()
    jobs.save_jobs(
        [
            {
                "id": "ordinary",
                "prompt": "ordinary",
                "schedule": {"kind": "interval", "minutes": 5},
                "next_run_at": due_at,
                "enabled": True,
                "session_target": "isolated",
            },
            {
                "id": "contextual",
                "prompt": "contextual",
                "schedule": {"kind": "interval", "minutes": 5},
                "next_run_at": due_at,
                "enabled": True,
                "session_target": "current",
                "_contextual_binding_version": 1,
                **_physical_binding(),
            },
        ]
    )
    jobs_file = jobs._current_cron_store().jobs_file
    before = jobs_file.read_bytes()

    @contextmanager
    def degraded_lock():
        yield False

    monkeypatch.setattr(jobs, "_jobs_lock", degraded_lock)
    with pytest.raises(RuntimeError, match="contextual jobs"):
        jobs.update_job("ordinary", {"enabled": False})

    assert jobs_file.read_bytes() == before


def test_degraded_jobs_lock_preserves_ordinary_only_writer(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    jobs.save_jobs(
        [
            {
                "id": "ordinary",
                "prompt": "ordinary",
                "schedule": {"kind": "interval", "minutes": 5},
                "enabled": True,
            }
        ]
    )

    @contextmanager
    def degraded_lock():
        yield False

    monkeypatch.setattr(jobs, "_jobs_lock", degraded_lock)
    updated = jobs.update_job("ordinary", {"enabled": False})

    assert updated is not None
    assert jobs.get_job("ordinary")["enabled"] is False


def test_contextual_oneshot_due_scan_does_not_claim_before_ledger(
    monkeypatch, tmp_path
):
    jobs = _point_store(monkeypatch, tmp_path)
    due_at = (jobs._hermes_now() - timedelta(seconds=1)).isoformat()
    jobs.save_jobs(
        [
            {
                "id": "contextual-once",
                "prompt": "continue",
                "schedule": {"kind": "once", "run_at": due_at},
                "next_run_at": due_at,
                "enabled": True,
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
                **_physical_binding("telegram:dm:42:42"),
            }
        ]
    )

    assert [job["id"] for job in jobs.get_due_jobs()] == ["contextual-once"]
    assert jobs.get_job("contextual-once").get("run_claim") is None


def test_contextual_occurrence_claim_is_exact_and_execution_specific(
    monkeypatch, tmp_path
):
    jobs = _point_store(monkeypatch, tmp_path)
    due_at = (jobs._hermes_now() - timedelta(seconds=1)).isoformat()
    jobs.save_jobs(
        [
            {
                "id": "contextual-once",
                "prompt": "continue",
                "schedule": {"kind": "once", "run_at": due_at},
                "next_run_at": due_at,
                "enabled": True,
                "session_target": "current",
            }
        ]
    )

    assert jobs.claim_contextual_occurrence(
        "contextual-once",
        execution_id="execution-1",
        expected_next_run_at=due_at,
    )
    assert jobs.verify_contextual_occurrence_claim(
        "contextual-once", execution_id="execution-1"
    )
    assert not jobs.claim_contextual_occurrence(
        "contextual-once",
        execution_id="execution-2",
        expected_next_run_at=due_at,
    )
    stored = jobs.get_job("contextual-once")
    assert stored is not None
    assert stored["run_claim"]["execution_id"] == "execution-1"


def test_contextual_recurring_occurrence_cas_advances_only_exact_cursor(
    monkeypatch, tmp_path
):
    jobs = _point_store(monkeypatch, tmp_path)
    due_at = (jobs._hermes_now() - timedelta(seconds=1)).isoformat()
    jobs.save_jobs(
        [
            {
                "id": "contextual-recurring",
                "prompt": "continue",
                "schedule": {"kind": "interval", "minutes": 5},
                "next_run_at": due_at,
                "enabled": True,
                "session_target": "current",
            }
        ]
    )

    assert not jobs.claim_contextual_occurrence(
        "contextual-recurring",
        execution_id="stale-execution",
        expected_next_run_at="2020-01-01T00:00:00+00:00",
    )
    assert jobs.claim_contextual_occurrence(
        "contextual-recurring",
        execution_id="execution-1",
        expected_next_run_at=due_at,
    )
    stored = jobs.get_job("contextual-recurring")
    assert stored["next_run_at"] != due_at
    assert stored["_contextual_occurrence_claim"]["execution_id"] == "execution-1"
    assert jobs.verify_contextual_occurrence_claim(
        "contextual-recurring", execution_id="execution-1"
    )


@pytest.mark.parametrize("kind", ["once", "interval"])
def test_tick_ledger_failure_never_claims_contextual_occurrence(
    monkeypatch, tmp_path, kind
):
    import cron.scheduler as scheduler

    scheduler._running_job_ids.clear()
    due_at = "2026-08-08T00:00:00+00:00"
    schedule = (
        {"kind": "once", "run_at": due_at}
        if kind == "once"
        else {"kind": "interval", "minutes": 5}
    )
    job = {
        "id": f"contextual-{kind}",
        "name": f"contextual-{kind}",
        "prompt": "continue",
        "schedule": schedule,
        "next_run_at": due_at,
        "enabled": True,
        "session_target": "current",
        "deliver": "local",
    }
    mutations = []
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [job])
    monkeypatch.setattr(scheduler, "advance_next_runs", lambda _ids: 0)
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_occurrence",
        lambda *_a, **_k: mutations.append("claim") or True,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("ledger unavailable")),
    )
    monkeypatch.setattr(
        scheduler,
        "_get_lock_paths",
        lambda: (tmp_path / "cron", tmp_path / "cron" / ".tick.lock"),
    )

    assert scheduler.tick(verbose=False) == 0
    assert mutations == []


def test_tick_executor_rejection_terminalizes_and_accounts_claimed_contextual_occurrence(
    monkeypatch, tmp_path
):
    import cron.scheduler as scheduler

    scheduler._running_job_ids.clear()
    events = []
    job = {
        "id": "contextual-recurring",
        "name": "contextual-recurring",
        "prompt": "continue",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-08-08T00:00:00+00:00",
        "enabled": True,
        "session_target": "current",
        "deliver": "local",
    }

    class RejectingPool:
        def submit(self, _callable):
            events.append("submit")
            raise RuntimeError("executor rejected")

    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [job])
    monkeypatch.setattr(scheduler, "advance_next_runs", lambda _ids: 0)
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda *_a, **_k: events.append("ledger") or {"id": "execution-1"},
    )
    monkeypatch.setattr(
        scheduler,
        "seal_contextual_delivery_target",
        lambda *_a, **_k: events.append("seal-target") or True,
    )
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_occurrence",
        lambda *_a, **_k: events.append("claim-occurrence") or True,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_execution",
        lambda *_a, **_k: events.append("finish-contextual") or {},
    )
    monkeypatch.setattr(
        scheduler,
        "_account_contextual_job_run",
        lambda **_k: events.append("account-job") or True,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda *_a, **_k: events.append("finish-ordinary") or {},
    )
    monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda _workers: RejectingPool())
    monkeypatch.setattr(
        scheduler,
        "_get_lock_paths",
        lambda: (tmp_path / "cron", tmp_path / "cron" / ".tick.lock"),
    )

    assert scheduler.tick(verbose=False) == 0
    assert events == [
        "ledger",
        "seal-target",
        "claim-occurrence",
        "submit",
        "finish-contextual",
        "account-job",
    ]


@pytest.mark.parametrize("interrupt_at", ["claim", "submit"])
def test_tick_interruption_after_ledger_is_terminal_and_releases_guard(
    monkeypatch, tmp_path, interrupt_at
):
    import cron.scheduler as scheduler

    scheduler._running_job_ids.clear()
    events = []
    job = {
        "id": "contextual-interrupted",
        "name": "contextual-interrupted",
        "prompt": "continue",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-08-08T00:00:00+00:00",
        "enabled": True,
        "session_target": "current",
        "deliver": "local",
    }

    class InterruptingPool:
        def submit(self, _callable):
            events.append("submit")
            raise KeyboardInterrupt("submit interrupted")

    def claim_occurrence(*_args, **_kwargs):
        events.append("claim-occurrence")
        if interrupt_at == "claim":
            raise KeyboardInterrupt("claim interrupted")
        return True

    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [job])
    monkeypatch.setattr(scheduler, "advance_next_runs", lambda _ids: 0)
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda *_a, **_k: events.append("ledger") or {"id": "execution-1"},
    )
    monkeypatch.setattr(
        scheduler,
        "seal_contextual_delivery_target",
        lambda *_a, **_k: events.append("seal-target") or True,
    )
    monkeypatch.setattr(scheduler, "claim_contextual_occurrence", claim_occurrence)
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_execution",
        lambda *_a, **_k: events.append("finish-contextual") or {},
    )
    monkeypatch.setattr(
        scheduler,
        "_account_contextual_job_run",
        lambda **_k: events.append("account-job") or True,
    )
    monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda _workers: InterruptingPool())
    monkeypatch.setattr(
        scheduler,
        "_get_lock_paths",
        lambda: (tmp_path / "cron", tmp_path / "cron" / ".tick.lock"),
    )

    with pytest.raises(KeyboardInterrupt):
        scheduler.tick(verbose=False)

    expected = ["ledger", "seal-target", "claim-occurrence"]
    if interrupt_at == "submit":
        expected.append("submit")
    expected.extend(["finish-contextual", "account-job"])
    assert events == expected
    assert "contextual-interrupted" not in scheduler._running_job_ids


def test_legacy_isolated_reads_are_byte_preserving(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    jobs.CRON_DIR.mkdir(parents=True, exist_ok=True)
    legacy_bytes = (
        b'{\n  "version": 1,\n  "jobs": [\n    {\n'
        b'      "id": "legacy-bytes",\n'
        b'      "prompt": "standalone",\n'
        b'      "schedule": {"kind": "interval", "minutes": 5},\n'
        b'      "enabled": true\n'
        b'    }\n  ]\n}\n'
    )
    jobs.JOBS_FILE.write_bytes(legacy_bytes)

    loaded = jobs.get_job("legacy-bytes")
    listed = jobs.list_jobs()

    assert loaded["session_target"] == "isolated"
    assert listed[0]["session_target"] == "isolated"
    assert jobs.JOBS_FILE.read_bytes() == legacy_bytes


def test_legacy_isolated_update_keeps_contextual_fields_absent_on_disk(
    monkeypatch, tmp_path
):
    jobs = _point_store(monkeypatch, tmp_path)
    jobs.save_jobs(
        [
            {
                "id": "legacy-update",
                "prompt": "standalone",
                "schedule": {"kind": "interval", "minutes": 5},
                "enabled": True,
            }
        ]
    )

    jobs.update_job("legacy-update", {"name": "still isolated"})
    raw = json.loads(jobs.JOBS_FILE.read_text(encoding="utf-8"))["jobs"][0]

    assert raw["name"] == "still isolated"
    assert "session_target" not in raw
    assert "session_key" not in raw
    assert "context_binding" not in raw


def test_current_update_binding_is_immutable(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    _bind_session(
        "telegram:dm:42:42",
        route_principal={
            "scope_id": "tenant-a",
            "parent_chat_id": "parent-a",
            "user_id_alt": "user-alt-a",
            "chat_id_alt": "chat-alt-a",
        },
    )
    created = jobs.create_job(
        prompt="continue the discussion",
        schedule="every 1h",
        origin={"platform": "telegram", "chat_id": "42", "user_id": "42"},
        session_target="current",
    )
    assert created["session_target"] == "current"
    assert created["session_key"] == "telegram:dm:42:42"
    assert created["context_binding"]["route_instance_id"] == "route-instance-a"
    assert created["context_binding"]["scope_id"] == "tenant-a"
    assert created["context_binding"]["parent_chat_id"] == "parent-a"
    assert created["context_binding"]["user_id_alt"] == "user-alt-a"
    assert created["context_binding"]["chat_id_alt"] == "chat-alt-a"
    assert "session_id" not in created["context_binding"]
    assert "routing_revision" not in created["context_binding"]
    assert created["_contextual_binding_version"] == 2

    _bind_session(
        "telegram:dm:99:99", session_id="different-session", routing_revision=7
    )
    updated = jobs.update_job(created["id"], {"name": "renamed"})
    assert updated["session_key"] == "telegram:dm:42:42"

    recaptured = jobs.update_job(created["id"], {"session_target": "current"})
    assert recaptured["session_key"] == "telegram:dm:42:42"
    assert recaptured["context_binding"]["chat_id"] == "42"
    assert recaptured["context_binding"]["route_instance_id"] == "route-instance-a"
    assert "session_id" not in recaptured["context_binding"]
    assert "routing_revision" not in recaptured["context_binding"]
    assert recaptured["origin"]["chat_id"] == "42"

    with pytest.raises(ValueError, match="binding is permanent"):
        jobs.update_job(created["id"], {"session_target": "isolated"})

    still_contextual = jobs.get_job(created["id"])
    assert still_contextual is not None
    assert still_contextual["session_target"] == "current"
    assert still_contextual["session_key"] == "telegram:dm:42:42"


def test_contextual_validation_rejects_missing_physical_session_binding(monkeypatch):
    from cron import contextual

    monkeypatch.setattr(contextual, "_validate_contextual_scheduler_provider", lambda: None)
    with pytest.raises(ValueError, match="immutable session"):
        contextual.validate_contextual_job_shape(
            {
                "id": "missing-physical-binding",
                "prompt": "continue",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
                "context_binding": {
                    "session_key": "telegram:dm:42:42",
                    "platform": "telegram",
                    "chat_id": "42",
                    "user_id": "42",
                },
                "deliver": "origin",
            }
        )


def test_current_rejects_when_no_live_session_is_bound(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    from gateway.session_context import reset_session_vars

    reset_session_vars()
    with pytest.raises(ValueError, match="gateway-bound"):
        jobs.create_job(
            prompt="continue",
            schedule="every 1h",
            session_target="current",
        )


def test_current_rejects_incompatible_scheduler_before_persistence(
    monkeypatch, tmp_path
):
    jobs = _point_store(monkeypatch, tmp_path)
    _bind_session("telegram:dm:42:42")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"cron": {"provider": "chronos"}},
    )

    with pytest.raises(ValueError, match="requires cron.provider='builtin'"):
        jobs.create_job(
            prompt="continue",
            schedule="every 1h",
            session_target="current",
        )

    assert jobs.load_jobs() == []


def test_current_capture_rejects_process_environment_spoof(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    from gateway.session_context import reset_session_vars

    reset_session_vars()
    monkeypatch.setenv("HERMES_SESSION_KEY", "telegram:dm:attacker:attacker")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "attacker")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "attacker")

    with pytest.raises(ValueError, match="gateway-bound"):
        jobs.create_job(
            prompt="continue",
            schedule="every 1h",
            session_target="current",
        )


def test_current_create_derives_origin_from_same_trusted_binding(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    _bind_session("telegram:dm:42:42")

    created = jobs.create_job(
        prompt="continue",
        schedule="every 1h",
        origin={"platform": "telegram", "chat_id": "WRONG", "user_id": "WRONG"},
        session_target="current",
    )

    assert created["context_binding"]["session_key"] == "telegram:dm:42:42"
    assert created["origin"]["platform"] == "telegram"
    assert created["origin"]["chat_id"] == "42"
    assert created["origin"]["user_id"] == "42"


def test_new_isolated_job_omits_contextual_fields_on_disk(monkeypatch, tmp_path):
    jobs = _point_store(monkeypatch, tmp_path)
    created = jobs.create_job(prompt="standalone", schedule="every 1h")

    raw = json.loads(jobs.JOBS_FILE.read_text(encoding="utf-8"))["jobs"][0]
    assert raw["id"] == created["id"]
    assert "session_target" not in raw
    assert "session_key" not in raw
    assert "context_binding" not in raw


def test_contextual_delivery_never_uses_session_mirroring():
    from cron.scheduler import _cron_mirror_delivery_enabled

    contextual = {
        "session_target": "current",
        "attach_to_session": True,
    }
    assert not _cron_mirror_delivery_enabled(
        contextual,
        {"cron": {"mirror_delivery": True}},
    )
    assert _cron_mirror_delivery_enabled(
        {"attach_to_session": True},
        {"cron": {"mirror_delivery": False}},
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("no_agent", True),
        ("script", "/tmp/task.py"),
        ("skills", ["web"]),
        ("workdir", "/tmp"),
        ("context_from", ["upstream"]),
        ("enabled_toolsets", ["web"]),
        ("model", "other"),
        ("provider", "other"),
        ("base_url", "https://example.invalid"),
        ("monitor_script", "watch.py"),
        ("monitor_url", "https://example.invalid/watch"),
        ("attach_to_session", True),
        ("deliver", "all"),
        ("deliver", "telegram:other"),
    ],
)
def test_contextual_shape_rejects_incompatible_settings(field, value):
    from cron.contextual import validate_contextual_job_shape

    with pytest.raises(ValueError, match=field):
        validate_contextual_job_shape(
            {
                "session_target": "current",
                "session_key": "stable",
                **_physical_binding(),
                "prompt": "continue",
                field: value,
            }
        )


def test_contextual_live_tool_policy_is_an_explicit_fail_closed_allowlist():
    from cron.contextual import (
        CONTEXTUAL_ALLOWED_TOOLSETS,
        CONTEXTUAL_DISABLED_TOOLSETS,
        contextual_live_tool_policy,
    )

    enabled, disabled = contextual_live_tool_policy(
        ["web", "cronjob", "messaging", "clarify", "plugin_side_channel"],
        ["terminal"],
    )

    assert enabled == list(CONTEXTUAL_ALLOWED_TOOLSETS)
    assert enabled == []
    assert disabled[0] == "terminal"
    assert set(CONTEXTUAL_DISABLED_TOOLSETS).issubset(disabled)
    assert "plugin_side_channel" not in enabled
    assert len(disabled) == len(set(disabled))


def test_contextual_final_schema_gate_rejects_unknown_and_side_effect_tools():
    from cron.contextual import filter_contextual_tool_schemas

    def schema(name):
        return {"type": "function", "function": {"name": name}}

    assert filter_contextual_tool_schemas(
        [
            schema("read_file"),
            schema("web_search"),
            schema("terminal"),
            schema("delegate_task"),
            schema("send_message"),
            schema("plugin_side_channel"),
            {"type": "function", "function": {}},
            "malformed",
        ]
    ) == []


def test_task_local_contextual_authority_blocks_even_a_widened_agent_allowlist():
    from types import SimpleNamespace

    from agent.tool_executor import _tool_execution_policy_block
    from gateway.session_context import _bind_contextual_turn_authority

    agent = SimpleNamespace(allowed_tool_names=frozenset({"terminal"}))
    assert _tool_execution_policy_block(agent, "terminal") is None

    with _bind_contextual_turn_authority(
        execution_id="execution-1",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        admitted_routing_revision=7,
    ):
        blocked = _tool_execution_policy_block(agent, "terminal")

    assert blocked is not None
    assert "not permitted by this execution policy" in blocked


def test_model_schema_exposes_target_but_no_raw_session_key():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    properties = cast(Any, CRONJOB_SCHEMA)["parameters"]["properties"]
    assert properties["session_target"]["enum"] == ["isolated", "current"]
    assert "session_key" not in properties


def test_model_tool_cannot_synchronously_run_current_session_job(monkeypatch):
    """A tool call runs inside the human turn that contextual execution waits on."""
    import tools.cronjob_tools as cron_tools

    job = {
        "id": "job-current",
        "name": "current",
        "session_target": "current",
    }
    executed = []
    monkeypatch.setattr(cron_tools, "resolve_job_ref", lambda _ref: job)
    monkeypatch.setattr(
        cron_tools,
        "_execute_job_now",
        lambda _job: executed.append(_job) or {"claimed": True, "success": True},
    )

    result = json.loads(cron_tools.cronjob(action="run", job_id=job["id"]))

    assert result["success"] is False
    assert "after the active human turn" in result["error"]
    assert executed == []


def _patch_scheduler_bookkeeping(monkeypatch, scheduler):
    monkeypatch.setattr(
        scheduler, "_CONTEXTUAL_AUTHORIZER", lambda _target: True
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "get_execution", lambda _id: None)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _id: {"status": "running"})
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_: "output.json")
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_job_accounting",
        lambda _execution_id: True,
    )
    monkeypatch.setattr(
        scheduler,
        "seal_contextual_delivery_target",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(scheduler, "finish_contextual_execution", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scheduler,
        "persist_contextual_agent_result",
        lambda execution_id, **kwargs: {
            "id": execution_id,
            "outcome": kwargs.get("outcome"),
            "delivery_state": (
                "pending" if kwargs.get("outcome") == "notify" else "not_applicable"
            ),
        },
    )
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_delivery",
        lambda execution_id: {"id": execution_id, "delivery_state": "claimed"},
    )
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_delivery",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        scheduler,
        "suppress_contextual_delivery",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        scheduler,
        "prepare_contextual_retry",
        lambda execution_id: {"id": execution_id},
    )
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda _job: [{"platform": "telegram"}])
    monkeypatch.setattr(scheduler, "_consume_interrupted_flag", lambda _job_id: False)


def test_contextual_execution_rejects_malformed_sealed_delivery_target(monkeypatch):
    import cron.scheduler as scheduler

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    monkeypatch.setattr(
        scheduler,
        "get_execution",
        lambda _id: {
            "id": "execution",
            "status": "claimed",
            "phase": "claimed",
            "delivery_target_json": "{malformed",
        },
    )
    monkeypatch.setattr(
        scheduler,
        "verify_contextual_occurrence_claim",
        lambda *_args, **_kwargs: False,
    )
    finished = []
    accounted = []
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)) or {},
    )
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_job_accounting",
        lambda execution_id: accounted.append(execution_id) or True,
    )

    assert scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "must not run",
            "deliver": "origin",
        },
        contextual_dispatch=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("malformed target must fail before gateway dispatch")
        ),
    ) is True
    assert finished[0][1]["outcome"] == "rejected"
    assert "malformed" in finished[0][1]["error"]
    assert accounted == ["execution"]


def test_scheduler_routes_contextual_job_only_through_gateway_and_delivers_once(monkeypatch):
    import cron.scheduler as scheduler
    from gateway.contextual_cron import ContextualCronOutcome

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("isolated runner must not be used")
        ),
    )
    delivered = []
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda job, content, **kwargs: delivered.append(content) or None,
    )

    assert scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "continue",
            "deliver": "origin",
        },
        contextual_dispatch=lambda job, *, execution_id: ContextualCronOutcome.notify(
            "one final notification"
        ),
    ) is True
    assert delivered == ["one final notification"]


def test_contextual_delivery_rechecks_authorization_before_claim(monkeypatch):
    import cron.scheduler as scheduler
    from gateway.contextual_cron import ContextualCronOutcome

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", lambda _target: False)
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_delivery",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("revoked delivery must not be claimed")
        ),
    )
    delivered = []
    suppressed = []
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda *_args, **_kwargs: delivered.append(1),
    )
    monkeypatch.setattr(
        scheduler,
        "suppress_contextual_delivery",
        lambda execution_id, **kwargs: suppressed.append((execution_id, kwargs)) or {},
    )

    assert scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "continue",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "42", "user_id": "42"},
        },
        contextual_dispatch=lambda *_args, **_kwargs: ContextualCronOutcome.notify(
            "must not send"
        ),
    ) is True
    assert delivered == []
    assert suppressed == [
        (
            "execution",
            {"error": "Contextual cron authorization was revoked before delivery."},
        )
    ]


def test_contextual_delivery_recovery_rechecks_authorization_before_claim(monkeypatch):
    import json

    import cron.scheduler as scheduler

    events = []
    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", lambda _target: False)
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_delivery",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("claim forbidden")),
    )
    monkeypatch.setattr(
        scheduler,
        "suppress_contextual_delivery",
        lambda *_a, **_k: events.append("suppressed") or {},
    )
    monkeypatch.setattr(
        scheduler,
        "_account_contextual_job_run",
        lambda **_k: events.append("accounted") or True,
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("send forbidden")),
    )

    resumed = scheduler._resume_contextual_delivery_record(
        {"id": "job"},
        {
            "id": "execution",
            "result_json": json.dumps({"final_response": "notify"}),
            "delivery_target_json": json.dumps(
                {
                    "id": "job",
                    "deliver": "origin",
                    "origin": {
                        "platform": "telegram",
                        "chat_id": "42",
                        "user_id": "42",
                    },
                }
            ),
        },
    )

    assert resumed is False
    assert events == ["suppressed", "accounted"]


def test_v2_delivery_recovery_rejects_incomplete_creator_authority(monkeypatch):
    import json

    import cron.scheduler as scheduler

    finished = []
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)) or True,
    )
    monkeypatch.setattr(
        scheduler,
        "_CONTEXTUAL_AUTHORIZER",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("auth forbidden")),
    )

    resumed = scheduler._resume_contextual_delivery_record(
        {"id": "job"},
        {
            "id": "execution",
            "admitted_route_instance_id": "route-instance-a",
            "admitted_binding_version": 2,
            "result_json": json.dumps({"final_response": "notify"}),
            "delivery_target_json": json.dumps(
                {
                    "id": "job",
                    "deliver": "origin",
                    "origin": {
                        "platform": "telegram",
                        "chat_type": "dm",
                        "chat_id": "42",
                        "user_id": "",
                    },
                }
            ),
        },
    )

    assert resumed is False
    assert finished == [
        (
            "execution",
            {
                "outcome": "unknown",
                "error": "Pending contextual delivery has invalid creator authority.",
            },
        )
    ]


def test_v2_delivery_recovery_passes_sealed_route_authority_to_gateway(monkeypatch):
    import json

    import cron.scheduler as scheduler

    seen = []
    monkeypatch.setattr(
        scheduler,
        "_CONTEXTUAL_AUTHORIZER",
        lambda target: seen.append(target) or False,
    )
    monkeypatch.setattr(
        scheduler, "suppress_contextual_delivery", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        scheduler, "_account_contextual_job_run", lambda **_k: True
    )

    assert scheduler._resume_contextual_delivery_record(
        {"id": "job"},
        {
            "id": "execution",
            "session_key": "telegram:dm:42",
            "admitted_binding_version": 2,
            "admitted_route_instance_id": "route-instance-a",
            "admitted_session_id": "session-a",
            "admitted_routing_revision": 7,
            "result_json": json.dumps({"final_response": "notify"}),
            "delivery_target_json": json.dumps(
                {
                    "id": "job",
                    "deliver": "origin",
                    "origin": {
                        "platform": "telegram",
                        "chat_type": "dm",
                        "chat_id": "42",
                        "user_id": "42",
                    },
                }
            ),
        },
    ) is False

    assert seen[0]["_contextual_authority"] == {
        "execution_id": "execution",
        "session_key": "telegram:dm:42",
        "binding_version": 2,
        "route_instance_id": "route-instance-a",
        "session_id": "session-a",
        "routing_revision": 7,
    }


def test_recovery_never_retries_a_lost_route_locked_delivery_claim(monkeypatch):
    import json

    import cron.scheduler as scheduler

    def authorize(target):
        target["_contextual_delivery_claim_attempted"] = True
        return True

    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", authorize)
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_delivery",
        lambda *_a, **_k: pytest.fail("claim retried outside the route lock"),
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda *_a, **_k: pytest.fail("lost claim must not deliver"),
    )

    assert scheduler._resume_contextual_delivery_record(
        {"id": "job"},
        {
            "id": "execution",
            "session_key": "telegram:dm:42",
            "admitted_binding_version": 2,
            "admitted_route_instance_id": "route-instance-a",
            "admitted_session_id": "session-a",
            "admitted_routing_revision": 7,
            "result_json": json.dumps({"final_response": "notify"}),
            "delivery_target_json": json.dumps(
                {
                    "id": "job",
                    "deliver": "origin",
                    "origin": {
                        "platform": "telegram",
                        "chat_type": "dm",
                        "chat_id": "42",
                        "user_id": "42",
                    },
                }
            ),
        },
    ) is False


def test_live_completion_never_accounts_a_lost_route_locked_claim(monkeypatch):
    import cron.scheduler as scheduler
    from gateway.contextual_cron import ContextualCronOutcome

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)

    def authorize(target):
        target["_contextual_delivery_claim_attempted"] = True
        return True

    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", authorize)
    monkeypatch.setattr(
        scheduler,
        "claim_contextual_delivery",
        lambda *_a, **_k: pytest.fail("claim retried outside the route lock"),
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda *_a, **_k: pytest.fail("lost claim must not deliver"),
    )
    monkeypatch.setattr(
        scheduler,
        "_account_contextual_job_run",
        lambda **_k: pytest.fail("non-owner must not account the occurrence"),
    )

    assert scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "continue",
            "deliver": "origin",
        },
        contextual_dispatch=lambda *_a, **_k: ContextualCronOutcome.notify(
            "private result"
        ),
    ) is False


def test_contextual_delivery_exception_is_persisted_unknown_without_retry(monkeypatch):
    import cron.scheduler as scheduler
    from gateway.contextual_cron import ContextualCronOutcome

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    sends = []
    finishes = []

    def uncertain_send(*_args, **_kwargs):
        sends.append(1)
        raise scheduler.ContextualDeliveryUnknown("acknowledgement lost")

    monkeypatch.setattr(scheduler, "_deliver_result", uncertain_send)
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_delivery",
        lambda execution_id, **kwargs: finishes.append((execution_id, kwargs)),
    )

    assert scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "continue",
            "deliver": "origin",
        },
        contextual_dispatch=lambda *_args, **_kwargs: ContextualCronOutcome.notify(
            "one final notification"
        ),
    )

    assert sends == [1]
    assert finishes == [
        (
            "execution",
            {"delivery_state": "unknown", "error": "acknowledgement lost"},
        )
    ]


def test_contextual_no_action_is_successful_and_silent(monkeypatch):
    import cron.scheduler as scheduler
    from gateway.contextual_cron import ContextualCronOutcome

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    delivered = []
    marks = []
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: delivered.append(1))
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda job_id, success, error, **kwargs: marks.append((success, error)),
    )

    scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "check",
            "deliver": "origin",
        },
        contextual_dispatch=lambda job, *, execution_id: ContextualCronOutcome.no_action(),
    )
    assert delivered == []
    assert marks == [(True, None)]


def test_contextual_retryable_reuses_occurrence_then_delivers_once(monkeypatch):
    import cron.scheduler as scheduler
    from gateway.contextual_cron import ContextualCronOutcome

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    outcomes = [
        ContextualCronOutcome.retryable("human turn still finishing"),
        ContextualCronOutcome.notify("after the wait"),
    ]
    prepared = []
    delivered = []
    monkeypatch.setattr(
        scheduler,
        "prepare_contextual_retry",
        lambda execution_id: prepared.append(execution_id) or {"id": execution_id},
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kwargs: delivered.append(content) or None,
    )

    assert scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "same-execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "continue",
            "deliver": "origin",
        },
        contextual_dispatch=lambda *_args, **_kwargs: outcomes.pop(0),
    ) is True

    assert prepared == ["same-execution"]
    assert delivered == ["after the wait"]


def test_contextual_without_gateway_is_rejected_never_isolated(monkeypatch):
    import cron.scheduler as scheduler

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    scheduler.set_contextual_dispatcher(None)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no isolation fallback")
        ),
    )
    marks = []
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda job_id, success, error, **kwargs: marks.append((success, error)),
    )
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)

    scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "continue",
            "deliver": "local",
        }
    )
    assert marks[0][0] is False
    assert "fallback" in marks[0][1]


def test_scheduler_revalidates_persisted_contextual_shape_before_dispatch(monkeypatch):
    import cron.scheduler as scheduler

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    dispatched = []
    marks = []
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda job_id, success, error, **kwargs: marks.append((success, error)),
    )
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)

    scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "continue",
            "deliver": "telegram:other-user",
        },
        contextual_dispatch=lambda *_args, **_kwargs: dispatched.append(True),
    )

    assert dispatched == []
    assert marks and marks[0][0] is False
    assert "deliver" in marks[0][1]


def test_invalid_session_target_never_falls_back_to_isolated(monkeypatch):
    import cron.scheduler as scheduler

    _patch_scheduler_bookkeeping(monkeypatch, scheduler)
    isolated = []
    marks = []
    monkeypatch.setattr(scheduler, "run_job", lambda *_a, **_k: isolated.append(True))
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda job_id, success, error, **kwargs: marks.append((success, error)),
    )
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_a, **_k: None)

    scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "typo",
            "prompt": "must not run",
            "deliver": "local",
        }
    )

    assert isolated == []
    assert marks and marks[0][0] is False
    assert "session_target" in marks[0][1]


def test_terminal_contextual_occurrence_is_not_dispatched_or_redelivered(monkeypatch):
    import cron.scheduler as scheduler

    monkeypatch.setattr(
        scheduler,
        "get_execution",
        lambda _execution_id: {"status": "completed", "outcome": "notify"},
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "claim_dispatch",
        lambda _job_id: (_ for _ in ()).throw(AssertionError("must not reclaim")),
    )
    dispatched = []

    assert scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "already-terminal",
            "session_target": "current",
            "session_key": "stable",
            **_physical_binding(),
            "prompt": "must not repeat",
            "deliver": "origin",
        },
        contextual_dispatch=lambda *_a, **_k: dispatched.append(True),
    ) is True
    assert dispatched == []


def test_contextual_delivery_ack_is_durable_before_job_success_accounting(monkeypatch):
    import cron.contextual as contextual
    import cron.scheduler as scheduler
    from gateway.contextual_cron import ContextualCronOutcome

    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", lambda _target: True)
    events = []
    monkeypatch.setattr(
        scheduler,
        "get_execution",
        lambda _execution_id: {
            "status": "claimed",
            "phase": "claimed",
            "delivery_target_json": '{"id":"job"}',
        },
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(
        scheduler, "mark_execution_running", lambda _execution_id: {"status": "running"}
    )
    monkeypatch.setattr(
        scheduler, "persist_contextual_agent_result", lambda *_a, **_k: {"phase": "agent_completed"}
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a, **_k: "output")
    monkeypatch.setattr(scheduler, "claim_contextual_delivery", lambda *_a, **_k: {})
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda *_a, **_k: events.append("send") or None,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_delivery",
        lambda *_a, **_k: events.append("delivery-ack") or {},
    )
    monkeypatch.setattr(
        scheduler,
        "_account_contextual_job_run",
        lambda **_k: events.append("job-accounting") or True,
    )
    monkeypatch.setattr(contextual, "validate_contextual_job_shape", lambda _job: None)

    assert scheduler.run_one_job(
        {
            "id": "job",
            "execution_id": "execution",
            "session_target": "current",
            "session_key": "telegram:dm:42:42",
            "prompt": "continue",
            "deliver": "telegram",
            "origin": {"platform": "telegram", "chat_id": "42", "user_id": "42"},
        },
        contextual_dispatch=lambda *_a, **_k: ContextualCronOutcome.notify("done"),
    ) is True

    assert events == ["send", "delivery-ack", "job-accounting"]


def test_recovered_contextual_delivery_ack_precedes_job_success_accounting(monkeypatch):
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", lambda _target: True)
    events = []
    monkeypatch.setattr(scheduler, "claim_contextual_delivery", lambda *_a, **_k: {})
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a, **_k: "output")
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda *_a, **_k: events.append("send") or None,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_delivery",
        lambda *_a, **_k: events.append("delivery-ack") or {},
    )
    monkeypatch.setattr(
        scheduler,
        "_account_contextual_job_run",
        lambda **_k: events.append("job-accounting") or True,
    )

    assert scheduler._resume_contextual_delivery_record(
        {"id": "job"},
        {
            "id": "execution",
            "result_json": '{"final_response":"done"}',
            "delivery_target_json": '{"id":"job","deliver":"telegram",'
            '"origin":{"platform":"telegram","chat_id":"42"}}',
        },
    ) is True

    assert events == ["send", "delivery-ack", "job-accounting"]


def test_interrupted_contextual_delivery_records_unknown_before_job_accounting(monkeypatch):
    import cron.contextual as contextual
    import cron.scheduler as scheduler
    from gateway.contextual_cron import ContextualCronOutcome

    monkeypatch.setattr(scheduler, "_CONTEXTUAL_AUTHORIZER", lambda _target: True)
    events = []
    records = iter(
        [
            {
                "status": "claimed",
                "phase": "claimed",
                "delivery_target_json": '{"id":"job"}',
            },
            {
                "status": "running",
                "phase": "delivering",
                "delivery_state": "claimed",
            },
        ]
    )
    monkeypatch.setattr(scheduler, "get_execution", lambda _execution_id: next(records))
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(
        scheduler, "mark_execution_running", lambda _execution_id: {"status": "running"}
    )
    monkeypatch.setattr(
        scheduler, "persist_contextual_agent_result", lambda *_a, **_k: {"phase": "agent_completed"}
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a, **_k: "output")
    monkeypatch.setattr(scheduler, "claim_contextual_delivery", lambda *_a, **_k: {})

    def _interrupt(*_args, **_kwargs):
        events.append("send")
        raise KeyboardInterrupt("transport boundary crossed")

    monkeypatch.setattr(scheduler, "_deliver_result", _interrupt)
    monkeypatch.setattr(
        scheduler,
        "finish_contextual_delivery",
        lambda *_a, **_k: events.append("delivery-unknown") or {},
    )
    monkeypatch.setattr(
        scheduler,
        "_account_contextual_job_run",
        lambda **_k: events.append("job-accounting") or True,
    )
    monkeypatch.setattr(contextual, "validate_contextual_job_shape", lambda _job: None)

    with pytest.raises(KeyboardInterrupt, match="transport boundary crossed"):
        scheduler.run_one_job(
            {
                "id": "job",
                "execution_id": "execution",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
                "prompt": "continue",
                "deliver": "telegram",
                "origin": {
                    "platform": "telegram",
                    "chat_id": "42",
                    "user_id": "42",
                },
            },
            contextual_dispatch=lambda *_a, **_k: ContextualCronOutcome.notify("done"),
        )

    assert events == ["send", "delivery-unknown", "job-accounting"]
