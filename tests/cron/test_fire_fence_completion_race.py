"""Regression tests for fire-fence heartbeat settlement races.

A worker's own side-effect fence can make its heartbeat return ``None`` while
saving or delivering. That self-contention must not become ownership loss by
duration alone. Confirmed takeover and external cancellation still fail closed.
"""

import contextlib
import errno
import threading
import time
from unittest.mock import patch

import pytest

import cron.scheduler as scheduler


@contextlib.contextmanager
def _owned_fence(*_args, **_kwargs):
    yield True


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _make_claimed_job(jobs, name):
    job = jobs.create_job(prompt="x", schedule="every 5m", name=name)
    assert jobs.claim_job_for_fire(job["id"]) is True
    stored = jobs.get_job(job["id"])
    assert stored is not None and stored.get("fire_claim")
    return job, stored["fire_claim"]["by"]


def test_fire_fence_contention_errno_is_portable():
    import cron.jobs as jobs

    assert jobs._is_fire_fence_contention(OSError(errno.EACCES, "busy"))
    assert jobs._is_fire_fence_contention(OSError(errno.EAGAIN, "busy"))
    assert not jobs._is_fire_fence_contention(OSError(errno.EMFILE, "unavailable"))


def test_heartbeat_lock_backend_error_propagates_as_store_uncertainty(
    temp_home, monkeypatch
):
    """A fence I/O failure is not the known-contention ``None`` verdict."""
    import cron.jobs as jobs

    job, owner = _make_claimed_job(jobs, "fence-io-error")

    def unavailable(*_args, **_kwargs):
        raise OSError("lock backend unavailable")

    monkeypatch.setattr(jobs, "open", unavailable, raising=False)

    with pytest.raises(OSError, match="lock backend unavailable"):
        jobs.heartbeat_fire_claim(job["id"], expected_owner=owner)


def test_store_error_gets_full_grace_after_prolonged_fence_contention(monkeypatch):
    """Known contention does not pre-spend later backend-error grace."""
    first_error = threading.Event()
    calls = 0

    def heartbeat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return True
        if calls <= 16:
            return None
        first_error.set()
        raise OSError("store unavailable")

    def run_body(_job, **kwargs):
        cancel = kwargs["fire_claim_lost"]
        assert first_error.wait(timeout=1)
        assert not cancel.wait(timeout=0.02)
        assert cancel.wait(timeout=0.3)
        return True

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.005)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.05)

    assert scheduler.run_one_job(
        {"id": "mixed-uncertainty", "fire_claim": {"by": "owner"}}
    ) is True


def test_successful_long_delivery_is_not_demoted_after_heartbeat_grace(
    temp_home, monkeypatch
):
    """A delivery holding its own fence past heartbeat grace stays successful."""
    import cron.jobs as jobs
    from cron.executions import latest_execution

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.005)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.02)

    job, owner = _make_claimed_job(jobs, "long-delivery")
    jid = job["id"]
    delivered = []

    def slow_delivery(_job, content, **_kwargs):
        # _deliver_result runs under the real side-effect fence. Keep it held
        # long enough for repeated heartbeat lock timeouts to exceed grace.
        time.sleep(0.12)
        delivered.append(content)
        return None

    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: {})
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_a, **_kw: (True, "out", "final response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "out.md")
    monkeypatch.setattr(scheduler, "_deliver_result", slow_delivery)

    with (
        patch("cron.scheduler._launch_external_cron_worker", return_value=False),
        patch("agent.secret_scope.set_secret_scope", return_value=None),
        patch("agent.secret_scope.build_profile_secret_scope", return_value=None),
        patch("agent.secret_scope.reset_secret_scope"),
        patch("tools.terminal_scope.install_profile_terminal_scope", return_value=None),
        patch("tools.terminal_scope.reset_terminal_scope"),
    ):
        assert scheduler.run_one_job(
            {
                "id": jid,
                "name": "long-delivery",
                "deliver": "all",
                "fire_claim": {"at": "2026-09-01T00:00:00+00:00", "by": owner},
            }
        ) is True

    assert delivered == ["final response"]
    final = jobs.get_job(jid)
    assert final is not None
    assert final["last_status"] == "ok", final
    assert not final.get("last_error"), final
    assert not final.get("failure_streak"), final
    row = latest_execution(jid)
    assert row is not None and row["status"] == "completed", row


def test_shutdown_during_long_delivery_still_fails_closed(temp_home, monkeypatch):
    """A real transport cancellation is not mistaken for fence self-contention."""
    import cron.jobs as jobs
    from cron.executions import latest_execution

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.005)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.02)

    job, owner = _make_claimed_job(jobs, "cancelled-long-delivery")
    jid = job["id"]
    external_cancel = threading.Event()

    def cancelled_delivery(*_args, **_kwargs):
        time.sleep(0.12)
        external_cancel.set()
        return None

    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: {})
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_a, **_kw: (True, "out", "final response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "out.md")
    monkeypatch.setattr(scheduler, "_deliver_result", cancelled_delivery)

    with (
        patch("cron.scheduler._launch_external_cron_worker", return_value=False),
        patch("agent.secret_scope.set_secret_scope", return_value=None),
        patch("agent.secret_scope.build_profile_secret_scope", return_value=None),
        patch("agent.secret_scope.reset_secret_scope"),
        patch("tools.terminal_scope.install_profile_terminal_scope", return_value=None),
        patch("tools.terminal_scope.reset_terminal_scope"),
    ):
        assert scheduler.run_one_job(
            {
                "id": jid,
                "name": "cancelled-long-delivery",
                "deliver": "all",
                "fire_claim": {"at": "2026-09-01T00:00:00+00:00", "by": owner},
            },
            cancel_event=external_cancel,
        ) is True

    final = jobs.get_job(jid)
    assert final is not None
    assert final["last_status"] == "error", final
    assert final["failure_streak"] == 1, final
    assert final["last_error"] == "Interrupted by shutdown before terminal completion."
    row = latest_execution(jid)
    assert row is not None and row["status"] == "failed", row


def test_genuine_ownership_loss_still_fails_closed(monkeypatch):
    """A confirmed takeover discards the stale result without delivering it."""
    finished = []
    delivered = []

    monkeypatch.setattr(
        scheduler, "create_execution", lambda *_a, **_kw: {"id": "exec-loss"}
    )
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda *_a: False)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: {})
    monkeypatch.setattr(
        scheduler, "run_job", lambda *_a, **_kw: (True, "out", "stale", None)
    )
    monkeypatch.setattr(scheduler, "fire_claim_fence", _owned_fence, raising=False)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "out.md")
    monkeypatch.setattr(
        scheduler, "_deliver_result", lambda *_a, **_kw: delivered.append(True)
    )

    probes = {"n": 0}

    def heartbeat(_job_id, *, expected_owner):
        probes["n"] += 1
        return probes["n"] == 1

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )

    with (
        patch("agent.secret_scope.set_secret_scope", return_value=None),
        patch("agent.secret_scope.build_profile_secret_scope", return_value=None),
        patch("agent.secret_scope.reset_secret_scope"),
        patch("tools.terminal_scope.install_profile_terminal_scope", return_value=None),
        patch("tools.terminal_scope.reset_terminal_scope"),
    ):
        assert scheduler.run_one_job(
            {
                "id": "taken-over",
                "name": "taken-over",
                "deliver": "all",
                "fire_claim": {
                    "at": "2026-09-01T00:00:00+00:00",
                    "by": "owner-A",
                },
            }
        ) is True

    assert delivered == []
    assert finished == [
        (
            "exec-loss",
            {
                "success": False,
                "error": "Fire claim ownership lost; stale result was discarded.",
            },
        )
    ]


def test_ownership_loss_recording_distinguishes_fence_uncertainty(monkeypatch):
    """A contended ownership probe is not a confirmed stale-owner loss."""
    finished = []
    marked = []
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda *_a, **_kw: marked.append((_a, _kw)),
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )

    scheduler._record_fire_ownership_lost("job-none", "owner-A", "exec-none")

    assert marked == []
    assert finished == [
        (
            "exec-none",
            {
                "success": False,
                "error": (
                    "Fire claim fence unavailable while resolving cancellation; "
                    "outcome uncertain."
                ),
            },
        )
    ]


def test_transport_cancel_while_waiting_for_delivery_fence_suppresses_send(monkeypatch):
    """Cancellation won under the delivery fence cannot leak normal output."""
    cancel = threading.Event()
    fence_entries = 0
    delivered = []
    finished = []

    @contextlib.contextmanager
    def cancelling_fence(*_args, **_kwargs):
        nonlocal fence_entries
        fence_entries += 1
        if fence_entries == 2:
            cancel.set()
        yield True

    monkeypatch.setattr(
        scheduler, "create_execution", lambda *_a, **_kw: {"id": "exec-cancel-race"}
    )
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda *_a: False)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: {})
    monkeypatch.setattr(
        scheduler, "run_job", lambda *_a, **_kw: (True, "out", "normal output", None)
    )
    monkeypatch.setattr(scheduler, "fire_claim_fence", cancelling_fence)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "out.md")
    monkeypatch.setattr(
        scheduler, "_deliver_result", lambda *_a, **_kw: delivered.append(True)
    )
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )

    with (
        patch("agent.secret_scope.set_secret_scope", return_value=None),
        patch("agent.secret_scope.build_profile_secret_scope", return_value=None),
        patch("agent.secret_scope.reset_secret_scope"),
        patch("tools.terminal_scope.install_profile_terminal_scope", return_value=None),
        patch("tools.terminal_scope.reset_terminal_scope"),
    ):
        assert scheduler.run_one_job(
            {
                "id": "cancel-race",
                "name": "cancel-race",
                "deliver": "all",
                "fire_claim": {
                    "at": "2026-09-01T00:00:00+00:00",
                    "by": "owner-A",
                },
            },
            cancel_event=cancel,
        ) is True

    assert delivered == []
    assert finished == [
        (
            "exec-cancel-race",
            {
                "success": False,
                "error": "Interrupted by shutdown before terminal completion.",
            },
        )
    ]
