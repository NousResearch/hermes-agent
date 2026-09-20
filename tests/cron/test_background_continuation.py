"""Opt-in job-scoped continuation for background processes launched by cron (#110650).

A cron execution has no turn left to re-enter, so a finished ``terminal(background=true)`` child
must not be announced through the session-key-routed completion path — an ambient session key
could inject a late child's output into an unrelated chat (#53027 / #63142, see
``cron.scheduler._CronRunScope``).

A job that declares ``background_continuation: true`` opts into a job-scoped path instead: the
child's exit is persisted as a durable record carrying the job's OWN routing, and a later
scheduler tick claims it and runs ONE continuation agent turn, delivered through the job's own
delivery config. Default (field absent) must be byte-identical to pre-feature behaviour.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

import cron.scheduler as scheduler_mod
from cron import continuation as cont

_JOB_ID = "jobabc123"
_PAYLOAD = {
    "job_id": _JOB_ID, "job_name": "build", "execution_id": "exec1",
    "platform": "telegram", "chat_id": "-100", "thread_id": "7",
}


@pytest.fixture(autouse=True)
def hermes_home(tmp_path, monkeypatch):
    """Profile-local storage (records, receipts, checkpoint, tick lock) inside tmp_path."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _restore_async_delivery():
    """cron binds async_delivery=False; make sure no test leaks that into the next one."""
    from gateway.session_context import _SESSION_ASYNC_DELIVERY, _UNSET

    token = _SESSION_ASYNC_DELIVERY.set(_UNSET)
    yield
    _SESSION_ASYNC_DELIVERY.reset(token)


def _job(**extra):
    job = {
        "id": _JOB_ID, "name": "build", "prompt": "start the build", "enabled": True,
        "state": "scheduled", "deliver": "telegram:-100:7",
        "origin": {"platform": "telegram", "chat_id": "-100", "thread_id": "7"},
    }
    job.update(extra)
    return job


def _stateless_scope():
    """Context manager-ish helper: bind a stateless channel exactly like a cron execution."""
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform="", chat_id="", chat_name="", cwd="", async_delivery=False)
    return lambda: clear_session_vars(tokens)


def _session(**kwargs):
    from tools.process_registry import ProcessSession

    defaults = dict(id="proc_cont1", command="make build", output_buffer="boom\n")
    defaults.update(kwargs)
    return ProcessSession(**defaults)


class TestDefaultOff:
    """No opt-in -> nothing about the existing safety boundary changes."""

    def test_job_without_optin_never_binds_a_continuation(self, monkeypatch):
        monkeypatch.setattr(
            scheduler_mod, "_resolve_delivery_target",
            lambda job: {"platform": "telegram", "chat_id": "-100", "thread_id": "7"})

        scope = scheduler_mod._CronRunScope(_job(), _JOB_ID, "exec1")
        scope.enter()
        try:
            assert cont.active() is None
        finally:
            scope.exit()
        assert cont.active() is None

    def test_stateless_child_without_optin_still_has_notify_stripped(self):
        """The pre-existing cron behaviour: notify/watch dropped, agent told to poll."""
        from tools.terminal_tool_background import _apply_async_support

        release = _stateless_scope()
        try:
            session = _session()
            result: dict = {}
            notify, watch = _apply_async_support(session, result, True, ["done"])

            assert (notify, watch) == (False, None)
            assert result["notify_on_complete"] is False
            assert result["notify_unsupported"]
            assert session.cron_continuation is None
            assert session.notify_on_complete is False
        finally:
            release()

    def test_session_without_payload_writes_no_record(self, hermes_home):
        assert cont.write_record(_session()) is None
        assert not (cont._dir().exists() and list(cont._dir().glob("*.json")))


class TestOptInArmsJobScopedRouting:
    def test_scope_binds_job_routing_and_ignores_ambient_session_env(self, monkeypatch):
        """Routing comes from the job's own delivery target — never the ambient session key."""
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "slack")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "C-AMBIENT")
        monkeypatch.setenv("HERMES_SESSION_KEY", "slack:C-AMBIENT")
        monkeypatch.setattr(
            scheduler_mod, "_resolve_delivery_target",
            lambda job: {"platform": "telegram", "chat_id": "-100", "thread_id": "7"})

        scope = scheduler_mod._CronRunScope(
            _job(background_continuation=True), _JOB_ID, "exec1")
        scope.enter()
        try:
            payload = cont.active()
            assert payload["job_id"] == _JOB_ID
            assert payload["execution_id"] == "exec1"
            assert (payload["platform"], payload["chat_id"], payload["thread_id"]) == (
                "telegram", "-100", "7")
            # The ambient (unrelated-chat) identity must not appear anywhere in the payload.
            assert "C-AMBIENT" not in json.dumps(payload)
            assert "slack" not in json.dumps(payload)

            from gateway.session_context import async_delivery_supported

            # Delegation still runs inline: the safety boundary is untouched.
            assert async_delivery_supported() is False
        finally:
            scope.exit()
        assert cont.active() is None, "exit() must unbind"

    def test_continuation_run_does_not_rearm_the_context(self, monkeypatch):
        """Recursion guard: a continuation turn can never spawn another continuation."""
        monkeypatch.setattr(
            scheduler_mod, "_resolve_delivery_target",
            lambda job: {"platform": "telegram", "chat_id": "-100", "thread_id": "7"})

        scope = scheduler_mod._CronRunScope(
            _job(background_continuation=True), _JOB_ID, "exec1", continuation=True)
        scope.enter()
        try:
            assert cont.active() is None
        finally:
            scope.exit()

    def test_notify_survives_the_gate_and_arms_a_job_scoped_continuation(self):
        from tools.terminal_tool_background import _apply_async_support

        release = _stateless_scope()
        token = cont.bind(_job(), _JOB_ID, "exec1", {"platform": "telegram", "chat_id": "-100", "thread_id": "7"})
        try:
            session = _session(id="proc_armed")
            result: dict = {}
            notify, watch = _apply_async_support(session, result, True, None)

            # The tool result reports the exit WILL be acted on...
            assert result["notify_on_complete"] is True
            assert result["cron_continuation"] == {"job_id": _JOB_ID, "process_id": "proc_armed"}
            assert "notify_unsupported" not in result
            # ...via the job-scoped record, with the generic session-routed path still closed.
            assert notify is False and watch is None
            assert session.cron_continuation["job_id"] == _JOB_ID
            assert session.notify_on_complete is False
            assert session.watcher_platform == ""  # no gateway watcher attached
        finally:
            cont.unbind(token)
            release()


class TestCompletionRecord:
    def test_finished_child_writes_a_job_scoped_record_and_no_queue_event(self):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        session = _session(
            id="proc_record1", command="make build", cron_continuation=dict(_PAYLOAD),
            task_id="cron:jobabc123:exec1")
        session.exit_code = 3
        session.completion_reason = "exited"
        session.output_buffer = "3 tests failed\n"
        with registry._lock:
            registry._running[session.id] = session

        registry._move_to_finished(session)

        # Nothing for a live chat session to drain: no ambient-routed completion event.
        assert registry.completion_queue.empty()

        records = cont.claim_pending()
        assert len(records) == 1
        record = records[0]
        assert record["job_id"] == _JOB_ID
        assert record["process_id"] == "proc_record1"
        assert record["command"] == "make build"
        assert record["exit_code"] == 3
        assert record["chat_id"] == "-100"
        assert "3 tests failed" in record["output"]
        # Idempotent: a claimed record is gone, so a replayed completion cannot double-fire.
        assert cont.claim_pending() == []

    def test_claim_survives_a_scheduler_restart(self):
        """The record is on disk, so a replacement scheduler claims it (and only once)."""
        session = _session(id="proc_restart", cron_continuation=dict(_PAYLOAD))
        session.exit_code = 0
        cont.write_record(session)

        assert len(cont.claim_pending()) == 1
        assert cont.claim_pending() == []


class TestSchedulerResumes:
    def _write_record(self, process_id="proc_resume", **kwargs):
        session = _session(id=process_id, cron_continuation=dict(_PAYLOAD), command="make build",
                           output_buffer="tests failed\n")
        session.exit_code = kwargs.pop("exit_code", 2)
        session.completion_reason = kwargs.pop("completion_reason", "exited")
        cont.write_record(session)

    def test_resume_builds_a_turn_and_delivers_by_job_config(self, monkeypatch):
        job = _job(background_continuation=True)
        self._write_record()
        monkeypatch.setattr("cron.jobs.get_job", lambda job_id: job)
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        calls: dict = {}
        monkeypatch.setattr(
            scheduler_mod, "run_job",
            lambda j, **kw: (calls.update(job=j, kwargs=kw), (True, "doc", "Build failed: 2 tests", None))[1])
        delivered: dict = {}
        monkeypatch.setattr(
            scheduler_mod, "_deliver_result",
            lambda j, content, **kw: (delivered.update(job=j, content=content, kwargs=kw), None)[1])

        assert scheduler_mod.resume_background_continuations() == 1

        assert calls["kwargs"]["continuation"] is True
        prompt = calls["kwargs"]["extra_prompt"]
        assert _JOB_ID in prompt and "make build" in prompt and "proc_resume" in prompt
        assert "Exit code:** 2" in prompt and "tests failed" in prompt
        # Delivery uses the job's own config (its deliver target / failure flag).
        assert delivered["job"]["id"] == _JOB_ID
        assert delivered["content"] == "Build failed: 2 tests"
        assert delivered["kwargs"]["for_failure"] is False
        # One continuation per process, ever.
        assert scheduler_mod.resume_background_continuations() == 0

    def test_record_is_dropped_when_job_is_gone_paused_or_opted_out(self, monkeypatch):
        self._write_record()
        calls = []
        monkeypatch.setattr(scheduler_mod, "run_job", lambda *a, **kw: calls.append(1))
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        monkeypatch.setattr("cron.jobs.get_job", lambda job_id: None)

        assert scheduler_mod.resume_background_continuations() == 0
        assert calls == []
        # Claimed and dropped, not left to re-fire forever.
        assert cont.claim_pending() == []

        self._write_record(process_id="proc_opted_out")
        monkeypatch.setattr("cron.jobs.get_job", lambda job_id: _job())
        assert scheduler_mod.resume_background_continuations() == 0
        assert calls == []

        self._write_record(process_id="proc_paused")
        monkeypatch.setattr("cron.jobs.get_job", lambda job_id: _job(background_continuation=True))
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: False)
        assert scheduler_mod.resume_background_continuations() == 0
        assert calls == []

    def test_failure_result_is_delivered_as_a_failure_notice(self, monkeypatch):
        self._write_record()
        monkeypatch.setattr("cron.jobs.get_job", lambda job_id: _job(background_continuation=True))
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        monkeypatch.setattr(scheduler_mod, "run_job", lambda j, **kw: (False, "doc", "", "boom"))
        delivered: dict = {}
        monkeypatch.setattr(
            scheduler_mod, "_deliver_result",
            lambda j, content, **kw: (delivered.update(content=content, kwargs=kw), None)[1])

        assert scheduler_mod.resume_background_continuations() == 1
        assert delivered["kwargs"]["for_failure"] is True

    def test_tick_drains_the_record_even_with_no_due_jobs(self, monkeypatch):
        job = _job(background_continuation=True)
        self._write_record(process_id="proc_tick")
        monkeypatch.setattr("cron.jobs.get_job", lambda job_id: job)
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        monkeypatch.setattr(
            scheduler_mod, "run_job", lambda j, **kw: (True, "doc", "done", None))
        monkeypatch.setattr(scheduler_mod, "_deliver_result", lambda j, content, **kw: None)

        with (
            patch.object(scheduler_mod, "get_due_jobs", return_value=[]),
            patch.object(scheduler_mod, "_maybe_reap_dead_owners", lambda: None),
            patch.object(scheduler_mod, "_maybe_run_worktree_maintenance", lambda: None),
            patch("tools.mcp_tool_lifecycle._kill_orphaned_mcp_children", lambda: None),
        ):
            executed = scheduler_mod.tick(verbose=False)

        assert executed == 1, "a continuation counts as one execution"
        assert cont.claim_pending() == []


class TestLiveChildEndToEnd:
    """Real Popen + real reader thread: a cron-launched child's exit reaches the scheduler."""

    def test_opted_in_child_exit_writes_the_job_scoped_record(self, hermes_home):
        import os

        from tools.terminal_tool_background import spawn_background_process

        release = _stateless_scope()
        token = cont.bind(_job(), _JOB_ID, "exec1",
                          {"platform": "telegram", "chat_id": "-100", "thread_id": "7"})
        try:
            raw = spawn_background_process(
                command="sleep 2; echo continuation-e2e; exit 4", env=None, env_type="local",
                effective_task_id="cron:jobabc123:exec1", task_id="cron:jobabc123:exec1",
                session_key="", workdir=None, cwd=os.getcwd(), effective_pty=False,
                notify_on_complete=True, watch_patterns=None, approval_note=None,
                pty_disabled_reason=None)
            result = json.loads(raw)
        finally:
            cont.unbind(token)
            release()

        assert result["cron_continuation"]["job_id"] == _JOB_ID
        assert result["exit_code"] == 0  # spawn result, not the child's exit

        deadline = time.time() + 60
        files = []
        while time.time() < deadline:
            files = list(cont._dir().glob("*.json")) if cont._dir().exists() else []
            if files:
                break
            time.sleep(0.25)
        assert files, "child exit never produced a continuation record"

        record = json.loads(files[0].read_text(encoding="utf-8"))
        assert record["job_id"] == _JOB_ID
        assert record["chat_id"] == "-100"
        assert record["exit_code"] == 4
        assert "continuation-e2e" in record["output"]
        assert cont.claim_pending(), "the record must be claimable by the scheduler"


class TestJobOptInSurface:
    @pytest.fixture()
    def cron_env(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "prof"
        (hermes_home / "cron" / "output").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import cron.jobs as jobs_mod

        monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
        monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
        monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
        return hermes_home

    def test_default_create_stores_no_field(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(prompt="nightly", schedule="every 1h")
        assert "background_continuation" not in job
        assert "background_continuation" not in (get_job(job["id"]) or {})

    def test_optin_round_trips_through_the_store_and_update(self, cron_env):
        from cron.jobs import create_job, get_job, update_job

        job = create_job(prompt="build", schedule="every 1h", background_continuation=True)
        assert job["background_continuation"] is True
        assert get_job(job["id"])["background_continuation"] is True

        update_job(job["id"], {"background_continuation": False})
        assert not get_job(job["id"]).get("background_continuation")


class TestReviewBlockers:
    """Regression tests for the three P1s on #111049."""

    def _write_record(self, process_id="proc_resume", **kwargs):
        session = _session(id=process_id, cron_continuation=dict(_PAYLOAD), command="make build",
                           output_buffer="tests failed\n")
        session.exit_code = kwargs.pop("exit_code", 2)
        session.completion_reason = kwargs.pop("completion_reason", "exited")
        cont.write_record(session)

    def test_retargeted_job_drops_the_stale_child_record(self, monkeypatch):
        """P1: a child spawned for A must never be delivered through B after a job edit."""
        self._write_record(process_id="proc_stale")
        retargeted = _job(deliver="slack:C99",
                          origin={"platform": "slack", "chat_id": "C99"},
                          background_continuation=True)
        monkeypatch.setattr("cron.jobs.get_job", lambda job_id: retargeted)
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        calls = []
        monkeypatch.setattr(scheduler_mod, "run_job", lambda *a, **kw: calls.append(1))

        assert scheduler_mod.resume_background_continuations() == 0
        assert calls == []
        assert cont.claim_pending() == []

    def test_matching_route_still_resumes(self, monkeypatch):
        """The fence must not swallow the normal case: unchanged target resumes."""
        self._write_record(process_id="proc_same")
        monkeypatch.setattr(
            "cron.jobs.get_job", lambda job_id: _job(background_continuation=True))
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        monkeypatch.setattr(
            scheduler_mod, "run_job", lambda j, **kw: (True, "doc", "done", None))
        monkeypatch.setattr(scheduler_mod, "_deliver_result", lambda j, content, **kw: None)

        assert scheduler_mod.resume_background_continuations() == 1

    def test_crash_during_first_continuation_preserves_the_rest(self, monkeypatch):
        """P1: records are claimed one at a time — a failed first run keeps its siblings."""
        self._write_record(process_id="proc_first")
        self._write_record(process_id="proc_second")
        monkeypatch.setattr(
            "cron.jobs.get_job", lambda job_id: _job(background_continuation=True))
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        ran = []
        calls = {"n": 0}

        def _flaky(job, record, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("provider blew up mid-continuation")
            ran.append(record["process_id"])
            return True

        monkeypatch.setattr(scheduler_mod, "_run_cron_continuation", _flaky)

        assert scheduler_mod.resume_background_continuations() == 1
        assert ran == ["proc_second"]
        assert cont.claim_pending() == []

    def test_continuation_runs_after_due_job_dispatch(self, monkeypatch):
        """P1: continuations execute outside the tick lock — due jobs dispatch first."""
        events: list = []
        self._write_record(process_id="proc_ordered")
        monkeypatch.setattr(
            "cron.jobs.get_job", lambda job_id: _job(background_continuation=True))
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        monkeypatch.setattr(
            scheduler_mod, "run_job",
            lambda j, **kw: (events.append("cont"), (True, "doc", "done", None))[1])
        monkeypatch.setattr(scheduler_mod, "_deliver_result", lambda j, content, **kw: None)

        due = _job()
        due["id"] = "due-job-1"
        monkeypatch.setattr(scheduler_mod, "get_due_jobs", lambda: [due])
        monkeypatch.setattr(scheduler_mod, "advance_next_runs", lambda ids: None)
        monkeypatch.setattr(scheduler_mod, "_resolve_max_parallel_workers", lambda: 1)
        monkeypatch.setattr(
            scheduler_mod, "_process_due_job",
            lambda job, adapters, loop, verbose: events.append("due") or True)
        monkeypatch.setattr(scheduler_mod, "_sweep_stale_inflight_for_tick", lambda jobs: None)
        monkeypatch.setattr(scheduler_mod, "_sweep_mcp_orphans", lambda: None)

        with (
            patch.object(scheduler_mod, "_maybe_reap_dead_owners", lambda: None),
            patch.object(scheduler_mod, "_maybe_run_worktree_maintenance", lambda: None),
            patch("tools.mcp_tool_lifecycle._kill_orphaned_mcp_children", lambda: None),
        ):
            executed = scheduler_mod.tick(verbose=False)

        assert events == ["due", "cont"]
        assert executed == 2
        assert cont.claim_pending() == []
