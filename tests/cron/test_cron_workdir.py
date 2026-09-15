"""Tests for per-job workdir support in cron jobs.

Covers:
  - jobs.create_job: param plumbing, validation, default-None preserved
  - jobs._normalize_workdir: absolute / relative / missing / file-not-dir
  - jobs.update_job: set, clear, re-validate
  - tools.cronjob_tools.cronjob: create + update JSON round-trip, schema
    includes workdir, _format_job exposes it when set
  - scheduler.tick(): partitions workdir jobs off the thread pool, restores
    TERMINAL_CWD in finally, honours the env override during run_job
  - scheduler.run_job() no_agent path: workdir becomes the script
    subprocess's cwd WITHOUT mutating the process-global cwd (no bleed into
    concurrently running parallel-pool jobs)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate cron job storage into a temp dir so tests don't stomp on real jobs."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# ---------------------------------------------------------------------------
# jobs._normalize_workdir
# ---------------------------------------------------------------------------

class TestNormalizeWorkdir:
    def test_none_returns_none(self):
        from cron.jobs import _normalize_workdir
        assert _normalize_workdir(None) is None

    def test_empty_string_returns_none(self):
        from cron.jobs import _normalize_workdir
        assert _normalize_workdir("") is None
        assert _normalize_workdir("   ") is None

    def test_absolute_existing_dir_returns_resolved_str(self, tmp_path):
        from cron.jobs import _normalize_workdir
        result = _normalize_workdir(str(tmp_path))
        assert result == str(tmp_path.resolve())

    def test_tilde_expands(self, tmp_path, monkeypatch):
        from cron.jobs import _normalize_workdir
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _normalize_workdir("~")
        assert result == str(tmp_path.resolve())

    def test_relative_path_rejected(self):
        from cron.jobs import _normalize_workdir
        with pytest.raises(ValueError, match="absolute path"):
            _normalize_workdir("some/relative/path")

    def test_missing_dir_rejected(self, tmp_path):
        from cron.jobs import _normalize_workdir
        missing = tmp_path / "does-not-exist"
        with pytest.raises(ValueError, match="does not exist"):
            _normalize_workdir(str(missing))

    def test_file_not_dir_rejected(self, tmp_path):
        from cron.jobs import _normalize_workdir
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="not a directory"):
            _normalize_workdir(str(f))


# ---------------------------------------------------------------------------
# jobs.create_job and update_job
# ---------------------------------------------------------------------------

class TestCreateJobWorkdir:
    def test_workdir_stored_when_set(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job
        job = create_job(
            prompt="hello",
            schedule="every 1h",
            workdir=str(tmp_cron_dir),
        )
        stored = get_job(job["id"])
        assert stored["workdir"] == str(tmp_cron_dir.resolve())


    def test_create_rejects_invalid_workdir(self, tmp_cron_dir):
        from cron.jobs import create_job
        with pytest.raises(ValueError):
            create_job(
                prompt="hello",
                schedule="every 1h",
                workdir="not/absolute",
            )


class TestUpdateJobWorkdir:
    def test_set_workdir_via_update(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(prompt="x", schedule="every 1h")
        update_job(job["id"], {"workdir": str(tmp_cron_dir)})
        assert get_job(job["id"])["workdir"] == str(tmp_cron_dir.resolve())

    def test_clear_workdir_with_none(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(
            prompt="x", schedule="every 1h", workdir=str(tmp_cron_dir)
        )
        update_job(job["id"], {"workdir": None})
        assert get_job(job["id"])["workdir"] is None


    def test_update_rejects_invalid_workdir(self, tmp_cron_dir):
        from cron.jobs import create_job, update_job
        job = create_job(prompt="x", schedule="every 1h")
        with pytest.raises(ValueError):
            update_job(job["id"], {"workdir": "nope/relative"})


# ---------------------------------------------------------------------------
# tools.cronjob_tools: end-to-end JSON round-trip
# ---------------------------------------------------------------------------

class TestCronjobToolWorkdir:


    def test_schema_advertises_workdir(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA
        assert "workdir" in CRONJOB_SCHEMA["parameters"]["properties"]
        desc = CRONJOB_SCHEMA["parameters"]["properties"]["workdir"]["description"]
        assert "absolute" in desc.lower()


# ---------------------------------------------------------------------------
# scheduler.tick(): workdir jobs use the parallel execution lane
# ---------------------------------------------------------------------------

class TestTickWorkdirPartition:
    """Workdir is per execution, so it must not force a global serial lane."""

    def test_workdir_jobs_overlap_on_parallel_pool(self, tmp_path, monkeypatch):
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery
        import threading

        workdir_a = tmp_path / "a"
        workdir_b = tmp_path / "b"
        workdir_a.mkdir()
        workdir_b.mkdir()
        jobs = [
            {"id": "a", "name": "A", "workdir": str(workdir_a)},
            {"id": "b", "name": "B", "workdir": str(workdir_b)},
        ]
        monkeypatch.setattr(sched, "get_due_jobs", lambda: jobs)
        monkeypatch.setattr(sched, "claim_job_for_fire", lambda *_a, **_kw: True)

        barrier = threading.Barrier(2, timeout=5)
        calls: list[tuple[str, str]] = []
        calls_lock = threading.Lock()

        def fake_run_job(job, *, defer_agent_teardown=None, **_kw):
            with calls_lock:
                calls.append((job["id"], threading.current_thread().name))
            barrier.wait()
            return True, "output", "response", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda _jid, _o: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        assert sched.tick(verbose=False, sync=True) == 2
        assert {job_id for job_id, _thread in calls} == {"a", "b"}
        assert all(thread.startswith("cron-parallel") for _job, thread in calls)


# ---------------------------------------------------------------------------
# scheduler.run_job: per-task cwd + skip_context_files wiring
# ---------------------------------------------------------------------------

class TestRunJobTerminalCwd:
    """
    run_job binds workdir to its unique task CWD without mutating ambient
    TERMINAL_CWD, and clears the task record in finally — even on error.
    AIAgent is stubbed so no real API call happens.
    """

    @staticmethod
    def _install_stubs(monkeypatch, observed: dict):
        """Patch enough of run_job's deps that it executes without real creds."""
        import os
        import sys
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["skip_context_files"] = kwargs.get("skip_context_files")
                observed["load_soul_identity"] = kwargs.get("load_soul_identity")
                observed["terminal_cwd_during_init"] = os.environ.get(
                    "TERMINAL_CWD", "_UNSET_"
                )

            def run_conversation(self, *_a, task_id=None, **_kw):
                from tools.terminal_tool import get_session_cwd

                observed["task_id"] = task_id
                observed["task_cwd_during_run"] = get_session_cwd(task_id)
                observed["terminal_cwd_during_run"] = os.environ.get(
                    "TERMINAL_CWD", "_UNSET_"
                )
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        # Bypass the real provider resolver — it reads ~/.hermes and credentials.
        from hermes_cli import runtime_provider as _rtp
        monkeypatch.setattr(
            _rtp,
            "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "test",
                "api_key": "k",
                "base_url": "http://test.local",
                "api_mode": "chat_completions",
            },
        )

        # Stub scheduler helpers that would otherwise hit the filesystem / config.
        monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None, **kw: "hi")
        monkeypatch.setattr(sched_delivery, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
        # Unlimited inactivity so the poll loop returns immediately.
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        # run_job calls load_dotenv(~/.hermes/.env, override=True), which will
        # happily clobber TERMINAL_CWD out from under us if the real user .env
        # has TERMINAL_CWD set (common on dev boxes).  Stub it out.
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)


    def test_no_workdir_leaves_terminal_cwd_untouched(self, monkeypatch):
        """When workdir is absent, run_job must not touch TERMINAL_CWD at all —
        whatever value was present before the call should be present after.

        We don't assert on the *content* of TERMINAL_CWD (other tests in the
        same xdist worker may leave it set to something like '.'); we just
        check it's unchanged by run_job.
        """
        import os
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery

        # Pin TERMINAL_CWD to a sentinel via monkeypatch so we control both
        # the before-value and the after-value regardless of cross-test state.
        monkeypatch.setenv("TERMINAL_CWD", "/cron-test-sentinel")
        before = os.environ["TERMINAL_CWD"]

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        job = {
            "id": "xyz",
            "name": "no-wd-job",
            "workdir": None,
            "schedule_display": "manual",
        }

        success, *_ = sched.run_job(job)
        assert success is True

        # Feature is OFF — skip_context_files stays True.
        assert observed["skip_context_files"] is True
        # Cron still forces SOUL.md identity even when cwd context files stay off.
        assert observed["load_soul_identity"] is True
        # TERMINAL_CWD saw the same value during init as it had before.
        assert observed["terminal_cwd_during_init"] == before
        # And after run_job completes, it's still the sentinel (nothing
        # overwrote or cleared it).
        assert os.environ["TERMINAL_CWD"] == before

    def test_workdir_is_bound_to_unique_task_without_mutating_process_env(
        self, monkeypatch, tmp_path
    ):
        import os
        import cron.scheduler as sched
        from cron import scheduler_delivery as sched_delivery
        from tools.terminal_tool import get_session_cwd

        baseline = str(tmp_path / "baseline")
        workdir = tmp_path / "project"
        (tmp_path / "baseline").mkdir()
        workdir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", baseline)

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)
        success, *_ = sched.run_job(
            {
                "id": "cwd-bound",
                "name": "cwd-bound",
                "workdir": str(workdir),
                "schedule_display": "manual",
            }
        )

        assert success is True
        assert observed["skip_context_files"] is False
        assert observed["task_id"].startswith("cron:cwd-bound:")
        assert observed["task_cwd_during_run"] == str(workdir)
        assert observed["terminal_cwd_during_run"] == baseline
        assert os.environ["TERMINAL_CWD"] == baseline
        assert get_session_cwd(observed["task_id"]) is None

    def test_agent_prerun_script_receives_configured_workdir(
        self, monkeypatch, tmp_path
    ):
        import cron.scheduler as sched

        workdir = tmp_path / "project"
        workdir.mkdir()
        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        def run_script(job, script_path, workdir=None, cancel_event=None):
            observed["script_workdir"] = workdir
            return True, '{"wakeAgent": false}'

        monkeypatch.setattr(
            sched, "_run_job_script_with_claim_heartbeat", run_script
        )
        success, *_ = sched.run_job(
            {
                "id": "agent-script-workdir",
                "name": "agent-script-workdir",
                "prompt": "Review the project.",
                "script": "collect.py",
                "workdir": str(workdir),
                "schedule_display": "manual",
            }
        )

        assert success is True
        assert observed["script_workdir"] == str(workdir)


# ---------------------------------------------------------------------------
# scheduler.run_job() no_agent path: workdir → subprocess cwd, no process
# cwd mutation (regression for cross-job cwd bleed between the sequential
# and parallel pools)
# ---------------------------------------------------------------------------


class TestNoAgentWorkdir:
    """A no_agent job's workdir must apply to the script subprocess only.

    The old implementation used a process-global ``os.chdir()`` outside
    ``_terminal_cwd_lock`` — and ``_run_job_script`` hardcoded the subprocess
    cwd to the script's directory anyway, so the chdir bled the workdir into
    concurrently running parallel-pool jobs while never actually reaching the
    script.  The workdir must instead be passed to ``subprocess.run(cwd=...)``.
    """

    @pytest.fixture()
    def hermes_home(self, tmp_path, monkeypatch):
        """Isolate HERMES_HOME so the scripts dir resolves under tmp_path."""
        home = tmp_path / ".hermes"
        (home / "scripts").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    @staticmethod
    def _job(workdir: str) -> dict:
        return {
            "id": "no-agent-wd",
            "name": "no-agent-wd",
            "no_agent": True,
            "script": "whereami.py",
            "workdir": workdir,
        }

    def test_workdir_is_script_subprocess_cwd(self, hermes_home, tmp_path):
        """The script actually runs with the configured workdir as its cwd."""
        from cron.scheduler import run_job

        workdir = tmp_path / "wd"
        workdir.mkdir()
        (hermes_home / "scripts" / "whereami.py").write_text(
            "import os\nprint(os.getcwd())\n"
        )

        success, _doc, final_response, error = run_job(self._job(str(workdir)))

        assert success is True, error
        assert os.path.realpath(final_response.strip()) == os.path.realpath(
            str(workdir)
        )

    def test_workdir_does_not_mutate_process_cwd(self, hermes_home, tmp_path):
        """While a workdir job's script runs, the process cwd stays put.

        Reproduces the bleed: with the old os.chdir() implementation, any
        concurrent parallel-pool job observed the workdir as its cwd for the
        whole script run.
        """
        from cron.scheduler import run_job

        workdir = tmp_path / "wd"
        workdir.mkdir()
        marker = workdir / "started.marker"
        (hermes_home / "scripts" / "whereami.py").write_text(
            "import pathlib, time\n"
            f"pathlib.Path({str(marker)!r}).write_text('1')\n"
            "time.sleep(1.0)\n"
            "print('done')\n"
        )

        original_cwd = os.getcwd()
        observed: dict[str, object] = {}

        def _run():
            observed["result"] = run_job(self._job(str(workdir)))

        worker = threading.Thread(target=_run, name="no-agent-workdir-test")
        worker.start()
        try:
            deadline = time.time() + 10
            while not marker.exists() and time.time() < deadline:
                time.sleep(0.01)
            assert marker.exists(), "script never started"
            # The workdir job's script is running right now — a concurrent
            # job must still see the original process cwd.
            assert os.getcwd() == original_cwd
        finally:
            worker.join(timeout=15)

        assert not worker.is_alive()
        assert os.getcwd() == original_cwd
        assert observed["result"][0] is True

    def test_missing_workdir_warns_and_falls_back_to_script_dir(
        self, hermes_home, tmp_path, caplog
    ):
        """A workdir that vanished on disk is dropped with a warning, same as
        the agent path — the script runs in its own directory."""
        from cron.scheduler import run_job

        (hermes_home / "scripts" / "whereami.py").write_text(
            "import os\nprint(os.getcwd())\n"
        )
        missing = str(tmp_path / "gone")

        with caplog.at_level(logging.WARNING):
            success, _doc, final_response, error = run_job(self._job(missing))

        assert success is True, error
        assert "no longer exists" in caplog.text
        assert os.path.realpath(final_response.strip()) == os.path.realpath(
            str(hermes_home / "scripts")
        )

    def test_no_workdir_keeps_script_dir_cwd(self, hermes_home):
        """Without a workdir the script still runs in its own directory."""
        from cron.scheduler import run_job

        (hermes_home / "scripts" / "whereami.py").write_text(
            "import os\nprint(os.getcwd())\n"
        )
        job = self._job("")
        del job["workdir"]

        success, _doc, final_response, error = run_job(job)

        assert success is True, error
        assert os.path.realpath(final_response.strip()) == os.path.realpath(
            str(hermes_home / "scripts")
        )


def test_build_job_prompt_inline_script_receives_configured_workdir(monkeypatch, tmp_path):
    """Callers that skip the wake-gate (no cached ``prerun_script``) run the script inline from
    ``_build_job_prompt``; that path must honour the job's workdir too."""
    from cron import scheduler_prompt, scheduler_script

    workdir = tmp_path / "project"
    workdir.mkdir()
    observed: dict = {}

    def run_script(script_path, workdir=None, cancel_event=None):
        observed["script_workdir"] = workdir
        return True, "collected data"

    monkeypatch.setattr(scheduler_script, "_run_job_script", run_script)
    prompt = scheduler_prompt._build_job_prompt(
        {"id": "inline", "name": "inline", "prompt": "Review.", "script": "collect.py",
         "workdir": str(workdir)})

    assert observed["script_workdir"] == str(workdir)
    assert "collected data" in prompt
