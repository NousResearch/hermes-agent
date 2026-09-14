"""Per-job cron script timeout override (``script_timeout_seconds``).

WHY THIS EXISTS
---------------
``cron/jobs.json`` had no timeout field at all, and ``cron/scheduler_script._get_script_timeout()``
resolved HERMES_CRON_SCRIPT_TIMEOUT -> ``cron.script_timeout_seconds`` -> 3600 with NO job argument.
So the only way to give ONE long-running script job (a multi-hour report, a full-market ingest) more
wall clock than the profile global was to raise the global — which raised the budget for every other
script job in that profile at the same time. Acceptance: the job's own declared timeout wins, and
nothing else about resolution changes.

WHAT IS PINNED HERE
-------------------
* Every rung of the chain, most specific first: job -> test seam -> env -> config -> default.
* A stored-but-unusable job value (0, negative, unparsable, wrong type) WARNs and falls through
  instead of wedging the job — a hand-edited jobs.json can never make a job un-runnable.
* A stored valid value really bounds a live script run: a script sleeping past a 2s per-job budget
  is killed and reported as ``Script timed out after 2s: ...``.
* create_job/update_job validation refuses 0 / -5 / "abc" BEFORE storing and accepts "11700".

Resolution never touches the operator's real config.yaml or env: ``resolution_env`` monkeypatches
``cron.scheduler.load_config`` and clears ``HERMES_CRON_SCRIPT_TIMEOUT``, so a rung can be isolated
by setting exactly one input.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

# cron.scheduler._DEFAULT_SCRIPT_TIMEOUT — restated so the tests fail loudly if it moves.
DEFAULT_SCRIPT_TIMEOUT = 3600


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment (same shape as tests/cron/test_cron_script.py)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Clear cached module-level paths
    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


@pytest.fixture
def resolution_env(cron_env, monkeypatch):
    """Deterministic timeout-resolution inputs; returns the mutable config dict.

    Seam pinned to the default (so it is inert), env var removed, ``load_config()`` stubbed to an
    empty dict — the caller fills in exactly the rungs it wants to exercise.
    """
    import cron.scheduler as sched

    monkeypatch.setattr(sched, "_SCRIPT_TIMEOUT", sched._DEFAULT_SCRIPT_TIMEOUT)
    monkeypatch.delenv("HERMES_CRON_SCRIPT_TIMEOUT", raising=False)
    config: dict = {}
    monkeypatch.setattr(sched, "load_config", lambda: config)
    return config


def _timeout(job=None) -> int:
    from cron.scheduler_script import _get_script_timeout

    return _get_script_timeout(job)


def _write_sleeper(cron_env: Path, seconds: int = 30, name: str = "sleeper.py") -> Path:
    script = cron_env / "scripts" / name
    script.write_text(f"import time\ntime.sleep({seconds})\n", encoding="utf-8")
    return script


# ---------------------------------------------------------------------------
# Precedence: job -> seam -> env -> config -> default
# ---------------------------------------------------------------------------


class TestResolutionPrecedence:
    def test_default_when_nothing_is_set(self, resolution_env):
        assert _timeout() == DEFAULT_SCRIPT_TIMEOUT

    def test_config_rung(self, resolution_env):
        resolution_env["cron"] = {"script_timeout_seconds": 111}
        assert _timeout() == 111

    def test_env_beats_config(self, resolution_env, monkeypatch):
        resolution_env["cron"] = {"script_timeout_seconds": 111}
        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "222")
        assert _timeout() == 222

    def test_seam_beats_env_and_config(self, resolution_env, monkeypatch):
        import cron.scheduler as sched

        resolution_env["cron"] = {"script_timeout_seconds": 111}
        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "222")
        monkeypatch.setattr(sched, "_SCRIPT_TIMEOUT", 333)
        assert _timeout() == 333

    def test_job_beats_every_other_rung(self, resolution_env, monkeypatch):
        import cron.scheduler as sched

        resolution_env["cron"] = {"script_timeout_seconds": 111}
        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "222")
        monkeypatch.setattr(sched, "_SCRIPT_TIMEOUT", 333)
        assert _timeout({"id": "slow", "script_timeout_seconds": 444}) == 444

    def test_job_beats_env_and_config_without_the_seam(self, resolution_env, monkeypatch):
        resolution_env["cron"] = {"script_timeout_seconds": 111}
        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "222")
        assert _timeout({"id": "slow", "script_timeout_seconds": 444}) == 444

    def test_job_value_may_be_a_numeric_string(self, resolution_env):
        # The tool/CLI pass the raw argparse string through; "11700" is the real-world case.
        assert _timeout({"script_timeout_seconds": "11700"}) == 11700

    def test_job_without_the_field_falls_through(self, resolution_env, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "222")
        assert _timeout({"id": "slow"}) == 222

    @pytest.mark.parametrize("value", [0, -5, "0", "abc", "1h", "", [], {"a": 1}])
    def test_unusable_job_value_warns_and_falls_through(self, resolution_env, monkeypatch, caplog, value):
        """A bad stored value must never wedge a job: resolution warns and keeps going."""
        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "222")
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            assert _timeout({"id": "small-window-job", "script_timeout_seconds": value}) == 222
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("script_timeout_seconds" in message for message in warnings), warnings
        assert any("small-window-job" in message for message in warnings), warnings

    def test_unusable_job_value_with_no_other_rung_uses_the_default(self, resolution_env):
        assert _timeout({"script_timeout_seconds": "abc"}) == DEFAULT_SCRIPT_TIMEOUT

    def test_non_mapping_job_is_ignored(self, resolution_env):
        # Defensive: callers that hand over a bare path/None must not crash resolution.
        assert _timeout("sleeper.py") == DEFAULT_SCRIPT_TIMEOUT


# ---------------------------------------------------------------------------
# The stored value really bounds a run
# ---------------------------------------------------------------------------


class TestStoredValueBoundsARealRun:
    def test_job_budget_kills_a_longer_script(self, cron_env, resolution_env, monkeypatch):
        """2s per-job budget, 600s profile budget, 30s script -> the JOB value must win."""
        import cron.scheduler_script as sched_script

        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "600")
        script = _write_sleeper(cron_env, seconds=30)

        started = time.monotonic()
        ok, output = sched_script._run_job_script(
            str(script), job={"id": "slow", "script_timeout_seconds": 2})
        elapsed = time.monotonic() - started

        assert ok is False, f"the job's 2s budget did not bound the run: {output!r}"
        assert output.startswith("Script timed out after 2s:"), output
        assert elapsed < 20, f"script ran for {elapsed:.1f}s past a 2s per-job budget"

    def test_job_without_the_field_still_uses_the_profile_chain(self, cron_env, resolution_env, monkeypatch):
        """The acceptance criterion's other half: nothing else changed for jobs that don't opt in."""
        import cron.scheduler_script as sched_script

        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "1")
        script = _write_sleeper(cron_env, seconds=30, name="sleeper_no_override.py")

        ok, output = sched_script._run_job_script(str(script), job={"id": "plain"})

        assert ok is False
        assert output.startswith("Script timed out after 1s:"), output


# ---------------------------------------------------------------------------
# jobs.py create/update validation
# ---------------------------------------------------------------------------


class TestCreateUpdateValidation:
    def test_create_accepts_numeric_string_and_persists_int(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(prompt="Nightly", schedule="every 1h", script_timeout_seconds="11700")

        assert job["script_timeout_seconds"] == 11700
        assert isinstance(job["script_timeout_seconds"], int)
        assert get_job(job["id"])["script_timeout_seconds"] == 11700
        # Canonical on-disk shape is {"jobs": [...]}; the stored value is the normalized INT, not
        # the string the caller passed.
        stored = json.loads((cron_env / "cron" / "jobs.json").read_text(encoding="utf-8"))
        assert stored["jobs"][0]["script_timeout_seconds"] == 11700

    def test_create_accepts_int(self, cron_env):
        from cron.jobs import create_job

        assert create_job(prompt="Nightly", schedule="every 1h",
                          script_timeout_seconds=900)["script_timeout_seconds"] == 900

    def test_create_omits_the_field_when_unset(self, cron_env):
        from cron.jobs import create_job

        assert create_job(prompt="Nightly", schedule="every 1h").get("script_timeout_seconds") is None

    # "" is NOT here on purpose: an empty string means "no override" (clear), same as None.
    @pytest.mark.parametrize("value", [0, -5, "abc", "0", True, []])
    def test_create_rejects_before_storing(self, cron_env, value):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="script_timeout_seconds"):
            create_job(prompt="Nightly", schedule="every 1h", script_timeout_seconds=value)
        # Refused BEFORE storing: nothing was written at all.
        assert not (cron_env / "cron" / "jobs.json").exists()

    def test_create_treats_falsey_empty_as_no_override(self, cron_env):
        """``None``/``False``/``""`` clear rather than meaning "zero seconds" (sibling convention)."""
        from cron.jobs import create_job

        assert create_job(prompt="A", schedule="every 1h",
                          script_timeout_seconds=False).get("script_timeout_seconds") is None
        assert create_job(prompt="B", schedule="every 1h",
                          script_timeout_seconds="").get("script_timeout_seconds") is None

    def test_update_sets_and_clears(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="Nightly", schedule="every 1h")
        updated = update_job(job["id"], {"script_timeout_seconds": 900})
        assert updated["script_timeout_seconds"] == 900

        cleared = update_job(job["id"], {"script_timeout_seconds": ""})
        assert cleared.get("script_timeout_seconds") is None

    def test_update_rejects_junk_and_keeps_the_previous_value(self, cron_env):
        from cron.jobs import create_job, get_job, update_job

        job = create_job(prompt="Nightly", schedule="every 1h", script_timeout_seconds=900)

        for bad in (0, -5, "abc"):
            with pytest.raises(ValueError, match="script_timeout_seconds"):
                update_job(job["id"], {"script_timeout_seconds": bad})

        assert get_job(job["id"])["script_timeout_seconds"] == 900

    def test_stored_value_is_what_resolution_reads(self, cron_env, resolution_env, monkeypatch):
        """End-to-end: create -> jobs.json -> _get_script_timeout beats env+config for that job."""
        from cron.jobs import create_job, get_job

        resolution_env["cron"] = {"script_timeout_seconds": 111}
        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "222")

        job = create_job(prompt="Slow report", schedule="every 1h",
                         script_timeout_seconds="11700")

        assert _timeout(get_job(job["id"])) == 11700
        # ...and a job that did NOT opt in still follows the profile chain (env rung here).
        other = create_job(prompt="Plain", schedule="every 1h")
        assert _timeout(get_job(other["id"])) == 222


# ---------------------------------------------------------------------------
# The cronjob tool lane the CLI drives
# ---------------------------------------------------------------------------


class TestCronjobToolLane:
    def test_tool_create_stores_and_echoes_the_override(self, cron_env):
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create", prompt="Slow report", schedule="every 1h",
            script_timeout_seconds="600"))

        assert result["success"] is True
        assert result["job"]["script_timeout_seconds"] == 600

    def test_tool_update_sets_and_clears(self, cron_env):
        from tools.cronjob_tools import cronjob

        created = json.loads(cronjob(action="create", prompt="Slow report", schedule="every 1h"))
        job_id = created["job_id"]

        updated = json.loads(cronjob(action="update", job_id=job_id, script_timeout_seconds="600"))
        assert updated["success"] is True
        assert updated["job"]["script_timeout_seconds"] == 600

        cleared = json.loads(cronjob(action="update", job_id=job_id, script_timeout_seconds=""))
        assert cleared["success"] is True
        assert cleared["job"].get("script_timeout_seconds") is None

    def test_per_job_timeout_is_not_exposed_to_models(self, cron_env):
        """CLI-only lane, mirroring reasoning_effort (see tests/cron/test_cron_reasoning_effort.py):
        a model must not be able to extend its own script budget."""
        from tools.cronjob_tools import CRONJOB_SCHEMA

        props = CRONJOB_SCHEMA["parameters"]["properties"]
        assert "script_timeout_seconds" not in props

    def test_tool_update_rejects_zero_without_changing_the_job(self, cron_env):
        from cron.jobs import get_job
        from tools.cronjob_tools import cronjob

        created = json.loads(cronjob(action="create", prompt="Slow report", schedule="every 1h",
                                     script_timeout_seconds="600"))
        job_id = created["job_id"]

        result = json.loads(cronjob(action="update", job_id=job_id, script_timeout_seconds=0))

        assert result["success"] is False
        assert "script_timeout_seconds" in result["error"]
        assert get_job(job_id)["script_timeout_seconds"] == 600
