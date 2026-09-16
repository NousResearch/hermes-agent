"""Regression tests for the Civic Assure scheduler-state save boundary."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import cron.jobs as jobs
import cron.scheduler as scheduler
import agent.chat_completion_helpers as chat_helpers
import tools.delegate_tool as delegate_tool


JOB_ID = "406ba6820205"


def _canonical_job():
    return {
        "id": JOB_ID,
        "name": "Civic Assure Issue-to-PR Maintainer",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-09-16T10:05:00+00:00",
        "fire_claim": None,
        "run_claim": None,
        "repeat": {"times": None, "completed": 0},
    }


def test_target_scheduler_state_save_requires_scheduler_owned_context(monkeypatch, tmp_path):
    """A direct caller cannot fabricate a target fire claim via save_jobs helpers."""
    profile = tmp_path / "civic-assure-maintainer"
    cron_dir = profile / "cron"
    cron_dir.mkdir(parents=True)
    jobs_file = cron_dir / "jobs.json"
    current = _canonical_job()
    jobs_file.write_text(json.dumps({"jobs": [current]}), encoding="utf-8")

    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))
    monkeypatch.setattr(
        jobs,
        "_current_cron_store",
        lambda: SimpleNamespace(jobs_file=Path(jobs_file)),
    )

    proposed = dict(current)
    proposed["fire_claim"] = {
        "at": "2026-09-16T10:00:00+00:00",
        "by": "fabricated-owner",
    }

    assert jobs._civic_assure_hermes_scheduler_control_save([proposed]) is False


def test_save_jobs_rejects_unscoped_target_scheduler_state(monkeypatch, tmp_path):
    """The public persistence boundary rejects an unscoped target claim."""
    profile = tmp_path / "civic-assure-maintainer"
    cron_dir = profile / "cron"
    cron_dir.mkdir(parents=True)
    jobs_file = cron_dir / "jobs.json"
    current = _canonical_job()
    jobs_file.write_text(json.dumps({"jobs": [current]}), encoding="utf-8")

    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))
    monkeypatch.setattr(
        jobs,
        "_current_cron_store",
        lambda: SimpleNamespace(jobs_file=Path(jobs_file)),
    )
    monkeypatch.setattr(jobs, "_jobs_lock", lambda: nullcontext())
    monkeypatch.setattr(jobs, "_save_jobs_unlocked", lambda *_a, **_kw: None)

    proposed = dict(current)
    proposed["fire_claim"] = {
        "at": "2026-09-16T10:00:00+00:00",
        "by": "fabricated-owner",
    }

    with pytest.raises(RuntimeError, match="internal scheduler capability"):
        jobs.save_jobs([proposed])


def test_target_scheduler_state_save_accepts_scheduler_owned_context(monkeypatch, tmp_path):
    """The internal scheduler capability still admits its own transition."""
    profile = tmp_path / "civic-assure-maintainer"
    cron_dir = profile / "cron"
    cron_dir.mkdir(parents=True)
    jobs_file = cron_dir / "jobs.json"
    current = _canonical_job()
    jobs_file.write_text(json.dumps({"jobs": [current]}), encoding="utf-8")

    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))
    monkeypatch.setattr(
        jobs,
        "_current_cron_store",
        lambda: SimpleNamespace(jobs_file=Path(jobs_file)),
    )

    proposed = dict(current)
    proposed["fire_claim"] = {
        "at": "2026-09-16T10:00:00+00:00",
        "by": "scheduler-owner",
    }

    with jobs._civic_assure_scheduler_save_context("fire_claim"):
        assert jobs._civic_assure_hermes_scheduler_control_save([proposed]) is True


def test_due_hold_filter_ignores_same_id_in_non_maintainer_profile(monkeypatch, tmp_path):
    """A job-ID collision outside the maintainer profile is unrelated work."""
    profile = tmp_path / "ordinary-profile"
    profile.mkdir()
    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))
    due = [{"id": JOB_ID, "name": "ordinary job"}]

    assert scheduler._civic_assure_filter_due_jobs(due) == due


def test_claim_job_for_fire_honors_maintainer_hold(monkeypatch):
    """Direct/manual claims cannot bypass an active maintainer hold."""
    monkeypatch.setattr(jobs, "_civic_assure_native_config_target_profile", lambda: True)
    monkeypatch.setattr(
        scheduler,
        "_civic_assure_maintenance_hold_blocks",
        lambda _job_id: True,
    )
    monkeypatch.setattr(jobs, "_fire_job_lock", lambda *_a, **_kw: nullcontext(True))
    monkeypatch.setattr(jobs, "_jobs_lock", lambda: nullcontext())
    monkeypatch.setattr(jobs, "load_jobs", lambda: [_canonical_job()])
    monkeypatch.setattr(jobs, "save_jobs", lambda *_a, **_kw: None)

    assert jobs.claim_job_for_fire(JOB_ID, force=True) is False


def test_civic_assure_session_id_uses_fire_claim_timestamp():
    """Canonical builtin runs use the claim timestamp, not run start time."""
    job = {
        "id": JOB_ID,
        "fire_claim": {"at": "2026-09-16T10:00:07+00:00", "by": "owner"},
    }

    assert scheduler._civic_assure_session_id_for_job(job) == (
        "cron_406ba6820205_20260916_100007"
    )


def test_governed_summary_dispatch_uses_interruptible_boundary():
    """Iteration summaries must not call a provider around the boundary."""
    calls = []

    class Agent:
        _civic_assure_model_request_binding = {"policy": {"max_output_tokens": 16_000}}
        _current_api_request_id = "prior-request"

        def _interruptible_api_call(self, request):
            calls.append((self._current_api_request_id, request))
            return "summary"

    agent = Agent()

    assert chat_helpers._execute_summary_request(
        agent,
        {"max_tokens": 16_000},
        lambda _request: "unbounded callback",
        summary_api_request_id="iteration-summary:test",
        retry_count=1,
    ) == "summary"
    assert calls == [("iteration-summary:test:1", {"max_tokens": 16_000})]
    assert agent._current_api_request_id == "prior-request"


def test_review_child_tool_surface_filters_actual_schemas():
    """Review mode must not leave write/execute schemas on the child."""
    child = SimpleNamespace(
        valid_tool_names=["read_file", "search_files", "write_file"],
        tools=[
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "search_files"}},
            {"type": "function", "function": {"name": "write_file"}},
        ],
    )

    delegate_tool._restrict_review_child_tool_surface(child)

    assert child.valid_tool_names == ["read_file", "search_files"]
    assert [schema["function"]["name"] for schema in child.tools] == [
        "read_file",
        "search_files",
    ]


def test_review_route_prefers_canonical_profile_config(monkeypatch, tmp_path):
    """Maintainer review routing cannot be replaced by user config."""
    profile = tmp_path / "civic-assure-maintainer"
    (profile / "config").mkdir(parents=True)
    (profile / "config" / "cron.yaml").write_text(
        "jobs:\n"
        "  civic-assure-maintainer-queue:\n"
        "    review:\n"
        "      enabled: true\n"
        "      provider: openai-codex\n"
        "      model: gpt-5.6-luna\n"
        "      reasoning_effort: max\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"review": {"provider": "same-provider", "model": "unsafe", "reasoning_effort": "low"}},
    )

    assert delegate_tool._load_review_config() == {
        "enabled": True,
        "provider": "openai-codex",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "max",
    }


def test_model_request_policy_uses_canonical_profile_config(monkeypatch, tmp_path):
    """Profile request limits must override same-named global defaults."""
    profile = tmp_path / "civic-assure-maintainer"
    (profile / "config").mkdir(parents=True)
    (profile / "config" / "automation-telemetry.yaml").write_text(
        "model_request:\n"
        "  max_wall_seconds: 77\n"
        "  max_output_tokens: 1234\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scheduler, "_civic_assure_profile_home", lambda: profile)

    config = scheduler._civic_assure_model_request_config(
        {"model_request": {"max_output_tokens": 9999}}
    )

    assert config["model_request"] == {
        "max_wall_seconds": 77,
        "max_output_tokens": 1234,
    }


def test_model_request_authority_ignores_same_id_in_non_maintainer_profile(monkeypatch, tmp_path):
    """A colliding job ID must not activate maintainer request policy."""
    profile = tmp_path / "ordinary-profile"
    profile.mkdir()
    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))

    assert scheduler._civic_assure_model_request_max_tokens(
        JOB_ID,
        "dgx-spark",
        {"provider": "custom"},
        {},
    ) is None


def test_maintainer_guard_requires_the_active_cron_store(monkeypatch, tmp_path):
    """A profile-name collision must not activate the native maintainer guard."""
    profile = tmp_path / "civic-assure-maintainer"
    profile.mkdir()
    unrelated_store = tmp_path / "other-profile" / "cron" / "jobs.json"
    unrelated_store.parent.mkdir(parents=True)

    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))
    monkeypatch.setattr(
        jobs,
        "_current_cron_store",
        lambda: SimpleNamespace(jobs_file=unrelated_store),
    )

    assert jobs._civic_assure_native_config_target_profile() is False


def test_due_scan_cold_start_authority_is_probe_only(monkeypatch, tmp_path):
    """A consumed cold-start admission must not be retried for normal scans."""
    profile = tmp_path / "civic-assure-maintainer"
    profile.mkdir()
    monkeypatch.setattr(jobs, "_civic_assure_native_config_target_profile", lambda: True)
    monkeypatch.setattr(jobs, "_civic_assure_native_config_profile", lambda: profile)

    def admission_rejected(_profile):
        raise RuntimeError("budget already consumed")

    monkeypatch.setattr(
        jobs,
        "_civic_assure_due_scan_module",
        lambda: (None, None, admission_rejected),
    )

    assert jobs._civic_assure_due_scan_is_cold_start() is False


def test_due_scan_cold_start_authority_is_used_when_admission_is_valid(monkeypatch, tmp_path):
    """The stronger due-scan authority remains available on a real cold start."""
    profile = tmp_path / "civic-assure-maintainer"
    profile.mkdir()
    monkeypatch.setattr(jobs, "_civic_assure_native_config_target_profile", lambda: True)
    monkeypatch.setattr(jobs, "_civic_assure_native_config_profile", lambda: profile)
    monkeypatch.setattr(
        jobs,
        "_civic_assure_due_scan_module",
        lambda: (None, None, lambda _profile: {"admitted": True}),
    )

    assert jobs._civic_assure_due_scan_is_cold_start() is True


def test_due_scan_falls_back_to_scheduler_bookkeeping_after_cold_start(monkeypatch):
    """Routine overdue recovery must not invoke the consumed cold-start gate."""
    now = scheduler.datetime.fromisoformat("2026-09-16T10:00:00+00:00")
    current = {
        "id": JOB_ID,
        "name": "Civic Assure Issue-to-PR Maintainer",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2020-01-01T00:00:00+00:00",
        "fire_claim": None,
        "run_claim": None,
        "last_status": "ok",
        "repeat": {"times": None, "completed": 0},
    }
    saves = []

    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(jobs, "load_jobs", lambda: [dict(current)])
    monkeypatch.setattr(jobs, "_sweep_completed_oneshots", lambda *_a, **_k: False)
    monkeypatch.setattr(jobs, "_civic_assure_native_one_run_blocks_dispatch", lambda _id: False)
    monkeypatch.setattr(jobs, "_civic_assure_native_config_target_profile", lambda: True)
    monkeypatch.setattr(jobs, "_civic_assure_due_scan_is_cold_start", lambda: False)
    monkeypatch.setattr(
        jobs,
        "_civic_assure_due_scan_context",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cold-start gate was used")),
    )
    monkeypatch.setattr(
        jobs,
        "_save_scheduler_jobs",
        lambda rows, **kwargs: saves.append((rows, kwargs)),
    )

    due = jobs._get_due_jobs_locked()

    assert due and due[0]["id"] == JOB_ID
    assert saves and saves[0][1]["operation"] == "due_scan"


def test_scheduler_hold_guard_requires_the_active_cron_store(monkeypatch, tmp_path):
    """The due-job filter must not block a same-ID job in another store."""
    profile = tmp_path / "civic-assure-maintainer"
    profile.mkdir()
    unrelated_store = tmp_path / "other-profile" / "cron" / "jobs.json"
    unrelated_store.parent.mkdir(parents=True)

    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))
    monkeypatch.setattr(
        "cron.jobs._current_cron_store",
        lambda: SimpleNamespace(jobs_file=unrelated_store),
    )

    assert scheduler._civic_assure_profile_is_active() is False


def test_maintainer_review_config_does_not_fall_back_to_global_config(monkeypatch, tmp_path):
    """A missing canonical route must fail closed instead of using user config."""
    profile = tmp_path / "civic-assure-maintainer"
    (profile / "config").mkdir(parents=True)
    monkeypatch.setenv("HERMES_PROFILE_HOME", str(profile))
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "review": {
                "provider": "unsafe-provider",
                "model": "unsafe-model",
                "reasoning_effort": "low",
            }
        },
    )

    assert delegate_tool._load_review_config() == {}


def test_maintainer_boundary_rejects_unwrapped_codex_app_server(monkeypatch):
    """The app-server turn cannot run under the transport request boundary."""
    agent = SimpleNamespace(
        _interruptible_api_call=lambda _request: None,
        _interruptible_streaming_api_call=lambda _request: None,
    )
    job = {
        "id": JOB_ID,
        "execution_id": "execution-1",
        "fire_claim": {"by": "scheduler-owner", "at": "2026-09-16T10:00:00+00:00"},
    }
    monkeypatch.setattr(scheduler, "_civic_assure_profile_is_active", lambda: True)

    with pytest.raises(RuntimeError, match="app-server"):
        scheduler._civic_assure_install_model_request_boundary(
            agent,
            job,
            {"api_mode": "codex_app_server"},
            {},
            "cron_406ba6820205_20260916_100000",
        )


def test_review_dispatch_is_controller_owned(monkeypatch, tmp_path):
    """The parent model cannot grant itself the independent-review capability."""
    parent = SimpleNamespace(_delegate_depth=0)
    monkeypatch.setattr(delegate_tool, "is_spawn_paused", lambda: False)

    result = json.loads(
        delegate_tool.delegate_task(
            goal="Inspect the exact candidate.",
            review=True,
            review_root=str(tmp_path),
            parent_agent=parent,
        )
    )

    assert "controller" in result["error"].lower()


def test_review_flag_rejects_integer_coercion(monkeypatch, tmp_path):
    """A numeric value must not be accepted as the boolean review flag."""
    parent = SimpleNamespace(_delegate_depth=0)
    monkeypatch.setattr(delegate_tool, "is_spawn_paused", lambda: False)

    result = json.loads(
        delegate_tool.delegate_task(
            goal="Inspect the exact candidate.",
            review=1,
            review_root=str(tmp_path),
            parent_agent=parent,
        )
    )

    assert "boolean" in result["error"].lower()
