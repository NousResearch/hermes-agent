"""Tests for monitor-mode cron jobs — cheap source each tick, hash-suppressed agent runs.

A monitor job runs a cheap *monitor source* (``monitor_script`` or
``monitor_url``) on every tick, hashes the exact output bytes, and:

* unchanged output  → suppressed run: NO agent invocation, NO delivery,
  visible in the executions ledger as a silent no-change tick;
* changed output    → a "MONITOR CHANGE DETECTED" block (unified diff of
  old vs new, capped, plus the new output) is injected into the prompt and
  the agent runs normally;
* first run         → always runs the agent (nothing to compare against);
* source failure    → treated as an ERROR (alert delivered), never as a
  change — and the stored hash is NOT updated.

State (`monitor_state.last_output_hash` / `last_changed_at`) lives on the
job record in jobs.json plus a snapshot file, so suppression survives
scheduler restarts.

Inspired by: ChatGPT Work monitor tasks (idea-level, docs-only);
enabler: #80774.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts/snapshots don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.monitor
    importlib.reload(cron.monitor)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


def _write_script(home, name: str, body: str) -> str:
    path = home / "scripts" / name
    path.write_text(body, encoding="utf-8")
    return name


def _install_agent_stubs(monkeypatch, observed: dict):
    """Stub the agent machinery so run_job's LLM path executes without creds.

    ``observed["prompts"]`` collects the prompt each agent run received;
    ``observed["agent_runs"]`` counts real agent invocations.
    """
    import cron.scheduler as sched

    observed.setdefault("prompts", [])
    observed.setdefault("agent_runs", 0)

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_conversation(self, prompt, *_a, **_kw):
            observed["agent_runs"] += 1
            observed["prompts"].append(prompt)
            if observed.get("raise_error"):
                raise RuntimeError(str(observed["raise_error"]))
            return {
                "final_response": observed.get("final_response", "agent done"),
                "messages": [],
            }

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    fake_mod = type(sys)("run_agent")
    fake_mod.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

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

    monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)


# ---------------------------------------------------------------------------
# create_job: data-layer semantics for monitor fields
# ---------------------------------------------------------------------------


def test_create_job_stores_monitor_script(hermes_env):
    from cron.jobs import create_job, get_job

    _write_script(hermes_env, "mon.sh", "echo stable\n")
    job = create_job(
        prompt="React to the change",
        schedule="every 5m",
        monitor_script="mon.sh",
        deliver="local",
    )
    reloaded = get_job(job["id"])
    assert reloaded["monitor_script"] == "mon.sh"
    assert reloaded.get("monitor_url") is None
    assert reloaded.get("monitor_state") is None


def test_create_job_monitor_script_and_url_mutually_exclusive(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="monitor_script and monitor_url"):
        create_job(
            prompt="p",
            schedule="every 5m",
            monitor_script="mon.sh",
            monitor_url="https://example.com/status",
        )


def test_create_job_monitor_rejected_with_no_agent(hermes_env):
    from cron.jobs import create_job

    _write_script(hermes_env, "w.sh", "echo hi\n")
    with pytest.raises(ValueError, match="no_agent"):
        create_job(
            prompt=None,
            schedule="every 5m",
            script="w.sh",
            no_agent=True,
            monitor_script="w.sh",
        )


def test_update_job_rejects_no_agent_on_monitor_job(hermes_env):
    """The create-time monitor×no_agent invariant must hold through the
    update door too — the scheduler's no_agent short-circuit runs before
    the monitor gate, so flipping no_agent=True on a monitor job would
    silently disable the monitor (post-merge audit of #81138)."""
    from cron.jobs import create_job, update_job

    _write_script(hermes_env, "mon.sh", "echo stable\n")
    _write_script(hermes_env, "w.sh", "echo hi\n")
    job = create_job(
        prompt="React to the change",
        schedule="every 5m",
        monitor_script="mon.sh",
        deliver="local",
    )
    with pytest.raises(ValueError, match="no_agent"):
        update_job(job["id"], {"no_agent": True, "script": "w.sh"})


def test_update_job_rejects_adding_monitor_to_no_agent_job(hermes_env):
    from cron.jobs import create_job, update_job

    _write_script(hermes_env, "w.sh", "echo hi\n")
    _write_script(hermes_env, "mon.sh", "echo stable\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="w.sh",
        no_agent=True,
        deliver="local",
    )
    with pytest.raises(ValueError, match="no_agent"):
        update_job(job["id"], {"monitor_script": "mon.sh"})


def test_update_job_rejects_second_monitor_source(hermes_env):
    from cron.jobs import create_job, update_job

    _write_script(hermes_env, "mon.sh", "echo stable\n")
    job = create_job(
        prompt="React",
        schedule="every 5m",
        monitor_script="mon.sh",
        deliver="local",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        update_job(job["id"], {"monitor_url": "https://example.com/status"})


def test_update_job_allows_clearing_monitor_then_no_agent(hermes_env):
    """Clearing the monitor and flipping no_agent in ONE update is valid —
    the invariant is checked on the merged record, not per-field."""
    from cron.jobs import create_job, get_job, update_job

    _write_script(hermes_env, "mon.sh", "echo stable\n")
    _write_script(hermes_env, "w.sh", "echo hi\n")
    job = create_job(
        prompt="React",
        schedule="every 5m",
        monitor_script="mon.sh",
        deliver="local",
    )
    update_job(job["id"], {"monitor_script": "", "no_agent": True, "script": "w.sh"})
    reloaded = get_job(job["id"])
    assert reloaded.get("monitor_script") is None
    assert reloaded["no_agent"] is True


def test_update_job_unrelated_fields_skip_mode_validation(hermes_env):
    """A legacy/odd record must keep accepting updates that don't touch the
    mode fields — the invariant re-check is scoped to changed fields."""
    from cron.jobs import create_job, update_job

    _write_script(hermes_env, "mon.sh", "echo stable\n")
    job = create_job(
        prompt="React",
        schedule="every 5m",
        monitor_script="mon.sh",
        deliver="local",
    )
    updated = update_job(job["id"], {"name": "renamed"})
    assert updated["name"] == "renamed"


# ---------------------------------------------------------------------------
# cron.monitor: hashing + diff unit behavior
# ---------------------------------------------------------------------------


def test_hash_is_exact_bytes(hermes_env):
    from cron.monitor import hash_monitor_output

    assert hash_monitor_output("a\nb") == hash_monitor_output("a\nb")
    # Exact-bytes contract: even whitespace-only differences are changes.
    assert hash_monitor_output("a\nb") != hash_monitor_output("a\nb ")


def test_deferred_monitor_commit_propagates_snapshot_write_failure(
    hermes_env, monkeypatch
):
    from pathlib import Path

    from cron.jobs import create_job, get_job
    from cron.monitor import commit_monitor_state

    job = create_job(prompt="p", schedule="every 5m", deliver="local")

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated snapshot write failure")

    monkeypatch.setattr(Path, "write_text", fail_write)

    with pytest.raises(OSError, match="snapshot write failure"):
        commit_monitor_state(job["id"], "abc123", "state A")

    assert get_job(job["id"]).get("monitor_state") is None


def test_deferred_monitor_commit_rolls_back_snapshot_when_hash_update_fails(
    hermes_env, monkeypatch
):
    import cron.jobs as jobs
    from cron.jobs import create_job, get_job
    from cron.monitor import _read_last_output, commit_monitor_state

    job = create_job(prompt="p", schedule="every 5m", deliver="local")
    commit_monitor_state(job["id"], "a" * 64, "state A")

    def fail_update(*_args, **_kwargs):
        raise OSError("simulated jobs write failure")

    monkeypatch.setattr(jobs, "update_job", fail_update)
    with pytest.raises(OSError, match="jobs write failure"):
        commit_monitor_state(job["id"], "b" * 64, "state B")

    assert _read_last_output(job["id"]) == "state A"
    assert get_job(job["id"])["monitor_state"]["last_output_hash"] == "a" * 64


def test_deferred_monitor_commit_fails_before_write_if_old_snapshot_is_unreadable(
    hermes_env, monkeypatch
):
    from pathlib import Path

    from cron.jobs import create_job, get_job
    from cron.monitor import _read_last_output, commit_monitor_state

    job = create_job(prompt="p", schedule="every 5m", deliver="local")
    commit_monitor_state(job["id"], "a" * 64, "state A")
    real_read_text = Path.read_text

    def fail_snapshot_read(path, *args, **kwargs):
        if path.name == "monitor_last_output.txt":
            raise OSError("simulated snapshot read failure")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_snapshot_read)
    with pytest.raises(OSError, match="snapshot read failure"):
        commit_monitor_state(job["id"], "b" * 64, "state B")
    monkeypatch.setattr(Path, "read_text", real_read_text)

    assert _read_last_output(job["id"]) == "state A"
    assert get_job(job["id"])["monitor_state"]["last_output_hash"] == "a" * 64


def test_after_delivery_policy_records_commit_failure_without_advancing_hash(
    hermes_env, monkeypatch
):
    import cron.monitor as monitor
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    def fail_commit(*_args, **_kwargs):
        raise OSError("simulated commit failure")

    monkeypatch.setattr(monitor, "commit_monitor_state", fail_commit)

    assert sched.run_one_job(job) is False

    stored = get_job(job["id"])
    assert stored.get("monitor_state") is None
    assert stored["last_status"] == "error"


def test_unified_diff_is_capped(hermes_env):
    from cron.monitor import MAX_DIFF_CHARS, build_monitor_diff

    old = "\n".join(f"line {i}" for i in range(5000))
    new = "\n".join(f"LINE {i}" for i in range(5000))
    diff = build_monitor_diff(old, new)
    assert len(diff) <= MAX_DIFF_CHARS + 200  # cap + truncation notice
    assert "truncated" in diff


# ---------------------------------------------------------------------------
# scheduler.run_job: monitor gate behavior
# ---------------------------------------------------------------------------


def _make_monitor_job(hermes_env, script_body: str):
    from cron.jobs import create_job

    _write_script(hermes_env, "mon.sh", script_body)
    return create_job(
        prompt="Summarize what changed",
        schedule="every 5m",
        monitor_script="mon.sh",
        deliver="local",
    )


def test_first_run_always_runs_agent(hermes_env, monkeypatch):
    from cron.scheduler import run_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    success, doc, final, error = run_job(job)
    assert success is True
    assert error is None
    assert observed["agent_runs"] == 1
    # First run: new output is injected as monitor context.
    assert "state A" in observed["prompts"][0]


def test_after_delivery_policy_defers_monitor_hash_commit(hermes_env, monkeypatch):
    from cron.jobs import get_job, update_job
    from cron.scheduler import run_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    pending_commits = []

    success, doc, final, error = run_job(
        job, defer_monitor_commit=pending_commits
    )

    assert success is True
    assert error is None
    assert observed["agent_runs"] == 1
    assert get_job(job["id"]).get("monitor_state") is None
    pending = job.get("_monitor_pending_commit")
    assert pending and pending["new_hash"] and pending["output"] == "state A"
    assert pending_commits == [pending]


def test_after_delivery_policy_commits_after_successful_end_to_end_run(
    hermes_env, monkeypatch
):
    from cron.jobs import get_job, update_job
    from cron.scheduler import run_one_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    assert run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored["last_status"] == "ok"
    assert stored["monitor_state"]["last_output_hash"]


def test_after_delivery_policy_silent_success_commits_without_delivery(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed = {"final_response": "[SILENT]"}
    _install_agent_stubs(monkeypatch, observed)
    deliveries = []
    monkeypatch.setattr(
        sched,
        "_deliver_result",
        lambda *_a, **_kw: deliveries.append(_a[1]) or None,
    )

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored["last_status"] == "ok"
    assert stored["monitor_state"]["last_output_hash"]
    assert deliveries == []


def test_required_delivery_event_rejects_silent_agent_response(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    body = (
        "echo '{\"version\":4,\"events\":{\"event\":"
        "{\"event_id\":\"health-1\",\"requires_delivery\":true}}}'\n"
    )
    job = _make_monitor_job(hermes_env, body)
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed = {"final_response": "[SILENT]"}
    _install_agent_stubs(monkeypatch, observed)
    deliveries = []
    monkeypatch.setattr(
        sched,
        "_deliver_result",
        lambda *_a, **_kw: deliveries.append(_a[1]) or None,
    )

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored["last_status"] == "monitor_retry"
    assert stored.get("monitor_state") is None
    assert deliveries == []


def test_required_delivery_event_rejects_local_only_delivery(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    body = (
        "echo '{\"version\":4,\"events\":{\"event\":"
        "{\"event_id\":\"health-1\",\"requires_delivery\":true}}}'\n"
    )
    job = _make_monitor_job(hermes_env, body)
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed = {"final_response": "health alert"}
    _install_agent_stubs(monkeypatch, observed)
    monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored["last_status"] == "monitor_retry"
    assert stored.get("monitor_state") is None


def test_after_delivery_policy_retry_marker_suppresses_delivery_and_commit(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed = {"final_response": "[MONITOR_RETRY]"}
    _install_agent_stubs(monkeypatch, observed)
    deliveries = []
    monkeypatch.setattr(
        sched,
        "_deliver_result",
        lambda *_a, **_kw: deliveries.append(_a[1]) or None,
    )

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored["last_status"] == "monitor_retry"
    assert stored.get("monitor_state") is None
    assert deliveries == []


def test_after_delivery_policy_retry_then_success_commits_and_suppresses_next_tick(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed = {"final_response": "[MONITOR_RETRY]"}
    _install_agent_stubs(monkeypatch, observed)
    deliveries = []
    monkeypatch.setattr(
        sched,
        "_deliver_result",
        lambda *_a, **_kw: deliveries.append(_a[1]) or None,
    )

    assert sched.run_one_job(job) is True
    assert get_job(job["id"]).get("monitor_state") is None
    assert observed["agent_runs"] == 1

    observed["final_response"] = "verified report"
    assert sched.run_one_job(get_job(job["id"])) is True
    assert get_job(job["id"])["monitor_state"]["last_output_hash"]
    assert observed["agent_runs"] == 2

    assert sched.run_one_job(get_job(job["id"])) is True
    assert observed["agent_runs"] == 2
    assert deliveries == ["verified report"]


def test_plain_cron_treats_monitor_retry_marker_as_normal_output(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import create_job, get_job

    job = create_job(
        prompt="ordinary job",
        schedule="every 5m",
        deliver="local",
    )
    observed = {"final_response": "[MONITOR_RETRY]"}
    _install_agent_stubs(monkeypatch, observed)
    deliveries = []
    monkeypatch.setattr(
        sched,
        "_deliver_result",
        lambda *_a, **_kw: deliveries.append(_a[1]) or None,
    )

    assert sched.run_one_job(job) is True

    assert deliveries == ["[MONITOR_RETRY]"]
    assert get_job(job["id"])["last_status"] == "ok"


def test_default_eager_monitor_treats_retry_marker_as_normal_output(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    observed = {"final_response": "[MONITOR_RETRY]"}
    _install_agent_stubs(monkeypatch, observed)
    deliveries = []
    monkeypatch.setattr(
        sched,
        "_deliver_result",
        lambda *_a, **_kw: deliveries.append(_a[1]) or None,
    )

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert deliveries == ["[MONITOR_RETRY]"]
    assert stored["monitor_state"]["last_output_hash"]


def test_after_delivery_policy_does_not_commit_on_delivery_failure(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    monkeypatch.setattr(
        sched,
        "_deliver_result",
        lambda *_a, **_kw: "simulated delivery failure",
    )

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored.get("monitor_state") is None
    assert stored["last_delivery_error"] == "simulated delivery failure"


def test_after_delivery_policy_does_not_commit_on_empty_message_delivery_exception(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    def fail_delivery(*_a, **_kw):
        raise TimeoutError()

    monkeypatch.setattr(sched, "_deliver_result", fail_delivery)

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored.get("monitor_state") is None
    assert stored["last_delivery_error"] == "TimeoutError"


def test_after_delivery_policy_does_not_commit_when_origin_is_unresolved(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(
        job["id"],
        {"monitor_commit_policy": "after_delivery", "deliver": "origin"},
    )
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored.get("monitor_state") is None


def test_after_delivery_policy_fences_monitor_commit_against_owner_loss(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(
        job["id"],
        {
            "monitor_commit_policy": "after_delivery",
            "fire_claim": {"by": "owner-a", "at": "2026-01-01T00:00:00+00:00"},
        },
    )
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    fence_calls = []

    @contextlib.contextmanager
    def lose_owner_before_commit(_job_id, *, expected_owner):
        fence_calls.append(expected_owner)
        # Saving output and delivery still belong to owner-a. The claim is
        # lost only at the deferred monitor-commit boundary.
        yield len(fence_calls) <= 2

    monkeypatch.setattr(sched, "fire_claim_fence", lose_owner_before_commit)

    assert sched._run_one_job_body(
        job,
        fire_claim_lost=threading.Event(),
        execution_token=object(),
    ) is True

    assert len(fence_calls) >= 3
    assert get_job(job["id"]).get("monitor_state") is None


def test_after_delivery_policy_real_fire_owner_commits_without_nested_fence(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(
        job["id"],
        {
            "monitor_commit_policy": "after_delivery",
            "fire_claim": {"by": "owner-a", "at": "2026-08-20T00:00:00+00:00"},
        },
    )
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    assert sched._run_one_job_body(
        job,
        fire_claim_lost=threading.Event(),
        execution_token=object(),
    ) is True

    stored = get_job(job["id"])
    assert stored["last_status"] == "ok"
    assert stored["monitor_state"]["last_output_hash"]


def test_after_delivery_policy_rechecks_interrupt_inside_commit_fence(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(
        job["id"],
        {
            "monitor_commit_policy": "after_delivery",
            "fire_claim": {"by": "owner-a", "at": "2026-01-01T00:00:00+00:00"},
        },
    )
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    cancellation = threading.Event()
    fence_calls = []

    @contextlib.contextmanager
    def interrupt_at_commit(_job_id, *, expected_owner):
        fence_calls.append(expected_owner)
        if len(fence_calls) == 3:
            cancellation.set()
        yield True

    monkeypatch.setattr(sched, "fire_claim_fence", interrupt_at_commit)

    assert sched._run_one_job_body(
        job,
        fire_claim_lost=cancellation,
        execution_token=object(),
    ) is True

    assert len(fence_calls) >= 3
    assert get_job(job["id"]).get("monitor_state") is None


def test_after_delivery_commit_serializes_against_shutdown_interrupt_flag(
    hermes_env, monkeypatch
):
    import cron.monitor as monitor
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(
        job["id"],
        {
            "monitor_commit_policy": "after_delivery",
            "fire_claim": {"by": "owner-a", "at": "2026-01-01T00:00:00+00:00"},
        },
    )
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    commit_entered = threading.Event()
    allow_commit = threading.Event()
    interrupt_mark_called = threading.Event()

    def slow_commit(*_args, **_kwargs):
        commit_entered.set()
        assert allow_commit.wait(timeout=2)

    def record_mark(_job_id, success, *_args, **_kwargs):
        if success is False:
            interrupt_mark_called.set()
        return True

    monkeypatch.setattr(monitor, "commit_monitor_state", slow_commit)
    monkeypatch.setattr(sched, "mark_job_run", record_mark)

    @contextlib.contextmanager
    def in_memory_fire_fence(*_args, **_kwargs):
        yield True

    monkeypatch.setattr(sched, "fire_claim_fence", in_memory_fire_fence)

    execution_token = object()
    with sched._running_lock:
        sched._running_fire_owners.setdefault(job["id"], {})[execution_token] = (
            "owner-a",
            hermes_env,
        )

    run_thread = threading.Thread(
        target=sched._run_one_job_body,
        kwargs={
            "job": job,
            "fire_claim_lost": threading.Event(),
            "execution_token": execution_token,
        },
    )
    run_thread.start()
    assert commit_entered.wait(timeout=2)

    interrupt_thread = threading.Thread(
        target=sched.mark_running_jobs_interrupted,
        args=("simulated shutdown",),
    )
    interrupt_thread.start()
    interrupt_reached_terminal_mark_before_commit = interrupt_mark_called.wait(
        timeout=0.2
    )

    allow_commit.set()
    run_thread.join(timeout=3)
    interrupt_thread.join(timeout=3)
    with sched._running_lock:
        sched._running_fire_owners.pop(job["id"], None)

    assert not run_thread.is_alive()
    assert not interrupt_thread.is_alive()
    assert interrupt_reached_terminal_mark_before_commit is False


def test_after_delivery_commit_stays_linearized_through_terminal_mark(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)
    execution_token = object()
    protected_at_mark = []

    def record_mark(_job_id, success, *_args, **_kwargs):
        if success:
            protected_at_mark.append(
                execution_token in sched._monitor_commit_in_progress
            )
        return True

    monkeypatch.setattr(sched, "mark_job_run", record_mark)

    assert sched._run_one_job_body(
        job,
        fire_claim_lost=threading.Event(),
        execution_token=execution_token,
    ) is True
    sched._monitor_commit_in_progress.discard(execution_token)

    assert protected_at_mark == [True]


def test_after_delivery_policy_does_not_commit_on_agent_failure(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed = {"raise_error": "simulated agent failure"}
    _install_agent_stubs(monkeypatch, observed)

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored.get("monitor_state") is None
    assert stored["last_status"] == "error"


def test_after_delivery_policy_does_not_commit_on_empty_agent_response(
    hermes_env, monkeypatch
):
    import cron.scheduler as sched
    from cron.jobs import get_job, update_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    update_job(job["id"], {"monitor_commit_policy": "after_delivery"})
    job = get_job(job["id"])
    observed = {"final_response": ""}
    _install_agent_stubs(monkeypatch, observed)

    assert sched.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored.get("monitor_state") is None
    assert stored["last_status"] == "error"


def test_unchanged_output_suppresses_agent_run(hermes_env, monkeypatch):
    from cron.jobs import get_job
    from cron.scheduler import SILENT_MARKER, run_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    run_job(job)
    assert observed["agent_runs"] == 1

    # Second tick with identical output → suppressed: no agent, silent.
    job = get_job(job["id"])
    success, doc, final, error = run_job(job)
    assert success is True
    assert error is None
    assert final == SILENT_MARKER
    assert observed["agent_runs"] == 1  # unchanged — agent NOT re-invoked
    assert "no_change" in doc


def test_changed_output_injects_diff(hermes_env, monkeypatch):
    from cron.jobs import get_job
    from cron.scheduler import run_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    run_job(job)

    # Mutate the monitored source, then fire again.
    _write_script(hermes_env, "mon.sh", "echo 'state B'\n")
    job = get_job(job["id"])
    success, doc, final, error = run_job(job)
    assert success is True
    assert observed["agent_runs"] == 2
    prompt = observed["prompts"][1]
    assert "MONITOR CHANGE DETECTED" in prompt
    assert "-state A" in prompt
    assert "+state B" in prompt
    assert "state B" in prompt  # new output included verbatim


def test_hash_persists_across_scheduler_restart(hermes_env, monkeypatch):
    """Suppression state must survive a scheduler restart (module reload)."""
    import importlib

    from cron.scheduler import run_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    run_job(job)
    assert observed["agent_runs"] == 1

    # Simulate restart: reload the cron modules, dropping in-memory state.
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.monitor
    importlib.reload(cron.monitor)
    import cron.scheduler
    importlib.reload(cron.scheduler)
    _install_agent_stubs(monkeypatch, observed)

    job = cron.jobs.get_job(job["id"])
    assert job["monitor_state"]["last_output_hash"]
    success, doc, final, error = cron.scheduler.run_job(job)
    assert success is True
    assert final == cron.scheduler.SILENT_MARKER
    assert observed["agent_runs"] == 1  # still suppressed after restart


def test_monitor_script_failure_is_error_not_change(hermes_env, monkeypatch):
    from cron.jobs import get_job
    from cron.scheduler import run_job

    job = _make_monitor_job(hermes_env, "echo 'state A'\n")
    observed: dict = {}
    _install_agent_stubs(monkeypatch, observed)

    run_job(job)
    stored_hash = get_job(job["id"])["monitor_state"]["last_output_hash"]

    # Break the source: non-zero exit must be an error, never a "change".
    _write_script(hermes_env, "mon.sh", "echo boom >&2\nexit 3\n")
    job = get_job(job["id"])
    success, doc, final, error = run_job(job)
    assert success is False
    assert error is not None
    assert observed["agent_runs"] == 1  # agent NOT invoked on source failure
    # Stored hash untouched — a later recovery to 'state A' still suppresses.
    assert get_job(job["id"])["monitor_state"]["last_output_hash"] == stored_hash


# ---------------------------------------------------------------------------
# cronjob tool: API-layer wiring
# ---------------------------------------------------------------------------


def test_cronjob_tool_create_with_monitor_script(hermes_env):
    from cron.jobs import get_job
    from tools.cronjob_tools import cronjob

    _write_script(hermes_env, "mon.sh", "echo hi\n")
    result = json.loads(
        cronjob(
            action="create",
            prompt="React",
            schedule="every 5m",
            monitor_script="mon.sh",
            deliver="local",
        )
    )
    assert result.get("success") is True
    job = get_job(result["job_id"])
    assert job["monitor_script"] == "mon.sh"


def test_cronjob_tool_rejects_monitor_script_path_escape(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            prompt="React",
            schedule="every 5m",
            monitor_script="../evil.sh",
            deliver="local",
        )
    )
    assert result.get("success") is False


def test_cronjob_tool_update_clears_monitor_script(hermes_env):
    from cron.jobs import get_job
    from tools.cronjob_tools import cronjob

    _write_script(hermes_env, "mon.sh", "echo hi\n")
    created = json.loads(
        cronjob(
            action="create",
            prompt="React",
            schedule="every 5m",
            monitor_script="mon.sh",
            deliver="local",
        )
    )
    result = json.loads(
        cronjob(action="update", job_id=created["job_id"], monitor_script="")
    )
    assert result.get("success") is True
    assert get_job(created["job_id"]).get("monitor_script") is None
