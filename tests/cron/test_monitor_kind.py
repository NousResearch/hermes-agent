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

import json
import sys

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
            return {"final_response": "agent done", "messages": []}

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


# ---------------------------------------------------------------------------
# cron.monitor: hashing + diff unit behavior
# ---------------------------------------------------------------------------


def test_hash_is_exact_bytes(hermes_env):
    from cron.monitor import hash_monitor_output

    assert hash_monitor_output("a\nb") == hash_monitor_output("a\nb")
    # Exact-bytes contract: even whitespace-only differences are changes.
    assert hash_monitor_output("a\nb") != hash_monitor_output("a\nb ")


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
