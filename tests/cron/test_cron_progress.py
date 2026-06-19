"""Tests for generic cron progress heartbeats."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# Ensure project root is importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for tests that create real cron jobs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants

    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)
    import tools.cronjob_tools
    importlib.reload(tools.cronjob_tools)
    return home


class FakeProgressAgent:
    def get_activity_summary(self):
        return {
            "last_activity_desc": "terminal running focused tests",
            "seconds_since_activity": 3.0,
            "current_tool": "terminal",
            "api_call_count": 4,
            "max_iterations": 90,
        }


def test_auto_progress_policy_skips_no_agent_watchdogs():
    from cron.scheduler import _cron_progress_auto_enabled

    assert not _cron_progress_auto_enabled(
        {
            "name": "memory watchdog implement alert",
            "prompt": "implement a check",
            "script": "memory.sh",
            "no_agent": True,
        }
    )


def test_auto_progress_policy_uses_workdir_and_skill_markers():
    from cron.scheduler import _cron_progress_auto_enabled

    assert _cron_progress_auto_enabled({"name": "repo worker", "workdir": "/repo", "no_agent": False})
    assert _cron_progress_auto_enabled(
        {
            "name": "continuable worker",
            "skills": ["github-pr-workflow"],
            "no_agent": False,
        }
    )
    assert not _cron_progress_auto_enabled(
        {"name": "daily learning drip", "prompt": "teach one lesson", "no_agent": False}
    )


def test_progress_config_defaults_to_auto_mode(monkeypatch):
    from cron.scheduler import _resolve_cron_progress_config

    monkeypatch.delenv("HERMES_CRON_PROGRESS_ENABLED", raising=False)
    likely_long_cfg = _resolve_cron_progress_config({"workdir": "/repo", "no_agent": False}, {})
    report_cfg = _resolve_cron_progress_config({"name": "daily report", "no_agent": False}, {})
    no_agent_cfg = _resolve_cron_progress_config(
        {"name": "watchdog", "no_agent": True, "script": "watchdog.sh"}, {}
    )

    assert likely_long_cfg["enabled"] is True
    assert report_cfg["enabled"] is False
    assert no_agent_cfg["enabled"] is False
    assert likely_long_cfg["initial_delay_seconds"] == 90.0
    assert likely_long_cfg["interval_seconds"] == 120.0


def test_progress_config_true_enables_all_cron_jobs_including_no_agent(monkeypatch):
    from cron.scheduler import _resolve_cron_progress_config

    monkeypatch.delenv("HERMES_CRON_PROGRESS_ENABLED", raising=False)
    cfg = {"cron": {"progress": {"enabled": True, "interval_seconds": 30}}}

    report_cfg = _resolve_cron_progress_config({"name": "daily report", "no_agent": False}, cfg)
    no_agent_cfg = _resolve_cron_progress_config(
        {"name": "watchdog", "no_agent": True, "script": "watchdog.sh"}, cfg
    )

    assert report_cfg["enabled"] is True
    assert no_agent_cfg["enabled"] is True
    assert report_cfg["interval_seconds"] == 30.0


def test_progress_config_honours_job_and_env_overrides(monkeypatch):
    from cron.scheduler import _resolve_cron_progress_config

    job = {
        "workdir": "/repo",
        "no_agent": False,
        "progress": {"enabled": False, "initial_delay_seconds": 5, "interval_seconds": 7},
    }
    cfg = {"cron": {"progress": {"enabled": True, "interval_seconds": 30}}}
    disabled = _resolve_cron_progress_config(job, cfg)
    assert disabled["enabled"] is False
    assert disabled["initial_delay_seconds"] == 5.0
    assert disabled["interval_seconds"] == 7.0

    monkeypatch.setenv("HERMES_CRON_PROGRESS_ENABLED", "all")
    monkeypatch.setenv("HERMES_CRON_PROGRESS_INTERVAL", "11")
    enabled = _resolve_cron_progress_config(job, cfg)
    assert enabled["enabled"] is True
    assert enabled["interval_seconds"] == 11.0



def test_create_job_stores_progress_override(hermes_env):
    from cron.jobs import create_job

    job = create_job(
        prompt="run a long report",
        schedule="every 5m",
        deliver="local",
        progress={
            "enabled": "all",
            "interval_seconds": 10,
            "state_path": "state.json",
            "ignored": "drop me",
        },
    )

    assert job["progress"] == {
        "enabled": "all",
        "interval_seconds": 10,
        "state_path": "state.json",
    }


def test_cronjob_tool_create_update_and_clear_progress(hermes_env):
    import json

    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            prompt="run a long report",
            deliver="local",
            progress={"enabled": True, "initial_delay_seconds": 1},
        )
    )
    assert created["success"] is True
    assert created["job"]["progress"] == {"enabled": True, "initial_delay_seconds": 1}

    job_id = created["job_id"]
    updated = json.loads(cronjob(action="update", job_id=job_id, progress="off"))
    assert updated["success"] is True
    assert updated["job"]["progress"] == {"enabled": "off"}

    cleared = json.loads(cronjob(action="update", job_id=job_id, progress={}))
    assert cleared["success"] is True
    assert "progress" not in cleared["job"]


def test_progress_message_is_human_readable_and_includes_state_path():
    from cron.scheduler import _format_cron_progress_message

    message = _format_cron_progress_message(
        {"id": "abc123", "name": "repo-worker"},
        {
            "current_tool": "terminal",
            "last_activity_desc": "api call",
            "seconds_since_activity": 12.2,
            "api_call_count": 6,
            "max_iterations": 90,
        },
        elapsed_seconds=367,
        state_path=".hermes/plans/router/state.json",
    )

    assert "⏳ Cron job: repo-worker (abc123)" in message
    assert "6 min elapsed — iteration 6/90" in message
    assert "Activity: terminal (12s since last activity)" in message
    assert "State: .hermes/plans/router/state.json" in message


def test_maybe_send_progress_is_rate_limited(monkeypatch):
    import cron.scheduler as scheduler

    sent = []

    def fake_deliver(job, content, progress_state, **kwargs):
        sent.append(content)
        progress_state.setdefault("message_ids", {})["telegram:1:"] = "msg-1"
        return None

    monkeypatch.setattr(scheduler, "_deliver_cron_progress_update", fake_deliver)
    monkeypatch.setattr(scheduler.time, "time", lambda: 125.0)

    job = {
        "id": "job1",
        "name": "repo worker",
        "prompt": "Use .hermes/plans/dev/state.json as durable state",
        "workdir": "/repo",
        "no_agent": False,
    }
    cfg = {"enabled": True, "initial_delay_seconds": 10.0, "interval_seconds": 60.0}
    state = {"started_at": 100.0, "last_sent_at": 0.0, "message_ids": {}}

    scheduler._maybe_send_cron_progress(job, FakeProgressAgent(), cfg, state, workdir="/repo")
    scheduler._maybe_send_cron_progress(job, FakeProgressAgent(), cfg, state, workdir="/repo")

    assert len(sent) == 1
    assert state["last_sent_at"] == 125.0
    assert "State: .hermes/plans/dev/state.json" in sent[0]


def test_maybe_send_progress_waits_for_initial_delay(monkeypatch):
    import cron.scheduler as scheduler

    sent = []
    monkeypatch.setattr(
        scheduler,
        "_deliver_cron_progress_update",
        lambda job, content, progress_state, **kwargs: sent.append(content),
    )
    monkeypatch.setattr(scheduler.time, "time", lambda: 105.0)

    scheduler._maybe_send_cron_progress(
        {"id": "job1", "name": "repo worker", "workdir": "/repo"},
        FakeProgressAgent(),
        {"enabled": True, "initial_delay_seconds": 10.0, "interval_seconds": 60.0},
        {"started_at": 100.0, "last_sent_at": 0.0, "message_ids": {}},
        workdir="/repo",
    )

    assert sent == []


def test_maybe_send_progress_accepts_script_activity_override(monkeypatch):
    import cron.scheduler as scheduler

    sent = []
    monkeypatch.setattr(
        scheduler,
        "_deliver_cron_progress_update",
        lambda job, content, progress_state, **kwargs: sent.append(content),
    )
    monkeypatch.setattr(scheduler.time, "time", lambda: 125.0)

    scheduler._maybe_send_cron_progress(
        {"id": "job1", "name": "script worker", "no_agent": True},
        None,
        {"enabled": True, "initial_delay_seconds": 10.0, "interval_seconds": 60.0},
        {"started_at": 100.0, "last_sent_at": 0.0, "message_ids": {}},
        activity_override={
            "current_tool": "script",
            "last_activity_desc": "running script: watchdog.sh",
            "seconds_since_activity": 0.0,
        },
    )

    assert len(sent) == 1
    assert "Activity: script (0s since last activity)" in sent[0]


def test_progress_delivery_sends_then_edits_live_adapter(monkeypatch):
    import asyncio

    import agent.async_utils as async_utils
    import cron.scheduler as scheduler
    import gateway.config as gateway_config

    platform = gateway_config.Platform("telegram")

    class FakeLoop:
        def is_running(self):
            return True

    class FakeAdapter:
        def __init__(self):
            self.sent = []
            self.edited = []

        async def send(self, chat_id, text, metadata=None):
            self.sent.append((chat_id, text, metadata))
            return SimpleNamespace(success=True, message_id="msg-1")

        async def edit_message(self, chat_id, message_id, text):
            self.edited.append((chat_id, message_id, text))
            return SimpleNamespace(success=True)

    def fake_safe_schedule(coro, loop):
        result = asyncio.run(coro)
        return SimpleNamespace(result=lambda timeout=None: result)

    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "telegram", "chat_id": "123", "thread_id": None}],
    )
    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: SimpleNamespace(platforms={platform: SimpleNamespace(enabled=True)}),
    )
    monkeypatch.setattr(async_utils, "safe_schedule_threadsafe", fake_safe_schedule)

    adapter = FakeAdapter()
    state = {"message_ids": {}}
    job = {"id": "job1", "name": "repo worker"}

    assert scheduler._deliver_cron_progress_update(
        job,
        "first heartbeat",
        state,
        adapters={platform: adapter},
        loop=FakeLoop(),
        edit_in_place=True,
    ) is None
    assert adapter.sent == [("123", "first heartbeat", None)]
    assert state["message_ids"]["telegram:123:"] == "msg-1"

    assert scheduler._deliver_cron_progress_update(
        job,
        "second heartbeat",
        state,
        adapters={platform: adapter},
        loop=FakeLoop(),
        edit_in_place=True,
    ) is None
    assert adapter.edited == [("123", "msg-1", "second heartbeat")]
