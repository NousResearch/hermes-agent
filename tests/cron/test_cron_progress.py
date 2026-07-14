"""Regression tests for configurable cron progress updates."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest


class FakeProgressAgent:
    def get_activity_summary(self):
        return {
            "last_activity_desc": "running focused tests",
            "seconds_since_activity": 3.0,
            "current_tool": "terminal",
            "api_call_count": 4,
            "max_iterations": 90,
        }


def test_auto_progress_policy_targets_likely_long_agent_jobs():
    from cron.scheduler import _cron_progress_auto_enabled

    assert _cron_progress_auto_enabled(
        {"name": "repo worker", "workdir": "/repo", "no_agent": False}
    )
    assert _cron_progress_auto_enabled(
        {"skills": ["github-ci-repair"], "no_agent": False}
    )
    assert not _cron_progress_auto_enabled(
        {"name": "daily report", "no_agent": False}
    )
    assert not _cron_progress_auto_enabled(
        {
            "name": "implementation watchdog",
            "no_agent": True,
            "script": "watch.py",
        }
    )


def test_progress_config_uses_global_and_per_job_values():
    from cron.scheduler import _resolve_cron_progress_config

    global_cfg = {
        "cron": {
            "progress": {
                "enabled": True,
                "initial_delay_seconds": 30,
                "interval_seconds": 45,
                "edit_in_place": False,
            }
        }
    }
    job = {
        "name": "worker",
        "progress": {
            "enabled": False,
            "initial_delay_seconds": 5,
            "interval_seconds": 7,
            "edit_in_place": True,
            "state_path": "state.json",
        },
    }

    resolved = _resolve_cron_progress_config(job, global_cfg)

    assert resolved == {
        "enabled": False,
        "initial_delay_seconds": 5.0,
        "interval_seconds": 7.0,
        "edit_in_place": True,
        "state_path": "state.json",
    }


def test_progress_config_rejects_non_finite_timing_values():
    from cron.scheduler import _resolve_cron_progress_config

    resolved = _resolve_cron_progress_config(
        {
            "progress": {
                "enabled": True,
                "initial_delay_seconds": float("nan"),
                "interval_seconds": float("inf"),
            }
        },
        {},
    )

    assert resolved["initial_delay_seconds"] == 90.0
    assert resolved["interval_seconds"] == 120.0


def test_default_config_exposes_cron_progress_settings():
    from hermes_cli.config import DEFAULT_CONFIG

    cron_config = cast(dict[str, Any], DEFAULT_CONFIG["cron"])
    assert cron_config["progress"] == {
        "enabled": "auto",
        "initial_delay_seconds": 90,
        "interval_seconds": 120,
        "edit_in_place": True,
    }


def test_job_progress_override_is_normalized_and_persisted(tmp_path):
    from cron.jobs import create_job, get_job, update_job, use_cron_store

    with use_cron_store(tmp_path):
        job = create_job(
            prompt="",
            schedule="every 5m",
            deliver="local",
            script="watch.py",
            no_agent=True,
            progress={
                "enabled": "all",
                "interval_seconds": 10,
                "state_path": "state.json",
                "ignored": "drop me",
            },
        )
        stored = get_job(job["id"])
        assert stored is not None
        assert stored["progress"] == {
            "enabled": "all",
            "interval_seconds": 10,
            "state_path": "state.json",
        }

        updated = update_job(job["id"], {"progress": "off"})
        assert updated is not None
        assert updated["progress"] == {"enabled": "off"}

        cleared = update_job(job["id"], {"progress": {}})
        assert cleared is not None
        assert cleared.get("progress") is None


def test_cronjob_schema_exposes_progress_override():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    parameters = cast(dict[str, Any], CRONJOB_SCHEMA["parameters"])
    progress = parameters["properties"]["progress"]
    assert {entry["type"] for entry in progress["anyOf"][:2]} == {
        "boolean",
        "string",
    }
    assert set(progress["anyOf"][2]["properties"]) == {
        "enabled",
        "initial_delay_seconds",
        "interval_seconds",
        "edit_in_place",
        "state_path",
    }


def test_cronjob_tool_create_update_and_clear_progress(tmp_path, monkeypatch):
    from cron.jobs import use_cron_store
    from tools.cronjob_tools import cronjob

    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    with use_cron_store(home):
        created = json.loads(
            cronjob(
                action="create",
                schedule="every 5m",
                deliver="local",
                script="watch.py",
                no_agent=True,
                progress={"enabled": True, "initial_delay_seconds": 1},
            )
        )
        assert created["success"] is True
        assert created["job"]["progress"] == {
            "enabled": True,
            "initial_delay_seconds": 1,
        }

        job_id = created["job_id"]
        updated = json.loads(
            cronjob(action="update", job_id=job_id, progress="off")
        )
        assert updated["job"]["progress"] == {"enabled": "off"}

        cleared = json.loads(
            cronjob(action="update", job_id=job_id, progress={})
        )
        assert "progress" not in cleared["job"]


def test_progress_message_is_compact_and_includes_state_reference():
    from cron.scheduler import _format_cron_progress_message

    message = _format_cron_progress_message(
        {"id": "abc123", "name": "repo worker"},
        FakeProgressAgent().get_activity_summary(),
        elapsed_seconds=367,
        state_path=".hermes/plans/dev/state.json",
    )

    assert message.splitlines() == [
        "⏳ repo worker",
        "6 min elapsed — iteration 4/90",
        "Activity: terminal (3s since last activity)",
        "State: .hermes/plans/dev/state.json",
    ]


def test_maybe_send_progress_is_rate_limited(monkeypatch):
    import cron.scheduler as scheduler

    sent = []
    monkeypatch.setattr(
        scheduler,
        "_deliver_cron_progress_update",
        lambda job, content, progress_state, **kwargs: sent.append(content),
    )
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: 125.0)

    cfg = {
        "enabled": True,
        "initial_delay_seconds": 10.0,
        "interval_seconds": 60.0,
        "edit_in_place": True,
        "state_path": "",
    }
    state = {"started_at": 100.0, "last_sent_at": 0.0, "message_ids": {}}
    job = {
        "id": "job1",
        "name": "repo worker",
        "workdir": "/repo",
        "prompt": "Use .hermes/plans/dev/state.json",
    }

    scheduler._maybe_send_cron_progress(
        job,
        FakeProgressAgent(),
        cfg,
        state,
        workdir="/repo",
    )
    scheduler._maybe_send_cron_progress(
        job,
        FakeProgressAgent(),
        cfg,
        state,
        workdir="/repo",
    )

    assert len(sent) == 1
    assert state["last_sent_at"] == 125.0


def test_live_route_helper_distinguishes_forum_and_direct_message_topics(monkeypatch):
    import cron.scheduler as scheduler
    from gateway.config import Platform

    job = {"id": "job1"}
    adapter = object()
    loop = object()

    monkeypatch.setattr(scheduler, "_is_channel_dm_topic", lambda *args: False)
    forum_target, forum_metadata, forum_media = scheduler._resolve_live_cron_route(
        job,
        platform=Platform.TELEGRAM,
        runtime_adapter=adapter,
        chat_id="123",
        thread_id="42",
        loop=loop,
    )
    assert forum_target.thread_id == "42"
    assert forum_metadata == {"job_id": "job1", "thread_id": "42"}
    assert forum_media == {"thread_id": "42"}

    monkeypatch.setattr(scheduler, "_is_channel_dm_topic", lambda *args: True)
    dm_target, dm_metadata, dm_media = scheduler._resolve_live_cron_route(
        job,
        platform=Platform.TELEGRAM,
        runtime_adapter=adapter,
        chat_id="123",
        thread_id="42",
        loop=loop,
    )
    assert dm_target.thread_id is None
    assert dm_metadata == {
        "job_id": "job1",
        "direct_messages_topic_id": "42",
    }
    assert dm_media == {"direct_messages_topic_id": "42"}


def test_progress_initial_send_uses_router_topic_metadata(monkeypatch):
    import agent.async_utils as async_utils
    import cron.scheduler as scheduler
    import gateway.config as gateway_config
    import gateway.delivery as delivery

    platform = gateway_config.Platform.TELEGRAM
    routed = []

    class FakeLoop:
        def is_running(self):
            return True

    class FakeAdapter:
        async def send(self, *args, **kwargs):
            raise AssertionError("progress must send through DeliveryRouter")

    async def fake_deliver(self, target, content, metadata):
        routed.append((target, content, metadata))
        return SimpleNamespace(success=True, message_id="msg-1")

    def fake_safe_schedule(coro, loop):
        result = asyncio.run(coro)
        return SimpleNamespace(result=lambda timeout=None: result, cancel=lambda: True)

    config = SimpleNamespace(
        platforms={platform: SimpleNamespace(enabled=True)},
        filter_silence_narration=True,
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [
            {"platform": "telegram", "chat_id": "123", "thread_id": "42"}
        ],
    )
    monkeypatch.setattr(scheduler, "_is_channel_dm_topic", lambda *args: True)
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: config)
    monkeypatch.setattr(delivery.DeliveryRouter, "_deliver_to_platform", fake_deliver)
    monkeypatch.setattr(async_utils, "safe_schedule_threadsafe", fake_safe_schedule)

    state = {"message_ids": {}}
    error = scheduler._deliver_cron_progress_update(
        {"id": "job1", "name": "worker"},
        "working",
        state,
        adapters={platform: FakeAdapter()},
        loop=FakeLoop(),
    )

    assert error is None
    target, content, metadata = routed[0]
    assert content == "working"
    assert target.thread_id is None
    assert metadata == {
        "job_id": "job1",
        "direct_messages_topic_id": "42",
    }
    assert state["message_ids"]["telegram:123:42"] == "msg-1"


@pytest.mark.parametrize(
    ("future_mode", "expect_error"),
    (("failure", True), ("ambiguous_timeout", False)),
)
def test_live_router_failure_never_falls_back_to_raw_send(
    monkeypatch,
    future_mode,
    expect_error,
):
    import agent.async_utils as async_utils
    import cron.scheduler as scheduler
    import gateway.config as gateway_config
    import tools.send_message_tool as send_message_tool

    platform = gateway_config.Platform.TELEGRAM
    raw_sends = []

    class FakeLoop:
        def is_running(self):
            return True

    class FakeAdapter:
        pass

    class FakeFuture:
        def result(self, timeout=None):
            if future_mode == "ambiguous_timeout":
                raise TimeoutError
            return SimpleNamespace(success=False)

        def cancel(self):
            return future_mode != "ambiguous_timeout"

    def fake_safe_schedule(coro, loop):
        coro.close()
        return FakeFuture()

    async def must_not_raw_send(*args, **kwargs):
        raw_sends.append((args, kwargs))
        raise AssertionError("live router failures must not fall back to raw send")

    config = SimpleNamespace(
        platforms={platform: SimpleNamespace(enabled=True)},
        filter_silence_narration=True,
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [
            {"platform": "telegram", "chat_id": "123", "thread_id": "42"}
        ],
    )
    monkeypatch.setattr(scheduler, "_is_channel_dm_topic", lambda *args: True)
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: config)
    monkeypatch.setattr(async_utils, "safe_schedule_threadsafe", fake_safe_schedule)
    monkeypatch.setattr(send_message_tool, "_send_to_platform", must_not_raw_send)

    error = scheduler._deliver_cron_progress_update(
        {"id": "job1"},
        "working",
        {"message_ids": {}},
        adapters={platform: FakeAdapter()},
        loop=FakeLoop(),
    )

    assert bool(error) is expect_error
    assert raw_sends == []


def test_progress_existing_message_edits_without_resending(monkeypatch):
    import agent.async_utils as async_utils
    import cron.scheduler as scheduler
    import gateway.config as gateway_config

    platform = gateway_config.Platform.TELEGRAM

    class FakeLoop:
        def is_running(self):
            return True

    class FakeAdapter:
        def __init__(self):
            self.edits = []

        async def edit_message(self, chat_id, message_id, text):
            self.edits.append((chat_id, message_id, text))
            return SimpleNamespace(success=True)

    def fake_safe_schedule(coro, loop):
        result = asyncio.run(coro)
        return SimpleNamespace(result=lambda timeout=None: result, cancel=lambda: True)

    config = SimpleNamespace(platforms={platform: SimpleNamespace(enabled=True)})
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [
            {"platform": "telegram", "chat_id": "123", "thread_id": "42"}
        ],
    )
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: config)
    monkeypatch.setattr(async_utils, "safe_schedule_threadsafe", fake_safe_schedule)

    adapter = FakeAdapter()
    state = {"message_ids": {"telegram:123:42": "msg-1"}}
    error = scheduler._deliver_cron_progress_update(
        {"id": "job1"},
        "still working",
        state,
        adapters={platform: adapter},
        loop=FakeLoop(),
    )

    assert error is None
    assert adapter.edits == [("123", "msg-1", "still working")]


def test_progress_script_wrapper_preserves_claim_heartbeat_path(monkeypatch):
    import cron.scheduler as scheduler

    calls = []
    monkeypatch.setattr(
        scheduler,
        "_run_job_script_with_claim_heartbeat",
        lambda job, script_path: calls.append((job["id"], script_path))
        or (True, "done"),
    )

    result = scheduler._run_job_script_with_progress(
        {"id": "job1"},
        "watch.py",
        {"enabled": True},
        {"started_at": 0.0, "last_sent_at": 0.0, "message_ids": {}},
    )

    assert result == (True, "done")
    assert calls == [("job1", "watch.py")]
