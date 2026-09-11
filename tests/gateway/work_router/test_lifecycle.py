"""Behavioral lifetime seams; no live profiles, model calls or Slack sends."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router import integration
from gateway.run_startup import GatewayStartupMixin
from plugins.platforms.slack.adapter import SlackAdapter
from gateway.config import PlatformConfig


def config_at(home):
    home.mkdir(parents=True, exist_ok=True)
    raw = {"enabled": True, "channel_allowlist": ["CEXEC"],
           "db_path": str(home / "router.db"),
           "bot_registry": {n: f"UBOT{i}" for i, n in enumerate(DEFAULT_PROFILES)}}
    (home / "config.yaml").write_text(yaml.safe_dump({"work_router": raw}))
    return RouterConfig.from_mapping(raw).require_ready()


def test_enabled_without_native_sender_refuses_before_opening_store(tmp_path):
    config = config_at(tmp_path)
    with pytest.raises(RuntimeError, match="native sender is not bound"):
        integration.RouterRuntime(config, sender=None)
    assert not (tmp_path / "router.db").exists()
    runner = SimpleNamespace()
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    assert integration.ensure_runtime(runner, profile_home=secondary) is None


@pytest.mark.asyncio
async def test_stop_fences_all_receivers_waits_children_and_closes_once(tmp_path):
    runtime = integration.RouterRuntime(config_at(tmp_path), sender=AsyncMock())
    first = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    rebuilt = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    runtime.attach(first)
    runtime.attach(rebuilt)
    runtime.attach(first)
    started = asyncio.Event()
    cleaned = []

    async def child():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert first._work_router_stopping and rebuilt._work_router_stopping
            runtime.router.store.inbox_count()  # SQLite must still be open.
            cleaned.append(True)

    task = asyncio.create_task(child())
    runtime.router._lounge_candidate_tasks.add(task)
    close = Mock(wraps=runtime.router.store.close)
    runtime.router.store.close = close
    await started.wait()
    await asyncio.gather(runtime.stop(), runtime.stop())
    assert cleaned == [True]
    assert runtime.closed and runtime.task.done() and task.done()
    close.assert_called_once()
    with pytest.raises(RuntimeError, match="stopping"):
        runtime.attach(first)


@pytest.mark.asyncio
async def test_timed_out_stop_retains_store_then_retry_closes(tmp_path, monkeypatch):
    runtime = integration.RouterRuntime(config_at(tmp_path), sender=AsyncMock())
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def resistant_child():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    task = asyncio.create_task(resistant_child())
    runtime.router._lounge_candidate_tasks.add(task)
    await started.wait()
    real_wait = asyncio.wait

    async def bounded_wait(tasks, *, timeout):
        await cancelled.wait()
        return await real_wait(tasks, timeout=0)

    monkeypatch.setattr(integration.asyncio, "wait", bounded_wait)
    await runtime.stop()
    assert runtime.stopping and not runtime.closed
    assert runtime.router.store.inbox_count() == 0
    release.set()
    await task
    monkeypatch.setattr(integration.asyncio, "wait", real_wait)
    await runtime.stop()
    assert runtime.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [False, "raise", "cancel", True])
async def test_start_failure_cleans_even_after_running_flag_set(outcome):
    stopped = AsyncMock()
    runner = SimpleNamespace(_running=True,
                             _work_router_runtimes={"isolated": SimpleNamespace(stop=stopped)})

    async def start():
        if outcome == "raise":
            raise ValueError("connect failed")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return outcome

    runner._start_with_work_router = start
    if outcome in ("raise", "cancel"):
        with pytest.raises(ValueError if outcome == "raise" else asyncio.CancelledError):
            await GatewayStartupMixin.start(runner)
    else:
        assert await GatewayStartupMixin.start(runner) is outcome
    assert stopped.await_count == (0 if outcome is True else 1)
