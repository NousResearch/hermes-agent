"""Session-start model picker (``gateway.session_start_model_picker``).

On a NEW session on a listed platform the gateway sends the ``/model`` picker
and holds the first turn until the user picks (or
``gateway.session_start_model_picker_timeout`` elapses, default 60s; 0 = do not
wait). The model resolves at agent-run time, after this hook, so a pick made
during the wait applies to the very first message. On timeout a one-line
"continuing with <model>" note is posted.

These drive the real hook and the real ``_send_model_picker`` with a fake
picker-capable adapter that captures the ``on_model_selected`` closure, so a
"click" is a call to that closure.
"""

import asyncio
import time
import types

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_session_start import _maybe_send_session_start_model_picker, run_session_start_hooks
from gateway.session import SessionSource
from tests.waiting import async_wait_until


class _FakePickerAdapter:
    def __init__(self):
        self.callback = None
        self.picker_sent = 0
        self.notes = []

    async def send_model_picker(self, *, on_model_selected, **kwargs):
        self.picker_sent += 1
        self.callback = on_model_selected
        return types.SimpleNamespace(success=True)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.notes.append(content)
        return types.SimpleNamespace(success=True)


def _make_runner(adapter, *, commit_raises=False):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner.committed = []

    async def _perform(ctx, model_id, provider_slug, source):
        return (model_id, provider_slug), None

    async def _commit(result, ctx, source=None, picker=False):
        if commit_raises:
            raise RuntimeError("switch failed")
        runner.committed.append(result)
        return "ok"

    runner._perform_model_switch = _perform
    runner._commit_model_switch = _commit
    return runner


def _event():
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="dm"),
    )


@pytest.fixture
def gw_config(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    cfg = {"model": {"default": "default-model", "provider": "openrouter"}}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: cfg)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        lambda **kw: [{"slug": "openrouter", "name": "OpenRouter", "models": ["gpt-5.5"]}],
    )
    return cfg


def _enable(cfg, *, platforms=("discord",), timeout=None):
    cfg["gateway"] = {"session_start_model_picker": list(platforms)}
    if timeout is not None:
        cfg["gateway"]["session_start_model_picker_timeout"] = timeout


def _start(runner, event):
    return asyncio.create_task(_maybe_send_session_start_model_picker(runner, event, event.source, "sk"))


@pytest.mark.asyncio
async def test_first_turn_is_held_until_the_pick(gw_config):
    _enable(gw_config, timeout=30)
    adapter = _FakePickerAdapter()
    runner = _make_runner(adapter)

    task = _start(runner, _event())
    await async_wait_until(lambda: adapter.callback is not None, what="picker sent")
    await asyncio.sleep(0)
    assert not task.done()  # the turn is waiting on the pick

    await adapter.callback("chat", "gpt-5.5", "openrouter")
    await asyncio.wait_for(task, timeout=5)

    assert runner.committed == [("gpt-5.5", "openrouter")]
    assert adapter.notes == []


@pytest.mark.asyncio
async def test_no_pick_times_out_and_posts_fallback_note(gw_config):
    _enable(gw_config, timeout=0.2)
    adapter = _FakePickerAdapter()
    runner = _make_runner(adapter)

    t0 = time.monotonic()
    await asyncio.wait_for(_start(runner, _event()), timeout=5)

    assert time.monotonic() - t0 >= 0.15
    assert runner.committed == []
    assert len(adapter.notes) == 1 and "default-model" in adapter.notes[0]


@pytest.mark.asyncio
async def test_failed_switch_still_releases_the_wait(gw_config):
    _enable(gw_config, timeout=30)
    adapter = _FakePickerAdapter()
    runner = _make_runner(adapter, commit_raises=True)

    task = _start(runner, _event())
    await async_wait_until(lambda: adapter.callback is not None, what="picker sent")
    with pytest.raises(RuntimeError):
        await adapter.callback("chat", "gpt-5.5", "openrouter")
    await asyncio.wait_for(task, timeout=5)

    assert adapter.notes == []


@pytest.mark.asyncio
async def test_timeout_zero_sends_picker_without_waiting(gw_config):
    _enable(gw_config, timeout=0)
    adapter = _FakePickerAdapter()
    runner = _make_runner(adapter)

    await asyncio.wait_for(_start(runner, _event()), timeout=5)

    assert adapter.picker_sent == 1
    assert adapter.notes == []


@pytest.mark.asyncio
async def test_unlisted_platform_is_a_no_op(gw_config):
    _enable(gw_config, platforms=("telegram",))
    adapter = _FakePickerAdapter()

    await asyncio.wait_for(_start(_make_runner(adapter), _event()), timeout=5)

    assert adapter.picker_sent == 0


@pytest.mark.asyncio
async def test_absent_key_is_a_no_op(gw_config):
    adapter = _FakePickerAdapter()

    await asyncio.wait_for(_start(_make_runner(adapter), _event()), timeout=5)

    assert adapter.picker_sent == 0


@pytest.mark.asyncio
async def test_picker_failure_never_blocks_the_turn(gw_config, monkeypatch):
    import gateway.run as gateway_run

    def _boom(*a, **k):
        raise OSError("config unreadable")

    monkeypatch.setattr(gateway_run, "_load_gateway_config", _boom)
    event = _event()

    await run_session_start_hooks(_make_runner(_FakePickerAdapter()), event, event.source, "sk", "qk")
