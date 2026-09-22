import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import sprites_activity
from hermes_cli import sprites_api


@pytest.mark.asyncio
async def test_idle_release_fences_renewals_and_wake_reacquires(monkeypatch, tmp_path):
    api = Mock(return_value={})
    marker = tmp_path / 'wake'
    monkeypatch.setattr(sprites_api, 'request', api)
    monkeypatch.setattr(sprites_activity, 'WAKE_MARKER', marker)
    hold = sprites_activity.ActivityHold('default')
    runner = SimpleNamespace(_running=True, _scale_to_zero_is_idle=lambda: True)
    await hold.renew()
    sleeper = asyncio.create_task(hold.sleep_until_wake(runner))
    try:
        async with asyncio.timeout(5):
            while not any(c.args[0] == 'DELETE' for c in api.call_args_list):
                await asyncio.sleep(0.01)
        await hold.renew()
        assert [c.args[0] for c in api.call_args_list] == ['PUT', 'DELETE']
        marker.touch()
        await asyncio.wait_for(sleeper, 5)
        assert [c.args[0] for c in api.call_args_list] == ['PUT', 'DELETE', 'PUT']
        assert api.call_args_list[-1].args[2] == {'expire': '90s'}
    finally:
        sleeper.cancel()
        await asyncio.gather(sleeper, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_releases_the_bounded_activity_hold(monkeypatch):
    runner = SimpleNamespace(_running=True)
    actions = []

    def api(method, path, payload=None):
        actions.append((method, path, payload))
        runner._running = False

    monkeypatch.setattr(sprites_api, 'request', api)
    hold = sprites_activity.ActivityHold('work')
    task = asyncio.create_task(hold.watch(runner))
    async with asyncio.timeout(5):
        while not actions:
            await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert actions == [('PUT', '/tasks/hermes-work', {'expire': '90s'}), ('DELETE', '/tasks/hermes-work', None)]
