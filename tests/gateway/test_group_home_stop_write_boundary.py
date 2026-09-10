"""Stop stays available before and after a consent writer replaces config."""

import asyncio
from threading import Event

import pytest

from gateway import group_home_consent as consent
from hermes_cli import config
from tests.gateway.test_group_home_consent import home, command


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before_write", "after_replace"])
async def test_stop_does_not_wait_for_consent_publication(home, monkeypatch, boundary):
    await command(home, "/group")
    entered, release = Event(), Event()
    original = config.atomic_yaml_write

    def held_write(path, data, *args, **kwargs):
        binding = data.get("platforms", {}).get("telegram", {}).get("home_channel", {})
        if not binding.get("group_audience_ack"):
            return original(path, data, *args, **kwargs)
        if boundary == "after_replace":
            original(path, data, *args, **kwargs)
        entered.set()
        assert release.wait(10)
        if boundary == "before_write":
            return original(path, data, *args, **kwargs)

    monkeypatch.setattr(config, "atomic_yaml_write", held_write)
    confirming = asyncio.create_task(command(home, "/group confirm"))
    assert await asyncio.to_thread(entered.wait, 5)
    stopping = None
    try:
        assert await asyncio.wait_for(command(home, "/group cancel"), 2) == consent.text("cancel_late")
        stopping = asyncio.create_task(command(home, "/group 1 stop"))
        done, _pending = await asyncio.wait({stopping}, timeout=2)
        assert stopping in done, "Stop waited for consent config publication"
        assert home.service.stopped
        assert stopping.result() == consent.text("stop_requested")
    finally:
        release.set()
        results = await asyncio.gather(
            confirming, *([stopping] if stopping is not None else []), return_exceptions=True
        )
        assert not any(isinstance(result, BaseException) for result in results), results
    assert "Release room" not in str(results[0])
