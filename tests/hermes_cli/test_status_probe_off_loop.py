"""/api/status does its local file reads off the event loop (#143561).

The handler runs on the loop that also serves the session WebSocket (Desktop), so a blocking read
of config.yaml / auth.json there stalls every other request for as long as the disk takes.
"""

import asyncio
import threading

import pytest


class _Stop(Exception):
    pass


@pytest.mark.asyncio
async def test_status_config_stamp_read_runs_off_the_event_loop(monkeypatch):
    import hermes_cli.config as config_mod
    from hermes_cli.web_routers import status as status_router

    loop_thread = threading.get_ident()
    seen = []

    def _check_config_version(*_a, **_kw):
        seen.append(threading.get_ident())
        raise _Stop()

    monkeypatch.setattr(config_mod, "check_config_version", _check_config_version)
    with pytest.raises(_Stop):
        await asyncio.wait_for(status_router.get_status(), 10)
    assert seen, "the config stamp was never read"
    assert seen[0] != loop_thread, "check_config_version ran on the event loop thread"
