"""Scheduler ownership belongs to the backend surface, not gateway ownership."""

import asyncio
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize(
    "surface, desktop, owns_scheduler",
    [
        ("webapp", False, True),
        ("dashboard", True, True),
        ("webapp", True, True),
        ("dashboard", False, False),
        (None, False, False),
    ],
)
def test_lifespan_owns_scheduler_without_adopting_native_gateway(
    monkeypatch, tmp_path, surface, desktop, owns_scheduler
):
    from cron import scheduler_provider
    from hermes_cli import gateway, web_server
    from hermes_cli.local_runtime import bootstrap
    from tui_gateway import methods_groups

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if desktop:
        monkeypatch.setenv("HERMES_DESKTOP", "1")
    else:
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)

    started = threading.Event()
    stopped = threading.Event()
    starts = []

    def start(stop_event, **kwargs):
        starts.append((stop_event, kwargs))
        started.set()
        if stop_event.wait(5):
            stopped.set()

    # Exercise the real lifespan and existing ticker; replace only the provider
    # execution boundary so no scheduled job or model inference can run.
    provider = SimpleNamespace(name="test", start=start)
    monkeypatch.setattr(scheduler_provider, "resolve_cron_scheduler", lambda: provider)
    monkeypatch.setattr(web_server, "_eager_reconcile_own_session_db", lambda: None)
    monkeypatch.setattr(web_server, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(methods_groups, "start_hosted_room_service", lambda: None)
    monkeypatch.setattr(methods_groups, "stop_hosted_room_service", lambda **kw: None)
    monkeypatch.setattr(bootstrap, "ensure_local_runtime", lambda config: None)
    monkeypatch.setattr(bootstrap, "shutdown_local_runtime", lambda: None)
    monkeypatch.setattr(web_server, "PTY_REGISTRY", SimpleNamespace(close_all=AsyncMock()))

    async def idle(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(web_server, "run_reaper", idle)
    monkeypatch.setattr(web_server, "_dashboard_selftest_loop", idle)
    monkeypatch.setattr(web_server, "_auto_archive_ticker_loop", idle)
    reap = Mock()
    terminate = Mock()
    monkeypatch.setattr(gateway, "_reap_unsupervised_gateway_orphans", reap)
    monkeypatch.setattr(web_server, "_terminate_desktop_managed_gateway", terminate)
    app = SimpleNamespace(state=SimpleNamespace())
    if surface is not None:
        app.state.ui_surface = surface

    async def exercise():
        async with web_server._lifespan(app):
            if owns_scheduler:
                assert started.wait(2), "standalone surface never started its scheduler"
                assert len(starts) == 1
                assert not starts[0][0].is_set()
                assert starts[0][1] == {"interval": 60}
            else:
                assert not starts
            assert reap.call_count == int(desktop)
            assert not terminate.called
            assert os.getenv("HERMES_DESKTOP") == ("1" if desktop else None)

    asyncio.run(exercise())
    if owns_scheduler:
        assert starts[0][0].is_set(), "backend shutdown did not signal its scheduler"
        assert stopped.wait(2), "scheduler did not observe backend shutdown"
    assert terminate.call_count == int(desktop)
