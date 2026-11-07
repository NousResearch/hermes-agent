"""start_gateway's startup tail: the unclean-exit state.db scan runs off the event loop after the runner
is live, and READY leaves the dashboard app unloaded until native Desktop attaches."""
import sys
import threading
from unittest.mock import AsyncMock

import asyncio

import pytest

from gateway import run_bootstrap
from gateway.run_runtime import publish_gateway_runtime_ready as _real_publish_ready


def test_scan_runs_on_a_daemon_thread_with_the_1800s_deadline(monkeypatch):
    from gateway import lifecycle_ledger as ledger
    seen, release, done = {}, threading.Event(), threading.Event()

    def scan(evidence, *, deadline_s):
        seen.update(evidence=evidence, deadline_s=deadline_s, thread=threading.current_thread())
        release.wait(5)  # still running when the caller returns: nothing waits on it
        done.set()
    monkeypatch.setattr(ledger, "run_deferred_integrity_check", scan)

    evidence = {"prior_pid": 1234}
    run_bootstrap._start_deferred_state_db_integrity_check(evidence)
    assert not done.is_set()
    release.set()
    assert done.wait(5)
    assert seen["evidence"] is evidence and seen["deadline_s"] == 1800.0
    assert seen["thread"].daemon and seen["thread"].name == "state-db-integrity-check"


@pytest.fixture
def _startup(monkeypatch, tmp_path):
    """Drive the real ``start_gateway`` with a fake runner; record the order of startup milestones."""
    from agent import memory_provider
    from gateway import lifecycle_ledger as ledger
    from gateway import run_runtime

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    events, release, scan_entered = [], threading.Event(), threading.Event()
    state = {"evidence": None}

    class _LiveRunner:
        def __init__(self, config):
            from gateway.session import SessionStore
            from hermes_constants import get_hermes_home
            from hermes_state import SessionDB
            self.session_store = SessionStore(get_hermes_home() / "sessions", config)
            self._session_db = SessionDB(get_hermes_home() / "state.db")
            self.session_store._db = self._session_db
            self._draining = False
            self._running = False
            self._closed = False
            self.config = config
            self.should_exit_cleanly = False
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            events.append("runner_started")
            self._running = True
            return True

        def _start_systemd_watchdog(self):
            pass

        async def stop(self):
            self._running = False
            if not self._closed:
                self._closed = True
                self.session_store.close_all_db_handles()
                self._session_db.close()

    real_spawn = memory_provider.spawn_context_thread

    def spawn(target, *, name, **kwargs):
        if name == "state-db-integrity-check":
            events.append("scan_launched")
        return real_spawn(target, name=name, **kwargs)

    def scan(evidence, *, deadline_s):
        state["scanned"] = evidence
        scan_entered.set()
        release.wait(10)

    async def wait_runtime(runner):
        events.append(("runtime_waited", release.is_set()))

    monkeypatch.setattr("gateway.run._host_attach_or_none", AsyncMock(return_value=None))
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _LiveRunner)
    monkeypatch.setattr(ledger, "record_startup", lambda: state["evidence"])
    monkeypatch.setattr(ledger, "run_deferred_integrity_check", scan)
    monkeypatch.setattr(memory_provider, "spawn_context_thread", spawn)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_start_cron_and_housekeeping",
                        lambda runner: (threading.Event(), None, None, None))
    monkeypatch.setattr(run_bootstrap, "_start_gateway_shutdown_tail", AsyncMock(return_value=True))
    monkeypatch.setattr(run_runtime, "publish_gateway_runtime_ready", lambda runner: events.append("ready"))
    monkeypatch.setattr(run_runtime, "wait_gateway_runtime", wait_runtime)
    yield events, state, scan_entered
    release.set()
    from gateway import host_rendezvous as hr
    hr.release_host_lock(hr.ROLE_GATEWAY)
    hr.clear_record(hr.ROLE_GATEWAY)


@pytest.mark.asyncio
async def test_unclean_exit_scan_starts_after_the_runner_and_does_not_hold_startup(_startup):
    from gateway.config import GatewayConfig
    events, state, scan_entered = _startup
    state["evidence"] = evidence = {"prior_pid": 1234}

    assert await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False, verbosity=None) is True

    # The scan is still blocked: startup reached READY and the runtime wait without it.
    assert events == ["runner_started", "scan_launched", "ready", ("runtime_waited", False)]
    assert scan_entered.wait(5) and state["scanned"] is evidence


@pytest.mark.asyncio
async def test_clean_previous_exit_launches_no_scan(_startup):
    from gateway.config import GatewayConfig
    events, _, scan_entered = _startup

    assert await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False, verbosity=None) is True

    assert events == ["runner_started", "ready", ("runtime_waited", False)]
    assert not scan_entered.is_set()


class _ImportSpy:
    """A ``sys.meta_path`` finder that counts import attempts of one module (and imports nothing)."""

    def __init__(self, name):
        self.name, self.attempts = name, 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            self.attempts += 1


@pytest.mark.asyncio
async def test_ready_gateway_loads_the_dashboard_only_for_its_first_desktop_client(_startup, monkeypatch):
    """A headless gateway (messaging, cron, kanban) never pays for the dashboard app (~20 MB): READY,
    CLI/TUI and worker sockets and refused clients leave it unimported; the first admitted host-scoped
    client (native Desktop) imports it, once."""
    import tui_gateway.ws
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidStatus

    from gateway import run_runtime, session_authorities
    from gateway.config import GatewayConfig
    from tests.waiting import async_wait_until

    dashboard = "hermes_cli.web_server"
    monkeypatch.delitem(sys.modules, dashboard, raising=False)
    spy = _ImportSpy(dashboard)
    monkeypatch.setattr(sys, "meta_path", [spy, *sys.meta_path])

    async def fake_handle_ws(ws, **kwargs):
        await ws.accept(subprotocol="hermes-gateway-v1")
        await ws.send_json({"native": True})
        await ws.close()
    monkeypatch.setattr(tui_gateway.ws, "handle_ws", fake_handle_ws)
    seen = {}

    async def wait_runtime(runner):
        home = runner.session_runtime_descriptor["served_profiles"][0]["profile_id"]
        url = runner.session_api.api_origin.replace("http:", "ws:") + "/api/ws"

        async def attach(purpose, scope="profile"):
            ticket = runner.session_ticket_store.mint(profile_id=home, subject="human", purpose=purpose,
                                                      scope=scope)
            try:
                async with connect(url, subprotocols=["hermes-gateway-v1", "hermes-gateway-ticket." + ticket],
                                   proxy=None) as ws:
                    await ws.recv()
                    return "admitted"
            except (InvalidStatus, ConnectionClosed):
                return "refused"

        async def imported_after(step):
            await step
            # A negative: nothing may start importing in the window a warm-up takes to begin.
            await asyncio.sleep(0.3)  # health: allow HX013 -- proving nothing starts in the window
            return spy.attempts

        seen["ready"] = await imported_after(asyncio.sleep(0))
        seen["cli"] = (await attach("interactive"), spy.attempts)  # hermes chat/TUI, kanban, cron
        seen["worker"] = (await attach("worker-adoption"), spy.attempts)
        runner._draining = True
        seen["draining"] = (await attach("interactive", "host"), await imported_after(asyncio.sleep(0)))
        runner._draining = False
        with monkeypatch.context() as patch:
            patch.setattr(session_authorities, "authority_for_profile_id", lambda runner, profile_id: None)
            seen["no_authority"] = (await attach("interactive", "host"),
                                    await imported_after(asyncio.sleep(0)))
        seen["desktop"] = await attach("interactive", "host")
        await async_wait_until(lambda: dashboard in sys.modules and runner.session_api.app.loaded,
                               timeout=60, what="dashboard mounted after the Desktop client")
        seen["second_desktop"] = await attach("interactive", "host")
        seen["attempts"] = spy.attempts

    monkeypatch.setattr(run_runtime, "publish_gateway_runtime_ready", _real_publish_ready)
    monkeypatch.setattr(run_runtime, "wait_gateway_runtime", wait_runtime)
    assert await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False, verbosity=None) is True

    assert seen == {
        "ready": 0, "cli": ("admitted", 0), "worker": ("admitted", 0),
        "draining": ("refused", 0), "no_authority": ("refused", 0),
        "desktop": "admitted", "second_desktop": "admitted", "attempts": 1,
    }, seen
