"""Secondary adapter success paths must stamp their ``<profile>:<platform>`` runtime-status row.

The primary paths stamp on startup (``run_startup.py::_start_aggregate_connect_results``) and on
reconnect (``run_adapters.py::_install_reconnected_adapter``); the secondary *success* paths
stamped nothing, so a healthy secondary channel had no ``gateway_state.json`` row at all while
every failure path wrote one (unserved / unbuilt / fatal / escalation) — measured 2026-10-25:
a secondary profile's channel served traffic for an hour with no row for it, and a channel
that reconnected after a failure kept its stale row. Contract: both secondary success paths write
the row, a degraded receive path reports ``retrying`` instead of ``connected``, and the write
clears the escalation flags a failure had set.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.run import GatewayRunner


class _Adapter:
    """Minimal secondary adapter: handler wiring + dedup, no transport (connect is stubbed)."""

    platform = Platform.DISCORD
    DEGRADED_STATUS_MESSAGE = "connected but not yet confirmed active; recovering in background"

    def __init__(self, *, degraded=False):
        self.send_path_degraded = degraded
        self._dedup = MessageDeduplicator()

    def set_message_handler(self, handler):
        self.message_handler = handler

    def set_fatal_error_handler(self, handler):
        self.fatal_error_handler = handler

    def set_session_store(self, store):
        self.session_store = store

    def set_busy_session_handler(self, handler):
        self.busy_session_handler = handler

    def set_topic_recovery_fn(self, handler):
        self.topic_recovery_fn = handler

    def set_authorization_check(self, handler):
        self.authorization_check = handler

    def set_platform_event_handler(self, handler):
        self.platform_event_handler = handler


def _runner():
    """Minimal GatewayRunner via object.__new__ (same pattern as test_multiplex_adapter_registry)."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._background_tasks = set()
    runner.session_store = object()
    runner._handle_active_session_busy_message = object()
    runner._recover_telegram_topic_thread_id = object()
    runner._busy_text_mode = "queue"
    runner._make_adapter_auth_check = lambda platform, profile_name=None: object()
    runner._adapter_disconnect_timeout_secs = lambda: 0
    runner._sync_voice_mode_state_to_adapter = lambda adapter: None
    runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=0)
    return runner


def _stub(monkeypatch, runner, adapter) -> list:
    """Stub every seam both secondary drives touch; return the recorded status writes."""

    @contextmanager
    def fake_scope(profile_home, *, hydrate_secrets=True):
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda name: Path("/profiles") / name
    )
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="profile-token")
            },
        ),
    )
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: adapter)
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda h: {})
    monkeypatch.setattr(gateway_run, "_load_gateway_config", dict)
    monkeypatch.setattr(runner, "_snapshot_profile_busy_modes", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)

    async def connect(adapter, platform, **kwargs):
        return True

    monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", connect)
    monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)

    writes: list = []
    monkeypatch.setattr(
        runner, "_update_platform_runtime_status", lambda key, **kw: writes.append((key, kw))
    )
    return writes


def _expected(state: str, error_message) -> list:
    return [
        (
            "reviewer:discord",
            {
                "platform_state": state,
                "error_code": None,
                "error_message": error_message,
                "needs_attention": False,
                "retrying_since": None,
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "degraded, state",
    [(False, "connected"), (True, "retrying")],
    ids=["healthy", "degraded"],
)
async def test_secondary_startup_success_stamps_row(monkeypatch, degraded, state):
    """A secondary that connects at startup gets its profile-scoped row; degraded reports retrying."""
    runner = _runner()
    adapter = _Adapter(degraded=degraded)
    writes = _stub(monkeypatch, runner, adapter)

    connected = await runner._start_one_profile_adapters(
        "reviewer", Path("/profiles/reviewer"), {}
    )

    assert connected == 1
    assert writes == _expected(state, adapter.DEGRADED_STATUS_MESSAGE if degraded else None)


@pytest.mark.asyncio
async def test_secondary_reconnect_success_stamps_row(monkeypatch):
    """The reconnect that succeeds writes connected and clears a failure's attention escalation."""
    runner = _runner()
    adapter = _Adapter()
    writes = _stub(monkeypatch, runner, adapter)

    await runner._run_secondary_profile_reconnect("reviewer", Platform.DISCORD)

    assert writes == _expected("connected", None)
