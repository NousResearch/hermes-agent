"""Teams socket mode (SDK >= 2.2.1, public preview): the adapter dials out instead of listening."""
import asyncio
import enum
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_teams import TeamsAdapter, _make_config, _teams_mod  # installs the SDK stand-in


class _Status(str, enum.Enum):
    CONNECTING = "connecting"
    READY = "ready"
    STOPPED = "stopped"


class _Options:
    def __init__(self, geos=("amer", "emea", "apac")):
        self.geos = tuple(geos)


class _SocketAdapter:
    def __init__(self, geos, *, fail: Exception | None = None):
        self._geos = geos
        self._fail = fail
        self._stop = asyncio.Event()
        self.stopped = False

    @property
    def geo_statuses(self):
        return tuple((g, _Status.STOPPED if self._fail else _Status.READY) for g in self._geos)

    @property
    def geo_list(self):
        return self._geos

    async def start(self, port: int = 0):
        if self._fail:
            raise self._fail
        await self._stop.wait()

    async def stop(self):
        self.stopped = True
        self._stop.set()


def _app_factory(created: list, *, fail: Exception | None = None):
    class _App:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            opts = kwargs.get("socket_mode")
            self.socket_mode = _SocketAdapter(opts.geos, fail=fail) if opts else None
            self.initialize = AsyncMock()
            created.append(self)

        def on_message(self, func):
            return func

        def on_card_action(self, func):
            return func

    return _App


@pytest.fixture
def socket_sdk(monkeypatch):
    mod = types.ModuleType("microsoft_teams.apps.socket_mode")
    mod.SocketModeOptions = _Options
    mod.SocketModeStatus = _Status
    monkeypatch.setitem(sys.modules, "microsoft_teams.apps.socket_mode", mod)
    monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
    monkeypatch.setattr(_teams_mod, "ClientOptions", lambda **kw: kw)
    return mod


def _socket_adapter(monkeypatch, **extra):
    monkeypatch.setenv("TEAMS_SOCKET_MODE", "true")
    return TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant", **extra))


class TestTeamsSocketModeConfig:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("TEAMS_SOCKET_MODE", raising=False)
        adapter = TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant"))
        assert adapter._socket_mode is False

    def test_env_switch_and_geos(self, monkeypatch):
        monkeypatch.setenv("TEAMS_SOCKET_GEOS", "emea, amer")
        adapter = _socket_adapter(monkeypatch)
        assert adapter._socket_mode is True
        assert adapter._socket_geos == ("emea", "amer")

    def test_extra_keys(self, monkeypatch):
        monkeypatch.delenv("TEAMS_SOCKET_MODE", raising=False)
        adapter = TeamsAdapter(_make_config(client_id="id", client_secret="secret", tenant_id="tenant",
                                            socket_mode=True, socket_geos="emea"))
        assert adapter._socket_mode is True
        assert adapter._socket_geos == ("emea",)


class TestTeamsSocketModeConnect:
    @pytest.mark.anyio
    async def test_connects_without_a_listener(self, monkeypatch, socket_sdk):
        created: list = []
        monkeypatch.setattr(_teams_mod, "App", _app_factory(created))
        bind = AsyncMock()
        monkeypatch.setattr("gateway.platforms.shared_ingress.bind_listener", bind)
        adapter = _socket_adapter(monkeypatch)
        assert await adapter.connect() is True
        app = created[0]
        assert "http_server_adapter" not in app.kwargs
        assert app.kwargs["socket_mode"].geos == ("amer", "emea", "apac")
        app.initialize.assert_awaited_once()
        bind.assert_not_called()
        assert adapter._runner is None
        assert adapter._socket_task is not None and not adapter._socket_task.done()
        await adapter.disconnect()
        assert app.socket_mode.stopped is True
        assert adapter._socket_task is None

    @pytest.mark.anyio
    async def test_selected_geos_reach_the_sdk(self, monkeypatch, socket_sdk):
        created: list = []
        monkeypatch.setattr(_teams_mod, "App", _app_factory(created))
        monkeypatch.setenv("TEAMS_SOCKET_GEOS", "emea")
        adapter = _socket_adapter(monkeypatch)
        assert await adapter.connect() is True
        assert created[0].kwargs["socket_mode"].geos == ("emea",)
        await adapter.disconnect()

    @pytest.mark.anyio
    async def test_refused_negotiate_is_a_retryable_fatal_with_hint(self, monkeypatch, socket_sdk):
        created: list = []
        monkeypatch.setattr(_teams_mod, "App", _app_factory(
            created, fail=RuntimeError("Socket Mode negotiate failed: HTTP 403")))
        adapter = _socket_adapter(monkeypatch)
        assert await adapter.connect() is False
        assert adapter._fatal_error_code == "CONNECT_FAILED"
        assert adapter._fatal_error_retryable is True
        assert "socket mode enabled in the bot registration" in adapter._fatal_error_message
        assert adapter._socket_task is None

    @pytest.mark.anyio
    async def test_sdk_without_socket_mode_is_a_permanent_fatal(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "microsoft_teams.apps.socket_mode", None)  # import fails
        monkeypatch.setattr(_teams_mod, "AIOHTTP_AVAILABLE", True)
        created: list = []
        monkeypatch.setattr(_teams_mod, "App", _app_factory(created))
        adapter = _socket_adapter(monkeypatch)
        assert await adapter.connect() is False
        assert adapter._fatal_error_code == "MISSING_SDK"
        assert adapter._fatal_error_retryable is False
        assert created == []

    @pytest.mark.anyio
    async def test_socket_ending_while_running_reports_a_fatal(self, monkeypatch, socket_sdk):
        created: list = []
        monkeypatch.setattr(_teams_mod, "App", _app_factory(created))
        adapter = _socket_adapter(monkeypatch)
        notified = MagicMock()

        async def _notify():
            notified()

        adapter._notify_fatal_error = _notify
        assert await adapter.connect() is True
        await created[0].socket_mode.stop()  # the SDK's start() returns: sockets gone
        for _ in range(20):
            await asyncio.sleep(0.01)
            if notified.called:
                break
        assert adapter._fatal_error_code == "SOCKET_CLOSED"
        assert adapter._fatal_error_retryable is True
        notified.assert_called_once()
