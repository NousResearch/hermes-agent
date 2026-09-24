"""Gateway startup listener reporting: actual sockets, startup-only summary, console notice."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from gateway.config import Platform  # noqa: E402
from gateway.platforms.shared_ingress import bind_listener, bound_site_endpoints  # noqa: E402
from gateway.run_startup import GatewayStartupMixin  # noqa: E402


class _Socket:
    def __init__(self, address):
        self._address = address

    def getsockname(self):
        return self._address


def test_bound_site_endpoints_uses_actual_dual_stack_socket_addresses() -> None:
    site = SimpleNamespace(
        _server=SimpleNamespace(
            sockets=[_Socket(("127.0.0.1", 8642)), _Socket(("::1", 8642, 0, 0))]
        )
    )

    assert bound_site_endpoints(site, "127.0.0.1", 0) == (
        "127.0.0.1:8642",
        "[::1]:8642",
    )


def test_bound_site_endpoints_falls_back_without_socket_introspection() -> None:
    assert bound_site_endpoints(SimpleNamespace(), "::", 8644) == ("[::]:8644",)
    assert bound_site_endpoints(SimpleNamespace(), None, 0) == ()


def test_startup_listener_summary_is_info_operator_notice(caplog) -> None:
    runner = SimpleNamespace(
        adapters={
            Platform.WEBHOOK: SimpleNamespace(
                _bound_listener_endpoints=("0.0.0.0:8644", "[::]:8644")
            ),
            Platform.API_SERVER: SimpleNamespace(
                _bound_listener_endpoints=("127.0.0.1:8642",)
            ),
            Platform.TELEGRAM: SimpleNamespace(_bound_listener_endpoints=()),
        }
    )

    with caplog.at_level(logging.INFO):
        GatewayStartupMixin._log_startup_listeners(runner)

    record = next(r for r in caplog.records if r.getMessage().startswith("Gateway listeners:"))
    assert record.levelno == logging.INFO
    assert getattr(record, "gateway_console_notice", False) is True
    assert record.getMessage() == (
        "Gateway listeners: api_server=127.0.0.1:8642; "
        "webhook=0.0.0.0:8644, [::]:8644"
    )


def test_startup_listener_summary_reports_physical_sockets_not_multiplex_routes(caplog) -> None:
    runner = SimpleNamespace(
        adapters={
            Platform.API_SERVER: SimpleNamespace(
                _bound_listener_endpoints=("127.0.0.1:8642",)
            )
        },
        _profile_adapters={
            "coder": {
                Platform.LINE: SimpleNamespace(
                    _shared_listener_profile="coder",
                    _bound_listener_endpoints=(),
                )
            },
            "maintainer": {
                Platform.TEAMS: SimpleNamespace(
                    _shared_listener_profile="maintainer",
                    _bound_listener_endpoints=(),
                )
            },
        },
    )

    with caplog.at_level(logging.INFO):
        GatewayStartupMixin._log_startup_listeners(runner)

    records = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Gateway listeners:")]
    assert records == ["Gateway listeners: api_server=127.0.0.1:8642"]
    assert "/p/coder/" not in records[0]
    assert "/p/maintainer/" not in records[0]


@pytest.mark.asyncio
async def test_bind_listener_records_the_actual_ephemeral_port() -> None:
    adapter = SimpleNamespace(_shared_listener_profile=None)
    app = web.Application()
    runner = await bind_listener(adapter, app, "127.0.0.1", 0, "/health")
    assert runner is not None
    try:
        endpoints = adapter._bound_listener_endpoints
        assert len(endpoints) == 1
        host, port = endpoints[0].rsplit(":", 1)
        assert host == "127.0.0.1"
        assert 0 < int(port) <= 65535
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_shared_listener_secondary_is_not_reported_as_a_bound_port() -> None:
    adapter = SimpleNamespace(
        _shared_listener_profile="coder",
        gateway_runner=SimpleNamespace(adapters={}),
        platform=SimpleNamespace(value="line"),
    )
    app = web.Application()

    runner = await bind_listener(adapter, app, "127.0.0.1", 9999, "/line/webhook")

    assert runner is None
    assert getattr(adapter, "_bound_listener_endpoints", ()) == ()
