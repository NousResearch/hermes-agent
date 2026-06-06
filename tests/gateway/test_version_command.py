"""Tests for gateway /version command."""

import asyncio

from hermes_cli import __release_date__, __version__


def test_gateway_version_command_returns_release_line():
    from gateway.run import GatewayRunner

    result = asyncio.run(GatewayRunner._handle_version_command(None, None))  # type: ignore[arg-type]
    assert result == f"Hermes Agent v{__version__} ({__release_date__})"
