"""#91564: Desktop serve must not reload MCP when a live gateway already owns them.

The scheduled gateway's api_server is not the Desktop /api/ws control plane,
so Desktop still spawns serve. It must skip configured-MCP discovery when
gateway.status proves a live process incarnation for this HERMES_HOME.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli.mcp_startup import desktop_serve_should_skip_configured_mcp


def test_standalone_serve_never_skips_mcp():
    assert (
        desktop_serve_should_skip_configured_mcp(
            desktop_env=None,
            record={"pid": 9, "start_time": 1, "hermes_home": "/h"},
            current_home=Path("/h"),
            pid_is_live=lambda _r: True,
        )
        is False
    )


def test_desktop_skips_mcp_when_gateway_incarnation_is_live():
    home = Path("/tmp/hermes-home")
    assert (
        desktop_serve_should_skip_configured_mcp(
            desktop_env="1",
            record={"pid": 14504, "start_time": 100, "hermes_home": str(home)},
            current_home=home,
            pid_is_live=lambda _r: True,
        )
        is True
    )


def test_desktop_does_not_skip_when_pid_incarnation_is_dead():
    home = Path("/tmp/hermes-home")
    assert (
        desktop_serve_should_skip_configured_mcp(
            desktop_env="1",
            record={"pid": 14504, "start_time": 100, "hermes_home": str(home)},
            current_home=home,
            pid_is_live=lambda _r: False,
        )
        is False
    )


def test_desktop_does_not_skip_when_record_home_is_a_different_profile():
    assert (
        desktop_serve_should_skip_configured_mcp(
            desktop_env="1",
            record={"pid": 1, "start_time": 1, "hermes_home": "/other"},
            current_home=Path("/mine"),
            pid_is_live=lambda _r: True,
        )
        is False
    )


def test_desktop_does_not_skip_without_a_runtime_record():
    assert (
        desktop_serve_should_skip_configured_mcp(
            desktop_env="1",
            record=None,
            current_home=Path("/h"),
            pid_is_live=lambda _r: True,
        )
        is False
    )
