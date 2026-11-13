"""The Screen pane dials the RFB socket a sandbox start marker names.

A pane is served by the dashboard/serve process, which never started the screen inside the terminal
backend, so the exec relay — built from that process's own sandbox environment — cannot be built for a
backend that publishes its socket on a reachable filesystem. The marker's socket is dialled directly,
a socket file whose Xvnc is gone falls through, and everything without a marker keeps the old paths.
"""

import asyncio
import json
import socket

import pytest

from hermes_cli.web_routers import display


async def _serve(path, payload):
    """An RFB stand-in that writes ``payload`` at whichever client connects first."""

    async def handler(_reader, writer):
        writer.write(payload)
        await writer.drain()

    return await asyncio.start_unix_server(handler, path=str(path))


def _profile_home(tmp_path, sandbox_dir=None):
    home = tmp_path / "home"
    (home / "bot-desktop").mkdir(parents=True)
    if sandbox_dir is not None:
        marker = home / "bot-desktop" / "sandbox.json"
        marker.write_text(json.dumps({"dir": str(sandbox_dir)}), encoding="utf-8")
    return home


def _stale_socket_file(path):
    """A socket file nothing listens on — what a killed Xvnc leaves behind."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.close()


def test_the_pane_dials_the_socket_the_start_marker_names(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    home = _profile_home(tmp_path, sandbox)

    async def scenario():
        server = await _serve(sandbox / "rfb.sock", b"RFB 003.008\n")
        try:
            reader, writer, relay = await display._open_rfb(home)
            assert relay is None, "a socket carries the stream itself, so there is no relay process"
            assert await reader.readexactly(12) == b"RFB 003.008\n"
            writer.close()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_a_socket_file_left_by_a_dead_screen_falls_through_to_the_gateway_socket(tmp_path, monkeypatch):
    from tools.bot_desktop import runtime

    monkeypatch.setattr(runtime, "sandbox_screen_running", lambda: False)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    _stale_socket_file(sandbox / "rfb.sock")
    home = _profile_home(tmp_path, sandbox)

    async def scenario():
        server = await _serve(home / "bot-desktop" / "rfb.sock", b"gateway\n")
        try:
            reader, writer, relay = await display._open_rfb(home)
            assert relay is None
            assert await reader.readexactly(8) == b"gateway\n"
            writer.close()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_a_gateway_hosted_screen_is_opened_exactly_as_before(tmp_path, monkeypatch):
    from tools.bot_desktop import runtime

    monkeypatch.setattr(runtime, "sandbox_screen_running", lambda: False)
    home = _profile_home(tmp_path)

    async def scenario():
        server = await _serve(home / "bot-desktop" / "rfb.sock", b"gateway\n")
        try:
            reader, writer, relay = await display._open_rfb(home)
            assert relay is None
            assert await reader.readexactly(8) == b"gateway\n"
            writer.close()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_nothing_running_is_still_an_oserror(tmp_path, monkeypatch):
    from tools.bot_desktop import runtime

    monkeypatch.setattr(runtime, "sandbox_screen_running", lambda: False)
    home = _profile_home(tmp_path)
    with pytest.raises(OSError, match="rfb.sock missing"):
        asyncio.run(display._open_rfb(home))
