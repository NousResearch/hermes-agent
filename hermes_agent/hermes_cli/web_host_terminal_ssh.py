"""Webapp SSH terminals use the saved connection's SSH shell, not remote ``serve`` HTTP.

The serving Webapp owns the PTY and its retention. Remote ``serve --isolated`` stays headless,
with no host-terminal endpoint; the shell rides the same SSH master as native Desktop's terminal.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from fastapi import WebSocket

from hermes_agent.hermes_cli import gateway_connections as store
from hermes_agent.hermes_cli.web_host_terminal_sessions import HostOwner, TerminalExpired


def _entry(path: Path, connection_id: str) -> dict | None:
    return next((c for c in store.read_registry(path)["connections"] if c["id"] == connection_id), None)


def _profile(ws: WebSocket) -> str:
    return (ws.query_params.get("profile") or "current").strip()


@dataclass(frozen=True)
class SshOwner(HostOwner):
    store_path: Path
    connection: dict
    profile: str

    def current(self) -> bool:
        # Pin the store path: the reaper and the PTY input worker do not run in a request's scope.
        try:
            current = _entry(self.store_path, self.connection["id"])
        except (store.ConnectionStoreError, store.ConnectionStoreBusy):
            return False  # An unreadable registry cannot authorize more input to the old host.
        return (super().current() and current is not None
                and not store.connection_dial_fields_changed(self.connection, current))

    def matches(self, ws: WebSocket) -> bool:
        selected = getattr(ws.state, "ssh_terminal", None)
        return (selected is not None and selected[0]["id"] == self.connection["id"]
                and selected[1] == self.store_path and _profile(ws) == self.profile)


def warm_tunnel(ws: WebSocket) -> None:
    """Dial before the registry's spawn lock: a cold connect would hold up every new terminal.

    The dial inside ``prepare_shell`` then takes the reuse path and keeps its fingerprint re-check."""
    from hermes_agent.hermes_cli.gateway_connections_ssh import ensure_tunnel

    connection, path = ws.state.ssh_terminal
    ensure_tunnel(connection, root=path.parent)


def prepare_shell(ws: WebSocket, identity: tuple[str, str]):
    from hermes_agent.hermes_cli.gateway_connections_ssh import interactive_shell_argv
    from hermes_agent.hermes_cli.web_host_terminal import resolve_argv
    from hermes_constants import get_process_hermes_home

    connection, path = ws.state.ssh_terminal
    home = get_process_hermes_home().resolve()
    remote_cwd = (ws.query_params.get("cwd") or "").strip()
    owner = SshOwner(identity, home, "ssh", remote_cwd, path, connection, _profile(ws))
    owner.admit(lambda: None)
    # The remote profile is an attachment scope, never a home resolved on this host.
    _argv, cwd, env, _name = resolve_argv(home=home)
    argv = interactive_shell_argv(connection, remote_cwd, root=path.parent)
    if not owner.current():
        raise TerminalExpired()
    return owner, argv, cwd, env


async def serve_ssh_terminal(ws: WebSocket, connection_id: str, admitted: str) -> bool:
    """After relay admission, serve an SSH terminal; False leaves other transports to the relay."""
    from hermes_agent.hermes_cli.web_host_terminal import request_allowed
    from hermes_agent.hermes_cli.web_host_terminal_sessions import host_terminal
    from hermes_agent.hermes_cli.web_server_chat import _PTY_BRIDGE_AVAILABLE, PtyUnavailableError, _pty_fail
    from hermes_agent.hermes_cli.web_server_surface import policy
    from hermes_agent.hermes_cli.web_session_sockets import track

    path = store.store_path()
    connection = await asyncio.to_thread(_entry, path, connection_id)
    if connection is None or connection["kind"] != "ssh":
        return False
    if not policy(ws.app.state).gateway_connections or not request_allowed():
        await ws.close(code=4403)
        return True
    ws.state.ssh_terminal = (connection, path)
    track(ws, admitted)
    await ws.accept()
    if not _PTY_BRIDGE_AVAILABLE:
        await _pty_fail(ws, PtyUnavailableError("Pseudo-terminal support is not installed on this host."),
                        surface="Terminal")
    else:
        await host_terminal(ws)
    return True
