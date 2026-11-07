"""Gateway SSH connections: a ControlMaster tunnel to a remote ``hermes serve``.

Server-side port of Desktop's SSH attach path (``apps/desktop/electron/ssh-connection.ts`` +
``remote-lifecycle.ts``). ``ensure_tunnel`` opens (or reuses) a ControlMaster to the registry
entry's host, starts or attaches a provably-owned ``serve --isolated`` there and forwards a
127.0.0.1-only local port to it. Linux/macOS on both ends.

Ownership is scoped to THIS gateway's connections store, not to a Desktop install: two clients
sharing one remote never reap each other's serve. The serve's session token lives in process memory
only; a fresh process cannot prove the old serve's token, so it reaps that (provably owned) serve
and spawns anew.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.gateway_connections_ssh_remote import (
    DEFAULT_READY_TIMEOUT_S,
    ConnectDeps,
    Connected,
    classify_ssh_reuse_proof,
    connect,
    disconnect_remote,
)
from hermes_cli.gateway_connections_ssh_transport import (
    SshConnection,
    SshEndpoint,
    SshReachError,
    control_socket_path,
    default_control_dir,
    pick_local_port,
    resolve_ssh_binary,
)

__all__ = ["SshReachError", "SshTarget", "close_all", "disconnect", "ensure_tunnel", "tunnel_status"]

HEALTH_TIMEOUT_S = 30.0
_HTTP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class SshTarget:
    base_url: str
    token: str
    local_port: int
    remote_port: int
    pid: int
    reused: bool
    hermes_version: str


@dataclass
class _Tunnel:
    ssh: SshConnection
    ownership_id: str
    connected: Connected
    host: str
    fingerprint: tuple


_tunnels: dict[str, _Tunnel] = {}
_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _connection_lock(connection_id: str) -> threading.Lock:
    with _registry_lock:
        return _locks.setdefault(connection_id, threading.Lock())


def store_root(root: Path | None = None) -> Path:
    """The gateway-connections store root: a profile home names its root (ui_preferences rule)."""
    if root is not None:
        return Path(os.path.realpath(root))
    from hermes_constants import get_hermes_home

    home = Path(os.path.abspath(get_hermes_home()))
    return (home.parent.parent if home.parent.name.lower() == "profiles" else home).resolve()


def ownership_id(root: Path, connection_id: str) -> str:
    return hashlib.sha256(f"hermes-gateway-connections\0{root}\0{connection_id}".encode()).hexdigest()[:32]


def _http_get(url: str, token: str = ""):
    import httpx

    headers = {"X-Hermes-Session-Token": token} if token else {}
    return httpx.get(url, headers=headers, timeout=_HTTP_TIMEOUT_S, trust_env=False)


def _health_ok(base_url: str) -> bool:
    import httpx

    try:
        response = _http_get(f"{base_url}/api/health")
    except httpx.HTTPError:
        return False
    return response.status_code == 200 and response.json().get("ok") is True


def _wait_for_health(base_url: str) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if _health_ok(base_url):
            return
        time.sleep(0.25)
    raise SshReachError("ready-timeout", f"The forwarded remote serve never answered /api/health ({HEALTH_TIMEOUT_S:.0f}s).")


def _probe_reuse_proof(base_url: str, token: str, spawn_nonce: str) -> str:
    response = _http_get(f"{base_url}/api/ssh/ownership", token)
    if response.status_code in (401, 403, 404):
        return "authenticated-stale"
    response.raise_for_status()
    return classify_ssh_reuse_proof(response.json(), spawn_nonce)


def _endpoint(connection: dict, root: Path, oid: str) -> SshEndpoint:
    host, user = str(connection.get("host") or ""), str(connection.get("user") or "")
    port = int(connection.get("port") or 22)
    key_path = str(connection.get("keyPath") or "")
    path = control_socket_path(user, host, port, default_control_dir(root), key_path=key_path, ownership_id=oid)
    return SshEndpoint(host=host, user=user, port=port, key_path=key_path, control_path=str(path))


def _fingerprint(connection: dict) -> tuple:
    return tuple(str(connection.get(k) or "") for k in ("host", "user", "port", "keyPath", "remoteHermesPath",
                                                         "remoteProfile"))


def _target(tunnel: _Tunnel) -> SshTarget:
    c = tunnel.connected
    return SshTarget(c.base_url, c.token, c.local_port, c.remote_port, c.pid, c.reused, c.hermes_version)


def _live_reuse(tunnel: _Tunnel) -> bool:
    if not tunnel.ssh.is_alive() or not _health_ok(tunnel.connected.base_url):
        return False
    import httpx

    try:
        verdict = _probe_reuse_proof(tunnel.connected.base_url, tunnel.connected.token, tunnel.connected.spawn_nonce)
    except httpx.HTTPError:
        return False
    return verdict == "authenticated-ok"


def _teardown(tunnel: _Tunnel) -> None:
    c = tunnel.connected
    tunnel.ssh.cancel_forward(c.local_port, c.remote_port)
    try:
        disconnect_remote(tunnel.ssh, tunnel.ownership_id)
    finally:
        tunnel.ssh.close()


def ensure_tunnel(connection: dict, *, root: Path | None = None) -> SshTarget:
    """A live tunnel to the connection's remote serve; synchronous (run it in a threadpool)."""
    connection_id = str(connection.get("id") or "")
    if not connection_id:
        raise SshReachError("unknown", "SSH connection entry has no id.")
    with _connection_lock(connection_id):
        existing = _tunnels.get(connection_id)
        if existing and existing.fingerprint == _fingerprint(connection) and _live_reuse(existing):
            return _target(existing)
        if existing:
            _tunnels.pop(connection_id, None)
            if existing.fingerprint != _fingerprint(connection):
                _teardown(existing)
                existing = None
            else:
                # Same target, failed liveness: drop only this forward; the redial below
                # exec-verifies (or evicts) the shared master and re-proves the serve with its token.
                existing.ssh.cancel_forward(existing.connected.local_port, existing.connected.remote_port)
        return _bootstrap(connection, store_root(root), existing)


def _bootstrap(connection: dict, root: Path, previous: _Tunnel | None) -> SshTarget:
    connection_id = str(connection["id"])
    oid = ownership_id(root, connection_id)
    ssh = SshConnection(_endpoint(connection, root, oid), ssh_binary=resolve_ssh_binary())
    ssh.open()
    reuse_token = previous.connected.token if previous else ""
    try:
        connected = connect(ConnectDeps(
            ssh=ssh, ownership_id=oid, pick_local_port=pick_local_port, wait_for_health=_wait_for_health,
            probe_reuse_proof=_probe_reuse_proof, profile=str(connection.get("remoteProfile") or ""),
            remote_hermes_path=str(connection.get("remoteHermesPath") or ""), reuse_token=reuse_token,
            ready_timeout_s=DEFAULT_READY_TIMEOUT_S))
    except BaseException:
        ssh.close()
        raise
    tunnel = _Tunnel(ssh, oid, connected, str(connection.get("host") or ""), _fingerprint(connection))
    _tunnels[connection_id] = tunnel
    return _target(tunnel)


def disconnect(connection_id: str) -> bool:
    """Cancel the forward, kill the provably-owned remote serve, close the master."""
    with _connection_lock(connection_id):
        tunnel = _tunnels.pop(connection_id, None)
        if tunnel is None:
            return False
        _teardown(tunnel)
        return True


def close_all() -> None:
    for connection_id in list(_tunnels):
        try:
            disconnect(connection_id)
        except SshReachError:
            continue


def tunnel_status(connection_id: str) -> dict | None:
    tunnel = _tunnels.get(connection_id)
    if tunnel is None:
        return None
    c = tunnel.connected
    return {"localPort": c.local_port, "remotePort": c.remote_port, "pid": c.pid, "reused": c.reused,
            "hermesVersion": c.hermes_version, "host": tunnel.host}
