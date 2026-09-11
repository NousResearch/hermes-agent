"""Loopback-only, capability-authenticated binary WayVNC bridge.

Hermes authenticates capability issuance through its plugin REST router. This
separate listener deliberately does not rely on HTTP middleware protecting WS.
"""

import asyncio
import contextlib
import socket
import threading
import time
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

from .rfb import ClientFilter
from .viewer_auth import Tickets

_profile_servers = {}
_profile_lock = threading.RLock()


def get_profile_viewer(home):
    """Lazy process-local listener with profile-wide takeover/revocation state."""
    from .manager import Manager

    key = str(Path(home).expanduser().resolve())
    with _profile_lock:
        server = _profile_servers.get(key)
        if server is None:
            manager = Manager(key)

            def resolve(realm_id):
                from .lifecycle import RealmError, validate_live

                if realm_id.startswith("v-"):
                    # A VM realm's viewer endpoint is QEMU's VNC unix socket.
                    # Same RFB protocol, same ownership rules, different manager.
                    from .vm_manager import VmManager, VmError

                    try:
                        return VmManager(key).validate(realm_id)
                    except (RealmError, VmError, OSError, ValueError):
                        return None
                try:
                    with manager.registry.lock():
                        record = manager.registry.get(realm_id)
                        if record["status"] != "running":
                            return None
                        # Read-only: viewing must not renew the realm's idle TTL.
                        validate_live(record)
                        return record
                except (RealmError, OSError):
                    return None

            server = ViewerServer(resolve, state_dir=Path(key) / "realms/viewer")
            _profile_servers[key] = server
        return server


def close_profile_viewer(home):
    key = str(Path(home).expanduser().resolve())
    with _profile_lock:
        server = _profile_servers.pop(key, None)
    if server:
        server.stop()


def _peer_in_realm_scope(pid, realm):
    """Whether *pid* runs inside this VM realm's own systemd scope.

    A labwc realm proves socket ownership by matching the listener against the
    worker/VNC processes it started. A VM realm has neither: QEMU is the
    listener and systemd owns it. The equivalent proof is cgroup membership —
    the realm's record pins the scope at launch, so a PID inside it is that
    realm's QEMU and nothing else. Reading ``/proc`` also closes the PID-reuse
    window that comparing bare PIDs would leave open.
    """
    scope = realm.get("cgroup")
    if not scope:
        return False
    try:
        with open(f"/proc/{int(pid)}/cgroup", encoding="utf-8") as handle:
            current = handle.read()
    except (OSError, ValueError):
        return False
    return any(line.split(":", 2)[-1].strip() == scope
               for line in current.splitlines() if line.strip())


class ViewerServer:
    def __init__(self, resolve_realm, *, state_dir=None):
        from .viewer_state import ControlAuthority

        self.authority = ControlAuthority(state_dir) if state_dir is not None else None
        self.resolve_realm = resolve_realm
        self.tickets = Tickets()
        self.origin = ""
        self._server = None
        self._thread = None
        self._controls = {}
        self._lock = threading.RLock()
        self.app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        self.app.websocket("/api/realms/{realm_id}/vnc")(self._vnc)
        web = Path(__file__).parent / "web"
        self.app.mount("/assets", StaticFiles(directory=web), name="assets")

        @self.app.get("/realms/{realm_id}/view")
        async def view(realm_id: str):
            # Public generic shell contains no realm data or capabilities.
            return FileResponse(web / "viewer.html")

        @self.app.middleware("http")
        async def headers(request, call_next):
            if request.headers.get("host") != self.origin.removeprefix("http://"):
                return JSONResponse({"detail": "Invalid host"}, status_code=400)
            response = await call_next(request)
            response.headers.update(
                {
                    "Referrer-Policy": "no-referrer",
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:; object-src 'none'; base-uri 'none'",
                }
            )
            return response

    def start(self):
        with self._lock:
            if self._thread is not None:
                return self
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(64)
            self.origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
            config = uvicorn.Config(
                self.app,
                host="127.0.0.1",
                log_level="error",
                access_log=False,
                ws="websockets-sansio",
                ws_max_size=1_048_600,
                timeout_graceful_shutdown=2,
            )
            self._server = uvicorn.Server(config)
            self._thread = threading.Thread(
                target=self._server.run,
                kwargs={"sockets": [listener]},
                name="realm-viewer",
                daemon=True,
            )
            self._thread.start()
        deadline = time.monotonic() + 10
        while (
            not self._server.started
            and self._thread.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        if not self._server.started:
            self.stop()
            raise RuntimeError("Viewer listener did not start")
        return self

    def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(5)
            if self._thread.is_alive():
                raise RuntimeError("Viewer listener did not stop")
        self._thread = None
        self._server = None
        with self._lock:
            self._controls.clear()

    def issue(self, realm_id, *, can_control=False, ttl=900):
        realm = self.resolve_realm(realm_id)
        if realm is None:
            raise ValueError("Realm is not running")
        return self.tickets.issue(
            realm_id, self._ticket_generation(realm), can_control=can_control, ttl=ttl
        )

    def _ticket_generation(self, realm):
        epoch = self.authority.epoch(realm["id"]) if self.authority else 0
        return f"{realm['generation']}:{epoch}"

    def revoke(self, realm_id):
        with self._lock:
            self.tickets.revoke(realm_id)
            if self.authority:
                self.authority.revoke(realm_id)

    def is_controlled(self, realm_id):
        if self.authority:
            return self.authority.controlled(realm_id)
        with self._lock:
            return realm_id in self._controls

    @staticmethod
    def _endpoint(realm):
        import os
        import stat

        endpoint = realm.get("vnc_socket")
        if endpoint is not None:
            try:
                runtime = Path(realm["runtime_dir"])
                path = Path(endpoint)
                parent = runtime.lstat()
                info = path.lstat()
                if (
                    runtime.is_symlink()
                    or not stat.S_ISDIR(parent.st_mode)
                    or parent.st_uid != os.getuid()  # windows-footgun: ok — runtime package rejects non-Linux hosts
                    or stat.S_IMODE(parent.st_mode) != 0o700
                ):
                    return None
                if (
                    path.parent != runtime
                    or not stat.S_ISSOCK(info.st_mode)
                    or info.st_uid != os.getuid()  # windows-footgun: ok — runtime package rejects non-Linux hosts
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    return None
                return ("unix", str(path), info.st_dev, info.st_ino)
            except (KeyError, OSError, TypeError):
                return None
        # Legacy manual spike transport only; product managers use private Unix.
        port = realm.get("vnc_port")
        if type(port) is int and 1024 <= port <= 65535:
            return ("tcp", port)
        return None

    async def _vnc(self, ws: WebSocket, realm_id: str):
        # A page served by this exact listener is the only allowed browser origin.
        if ws.headers.get("origin") != self.origin or ws.headers.get(
            "host"
        ) != self.origin.removeprefix("http://"):
            await ws.close(code=1008)
            return
        protocols = [
            p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",")
        ]
        capabilities = [p[6:] for p in protocols if p.startswith("realm.")]
        realm = self.resolve_realm(realm_id)
        control = ws.query_params.get("control") == "1"
        if len(capabilities) != 1 or realm is None:
            await ws.close(code=1008)
            return
        token = capabilities[0]
        generation = self._ticket_generation(realm)
        if not self.tickets.check(token, realm_id, generation, control=control):
            await ws.close(code=1008)
            return
        endpoint = self._endpoint(realm)
        if endpoint is None:
            await ws.close(code=1008)
            return
        lease = uuid.uuid4().hex
        with self._lock:
            taken = control and (
                not self.authority.acquire(realm_id, generation, lease)
                if self.authority
                else realm_id in self._controls
            )
            if control and not taken:
                self._controls[realm_id] = lease
        if taken:
            await ws.close(code=1008)
            return
        writer = None
        tasks = []
        try:
            connection = (
                asyncio.open_unix_connection(endpoint[1])
                if endpoint[0] == "unix"
                else asyncio.open_connection("127.0.0.1", endpoint[1])
            )
            reader, writer = await asyncio.wait_for(connection, 5)
            if endpoint[0] == "unix":
                import os
                import struct
                from .lifecycle import identity

                peer = writer.get_extra_info("socket")
                pid, uid, _ = struct.unpack(
                    "3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                )
                process = identity(pid)
                # SO_PEERCRED identifies the listener creator. The worker
                # prebinds the product socket and passes it to WayVNC; a
                # directly bound WayVNC listener identifies the VNC process.
                # A VM realm has no such processes: the listener is QEMU, and
                # its proof of belonging is the realm's own systemd scope.
                owners = realm.get("processes", {})
                known = (
                    _peer_in_realm_scope(pid, realm) if realm.get("kind") == "omarchy-vm"
                    else process is not None
                    and process in (owners.get("worker"), owners.get("vnc"))
                )
                if (
                    uid != os.getuid()  # windows-footgun: ok — runtime package rejects non-Linux hosts
                    or process is None
                    or not known
                    or self._endpoint(realm) != endpoint
                ):
                    raise ValueError("Realm VNC peer ownership changed")
            await ws.accept(subprotocol="binary")
            parser = ClientFilter(control=control)

            async def receive():
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    data = message.get("bytes")
                    if data is None:
                        raise ValueError("Binary VNC frames required")
                    forwarded = parser.feed(data)
                    if forwarded:
                        current = self.resolve_realm(realm_id)
                        if (
                            current is None
                            or self._ticket_generation(current) != generation
                            or self._endpoint(current) != endpoint
                        ):
                            return
                        with self._lock:
                            guard = (
                                self.authority.forwarding(
                                    realm_id, generation, lease if control else None
                                )
                                if self.authority
                                else contextlib.nullcontext(
                                    not control or self._controls.get(realm_id) == lease
                                )
                            )
                            with guard as authorized:
                                if not authorized or not self.tickets.check(
                                    token, realm_id, generation, control=control
                                ):
                                    return
                                writer.write(forwarded)
                        await writer.drain()

            async def transmit():
                while True:
                    data = await reader.read(65536)
                    if not data:
                        return
                    try:
                        await ws.send_bytes(data)
                    except RuntimeError as exc:
                        # uvloop can close the transport between receive and send.
                        if "handler is closed" not in str(exc):
                            raise
                        return

            async def lifetime():
                while True:
                    await asyncio.sleep(0.25)
                    current = self.resolve_realm(realm_id)
                    if (
                        current is None
                        or self._ticket_generation(current) != generation
                        or self._endpoint(current) != endpoint
                    ):
                        return
                    if (
                        control
                        and self.authority
                        and not self.authority.refresh(realm_id, lease)
                    ):
                        return
                    if not self.tickets.check(
                        token, realm_id, generation, control=control
                    ):
                        return

            tasks = [asyncio.create_task(f()) for f in (receive, transmit, lifetime)]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (
            WebSocketDisconnect,
            ConnectionError,
            OSError,
            asyncio.TimeoutError,
            ValueError,
        ) as exc:
            import logging

            logging.getLogger(__name__).debug("Viewer protocol closed: %s", exc)
            # Invalid wire frames and disconnected/stopped private listeners close,
            # never reconnect to a different realm or to the host daemon.
            pass
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if writer:
                writer.close()
                with contextlib.suppress(ConnectionError):
                    await writer.wait_closed()
            if control and self.authority:
                self.authority.release(realm_id, lease)
            with self._lock:
                if self._controls.get(realm_id) == lease:
                    self._controls.pop(realm_id, None)
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await ws.close()
