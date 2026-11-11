"""Gateway-owned ACP transport. Each authenticated local editor gets its own projection."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
import os
from pathlib import Path
import stat

from acp_adapter.relay import HEADER_VERSION, SOCKET_NAME

logger = logging.getLogger(__name__)
_MAX_FRAME = 50 * 1024 * 1024  # ACP stdio's image-bearing frame limit
_MAX_HEADER = 64 * 1024


class HostedACPServer:
    def __init__(self, runner, home):
        self.runner = runner
        self.home = Path(home).resolve()
        self.path = self.home / SOCKET_NAME
        self.server = None
        self.tasks = set()
        self.identity = None

    async def start(self):
        from gateway.control_socket import _fits_sun_path
        if os.name == 'nt' or not _fits_sun_path(self.path):
            return False
        # ACP is optional; installations without its extra keep the stdio viewer.
        import acp
        if self.path.is_symlink():
            raise PermissionError('ACP socket cannot be a symlink')
        if self.path.exists():
            info = self.path.lstat()
            if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode):  # windows-footgun: ok — POSIX start only
                raise PermissionError('ACP socket path is not an owned socket')
            # The gateway has already acquired its exclusive runtime lock.
            self.path.unlink()
        # Bind synchronously so no task observes the process-wide umask change.
        import socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        mask = os.umask(0o177)
        try:
            sock.bind(str(self.path))
            os.chmod(self.path, 0o600)
            info = self.path.lstat()
            self.identity = (info.st_dev, info.st_ino)
            sock.listen()
            sock.setblocking(False)
        except BaseException:
            sock.close()
            raise
        finally:
            os.umask(mask)
        try:
            self.server = await asyncio.start_unix_server(self._serve, sock=sock, limit=_MAX_HEADER)
        except BaseException:
            sock.close()
            await self.stop()
            raise
        logger.info('Gateway ACP listening at %s', self.path)
        return True

    @asynccontextmanager
    async def _connect(self):
        from hermes_cli.gateway_runtime import attachable, discover_gateway_endpoint
        from hermes_cli.gateway_client import GatewayClientError, connect_gateway
        result = await asyncio.to_thread(discover_gateway_endpoint, self.home)
        if not attachable(result):
            raise GatewayClientError('Gateway is not ready for ACP')
        # Never ensure/start a replacement owner from inside the owner itself.
        async with connect_gateway(endpoint=result.endpoint) as client:
            yield client

    async def _serve(self, reader, writer):
        from hermes_cli.gateway_runtime_discovery import socket_peer_uid
        from gateway.run_idle_exit import attached_client
        task = asyncio.current_task()
        self.tasks.add(task)
        agent = connection = None
        try:
            if socket_peer_uid(writer.get_extra_info('socket')) != os.getuid():  # windows-footgun: ok — Unix listener only
                raise PermissionError('ACP peer uid mismatch')
            # Bound the handshake separately from the much larger ACP image frame.
            header = await asyncio.wait_for(reader.readuntil(b'\n'), 10)
            if len(header) > _MAX_HEADER:
                raise ValueError('ACP handshake too large')
            header = json.loads(header)
            if not isinstance(header, dict) or type(header.get('hermes_acp_relay')) is not int \
                    or header['hermes_acp_relay'] != HEADER_VERSION:
                raise ValueError('Unsupported ACP handshake')
            env, cwd = header.get('env'), header.get('cwd')
            if not isinstance(env, dict) or not all(isinstance(k, str) and k.startswith('HERMES_')
                    and isinstance(v, str) and '\0' not in v for k, v in env.items()):
                raise ValueError('Invalid ACP environment')
            if not isinstance(cwd, str) or not os.path.isabs(cwd) or '\0' in cwd:
                raise ValueError('Invalid ACP working directory')
            reader._limit = _MAX_FRAME
            from acp import AgentSideConnection
            from acp_adapter.gateway_server import GatewayACPAgent
            # Do not merge the gateway's launch flags into an unrelated editor's policy.
            agent = GatewayACPAgent(environ=env, connector=self._connect)
            with attached_client(self.runner):
                connection = AgentSideConnection(agent, writer, reader, listening=False, use_unstable_protocol=True)
                await connection.listen()
        except (ValueError, OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
            logger.warning('Rejected ACP connection', exc_info=True)
        except Exception:
            logger.exception('Hosted ACP connection failed')
        finally:
            try:
                await _close_projection(writer, connection, agent)
            finally:
                self.tasks.discard(task)

    async def stop(self):
        server, self.server = self.server, None
        if server is not None:
            server.close()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if server is not None:
            await server.wait_closed()
        if self.identity is not None:
            with suppress(FileNotFoundError):
                info = self.path.lstat()
                if (info.st_dev, info.st_ino) == self.identity:
                    self.path.unlink()
            self.identity = None


async def _close_projection(writer, connection, agent):
    # Abort before the SDK drains its sender: a client that stopped reading must not
    # hold the gateway's entire stop path hostage.
    writer.transport.abort()
    for close in (connection._conn.close if connection is not None else None,
                  agent.aclose if agent is not None else None, writer.wait_closed):
        if close is not None:
            try:
                await asyncio.wait_for(close(), 5)
            except Exception:
                logger.warning('ACP cleanup did not finish cleanly', exc_info=True)


async def start_hosted_acp(runner):
    from hermes_constants import get_process_hermes_home
    from acp_adapter.relay import disabled
    if disabled():
        return None
    server = HostedACPServer(runner, get_process_hermes_home())
    try:
        if await server.start():
            return server
    except ImportError:
        logger.debug('ACP extra not installed; hosted ACP disabled')
    except OSError:
        logger.warning('Hosted ACP unavailable; keeping stdio fallback', exc_info=True)
    await server.stop()
    return None
