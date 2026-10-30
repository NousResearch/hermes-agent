"""Bound stalled dashboard HTTP writes, including close-with-data-pending.

A loopback peer can be an SSH forward whose remote reader has gone away.
Keep-alive/ping timers do not cover a queued HTTP response, and close() on a
Windows proactor transport waits for the pending overlapped send to finish.
"""
import os
import socket
import sys

from uvicorn.protocols.http.h11_impl import H11Protocol


class BoundedH11Protocol(H11Protocol):
    write_stall_timeout = 15.0
    _write_stall_timer = None

    def connection_made(self, transport):
        super().connection_made(transport)
        # Windows buffered sends can complete into AfdB before the peer reads,
        # bypassing asyncio backpressure. Zero requests unbuffered WSASend there.
        # Unconditional SO_SNDBUF=0 makes every send() on a non-blocking Windows socket
        # complete synchronously into the peer (unbuffered WSASend), so a multi-megabyte
        # /api/plugins/kanban/board reply over an SSH tunnel can hold the event loop for
        # tens of seconds, dropping websockets and timing out persistence receipts. Keep
        # kernel buffering by default; asyncio's 64 KB high-water mark and the stall timer
        # below still apply backpressure and abort a non-reading peer.
        sock = transport.get_extra_info('socket')
        if sock is not None and sys.platform == 'win32' and os.environ.get('HERMES_DASHBOARD_UNBUFFERED_SEND') == '1':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 0)
        transport.set_write_buffer_limits(high=64 * 1024, low=16 * 1024)

    def pause_writing(self):
        super().pause_writing()
        if self._write_stall_timer is None:
            self._write_stall_timer = self.loop.call_later(
                self.write_stall_timeout, self._abort_stalled_write)

    def resume_writing(self):
        self._cancel_write_stall_timer()
        super().resume_writing()

    def connection_lost(self, exc):
        self._cancel_write_stall_timer()
        super().connection_lost(exc)

    def _cancel_write_stall_timer(self):
        if self._write_stall_timer is not None:
            self._write_stall_timer.cancel()
            self._write_stall_timer = None

    def _abort_stalled_write(self):
        self._write_stall_timer = None
        self.logger.warning('Closing dashboard HTTP connection: stalled response write')
        # abort, not close: close waits for a non-reading tunnel to drain on Windows.
        self.transport.abort()
