"""Immutable loopback capability and bounded HTTP worker lifetime."""
from __future__ import annotations

import atexit
import hmac
import threading
from http.server import ThreadingHTTPServer

from .handler import ChatCompletionsHandler
from .pool import pool

_proxy_server = None
_proxy_lock = threading.RLock()


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 16

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(10)
        return request, address

    def process_request(self, request, address):
        if not self.http_slots.acquire(False):
            # Do not allocate another handler thread for a slow/malicious client.
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.http_slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.http_slots.release()


def spawn_proxy(*, port, token, pool_initializer=None, runtime_guard=None, backend_pool=None):
    global _proxy_server
    if not isinstance(token, str) or not token.strip():
        raise ValueError('freemaxxing proxy requires a non-empty token')
    with _proxy_lock:
        if _proxy_server is not None:
            existing = _proxy_server
            if not hmac.compare_digest(token.encode(), existing.auth_token.encode()):
                raise RuntimeError('existing Freemaxxing listener has different authority')
            if pool_initializer is not None and existing.pool_initializer is not pool_initializer:
                raise RuntimeError('existing Freemaxxing listener has a different initializer')
            if backend_pool is not None and existing.backend_pool is not backend_pool:
                raise RuntimeError('existing Freemaxxing listener has a different pool')
            if runtime_guard is not None and existing.runtime_guard is not runtime_guard:
                raise RuntimeError('existing Freemaxxing listener has a different runtime guard')
            if port and port != existing.server_address[1]:
                raise RuntimeError('existing Freemaxxing listener has a different port')
            if not existing.worker_thread.is_alive():
                raise RuntimeError('Freemaxxing listener stopped; explicit restart required')
            return existing
        owner = backend_pool or pool
        server = BoundedServer(('127.0.0.1', int(port)), ChatCompletionsHandler)
        server.auth_token = token
        server.backend_pool = owner
        server.pool_initializer = pool_initializer
        server.pool_ready = pool_initializer is None
        server.runtime_guard = runtime_guard
        server.pool_init_lock = threading.Lock()
        server.inference_slots = threading.BoundedSemaphore(owner.limits.concurrency)
        server.http_slots = threading.BoundedSemaphore(owner.limits.concurrency + 4)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05},
                                  name='freemaxxing-proxy', daemon=True)
        server.worker_thread = thread
        _proxy_server = server
        thread.start()
        return server


def stop_proxy(server=None):
    global _proxy_server
    with _proxy_lock:
        target = server or _proxy_server
        if target is None:
            return
        target.shutdown()
        target.server_close()
        if target.worker_thread is not threading.current_thread():
            target.worker_thread.join(timeout=2)
        target.backend_pool.clear()
        if target is _proxy_server:
            _proxy_server = None


atexit.register(stop_proxy)
