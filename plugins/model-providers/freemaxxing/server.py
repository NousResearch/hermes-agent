"""Freemaxxing loopback-listener lifecycle."""

from __future__ import annotations

import atexit
import hmac
import logging
import threading
from http.server import ThreadingHTTPServer
from typing import Callable, Optional

from .handler import ChatCompletionsHandler

logger = logging.getLogger("freemaxxing.proxy")

_proxy_server: Optional[ThreadingHTTPServer] = None
_proxy_lock = threading.Lock()

def spawn_proxy(
    *,
    port: int,
    token: str,
    pool_initializer: Optional[Callable[[], None]] = None,
) -> ThreadingHTTPServer:
    """Start one authenticated listener; auth mode cannot change in-place."""
    global _proxy_server
    token = str(token or "")
    if not token:
        raise ValueError("freemaxxing proxy requires a non-empty auth token")

    with _proxy_lock:
        if _proxy_server is not None:
            actual = int(_proxy_server.server_address[1])
            if port and port != actual:
                logger.warning(
                    "freemaxxing: existing proxy uses port %d; requested %d",
                    actual,
                    port,
                )
            existing_token = str(
                getattr(_proxy_server, "auth_token", "") or ""
            )
            if not hmac.compare_digest(token, existing_token):
                raise RuntimeError(
                    "freemaxxing proxy already runs with a different local "
                    "authentication token"
                )
            existing_initializer = getattr(
                _proxy_server,
                "pool_initializer",
                None,
            )
            if (
                pool_initializer is not None
                and existing_initializer is not None
                and pool_initializer is not existing_initializer
            ):
                raise RuntimeError(
                    "freemaxxing proxy already runs with a different pool "
                    "initializer"
                )
            return _proxy_server

        server = ThreadingHTTPServer(
            ("127.0.0.1", int(port)),
            ChatCompletionsHandler,
        )
        server.daemon_threads = True
        server.auth_token = token
        server.pool_initializer = pool_initializer
        server.pool_ready = pool_initializer is None
        server.pool_init_lock = threading.Lock()

        thread = threading.Thread(
            target=server.serve_forever,
            name="freemaxxing-proxy",
            daemon=True,
        )
        server.worker_thread = thread
        thread.start()
        _proxy_server = server
        logger.info(
            "freemaxxing: proxy listening on 127.0.0.1:%d",
            server.server_address[1],
        )
        return server


def stop_proxy(server: Optional[ThreadingHTTPServer] = None) -> None:
    global _proxy_server
    with _proxy_lock:
        target = server or _proxy_server
        if target is None:
            return
        try:
            target.shutdown()
            target.server_close()
        except Exception as exc:
            logger.debug("freemaxxing: proxy shutdown error: %s", exc)
        thread = getattr(target, "worker_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if target is _proxy_server:
            _proxy_server = None


atexit.register(stop_proxy)
