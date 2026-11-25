"""Process-wide A2A listener registry: one stdlib HTTP listener per ``(host, port)``, many adapters.

A multiplexed gateway builds one A2A adapter per served profile, and every profile home imports
its own copy of this plugin (``hermes_plugins.platforms__a2a__home_<hash>``), so module globals in
``adapter.py`` are per profile. This module is imported by its canonical name
(``plugins.platforms.a2a.listeners``) and is therefore ONE module per process: it is where the
adapters of different profiles find the listener another one already bound.

Mounts follow the shared-ingress convention (``gateway/platforms/shared_ingress.py``):
``""`` serves the un-prefixed root (the default profile, or a standalone gateway) and ``<profile>``
serves a multiplexed secondary at ``/p/<profile>/``. A path whose profile has no mount gets a 404
from the handler, never another profile's adapter. Each mounted adapter still authenticates its own
requests with its own security context; mounts share only the socket.
"""

from __future__ import annotations

import errno
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

PREFIX = "p"  # /p/<profile>/ — reserved as a served-agent path segment

_lock = threading.Lock()
_listeners: dict[tuple[str, int], Listener] = {}


@dataclass(eq=False)
class Listener:
    host: str
    port: int
    server: Any
    thread: Optional[threading.Thread] = None
    mounts: dict[str, Any] = field(default_factory=dict)


def _key(host: str, port: int) -> tuple[str, int]:
    return (str(host or "").strip().lower(), int(port))


def mount_prefix(slug: str) -> str:
    """URL path prefix of a mount ("" for the root mount)."""
    return f"/{PREFIX}/{urllib.parse.quote(slug, safe='')}" if slug else ""


def mount(host: str, port: int, slug: str, adapter: Any, make_server: Callable[[str, int], Any]) -> Listener:
    """Mount ``adapter`` at ``slug`` on the process listener for ``host:port``, binding it first when
    no adapter of this process holds it. Raises ``OSError`` (EADDRINUSE) when the port belongs to
    another process, or when another live adapter already owns this mount."""
    key = _key(host, port)
    with _lock:
        listener = _listeners.get(key)
        if listener is None:
            server = make_server(host, port)  # OSError when another process holds the port
            listener = Listener(host=host, port=port, server=server)
            server.a2a_listener = listener
            listener.thread = threading.Thread(target=server.serve_forever, name="a2a-http", daemon=True)
            listener.thread.start()
            _listeners[key] = listener
        holder = listener.mounts.get(slug)
        if holder is not None and holder is not adapter:
            where = mount_prefix(slug) or "/"
            raise OSError(errno.EADDRINUSE, f"{host}:{port}{where} is already served by another A2A adapter "
                                            f"in this process ({getattr(holder, '_active_profile', '?')})")
        listener.mounts[slug] = adapter
        return listener


def unmount(listener: Optional[Listener], slug: str, adapter: Any) -> None:
    """Drop ``adapter``'s mount; the last mount out closes the socket."""
    if listener is None:
        return
    with _lock:
        if listener.mounts.get(slug) is adapter:
            del listener.mounts[slug]
        if listener.mounts:
            return
        key = _key(listener.host, listener.port)
        if _listeners.get(key) is listener:
            del _listeners[key]
        # Inside the lock: a remount of this port must not race the old socket's close.
        try:
            listener.server.shutdown()
        finally:
            listener.server.server_close()


def resolve(listener: Listener, raw_path: str) -> tuple[Optional[Any], str]:
    """``(adapter, path the adapter sees)`` for a request path; adapter None = no such mount."""
    parts = urllib.parse.urlsplit(raw_path or "/")
    segments = (parts.path or "/").split("/", 3)  # ["", "p", "<profile>", "rest"]
    if len(segments) >= 3 and segments[1] == PREFIX and segments[2]:
        slug = urllib.parse.unquote(segments[2])
        rest = "/" + (segments[3] if len(segments) > 3 else "")
        query = f"?{parts.query}" if parts.query else ""
        return listener.mounts.get(slug), rest + query
    return listener.mounts.get(""), raw_path or "/"


def profiles(listener: Optional[Listener]) -> list[str]:
    """The profiles mounted under ``/p/`` on ``listener``, sorted."""
    if listener is None:
        return []
    with _lock:
        return sorted(slug for slug in listener.mounts if slug)


def local_url(profile: str, exclude: Any = None) -> Optional[str]:
    """``http://host:port/p/<profile>/`` of a profile mounted on any listener in this process, so a
    served profile can call another by name (``default`` is the root mount). None when no listener
    mounts it (or only ``exclude``)."""
    slug = "" if profile == "default" else profile
    if not slug and profile != "default":
        return None
    with _lock:
        for listener in _listeners.values():
            adapter = listener.mounts.get(slug)
            if adapter is not None and adapter is not exclude:
                host = listener.host
                host = "127.0.0.1" if host in ("", "0.0.0.0") else "::1" if host == "::" else host
                host = f"[{host}]" if ":" in host else host
                return f"http://{host}:{listener.port}{mount_prefix(slug)}/"
    return None
