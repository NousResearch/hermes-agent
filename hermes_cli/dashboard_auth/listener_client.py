"""The Nous Portal client the listener this process serves signs in with instead of the configured one.

The gateway's Webapp listener signs browsers in with the client Web Access registered for it
(``hermes_cli.webapp_access.sign_in_client``), so the dashboard's ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``
keeps the redirect URI its own sign-in needs. Plugin discovery imports the bundled Nous sign-in
plugin under a module name of its own, so the plugin and the gateway meet here: the plugin offers
its provider swap when discovery loads it (:func:`offer`), and reads :func:`current` whenever it
builds its provider, a plugin reload included.
"""
from __future__ import annotations

from typing import Callable, Optional

_current: Optional[tuple[str, str]] = None
_swap: Optional[Callable[[], Callable[[], None]]] = None


def current() -> Optional[tuple[str, str]]:
    """``(client_id, portal_url)`` while a listener signs in with its own client, else None."""
    return _current


def offer(swap: Callable[[], Callable[[], None]]) -> None:
    """Called by the Nous sign-in plugin as discovery loads it: ``swap()`` puts its provider on the
    :func:`current` client in place of the registered one and returns the undo."""
    global _swap
    _swap = swap


def use(client_id: str, portal_url: str) -> Optional[Callable[[], None]]:
    """Sign in with *client_id* on *portal_url* until the returned undo runs, which puts the
    configured provider (or none) back. None, changing nothing, where the Nous sign-in plugin is
    not loaded (``plugins.disabled``)."""
    global _current
    if _swap is None:
        return None
    _current = (client_id, portal_url)
    try:
        restore = _swap()
    except BaseException:
        _current = None
        raise

    def undo() -> None:
        global _current
        _current = None
        restore()

    return undo
