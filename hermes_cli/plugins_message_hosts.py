"""Process-wide published message hosts (messaging gateway and Ink TUI/desktop).

Split out of ``plugins.py``; the names are re-imported there so
``hermes_cli.plugins.<name>`` keeps resolving for its importers and patch seams.

A multiplexed gateway or desktop process owns one message host while plugins are isolated in
per-profile managers, so every manager in the process must see the same live host. Publishing is
owner-scoped: a clear by one owner never clobbers a newer owner's slot.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from hermes_cli.plugins import PluginManager

# Process-wide messaging-gateway host.
_published_gateway_message_injector: tuple[object, Callable] | None = None
_published_gateway_host_lock = threading.Lock()

# Process-wide Ink TUI / desktop host. Not the messaging-gateway slot. Stamped onto
# each profile's manager so a multiplexed desktop process does not drop injects
# aimed at a non-launch profile. ``None`` until the TUI/desktop process installs it.
_published_tui_message_injector: tuple[object, Callable] | None = None
_published_tui_host_lock = threading.Lock()


def publish_gateway_message_host(owner: object, injector: Callable[..., bool]) -> None:
    """Remember the process gateway host and stamp managers that already exist."""
    global _published_gateway_message_injector
    from hermes_cli.plugins import _known_plugin_managers

    # Keep publication + stamping atomic with owner-safe clear. Managers created concurrently are
    # registered before _attach_published_gateway_host(), which takes this same lock.
    with _published_gateway_host_lock:
        _published_gateway_message_injector = (owner, injector)
        for manager in _known_plugin_managers():
            manager.set_gateway_message_injector(owner, injector)


def clear_published_gateway_message_host(owner: object) -> None:
    """Forget this owner's process gateway host without clobbering a newer runner."""
    global _published_gateway_message_injector
    from hermes_cli.plugins import _known_plugin_managers

    with _published_gateway_host_lock:
        if (_published_gateway_message_injector is not None
                and _published_gateway_message_injector[0] is owner):
            _published_gateway_message_injector = None
        for manager in _known_plugin_managers():
            manager.clear_gateway_message_injector(owner)


def _attach_published_gateway_host(manager: PluginManager) -> None:
    """Give a newly resolved profile manager the live process gateway host, if any."""
    with _published_gateway_host_lock:
        host = _published_gateway_message_injector
        if host is not None and manager._gateway_message_injector is None:
            manager.set_gateway_message_injector(*host)


def publish_tui_message_host(owner: object, injector: Callable[..., bool]) -> None:
    """Remember the process TUI/desktop host and stamp managers that already exist.

    Does not call ``set_gateway_message_injector``.
    """
    global _published_tui_message_injector
    from hermes_cli.plugins import _known_plugin_managers

    with _published_tui_host_lock:
        _published_tui_message_injector = (owner, injector)
    for manager in _known_plugin_managers():
        manager.set_tui_message_injector(owner, injector)


def clear_published_tui_message_host(owner: object) -> None:
    """Forget the process TUI host and clear it where this owner still holds the slot."""
    global _published_tui_message_injector
    from hermes_cli.plugins import _known_plugin_managers

    with _published_tui_host_lock:
        if (_published_tui_message_injector is not None
                and _published_tui_message_injector[0] is owner):
            _published_tui_message_injector = None
    for manager in _known_plugin_managers():
        manager.clear_tui_message_injector(owner)


def _attach_published_tui_host(manager: PluginManager) -> None:
    """Give a newly resolved manager the process TUI host, if one is installed and the slot is empty."""
    with _published_tui_host_lock:
        host = _published_tui_message_injector
    if host is not None and manager._tui_message_injector is None:
        manager._tui_message_injector = host
