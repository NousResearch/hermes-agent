"""Public event bridge for plugin backends (``plugin_api.py``, slash commands).

A plugin's backend runs inside the gateway process. To push an update to its
OWN desktop half it emits on the app's global event stream — the same stream
``host.onEvent`` subscribes to in the renderer::

    from hermes_cli.plugin_events import broadcast_plugin_event

    broadcast_plugin_event("rss-reader", "feed.updated", {"count": 3})
    # event "plugin.rss-reader.feed.updated" reaches every connected desktop client

The desktop half filters for its own name::

    host.onEvent('plugin.rss-reader.feed.updated', ({ payload }) => …)

This module is the sanctioned door for that: plugin backends must never import
``tui_gateway.server`` privates (``_broadcast_global_event``), whose signature
is core-internal. The ``plugin.`` prefix keeps plugin traffic out of core's own
event names (``skin.changed``, ``session.reclaimed``, …).

Delivery is per process: the frame goes to the clients of the gateway the
caller runs in. Under ``hermes serve`` (the Desktop backend, where plugin
routers, slash commands and the agent turn's tools/hooks run) that is every
connected window; a call from the ``dashboard.turn_isolation`` compute-host
child rides the host pipe to ``hermes serve`` and fans out there. A process
with no Desktop client at all (``hermes gateway run``, ``hermes chat``, cron)
has nobody to deliver to: the call is a logged no-op.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Every plugin event name starts with this — core owns all other names.
PLUGIN_EVENT_PREFIX = "plugin."

# Plugin ids are the catalog names (``plugin_catalog._NAME_RE``): lowercase, digits, ``_``/``-``.
# No dot: ``plugin.<id>.`` must be an unambiguous prefix, so an id can never spell a neighbour's
# namespace (``other.x`` would read as plugin ``other``).
_PLUGIN_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
# Dotted hierarchy like core's own names (``display.install.done``): non-empty segments of
# [A-Za-z0-9_-], so ``../x``, ``a..b``, a leading/trailing dot, slashes and whitespace are refused.
_EVENT_RE = re.compile(r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")


def plugin_event_name(plugin_id: str, event: str) -> str:
    """The wire name for one plugin event: ``plugin.<plugin_id>.<event>``.

    Raises ``ValueError`` for an id or event that cannot form one — a silently
    mangled name would strand the desktop half waiting on a name nobody emits.
    """
    if not isinstance(plugin_id, str) or not _PLUGIN_ID_RE.match(plugin_id):
        raise ValueError(f"invalid plugin id {plugin_id!r}: expected [a-z0-9_-]{{1,64}}")
    if not isinstance(event, str) or len(event) > 128 or not _EVENT_RE.match(event):
        raise ValueError(
            f"invalid plugin event {event!r}: expected dot-separated segments of [A-Za-z0-9_-] "
            "(the plugin id already namespaces the name)"
        )
    return f"{PLUGIN_EVENT_PREFIX}{plugin_id}.{event}"


def broadcast_plugin_event(plugin_id: str, event: str, payload: Optional[dict[str, Any]] = None) -> None:
    """Emit ``plugin.<plugin_id>.<event>`` to every connected client.

    Fire-and-forget and safe to call from any request handler: delivery fans out
    over the gateway's live transports, and a wedged peer is skipped rather than
    stalling the caller. ``payload`` must be a dict (or ``None`` for ``{}``).
    """
    if payload is not None and not isinstance(payload, dict):
        raise TypeError(f"plugin event payload must be a dict or None, got {type(payload).__name__}")
    name = plugin_event_name(plugin_id, event)

    # Late import: plugin backends load before the gateway server is up in some
    # hosts (CLI tooling imports plugin_api modules for route inspection), and
    # tui_gateway.server pulls in the transport stack.
    try:
        from tui_gateway.server import _broadcast_global_event
    except ImportError:
        # A plugin-only process (``hermes plugins validate`` importing the backend, a trimmed
        # install) has no gateway at all: nobody to deliver to, and the handler must not die for it.
        logger.warning("plugin event %s dropped: no tui_gateway in this process", name, exc_info=True)
        return

    _broadcast_global_event(name, dict(payload or {}))
