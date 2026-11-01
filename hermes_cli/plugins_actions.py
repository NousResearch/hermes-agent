"""Interactive-action registrars on ``PluginContext``: Slack Block Kit action handlers and
platform-agnostic card action handlers (contract: ``gateway/plugin_card_actions.py``).

Bound into the class body (``from hermes_cli.plugins_actions import ...``), so they are ordinary
``ctx.register_*`` methods and get the same abandoned-load guard as every other registrar.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from hermes_cli.plugins_ledger import PluginRegistration
from hermes_cli.plugins_loader import _serialized_replacement

# Same logger as the rest of the ctx surface, so plugin-load diagnostics stay in one place.
logger = logging.getLogger("hermes_cli.plugins")

__all__ = ["register_card_action_handler", "register_slack_action_handler"]

_CARD_NAMESPACE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")


def register_slack_action_handler(
    self, action_id: Any, callback: Callable,
) -> PluginRegistration:
    """Register a Slack Block Kit action handler, wired into ``slack_bolt.AsyncApp`` at connect.
    ``action_id`` is anything ``slack_bolt.App.action()`` accepts; ``callback`` is
    ``async def handler(ack, body, action)`` (``await ack()`` within 3s). Raises ``ValueError`` for
    a non-callable callback or empty ``action_id``."""
    if not callable(callback):
        raise self._refuse("a Slack action handler with a non-callable callback")
    if action_id is None or (isinstance(action_id, str) and not action_id.strip()):
        raise self._refuse("a Slack action handler with an empty action_id")
    entry = (action_id, callback, self.manifest.name)
    handlers = self._manager._slack_action_handlers
    handlers.append(entry)
    handle = self._track("slack_action_handler", repr(action_id),
                         lambda: self._manager._remove_identity(handlers, entry))
    logger.debug("Plugin %s registered Slack action handler: %s", self.manifest.name, action_id)
    return handle


@_serialized_replacement
def register_card_action_handler(self, namespace: str, callback: Callable) -> Optional[PluginRegistration]:
    """Answer clicks on Adaptive Card ``Action.Execute`` buttons whose ``data`` carries
    ``{"hermes_plugin_action": namespace}`` (Teams). ``callback(event)`` (sync or async) receives a
    ``gateway.plugin_card_actions.CardActionEvent`` and returns a ``CardActionResult``, a card ``dict``
    (replaces the clicked card) or a ``str`` (shown briefly). ``namespace`` is 1-64 characters of
    ``[a-z0-9_.-]`` starting with a letter or digit, owned by one plugin: another plugin's registration
    is refused with a warning (``None``), the owner registering again replaces its handler. Raises
    ``ValueError`` for an invalid namespace or a non-callable callback."""
    if not callable(callback):
        raise self._refuse("a card action handler with a non-callable callback")
    if not isinstance(namespace, str) or not _CARD_NAMESPACE_RE.fullmatch(namespace):
        raise self._refuse(f"a card action handler with an invalid namespace {namespace!r} "
                           "(1-64 of [a-z0-9_.-], starting with a letter or digit)")
    handlers = self._manager._card_action_handlers
    owner = handlers.get(namespace)
    if owner is not None and owner["plugin_key"] != self.plugin_id:
        logger.warning("Plugin '%s' tried to register card action namespace %r, which plugin '%s' owns; "
                       "ignoring it.", self.manifest.name, namespace, owner["plugin"])
        return None
    entry = {"callback": callback, "plugin": self.manifest.name, "plugin_key": self.plugin_id}
    return self._register_entry("card_action_handler", namespace, handlers, entry,
                                "Plugin %s registered card action handler: %s", namespace)


def card_action_handler(namespace: str) -> Optional[dict]:
    """``{callback, plugin, plugin_key}`` registered for *namespace* by the active profile's plugins."""
    from hermes_cli.plugins import _ensure_plugins_discovered
    return _ensure_plugins_discovered()._card_action_handlers.get(namespace)
