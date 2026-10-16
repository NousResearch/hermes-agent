"""``ctx.register_footer_item``: plugin items in the classic CLI live-work footer.

Bound into the ``PluginContext`` class body like ``plugins_content``, so it is an ordinary
``ctx.register_*`` method with the same abandoned-load guard. Rendering, polling and actions live in
``hermes_cli.cli_plugin_footer``; this module only validates and stores the registration.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from hermes_cli.plugins_ledger import PluginRegistration
from hermes_cli.plugins_loader import _serialized_replacement

__all__ = ["get_plugin_footer_items", "register_footer_item"]

_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")


@_serialized_replacement
def register_footer_item(
    self, name: str, *, snapshot: Callable[[dict], Optional[dict]],
    on_action: Optional[Callable[..., Any]] = None, order: int = 100,
) -> PluginRegistration:
    """Register an item in the classic CLI's live-work footer (display only: nothing reaches
    the conversation). ``snapshot(session)`` returns ``{"label", "tone", "rows", "actions"}``
    or ``None`` to hide; it is polled about once a second off the render path, in a worker
    thread with a short time budget. ``on_action(action_id, session, row_key)`` runs when a
    row's action key is pressed or the row is clicked. Lower ``order`` paints first."""
    clean = str(name or "").strip().lower()
    if not _NAME.fullmatch(clean):
        raise self._refuse(f"footer item {name!r} (name must match [a-z0-9][a-z0-9_.-]{{0,63}})")
    if not callable(snapshot):
        raise self._refuse(f"footer item {clean!r} with a non-callable snapshot")
    if on_action is not None and not callable(on_action):
        raise self._refuse(f"footer item {clean!r} with a non-callable on_action")
    if isinstance(order, bool) or not isinstance(order, int):
        raise self._refuse(f"footer item {clean!r} with a non-integer order")
    key = f"{self.plugin_id}:{clean}"
    entry = {"key": key, "name": clean, "plugin": self.manifest.name, "plugin_key": self.plugin_id,
             "snapshot": snapshot, "on_action": on_action, "order": order}
    return self._register_entry("footer_item", key, self._manager._footer_items, entry,
                                "Plugin %s registered footer item: %s", clean)


def get_plugin_footer_items() -> list[dict[str, Any]]:
    """Registered CLI footer items by ``(order, key)``. Never triggers discovery: it is read from
    the footer's one-second poll, and an undiscovered manager simply has none yet."""
    from hermes_cli.plugins import get_plugin_manager
    items = list(get_plugin_manager()._footer_items.values())
    return sorted(items, key=lambda item: (item["order"], item["key"]))
