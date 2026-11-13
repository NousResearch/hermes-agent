"""``manage_connections`` action ``display``: the desktop's browsable connector card.

The card is an operation with no targets, so only Continue, words typed in the composer, the deadline or /stop settle it. Connecting from the card runs the Connectors page's own account operation, never this one."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from tools.connectors.contract import SettleReason
from tools.connectors.operation import ConnectionOperation
from tools.connectors.run import Callback, Kind, open_and_drive

OFF_DESKTOP_NOTE = (
    "The connector card shows only in the Hermes desktop app. Tell the user to connect apps from the "
    "desktop app or the portal."
)


@dataclass
class DisplayOperation(ConnectionOperation):
    suggested: list[str] = field(default_factory=list)
    said: str | None = None

    def request_payload(self) -> dict[str, Any]:
        return {**super().request_payload(), "display": {"suggested": list(self.suggested)}}


def apply_answer(operation: DisplayOperation, raw: str) -> None:
    answer = json.loads(raw)
    said = str(answer.get("said") or "").strip()
    if said:
        operation.said = said
    if said or answer.get("settled_by") == SettleReason.continue_.value:
        operation.settle(SettleReason.continue_)


def connected_slugs(client: Any) -> set[str]:
    return {str(i.get("connector", "")).lower() for i in client.list_connectors()
            if isinstance(i, dict) and i.get("connected")}


def _result(client: Any, before: set[str], operation: DisplayOperation) -> dict[str, Any]:
    after = connected_slugs(client)
    result = {"connected": sorted(after), "newly_connected": sorted(after - before), "suggested": operation.suggested}
    if operation.said:
        result["said"] = operation.said
    # A user who connected an app and walked away still gets it reported.
    return {**result, "timed_out": True} if operation.settled_by == SettleReason.deadline else result


def _unchanged(_operation: Any) -> None:
    return None


def run_display(client: Any, suggested: list[str], *, session_key: str, tool_call_id: Optional[str],
                connection_callback: Callback, call: Optional[Any] = None) -> str:
    """Draw the card and hold the tool thread until it settles."""
    before = connected_slugs(client)
    operation = DisplayOperation([], session_key=session_key, tool_call_id=tool_call_id, call=call,
                                 suggested=suggested)
    kind = Kind(prepare=_unchanged, observe=_unchanged, note="",
                result=lambda _settled: _result(client, before, operation))
    return open_and_drive(operation, kind, connection_callback=connection_callback, with_urls_in_result=False)
