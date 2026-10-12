"""Connector work in the shared metrics: per-entry tool counting, the search-to-execute chain, and
the batch and search rows.

A ``tool_call`` batch of connector entries reaches the hooks once per entry (same ``tool_call_id``)
and once more for the wrapper. Each entry is its own logical call, so entries get distinct identities
(``<id>#<n>`` in run order) and the wrapper is not counted once an entry was. Connector slugs, tool
names and query text never leave this module: rows carry only buckets and closed enums.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import shared_metrics_contract as contract
from .shared_metrics_contract import CONNECTOR_NAME_PREFIX

# Described names are kept per conversation to answer "was this tool described before it ran".
_MAX_DESCRIBED = 256


def is_connector_name(name: Any) -> bool:
    return isinstance(name, str) and name.startswith(CONNECTOR_NAME_PREFIX)


def _call_entries(args: Any) -> list[dict[str, Any]]:
    """The ``calls[]`` entries of a ``tool_call`` invocation; [] when the arguments do not parse."""
    if not isinstance(args, dict):
        return []
    from tools.tool_search_validation import normalize_tool_call_entries

    entries, err = normalize_tool_call_entries(args)
    return [] if err else entries


def is_connector_batch(event: dict[str, Any]) -> bool:
    """A ``tool_call`` wrapper whose entries are all connector calls (each entry is counted itself)."""
    if event.get("tool_name") != "tool_call":
        return False
    entries = _call_entries(event.get("args"))
    return bool(entries) and all(is_connector_name(entry.get("name")) for entry in entries)


@dataclass
class ConnectorSessionState:
    """Per-conversation connector state; the caller holds the session lock."""

    searches_since_execute: int = 0
    described: set[str] = field(default_factory=set)
    # (task_id, api_request_id, turn_id, tool_call_id) -> connector entries already counted.
    counted_entries: dict[tuple[str, str, str, str], int] = field(default_factory=dict)

    def identity(self, task_id: str, event: dict[str, Any], base: tuple[str, str, str]) -> tuple[str, str, str]:
        """``base`` for any call but a connector entry, which gets ``<id>#<n>`` (n = entries counted)."""
        if not is_connector_name(event.get("tool_name")) or not base[2]:
            return base
        return (base[0], base[1], f"{base[2]}#{self.counted_entries.get((task_id, *base), 0)}")

    def batch_already_counted(self, task_id: str, event: dict[str, Any], base: tuple[str, str, str]) -> bool:
        """True for a connector batch wrapper once one of its entries was counted."""
        return bool(self.counted_entries.get((task_id, *base))) and is_connector_batch(event)

    def observe(self, task_id: str, event: dict[str, Any], base: tuple[str, str, str]) -> dict[str, str] | None:
        """Update the chain after a counted call; the ``hermes.connector_chain.count`` fields for an entry."""
        name = event.get("tool_name")
        if name == "tool_search":
            self.searches_since_execute += 1
            return None
        if name == "tool_describe":
            names = (event.get("args") or {}).get("names") if isinstance(event.get("args"), dict) else None
            for described in names if isinstance(names, list) else ():
                if is_connector_name(described) and len(self.described) < _MAX_DESCRIBED:
                    self.described.add(described)
            return None
        if not is_connector_name(name):
            return None
        if base[2]:
            key = (task_id, *base)
            self.counted_entries[key] = self.counted_entries.get(key, 0) + 1
        fields = {
            "described_first": "yes" if name in self.described else "no",
            "outcome": contract.tool_outcome(event),
            "searches_before_bucket": contract.count_bucket(self.searches_since_execute),
        }
        self.searches_since_execute = 0
        return fields

    def forget_task(self, task_id: str) -> None:
        for key in [key for key in self.counted_entries if key[0] == task_id]:
            del self.counted_entries[key]


# ---- row builders ----------------------------------------------------------------------------

def tool_batch_fields(route: dict[str, str], tool_calls: Iterable[Any]) -> dict[str, str] | None:
    """One model reply's tool calls: how many logical calls (``tool_call`` entries counted one by
    one) and how many of them were connector calls. None for a reply with no calls."""
    total = connectors = 0
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None)
        names = [name]
        if name == "tool_call":
            entries = _call_entries(_parsed_arguments(getattr(fn, "arguments", None)))
            names = [entry.get("name") for entry in entries] or [name]
        total += len(names)
        connectors += sum(1 for entry_name in names if is_connector_name(entry_name))
    if not total:
        return None
    share = "none" if not connectors else "all" if connectors == total else "some"
    return {**route, "connector_share": share, "size_bucket": contract.count_bucket(total)}


def _parsed_arguments(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def tool_search_fields(*, scope: str, hits: int, outcome: str) -> dict[str, str] | None:
    if scope not in contract.TOOL_SEARCH_SCOPES or outcome not in contract.TOOL_SEARCH_OUTCOMES:
        return None
    return {"hits_bucket": contract.count_bucket(hits), "outcome": outcome, "scope": scope}
