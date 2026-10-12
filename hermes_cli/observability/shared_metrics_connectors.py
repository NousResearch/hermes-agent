"""Connector work in the shared metrics: the search-to-execute chain and the batch and search rows.

Each entry of a connector ``tool_call`` batch dispatches with its own call id (``<id>#<position>``,
``tools/connectors/dispatch.py``), so entries count as their own calls; the wrapper's post under the
bare id is dropped once an entry ran. Connector slugs, tool names and query text never leave this
module: rows carry only buckets and closed enums.
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


def batch_entry_ran(identities: Iterable[tuple[str, str, str]], wrapper: tuple[str, str, str]) -> bool:
    """Whether an entry of the batch whose wrapper has identity ``wrapper`` was already counted."""
    prefix = f"{wrapper[2]}#"
    return bool(wrapper[2]) and any(
        identity[2].startswith(prefix) and identity[:2] == wrapper[:2] for identity in identities)


def _described_names(args: Any) -> list[str]:
    """``tool_describe`` names as its dispatcher reads them: a bare string is one name, items stripped."""
    names = args.get("names") if isinstance(args, dict) else None
    names = [names] if isinstance(names, str) else names if isinstance(names, list) else []
    return [str(name or "").strip() for name in names]


@dataclass
class ConnectorSessionState:
    """Per-conversation chain state; the caller holds the session lock. Compression hands it to the
    rotated-to segment so the chain spans the whole conversation."""

    searches_since_execute: int = 0
    described: set[str] = field(default_factory=set)

    def observe(self, event: dict[str, Any]) -> dict[str, str] | None:
        """Update the chain after a counted call; the ``hermes.connector_chain.count`` fields for an entry."""
        name = event.get("tool_name")
        if name == "tool_search":
            self.searches_since_execute += 1
            return None
        if name == "tool_describe":
            for described in _described_names(event.get("args")):
                if is_connector_name(described) and len(self.described) < _MAX_DESCRIBED:
                    self.described.add(described)
            return None
        if not is_connector_name(name):
            return None
        fields = {
            "described_first": "yes" if name in self.described else "no",
            "outcome": contract.tool_outcome(event),
            "searches_before_bucket": contract.count_bucket(self.searches_since_execute),
        }
        self.searches_since_execute = 0
        return fields


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
