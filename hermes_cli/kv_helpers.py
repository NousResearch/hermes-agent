"""Structured key naming helpers for the Kanban swarm blackboard.

These helpers enforce a namespaced key convention on top of the existing
`post_blackboard_update` / `latest_blackboard` surface.  No new storage
or schema is introduced — the blackboard remains a flat string-keyed
dictionary backed by Kanban task comments.  The convention is:

    worker:{index}:{field}
    team:{field}
    coordinator:{field}

Adopting structured keys gives the blackboard **machine-parseable structure**
without any code changes to the core board.  It is a naming convention,
not a new persistence layer.

Usage::

    from hermes_cli.kv_helpers import worker_key, parse_worker_key

    post_blackboard_update(conn, root, author="worker-3",
                           key=worker_key(3, "status"), value="running")

    board = latest_blackboard(conn, root)
    for key in board:
        if key.startswith("worker:"):
            parsed = parse_worker_key(key)
            # → (3, "status")
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Key namespace prefixes
# ---------------------------------------------------------------------------

WORKER_NS = "worker"
TEAM_NS = "team"
COORDINATOR_NS = "coordinator"

# ---------------------------------------------------------------------------
# Reserved top-level blackboard keys (non-namespaced)
# ---------------------------------------------------------------------------

TOPOLOGY_KEY = "topology"
GOAL_KEY = "goal"
AUTHORS_KEY = "_authors"

# ---------------------------------------------------------------------------
# Well-known per-worker field names
# ---------------------------------------------------------------------------

FIELD_STATUS = "status"
FIELD_RESULT = "result"
FIELD_HANDOFF = "handoff"
FIELD_HEARTBEAT = "heartbeat"
FIELD_AGENT_TYPE = "agent_type"
FIELD_DIRECTIVE = "directive"
FIELD_TASK = "task"
FIELD_UPDATED_AT = "updated_at"


# ===========================================================================
# Builders
# ===========================================================================


def worker_key(worker_index: int, field: str) -> str:
    """Build a namespaced key for a worker's blackboard entry.

    >>> worker_key(3, "status")
    'worker:3:status'
    """
    return f"{WORKER_NS}:{worker_index}:{field}"


def worker_result_key(worker_index: int) -> str:
    """Shortcut for the worker's result field."""
    return worker_key(worker_index, FIELD_RESULT)


def worker_status_key(worker_index: int) -> str:
    """Shortcut for the worker's status field."""
    return worker_key(worker_index, FIELD_STATUS)


def worker_handoff_key(worker_index: int) -> str:
    """Shortcut for the worker's handoff (cross-worker note) field."""
    return worker_key(worker_index, FIELD_HANDOFF)


def team_key(field: str) -> str:
    """Build a team-level blackboard key.

    >>> team_key("sources")
    'team:sources'
    """
    return f"{TEAM_NS}:{field}"


def coordinator_key(field: str) -> str:
    """Build a coordinator-level blackboard key.

    >>> coordinator_key("decision")
    'coordinator:decision'
    """
    return f"{COORDINATOR_NS}:{field}"


# ===========================================================================
# Parsers
# ===========================================================================


def parse_worker_key(key: str) -> Optional[tuple[int, str]]:
    """Parse a ``worker:N:field`` key into its components.

    Returns ``(worker_index, field)`` if the key matches the worker namespace,
    otherwise ``None``.

    >>> parse_worker_key("worker:3:status")
    (3, 'status')
    >>> parse_worker_key("worker:0:result")
    (0, 'result')
    >>> parse_worker_key("topology") is None
    True
    >>> parse_worker_key("worker:abc:status") is None
    True
    """
    if not key.startswith(f"{WORKER_NS}:"):
        return None

    rest = key[len(WORKER_NS) + 1:]
    separator = rest.rfind(":")
    if separator <= 0 or separator == len(rest) - 1:
        return None

    index_str = rest[:separator]
    field = rest[separator + 1:]
    if not index_str or not field:
        return None

    try:
        worker_index = int(index_str)
    except ValueError:
        return None

    return (worker_index, field)


def parse_namespaced_key(key: str) -> Optional[tuple[str, str]]:
    """Parse any namespaced key into ``(namespace, field``).

    Returns ``None`` for non-namespaced keys (e.g. ``"topology"``).

    >>> parse_namespaced_key("worker:3:status")
    ('worker', 'status')
    >>> parse_namespaced_key("team:sources")
    ('team', 'sources')
    >>> parse_namespaced_key("topology") is None
    True
    """
    if ":" not in key:
        return None
    # For worker keys the namespace is the first segment before the first ":"
    ns_end = key.index(":")
    ns = key[:ns_end]
    # The "field" is everything after the last ":"
    field_start = key.rfind(":") + 1
    field = key[field_start:]
    if not field:
        return None
    return (ns, field)


# ===========================================================================
# Utilities
# ===========================================================================


def worker_prefix(worker_index: int) -> str:
    """Return the key prefix for all entries belonging to a worker.

    >>> worker_prefix(3)
    'worker:3:'
    """
    return f"{WORKER_NS}:{worker_index}:"


def worker_fields(
    blackboard: dict[str, Any],
    worker_index: int,
) -> dict[str, Any]:
    """Extract all blackboard fields for a given worker.

    Returns a dict with the per-worker field names (without the
    ``worker:N:`` prefix) as keys.

    >>> board = {"worker:3:status": "running", "topology": {...}}
    >>> worker_fields(board, 3)
    {'status': 'running'}
    """
    prefix = worker_prefix(worker_index)
    return {
        key.removeprefix(prefix): value
        for key, value in blackboard.items()
        if key.startswith(prefix)
    }


def get_worker_field(
    blackboard: dict[str, Any],
    worker_index: int,
    field: str,
) -> Any:
    """Read a single field from a worker's blackboard slice.

    Returns ``None`` if the key has never been written.
    """
    return blackboard.get(worker_key(worker_index, field))
