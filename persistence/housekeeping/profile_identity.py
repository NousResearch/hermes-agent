"""Profile identity lifecycle contract.

A profile is an identity inside a store. The store carries that identity in two shapes: the ``agent:<profile>:`` KEY NAMESPACE on session keys (sessions, gateway routing entries, delivery obligations, topic bindings) and OWNER TAGS on rows (``sessions.profile_name``, ``gateway_heartbeats.profile``, ``delivery_obligations.adapter_profile``, the telegram topic tables' ``profile_name``), plus the ``profile`` field of stored origin payloads. Renaming a profile rewrites every shape at once; deleting one purges its bookkeeping. A caller working on several stores (the routing home and the profile home, hermes_cli/profile_identity.py:128, :228; gateway/run_profile_reconcile.py:675-720) opens one handle per store and calls the verb on each.

Both verbs are one write transaction:

- Names are stripped. An empty name, or a rekey whose stripped names are equal, returns empty counts without touching the store. The built-in ``default`` profile is not refused here; the callers refuse it first.
- ``rekey_profile`` checks collisions BEFORE any write and fails the whole call with :class:`ProfileIdentityCollision` (a routing entry already stored under the target key in the same routing scope; a topic-mode or topic-binding row already owned by the target name for the same chat or thread). Matching is exact: ``agent:<old>:`` as a literal prefix, owner tags by equality, no case folding.
- ``purge_profile`` hard-deletes routing entries under the namespace, heartbeat rows and topic-mode rows by owner tag, and topic bindings by owner tag OR key namespace; it TERMINALIZES pending delivery obligations (``abandoned``, delivered history untouched) and never touches session rows.
- A facet whose table or column the store lacks (the lazily created telegram topic tables, the delivery ledger and its ``adapter_profile`` column) is skipped: its count is ``None`` and ``as_dict()`` leaves its key out.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Protocol, runtime_checkable

from ..core.errors import Conflict

__all__ = [
    "ProfileIdentityCollision",
    "ProfileIdentityRepository",
    "ProfilePurgeCounts",
    "ProfileRekeyCounts",
    "profile_namespace",
]


def profile_namespace(profile: str) -> str:
    """The key namespace a profile owns: ``agent:<profile>:`` over the stripped name."""
    return f"agent:{(profile or '').strip()}:"


class ProfileIdentityCollision(Conflict, ValueError):
    """A rekey target already holds the identity being moved; ``str`` is the collision text (``profile routing collision in scope ...`` or ``profile identity collision in <table>``)."""

    def __init__(self, new_name: str, message: str) -> None:
        super().__init__("profile", new_name)
        self.args = (message,)


def _present(counts: object) -> dict[str, int]:
    return {f.name: value for f in fields(counts) if (value := getattr(counts, f.name)) is not None}


@dataclass(frozen=True, slots=True)
class ProfileRekeyCounts:
    """Rows rewritten by one ``rekey_profile`` call; field names are the ``as_dict()`` keys, in the same order, ``None`` for a facet the store lacks."""

    sessions_profile_name: Optional[int] = None
    gateway_heartbeats_profile: Optional[int] = None
    sessions_session_key: Optional[int] = None
    sessions_origin_json: Optional[int] = None
    delivery_obligations_adapter_profile: Optional[int] = None
    delivery_obligations_session_key: Optional[int] = None
    telegram_dm_topic_mode_profile_name: Optional[int] = None
    telegram_dm_topic_bindings_profile_name: Optional[int] = None
    telegram_dm_topic_bindings_session_key: Optional[int] = None
    gateway_routing: Optional[int] = None

    def as_dict(self) -> dict[str, int]:
        """The count dict: one key per facet the store has."""
        return _present(self)


@dataclass(frozen=True, slots=True)
class ProfilePurgeCounts:
    """Rows settled by one ``purge_profile`` call; field names are the ``as_dict()`` keys, in the same order, ``None`` for a facet the store lacks."""

    gateway_routing: Optional[int] = None
    gateway_heartbeats: Optional[int] = None
    delivery_obligations: Optional[int] = None
    telegram_dm_topic_mode: Optional[int] = None
    telegram_dm_topic_bindings: Optional[int] = None

    def as_dict(self) -> dict[str, int]:
        """The count dict: one key per facet the store has."""
        return _present(self)


@runtime_checkable
class ProfileIdentityRepository(Protocol):
    """Profile identity rewrite and purge inside the handle's store."""

    def rekey_profile(self, old_name: str, new_name: str) -> ProfileRekeyCounts:
        """Rewrite ``old_name``'s identity to ``new_name`` in one write: owner tags (sessions, heartbeats, delivery obligations, topic tables), the ``agent:<old>:`` namespace on session keys (sessions, delivery obligations, topic bindings, routing entries with the payload's ``session_key``), and the ``profile`` field of session origins and routing payload origins. Empty or equal stripped names return empty counts with no write. A collision raises ``ProfileIdentityCollision`` before anything is written. Idempotent: a repeated rename counts zero."""
        ...

    def purge_profile(self, profile: str) -> ProfilePurgeCounts:
        """Settle ``profile``'s identity in one write: delete its routing entries (every routing scope), heartbeat rows, topic-mode rows and topic bindings (owner tag OR key namespace); mark its non-terminal delivery obligations ``abandoned``; never touch session rows. An empty stripped name returns empty counts with no write. Idempotent: a second purge counts zero."""
        ...
