"""In-memory reference implementation of ``ProfileIdentityRepository`` (test fixture).

Registered onto the assembled backend as the ``profile_identity`` attribute. The reference store has the sessions table (``profile_name``, ``session_key``, ``origin``), the heartbeat table, the routing index (routing scope -> session key -> a row with ``session_key`` and ``origin`` attributes) and, once ``RoutingRepository.prepare_topic_storage`` ran, the topic tables keyed by ``(profile, chat_id)`` and ``(profile, chat_id, thread_id)``. API binding receipts are the scope's ``api_bindings`` receipts and local creation receipts the authority state's ``binding:`` meta entries; both are rewritten in the authority transaction, as stored dicts. It has no delivery ledger: those facets count ``None``, as do the topic facets before the topic tables exist. Collisions are checked before the first write, so a refused rename leaves every table as it was.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Mapping, Optional

from persistence.housekeeping.profile_identity import ProfileIdentityCollision, ProfilePurgeCounts, ProfileRekeyCounts, profile_namespace

from .._authority import authority_state
from ..authority.bindings_lineage import RECEIPT_PREFIX, decode
from ..backend import InMemoryBackend, register_repository

__all__ = ["InMemoryProfileIdentityRepository"]


def _rekeyed(key: Any, old_ns: str, new_ns: str) -> Any:
    return new_ns + key[len(old_ns):] if isinstance(key, str) and key.startswith(old_ns) else key


def _rekeyed_receipt(raw: Any, old: str, new: str, old_ns: str, new_ns: str) -> Optional[dict[str, Any]]:
    """``raw`` with its ``route`` and ``entry['session_key']`` moved out of ``old_ns`` and ``entry['origin']['profile']`` renamed, or None when nothing changed or it does not decode as an object."""
    try:
        receipt = decode(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(receipt, dict):
        return None
    receipt, changed = copy.deepcopy(receipt), False
    if isinstance(receipt.get("route"), str) and receipt["route"].startswith(old_ns):
        receipt["route"], changed = _rekeyed(receipt["route"], old_ns, new_ns), True
    entry = receipt.get("entry")
    if isinstance(entry, dict):
        if isinstance(entry.get("session_key"), str) and entry["session_key"].startswith(old_ns):
            entry["session_key"], changed = _rekeyed(entry["session_key"], old_ns, new_ns), True
        origin = entry.get("origin")
        if isinstance(origin, dict) and origin.get("profile") == old:
            origin["profile"], changed = new, True
    return receipt if changed else None


def _rekey_stored(table: dict[str, Any], keys: list[str], *names: str) -> int:
    moved = 0
    for key in keys:
        if (receipt := _rekeyed_receipt(table[key], *names)) is not None:
            table[key] = receipt
            moved += 1
    return moved


class InMemoryProfileIdentityRepository:
    """See module docstring; semantics follow persistence/housekeeping/profile_identity.py."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    def _table(self, name: str) -> Any:
        return self._backend.scope_store(self._backend.scope).setdefault(name, {})

    def _topics(self) -> Optional[tuple[dict[tuple[str, str], Any], dict[tuple[str, str, str], Any]]]:
        """The topic-mode and topic-binding tables, or None before the topic storage exists."""
        if not self._backend.scope_store(self._backend.scope).get("topic_tables"):
            return None
        return self._table("topic_mode"), self._table("topic_bindings")

    def _check_collisions(self, old: str, new: str, old_ns: str, new_ns: str) -> None:
        for routing_scope, entries in self._table("routing").items():
            for key in entries:
                target = _rekeyed(key, old_ns, new_ns)
                if key.startswith(old_ns) and target in entries:
                    raise ProfileIdentityCollision(new, f"profile routing collision in scope {routing_scope!r}: {target!r}")
        if (topics := self._topics()) is None:
            return
        for table_name, table in zip(("telegram_dm_topic_mode", "telegram_dm_topic_bindings"), topics):
            if any(key[0] == old and (new, *key[1:]) in table for key in table):
                raise ProfileIdentityCollision(new, f"profile identity collision in {table_name}")

    def _rekey_sessions(self, old: str, new: str, old_ns: str, new_ns: str) -> tuple[int, int, int]:
        rows = self._table("sessions").values()
        owners = [row for row in rows if row.get("profile_name") == old]
        for row in owners:
            row["profile_name"] = new
        keys = [row for row in rows if isinstance(row.get("session_key"), str) and row["session_key"].startswith(old_ns)]
        for row in keys:
            row["session_key"] = _rekeyed(row["session_key"], old_ns, new_ns)
        origins = [row for row in rows if isinstance(row.get("origin"), Mapping) and row["origin"].get("profile") == old]
        for row in origins:
            row["origin"] = {**row["origin"], "profile": new}
        return len(owners), len(keys), len(origins)

    def _rekey_topics(self, old: str, new: str, old_ns: str, new_ns: str) -> dict[str, int]:
        if (topics := self._topics()) is None:
            return {}
        modes, bindings = topics
        moved_modes = [key for key in modes if key[0] == old]
        for key in moved_modes:
            modes[(new, *key[1:])] = modes.pop(key)
        rekeyed = [key for key, binding in bindings.items() if isinstance(binding.session_key, str) and binding.session_key.startswith(old_ns)]
        for key in rekeyed:
            bindings[key] = dataclasses.replace(bindings[key], session_key=_rekeyed(bindings[key].session_key, old_ns, new_ns))
        owned = [key for key in bindings if key[0] == old]
        for key in owned:
            bindings[(new, *key[1:])] = dataclasses.replace(bindings.pop(key), profile_name=new)
        return {
            "telegram_dm_topic_mode_profile_name": len(moved_modes),
            "telegram_dm_topic_bindings_profile_name": len(owned),
            "telegram_dm_topic_bindings_session_key": len(rekeyed),
        }

    def _rekey_routing(self, old: str, new: str, old_ns: str, new_ns: str) -> int:
        moved = 0
        for entries in self._table("routing").values():
            for key in [k for k in entries if k.startswith(old_ns)]:
                entry = entries.pop(key)
                origin: Mapping[str, Any] | None = entry.origin
                if isinstance(origin, Mapping) and origin.get("profile") == old:
                    origin = {**origin, "profile": new}
                entries[_rekeyed(key, old_ns, new_ns)] = dataclasses.replace(entry, session_key=_rekeyed(entry.session_key, old_ns, new_ns), origin=origin)
                moved += 1
        return moved

    def _rekey_receipts(self, *names: str) -> tuple[int, int]:
        """The API binding and local creation receipts moved, in one authority transaction."""
        with authority_state(self._backend, self._backend.scope).transaction() as staged:
            receipts = staged.scope.get("api_bindings", {}).get("receipts", {})
            meta = staged.authority["meta"]
            api = _rekey_stored(receipts, list(receipts), *names)
            local = _rekey_stored(meta, [key for key in meta if isinstance(key, str) and key.startswith(RECEIPT_PREFIX)], *names)
        return api, local

    def rekey_profile(self, old_name: str, new_name: str) -> ProfileRekeyCounts:
        old, new = (old_name or "").strip(), (new_name or "").strip()
        if not old or not new or old == new:
            return ProfileRekeyCounts()
        old_ns, new_ns = profile_namespace(old), profile_namespace(new)
        self._check_collisions(old, new, old_ns, new_ns)
        owners, keys, origins = self._rekey_sessions(old, new, old_ns, new_ns)
        heartbeats = [row for row in self._table("gateway_heartbeats").values() if row.get("profile") == old]
        for row in heartbeats:
            row["profile"] = new
        topics = self._rekey_topics(old, new, old_ns, new_ns)
        routing = self._rekey_routing(old, new, old_ns, new_ns)
        api, local = self._rekey_receipts(old, new, old_ns, new_ns)
        return ProfileRekeyCounts(
            sessions_profile_name=owners,
            gateway_heartbeats_profile=len(heartbeats),
            sessions_session_key=keys,
            sessions_origin_json=origins,
            gateway_routing=routing,
            api_binding_receipts=api,
            local_policy_receipts=local,
            **topics,
        )

    def _purge_topics(self, name: str, ns: str) -> dict[str, int]:
        if (topics := self._topics()) is None:
            return {}
        modes, bindings = topics
        doomed_modes = [key for key in modes if key[0] == name]
        doomed_bindings = [key for key, binding in bindings.items() if key[0] == name or (isinstance(binding.session_key, str) and binding.session_key.startswith(ns))]
        for key in doomed_modes:
            del modes[key]
        for key in doomed_bindings:
            del bindings[key]
        return {"telegram_dm_topic_mode": len(doomed_modes), "telegram_dm_topic_bindings": len(doomed_bindings)}

    def purge_profile(self, profile: str) -> ProfilePurgeCounts:
        name = (profile or "").strip()
        if not name:
            return ProfilePurgeCounts()
        ns = profile_namespace(name)
        routed = 0
        for entries in self._table("routing").values():
            for key in [k for k in entries if k.startswith(ns)]:
                del entries[key]
                routed += 1
        heartbeats = self._table("gateway_heartbeats")
        beats = [key for key, row in heartbeats.items() if row.get("profile") == name]
        for key in beats:
            del heartbeats[key]
        return ProfilePurgeCounts(gateway_routing=routed, gateway_heartbeats=len(beats), **self._purge_topics(name, ns))


register_repository("profile_identity", InMemoryProfileIdentityRepository)
