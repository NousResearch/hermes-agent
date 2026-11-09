"""The reference backend for the extension registry conformance suite.

An in-memory registry that holds one tenant's whole registry in one store, exactly the shape a
hosted backend serves, so the suite can exercise the ownership levels the local backend cannot
express (a local install has one agent per file and one owner, the local person). It is
executable documentation of the port's rules — the pin, the administration right, the
provisioning actor that alone writes tenant-wide entries and
:func:`hermes_extension_registry_port.entry_visible_to` — and the substrate the deliberately
broken backends in ``tests/conformance/extensions/mutants.py`` are cut from. It is test
infrastructure: no deployment selects it, and it stores nothing when it dies.

One store per tenant; ``open`` returns a view pinned to one agent and its owner, whose every
operation re-checks the bound scope, as a backend behind row-level security would.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from hermes_extension_registry_port import (
    ExtensionRegistry, RegistryAdminRefused, RegistryPinRefused, owner_key,
    require_provisioning,
)
from hermes_scope import Group, Owner, Principal, current_scope


def _owner_from_key(key: Optional[tuple[str, str]]) -> Optional[Owner]:
    return None if key is None else (Principal(key[1]) if key[0] == "principal" else Group(key[1]))


class TenantStore:
    """One tenant's rows, guarded by one lock (the registry is a read-modify-write)."""

    def __init__(self, tenant: str) -> None:
        self.tenant = tenant
        self.lock = threading.Lock()
        #: ``(owner_key, name)`` → the entry dict, whose ``owner`` is the owner object.
        self.entries: dict[tuple[Optional[tuple[str, str]], str], dict[str, Any]] = {}
        #: the per-agent switch rows: ``(agent, name)`` → on/off.
        self.switches: dict[tuple[str, str], bool] = {}


class ReferenceExtensionRegistry(ExtensionRegistry):
    """One agent's view of its tenant's registry. Every operation re-checks the bound scope is
    this view's ``(tenant, agent, owner)`` and refuses any other pin."""

    def __init__(self, store: TenantStore, *, agent: str, owner: Owner) -> None:
        self.store = store
        self.agent = agent
        self.owner = owner

    # --- the pin ---------------------------------------------------------------------------

    def _pin(self):
        """The bound scope, after checking it is this view's own ``(tenant, agent, owner)``."""
        scope = current_scope()
        if scope is None:
            raise RegistryPinRefused("an extension registry operation needs a scope bound")
        if scope.tenant != self.store.tenant:
            raise RegistryPinRefused(
                f"this extension registry serves tenant {self.store.tenant!r} only")
        if scope.agent != self.agent:
            raise RegistryPinRefused("the bound agent is not this extension registry's agent")
        if scope.owner != self.owner:
            raise RegistryPinRefused("the bound scope's owner is not this agent's owner")
        return scope

    def _administer(self, scope) -> None:
        """Refuse a scope that may not administer this agent, before any target is looked up."""
        if not scope.can_administer:
            raise RegistryAdminRefused(
                "installing, removing or switching a user extension needs the agent's administration right")

    # --- the keys the broken backends in the mutant suite mis-handle -----------------------

    def _visible(self, scope, entry_owner_key: Optional[tuple[str, str]]) -> bool:
        """Whether an entry at *entry_owner_key* resolves for the bound scope's agent owner."""
        return entry_owner_key is None or entry_owner_key == owner_key(scope.owner)

    def _switch_key(self, agent: str, name: str) -> tuple[str, str]:
        """The per-agent switch row's key: one row per ``(agent, entry)``."""
        return (agent, name)

    # --- resolution ------------------------------------------------------------------------

    def _resolve(self, scope, name: str) -> Optional[dict[str, Any]]:
        """The entry *name* resolves to for the bound owner: the owner-wide one when visible,
        else the tenant-wide one. Foreign and missing both come back ``None``."""
        candidates = [entry for key, entry in self.store.entries.items()
                      if key[1] == name and self._visible(scope, key[0])]
        for entry in candidates:  # the owner-wide level wins for its owner
            if entry["level"] == "owner":
                return entry
        return candidates[0] if candidates else None

    def _visible_rows(self, scope, *, kind: Optional[str]) -> list[dict[str, Any]]:
        rows = [dict(entry) for key, entry in self.store.entries.items()
                if self._visible(scope, key[0]) and (kind is None or entry["kind"] == kind)]
        return sorted(rows, key=lambda r: r["name"])

    def _row(self, entry: dict[str, Any]) -> dict[str, Any]:
        row = dict(entry)
        row["owner"] = _owner_from_key(owner_key(entry["owner"]))
        return row

    # --- the port --------------------------------------------------------------------------

    def install(self, *, name, kind, version, content_hash, manifest, scan_result=None, level="owner"):
        scope = self._pin()
        if level == "tenant":
            # The v1 rule: a tenant-wide entry is written by the tenant's provisioning actor
            # alone, before anything is read — an agent's administration right never reaches
            # past its owner's entries.
            require_provisioning(scope)
        else:
            self._administer(scope)
        self._check_install_args(name=name, kind=kind, version=version, content_hash=content_hash,
                                 manifest=manifest, scan_result=scan_result, level=level)
        owner = self.owner if level == "owner" else None
        entry = {"name": name, "kind": kind, "version": version, "content_hash": content_hash,
                 "manifest": manifest, "scan_result": scan_result, "enabled": True,
                 "level": level, "owner": owner}
        with self.store.lock:
            self.store.entries[(owner_key(owner), name)] = entry
        return self._row(entry)

    def get(self, name):
        scope = self._pin()
        with self.store.lock:
            row = self._resolve(scope, name)
            return self._row(row) if row is not None else None

    def list(self, *, kind=None):
        scope = self._pin()
        with self.store.lock:
            return [self._row(r) for r in self._visible_rows(scope, kind=kind)]

    def set_enabled(self, name, enabled):
        scope = self._pin()
        self._administer(scope)
        with self.store.lock:
            row = self._resolve(scope, name)
            if row is None:
                return False
            if row["level"] == "tenant":
                # The agent's administration right reaches its owner's entries, never the
                # tenant's (the v1 rule).
                require_provisioning(scope)
            row["enabled"] = bool(enabled)
            return True

    def remove(self, name):
        scope = self._pin()
        self._administer(scope)
        with self.store.lock:
            row = self._resolve(scope, name)
            if row is None:
                return False
            if row["level"] == "tenant":
                require_provisioning(scope)
            del self.store.entries[(owner_key(row["owner"]), name)]
            for agent in {key[0] for key in self.store.switches}:
                self.store.switches.pop(self._switch_key(agent, name), None)
            return True

    def set_agent_enabled(self, name, enabled):
        scope = self._pin()
        self._administer(scope)
        with self.store.lock:
            if self._resolve(scope, name) is None:
                return False
            self.store.switches[self._switch_key(self.agent, name)] = bool(enabled)
            return True

    def agent_enabled(self, name):
        scope = self._pin()
        with self.store.lock:
            if self._resolve(scope, name) is None:
                return False
            return bool(self.store.switches.get(self._switch_key(self.agent, name)))

    def enabled_extensions(self, *, kind=None):
        scope = self._pin()
        with self.store.lock:
            return [self._row(r) for r in self._visible_rows(scope, kind=kind)
                    if r["enabled"] and self.store.switches.get(self._switch_key(self.agent, r["name"]))]
