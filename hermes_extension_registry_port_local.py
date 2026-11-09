"""The local extension registry backend: a small file under the profile home behind the port.

The rows live in ``<profile home>/extension_registry.json`` — one small file per profile, read
and written atomically (``utils.atomic_json_write``). It is **not wired into local plugin
loading**: local installs keep today's in-process plugin loading (``hermes_cli.plugins``,
``$HERMES_HOME/plugins/``), which never reads this file, and nothing in config.yaml selects this
backend yet. Until a caller migrates, these rows are Hermes's own record only.

A local install is one tenant (``local``) whose one person owns every agent, and the file a
profile holds is that profile's registry: today profiles are islands, and a plugin installed
for one profile is not offered to another. This is the registry's local exception: an entry
never sits at the agent level by design (only the per-agent switch does), and the local backend
keeps entries owner-wide — the local owner's, which is every agent this file serves — and
refuses the tenant level outright, because one owner has nothing to share tenant-wide. Every
operation is pinned after the same fashion as the local session store
(``hermes_state_port_local``): the bound scope must be a local one naming this registry's own
profile home, and its owner must be the local person — any other tenant, agent or owner is
refused with :class:`~hermes_extension_registry_port.RegistryPinRefused` before anything is
read or written. With no scope bound, the pin is the ambient profile
(``get_hermes_home()``), as every local caller has it today.

The registry's home is frozen when it opens, from the absolute path it was given; a relative
one is refused, because which file it named could not be established later. A profile's
registry is served by one handle at a time (the file is unwired today); concurrent writers would
race a read-modify-write, and the last atomic replace wins.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from hermes_extension_registry_port import (
    ExtensionRegistry, ExtensionRegistryProvider, RegistryAdminRefused, RegistryPinRefused,
    entry_visible_to,
)
from hermes_scope import LOCAL_OWNER, LOCAL_TENANT, Group, Owner, Principal, current_scope

#: The registry file, under the profile home.
REGISTRY_FILE = "extension_registry.json"


def _owner_to_json(owner: Optional[Owner]) -> Optional[dict[str, str]]:
    return None if owner is None else {"kind": "group" if isinstance(owner, Group) else "principal", "id": owner.id}


def _owner_from_json(owner: Optional[dict[str, Any]]) -> Optional[Owner]:
    if owner is None:
        return None
    return Group(owner["id"]) if owner.get("kind") == "group" else Principal(owner["id"])


class LocalExtensionRegistry(ExtensionRegistry):
    """The registry rows of one profile home, in ``extension_registry.json``."""

    def __init__(self, home: Path) -> None:
        from hermes_constants import hermes_home_key
        if not os.path.isabs(os.fspath(home)):
            raise RegistryPinRefused(f"an extension registry needs an absolute profile home, not {home!r}")
        self.home = Path(os.fspath(home))
        self.path = self.home / REGISTRY_FILE
        self._home_key = hermes_home_key(self.home)

    # --- the pin ---------------------------------------------------------------------------

    def _pin(self):
        """The bound scope, after checking it is this profile's own local one."""
        from hermes_constants import get_hermes_home, hermes_home_key
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise RegistryPinRefused(f"a local extension registry serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        # Both sides canonical and absolute: a relative agent is the directory it names now.
        if hermes_home_key(agent) != self._home_key:
            raise RegistryPinRefused("the bound agent is not this extension registry's profile")
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        if owner != LOCAL_OWNER:
            raise RegistryPinRefused("a local extension registry serves the local owner only")
        return scope

    def _administer(self, scope) -> None:
        """Refuse a scope that may not administer this agent, before any target is looked up."""
        if scope is not None and scope.can_administer is False:
            raise RegistryAdminRefused(
                "installing, removing or switching a user extension needs the agent's administration right")

    def _check_levels(self, level: str) -> None:
        """The registry's local exception: one tenant (``local``) whose one person owns every
        agent, so an entry has nothing to be tenant-wide over — the local backend keeps
        owner-wide entries as the local owner's and refuses the tenant level outright. An entry
        never sits at the agent level by design; the per-agent switch alone is the agent's row
        (one file per profile, so the switch is stored at the agent level)."""
        if level == "tenant":
            raise ValueError(
                "the local extension registry stores owner-wide entries only: a local install "
                "is one owner, with nothing to share tenant-wide")

    # --- the file --------------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        from utils import read_json_or_empty
        data = read_json_or_empty(self.path)
        if not isinstance(data.get("entries"), list):
            data["entries"] = []
        if not isinstance(data.get("switches"), dict):
            data["switches"] = {}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        from utils import atomic_json_write
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(self.path, data)

    # --- resolution ------------------------------------------------------------------------

    def _row(self, entry: dict[str, Any]) -> dict[str, Any]:
        row = dict(entry)
        row["owner"] = _owner_from_json(entry.get("owner"))
        return row

    def _resolve(self, data: dict[str, Any], name: str, owner: Owner) -> Optional[dict[str, Any]]:
        """The entry *name* resolves to for *owner*: the owner-wide one when visible, else the
        tenant-wide one. Foreign and missing both come back ``None``."""
        candidates = [e for e in data["entries"] if e.get("name") == name
                      and entry_visible_to(owner, _owner_from_json(e.get("owner")))]
        # The owner-wide level wins for its owner; everyone else resolves the tenant-wide one.
        for row in candidates:
            if row.get("level") == "owner" and row.get("owner") is not None:
                return row
        return candidates[0] if candidates else None

    def _switch_key(self, entry: dict[str, Any]) -> str:
        """The per-agent switch row's key: the entry's identity — its level and name (the local
        owner is the only owner) — so a switch belongs to one entry alone: removing an entry
        drops its own switch, and another entry the same name resolves to never inherits it."""
        return f"{entry.get('level')}:{entry['name']}"

    def _visible(self, data: dict[str, Any], owner: Owner, *, kind: Optional[str]) -> list[dict[str, Any]]:
        rows = [self._row(e) for e in data["entries"]
                if entry_visible_to(owner, _owner_from_json(e.get("owner")))
                and (kind is None or e.get("kind") == kind)]
        return sorted(rows, key=lambda r: r["name"])

    # --- the port --------------------------------------------------------------------------

    def install(self, *, name, kind, version, content_hash, manifest, scan_result=None, level="owner"):
        scope = self._pin()
        self._administer(scope)
        self._check_install_args(name=name, kind=kind, version=version, content_hash=content_hash,
                                 manifest=manifest, scan_result=scan_result, level=level)
        self._check_levels(level)
        data = self._load()
        entry = {"name": name, "kind": kind, "version": version, "content_hash": content_hash,
                 "manifest": manifest, "scan_result": scan_result, "enabled": True, "level": level,
                 "owner": _owner_to_json(LOCAL_OWNER if level == "owner" else None)}
        data["entries"] = [e for e in data["entries"]
                           if not (e.get("name") == name and e.get("level") == level)]
        data["entries"].append(entry)
        self._save(data)
        return self._row(entry)

    def get(self, name):
        scope = self._pin()
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        row = self._resolve(self._load(), name, owner)
        return self._row(row) if row is not None else None

    def list(self, *, kind=None):
        scope = self._pin()
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        return self._visible(self._load(), owner, kind=kind)

    def set_enabled(self, name, enabled):
        scope = self._pin()
        self._administer(scope)
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        data = self._load()
        row = self._resolve(data, name, owner)
        if row is None:
            return False
        row["enabled"] = bool(enabled)
        self._save(data)
        return True

    def remove(self, name):
        scope = self._pin()
        self._administer(scope)
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        data = self._load()
        row = self._resolve(data, name, owner)
        if row is None:
            return False
        data["entries"].remove(row)
        data["switches"].pop(self._switch_key(row), None)
        self._save(data)
        return True

    def set_agent_enabled(self, name, enabled):
        scope = self._pin()
        self._administer(scope)
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        data = self._load()
        row = self._resolve(data, name, owner)
        if row is None:
            return False
        data["switches"][self._switch_key(row)] = bool(enabled)
        self._save(data)
        return True

    def agent_enabled(self, name):
        scope = self._pin()
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        data = self._load()
        row = self._resolve(data, name, owner)
        if row is None:
            return False
        return bool(data["switches"].get(self._switch_key(row)))

    def enabled_extensions(self, *, kind=None):
        scope = self._pin()
        owner = LOCAL_OWNER if scope is None or scope.owner is None else scope.owner
        data = self._load()
        # One entry per name, resolved exactly as ``get`` resolves it, then offered when it is
        # enabled and this agent's switch for that entry is on.
        offered = []
        for name in sorted({e.get("name") for e in data["entries"]
                            if entry_visible_to(owner, _owner_from_json(e.get("owner")))}):
            row = self._resolve(data, name, owner)
            if row is None or not row.get("enabled"):
                continue
            if kind is not None and row.get("kind") != kind:
                continue
            if not data["switches"].get(self._switch_key(row)):
                continue
            offered.append(self._row(row))
        return offered


class LocalExtensionRegistryProvider(ExtensionRegistryProvider):
    """One small ``extension_registry.json`` per profile home. Not wired into local plugin
    loading: local installs keep today's in-process plugins, which never read this file."""

    @property
    def name(self) -> str:
        return "local"

    def open(self, home: Any = None) -> ExtensionRegistry:
        """The registry of *home*, or of the ambient profile home. The home is frozen here; the
        pin is checked by every operation."""
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        return LocalExtensionRegistry(Path(os.fspath(home)))

    def assert_isolating(self) -> None:
        """A local registry is one file per profile and refuses every foreign pin
        (``LocalExtensionRegistry._pin``); there is no shared role that could bypass that."""
        return
