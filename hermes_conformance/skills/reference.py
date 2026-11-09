"""An in-memory skills index that stores every level, for the resolution conformance tests.

The shipped local backend stores the agent level only, so the suite's resolution tests
(agent → owner-wide → tenant-wide) need a backend that actually holds owner-wide and tenant-wide
rows: this one. It is test scaffolding inside the conformance package, never a backend
production code may use: rows are plain dicts in one process, but every operation is pinned to
the bound scope and resolves through the port's one policy function, exactly as a real hosted
backend must.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

from agent.runtime_scope import bind_state
from hermes_scope import Group, Principal, Scope, current_scope
from hermes_skills_port import (
    AGENT_LEVEL, SKILL_LEVELS, SkillsIndexing, SkillsLevelRefused, SkillIndex, StoragePinRefused,
    resolution_order,
)

#: The tenant the reference backend serves; a second one proves tenant isolation.
REFERENCE_TENANT = "conformance"
OTHER_TENANT = "other"

#: Every agent of the reference tenant, and its owner: ``a`` and ``c`` share one owner.
REFERENCE_AGENTS = ("a", "b", "c")
_OWNERS = {"a": Principal("o-a"), "b": Principal("o-b"), "c": Principal("o-a")}


def shared_reference_store() -> dict[str, Any]:
    """One store: ``{tenant: {(level, subject, agent): {skill_name: stored}}}``. Every index
    over it shares the storage, so two agents of one tenant — and two tenants — see one backend."""
    return {}


class ReferenceSkillsIndex(SkillIndex):
    """Dict-backed and hosted-shaped: no scope bound means no service, and every row is keyed by
    the scope's own tenant, agent and owner. Reads resolve through
    :func:`hermes_skills_port.resolution_order`."""

    levels = frozenset(SKILL_LEVELS)

    def __init__(self, store: dict[str, Any]) -> None:
        self.store = store

    # --- the pin and the levels

    def _pin(self):
        scope = current_scope()
        if scope is None:
            raise StoragePinRefused("a hosted skills index serves a bound scope only")
        return scope

    def _store_level(self, level: str) -> None:
        if level not in SKILL_LEVELS:
            raise SkillsLevelRefused(f"a skills index stores levels {sorted(SKILL_LEVELS)} only")

    def _read_keys(self, scope) -> list[tuple[str, Optional[str], Optional[str]]]:
        """The storage keys a read of the bound agent's skills resolves through, in the order
        :func:`hermes_skills_port.resolution_order` gives: the agent's own shelf, its owner's,
        the tenant's."""
        keys = []
        for level, subject in resolution_order(scope):
            if level == "agent":
                keys.append(("agent", None, scope.agent))
            elif level == "owner":
                keys.append(("owner", subject.id if subject is not None else None, None))
            else:
                keys.append(("tenant", None, None))
        return keys

    @staticmethod
    def _write_key(scope, level: str) -> tuple[str, Optional[str], Optional[str]]:
        """The one shelf a write at *level* lands on, from the scope alone."""
        if level == "agent":
            return ("agent", None, scope.agent)
        if level == "owner":
            return ("owner", scope.owner.id, None)
        return ("tenant", None, None)

    def _shelf(self, scope, key) -> dict[str, Any]:
        return self.store.setdefault(scope.tenant, {}).setdefault(key, {})

    def _resolved(self, scope) -> dict[str, tuple[str, dict[str, Any]]]:
        """``{name: (level, stored)}`` for every skill the scope resolves; the first level to hold
        a name wins, in :func:`hermes_skills_port.resolution_order` order."""
        tenant = self.store.setdefault(scope.tenant, {})
        resolved: dict[str, tuple[str, dict[str, Any]]] = {}
        for key in self._read_keys(scope):
            for name, stored in tenant.get(key, {}).items():
                resolved.setdefault(name, (key[0], stored))
        return resolved

    def _row(self, name: str, stored: dict[str, Any], level: str) -> dict[str, Any]:
        return {"name": name, "description": stored["description"], "category": stored["category"],
                "status": "unique", "load_name": name, "level": level}

    # --- reads

    def list(self) -> list[dict[str, Any]]:
        scope = self._pin()
        resolved = self._resolved(scope)
        return [self._row(name, stored, level) for name, (level, stored) in sorted(resolved.items())]

    def get(self, name: str) -> Optional[dict[str, Any]]:
        scope = self._pin()
        hit = self._resolved(scope).get(name)
        return None if hit is None else self._row(name, hit[1], hit[0])

    def read_body(self, name: str, file_path: Optional[str] = None) -> Optional[str]:
        scope = self._pin()
        hit = self._resolved(scope).get(name)
        if hit is None:
            return None
        stored = hit[1]
        if file_path is None:
            return stored["body"]
        return stored["files"].get(file_path)

    # --- writes

    def put(self, name: str, content: str, *, category: Optional[str] = None,
            level: str = AGENT_LEVEL) -> dict[str, Any]:
        from agent.skill_utils import parse_frontmatter
        scope = self._pin()
        self._store_level(level)
        frontmatter, _body = parse_frontmatter(content)
        stored = {"description": str(frontmatter.get("description", "")),
                  "category": category, "body": content, "files": {}}
        self._shelf(scope, self._write_key(scope, level))[name] = stored
        return self._row(name, stored, level)

    def put_file(self, name: str, file_path: str, content: str, *, level: str = AGENT_LEVEL) -> None:
        scope = self._pin()
        self._store_level(level)
        target = _relative_support_path(file_path)
        shelf = self._shelf(scope, self._write_key(scope, level))
        if name not in shelf:
            raise LookupError(f"skill {name!r} not found; create it first")
        shelf[name]["files"][target] = content

    def delete(self, name: str, *, level: str = AGENT_LEVEL) -> bool:
        scope = self._pin()
        self._store_level(level)
        return self._shelf(scope, self._write_key(scope, level)).pop(name, None) is not None

    def delete_file(self, name: str, file_path: str, *, level: str = AGENT_LEVEL) -> bool:
        scope = self._pin()
        self._store_level(level)
        target = _relative_support_path(file_path)
        shelf = self._shelf(scope, self._write_key(scope, level))
        return shelf.get(name, {}).pop(target, None) is not None


def _relative_support_path(file_path: str) -> str:
    """A supporting-file path as the reference stores it: relative, no escape."""
    from pathlib import PurePosixPath
    candidate = PurePosixPath(file_path)
    if not file_path or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Path traversal is not allowed; got {file_path!r}")
    return file_path


def reference_backend(store: Optional[dict[str, Any]] = None):
    """The suite's :class:`~hermes_conformance.skills.suite.Backend` over one shared reference
    store: every agent of the reference tenant (its owner comes with it), a second tenant, and
    no scope bound — which the backend refuses, as a hosted one must."""
    from hermes_conformance.skills.suite import Backend

    if store is None:
        store = shared_reference_store()

    def bind(agent: str):
        owner = _OWNERS[agent]
        return bind_state(Scope(REFERENCE_TENANT, agent, Principal("member"),
                                owner=owner, can_administer=True))

    def foreign_tenant(agent: str):
        return bind_state(Scope(OTHER_TENANT, agent, Principal("member"),
                               owner=Group("other"), can_administer=False))

    def open(agent: str) -> SkillsIndexing:
        return SkillsIndexing(ReferenceSkillsIndex(store))

    return Backend(open=open, bind=bind, foreign_tenant=foreign_tenant, unbound=nullcontext,
                   levels=frozenset(SKILL_LEVELS))
