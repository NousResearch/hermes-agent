"""SQLite-backed resource store for the enterprise control plane.

One store per Installation. The store enforces the invariants the resource
model declares:

  * unique (kind, namespace, name)
  * namespace-scoped resources require an existing, non-terminating Namespace
  * namespace-scoped references resolve within the same namespace only
    (REFERENCE_FIELDS), fail-closed
  * optimistic concurrency via meta.generation
  * AgentRevision rows are immutable after creation except for their status
  * deletion is refused while dependents reference the resource

WAL journaling is used when available (mirrors hermes_state.py behavior).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .errors import ConflictError, NotFoundError, ScopeError, ValidationError
from .resources import (
    Kind,
    NamespacePhase,
    REFERENCE_FIELDS,
    Resource,
    now_ts,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS resources (
    uid        TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    namespace  TEXT,
    name       TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 1,
    doc        TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_resources_identity
    ON resources(kind, COALESCE(namespace, ''), name);
CREATE INDEX IF NOT EXISTS idx_resources_kind_ns ON resources(kind, namespace);
"""


class ResourceStore:
    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, res: Resource) -> Resource:
        res.validate()
        with self._lock, self._conn:
            self._check_scope(res)
            self._check_references(res)
            try:
                self._conn.execute(
                    "INSERT INTO resources (uid, kind, namespace, name, generation,"
                    " doc, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        res.meta.uid,
                        res.meta.kind,
                        res.meta.namespace,
                        res.meta.name,
                        res.meta.generation,
                        json.dumps(res.to_dict()),
                        res.meta.created_at,
                        res.meta.updated_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError(
                    f"{res.meta.kind} {self._fqn(res)} already exists"
                ) from exc
        return res

    def get(self, kind: Kind | str, name: str, namespace: str | None = None) -> Resource:
        kind = Kind(kind).value
        with self._lock:
            row = self._conn.execute(
                "SELECT doc FROM resources WHERE kind=? AND"
                " COALESCE(namespace,'')=COALESCE(?,'') AND name=?",
                (kind, namespace, name),
            ).fetchone()
        if row is None:
            where = f"namespace {namespace!r}" if namespace else "installation scope"
            raise NotFoundError(f"{kind} {name!r} not found in {where}")
        return Resource.from_dict(json.loads(row["doc"]))

    def list(self, kind: Kind | str, namespace: str | None = None) -> list[Resource]:
        kind = Kind(kind).value
        with self._lock:
            if namespace is None:
                rows = self._conn.execute(
                    "SELECT doc FROM resources WHERE kind=? ORDER BY name", (kind,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT doc FROM resources WHERE kind=? AND namespace=?"
                    " ORDER BY name",
                    (kind, namespace),
                ).fetchall()
        return [Resource.from_dict(json.loads(r["doc"])) for r in rows]

    def update_spec(self, res: Resource) -> Resource:
        """Update spec with optimistic concurrency; bumps generation.

        AgentRevision specs are immutable: only status may change after
        creation (use update_status).
        """
        res.validate()
        if Kind(res.meta.kind) is Kind.AGENT_REVISION:
            raise ValidationError(
                "AgentRevision is immutable; only its status may be updated"
            )
        with self._lock, self._conn:
            current = self.get(res.meta.kind, res.meta.name, res.meta.namespace)
            if current.meta.generation != res.meta.generation:
                raise ConflictError(
                    f"{res.meta.kind} {self._fqn(res)} generation conflict: "
                    f"store has {current.meta.generation}, caller has "
                    f"{res.meta.generation}"
                )
            self._check_references(res)
            res.meta.generation += 1
            res.meta.updated_at = now_ts()
            res.meta.uid = current.meta.uid
            res.meta.created_at = current.meta.created_at
            self._conn.execute(
                "UPDATE resources SET generation=?, doc=?, updated_at=? WHERE uid=?",
                (res.meta.generation, json.dumps(res.to_dict()), res.meta.updated_at,
                 res.meta.uid),
            )
        return res

    def update_status(self, kind: Kind | str, name: str, namespace: str | None,
                      status: dict[str, Any]) -> Resource:
        """Controller-owned status update; does not bump generation."""
        with self._lock, self._conn:
            res = self.get(kind, name, namespace)
            res.status = dict(status)
            res.validate()
            res.meta.updated_at = now_ts()
            self._conn.execute(
                "UPDATE resources SET doc=?, updated_at=? WHERE uid=?",
                (json.dumps(res.to_dict()), res.meta.updated_at, res.meta.uid),
            )
        return res

    def delete(self, kind: Kind | str, name: str, namespace: str | None = None) -> None:
        kind_e = Kind(kind)
        with self._lock, self._conn:
            res = self.get(kind_e, name, namespace)
            dependents = self._find_dependents(res)
            if dependents:
                raise ConflictError(
                    f"cannot delete {kind_e.value} {name!r}: referenced by "
                    + ", ".join(sorted(dependents))
                )
            if kind_e is Kind.NAMESPACE:
                with_ns = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM resources WHERE namespace=?", (name,)
                ).fetchone()["n"]
                if with_ns:
                    raise ConflictError(
                        f"cannot delete Namespace {name!r}: {with_ns} resources "
                        "remain; drain the namespace first"
                    )
            self._conn.execute("DELETE FROM resources WHERE uid=?", (res.meta.uid,))

    # ------------------------------------------------------------------
    # Invariant enforcement
    # ------------------------------------------------------------------

    def _check_scope(self, res: Resource) -> None:
        if res.meta.namespace is None:
            return
        try:
            ns = self.get(Kind.NAMESPACE, res.meta.namespace)
        except NotFoundError as exc:
            raise ScopeError(
                f"{res.meta.kind} {res.meta.name!r} references missing "
                f"Namespace {res.meta.namespace!r}"
            ) from exc
        phase = ns.status.get("phase", NamespacePhase.PENDING.value)
        if phase == NamespacePhase.TERMINATING.value:
            raise ScopeError(
                f"Namespace {res.meta.namespace!r} is terminating and cannot "
                "admit new resources"
            )

    def _check_references(self, res: Resource) -> None:
        fields = REFERENCE_FIELDS.get(Kind(res.meta.kind), ())
        for key, target_kind, is_list in fields:
            raw = res.spec.get(key)
            if raw in (None, "", []):
                continue
            names = raw if is_list else [raw]
            for target in names:
                try:
                    self.get(target_kind, target, res.meta.namespace)
                except NotFoundError as exc:
                    raise ScopeError(
                        f"{res.meta.kind} {res.meta.name!r} references "
                        f"{target_kind.value} {target!r} which does not exist "
                        f"in namespace {res.meta.namespace!r} (cross-namespace "
                        "references are not permitted)"
                    ) from exc
        # Agent.harness is installation-scoped:
        if Kind(res.meta.kind) is Kind.AGENT:
            harness = res.spec.get("harness")
            if harness:
                self.get(Kind.HARNESS, harness, None)  # raises NotFoundError

    def _find_dependents(self, res: Resource) -> set[str]:
        """Names of same-namespace resources whose reference fields point here."""
        deps: set[str] = set()
        target_kind = Kind(res.meta.kind)
        for src_kind, fields in REFERENCE_FIELDS.items():
            relevant = [(k, lst) for k, tk, lst in fields if tk is target_kind]
            if not relevant:
                continue
            for candidate in self.list(src_kind, res.meta.namespace):
                for key, is_list in relevant:
                    raw = candidate.spec.get(key)
                    names = raw if is_list else [raw]
                    if names and res.meta.name in [n for n in names if n]:
                        deps.add(f"{src_kind.value}/{candidate.meta.name}")
        if target_kind is Kind.HARNESS:
            for ns in self.list(Kind.NAMESPACE):
                for agent in self.list(Kind.AGENT, ns.meta.name):
                    if agent.spec.get("harness") == res.meta.name:
                        deps.add(f"Agent/{ns.meta.name}/{agent.meta.name}")
        return deps

    @staticmethod
    def _fqn(res: Resource) -> str:
        if res.meta.namespace:
            return f"{res.meta.namespace}/{res.meta.name}"
        return res.meta.name
