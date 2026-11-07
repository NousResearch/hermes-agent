"""Test seams for the SQLite backend: the fixture hooks the contract suites call on the in-memory reference, given their SQLite meaning.

``ContractSQLiteBackend`` serves any profile the suites name, each at its place under the suites' fake root mirrored under the test's HERMES_HOME, and attaches to every store handle it opens the seeding seams of the reference repositories: ``search.seed_session``, ``search.seed_message``, ``search.seed_deferred_rebuild``, ``usage.seed_session``, ``usage.set_session_archived``, ``sessions.seed_ledger_admission``, ``runtime.seed_local_receipt`` and ``runtime.seed_retirement``. Each seeds the rows main stores, through the contract verbs where one exists and otherwise through direct SQL on the store's holder. Every handle also carries the reference handle's ``owner_profile``, the profile the backend stamps as the store's owner. The seams that need the conversation family's store modules import them on first use and, without the family, refuse as its unregistered repositories do. ``session_db(scope)`` is one profile's store holder (the suites' SQL reads), ``scope_store(scope)["sessions"][id]`` writes raw cells of one stored session row, the reference's row-dict seam, ``begin_epoch(scope, instance_id)`` begins an authority epoch with main's runtime epoch write, ``replace_store(scope)`` puts a copy of the store file at its path, and ``close_store_under(scope)`` force-closes the store under its handles as a profile delete's sweep does. ``root`` is the folder every served profile sits under, so a child process re-assembles the same backend with ``ContractSQLiteBackend({}, root=root)``. These are test plumbing; nothing here is product code.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from persistence import CapabilityUnsupported, MessageId, ProfileScope, RowVisibility, SessionId
from persistence.core.canonical import canonical_json
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.sqlite import SQLiteBackend
from persistence.sqlite._backend import SQLiteStore
from persistence.sqlite._holder import StoreHolder
from tests.contracts._scopes import CONTRACT_ROOT


def _store_module(name: str) -> Any:
    """One of the conversation family's store modules, imported on first use; without the family, the sessions refusal."""
    try:
        return importlib.import_module("persistence.sqlite.conversation._store." + name)
    except ImportError as exc:
        raise CapabilityUnsupported("sessions") from exc


def _write(store: SQLiteStore, sql: str, params: Sequence[Any]) -> int:
    """Run one statement in the holder's write transaction; returns the changed row count."""
    return store.holder()._execute_write(lambda conn: conn.execute(sql, tuple(params)).rowcount)


# ── search: sessions and rows the filters read ──────────────────────────────


def _search_seed_session(store: SQLiteStore, session_id: str, *, source: Optional[str] = None, model: Optional[str] = None,
                         started_at: Optional[float] = None, delegate_child: bool = False) -> SessionId:
    """A stored session row with the start time and source the filters read; a delegate child carries M1's ``_delegate_from`` creation marker."""
    store.sessions.create(NewSession(SessionId(session_id), source, model=model))
    if started_at is not None:
        _write(store, "UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, session_id))
    if delegate_child:
        _write(store, "UPDATE sessions SET model_config = ? WHERE id = ?", (json.dumps({"_delegate_from": "seed-parent"}), session_id))
    return SessionId(session_id)


_VISIBILITY_FLAGS = {RowVisibility.REWOUND: (0, 0), RowVisibility.COMPACTION_ARCHIVED: (0, 1)}


def _search_seed_message(store: SQLiteStore, session_id: str, role: str, content: Any, *, timestamp: float,
                         tool_name: Optional[str] = None, tool_calls: Sequence[Mapping[str, Any]] = (),
                         display_kind: Optional[str] = None, visibility: RowVisibility = RowVisibility.ACTIVE) -> MessageId:
    """One transcript row appended through the messages verb, then given the stored ``active``/``compacted`` flags of ``visibility``."""
    message = NewMessage(role, content, float(timestamp), tool_name=tool_name, tool_calls=tuple(tool_calls), display_kind=display_kind)
    row_id = store.messages.append(SessionId(session_id), message)
    if visibility is not RowVisibility.ACTIVE:
        _write(store, "UPDATE messages SET active = ?, compacted = ? WHERE id = ?", (*_VISIBILITY_FLAGS[visibility], row_id))
    return row_id


def _search_seed_deferred_rebuild(store: SQLiteStore, total: int, indexed: int) -> None:
    """The deferred FTS rebuild markers M1 keeps in ``state_meta``: the high-water id (``total``) and the backfill progress (``indexed``)."""
    holder = store.holder()
    holder.set_meta("fts_rebuild_high_water", str(int(total)))
    holder.set_meta("fts_rebuild_progress", str(int(indexed)))


# ── usage: the session columns the totals filter reads ──────────────────────


def _usage_seed_session(store: SQLiteStore, session_id: str, *, message_count: int = 0,
                        parent_session_id: Optional[str] = None, archived: bool = False) -> SessionId:
    """A ``cli`` session row with the stored message count, parent link and archived flag the totals filter reads."""
    parent = SessionId(parent_session_id) if parent_session_id else None
    store.sessions.create(NewSession(SessionId(session_id), "cli", parent_session_id=parent))
    _write(store, "UPDATE sessions SET message_count = ?, archived = ? WHERE id = ?", (message_count, int(archived), session_id))
    return SessionId(session_id)


def _usage_set_session_archived(store: SQLiteStore, session_id: str, archived: bool) -> None:
    if not _write(store, "UPDATE sessions SET archived = ? WHERE id = ?", (int(archived), session_id)):
        raise KeyError(session_id)


# ── sessions: the admissions ledger the delete fence reads ──────────────────


def _seed_ledger_admission(store: SQLiteStore, session_id: SessionId, *, status: str = "queued") -> None:
    """One ``session_admissions`` row targeting ``session_id`` in ``status``; a started row carries the owner epoch and generation the schema requires."""
    def write(conn: Any) -> None:
        (count,) = conn.execute("SELECT COUNT(*) FROM session_admissions").fetchone()
        admission_id = f"seed-adm-{count}"
        conn.execute(
            "INSERT INTO session_admissions(admission_id, request_id, principal_id, target_session_id, lineage_json,"
            " payload_json, payload_digest, intent, status, owner_epoch, generation) VALUES (?, ?, ?, ?, ?, ?, ?, 'queue', ?, ?, ?)",
            (admission_id, admission_id, "contract-seed", str(session_id), json.dumps([str(session_id)]), "{}", "", status,
             0, 0 if status == "started" else None))
    store.holder()._execute_write(write)


# ── runtime: the local policy receipt and the retirement marker ────────────


def _seed_local_receipt(store: SQLiteStore, session_id: str, receipt: Mapping[str, Any]) -> None:
    """``receipt`` stored as M1 keeps a local policy receipt: canonical JSON under ``POLICY_PREFIX + session_id``."""
    key = _store_module("local").POLICY_PREFIX + session_id
    _write(store, "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)", (key, canonical_json(dict(receipt))))


def _seed_retirement(store: SQLiteStore, session_id: str, *, expires: bool = False) -> None:
    """The retirement marker M1's ``retire_routes`` writes for ``session_id``, alone; ``expires`` makes it a sweep's marker."""
    retirement = _store_module("retirement")
    store.holder()._execute_write(lambda conn: retirement._mark_retired(conn, retirement.RETIRED_PREFIX + session_id, expires=expires, own=True))


# ── raw session cells (the reference's row-dict seam) ───────────────────────


def _cell(value: Any) -> Any:
    """A datetime is stored as M1 stores instants, epoch seconds; anything else lands as given."""
    return value.timestamp() if isinstance(value, datetime) else value


def _session_columns(store: SQLiteStore) -> set[str]:
    with store.holder()._read_ctx() as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}


class _SessionCells:
    """One stored session row: assigning a cell writes that column as given, bypassing every typed write path."""

    def __init__(self, store: SQLiteStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id

    def __setitem__(self, column: str, value: Any) -> None:
        self.update({column: value})

    def update(self, cells: Optional[Mapping[str, Any]] = None, **more: Any) -> None:
        cells = {**(cells or {}), **more}
        unknown = set(cells) - _session_columns(self._store)
        if unknown:
            raise KeyError(f"sessions has no column {sorted(unknown)}")
        assignments = ", ".join(f"{column} = ?" for column in cells)
        params = [_cell(value) for value in cells.values()]
        if not _write(self._store, f"UPDATE sessions SET {assignments} WHERE id = ?", (*params, self._session_id)):
            raise KeyError(self._session_id)


class _SessionRows:
    """The stored session rows of one profile, by id."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def __getitem__(self, session_id: str) -> _SessionCells:
        if self._store.holder()._read_one("SELECT 1 FROM sessions WHERE id = ?", (session_id,)) is None:
            raise KeyError(session_id)
        return _SessionCells(self._store, session_id)


# ── the authority epoch ─────────────────────────────────────────────────────


def _begin_epoch(conn: Any, instance_id: str) -> int:
    """M1 ``begin_runtime_epoch``'s write (hermes_state_runtime.py:110-117): the singleton epoch row advanced and stamped with ``instance_id``."""
    conn.execute('''INSERT INTO runtime_epoch(singleton,epoch,instance_id) VALUES(1,1,?)
            ON CONFLICT(singleton) DO UPDATE SET epoch=runtime_epoch.epoch+1, instance_id=excluded.instance_id''', (instance_id,))
    return int(conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()[0])


# ── the backend ─────────────────────────────────────────────────────────────


class _ContractStore(SQLiteStore):
    """A store handle carrying the reference handle's ``owner_profile``."""

    @property
    def owner_profile(self) -> Optional[str]:
        """The profile this store's owner fence stamps (M1 ``SessionDB._own_profile_name``)."""
        return _store_module("sessions")._own_profile_name(self.holder())


def _attach_seams(store: SQLiteStore) -> SQLiteStore:
    # SQLiteBackend.open builds the handle itself, so the owner property arrives by class.
    store.__class__ = _ContractStore
    for attribute, name, seam in (
        ("search", "seed_session", _search_seed_session),
        ("search", "seed_message", _search_seed_message),
        ("search", "seed_deferred_rebuild", _search_seed_deferred_rebuild),
        ("usage", "seed_session", _usage_seed_session),
        ("usage", "set_session_archived", _usage_set_session_archived),
        ("sessions", "seed_ledger_admission", _seed_ledger_admission),
        ("runtime", "seed_local_receipt", _seed_local_receipt),
        ("runtime", "seed_retirement", _seed_retirement),
    ):
        setattr(getattr(store, attribute), name, partial(seam, store))
    return store


def _contract_place(scope: ProfileScope) -> Path:
    """Where ``scope``'s home sits under the backend's root: its place under the suites' fake root, else ``profiles/<name>``."""
    try:
        return scope.profile_dir.relative_to(CONTRACT_ROOT)
    except ValueError:
        return Path("profiles") / scope.name


class ContractSQLiteBackend(SQLiteBackend):
    """The SQLite backend as the contract suites drive it: every profile they name is served, and every handle carries the seeding seams."""

    def __init__(self, stores: Mapping[str, Path], *, root: Path) -> None:
        super().__init__(stores)
        self.root = root

    def open(self, scope: ProfileScope) -> SQLiteStore:
        # SQLiteBackend has no public hook to serve one more profile, so this fixture class, and nothing else in the suite, registers the store path in its private map.
        if scope.hermes_home_key not in self._stores:
            home = self.root / _contract_place(scope)
            home.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._stores.setdefault(scope.hermes_home_key, home / "state.db")
        return _attach_seams(super().open(scope))

    def session_db(self, scope: ProfileScope) -> StoreHolder:
        """The store holder of ``scope``, for the suites' SQL reads and legacy-function calls."""
        return self.open(scope).holder()

    def scope_store(self, scope: ProfileScope) -> dict[str, Any]:
        """The reference's namespace shape over ``scope``'s store: ``["sessions"][id]`` writes raw cells of one stored row."""
        return {"sessions": _SessionRows(self.open(scope))}

    def begin_epoch(self, scope: ProfileScope, instance_id: str) -> int:
        """Begin an authority epoch on ``scope``'s store through its holder's write, as the reference's authority state does on its own."""
        return int(self.open(scope).holder()._execute_write(partial(_begin_epoch, instance_id=instance_id)))

    def close_store_under(self, scope: ProfileScope) -> None:
        """Force-close ``scope``'s store under its open handles: ``persistence_assembly.close_stores_under`` on its home."""
        from persistence_assembly import close_stores_under

        close_stores_under(self._stores[scope.hermes_home_key].parent)

    def replace_store(self, scope: ProfileScope) -> None:
        """Put a copy of ``scope``'s store file at its path, so the path no longer names the file its open handles opened."""
        path = self._stores[scope.hermes_home_key]
        copy = path.with_name("replacement.db")
        shutil.copyfile(path, copy)
        os.replace(copy, path)
