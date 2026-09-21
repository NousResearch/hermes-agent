"""``hermes sessions repair-profiles`` — settle crossed-profile durable state across every store.

The per-profile store model (#88734) and the identity fences that followed are forward-only: they
put NEW rows in the right place and refuse to widen existing damage, but nothing walks the stores
and settles what an earlier release left crossed. This does, for six kinds of crossing:

1. ``sessions.profile_name`` disagreeing with the profile in the row's own ``session_key``;
2. rows physically in the wrong store (a named profile's keys in the root ``state.db``, another
   profile's keys in ``profiles/<p>/state.db``) — moved to the owning profile's store;
3. ``parent_session_id`` crossing namespaces — severed (the child's own identity is untouched);
4. ``gateway_routing`` rows in a named profile's store (the routing index lives in the gateway's
   home; #66887 copied every profile's rows into whichever store was scoped) — moved to the routing
   store when absent there, dropped otherwise — and root rows for a namespace no profile claims;
5. profile-less Telegram topic bindings and ``gateway_voice_mode.json`` entries owned by a
   non-default bot — relabelled from the evidence of the rows themselves;
6. ``sessions.json`` mirror entries whose key namespace no served profile claims — dropped, since
   the legacy import re-injects any key the routing DB lacks on every boot.

Legacy ``agent:main:`` rows inside a named profile's store are reported but NOT repaired unless
``--legacy-main`` says how: they are either a standalone (pre-multiplex) gateway's own history that
must be rekeyed to ``agent:<p>:`` (the #113884 incident), or a default-profile chat that leaked in
under a scoped write (#102157) and must move to the root store. The row carries no evidence that
tells the two apart, and guessing wrong hands a conversation to the other bot.

Report-only by default. ``--apply`` refuses while a gateway owning any touched store is live (it
holds the routing index in memory and writes it back), takes a quick snapshot of every store it
will mutate, and is idempotent: a second run finds nothing.
"""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_LEGACY_MAIN_CHOICES = ("report", "rekey", "move")
_VOICE_MODE_FILE = "gateway_voice_mode.json"


@dataclass(frozen=True)
class Store:
    """One profile's ``state.db``; ``routing`` marks the store that owns the multiplexer's index."""
    profile: str
    home: Path
    routing: bool = False

    @property
    def db_path(self) -> Path:
        return self.home / "state.db"


@dataclass
class Finding:
    kind: str
    store: str
    subject: str
    detail: str
    action: Optional[str] = None      # what --apply does; None = report only
    reason: Optional[str] = None      # why it is report only
    _fix: Optional[Callable[["_Session"], Dict[str, int]]] = field(default=None, repr=False, compare=False)

    @property
    def repairable(self) -> bool:
        return self._fix is not None

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "store": self.store, "subject": self.subject, "detail": self.detail,
                "action": self.action, "reason": self.reason}


class _Session:
    """Open store handles for one run; every store opened at most once, all released at the end."""

    def __init__(self, *, read_only: bool) -> None:
        self.read_only = read_only
        self._dbs: Dict[Path, Any] = {}

    def db(self, store: Store):
        """The store's SessionDB. Read-only runs never create a file: a live profile that has no
        ``state.db`` yet reads as empty; an apply run opens it for real when a move lands there."""
        path = store.db_path
        if path not in self._dbs:
            if self.read_only:
                from hermes_state import SessionDB
                self._dbs[path] = SessionDB(path, read_only=True) if path.exists() else _EMPTY_STORE
            else:
                from hermes_state_registry import acquire
                self._dbs[path] = acquire(path)
        return self._dbs[path]

    def close(self) -> None:
        from hermes_state_registry import release_or_close
        for db in self._dbs.values():
            if db is _EMPTY_STORE:
                continue
            with contextlib.suppress(Exception):
                release_or_close(db) if not self.read_only else db.close()
        self._dbs.clear()


class _EmptyStore:
    """Stand-in for a live profile whose ``state.db`` does not exist yet: nothing to find."""
    def find_crossed_profile_sessions(self, owner):
        return {"mislabelled": [], "foreign": [], "crossed_parents": []}

    def find_profile_less_telegram_topic_rows(self):
        return []

    def list_gateway_routing_rows(self):
        return []

    def key_profiles_for_chat(self, platform, chat_id):
        return set()


_EMPTY_STORE = _EmptyStore()


# ── enumeration ───────────────────────────────────────────────────────────────

def enumerate_stores() -> List[Store]:
    """Default root plus every live named profile (a live profile claims its namespace whether or not
    it has written a ``state.db`` yet). The root store owns the routing index: the multiplexer's
    ``_routing_home`` is its launch home, the root."""
    from hermes_cli.profiles import _get_default_hermes_home, _iter_named_profile_dirs
    root = _get_default_hermes_home()
    stores = [Store("default", root, routing=True)]
    stores.extend(Store(entry.name, entry) for entry in _iter_named_profile_dirs())
    return stores


def _gateway_multiplexes(root: Path) -> bool:
    """Does the default gateway serve every profile? Same reader as every other CLI surface: the live
    record, else the explicit flag, else False — an unset flag is a verdict only the gateway reaches,
    and guessing "yes" would uproot a standalone gateway's own routing index."""
    from hermes_cli.gateway_multiplex_mode import default_gateway_multiplexes
    try:
        return default_gateway_multiplexes(root)
    except Exception:
        return False


def live_gateway_homes(stores: Iterable[Store]) -> List[Tuple[str, int]]:
    from gateway.status import live_gateway_pid_for_home
    live = []
    for store in stores:
        pid = live_gateway_pid_for_home(store.home)
        if pid is not None:
            live.append((store.profile, pid))
    return live


# ── scan ──────────────────────────────────────────────────────────────────────

class RepairPlan:
    def __init__(self, stores: List[Store], *, legacy_main: str = "report") -> None:
        if legacy_main not in _LEGACY_MAIN_CHOICES:
            raise ValueError(f"legacy_main must be one of {_LEGACY_MAIN_CHOICES}")
        self.stores = stores
        self.by_profile = {s.profile: s for s in stores}
        self.claimed: Set[str] = set(self.by_profile)
        self.routing_store = next((s for s in stores if s.routing), None)
        self.legacy_main = legacy_main
        self.multiplexes = _gateway_multiplexes(self.routing_store.home) if self.routing_store else False
        self.findings: List[Finding] = []
        self._moves: Dict[Tuple[Path, Path], "_MoveBatch"] = {}

    # -- helpers --
    def _add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def repairable(self) -> List[Finding]:
        return [f for f in self.findings if f.repairable]

    # -- scan --
    def scan(self, session: _Session) -> "RepairPlan":
        for store in self.stores:
            self._scan_sessions(session, store)
            self._scan_topics(session, store)
        for store in self.stores:
            self._scan_routing(session, store)
        if self.routing_store is not None:
            self._scan_voice_modes(session, self.routing_store)
            self._scan_sessions_json(self.routing_store)
        return self

    def _scan_sessions(self, session: _Session, store: Store) -> None:
        db = session.db(store)
        crossed = db.find_crossed_profile_sessions(store.profile)
        moving: Set[str] = set()
        for row in crossed["foreign"]:
            self._plan_foreign_row(store, row, moving)
        for row in crossed["mislabelled"]:
            if row["id"] in moving:
                continue  # the move stamps the right label
            self._add(Finding(
                "mislabelled", store.profile, row["id"],
                f"profile_name={row['profile_name']!r} but key {row['session_key']!r} belongs to "
                f"{row['key_profile']!r}", action=f"relabel to {row['key_profile']!r}",
                _fix=lambda s, st=store, sid=row["id"]: {
                    "relabelled": s.db(st).relabel_sessions_to_key_profile([sid])}))
        for row in crossed["crossed_parents"]:
            if row["id"] in moving:
                continue  # the import drops a parent that is not in the target store
            self._add(Finding(
                "crossed_parent", store.profile, row["id"],
                f"{row['key_profile']!r} row inherits from parent {row['parent_session_id']} keyed under "
                f"{row['parent_profile']!r}", action="sever parent_session_id",
                _fix=lambda s, st=store, sid=row["id"]: {"severed": s.db(st).sever_crossed_parents([sid])}))

    def _plan_foreign_row(self, store: Store, row: Dict[str, Any], moving: Set[str]) -> None:
        key_profile, sid = row["key_profile"], row["id"]
        detail = f"key {row['session_key']!r} ({row['message_count']} messages) sits in the {store.profile} store"
        if key_profile == "default" and store.profile != "default":
            self._plan_legacy_main_row(store, row, moving)
            return
        if key_profile not in self.claimed:
            self._add(Finding(
                "unclaimed_namespace", store.profile, sid, detail,
                reason=f"no live profile named {key_profile!r}; create it, then rerun, or "
                       f"`hermes profile migrate-identity {key_profile} <target>`"))
            return
        moving.add(sid)
        self._add(Finding(
            "wrong_store", store.profile, sid, detail, action=f"move to the {key_profile} store",
            _fix=self._move_batch(store, self.by_profile[key_profile]).fix_for(sid)))

    def _plan_legacy_main_row(self, store: Store, row: Dict[str, Any], moving: Set[str]) -> None:
        sid = row["id"]
        detail = (f"legacy default-namespace key {row['session_key']!r} ({row['message_count']} messages) "
                  f"in the {store.profile} store")
        if self.legacy_main == "rekey":
            self._add(Finding(
                "legacy_main", store.profile, sid, detail, action=f"rekey to agent:{store.profile}:",
                _fix=lambda s, st=store, i=sid: {"rekeyed": s.db(st).rekey_legacy_main_sessions([i], st.profile)}))
        elif self.legacy_main == "move" and self.routing_store is not None:
            moving.add(sid)
            self._add(Finding(
                "legacy_main", store.profile, sid, detail, action="move to the default store",
                _fix=self._move_batch(store, self.routing_store).fix_for(sid)))
        else:
            self._add(Finding(
                "legacy_main", store.profile, sid, detail,
                reason="either a standalone gateway's own history (--legacy-main rekey) or a default "
                       "chat that leaked in (--legacy-main move); the row cannot say which"))

    def _move_batch(self, src: Store, dst: Store) -> "_MoveBatch":
        return self._moves.setdefault((src.db_path, dst.db_path), _MoveBatch(src, dst))

    def _scan_topics(self, session: _Session, store: Store) -> None:
        rows = session.db(store).find_profile_less_telegram_topic_rows()
        for row in rows:
            self._add(Finding(
                "topic_profile_less", store.profile, f"chat {row['chat_id']} thread {row['thread_id']}",
                f"telegram topic binding labelled 'default' but keyed {row['session_key']!r}",
                action=f"relabel to {row['key_profile']!r}",
                _fix=lambda s, st=store, r=row: s.db(st).relabel_telegram_topic_rows([r])))

    def _scan_routing(self, session: _Session, store: Store) -> None:
        from hermes_state_profile_repair import session_key_profile
        rows = session.db(store).list_gateway_routing_rows()
        for row in rows:
            key_profile = session_key_profile(row["session_key"])
            if key_profile is None:
                continue
            subject = f"{row['session_key']} (scope {row['scope']!r})"
            if store.routing:
                if key_profile not in self.claimed:
                    self._add(Finding(
                        "routing_unclaimed", store.profile, subject,
                        f"routing row for a namespace no live profile claims ({key_profile!r}); every "
                        "inbound event on it logs a missing-profile error", action="delete",
                        _fix=lambda s, st=store, r=row: {
                            "routing_deleted": s.db(st).delete_gateway_routing_rows([(r["scope"], r["session_key"])])}))
                continue
            if key_profile == store.profile and not self.multiplexes:
                continue  # a standalone gateway's own index lives in its own store
            routing_store = self.routing_store
            if routing_store is None:
                continue
            self._add(Finding(
                "routing_stray", store.profile, subject,
                f"routing row in the {store.profile} store; the index lives in the default store",
                action="move to the default store (existing row there wins)",
                _fix=lambda s, st=store, r=row, rs=routing_store: self._move_routing_row(s, st, rs, r)))

    def _move_routing_row(self, session: _Session, store: Store, routing_store: Store,
                          row: Dict[str, Any]) -> Dict[str, int]:
        adopted = session.db(routing_store).insert_gateway_routing_rows_if_absent(
            [(row["scope"], row["session_key"], row["entry_json"], row["updated_at"])])
        deleted = session.db(store).delete_gateway_routing_rows([(row["scope"], row["session_key"])])
        return {"routing_adopted": adopted, "routing_deleted": deleted}

    def _scan_voice_modes(self, session: _Session, store: Store) -> None:
        path = store.home / _VOICE_MODE_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        for key in sorted(k for k in data if isinstance(k, str) and k.count(":") == 1):
            platform, chat_id = key.split(":", 1)
            owners: Set[str] = set()
            for st in self.stores:
                owners |= session.db(st).key_profiles_for_chat(platform, chat_id)
            if not owners or "default" in owners:
                continue  # the default bot speaks there: the unprefixed key is its own
            if len(owners) > 1:
                self._add(Finding(
                    "voice_profile_less", store.profile, key,
                    f"voice mode entry without a profile; chat is held by {sorted(owners)}",
                    reason="more than one non-default bot speaks in this chat"))
                continue
            owner = next(iter(owners))
            self._add(Finding(
                "voice_profile_less", store.profile, key,
                f"voice mode {data[key]!r} without a profile; chat is held only by {owner!r}",
                action=f"rekey to {owner}:{key}",
                _fix=lambda s, p=path, k=key, o=owner: _rekey_voice_mode_entry(p, k, o)))

    def _scan_sessions_json(self, store: Store) -> None:
        from hermes_state_profile_repair import session_key_profile
        path = store.home / "sessions" / "sessions.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        for key in sorted(k for k in data if isinstance(k, str) and not k.startswith("_")):
            key_profile = session_key_profile(key)
            if key_profile is None or key_profile in self.claimed:
                continue
            self._add(Finding(
                "sessions_json_unclaimed", store.profile, key,
                f"sessions.json mirror entry for a namespace no live profile claims ({key_profile!r}); "
                "the legacy import re-injects it into routing on every boot", action="delete entry",
                _fix=lambda s, p=path, k=key: _drop_sessions_json_entry(p, k)))

    # -- apply --
    def apply(self, session: _Session, *, snapshot: Callable[[Store], Optional[str]]) -> Dict[str, Any]:
        """Run every repairable fix; snapshots every store first. Fixes are independent and each
        re-checks its precondition, so a partial run leaves a state the next run completes."""
        snapshots = {s.profile: snapshot(s) for s in self.stores if s.db_path.exists()}
        totals: Dict[str, int] = {}
        failures: List[Dict[str, str]] = []
        for finding in self.repairable:
            fix = finding._fix
            assert fix is not None
            try:
                for key, count in (fix(session) or {}).items():
                    totals[key] = totals.get(key, 0) + int(count)
            except Exception as exc:  # one bad row must not abandon the rest of the plan
                logger.warning("repair-profiles: %s on %s failed", finding.kind, finding.subject, exc_info=True)
                failures.append({"kind": finding.kind, "subject": finding.subject,
                                 "error": f"{type(exc).__name__}: {exc}"})
        return {"snapshots": snapshots, "totals": totals, "failures": failures}


class _MoveBatch:
    """Every session moving from one store to another, moved together: export all, import all
    (parents before children, so a moved child keeps its moved parent), then delete from the source.
    Row by row, deleting a parent would detach the children still waiting in the source. Copy
    precedes delete, so a crash between the two leaves a duplicate the next run settles
    (``present`` on import, then the delete)."""

    def __init__(self, src: Store, dst: Store) -> None:
        self.src, self.dst = src, dst
        self.ids: List[str] = []
        self._result: Optional[Dict[str, Dict[str, int]]] = None

    def fix_for(self, sid: str) -> Callable[[_Session], Dict[str, int]]:
        self.ids.append(sid)
        return lambda session: self.run(session).get(sid, {"missing": 1})

    def run(self, session: _Session) -> Dict[str, Dict[str, int]]:
        if self._result is not None:
            return self._result
        src_db, dst_db = session.db(self.src), session.db(self.dst)
        payloads = {sid: src_db.export_session_for_move(sid) for sid in self.ids}
        ordered = _parents_first([p for p in payloads.values() if p is not None])
        result: Dict[str, Dict[str, int]] = {sid: {"missing": 1} for sid, p in payloads.items() if p is None}
        for payload in ordered:
            sid = payload["session"]["id"]
            outcome = dst_db.import_moved_session(payload, profile_name=self.dst.profile)
            if dst_db.count_messages_all(sid) < len(payload["messages"]):
                logger.warning("repair-profiles: %s copied into %s with fewer messages than the source; "
                               "source row kept", sid, self.dst.profile)
                result[sid] = {"copied_incomplete": 1}
                continue
            result[sid] = {outcome: 1}
        for payload in ordered:
            sid = payload["session"]["id"]
            if "copied_incomplete" not in result[sid] and src_db.delete_moved_session(sid):
                result[sid]["moved"] = 1
        self._result = result
        return result


def _parents_first(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order moved sessions so every parent in the batch precedes its children (a cycle, impossible
    for a well-formed store, falls back to insertion order)."""
    pending = {p["session"]["id"]: p for p in payloads}
    ordered: List[Dict[str, Any]] = []
    while pending:
        ready = [sid for sid, p in pending.items() if p["session"].get("parent_session_id") not in pending]
        if not ready:
            ready = list(pending)
        for sid in ready:
            ordered.append(pending.pop(sid))
    return ordered


def _rekey_voice_mode_entry(path: Path, key: str, owner: str) -> Dict[str, int]:
    from utils import atomic_json_write
    data = json.loads(path.read_text(encoding="utf-8"))
    if key not in data:
        return {}
    data.setdefault(f"{owner}:{key}", data.pop(key))
    data.pop(key, None)
    atomic_json_write(path, data)
    return {"voice_rekeyed": 1}


def _drop_sessions_json_entry(path: Path, key: str) -> Dict[str, int]:
    from utils import atomic_json_write
    data = json.loads(path.read_text(encoding="utf-8"))
    if key not in data:
        return {}
    del data[key]
    atomic_json_write(path, data, mode=0o600)
    return {"sessions_json_dropped": 1}


# ── entry points ──────────────────────────────────────────────────────────────

def scan_stores(stores: Optional[List[Store]] = None, *, legacy_main: str = "report",
                read_only: bool = True) -> Tuple[RepairPlan, _Session]:
    """Open every store and build the plan. The caller owns the returned session (close it)."""
    stores = enumerate_stores() if stores is None else stores
    session = _Session(read_only=read_only)
    try:
        plan = RepairPlan(stores, legacy_main=legacy_main).scan(session)
    except Exception:
        session.close()
        raise
    return plan, session


def default_snapshot(store: Store) -> Optional[str]:
    from hermes_cli.backup import create_quick_snapshot
    try:
        return create_quick_snapshot(label="repair-profiles", hermes_home=store.home)
    except Exception as exc:
        logger.warning("repair-profiles: snapshot of %s failed: %s", store.home, exc)
        return None
