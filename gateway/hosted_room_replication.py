"""Copies of a canonical Group Chat's history on its member installations, its custodians.

A member installation's grant carries ``replicate`` unless its operator opted out
(``groups.peer.invite`` with ``replication: false``), and ``work_records`` when task evidence was
added. Once the home registers that grant, this publisher sends the room's log there, one bounded
page at a time, and each frozen task-evidence record once the history it is anchored to has been
acknowledged.
Coverage is an acknowledgement, not failover: nothing here moves authority, replays work or
answers for a lost home. When the room owner prepared a retirement for a participant, its copy
waits for the participant's confirmation, stops once Disband closes that obligation, and the
same workers then deliver the notice that retires the copy.

An installation no member route reaches (a custodian-only installation, or the previous host
after a move) is copied on the copy-only grant it gave, its custody route. Its acknowledgments
renew that grant before it runs out, and a route it refused as unauthorized is probed again every
``REAUTHORIZATION_PROBE_SECONDS``.

Two workers rotate the routes. Checkpoints keep the pending page's coordinates (not its
events), so a lost reply or a restart re-sends the same page. No transaction spans HTTP, and
an OS lock per (room, participant) keeps two processes from sending one copy concurrently.
The workers start only once a route exists, so a gateway without one runs no extra thread.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import deque
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gateway import hosted_room_links as links
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_work_records as work_records
from gateway import hosted_rooms as rooms
from gateway.hosted_room_peer import PROTOCOL_VERSION, _split_token, gateway_room_grant_secret
from gateway.hosted_rooms_common import open_sqlite, table_exists
from gateway.status import _release_file_lock, _try_acquire_file_lock
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

POLL_SECONDS = 5.0
PAGE_TIMEOUT_SECONDS = 3.0
# A copy-only route its custodian refused as unauthorized is probed again this often.
REAUTHORIZATION_PROBE_SECONDS = 60 * 60
PAGE_LIMIT = 32
WORKERS = 2
ROUTES_TABLE = "hosted_room_replication_publishers"
TARGETS_TABLE = "hosted_room_replication_targets"
# Route states that wait for a new grant or a new route instead of retrying.
_BLOCKED = {"needs_reauthorization", "replica_rejected", "invalid_ack", "source_gap", "divergent_copy"}
# What a renewed copy-only grant must keep of the grant it replaces: everything but its life.
_RENEWAL_SCOPE = ("version", "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id",
                  "target_install_id", "target_profile", "execution_policy_digest", "permissions")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _generation(link: links.StoredRoomLink, room: dict) -> str:
    record = link.as_record()
    # Health timestamps change during ordinary execution and are not route identity.
    for key in ("status", "updated_at"):
        record.pop(key)
    return _digest([record, room["authority_gateway_id"], room["authority_epoch"], room["members"]])


def _replication_hint(token: str) -> dict:
    """The grant's unverified claims when they include ``replicate``; else ``{}``.

    Only a sending hint: the participant verifies signature, reservation and revocation.
    """
    if not isinstance(token, str) or len(token) > links.MAX_GRANT_CHARS:
        return {}
    try:
        hint = json.loads(_split_token(token)[0].decode("ascii"))
    except (ValueError, UnicodeError, RecursionError):
        return {}
    if isinstance(hint, dict) and isinstance(hint.get("permissions"), list) and "replicate" in hint["permissions"]:
        return hint
    return {}


# Publisher keys of installations no member route reaches: (room_id, "custody@<install_id>"). A
# custodian-only installation has no Bot in the room; a custodian whose Bots this host has no route
# to, such as the previous host after a move, keeps its copy the same way. Their copy-only grant
# lives in the custody routes table.
_CUSTODY_PREFIX = "custody@"


def _installation(catalog_json: Any) -> str | None:
    """The installation a stored member route's catalog names, if it is readable."""
    try:
        catalog = json.loads(catalog_json)
    except (TypeError, ValueError, RecursionError):
        return None
    return catalog.get("installation_id") if isinstance(catalog, dict) else None


def _member_routes_reach_locked(conn: sqlite3.Connection, room_id: str, install_id: str) -> bool:
    """Whether a member route that carries a copy reaches this installation: that route copies it."""
    return any(_replication_hint(row["grant"]) and _installation(row["catalog_json"]) == install_id
               for row in conn.execute("SELECT grant, catalog_json FROM hosted_room_links WHERE room_id=? LIMIT ?",
                                       (room_id, links.MAX_LINKS)))


def _custody_link_locked(conn: sqlite3.Connection, key: tuple[str, str]) -> links.StoredRoomLink | None:
    from gateway import hosted_room_custody as custody
    from gateway.hosted_room_peer import GatewayRoomCatalog
    if not table_exists(conn, custody.ROUTES_TABLE):
        return None
    row = conn.execute(f"""SELECT r.*, c.role FROM {custody.ROUTES_TABLE} r JOIN {custody.CUSTODIANS_TABLE} c
        ON c.room_id=r.room_id AND c.install_id=r.install_id WHERE r.room_id=? AND r.install_id=?
        AND c.role IN ('custodian_only','custodian') AND c.state IN ('active','unsupported')""",
                       (key[0], key[1][len(_CUSTODY_PREFIX):])).fetchone()
    if row is None or (row["role"] == "custodian" and _member_routes_reach_locked(conn, key[0], row["install_id"])):
        return None
    return links.make_stored_link(
        room_id=row["room_id"], member_id=custody.CUSTODY_MEMBER_ID, target_url=row["target_url"],
        target_profile=row["target_profile"], grant=row["grant"],
        catalog=GatewayRoomCatalog.from_mapping(json.loads(row["catalog_json"])),
        cancellation_scope_id="custody", trace_id="custody")


def _link_locked(conn: sqlite3.Connection, key: tuple[str, str]) -> links.StoredRoomLink | None:
    """The member route, or the copy-only custody route, behind one publisher key, when it carries a copy."""
    if key[1].startswith(_CUSTODY_PREFIX):
        return _custody_link_locked(conn, key)
    raw = conn.execute("SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?", key).fetchone()
    return links.StoredRoomLink.from_record(raw) if raw is not None and _replication_hint(raw["grant"]) else None


def _eligible(link: links.StoredRoomLink, room: dict, local_id: str) -> bool:
    if (
        room["authority_gateway_id"] != local_id
        or link.status == "needs_reauthorization" or room.get("safety_status")
        or link.catalog.installation_id == local_id
        or PROTOCOL_VERSION not in link.catalog.protocol_versions
        or link.catalog.execution_policy.target_profile != link.target_profile
    ):
        return False
    hint = _replication_hint(link.grant)
    expected = {
        "version": PROTOCOL_VERSION, "room_id": link.room_id, "member_id": link.member_id,
        "home_install_id": local_id, "authority_gateway_id": local_id, "authority_epoch": room["authority_epoch"],
        "target_install_id": link.catalog.installation_id, "target_profile": link.target_profile,
        "execution_policy_digest": link.catalog.execution_policy.policy_digest,
    }
    # No unsigned timing is trusted here: the participant enforces the grant's horizon.
    if any(hint.get(k) != v for k, v in expected.items()):
        return False
    from gateway.hosted_room_custody import CUSTODY_MEMBER_ID
    if link.member_id == CUSTODY_MEMBER_ID:
        # A copy-only grant names no Bot in the room, and it can never run work.
        return "dispatch" not in hint.get("permissions", ())
    matching = [m for m in room["members"] if m.get("member_id") == link.member_id]
    if len(matching) != 1:
        return False
    target = matching[0].get("target", {})
    if not isinstance(target, dict) or target.get("profile") != link.target_profile:
        return False
    return target.get("kind") == "peer" and target.get("installation_id") == link.catalog.installation_id


@dataclass(frozen=True)
class _Route:
    key: tuple[str, str]
    link: links.StoredRoomLink = field(repr=False)
    room: dict = field(repr=False)
    generation: str = field(repr=False)


@dataclass(frozen=True)
class _RetirementWork:
    enrollment_id: str


_Work = tuple[str, str] | _RetirementWork


def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {ROUTES_TABLE} (
        room_id TEXT NOT NULL, member_id TEXT NOT NULL, generation TEXT NOT NULL,
        target_install_id TEXT NOT NULL, target_profile TEXT NOT NULL,
        authority_gateway_id TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
        acked_seq INTEGER NOT NULL DEFAULT 0, source_latest_seq INTEGER NOT NULL DEFAULT 0,
        pending_end INTEGER, pending_latest INTEGER, pending_name TEXT,
        status TEXT NOT NULL DEFAULT 'pending', updated_at REAL NOT NULL,
        work_record_status TEXT NOT NULL DEFAULT 'pending', PRIMARY KEY(room_id, member_id))""")
    # One copy per participant installation, whichever of its member routes carries it.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TARGETS_TABLE} (
        room_id TEXT NOT NULL, target_install_id TEXT NOT NULL, lineage TEXT NOT NULL,
        selected_member_id TEXT NOT NULL, acked_seq INTEGER NOT NULL DEFAULT 0,
        source_latest_seq INTEGER NOT NULL DEFAULT 0, pending_end INTEGER,
        pending_latest INTEGER, pending_name TEXT, status TEXT NOT NULL DEFAULT 'pending',
        updated_at REAL NOT NULL, PRIMARY KEY(room_id, target_install_id))""")


class HostedRoomReplicationPublisher:
    """Gateway-owned asynchronous copy, deliberately independent of execution."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.local_id = rooms.local_authority_gateway_id()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._enabled = False
        self._threads: list[threading.Thread] = []
        self._routes: deque[_Work] = deque()
        self._inflight: set[_Work] = set()
        self._due: dict[_Work, float] = {}
        self._retirement_delays: dict[_RetirementWork, float] = {}
        self._scan_at = 0.0
        self._error: str | None = None
        self._work_record_error: str | None = None
        self._custody_error: str | None = None

    @contextmanager
    def _transaction(self):
        conn = open_sqlite(self.db_path, timeout=0.25)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _initialize(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- lifecycle ------------------------------------------------------------------------
    def _alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self) -> None:
        """Arm the publisher; its workers start when there is something to copy."""
        with self._condition:
            if self._alive():
                return  # A timed-out stop must not spawn replacement workers.
            self._enabled, self._threads = True, []
            self._stop.clear()
        self.wakeup()

    def wakeup(self) -> None:
        """Rescan now, starting the workers on the first route that wants a copy."""
        with self._condition:
            if not self._enabled or self._stop.is_set():
                return
            self._scan_at = 0.0
            self._condition.notify_all()
            if self._alive():
                return
        try:
            wanted = self._has_work()
        except (OSError, sqlite3.Error):
            wanted = True  # Let the workers retry an unavailable store rather than go quiet.
        if wanted:
            self._start_workers()

    def _has_work(self) -> bool:
        from gateway.hosted_room_custody import ROUTES_TABLE as CUSTODY_ROUTES
        with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
            if retirement.has_open_obligations(conn):
                return True
            if table_exists(conn, CUSTODY_ROUTES) and conn.execute(f"SELECT 1 FROM {CUSTODY_ROUTES} LIMIT 1").fetchone():
                return True
            if not table_exists(conn, "hosted_room_links"):
                return False
            return any(_replication_hint(row["grant"]) for row in conn.execute(
                "SELECT grant FROM hosted_room_links LIMIT ?", (links.MAX_LINKS,)))

    def _start_workers(self) -> None:
        with self._condition:
            if self._alive() or not self._enabled or self._stop.is_set():
                return
            self._scan_at = 0.0
            self._error = None
            self._threads = [threading.Thread(
                target=self._worker, name=f"hosted-room-replication-{i}", daemon=True,
            ) for i in range(WORKERS)]
            try:
                for thread in self._threads:
                    thread.start()
            except (RuntimeError, OSError):
                self._error = "publisher_start_failed"
                self._stop.set()
                self._threads = [thread for thread in self._threads if thread.ident is not None]
                self._condition.notify_all()

    def stop(self, *, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        self._stop.set()
        with self._condition:
            self._enabled = False
            self._condition.notify_all()
            threads = tuple(self._threads)
        for thread in threads:
            if thread.ident is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
        return not any(t.is_alive() for t in threads)

    # -- scheduling -----------------------------------------------------------------------
    def _scan(self, now: float) -> None:
        from gateway import hosted_room_custody as custody
        with self._transaction() as conn:
            custody.initialize_locked(conn)
            has_retirement = table_exists(conn, retirement.HOME_TABLE)
            keys: list[_Work] = []
            reached: set[tuple[str, str | None]] = set()
            for row in conn.execute("""SELECT room_id, member_id, grant, catalog_json FROM hosted_room_links
                    ORDER BY room_id, member_id LIMIT ?""", (links.MAX_LINKS,)):
                if _replication_hint(row["grant"]):
                    keys.append((row["room_id"], row["member_id"]))
                    reached.add((row["room_id"], _installation(row["catalog_json"])))
            # A custody route copies a custodian-only installation, and a custodian no member route reaches.
            keys.extend((row[0], _CUSTODY_PREFIX + row[1]) for row in conn.execute(
                f"""SELECT r.room_id, r.install_id, c.role FROM {custody.ROUTES_TABLE} r
                    JOIN {custody.CUSTODIANS_TABLE} c ON c.room_id=r.room_id AND c.install_id=r.install_id
                    WHERE c.role IN ('custodian_only','custodian') AND c.state IN ('active','unsupported')
                    ORDER BY r.room_id, r.install_id LIMIT ?""", (links.MAX_LINKS,))
                if row[2] == "custodian_only" or (row[0], row[1]) not in reached)
            conn.execute(f"""UPDATE {ROUTES_TABLE} SET status='stopped_route_removed'
                WHERE status!='stopped_route_removed' AND NOT EXISTS (
                    SELECT 1 FROM hosted_room_links AS l
                    WHERE l.room_id={ROUTES_TABLE}.room_id AND l.member_id={ROUTES_TABLE}.member_id)
                  AND NOT EXISTS (
                    SELECT 1 FROM {custody.ROUTES_TABLE} AS c
                    WHERE c.room_id={ROUTES_TABLE}.room_id AND '{_CUSTODY_PREFIX}' || c.install_id={ROUTES_TABLE}.member_id)""")
            conn.execute(f"""UPDATE {ROUTES_TABLE} SET source_latest_seq=MAX(source_latest_seq, COALESCE(
                (SELECT next_seq-1 FROM hosted_rooms WHERE room_id={ROUTES_TABLE}.room_id), source_latest_seq))
                WHERE status='stopped_route_removed'""")
            # Keep a bounded diagnostic tail after revocation, not an ever-growing outbox.
            conn.execute(f"""DELETE FROM {ROUTES_TABLE} WHERE rowid IN (
                SELECT rowid FROM {ROUTES_TABLE} WHERE status='stopped_route_removed'
                ORDER BY updated_at DESC LIMIT -1 OFFSET ?)""", (links.MAX_LINKS,))
            conn.execute(f"""DELETE FROM {TARGETS_TABLE} WHERE NOT EXISTS (
                SELECT 1 FROM {ROUTES_TABLE} AS r WHERE r.room_id={TARGETS_TABLE}.room_id
                    AND r.target_install_id={TARGETS_TABLE}.target_install_id)""")
        if has_retirement:
            keys.extend(_RetirementWork(enrollment_id) for enrollment_id in retirement.pending_notice_ids(
                self.db_path, local_gateway_id=self.local_id))
        current = set(keys)
        self._routes = deque([k for k in self._routes if k in current])
        self._routes.extend(k for k in keys if k not in self._routes)
        self._due = {k: due for k, due in self._due.items() if k in current}
        self._retirement_delays = {k: delay for k, delay in self._retirement_delays.items() if k in current}
        self._maintain_custody()
        self._scan_at = now + POLL_SECONDS

    def _maintain_custody(self) -> None:
        """Record each hosted room's custodians in its log when they changed."""
        from gateway import hosted_room_custody as custody
        from gateway.hosted_room_identity import local_public_key
        from gateway.hosted_room_peer import local_room_link_endpoint
        with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
            if not table_exists(conn, custody.CUSTODIANS_TABLE):
                return
            room_ids = [str(row[0]) for row in conn.execute(
                f"SELECT DISTINCT room_id FROM {custody.CUSTODIANS_TABLE} ORDER BY room_id")]
        endpoint = local_room_link_endpoint()
        url, key, failed = endpoint.get("url") if endpoint.get("available") else None, local_public_key(), None
        name, owner_name = custody.local_names()
        for room_id in room_ids:
            try:
                custody.maintain_configuration(
                    self.db_path, room_id=room_id, local_gateway_id=self.local_id, public_key=key, endpoint=url,
                    name=name, owner_name=owner_name)
            except (rooms.HostedRoomError, sqlite3.Error):
                failed = "custody_configuration_unavailable"  # one room never stalls the others' copies
        self._custody_error = failed

    def wait_protected(self, room_id: str, seq: int, timeout: float) -> bool:
        """Whether an eligible successor holds ``seq`` of this hosted room within ``timeout``."""
        from gateway.hosted_room_custody import wait_protected
        return wait_protected(self.db_path, room_id, seq, timeout)

    def _take(self) -> _Work | None:
        with self._condition:
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= self._scan_at:
                    self._scan(now)
                for _ in range(len(self._routes)):
                    key = self._routes.popleft()
                    self._routes.append(key)
                    if key not in self._inflight and self._due.get(key, 0) <= now:
                        self._inflight.add(key)
                        return key
                wake_at = min([self._scan_at, *(
                    self._due.get(k, now) for k in self._routes if k not in self._inflight
                )])
                self._condition.wait(timeout=min(POLL_SECONDS, max(0.01, wake_at - now)))
        return None

    def _worker(self) -> None:
        while not self._stop.is_set():
            key, more = None, False
            try:
                key = self._take()
                if key is None:
                    return
                more = self._publish_retirement(key) if isinstance(key, _RetirementWork) else self._publish_one(key)
                self._error = None
            except Exception:
                # Never include transport exceptions, grant material, URLs or raw rows.
                self._error = "publisher_local_error"
                self._stop.wait(POLL_SECONDS)
            finally:
                if key is not None:
                    with self._condition:
                        self._inflight.discard(key)
                        delay = self._retirement_delays.get(key, POLL_SECONDS) if isinstance(
                            key, _RetirementWork) else POLL_SECONDS
                        self._due[key] = time.monotonic() + (0 if more else delay)
                        self._condition.notify_all()

    def _publish_retirement(self, work: _RetirementWork) -> bool:
        """Deliver one closed obligation's notice until the participant acknowledges it."""
        route = retirement.notice_route(self.db_path, enrollment_id=work.enrollment_id, local_gateway_id=self.local_id)
        if route is None or self._stop.is_set():
            return False
        path = self._lock_path(route["room_id"], route["target_install_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            if not _try_acquire_file_lock(handle):
                return False
            try:
                notice = retirement.materialize_notice(
                    self.db_path, enrollment_id=work.enrollment_id, local_gateway_id=self.local_id,
                    secret_loader=gateway_room_grant_secret)
                if self._stop.is_set():
                    return False
                if not notice.proof_grant:
                    raise retirement.RetirementProofUnavailable('participant installation proof is unavailable')
                client = PeerRunsHTTPClient(base_url=notice.endpoint, api_key="", timeout_seconds=PAGE_TIMEOUT_SECONDS,
                                            proof_install_id=notice.target_install_id)
                retirement.acknowledge_notice(self.db_path, notice=notice, response=client.retire_replica(notice))
                self._retirement_delays.pop(work, None)
            except (retirement.RetirementError, PeerRunsHTTPError, OSError) as exc:
                code = ("retirement_key_unavailable" if isinstance(exc, retirement.RetirementKeyUnavailable)
                        else "retirement_target_proof_unavailable" if isinstance(exc, retirement.RetirementProofUnavailable)
                        else "retirement_delivery_unconfirmed")
                retirement.record_delivery_error(self.db_path, enrollment_id=work.enrollment_id, code=code)
                # Back off, up to five minutes; an obligation is never dropped.
                self._retirement_delays[work] = min(300.0, self._retirement_delays.get(work, POLL_SECONDS / 2) * 2)
            finally:
                _release_file_lock(handle)
        return False

    # -- one route ------------------------------------------------------------------------
    def _load_route(self, key: tuple[str, str]) -> _Route | None:
        try:
            with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
                link = _link_locked(conn, key)
            if link is None:
                return None
            room = rooms.room_state(self.db_path, room_id=key[0], include_disbanded=True)
            if _eligible(link, room, self.local_id):
                return _Route(key, link, room, _generation(link, room))
        except (rooms.HostedRoomError, ValueError):
            pass
        with self._transaction() as conn:
            conn.execute(f"""UPDATE {ROUTES_TABLE} SET status='stopped_ineligible'
                WHERE room_id=? AND member_id=? AND status NOT IN ({','.join('?' for _ in _BLOCKED)})""",
                         (*key, *sorted(_BLOCKED)))
        return None

    def _current(self, conn, route: _Route) -> bool:
        """Re-read the route and room inside a writer: a change since loading wins."""
        room = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (route.key[0],)).fetchone()
        quarantine = conn.execute(
            "SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (route.key[0],),
        ).fetchone()
        if room is None or quarantine is not None:
            return False
        try:
            link = _link_locked(conn, route.key)
            if link is None:
                return False
            state = {**dict(room), "members": json.loads(room["members_json"])}
            return _eligible(link, state, self.local_id) and _generation(link, state) == route.generation
        except (ValueError, rooms.HostedRoomError):
            return False

    def _checkpoint(self, route: _Route) -> dict | None:
        with self._transaction() as conn:
            if not self._current(conn, route):
                return None
            conn.execute(f"""INSERT INTO {ROUTES_TABLE} (
                room_id, member_id, generation, target_install_id, target_profile,
                authority_gateway_id, authority_epoch, updated_at) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(room_id,member_id) DO UPDATE SET generation=excluded.generation,
                target_install_id=excluded.target_install_id, target_profile=excluded.target_profile,
                authority_gateway_id=excluded.authority_gateway_id, authority_epoch=excluded.authority_epoch,
                acked_seq=0, source_latest_seq=0, pending_end=NULL, pending_latest=NULL,
                pending_name=NULL, status='pending', work_record_status='pending', updated_at=excluded.updated_at
                WHERE generation!=excluded.generation""", (
                    *route.key, route.generation, route.link.catalog.installation_id, route.link.target_profile,
                    route.room["authority_gateway_id"], route.room["authority_epoch"], time.time(),
                ))
            return dict(conn.execute(f"SELECT * FROM {ROUTES_TABLE} WHERE room_id=? AND member_id=?", route.key).fetchone())

    def _save(self, route: _Route, checkpoint: dict, **values) -> bool:
        """Advance the participant's checkpoint only from the exact state this attempt read."""
        with self._transaction() as conn:
            if not self._current(conn, route):
                return False
            values["updated_at"] = time.time()
            saved = conn.execute(
                f"UPDATE {TARGETS_TABLE} SET " + ",".join(f"{k}=?" for k in values)
                + " WHERE room_id=? AND target_install_id=? AND lineage=? AND acked_seq=?"
                + " AND pending_end IS ? AND pending_latest IS ?",
                (*values.values(), route.key[0], route.link.catalog.installation_id,
                 checkpoint["lineage"], checkpoint["acked_seq"],
                 checkpoint["pending_end"], checkpoint["pending_latest"]),
            ).rowcount == 1
            if saved:
                conn.execute(
                    f"UPDATE {ROUTES_TABLE} SET " + ",".join(f"{k}=?" for k in values)
                    + " WHERE room_id=? AND member_id=? AND generation=?",
                    (*values.values(), *route.key, route.generation),
                )
            return saved

    def _lock_path(self, room_id: str, target_install_id: str) -> Path:
        return self.db_path.parent / "room_replication_locks" / (_digest([room_id, target_install_id]) + ".lock")

    def _select_route(self, initial: _Route) -> _Route | None:
        """Pick a live authorized member route to this participant, under its OS lock.

        A route that just failed yields to an alternate. Work-record delivery and history share
        the routes fairly: an anchored record goes first only if some route can still deliver
        it, so a refused record never hides an otherwise healthy history path.
        """
        target_install_id = initial.link.catalog.installation_id
        if initial.key[1].startswith(_CUSTODY_PREFIX):
            # An installation no member route reaches has exactly one route: its copy-only grant.
            checkpoint = self._checkpoint(initial)
            if checkpoint is not None and checkpoint["status"] == "needs_reauthorization":
                checkpoint = self._probe_reauthorization(initial, checkpoint)
            return initial if checkpoint is not None and checkpoint["status"] not in _BLOCKED else None
        with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
            candidates = conn.execute(
                "SELECT * FROM hosted_room_links WHERE room_id=? ORDER BY member_id LIMIT ?",
                (initial.key[0], links.MAX_LINKS),
            ).fetchall()
            target = conn.execute(f"SELECT * FROM {TARGETS_TABLE} WHERE room_id=? AND target_install_id=?",
                                  (initial.key[0], target_install_id)).fetchone() if table_exists(
                                      conn, TARGETS_TABLE) else None
            work_ready = (target is not None and (target["acked_seq"] > 0 or target["status"] == "acked")
                          and work_records.pending_delivery_is_anchored_locked(
                              conn, room_id=initial.key[0], target_install_id=target_install_id,
                              through_seq=target["acked_seq"]))
        selected = []
        room = rooms.room_state(self.db_path, room_id=initial.key[0], include_disbanded=True)
        for raw in candidates:
            if not _replication_hint(raw["grant"]):
                continue
            try:
                link = links.StoredRoomLink.from_record(raw)
            except ValueError:
                continue
            if link.catalog.installation_id != target_install_id or not _eligible(link, room, self.local_id):
                continue
            route = _Route((link.room_id, link.member_id), link, room, _generation(link, room))
            checkpoint = self._checkpoint(route)
            if checkpoint is None or checkpoint["status"] in _BLOCKED:
                continue
            opted_in = work_records.PERMISSION in _replication_hint(link.grant).get("permissions", ())
            refused = checkpoint["work_record_status"] in work_records.BLOCKED_DELIVERY_STATUSES
            work_rank = 2 if opted_in and refused else 0 if opted_in else 1
            work_unavailable = checkpoint["work_record_status"] == "unavailable"
            # Transient work failures rotate between routes instead of pinning the asked-for one.
            turn_rank = route.key != initial.key if work_rank == 0 and work_unavailable else 0
            selected.append((checkpoint["status"] == "unavailable", work_rank, work_unavailable, turn_rank,
                             route.key, route))
        can_deliver_work = work_ready and any(item[1] == 0 for item in selected)
        history_needed = not can_deliver_work and (
            target is None or target["pending_end"] is not None or target["acked_seq"] < room["latest_seq"]
            or target["status"] != "acked")

        def rank(item):
            unavailable, work_rank, work_unavailable, turn_rank, key, _ = item
            if history_needed:
                return unavailable, work_rank, work_unavailable, turn_rank, key
            return work_rank, work_unavailable, turn_rank, unavailable, key

        return min(selected, key=rank)[5] if selected else None

    def _target_checkpoint(self, route: _Route) -> dict | None:
        lineage = _digest([route.room["authority_gateway_id"], route.room["authority_epoch"], route.room["members"]])
        key = (route.key[0], route.link.catalog.installation_id)
        with self._transaction() as conn:
            if not self._current(conn, route):
                return None
            conn.execute(f"""INSERT INTO {TARGETS_TABLE}
                (room_id,target_install_id,lineage,selected_member_id,updated_at) VALUES (?,?,?,?,?)
                ON CONFLICT(room_id,target_install_id) DO UPDATE SET lineage=excluded.lineage,
                acked_seq=0, source_latest_seq=0, pending_end=NULL, pending_latest=NULL,
                pending_name=NULL, status='pending', updated_at=excluded.updated_at
                WHERE lineage!=excluded.lineage""", (*key, lineage, route.key[1], time.time()))
            conn.execute(f"UPDATE {TARGETS_TABLE} SET selected_member_id=? WHERE room_id=? AND target_install_id=?",
                         (route.key[1], *key))
            return dict(conn.execute(
                f"SELECT * FROM {TARGETS_TABLE} WHERE room_id=? AND target_install_id=?", key,
            ).fetchone())

    def _publish_one(self, key: tuple[str, str]) -> bool:
        route = self._load_route(key)
        if route is None or self._stop.is_set():
            return False
        path = self._lock_path(route.key[0], route.link.catalog.installation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never unlink lock files: replacing a locked inode permits two owners.
        with path.open("a+", encoding="utf-8") as handle:
            if not _try_acquire_file_lock(handle):
                return False
            try:
                selected = self._select_route(route)
                return self._publish_locked(selected) if selected is not None else False
            finally:
                _release_file_lock(handle)

    def _publish_locked(self, route: _Route) -> bool:
        key = route.key
        checkpoint = self._target_checkpoint(route)
        if checkpoint is None:
            return False
        client = PeerRunsHTTPClient(base_url=route.link.target_url, api_key="", timeout_seconds=PAGE_TIMEOUT_SECONDS,
                                    proof_install_id=route.link.catalog.installation_id)
        enrollment = retirement.current_home_enrollment(
            self.db_path, room_id=key[0], target_install_id=route.link.catalog.installation_id)
        if enrollment is not None:
            # A prepared retirement is confirmed from the participant's probe before more copying;
            # once Disband closed it, the copy waits only for its retirement notice.
            if enrollment["state"] == "prepared":
                try:
                    proof = client.probe(grant=route.link.grant).get("retirement_enrollment")
                except PeerRunsHTTPError as exc:
                    return self._http_failure(route, checkpoint, exc)
                if not retirement.confirm_home_enrollment(
                        self.db_path, enrollment_id=enrollment["enrollment_id"], proof=proof):
                    return False
            elif enrollment["state"] != "enrolled":
                return False
        # Each turn can deliver one anchored record and one history page, so neither waits for
        # the conversation to go quiet and neither blocks the other on transport loss.
        self._publish_work_records(route, checkpoint, client)
        cursor, pending = checkpoint["acked_seq"], checkpoint["pending_end"]
        if pending is None and cursor >= route.room["latest_seq"] and checkpoint["status"] == "acked":
            return False
        limit = PAGE_LIMIT if pending is None else max(1, pending - cursor)
        page = rooms.read_events(self.db_path, room_id=key[0], since_seq=cursor, limit=limit, include_disbanded=True)
        expected_authority = {"gateway_id": route.room["authority_gateway_id"], "epoch": route.room["authority_epoch"]}
        if page["authority"] != expected_authority:
            return False
        name = route.room["name"]
        if pending is not None:
            # New source events cannot change an in-flight retry's coverage claim.
            if pending == cursor:
                page.update(events=[], cursor=cursor)
            if page["cursor"] != pending:
                self._save(route, checkpoint, status="source_gap")
                return False
            page.update(latest_seq=checkpoint["pending_latest"], has_more=pending < checkpoint["pending_latest"])
            name = checkpoint["pending_name"]
        else:
            values = dict(pending_end=page["cursor"], pending_latest=page["latest_seq"], pending_name=name,
                          source_latest_seq=page["latest_seq"], status="pending")
            if not self._save(route, checkpoint, **values):
                return False
            checkpoint.update(values)
        if self._stop.is_set():
            return False
        from gateway import hosted_room_custody as custody
        with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
            report = custody.report_locked(conn, key[0], route.link.catalog.installation_id)
        try:
            reply = client.replicate_page(
                grant=route.link.grant, room_id=key[0], room_name=name, members=route.room["members"], page=page,
                custody=report)
        except PeerRunsHTTPError as exc:
            return self._http_failure(route, checkpoint, exc)
        if (
            not isinstance(reply, dict) or reply.get("room_id") != key[0]
            or reply.get("authority") != expected_authority
            or type(reply.get("stored_seq")) is not int or reply["stored_seq"] < page["cursor"]
            or reply["stored_seq"] > page["latest_seq"]
        ):
            self._save(route, checkpoint, status="invalid_ack")
            return False
        install_id = route.link.catalog.installation_id
        reported = reply["custody"] if isinstance(reply.get("custody"), dict) else {}
        allowed = reported.get("allowed")
        if isinstance(allowed, bool):
            # The operator's current consent to continue the group, from an authenticated acknowledgment.
            custody.record_allowed(self.db_path, room_id=key[0], install_id=install_id, allowed=allowed)
        if "watermark" not in reply:
            # An older Hermes keeps the pages but no custody watermark: it never counts as holding history.
            custody.mark_unsupported(self.db_path, room_id=key[0], install_id=install_id)
        else:
            # The custodian's durable watermark counts only if it is a prefix of this log.
            try:
                acknowledged = custody.record_acknowledgment(
                    self.db_path, room_id=key[0], install_id=install_id, watermark=reply["watermark"])
            except custody.CustodyError:
                acknowledged = "invalid"
            if acknowledged in {"divergent", "invalid"}:
                self._save(route, checkpoint, status="divergent_copy" if acknowledged == "divergent" else "invalid_ack")
                return False
        saved = self._save(
            route, checkpoint, acked_seq=page["cursor"], source_latest_seq=page["latest_seq"],
            pending_end=None, pending_latest=None, pending_name=None,
            status="pending" if page["has_more"] else "acked",
        )
        if key[1].startswith(_CUSTODY_PREFIX) and isinstance(reported.get("renewed_grant"), str):
            self._keep_renewed_grant(route, reported["renewed_grant"])  # the next push uses it
        return saved and page["has_more"]

    def _keep_renewed_grant(self, route: _Route, renewed: str) -> None:
        """Keep the fresh copy-only grant a custodian returned as its old one nears its horizon.

        Only its life may move: the same room, host, epoch, installation, profile, policy and
        permissions, and a later horizon. The custodian checks its signature when the next push uses
        it; as a new route generation, it also lifts a block the old grant caused.
        """
        from gateway import hosted_room_custody as custody
        from gateway.hosted_room_peer import HostedRoomGrantError, unverified_room_grant_claims
        try:
            old, new = unverified_room_grant_claims(route.link.grant), unverified_room_grant_claims(renewed)
            later = float(new["status_expires_at"]) > float(old.get("status_expires_at", old["expires_at"]))
        except (HostedRoomGrantError, KeyError, TypeError, ValueError):
            return
        if len(renewed) > links.MAX_GRANT_CHARS or not later or any(
                new.get(field) != old.get(field) for field in _RENEWAL_SCOPE):
            return
        custody.save_custody_route(
            self.db_path, room_id=route.key[0], install_id=route.link.catalog.installation_id,
            target_url=route.link.target_url, target_profile=route.link.target_profile, grant=renewed,
            catalog=route.link.catalog.as_mapping())

    def _probe_reauthorization(self, route: _Route, checkpoint: dict) -> dict | None:
        """Probe a copy-only route its custodian refused as unauthorized, once per
        ``REAUTHORIZATION_PROBE_SECONDS``: copying resumes as soon as its grant is accepted again."""
        if time.time() - float(checkpoint["updated_at"]) < REAUTHORIZATION_PROBE_SECONDS or self._stop.is_set():
            return checkpoint
        client = PeerRunsHTTPClient(base_url=route.link.target_url, api_key="", timeout_seconds=PAGE_TIMEOUT_SECONDS,
                                    proof_install_id=route.link.catalog.installation_id)
        try:
            client.probe(grant=route.link.grant)
            status = "pending"
        except PeerRunsHTTPError:
            status = "needs_reauthorization"  # probed again a period later
        with self._transaction() as conn:
            if not self._current(conn, route):
                return None
            conn.execute(f"UPDATE {ROUTES_TABLE} SET status=?, updated_at=? WHERE room_id=? AND member_id=? "
                         "AND generation=?", (status, time.time(), *route.key, route.generation))
            return dict(conn.execute(f"SELECT * FROM {ROUTES_TABLE} WHERE room_id=? AND member_id=?",
                                     route.key).fetchone())

    def _publish_work_records(self, route: _Route, checkpoint: dict, client: PeerRunsHTTPClient) -> None:
        """Send this participant's frozen record once its history anchor is acknowledged."""
        if work_records.PERMISSION not in _replication_hint(route.link.grant).get("permissions", ()):
            return
        installation = route.link.catalog.installation_id
        try:
            with self._transaction() as conn:
                if not self._current(conn, route):
                    return
                state = conn.execute(f"SELECT work_record_status FROM {ROUTES_TABLE} WHERE room_id=? AND member_id=? "
                                     "AND generation=?", (*route.key, route.generation)).fetchone()
                if state is None or state[0] in work_records.BLOCKED_DELIVERY_STATUSES:
                    return
                try:
                    record = work_records.prepare_delivery_locked(
                        conn, room_id=route.key[0], target_install_id=installation, route_generation=route.generation,
                        local_gateway_id=self.local_id, through_seq=checkpoint["acked_seq"])
                except work_records.InvalidStoredWorkRecord:
                    # Commit the invalid disposition just discovered; never send those bytes.
                    conn.execute(f"UPDATE {ROUTES_TABLE} SET work_record_status='invalid_work_evidence' "
                                 "WHERE room_id=? AND member_id=? AND generation=?", (*route.key, route.generation))
                    return
                if record is None and state[0] in {"source_prefix_expired", "work_record_capture_unavailable"}:
                    # An unchanged capture after a failure needs no transport, but only an
                    # acknowledged current record clears that failure.
                    conn.execute(f"""UPDATE {ROUTES_TABLE} SET work_record_status='acked'
                        WHERE room_id=? AND member_id=? AND generation=? AND EXISTS (
                            SELECT 1 FROM {work_records.PENDING_TABLE} WHERE room_id=? AND target_install_id=?
                            AND status='acked' AND disposition='current')""",
                                 (*route.key, route.generation, route.key[0], installation))
            history_confirmed = checkpoint["acked_seq"] > 0 or checkpoint["status"] == "acked"
            if record is None or not history_confirmed or self._stop.is_set():
                return
            ack, status, prefix_gap = None, None, False
            try:
                reply = client.replicate_work_records(grant=route.link.grant, record=record)
                tag = "hermes.room_member.work_records"
                ack = dict(reply) if isinstance(reply, dict) else None
                if ack is not None and ack.pop("object", tag) != tag:
                    ack = None
            except PeerRunsHTTPError as exc:
                status = "unavailable"
                prefix_gap = exc.status_code == 409 and exc.error_code == "work_records_prefix"
                if exc.status_code in {401, 403}:
                    status = "needs_reauthorization"
                elif (exc.status_code is not None and 400 <= exc.status_code < 500
                      and exc.status_code not in {408, 429} and not prefix_gap):
                    status = "rejected"
            with self._transaction() as conn:
                if self._current(conn, route):
                    scope = dict(room_id=route.key[0], target_install_id=installation,
                                 route_generation=route.generation, record=record)
                    if status is None and ack is not None and work_records.acknowledge_locked(conn, **scope, ack=ack):
                        status, saved = "acked", True
                    else:
                        # Only the exact passive ACK completes a record; anything else is refused.
                        status = status or "invalid_ack"
                        saved = work_records.delivery_status_locked(conn, **scope, status=status)
                    if saved:
                        conn.execute(f"UPDATE {ROUTES_TABLE} SET work_record_status=?,updated_at=? "
                                     "WHERE room_id=? AND member_id=? AND generation=?",
                                     (status, time.time(), *route.key, route.generation))
            if prefix_gap:
                # The participant lacks the history this record is anchored to: copy it again.
                reset = dict(acked_seq=0, pending_end=None, pending_latest=None, pending_name=None, status="replica_gap")
                if self._save(route, checkpoint, **reset):
                    checkpoint.update(reset)
            self._work_record_error = None
        except work_records.WorkRecordPrefixError:
            self._record_work_failure(route, "source_prefix_expired")
        except (work_records.WorkRecordError, sqlite3.Error, OSError):
            self._record_work_failure(route, "work_record_capture_unavailable")

    def _record_work_failure(self, route: _Route, status: str) -> None:
        # Persist on the failing route: another worker's success or a restart must not erase it.
        try:
            with self._transaction() as conn:
                if self._current(conn, route):
                    conn.execute(f"UPDATE {ROUTES_TABLE} SET work_record_status=? WHERE room_id=? AND member_id=? "
                                 "AND generation=?", (status, *route.key, route.generation))
        except (sqlite3.Error, OSError):
            self._work_record_error = status

    def _http_failure(self, route: _Route, checkpoint: dict, exc: PeerRunsHTTPError) -> bool:
        if exc.status_code == 409 and exc.error_code == "room_replica_gap":
            # The participant's copy is behind this checkpoint (for example it was restored).
            self._save(route, checkpoint, acked_seq=0, pending_end=None, pending_latest=None,
                       pending_name=None, status="replica_gap")
            return False
        status = "unavailable"
        if exc.status_code in {401, 403}:
            status = "needs_reauthorization"
        elif exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code not in {408, 429}:
            status = "replica_rejected"
        self._save(route, checkpoint, status=status)
        return False

    def status(self, room_id: str | None = None) -> dict:
        error = self._error
        routes, deliveries, retirements = [], [], []
        try:
            with closing(open_sqlite(self.db_path, timeout=0.25)) as conn:
                deliveries = work_records.delivery_summaries_locked(conn, room_id)
                has_retirement = table_exists(conn, retirement.HOME_TABLE)
                if table_exists(conn, ROUTES_TABLE):
                    rows = conn.execute(f"""SELECT r.room_id, member_id, r.target_install_id, target_profile,
                        authority_gateway_id, authority_epoch, r.acked_seq, r.source_latest_seq, r.status,
                        r.updated_at, r.work_record_status, t.selected_member_id, t.acked_seq AS target_acked_seq,
                        t.status AS target_status, t.source_latest_seq AS target_source_latest_seq
                        FROM {ROUTES_TABLE} AS r LEFT JOIN {TARGETS_TABLE} AS t
                        ON r.room_id=t.room_id AND r.target_install_id=t.target_install_id
                        WHERE (? IS NULL OR r.room_id=?) ORDER BY r.room_id, member_id""", (room_id, room_id))
                    for row in rows:
                        role = "selected" if row["selected_member_id"] == row["member_id"] else "alternate"
                        if row["status"] in _BLOCKED or row["status"].startswith("stopped"):
                            role = "blocked"
                        routes.append({**dict(row), "delivery_unconfirmed": row["status"] != "acked", "role": role})
            if has_retirement:
                retirements = retirement.home_status(self.db_path, room_id=room_id)
        except (OSError, sqlite3.Error):
            routes, deliveries, retirements, error = None, None, None, "publisher_status_unavailable"
        custody = None
        if room_id is not None:
            from gateway.hosted_room_custody import custody_status
            try:
                custody = custody_status(self.db_path, room_id)
            except (rooms.HostedRoomError, OSError, sqlite3.Error):
                custody = None
        capture_errors = sorted({
            row["work_record_status"] for row in routes or []
            if row["work_record_status"] in {"source_prefix_expired", "work_record_capture_unavailable",
                                             "invalid_work_evidence"}
            and not row["status"].startswith("stopped")})
        return {
            "running": self._alive(), "stopping": self._stop.is_set(),
            "workers": sum(t.is_alive() for t in self._threads), "routes": routes, "work_records": deliveries,
            "retirements": retirements,
            "work_records_error": capture_errors[0] if capture_errors else self._work_record_error,
            "error": error, "mode": "passive_async_copy", "source_loss_safe": False,
            "custody": custody, "custody_error": self._custody_error,
        }
