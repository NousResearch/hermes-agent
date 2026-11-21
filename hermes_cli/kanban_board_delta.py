"""Incremental ``GET /board`` for the Kanban dashboard plugin: built cards kept between requests, an
etag, and deltas.

A board with a long history is thousands of cards, nearly all of them done and never changing again.
The endpoint used to rebuild every card (diagnostics over all of them) and send the whole snapshot on
every poll. This module keeps each card's built JSON per board database, rebuilds a card only when its
inputs changed, and answers a client that names the state it holds with ``unchanged`` or a delta.

When a card is rebuilt
    Each card has a signature: its whole ``tasks`` row, its comment count, its link counts, progress
    and direct neighbours (id, title, status), a rollup of its runs, its newest event id, and the
    diagnostics config. The row is in it because some writers (a claim extension, a failure-counter
    reset, a workspace path) change a card without leaving an event, so the event id alone cannot be
    the key. A card whose status is not done/archived, or that carries a diagnostic, is also
    re-evaluated every ``VOLATILE_SECONDS``, because its diagnostics read the clock (stranded in
    ready, running too long, a respawn loop leaving its window). A done card with no diagnostic cannot
    grow one by waiting: every rule that could fire on it later is exempt for done/archived cards or
    windowed. Every card is also rebuilt once its build is ``ENTRY_TTL_SECONDS`` old, a few per
    request, so a change the signature misses can live at most that long.

Clock drift is not a change
    A card re-evaluated only because the clock asked, from inputs that did not move, is a change only
    when its task fields moved or what its diagnostics say did: their kinds, severities, counts and
    actions, which is what a board card shows (the drawer reads the full diagnostics from
    ``GET /tasks/<id>``). Otherwise a "Ready for 49m" title would put every stranded card in every
    delta. The cached bytes are always the newest, so a full snapshot always carries the current text;
    a client holding snapshot + deltas holds diagnostic clock text at most as old as its last full read.

Wire contract (query parameters, because the desktop's plugin REST call hands back the parsed body only)
    no ``since``                  the full snapshot: the shape it always had, plus ``etag``
    ``etag`` == the current one   ``{"delta": true, "unchanged": true, "latest_event_id", "now", "etag"}``
    ``since=E`` (and ``etag=T``)  ``{"delta": true, "since", "latest_event_id", "now", "etag", "cards",
                                  "column_order": [{"name", "ids"}], "removed", "tenants", "assignees"}``:
                                  every column's ids in order, plus the cards that changed since T (when
                                  T comes from this server process) or that had an event after E (without
                                  a usable T, also every live-evaluated card and the neighbours of the
                                  changed ones)
    a ``since`` that is not an int, is negative, or is past the newest event: the full snapshot.
    Merge: held[id] = card for each of ``cards``; drop ``removed``; rebuild every column from
    ``column_order``. An id in ``column_order`` the client does not hold means it lost sync and must read
    the full snapshot. ``age`` is a function of the response's ``now``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from os import PathLike
from typing import Any, Callable, Iterator, NamedTuple, Optional

from hermes_cli import kanban_db
from hermes_cli import kanban_diagnostics as kd

VOLATILE_SECONDS = 30
ENTRY_TTL_SECONDS = 600
REFRESH_PER_CALL = 100
DIAGNOSTICS_CHUNK = 400
SUMMARIES_CHUNK = 2000
TOMBSTONES = 4096
TERMINAL = frozenset({"done", "archived"})
_NO_LINKS = (0, 0, 0, 0, (), ())
_NULL_AGE = b'{"created_age_seconds":null,"started_age_seconds":null,"time_to_complete_seconds":null}'


class BoardReply(NamedTuple):
    body: bytes
    etag: str


class CardHooks(NamedTuple):
    """The plugin's own card pieces. ``diagnostics(conn, task_ids, diag_config=...)`` returns
    ``{task_id: [diagnostic, ...]}``; ``build(task, summary, run_started_at, link_counts, comment_count,
    progress, diagnostics)`` returns the card dict exactly as a full snapshot lists it."""

    diagnostics: Callable[..., dict]
    build: Callable[..., dict]


def json_bytes(obj: Any) -> bytes:
    """The bytes FastAPI's JSONResponse writes for ``obj``."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def parse_since(raw: Optional[str]) -> Optional[int]:
    """The ``since`` query parameter as a non-negative int, else None (answered with a full snapshot)."""
    try:
        value = int(raw) if raw is not None and str(raw).strip() != "" else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and value >= 0 else None


def board_rows(conn: sqlite3.Connection) -> list:
    """Every ``tasks`` row in the board's dispatch order (``list_tasks``' default). One read, one place
    to intercept."""
    return conn.execute("SELECT * FROM tasks ORDER BY priority DESC, created_at ASC").fetchall()


def card_disc(diags: Optional[list[dict]]) -> tuple:
    """What a card's diagnostics say that does not depend on the clock: per diagnostic its kind,
    severity, count and actions. A title that counts minutes, ``data.age_seconds`` and a ``last_seen_at``
    of "now" are the clock talking; a change in these four is news to the client."""
    return tuple((d.get("kind"), d.get("severity"), d.get("count", 1), json_bytes(d.get("actions") or []))
                 for d in (diags or ()))


def _chunked(ids: list, size: int) -> Iterator[list]:
    for i in range(0, len(ids), size):
        yield ids[i:i + size]


def _card_parts(card: dict, task: kanban_db.Task) -> tuple[bytes, bytes, Optional[int], bool]:
    """``(head, tail, time_to_complete, age_is_null)``: the card's JSON is ``head + ,"age":<now> + tail``,
    so the age block is re-rendered for each response's ``now`` without rebuilding the card."""
    keys = list(card)
    cut = keys.index("age")
    age = card["age"]
    head = json_bytes({k: card[k] for k in keys[:cut]})[:-1]
    tail = json_bytes({k: card[k] for k in keys[cut + 1:]})[1:]
    # The plugin falls back to an all-null age when ``task_age`` raises; keep that exactly.
    age_null = any((got is None) != (stamp is None) for got, stamp in (
        (age.get("created_age_seconds"), task.created_at), (age.get("started_age_seconds"), task.started_at),
        (age.get("time_to_complete_seconds"), task.completed_at)))
    return head, (b"," + tail if tail != b"}" else tail), age.get("time_to_complete_seconds"), age_null


class _Card:
    __slots__ = ("age_null", "built_at", "changed_at", "completed_at", "created_at", "disc", "has_diag", "head",
                 "id", "rev", "sig", "started_at", "status", "tail", "ttc")

    def age_json(self, now: int) -> bytes:
        if self.age_null:
            return _NULL_AGE
        cr, st, ttc = self.created_at, self.started_at, self.ttc
        return b'{"created_age_seconds":%s,"started_age_seconds":%s,"time_to_complete_seconds":%s}' % (
            b"null" if cr is None else b"%d" % (now - cr), b"null" if st is None else b"%d" % (now - st),
            b"null" if ttc is None else b"%d" % ttc)

    def frag(self, now: int) -> bytes:
        return b'%s,"age":%s%s' % (self.head, self.age_json(now), self.tail)


class _Inputs(NamedTuple):
    """Everything one refresh reads, all inside the same read transaction."""

    rows: list
    by_id: dict
    latest: int
    comments: dict
    runs: dict
    parents_of: dict
    children_of: dict
    diag_config: dict
    cfg_key: str


class _Built(NamedTuple):
    task: kanban_db.Task
    head: bytes
    tail: bytes
    ttc: Optional[int]
    age_null: bool
    has_diag: bool
    disc: tuple


class BoardCache:
    """Built cards of one board database, kept between requests. Every method runs under ``lock``."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        # A new epoch on every reset: a token from before a reset must never match the counter after it.
        self.epoch = uuid.uuid4().hex[:8]
        self.cards: dict[str, _Card] = {}
        self.rev = 0
        self.tombstones: deque = deque(maxlen=TOMBSTONES)
        self.last_event: dict[str, int] = {}
        self.events_loaded = False
        self.event_mark = 0
        self.parents_of: dict[str, list[str]] = {}
        self.children_of: dict[str, list[str]] = {}

    # -- reading the database ------------------------------------------------

    def _fold_events(self, conn: sqlite3.Connection, latest: int) -> None:
        """Newest event id per task, kept up to date from the events after the last one seen."""
        if not self.events_loaded:
            self.last_event = {r[0]: int(r[1]) for r in conn.execute(
                "SELECT task_id, MAX(id) FROM task_events GROUP BY task_id")}
            self.events_loaded = True
        elif latest > self.event_mark:
            for r in conn.execute(
                    "SELECT task_id, MAX(id) FROM task_events WHERE id > ? GROUP BY task_id", (self.event_mark,)):
                self.last_event[r[0]] = max(self.last_event.get(r[0], 0), int(r[1]))
        self.event_mark = latest

    def _read(self, conn: sqlite3.Connection) -> _Inputs:
        from hermes_cli.config import load_config

        latest = int(conn.execute("SELECT COALESCE(MAX(id), 0) FROM task_events").fetchone()[0])
        if latest < self.event_mark:  # the database was replaced or rewound: nothing cached is trustworthy
            self.reset()
        rows = board_rows(conn)
        self._fold_events(conn, latest)
        comments = {r[0]: r[1] for r in conn.execute("SELECT task_id, COUNT(*) FROM task_comments GROUP BY task_id")}
        runs = {r[0]: tuple(r)[1:] for r in conn.execute(
            "SELECT task_id, COUNT(*), MAX(id), SUM(COALESCE(ended_at, 0)), SUM(LENGTH(COALESCE(summary, ''))), "
            "SUM(LENGTH(COALESCE(metadata, ''))), SUM(LENGTH(COALESCE(error, ''))) FROM task_runs GROUP BY task_id")}
        parents_of: dict[str, list[str]] = {}
        children_of: dict[str, list[str]] = {}
        for r in conn.execute("SELECT parent_id, child_id FROM task_links"):
            children_of.setdefault(r[0], []).append(r[1])
            parents_of.setdefault(r[1], []).append(r[0])
        diag_config = kd.config_from_runtime_config(load_config())
        return _Inputs(rows, {r["id"]: r for r in rows}, latest, comments, runs, parents_of, children_of,
                       diag_config, json.dumps(diag_config, sort_keys=True, default=str))

    @staticmethod
    def _neighbours(tid: str, inp: _Inputs) -> tuple:
        """``(parents, children, children done, live children, parent states, child states)``."""
        ps, cs = inp.parents_of.get(tid), inp.children_of.get(tid)
        if not ps and not cs:
            return _NO_LINKS
        by_id = inp.by_id
        live = [by_id[c] for c in (cs or ()) if c in by_id]
        return (len(ps or ()), len(cs or ()), sum(1 for c in live if c["status"] == "done"), len(live),
                tuple(sorted((p, by_id[p]["title"], by_id[p]["status"]) for p in (ps or ()) if p in by_id)),
                tuple(sorted((c["id"], c["title"], c["status"]) for c in live)))

    def _plan(self, inp: _Inputs, now: int) -> tuple[list[str], set[str], dict, dict]:
        """``(cards to rebuild, those rebuilt only because the clock asked, signature by id, neighbours by id)``."""
        sig_of: dict[str, int] = {}
        neigh_of: dict[str, tuple] = {}
        rebuild: list[str] = []
        timer: set[str] = set()
        stale: list[_Card] = []
        for r in inp.rows:
            tid = r["id"]
            neigh = neigh_of[tid] = self._neighbours(tid, inp)
            sig = sig_of[tid] = hash((tuple(r), inp.comments.get(tid, 0), neigh, inp.runs.get(tid),
                                      self.last_event.get(tid, 0), inp.cfg_key))
            c = self.cards.get(tid)
            if c is None or c.sig != sig:
                rebuild.append(tid)
                continue
            age = now - c.built_at
            if (c.status not in TERMINAL or c.has_diag) and age >= VOLATILE_SECONDS:
                rebuild.append(tid)
                timer.add(tid)
            elif age >= ENTRY_TTL_SECONDS:
                stale.append(c)
        stale.sort(key=lambda c: c.built_at)
        rebuild.extend(c.id for c in stale[:REFRESH_PER_CALL])
        return rebuild, timer, sig_of, neigh_of

    def _build(self, conn: sqlite3.Connection, inp: _Inputs, rebuild: list[str], neigh_of: dict,
               hooks: CardHooks) -> dict[str, _Built]:
        tasks = {tid: kanban_db.Task.from_row(inp.by_id[tid]) for tid in rebuild}
        summaries: dict[str, str] = {}
        run_starts: dict[str, int] = {}
        for part in _chunked(rebuild, SUMMARIES_CHUNK):
            summaries.update(kanban_db.latest_summaries(conn, part))
            run_starts.update(kanban_db.current_run_started_ats(conn, part))
        diags: dict[str, list[dict]] = {}
        for part in _chunked([tid for tid in rebuild if tasks[tid].status != "archived"], DIAGNOSTICS_CHUNK):
            diags.update(hooks.diagnostics(conn, part, diag_config=inp.diag_config))
        built: dict[str, _Built] = {}
        for tid in rebuild:
            t, nb, dg = tasks[tid], neigh_of[tid], diags.get(tid)
            card = hooks.build(t, summaries.get(tid), run_starts.get(tid), {"parents": nb[0], "children": nb[1]},
                               inp.comments.get(tid, 0), {"done": nb[2], "total": nb[3]} if nb[3] else None, dg)
            built[tid] = _Built(t, *_card_parts(card, t), bool(dg), card_disc(dg))
        return built

    def _store(self, built: dict[str, _Built], timer: set[str], sig_of: dict, now: int) -> list[_Card]:
        """Write the built cards into the cache; return the ones a client must be sent."""
        changed: list[_Card] = []
        cold = not self.cards
        for tid, b in built.items():
            old = self.cards.get(tid)
            if old is None:
                differs = True
            elif tid in timer:  # same inputs, rebuilt on the clock: news only if something the client shows moved
                differs = old.head != b.head or old.age_null != b.age_null or old.disc != b.disc
            else:
                differs = old.head != b.head or old.tail != b.tail or old.age_null != b.age_null
            card = old or _Card()
            t = b.task
            card.id, card.status, card.created_at, card.started_at, card.completed_at = (
                tid, t.status, t.created_at, t.started_at, t.completed_at)
            card.ttc, card.age_null, card.sig, card.head, card.tail = b.ttc, b.age_null, sig_of[tid], b.head, b.tail
            card.built_at, card.has_diag, card.disc = now, b.has_diag, b.disc
            if old is None:
                card.rev = card.changed_at = 0
                self.cards[tid] = card
            if differs:
                # A cold build has no earlier state to differ from: those cards are as old as the client's.
                card.changed_at = 0 if cold else now
                changed.append(card)
        return changed

    def refresh(self, conn: sqlite3.Connection, now: int, hooks: CardHooks) -> tuple[list, int]:
        """Bring every card up to date with the database; return ``(rows in board order, newest event id)``.
        Runs inside one read transaction, so every query sees the same database."""
        inp = self._read(conn)
        self.parents_of, self.children_of = inp.parents_of, inp.children_of
        rebuild, timer, sig_of, neigh_of = self._plan(inp, now)
        # Build everything before storing anything: a failure part-way through must leave the cache exactly
        # as the last successful build left it, never holding some cards newer than the counter that names them.
        changed = self._store(self._build(conn, inp, rebuild, neigh_of, hooks), timer, sig_of, now) if rebuild else []
        gone = [tid for tid in self.cards if tid not in inp.by_id]
        if changed or gone:
            self.rev += 1
            for card in changed:
                card.rev = self.rev
            for tid in gone:
                del self.cards[tid]
                self.last_event.pop(tid, None)
                self.tombstones.append((self.rev, tid))
        return inp.rows, inp.latest

    # -- answering -------------------------------------------------------------

    def reply(self, conn: sqlite3.Connection, *, columns: list[str], hooks: CardHooks, tenant: Optional[str],
              include_archived: bool, workflow_template_id: Optional[str], current_step_key: Optional[str],
              since: Optional[int], etag: Optional[str]) -> BoardReply:
        with self.lock:
            now = int(time.time())
            conn.execute("BEGIN")
            try:
                rows, latest = self.refresh(conn, now, hooks)
                tenants = [r[0] for r in conn.execute(
                    "SELECT DISTINCT tenant FROM tasks WHERE tenant IS NOT NULL ORDER BY tenant")]
                assignees = [entry["name"] for entry in kanban_db.known_assignees(conn)]
                since_ts = self._event_time(conn, since)
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            names = list(columns) + (["archived"] if include_archived else [])
            buckets = self._buckets(rows, names, tenant=tenant, include_archived=include_archived,
                                    workflow_template_id=workflow_template_id, current_step_key=current_step_key)
            shape = json.dumps([tenant, bool(include_archived), workflow_template_id, current_step_key, tenants,
                                assignees], default=str)
            token = f"{self.epoch}.{self.rev}.{latest}.{hashlib.sha256(shape.encode()).hexdigest()[:8]}"
            head = b'"latest_event_id":%d,"now":%d,"etag":"%s"' % (latest, now, token.encode())
            if etag == token:
                return BoardReply(b'{"delta":true,"unchanged":true,' + head + b"}", token)
            lists = b',"tenants":%s,"assignees":%s' % (json_bytes(tenants), json_bytes(assignees))
            if since is None or not 0 <= since <= latest:
                cols = b",".join(b'{"name":%s,"tasks":[%s]}' % (
                    json_bytes(n), b",".join(c.frag(now) for c in buckets[n])) for n in names)
                return BoardReply(b'{"columns":[%s]%s,%s}' % (cols, lists, head), token)
            send, gone = self._delta(buckets, names, since, since_ts, etag)
            order = [{"name": n, "ids": [c.id for c in buckets[n]]} for n in names]
            return BoardReply(b'{"delta":true,"since":%d,%s,"cards":[%s],"column_order":%s,"removed":%s%s}' % (
                since, head, b",".join(c.frag(now) for c in send), json_bytes(order), json_bytes(sorted(gone)),
                lists), token)

    @staticmethod
    def _event_time(conn: sqlite3.Connection, since: Optional[int]) -> int:
        """When the client's newest event happened (0 when unknown)."""
        if since is None:
            return 0
        row = conn.execute("SELECT created_at FROM task_events WHERE id <= ? ORDER BY id DESC LIMIT 1",
                           (since,)).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _buckets(self, rows: list, names: list[str], *, tenant: Optional[str], include_archived: bool,
                 workflow_template_id: Optional[str], current_step_key: Optional[str]) -> dict[str, list[_Card]]:
        """The cards of each column: the ``list_tasks`` filters, the unknown-status-to-todo rule, and the
        done column newest-completed first (two stable sorts: ``completed_at DESC NULLS LAST, id DESC``)."""
        buckets: dict[str, list[_Card]] = {n: [] for n in names}
        wanted = (("tenant", tenant), ("workflow_template_id", workflow_template_id),
                  ("current_step_key", current_step_key))
        for r in rows:
            if not include_archived and r["status"] == "archived":
                continue
            if any(value is not None and r[col] != value for col, value in wanted):
                continue
            card = self.cards[r["id"]]
            buckets[card.status if card.status in buckets else "todo"].append(card)
        if "done" in buckets:
            buckets["done"].sort(key=lambda c: c.id, reverse=True)
            buckets["done"].sort(key=lambda c: (c.completed_at is None, -(c.completed_at or 0)))
        return buckets

    def _token_rev(self, etag: Optional[str]) -> Optional[int]:
        """The change counter an etag from THIS cache names, or None (another process, a reset, junk)."""
        if not etag:
            return None
        epoch, _, rest = etag.partition(".")
        num = rest.partition(".")[0]
        if epoch == self.epoch and num.isascii() and num.isdigit() and int(num) <= self.rev:
            return int(num)
        return None

    def _delta(self, buckets: dict[str, list[_Card]], names: list[str], since: int, since_ts: int,
               etag: Optional[str]) -> tuple[list[_Card], set[str]]:
        """``(cards to send, ids to remove)``. A token from this process names the exact cards that changed
        since the client's response (writes that leave no event included); the events cursor covers the rest."""
        tok_rev = self._token_rev(etag)
        wanted = {tid for tid, ev in self.last_event.items() if ev > since}
        if tok_rev is not None:
            wanted |= {c.id for c in self.cards.values() if c.rev > tok_rev}
        else:
            # No token from this process (a restart, or a bare ``since``): the cards with events, their
            # neighbours, every card whose build changed since the cursor's second (a diagnostic that aged
            # out leaves no event), and every live-evaluated card.
            for tid in list(wanted):
                wanted.update(self.parents_of.get(tid, ()))
                wanted.update(self.children_of.get(tid, ()))
            wanted |= {c.id for c in self.cards.values()
                       if c.changed_at >= since_ts - 2 or c.status not in TERMINAL or c.has_diag}
        on_board = {c.id for n in names for c in buckets[n]}
        gone = wanted - on_board
        if tok_rev is not None:
            gone |= {tid for rev, tid in self.tombstones if rev > tok_rev and tid not in on_board}
        return [c for n in names for c in buckets[n] if c.id in wanted], gone


_caches: dict[str, BoardCache] = {}
_caches_lock = threading.Lock()


def cache_for(db_path: str | PathLike) -> BoardCache:
    """The one cache of the board database at ``db_path`` in this process."""
    key = str(db_path)
    with _caches_lock:
        cache = _caches.get(key)
        if cache is None:
            cache = _caches[key] = BoardCache()
        return cache
