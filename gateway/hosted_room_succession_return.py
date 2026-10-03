"""Stepping down: a returning old host, or the successor the owner did not keep, becomes a backup.

The old host asks the group's other computers who they follow (``.../succession/query``). Once one
shows a verified transition out of its own epoch, it steps down with ``demote_to_custody``. A
successor the owner did not keep after ``continued_on_two`` steps down the same way, in favour of
the kept one. In one writer transaction the events after the shared history move into a
divergent branch kept here, set aside and readable (``groups.succession.branch_log``); the shared
prefix becomes a custody copy that follows the kept host; the room's driver, link and policy state
goes; and a stepped-down successor's own transition mark is archived with its branch. Catch-up
then brings in the kept host's history, whose transition the copy verifies itself.

A host that learns its epoch was fenced elsewhere pauses at once and executes nothing until it can
step down. Afterwards it reports what it did while partitioned (its driver's turns and an index of
the set-aside events) to the kept host, which keeps the report beside its own reconciliation.
Nothing from a branch is merged, replayed or re-executed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from gateway import hosted_room_succession as succession
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import table_exists, utf8_len

BRANCHES = "hosted_room_divergent_branches"
BRANCH_EVENTS = "hosted_room_divergent_events"
# The groups whose branches ``prune_branches`` deletes in its own transaction (empty otherwise).
BRANCH_PRUNES = "hosted_room_divergent_prunes"
CHECK_INTERVAL_SECONDS = 60.0
MAX_REPORTED_RUNS = 64
MAX_REPORTED_TAIL = 64
# Authority-side state of a room; participant-side reservations and copy retirement stay.
_AUTHORITY_TABLES = (
    "hosted_room_driver_leases", "hosted_room_driver_tasks", "hosted_room_remote_runs", "hosted_room_links",
    "hosted_room_link_renewals", "hosted_room_policy_transcript_state", "hosted_room_policy_transcript",
    "hosted_room_policy_publications", "hosted_room_policy_watermarks", "hosted_room_policy_events",
    "hosted_room_policy_threads", "hosted_room_policy_cursors", "hosted_room_replication_publishers",
    "hosted_room_replication_targets", "hosted_room_work_records_source", "hosted_room_work_records_pending")
_EVENT_COLUMNS = "seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at"


def initialize_branch_schema(conn: sqlite3.Connection) -> None:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {BRANCHES} (
        room_id TEXT NOT NULL, branch_id TEXT NOT NULL, own_epoch INTEGER NOT NULL, fork_seq INTEGER NOT NULL,
        successor_gateway_id TEXT NOT NULL, to_epoch INTEGER NOT NULL, proof_kind TEXT NOT NULL,
        proof_digest TEXT NOT NULL, event_count INTEGER NOT NULL, local_runs_json TEXT NOT NULL,
        detected_at REAL NOT NULL, PRIMARY KEY (room_id, branch_id))""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {BRANCH_EVENTS} (
        room_id TEXT NOT NULL, branch_id TEXT NOT NULL, seq INTEGER NOT NULL, event_id TEXT NOT NULL,
        kind TEXT NOT NULL, actor_json TEXT NOT NULL, authority_epoch INTEGER, payload_json TEXT NOT NULL,
        created_at REAL NOT NULL, PRIMARY KEY (room_id, branch_id, seq))""")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {BRANCH_PRUNES} (room_id TEXT PRIMARY KEY)")
    # A set-aside branch is evidence: readable, never rewritten or merged. It goes only with its group,
    # once that is disbanded or retired here (``prune_branches``, which marks the group as it deletes).
    for table in (BRANCHES, BRANCH_EVENTS):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_{table}_kept_update BEFORE UPDATE ON {table}
            BEGIN SELECT RAISE(ABORT, 'a set-aside branch is kept'); END""")
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_{table}_kept_delete BEFORE DELETE ON {table}
            WHEN NOT EXISTS (SELECT 1 FROM {BRANCH_PRUNES} WHERE room_id=OLD.room_id)
            BEGIN SELECT RAISE(ABORT, 'a set-aside branch is kept'); END""")


def branch_id_for(own_epoch: int, fork_seq: int) -> str:
    """One branch per epoch this computer followed and the point its history left the kept one."""
    return f"epoch-{int(own_epoch)}-at-{int(fork_seq)}"


def _head_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any]:
    room = conn.execute("""SELECT room_id, name, members_json, authority_gateway_id, authority_epoch, next_seq,
        created_at, disbanded_at FROM hosted_rooms WHERE room_id=?""", (room_id,)).fetchone()
    if room is not None and room["authority_gateway_id"] == succession.local_install_id():
        if room["disbanded_at"] is not None:
            raise succession.SuccessionError("this group was disbanded", reason="room_not_found")
        return {"kind": "room", "table": "hosted_room_events", "epoch": int(room["authority_epoch"]),
                "latest": int(room["next_seq"]) - 1, "row": room}
    copy = conn.execute("""SELECT room_id, name, members_json, authority_gateway_id, authority_epoch, last_seq,
        created_at, disbanded_at, quarantine_reason FROM hosted_room_replicas WHERE room_id=?""",
                        (room_id,)).fetchone()
    if copy is None or copy["disbanded_at"] is not None:
        raise succession.SuccessionError("this computer keeps no copy of this group", reason="room_not_found")
    if copy["quarantine_reason"] is not None:
        raise succession.SuccessionError("a quarantined copy stays as it is", reason="room_quarantined")
    return {"kind": "copy", "table": "hosted_room_replica_events", "epoch": int(copy["authority_epoch"]),
            "latest": int(copy["last_seq"]), "row": copy}


def shared_fork_locked(conn: sqlite3.Connection, room_id: str, head: Mapping[str, Any],
                       transition: Mapping[str, Any], fork_event: Mapping[str, Any] | None,
                       decision: Mapping[str, Any] | None) -> int | None:
    """The last seq this computer's history shares with the kept host's, once the kept host is
    verified; ``None`` when this computer already holds that transition."""
    payload = transition["payload"]
    kept_seq, successor = int(transition["seq"]), str(payload["successor_gateway_id"])
    to_epoch, from_epoch = int(payload["to_epoch"]), int(payload["from_epoch"])
    if succession.same_event(succession.event_at_locked(conn, room_id, kept_seq), transition):
        return None
    if payload.get("proof_digest") != succession.proof_digest(payload.get("proof")):
        raise succession.ProofInvalid("the transition does not bind its proof")
    rival = succession.own_transition_locked(conn, room_id, from_epoch=from_epoch)
    if head["kind"] == "copy" and head["latest"] < kept_seq and (rival is None or rival["seq"] >= kept_seq):
        return None  # nothing diverged: the copy verifies the transition when it catches up
    shares_fork = kept_seq == 1 or head["latest"] < kept_seq - 1 or succession.same_event(
        succession.event_at_locked(conn, room_id, kept_seq - 1), fork_event)
    if head["epoch"] == from_epoch and shares_fork and (rival is None or rival["seq"] < kept_seq):
        fork = min(kept_seq - 1, head["latest"])
        succession.verify_proof_locked(
            conn, room_id, proof_kind=str(payload["proof_kind"]), proof=payload["proof"], from_epoch=from_epoch,
            to_epoch=to_epoch, successor=successor, fork_seq=kept_seq - 1,
            configuration=succession.configuration_through_locked(conn, room_id, fork))
        return fork
    if rival is not None and rival["payload"]["successor_gateway_id"] != successor:
        fork = min(kept_seq, int(rival["seq"])) - 1
        chosen = succession.verify_decision_locked(conn, room_id, decision)
        if chosen["keep_install_id"] != successor or to_epoch < chosen["epoch"]:
            raise succession.ProofInvalid("the owner did not choose this computer's other lineage")
        try:  # the kept host's own fresh epoch verifies in full against the history both share
            succession.verify_proof_locked(
                conn, room_id, proof_kind=str(payload["proof_kind"]), proof=payload["proof"], from_epoch=from_epoch,
                to_epoch=to_epoch, successor=successor,
                configuration=succession.configuration_through_locked(conn, room_id, fork))
        except succession.ProofInvalid:
            succession.verify_claim_locked(conn, room_id, payload, fork_seq=fork)
        return fork
    raise succession.SuccessionError("the transition does not replace this computer's history",
                                     reason="room_authority_superseded")


def demote_to_custody(db_path: Path | str, *, room_id: str, transition: Mapping[str, Any],
                      fork_event: Mapping[str, Any] | None, decision: Mapping[str, Any] | None = None,
                      local_runs: list[dict[str, Any]] | None = None,
                      now: float | None = None) -> dict[str, Any] | None:
    """Make this computer's room or copy follow the kept host of ``transition``; ``None`` when it already does.

    ``transition`` is the kept host's ``authority.transition`` event and ``fork_event`` the event
    just before it. Either this computer's history continues the epoch that transition leaves
    (the old host, or a copy holding more of its events than the kept host adopted), verified in
    full against that shared history; or it followed a rival successor at the same epoch, which
    needs the owner's choice (``decision``) and the kept host's signed claim. Whatever this
    computer holds beyond the shared history is set aside in one branch. One writer transaction
    does all of it, so a crash leaves either the room or copy as it was, or a clean copy with its
    branch, never a mix.
    """
    now = time.time() if now is None else float(now)
    payload = transition["payload"]
    successor, to_epoch, from_epoch = (str(payload["successor_gateway_id"]), int(payload["to_epoch"]),
                                       int(payload["from_epoch"]))
    with rooms._transaction(Path(db_path), immediate=True) as conn:
        initialize_branch_schema(conn)
        head = _head_locked(conn, room_id)
        fork = shared_fork_locked(conn, room_id, head, transition, fork_event, decision)
        if fork is None:
            return None
        branch_id = branch_id_for(head["epoch"], fork)
        table = head["table"]
        tail = conn.execute(f"SELECT {_EVENT_COLUMNS} FROM {table} WHERE room_id=? AND seq>? ORDER BY seq",
                            (room_id, fork)).fetchall()
        conn.executemany(f"""INSERT INTO {BRANCH_EVENTS}(room_id, branch_id, {_EVENT_COLUMNS})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", [(room_id, branch_id, *row) for row in tail])
        conn.execute(f"""INSERT INTO {BRANCHES}(room_id, branch_id, own_epoch, fork_seq, successor_gateway_id,
            to_epoch, proof_kind, proof_digest, event_count, local_runs_json, detected_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (room_id, branch_id, head["epoch"], fork, successor, to_epoch, payload["proof_kind"],
             payload["proof_digest"], len(tail), json.dumps((local_runs or [])[-MAX_REPORTED_RUNS:], sort_keys=True),
             now))
        prefix = conn.execute(f"SELECT {_EVENT_COLUMNS} FROM {table} WHERE room_id=? AND seq<=? ORDER BY seq",
                              (room_id, fork)).fetchall()
        prefix_bytes = sum(utf8_len(row["event_id"], row["kind"], row["actor_json"], row["payload_json"])
                           for row in prefix)
        # The copy follows the host the shared history was written under; the kept host's own
        # transition out of that epoch arrives with catch-up and is verified there.
        previous = str(payload["proof"].get("previous_authority") or (succession.host_entry(
            succession.configuration_through_locked(conn, room_id, fork)) or {}).get("install_id"))
        if head["kind"] == "room":
            room = head["row"]
            for authority_table in _AUTHORITY_TABLES:
                if table_exists(conn, authority_table):
                    conn.execute(f"DELETE FROM {authority_table} WHERE room_id=?", (room_id,))
            if table_exists(conn, "state_meta"):
                # The owner key stays: the owner still reads the copy and the messages set aside here.
                conn.execute("DELETE FROM state_meta WHERE key=?", ("gateway.peer.retiring.v1:" + room_id,))
            conn.execute("DELETE FROM hosted_room_events WHERE room_id=?", (room_id,))
            conn.execute("DELETE FROM hosted_rooms WHERE room_id=?", (room_id,))
            conn.execute("DELETE FROM hosted_room_id_reservations WHERE room_id=? AND owner_kind='authority'",
                         (room_id,))
            conn.execute("""INSERT INTO hosted_room_replicas (room_id, name, members_json, authority_gateway_id,
                    authority_epoch, last_seq, latest_seq, event_bytes, created_at, updated_at, disbanded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                         (room_id, room["name"], room["members_json"], previous, from_epoch, fork, fork, prefix_bytes,
                          float(room["created_at"]), now))
            conn.executemany(f"""INSERT INTO hosted_room_replica_events (room_id, {_EVENT_COLUMNS})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", [(room_id, *row) for row in prefix])
        else:
            conn.execute("DELETE FROM hosted_room_replica_events WHERE room_id=? AND seq>?", (room_id, fork))
            conn.execute("""UPDATE hosted_room_replicas SET authority_gateway_id=?, authority_epoch=?, last_seq=?,
                latest_seq=?, event_bytes=?, updated_at=? WHERE room_id=?""",
                         (previous, from_epoch, fork, fork, prefix_bytes, now, room_id))
        succession.reset_chain_locked(conn, room_id, after_seq=fork)
        for row in tail:
            if row["kind"] == "authority.transition":
                succession.mark_moved_to_branch(conn, room_id=room_id, to_epoch=int(row["authority_epoch"]),
                                                branch_id=branch_id)
        succession.record_lineage_locked(
            conn, room_id, origin_install_id=str(payload["proof"].get("origin_install_id") or ""),
            gateway_id=successor, epoch=to_epoch, role=str(payload["proof_kind"]), proof_digest=payload["proof_digest"])
        record = {"state": "stepped_down" if head["kind"] == "room" else "rebased", "branch_id": branch_id,
                  "own_epoch": head["epoch"], "fork_seq": fork, "successor": successor, "to_epoch": to_epoch,
                  "at": now, "separate_events": len(tail), "caught_up": False, "reported": head["kind"] != "room",
                  **({"decision": dict(decision)} if decision is not None else {})}
        if head["kind"] == "room" or tail:
            succession.save_record_locked(conn, room_id, "return", record)
    return {"room_id": room_id, "kind": head["kind"], **record}


def branches(db_path: Path | str, room_id: str) -> list[dict[str, Any]]:
    """The branches this computer set aside for a room, newest first."""
    with closing(rooms._read_connection(Path(db_path))) as conn:
        if not table_exists(conn, BRANCHES):
            return []
        return [{"branch_id": r[0], "own_epoch": int(r[1]), "fork_seq": int(r[2]), "successor_gateway_id": r[3],
                 "to_epoch": int(r[4]), "proof_kind": r[5], "proof_digest": r[6], "separate_events": int(r[7]),
                 "local_runs": json.loads(r[8]), "detected_at": float(r[9])}
                for r in conn.execute(f"""SELECT branch_id, own_epoch, fork_seq, successor_gateway_id, to_epoch,
                    proof_kind, proof_digest, event_count, local_runs_json, detected_at FROM {BRANCHES}
                    WHERE room_id=? ORDER BY own_epoch DESC""", (room_id,))]


def prune_branches(db_path: Path | str) -> int:
    """Delete the branches of groups this computer no longer holds: disbanded, retired or gone."""
    with rooms._transaction(Path(db_path), immediate=True) as conn:
        if not table_exists(conn, BRANCHES):
            return 0
        initialize_branch_schema(conn)
        live = _live_rooms_sql(conn)
        stale = [row[0] for row in conn.execute(f"SELECT DISTINCT room_id FROM {BRANCHES} WHERE room_id NOT IN ({live})")]
        for room_id in stale:
            conn.execute(f"INSERT OR IGNORE INTO {BRANCH_PRUNES}(room_id) VALUES (?)", (room_id,))
            conn.execute(f"DELETE FROM {BRANCH_EVENTS} WHERE room_id=?", (room_id,))
            conn.execute(f"DELETE FROM {BRANCHES} WHERE room_id=?", (room_id,))
            conn.execute(f"DELETE FROM {BRANCH_PRUNES} WHERE room_id=?", (room_id,))
        return len(stale)


def _live_rooms_sql(conn: sqlite3.Connection) -> str:
    """The rooms this computer still holds: hosted, or a copy neither disbanded nor retired."""
    from gateway.hosted_room_replica_retirement import RETIREMENT_TABLE
    parts = ["SELECT room_id FROM hosted_rooms WHERE disbanded_at IS NULL"]
    if table_exists(conn, "hosted_room_replicas"):
        retired = (f" AND room_id NOT IN (SELECT room_id FROM {RETIREMENT_TABLE})"
                   if table_exists(conn, RETIREMENT_TABLE) else "")
        parts.append("SELECT room_id FROM hosted_room_replicas WHERE disbanded_at IS NULL" + retired)
    return " UNION ".join(parts)


def branch_log(db_path: Path | str, room_id: str, branch_id: str, *, after_seq: int = 0,
               limit: int = 200) -> dict[str, Any]:
    """One page of a set-aside branch, in the ``groups.log`` page shape; gone with its disbanded or retired group."""
    limit = max(1, min(int(limit), rooms.MAX_LOG_LIMIT))
    with closing(rooms._read_connection(Path(db_path))) as conn:
        found = table_exists(conn, BRANCHES) and conn.execute(
            f"SELECT 1 FROM {BRANCHES} WHERE room_id=? AND branch_id=? AND room_id IN ({_live_rooms_sql(conn)})",
            (room_id, branch_id)).fetchone()
        if not found:
            raise succession.SuccessionError("no set-aside messages with that id", reason="branch_not_found")
        rows = conn.execute(f"""SELECT {_EVENT_COLUMNS} FROM {BRANCH_EVENTS} WHERE room_id=? AND branch_id=?
            AND seq>? ORDER BY seq LIMIT ?""", (room_id, branch_id, int(after_seq), limit + 1)).fetchall()
        latest = conn.execute(f"SELECT MAX(seq) FROM {BRANCH_EVENTS} WHERE room_id=? AND branch_id=?",
                              (room_id, branch_id)).fetchone()[0]
    events = [succession.event_dict(room_id, row) for row in rows[:limit]]
    cursor = events[-1]["seq"] if events else int(after_seq)
    return {"room_id": room_id, "branch_id": branch_id, "events": events, "cursor": cursor,
            "latest_seq": int(latest or 0), "has_more": len(rows) > limit}


# --- noticing the move, pausing and stepping down -------------------------------------------------
def _local_runs(db_path: Path, room_id: str) -> list[dict[str, Any]]:
    """What this host's driver knew about its own turns: reported, never replayed."""
    from gateway import hosted_room_driver as driver
    try:
        tasks = driver.list_tasks(db_path, room_id=room_id)
    except Exception:
        return []
    return [{"task_id": task["identity"].task_id, "status": task["status"],
             "execution_generation": task["execution_generation"],
             "member_id": task["payload"].get("target_member_id") or task["payload"].get("target_profile")}
            for task in tasks[-MAX_REPORTED_RUNS:]]


def _ask_everyone(ctx, room_id: str, epoch: int) -> list[dict[str, Any]]:
    """Signed answers from the group's other computers about the move out of ``epoch``."""
    from gateway import hosted_room_succession_move as move
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        configuration = succession.configuration_locked(conn, room_id)
    answers = move.survey(ctx, room_id, configuration, from_epoch=epoch)
    return [outcome["answer"] for _, outcome in sorted(answers.items()) if outcome.get("answer")]


def _fenced_elsewhere(answer: Mapping[str, Any], epoch: int) -> dict[str, Any] | None:
    """The computer an answer says this host's epoch was promised to, when it was."""
    fence = answer.get("fence") if isinstance(answer.get("fence"), Mapping) else {}
    promise, authority = fence.get("promise") or {}, fence.get("authority") or {}
    if int(authority.get("epoch") or 0) > epoch:
        return {"install_id": authority.get("install_id"), "epoch": int(authority["epoch"])}
    if int(promise.get("epoch") or 0) > epoch or int(fence.get("fenced_epoch") or 0) >= epoch:
        return {"install_id": promise.get("candidate_install_id"), "epoch": int(promise.get("epoch") or epoch + 1)}
    return None


def check(ctx, room_id: str) -> dict[str, Any] | None:
    """Ask the group's computers about this host's epoch: step down after a verified move, and
    pause at once while another computer holds a promise of a later epoch."""
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        room = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                            "AND disbanded_at IS NULL", (room_id,)).fetchone()
    if room is None or room[0] != succession.local_install_id():
        return None
    epoch, fenced = int(room[1]), None
    from gateway import hosted_room_succession_move as move
    answers = list(_ask_everyone(ctx, room_id, epoch))
    for answer in answers:
        # The owner kept this host while it was away: it continues instead of stepping down.
        if move.adopt_kept_decision(ctx, room_id, answer.get("decision")):
            try:
                move.keep_continue(ctx, room_id)
            except succession.SuccessionError:
                pass  # upkeep tries again; meanwhile the conflict pauses both
            return None
    for answer in answers:
        found = answer.get("transition")
        if isinstance(found, dict) and int(found["event"]["payload"].get("from_epoch", -1)) == epoch:
            demoted = step_down(ctx, room_id, found["event"], found.get("fork_event"))
            if demoted is not None:
                return demoted
        fenced = fenced or _fenced_elsewhere(answer, epoch)
    if fenced is not None:
        pause(ctx.db_path, room_id, fenced)
        try:  # a step promised long ago that its computer never took: continue past it (by rule)
            move.recover_stalled_promise(ctx, room_id, answers)
        except (succession.SuccessionError, rooms.HostedRoomError):
            pass  # the next check tries again
    return None


def pause(db_path: Path, room_id: str, fenced_by: Mapping[str, Any]) -> None:
    """Another computer holds a later epoch of this host's group: execute nothing from now on."""
    record = succession.load_record(db_path, room_id, "return") or {}
    if record.get("state") in {"paused", "stepped_down"}:
        return
    succession.save_record(db_path, room_id, "return", {"state": "paused", "fenced_by": dict(fenced_by),
                                                       "at": time.time()})


def paused_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    record = succession.load_record_locked(conn, room_id, "return") or {}
    return record.get("state") == "paused"


def paused(db_path: Path, room_id: str) -> bool:
    with closing(rooms._read_connection(Path(db_path))) as conn:
        return paused_locked(conn, room_id)


def step_down(ctx, room_id: str, transition: Mapping[str, Any], fork_event: Mapping[str, Any] | None, *,
              decision: Mapping[str, Any] | None = None, follow: bool = True) -> dict[str, Any] | None:
    """Stop this computer's work for the group, then follow the kept host.

    The kept host is verified first, so a transition this computer cannot verify stops nothing.
    """
    from gateway import hosted_room_fence as fence
    try:
        with closing(rooms._read_connection(ctx.db_path)) as conn:
            head = _head_locked(conn, room_id)
            fork = shared_fork_locked(conn, room_id, head, transition, fork_event, decision)
            if fork is None:
                return None
    except (succession.SuccessionError, KeyError, TypeError, ValueError):
        return None
    service = getattr(ctx, "service", None)
    if (head["kind"] == "room" and decision is None and transition["payload"].get("proof_kind") in
            succession.AUTOMATIC_PROOFS and _wrote_after(ctx.db_path, room_id, fork)):
        # The group moved by itself while this host kept writing: the owner chooses which history it
        # keeps. After the owner's own continuation, this host's later messages are set aside instead.
        return continued_on_two(ctx, room_id, transition, fork_event, fork=fork)
    if head["kind"] == "room" and service is not None:
        try:  # a stepping-down host appends nothing: no Stop event, only its own runs cancelled
            stop = getattr(service, "stop_work", None)
            if stop is not None:
                stop(room_id, cancel_id="authority-moved")
        except Exception:
            pass  # the branch is set aside either way; nothing from it is replayed
    demoted = demote_to_custody(ctx.db_path, room_id=room_id, transition=transition, fork_event=fork_event,
                                decision=decision, local_runs=_local_runs(ctx.db_path, room_id))
    if demoted is None:
        return None
    from gateway.hosted_room_succession_automatic import started_following
    started_following(ctx.db_path, room_id)  # its silence counts from now, never from before it hosted
    if head["kind"] == "room":
        keep_copy_consent(ctx.db_path, room_id)
    try:  # a host without a fence store has no participant side to fence
        if getattr(ctx, "runs_store", None) is not None:
            fence.learn_authority(ctx.runs_store.path, room_id=room_id, epoch=demoted["to_epoch"],
                                  install_id=demoted["successor"])
    except fence.RoomAuthorityConflict:
        pass  # a rival successor's own epoch: the kept host's next epoch supersedes it here
    if follow:
        follow_up(ctx, room_id)
    return demoted


# A copy-only grant the old host gives each later host: as long as a grant can last (#104601's limits).
COPY_GRANT_OPTIONS = {"replication": True, "work_records": False, "passive_only": True, "successor": True,
                      "ttl_seconds": 24 * 3600, "status_ttl_seconds": 30 * 24 * 3600}


def keep_copy_consent(db_path: Path | str, room_id: str) -> None:
    """A host that stepped down keeps a copy: its consent to the copy-only grant every later host gets
    (sent with its report, and with its fence receipts), so that host pushes the history and its lease
    requests to it. Its operator ran the group here, so it may continue it again, unless it said no."""
    custody = succession._custody()
    with rooms._transaction(Path(db_path), immediate=True) as conn:
        succession.record_consent_locked(conn, room_id=room_id, member_id=succession.custody_member_id(),
                                         target_profile="default", options=COPY_GRANT_OPTIONS)
        chosen = table_exists(conn, custody.CONSENT_TABLE) and conn.execute(
            f"SELECT 1 FROM {custody.CONSENT_TABLE} WHERE room_id=?", (room_id,)).fetchone() is not None
    if not chosen:
        custody.set_local_consent(db_path, room_id=room_id, allowed=True)


def follow_up(ctx, room_id: str) -> dict[str, Any] | None:
    """After stepping down: catch the copy up to the kept host, then report this computer's evidence."""
    from gateway import hosted_room_succession_move as move
    record = succession.load_record(ctx.db_path, room_id, "return")
    if not record or record.get("state") not in {"stepped_down", "rebased"} or (
            record.get("caught_up") and record.get("reported")):
        return record
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        others = sorted(set(succession.custodians_by_id(succession.configuration_locked(conn, room_id)))
                        - {record["successor"], succession.local_install_id()})
    # The kept host first; when it can't be reached, any other computer that holds its history.
    # Another computer's copy counts only as far as the kept host signed it: never the other side's.
    for source in [record["successor"], *others]:
        try:
            move.catch_up(ctx, room_id, source, signed_by=None if source == record["successor"]
                          else record["successor"])
        except Exception:
            continue  # the next source, or the next upkeep cycle, continues from the copy's own watermark
        if source == record["successor"]:
            record["caught_up"] = True
            break
    if not record.get("reported"):
        record["reported"] = _report(ctx, room_id, record)
    succession.save_record(ctx.db_path, room_id, "return", record)
    return record


def _report(ctx, room_id: str, record: Mapping[str, Any]) -> bool:
    from gateway import hosted_room_succession_move as move
    branch = next((item for item in branches(ctx.db_path, room_id) if item["branch_id"] == record["branch_id"]), None)
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        kept = succession.custodians_by_id(succession.configuration_locked(conn, room_id)).get(record["successor"])
        tail = [{"seq": int(r[0]), "event_id": r[1], "kind": r[2], "created_at": float(r[3])} for r in conn.execute(
            f"""SELECT seq, event_id, kind, created_at FROM {BRANCH_EVENTS} WHERE room_id=? AND branch_id=?
                ORDER BY seq LIMIT ?""", (room_id, record["branch_id"], MAX_REPORTED_TAIL))] if branch else []
    if branch is None or kept is None or not kept.get("endpoint"):
        return False
    grants = []
    if getattr(ctx, "mint_grants", None) is not None:
        try:  # the copy-only grant (and any member's) the kept host pushes to this copy with
            grants = ctx.mint_grants(room_id, record["successor"], int(record["to_epoch"]))
        except Exception:
            grants = []
    unsigned = {"room_id": room_id, "reporter_install_id": succession.local_install_id(), "issued_at": time.time(),
                "nonce": succession.nonce(), "from_epoch": branch["own_epoch"], "fork_seq": branch["fork_seq"],
                "divergent": {"events": branch["separate_events"]}, "runs": branch["local_runs"], "tail": tail,
                **({"grants": grants} if grants else {})}
    try:
        (ctx.post or move.http_post)(str(kept["endpoint"]), "/v1/room-members/succession/report",
                                     {**unsigned, "signature": succession.sign(succession.REPORT, unsigned)},
                                     ctx.timeout)
    except Exception:
        return False  # the branch stays here, readable; the next upkeep cycle reports again
    return True


_checked: dict[tuple[str, str], float] = {}


def maintain(ctx, room_ids, *, suspect=frozenset(), clock=None) -> list[dict[str, Any]]:
    """Check each hosted room once a minute, at once when a peer refused its authority."""
    from gateway import hosted_room_clock
    clock = clock or hosted_room_clock.now
    demoted = []
    for room_id in room_ids:
        key = (str(ctx.db_path), room_id)
        last = _checked.get(key)
        if last is not None and room_id not in suspect and clock() - last < CHECK_INTERVAL_SECONDS:
            continue
        _checked[key] = clock()
        try:
            result = check(ctx, room_id)
        except (succession.SuccessionError, rooms.HostedRoomError):
            continue
        if result is not None:
            demoted.append(result)
    return demoted


def _wrote_after(db_path: Path, room_id: str, fork: int) -> bool:
    """Whether this host holds events after ``fork``: history the other host lacks."""
    with closing(rooms._read_connection(Path(db_path))) as conn:
        row = conn.execute("SELECT next_seq - 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    return row is not None and int(row[0]) > fork


def continued_on_two(ctx, room_id: str, transition: Mapping[str, Any], fork_event: Mapping[str, Any] | None, *,
                     fork: int) -> dict[str, Any]:
    """Both hosts wrote after the split: record it here and tell the other host. The rule picks the one
    that keeps running (here, a host that never moved loses to the verified move); this one stops,
    keeps its own messages apart until the owner chooses, and its Bots follow the kept host."""
    from gateway import hosted_room_fence as fence
    from gateway import hosted_room_succession_move as move
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        room = conn.execute("SELECT authority_epoch, next_seq - 1 FROM hosted_rooms WHERE room_id=?",
                            (room_id,)).fetchone()
        first = conn.execute("SELECT created_at FROM hosted_room_events WHERE room_id=? AND seq=?",
                             (room_id, fork + 1)).fetchone()
    payload = transition["payload"]
    statement = payload.get("proof", {}).get("statement") if isinstance(payload.get("proof"), Mapping) else None
    start = statement.get("silent_since") if isinstance(statement, Mapping) else None
    record = move.record_conflict(
        ctx.db_path, room_id, start=start,
        mine={"event": None, "head": {"install_id": succession.local_install_id(), "epoch": int(room[0]),
                                      "since": float(first[0]) if first else None, "latest_seq": int(room[1])}},
        theirs={"event": dict(transition), "fork_event": dict(fork_event) if fork_event else None})
    if record.get("winner") == payload["successor_gateway_id"] and getattr(ctx, "runs_store", None) is not None:
        try:  # as a participant, this computer takes the kept host's work meanwhile
            fence.learn_authority(ctx.runs_store.path, room_id=room_id, epoch=int(payload["to_epoch"]),
                                  install_id=str(payload["successor_gateway_id"]))
        except fence.RoomFenceError:
            pass
    _tell_other_host(ctx, room_id, record, fork=fork)
    return {"room_id": room_id, "state": "continued_on_two", "fork_seq": fork}


def _tell_other_host(ctx, room_id: str, record: Mapping[str, Any], *, fork: int) -> bool:
    """Report to the other host that this one wrote after the split, so it records the conflict too."""
    from gateway import hosted_room_succession_move as move
    theirs = record["theirs"]["event"]["payload"]["successor_gateway_id"]
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        endpoint = succession.custodians_by_id(succession.configuration_locked(conn, room_id)).get(
            theirs, {}).get("endpoint")
    if not endpoint:
        return False
    mine = record["mine"]["head"]
    unsigned = {"room_id": room_id, "reporter_install_id": succession.local_install_id(), "issued_at": time.time(),
                "nonce": succession.nonce(), "from_epoch": mine["epoch"], "fork_seq": fork,
                "divergent": {"events": int(mine["latest_seq"]) - fork}, "post_split": {
                    "since": mine.get("since"), "start": record.get("start"), "latest_seq": mine["latest_seq"]}}
    try:
        (ctx.post or move.http_post)(str(endpoint), "/v1/room-members/succession/report",
                                     {**unsigned, "signature": succession.sign(succession.REPORT, unsigned)},
                                     ctx.timeout)
    except Exception:
        return False
    return True


def _with_epochs(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The chain with each event's epoch, derived from its transitions where a client left it out: an
    event before the first transition belongs to the epoch that transition leaves, any later one to
    the epoch of the transition before it. A wrong one only fails to match; it never verifies."""
    ordered = sorted((dict(event) for event in events), key=lambda event: (
        event.get("seq") if isinstance(event.get("seq"), int) else 0))
    first = next((event for event in ordered if event.get("kind") == "authority.transition"
                  and isinstance(event.get("payload"), Mapping)), None)
    epoch = (first or {}).get("payload", {}).get("from_epoch")
    for event in ordered:
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        if event.get("kind") == "authority.transition":
            epoch = payload.get("to_epoch")
        if event.get("authority_epoch") is None:
            event["authority_epoch"] = epoch
    return ordered


def learn(ctx, room_id: str, events: Any) -> dict[str, Any]:
    """``groups.succession.learn``: a chain of later hosts reaches this computer from any device.

    ``events`` are the ``authority.transition`` and ``custody.configured`` events after this
    computer's epoch, in log order, with the event just before the first transition. The first
    transition out of this computer's own epoch is verified here, with pinned keys only, against
    the history this computer holds; anything later is verified in order as the copy catches up. A
    forged, replayed or unverifiable chain changes nothing.
    """
    if not isinstance(events, list) or not all(isinstance(event, Mapping) for event in events):
        raise succession.ProofInvalid("the chain has the wrong shape")
    events = _with_epochs(events)
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        head = _head_locked(conn, room_id)
    by_seq = {int(event["seq"]): dict(event) for event in events if isinstance(event.get("seq"), int)}
    leaving = next((dict(event) for event in sorted(events, key=lambda item: int(item.get("seq") or 0))
                    if event.get("kind") == "authority.transition"
                    and (event.get("payload") or {}).get("from_epoch") == head["epoch"]), None)
    if leaving is None:
        return {"room_id": room_id, "learned": False, "reason": "not_superseded"}
    fork_event = by_seq.get(int(leaving["seq"]) - 1)
    outcome = step_down(ctx, room_id, leaving, fork_event, follow=False)
    if outcome is None:
        with closing(rooms._read_connection(ctx.db_path)) as conn:
            if succession.same_event(succession.event_at_locked(conn, room_id, int(leaving["seq"])), leaving):
                return {"room_id": room_id, "learned": False, "reason": "already_following"}
        raise succession.ProofInvalid("the chain could not be verified here")
    return {"room_id": room_id, "learned": True, **{key: outcome[key] for key in ("state",) if key in outcome}}
