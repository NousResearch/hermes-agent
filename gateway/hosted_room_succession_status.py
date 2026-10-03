"""What a computer knows about its group's host, as ``groups.succession.status`` reports it, and the upkeep
behind it.

``status`` answers ``groups.succession.status`` from this computer's own records: the room it
hosts or the copy it keeps, the configuration, its custody view, and the heartbeat record the
upkeep keeps. A backup asks the configured host every minute (the signed succession query); the
host counts as offline only after a bounded window without an answer, never after one missed
poll, and a planned restart the host announced is not loss. Every field is a code or parameter;
times are Unix seconds.

When the host stays offline, the eligible computer best placed to continue the group (ranked as
``actions[continue].targets``) sends the owner one notice per incident: through the owner's
private chats when messaging offers them, otherwise the home channel. Another eligible computer
waits five minutes and sends only while every better-placed one still looks offline from here.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from gateway import hosted_room_succession as succession
from gateway import hosted_rooms as rooms
from gateway.hosted_room_succession import SuccessionError

HEARTBEAT_INTERVAL_SECONDS = 60.0
UNREACHABLE_AFTER_SECONDS = 300.0
RESTART_GRACE_SECONDS = 120.0
NOTICE_DELAY_SECONDS = 300.0
UPKEEP_INTERVAL_SECONDS = 15.0
# On the host, a custodian that has acknowledged no push for this long counts as offline. The host
# pushes to a voter about every 5 s and to other custodians about every minute (#104601), so these
# are several missed rounds, never one.
VOTER_SILENT_SECONDS = 30.0
CUSTODIAN_SILENT_SECONDS = 180.0
# In majority mode the voters move the group by themselves; when that still hasn't happened this long
# after the host counted as offline, the owner may continue it by hand.
TAKEOVER_WAIT_SECONDS = 300.0
_READINESS_RANK = {"caught_up": 0, "behind": 1, "unknown": 2, "offline": 3, "needs_reauthorization": 3,
                   "unsupported": 4}
# Readiness values of a computer that can't take part now: it gets no copy, so it can't vote either.
UNAVAILABLE = frozenset({"offline", "needs_reauthorization", "unsupported"})
# Reads of one status while the room keeps changing hands under it, before the last one stands.
STATUS_READS = 3


# --- the room as this computer holds it ------------------------------------------------------------
def _holding(conn, room_id: str) -> dict[str, Any]:
    """The room this computer hosts or keeps a copy of, its membership, and what it set aside."""
    room = conn.execute("""SELECT name, members_json, authority_gateway_id, authority_epoch, next_seq - 1
        FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL""", (room_id,)).fetchone()
    if room is not None:
        return {"kind": "room" if room[2] == succession.local_install_id() else "foreign", "name": room[0],
                "members": room[1], "authority": room[2], "epoch": int(room[3]), "latest_seq": int(room[4])}
    copy = conn.execute("""SELECT name, members_json, authority_gateway_id, authority_epoch, latest_seq, last_seq
        FROM hosted_room_replicas WHERE room_id=? AND disbanded_at IS NULL AND quarantine_reason IS NULL""",
                        (room_id,)).fetchone()
    if copy is not None:
        return {"kind": "copy", "name": copy[0], "members": copy[1], "authority": copy[2], "epoch": int(copy[3]),
                "latest_seq": int(copy[4]), "held_seq": int(copy[5])}
    member = conn.execute("SELECT 1 FROM hosted_room_peer_reservations WHERE room_id=? LIMIT 1",
                          (room_id,)).fetchone() if succession.table_exists(conn, "hosted_room_peer_reservations") \
        else None
    if member is not None:
        return {"kind": "member"}
    if any(succession.load_record_locked(conn, room_id, kind) for kind in ("move", "return", "heartbeat")):
        return {"kind": "none"}
    raise SuccessionError("this computer holds no record of that group", reason="room_not_found")


def _readiness(entry: Mapping[str, Any], *, latest_seq: int, now: float, heard: Mapping[str, Any] | None,
               hosting: bool = False, route: str | None = None):
    if entry.get("state") == "unsupported":
        return "unsupported", None
    if hosting and route == "needs_reauthorization":
        # The host's copy to it is refused for lack of permission (its grant lapsed while the host couldn't
        # reach it): not offline, it needs its grant renewed before it gets the history again.
        return "needs_reauthorization", None
    watermark = entry.get("watermark") or (heard or {}).get("watermark")
    seen = entry.get("acknowledged_at") or (heard or {}).get("last_ok")
    if (heard or {}).get("last_attempt") and not (heard or {}).get("ok", True):
        if seen is None or now - float(seen) > UNREACHABLE_AFTER_SECONDS:
            return "offline", None
    if hosting and entry.get("acknowledged_at") is not None:
        # The host pushes to every custodian on a cadence: silence since then means it is offline.
        silent = VOTER_SILENT_SECONDS if entry.get("voter") else CUSTODIAN_SILENT_SECONDS
        if now - float(entry["acknowledged_at"]) > silent:
            return "offline", None
    if not isinstance(watermark, Mapping):
        return "unknown", None
    behind = max(0, int(latest_seq) - int(watermark["seq"]))
    return ("caught_up" if behind == 0 else "behind"), behind


def backups(configuration: Mapping[str, Any], custody: Mapping[str, Any] | None, *, host_id: str | None,
            latest_seq: int, heartbeat: Mapping[str, Any], now: float,
            routes: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """One row per computer that keeps a copy, best placed first among those that can continue. ``routes``
    is, on the host, the state of its copy to each computer (#104601's publisher)."""
    entries = {item["install_id"]: item for item in (custody or {}).get("custodians", ())}
    peers = heartbeat.get("peers") or {}
    hosting = host_id is not None and host_id == succession.local_install_id()
    rows = []
    for order, item in enumerate(configuration.get("custodians") or ()):
        install_id = item["install_id"]
        if install_id == host_id:
            continue
        entry = {**item, **entries.get(install_id, {})}
        readiness, behind = _readiness(entry, latest_seq=latest_seq, now=now, heard=peers.get(install_id),
                                       hosting=hosting, route=(routes or {}).get(install_id))
        allowed = entry.get("allowed")
        designated = entry.get("designated")
        rows.append({
            "install_id": install_id, "name": succession.label(configuration, install_id),
            "successor": bool(item.get("successor")), "readiness": readiness, "behind_by": behind,
            "last_seen": entry.get("acknowledged_at") or (peers.get(install_id) or {}).get("last_ok"),
            "allowed": bool(item.get("successor")) if allowed is None else bool(allowed),
            "designated": bool(item.get("successor")) if designated is None else bool(designated),
            "voter": bool(item.get("voter")), "always_on": bool(entry.get("always_on")),
            "kind": "backup" if item.get("role") == "custodian_only" else "member",
            "operator_name": succession.label(configuration, install_id, "operator_name"), "_order": order})
    return rows


def ranked_targets(rows: list[Mapping[str, Any]]) -> list[str]:
    """Eligible computers, best placed first: always on, then readiness, then the owner's order."""
    eligible = [row for row in rows if row["successor"] and row["readiness"] != "unsupported"]
    eligible.sort(key=lambda row: (not row.get("always_on", False), _READINESS_RANK[row["readiness"]], row["_order"]))
    return [row["install_id"] for row in eligible]


def _host_view(holding: Mapping[str, Any], configuration: Mapping[str, Any], heartbeat: Mapping[str, Any],
               restarting: float | None, now: float) -> dict[str, Any]:
    me = succession.local_install_id()
    host_id = (succession.host_entry(configuration) or {}).get("install_id") or holding.get("authority")
    host = {"install_id": host_id, "name": succession.label(configuration, host_id), "reachable": True, "since": None}
    if host_id == me:
        return host
    last_ok, first = heartbeat.get("last_ok"), heartbeat.get("first_failure")
    failing = heartbeat.get("failing", False)
    if failing:
        since = last_ok if last_ok is not None else first
        host.update(reachable=False, since=since)
    if restarting is not None:
        host["restarting_until"] = restarting
    return host


def _held_as(ctx, room_id: str) -> tuple[Any, ...] | None:
    """How this computer holds the room right now: host or copy, of which host, at which epoch."""
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        try:
            holding = _holding(conn, room_id)
        except SuccessionError:
            return None
    return holding["kind"], holding.get("authority"), holding.get("epoch")


def status(ctx, room_id: str) -> dict[str, Any]:
    """``groups.succession.status`` for this computer. Its parts come from several reads; when the room
    changed hands between them (a host stepping down just then), it is read again, so a host never
    shows as serving on a view that mixes before and after."""
    for _ in range(STATUS_READS):
        before = _held_as(ctx, room_id)
        result = _status(ctx, room_id)
        if _held_as(ctx, room_id) == before:
            break
    return result


def _status(ctx, room_id: str) -> dict[str, Any]:
    from gateway import hosted_room_succession_automatic as automatic
    from gateway.hosted_room_succession_move import host_bots, is_owner, placed_bots, restarting_until, work_counts
    now, me = time.time(), succession.local_install_id()
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        holding = _holding(conn, room_id)
        configuration = succession.configuration_locked(conn, room_id) if holding["kind"] in {"room", "copy"} \
            else {"configuration_seq": 0, "custodians": [], "owner_name": None}
        records = {kind: succession.load_record_locked(conn, room_id, kind) or {}
                   for kind in ("move", "conflict", "heartbeat", "return")}
        restarting = restarting_until(conn, room_id) if holding["kind"] == "copy" else None
        origin = succession.origin_locked(conn, room_id) if holding["kind"] == "room" else None
        routes = _routes(conn, room_id) if holding["kind"] == "room" else {}
        requested = succession.automatic_requested_locked(conn, room_id) if holding["kind"] == "room" else None
    custody = None
    if holding["kind"] in {"room", "copy"}:
        try:
            custody = succession.custody_status(ctx.db_path, room_id)
        except Exception:
            custody = None
    move, conflict, heartbeat, returned = (records[k] for k in ("move", "conflict", "heartbeat", "return"))
    host = _host_view(holding, configuration, heartbeat, restarting, now)
    latest = holding.get("latest_seq", 0)
    rows = backups(configuration, custody, host_id=host["install_id"], latest_seq=latest, heartbeat=heartbeat,
                   now=now, routes=routes)
    targets = ranked_targets(rows)
    owner = is_owner(ctx, room_id)
    restarting_now = restarting is not None and now < restarting + RESTART_GRACE_SECONDS
    paused = automatic.paused_view(ctx, room_id) if holding["kind"] == "room" else None
    stalled = _stalled_view(ctx, room_id, configuration, now) if holding["kind"] == "room" else None
    if returned.get("state") == "stepped_down" and holding["kind"] != "room":
        state = "moved_away"
    elif conflict.get("state") == "active":
        state = "continued_on_two"
    elif move.get("state") in {"moving", "handing_over"} or (returned.get("state") == "paused" and not stalled):
        state = "moving"
    elif holding["kind"] == "room":
        paused = paused or stalled
        state = "paused" if paused else "ok"
    elif restarting_now:
        state = "host_restarting"
    elif not host["reachable"]:
        state = "host_unreachable"
    else:
        state = "ok"
    role = {"room": "host", "copy": "backup", "member": "member"}.get(holding["kind"], "none")
    at_risk_after = (custody or {}).get("at_risk_after_seq") or 0
    result: dict[str, Any] = {
        "state": state, "host": host,
        "this_install": {"install_id": me, "name": succession.label(configuration, me), "role": role},
        "owner": {"name": succession.owner_label(configuration)},
        "backups": [{key: value for key, value in row.items() if key != "_order"} for row in rows],
        "at_risk": {"count": max(0, int(latest) - int(at_risk_after)) if custody else 0},
        "moving": None, "conflict": None, "moved": None, "work": None, "actions": [],
        "unavailable_reason": None, "previous_host": None, "unavailable_bots": [], "last_attempt": None,
        "automatic": automatic.automatic_view(configuration, rows, host_id=host["install_id"],
                                              pending=_pending(ctx, room_id, requested)),
        "paused": paused, "moved_in": None}
    if state == "moving":
        if move.get("state") in {"moving", "handing_over"}:
            to = move.get("to") if move.get("state") == "handing_over" else me
            result["moving"] = {"to": {"install_id": to, "name": succession.label(configuration, to)},
                                "step": move.get("step") or "fencing", "started_at": move.get("started_at"),
                                "reason": move.get("reason") or "manual"}
            if move.get("step") == "waiting_for_turns":
                from gateway.hosted_room_succession_handover import unsettled_turns
                result["moving"]["running"] = len(unsettled_turns(ctx.db_path, room_id))
                if owner:
                    result["actions"].append({"action": "move_now"})
        else:
            holder = (returned.get("fenced_by") or {}).get("install_id")
            result["moving"] = {"to": {"install_id": holder, "name": succession.label(configuration, holder)},
                                "step": "fencing", "started_at": returned.get("at"), "reason": "manual"}
    if state == "continued_on_two":
        winner = conflict.get("winner")
        result["conflict"] = {"hosts": [{"install_id": item["install_id"],
                                         "name": succession.label(configuration, item["install_id"]),
                                         "since": item.get("since")} for item in conflict.get("hosts") or ()],
                              "start": conflict.get("start"), "end": conflict.get("detected_at"),
                              "running_on": {"install_id": winner, "name": succession.label(configuration, winner)}
                              if winner else None}
        if owner:  # the host the group keeps running on first: keeping it is "keep going"
            result["actions"].append({"action": "keep", "targets": sorted(
                (item["install_id"] for item in conflict.get("hosts") or ()), key=lambda item: item != winner)})
    if state == "moved_away":
        result["moved"] = {"to": {"install_id": returned["successor"],
                                  "name": succession.label(configuration, returned["successor"])},
                           "at": returned.get("at"), "separate_events": returned.get("separate_events", 0),
                           "branch_id": returned.get("branch_id")}
        result["actions"].append({"action": "open_on", "target": returned["successor"]})
    if holding["kind"] == "room" and move.get("state") == "moved":
        reconciliation = move.get("reconciliation") or {}
        result["work"] = reconciliation.get("counts") or work_counts([])
        previous = move.get("previous_host")
        result["previous_host"] = {"install_id": previous, "name": succession.label(configuration, previous),
                                   "offline_since": move.get("offline_since")}
        # A Bot runs only on its own computer: it takes part again once the group moves back there.
        result["unavailable_bots"] = placed_bots(configuration, host_bots(
            _members(holding), previous, origin=origin or move.get("origin_install_id"), here=me), {
            row["install_id"] for row in rows if row["readiness"] in {"caught_up", "behind"}})
        kind = move.get("proof_kind")
        if kind in {"certified", "evidence", "handover"} and not move.get("reconciled_old_host"):
            # Until the old host is a copy again; after a careful move the owner may go back to it.
            result["moved_in"] = {"from": {"install_id": previous, "name": succession.label(configuration, previous)},
                                  "at": move.get("moved_at"), "proof_kind": kind}
            if kind == "evidence" and owner and state == "ok":
                result["actions"].append({"action": "keep", "targets": [previous]})
    if move.get("state") == "failed" and move.get("last_attempt"):
        attempt = move["last_attempt"]
        result["last_attempt"] = {"to": attempt.get("to"), "error": attempt.get("error"), "at": attempt.get("at")}
    if state == "host_unreachable":
        if not owner:
            result["unavailable_reason"] = "not_owner"
        elif not targets:
            result["unavailable_reason"] = "no_successor"
        elif all(row["readiness"] in UNAVAILABLE for row in rows if row["install_id"] in targets):
            result["unavailable_reason"] = "successor_behind_offline"
        elif not automatic.majority_reachable(configuration, rows, host_id=host["install_id"]):
            # In majority mode a reachable majority moves the group by itself; by hand only without one.
            result["actions"].append({"action": "continue", "targets": targets})
        else:
            result["unavailable_reason"] = "takeover_waiting"
            since = host.get("since")
            if since is not None and now - float(since) >= UNREACHABLE_AFTER_SECONDS + TAKEOVER_WAIT_SECONDS:
                result["actions"].append({"action": "continue", "targets": targets})
    elif state in {"ok", "host_restarting"}:
        result["unavailable_reason"] = "host_reachable"
    if state == "paused" and owner:
        # Without its lease layer, continuing here also turns automatic moves off for the group.
        result["actions"].append({"action": "continue_anyway", "turns_off_automatic": True}
                                 if (paused or {}).get("reason") == "no_lease_layer" else {"action": "continue_anyway"})
    if holding["kind"] == "room" and owner and state == "ok":
        # Computers that hold a verified copy and answered lately: the old host too, once it is back
        # as a copy, so the owner can always move the group back.
        movable = [target for target in targets if next(
            row for row in rows if row["install_id"] == target)["readiness"] in {"caught_up", "behind"}]
        if movable:
            result["actions"].append({"action": "move", "targets": movable})
        result["actions"].append({"action": "automatic", "enabled": configuration.get("automatic") is not False})
        # A switch per computer keeping a copy: on or off, with a hint where its operator didn't allow it.
        designate = [row["install_id"] for row in rows]
        if designate:
            result["actions"].append({"action": "designate", "targets": designate})
        result["actions"].append({"action": "add_backup"})
        removable = [row["install_id"] for row in rows if row["kind"] == "backup"]
        if removable:
            result["actions"].append({"action": "remove_backup", "targets": removable})
    return result


def _stalled_view(ctx, room_id: str, configuration: Mapping[str, Any], now: float) -> dict[str, Any] | None:
    """``status.paused`` for a host paused for a step another computer never took, once it has waited for
    it (``step_not_taken``): it waits to hear, from the computers that could have continued the group,
    that nothing happened, and the owner may continue it anyway."""
    from gateway.hosted_room_succession_move import stalled_promise
    stalled = stalled_promise(ctx.db_path, room_id, now=now)
    if stalled is None:
        return None
    return {"reason": "step_not_taken", "since": stalled["since"],
            "waiting_for": [{"install_id": install_id, "name": succession.label(configuration, install_id)}
                            for install_id in stalled["unconfirmed"]]}


def _pending(ctx, room_id: str, requested: bool | None) -> bool | None:
    """On the host, the owner's switch while it isn't in force yet (#104601's ``automatic_pending``)."""
    if requested is None:
        return None
    try:
        return requested if succession.automatic_pending(ctx.db_path, room_id, enabled=requested) else None
    except Exception:
        return None


def _routes(conn, room_id: str) -> dict[str, str]:
    """On the host, the state of its copy to each computer: ``needs_reauthorization`` once refused."""
    from gateway.hosted_room_replication import TARGETS_TABLE
    if not succession.table_exists(conn, TARGETS_TABLE):
        return {}
    return {str(row[0]): str(row[1]) for row in conn.execute(
        f"SELECT target_install_id, status FROM {TARGETS_TABLE} WHERE room_id=?", (room_id,))}


def _members(holding: Mapping[str, Any]) -> list[dict[str, Any]]:
    import json
    raw = holding.get("members")
    return json.loads(raw) if isinstance(raw, str) else []


# --- heartbeats, the offline window and the owner's notice ----------------------------------------
def heartbeat(ctx, room_id: str, *, now: float | None = None, survey_peers: bool = False) -> dict[str, Any]:
    """Ask the configured host once, and when it stays offline every other custodian too."""
    from gateway.hosted_room_succession_move import RemoteRefusal, _query, survey
    now = time.time() if now is None else float(now)
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        configuration = succession.configuration_locked(conn, room_id)
        record = dict(succession.load_record_locked(conn, room_id, "heartbeat") or {})
    host = succession.host_entry(configuration) or {}
    me = succession.local_install_id()
    if not host or host.get("install_id") == me:
        return record
    if record.get("host") != host["install_id"]:
        record = {"host": host["install_id"], "first_seen_at": now}
    ok = False
    try:
        if host.get("endpoint"):
            answer = _query(ctx, room_id, host["install_id"], str(host["endpoint"]))
            ok = bool(answer.get("hosting")) or isinstance(answer.get("restarting_until"), (int, float))
    except (RemoteRefusal, SuccessionError, OSError, ValueError):
        ok = False
    except Exception:
        ok = False
    record["last_attempt"] = now
    if ok:
        record.update(last_ok=now, failing=False, first_failure=None, notice_waiting_since=None)
    else:
        record.setdefault("first_failure", now)
        record["first_failure"] = record["first_failure"] or now
        reference = record.get("last_ok") or record.get("first_seen_at") or record["first_failure"]
        record["failing"] = now - float(reference) >= UNREACHABLE_AFTER_SECONDS
    if record.get("failing") and survey_peers:
        peers = dict(record.get("peers") or {})
        for install_id, outcome in survey(ctx, room_id, configuration).items():
            if install_id == host["install_id"]:
                continue
            answer = outcome.get("answer")
            peer = dict(peers.get(install_id) or {})
            peer.update(last_attempt=now, ok=answer is not None)
            if answer is not None:
                peer.update(last_ok=now, watermark=answer.get("watermark"))
            peers[install_id] = peer
        record["peers"] = peers
    succession.save_record(ctx.db_path, room_id, "heartbeat", record)
    return record


def notice_due(ctx, room_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    """The incident this computer should tell the owner about now, once; ``None`` otherwise."""
    now = time.time() if now is None else float(now)
    try:
        current = status(ctx, room_id)
    except SuccessionError:
        return None
    me = succession.local_install_id()
    record = succession.load_record(ctx.db_path, room_id, "heartbeat") or {}
    if current["state"] != "host_unreachable" or not succession.local_consent(ctx.db_path, room_id):
        return None
    targets = next((action["targets"] for action in current["actions"] if action["action"] == "continue"), None)
    if targets is None and current["automatic"]["mode"] == "majority":
        return None  # a reachable majority moves the group by itself; messaging tells the owner then
    if targets is None:
        targets = ranked_targets([{**row, "_order": index} for index, row in enumerate(current["backups"])])
    if me not in targets:
        return None
    incident = f"{current['host']['install_id']}:{current['host']['since']}"
    if record.get("notified") == incident:
        return None
    peers = record.get("peers") or {}
    better = targets[:targets.index(me)]
    reachable_better = [install_id for install_id in better if (peers.get(install_id) or {}).get("ok")]
    if better:
        waiting = record.get("notice_waiting_since")
        if waiting is None:
            succession.save_record(ctx.db_path, room_id, "heartbeat", {**record, "notice_waiting_since": now})
            return None
        if reachable_better or now - float(waiting) < NOTICE_DELAY_SECONDS:
            return None
    return {"incident": incident, "status": current}


def notice_text(*, group: str, host: str, minutes: int, here: str, hint: str | None) -> str:
    if hint is not None:
        return (f"“{group}” is paused: {host} has been offline for {minutes} min. "
                f"Reply {hint} to continue it on {here}.")
    return (f"“{group}” is paused: {host} has been offline for {minutes} min. Open Hermes Desktop on a computer "
            f"you own, or run `hermes groups continue {group}` on {here}.")


async def send_notice(runner, *, room_id: str, group: str, host: str | None, minutes: int, here: str) -> bool:
    """One notice: messaging renders it, with its own Continue button, when it offers that; otherwise
    our text goes to the owner's private chats it offers, else the home channel."""
    render = getattr(runner, "_group_chat_notify", None)
    if render is not None:
        return bool(await render(room_id, "host_offline", {"host": host, "minutes": minutes}))
    host = host or "the host"
    refs_for = getattr(runner, "_group_chat_continue_refs", None)
    refs = await refs_for(room_id) if refs_for is not None else []
    sent = False
    for adapter, chat_id, metadata, number in refs or ():
        hint = runner._typed_command_prefix_for(adapter.platform) + f"group {number} continue"
        try:
            await adapter.send(chat_id, notice_text(group=group, host=host, minutes=minutes, here=here, hint=hint),
                               metadata=metadata)
            sent = True
        except Exception:
            continue
    if refs:
        return sent
    for platform, _config, home, transport in runner._home_channel_transports():
        sent = await runner._send_home_channel_message(
            platform, home, transport, notice_text(group=group, host=host, minutes=minutes, here=here, hint=None),
            "Group Chat pause notice to %s:%s failed: %s") or sent
    return sent


def notify(ctx, room_id: str, runner, loop, *, now: float | None = None) -> bool:
    """Send the owner's notice for this incident when this computer is the one to send it."""
    due = notice_due(ctx, room_id, now=now)
    if due is None or runner is None or loop is None:
        return False
    now = time.time() if now is None else float(now)
    current = due["status"]
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        holding = _holding(conn, room_id)
    since = current["host"]["since"] or now
    future = asyncio.run_coroutine_threadsafe(send_notice(
        runner, room_id=room_id, group=holding.get("name") or "this group", host=current["host"]["name"],
        minutes=max(1, int((now - float(since)) // 60)), here=current["this_install"]["name"] or "this computer"),
        loop)
    try:
        sent = bool(future.result(timeout=30))
    except Exception:
        return False
    if sent:
        record = succession.load_record(ctx.db_path, room_id, "heartbeat") or {}
        succession.save_record(ctx.db_path, room_id, "heartbeat", {**record, "notified": due["incident"]})
    return sent


# --- the upkeep thread ------------------------------------------------------------------------------
def held_rooms(db_path: Path) -> dict[str, list[str]]:
    """Rooms this computer hosts and rooms it keeps a copy of, as succession sees them."""
    me = succession.local_install_id()
    with closing(rooms._read_connection(db_path)) as conn:
        hosted = [row[0] for row in conn.execute(
            "SELECT room_id FROM hosted_rooms WHERE authority_gateway_id=? AND disbanded_at IS NULL", (me,))]
        copies = [row[0] for row in conn.execute(
            "SELECT room_id FROM hosted_room_replicas WHERE disbanded_at IS NULL AND quarantine_reason IS NULL")] \
            if succession.table_exists(conn, "hosted_room_replicas") else []
    return {"hosted": hosted, "copies": copies}


def back_from_restart(db_path: Path, room_id: str) -> None:
    """A host running again ends the restart it announced, so backups stop waiting for it."""
    from gateway.hosted_room_succession_move import append_state, restarting_until
    with closing(rooms._read_connection(db_path)) as conn:
        announced = restarting_until(conn, room_id)
    if announced is not None:
        append_state(db_path, room_id, "ok")


class SuccessionUpkeep:
    """Background checks for the groups this computer hosts or keeps a copy of."""

    def __init__(self, context_factory, *, lease_context=None, runner=None, loop=None,
                 interval: float = UPKEEP_INTERVAL_SECONDS):
        """``context_factory`` gives this computer's full succession context (its fence store and its
        endpoint), or None; ``lease_context`` what holding its own groups' leases needs (its room store),
        which a host without a fence store or an endpoint still has."""
        from gateway.hosted_room_succession_automatic import Automatic
        self._context_factory, self._runner, self._loop = context_factory, runner, loop
        self._lease_context = lease_context or context_factory
        self._interval, self._stop = float(interval), threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_beat: dict[str, float] = {}
        self._restored = False
        self._installed = False
        self.suspect: set[str] = set()
        self.automatic = Automatic(self._lease_context, wake=self.wakeup)

    def _ensure_installed(self) -> bool:
        """Register the lease layer with #104601's hooks as soon as there is a room store to hold."""
        from gateway.hosted_room_succession_automatic import install
        if self._installed:
            return True
        try:
            if self._lease_context() is None:
                return False
            install(self.automatic)
        except Exception:
            return False  # tried again next tick; meanwhile its automatic groups stay paused
        self._installed = True
        return True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ensure_installed()
        self._thread = threading.Thread(target=self._run, name="group-succession-upkeep", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        from gateway.hosted_room_succession_automatic import uninstall
        self._stop.set()
        self._wake.set()
        uninstall(self.automatic)
        self._installed = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def wakeup(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        """The full upkeep every interval or when woken; automatic moves' fast checks in between."""
        from gateway.hosted_room_succession_automatic import TICK_SECONDS
        next_full = 0.0
        while not self._stop.is_set():
            woken = self._wake.is_set()
            self._wake.clear()
            try:
                if woken or time.monotonic() >= next_full:
                    next_full = time.monotonic() + self._interval
                    self.run_once()
                self.run_automatic()
            except Exception:
                pass  # upkeep never decides a room's authority by failing; the next cycle asks again
            self._wake.wait(TICK_SECONDS)

    def run_automatic(self) -> list[dict[str, Any]]:
        """Pause a host without its lease or cut off, and take over where this computer is the standby due."""
        if not self._ensure_installed():
            return []
        ctx = self._lease_context()
        if ctx is None:
            return []
        held = held_rooms(ctx.db_path)
        if not self._restored:
            self.automatic.restore(ctx, held["hosted"])
            self._restored = True
        return self.automatic.tick(ctx, held, standby=self._context_factory())

    def run_once(self, *, now: float | None = None) -> None:
        from gateway import hosted_room_succession_move as move
        from gateway import hosted_room_succession_return as returning
        ctx = self._context_factory()
        if ctx is None:
            return
        now = time.time() if now is None else float(now)
        held = held_rooms(ctx.db_path)
        returning.prune_branches(ctx.db_path)  # set-aside messages go with their disbanded or retired group
        for room_id in held["hosted"]:
            back_from_restart(ctx.db_path, room_id)
        move.maintain(ctx, held["hosted"])
        returning.maintain(ctx, held["hosted"], suspect=frozenset(self.suspect))
        self.suspect.clear()
        for room_id in held["copies"]:
            try:
                returning.follow_up(ctx, room_id)
                move.deliver_decision(ctx, room_id)
                if now - self._last_beat.get(room_id, 0.0) >= HEARTBEAT_INTERVAL_SECONDS:
                    self._last_beat[room_id] = now
                    record = heartbeat(ctx, room_id, now=now, survey_peers=True)
                    if record.get("failing"):
                        notify(ctx, room_id, self._runner, self._loop, now=now)
            except Exception:
                continue
