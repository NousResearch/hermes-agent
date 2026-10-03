"""Continuing a Group Chat on this computer: preview, continue, and keep after a conflict.

``preview`` checks, without changing anything, what continuing here would mean: how far this
copy is behind the most complete reachable one, which work is finished, still running elsewhere,
unknown or waiting for the old host, and which Bots stay unavailable. ``continue_here`` then runs
the owner's decision: it fences the host's epoch at every reachable backup, adopts the most
complete copy, writes the marked transition, records the next configuration, reconciles accepted
work, takes ownership, registers the members' continuation routes, and announces the move. Each
step is recorded, so a crash resumes instead of repeating one, and a failed attempt keeps its
reason.

When two computers hosted the group at once (a partition and two moves), the first contact between
them records ``continued_on_two``. The group keeps running on one of them by rule: the higher
epoch, and on a tie ``certified`` before ``evidence`` before ``attested``, then the lower
installation id. That host keeps serving (on a tie it continues at a fresh epoch every computer can
follow), every copy follows it, and the other host stops serving with its own messages kept apart
until the owner chooses. ``keep`` from either side: keeping the running host steps the other one
down to a copy, its messages set aside; keeping the other one switches, at a fresh epoch, and the
running host steps aside. The owner's choice is signed by the computer it was made on.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway import hosted_room_succession as succession
from gateway import hosted_rooms as rooms
from gateway.hosted_room_succession import ProofInvalid, SuccessionError

REQUEST_TIMEOUT_SECONDS = 5.0
# How far past every epoch this computer itself knows a rival's signed claim may move the next one.
MAX_EPOCH_STEP = 64
# Refusals whose detail names a holder or an epoch: trusted only when signed by the computer that refused.
_CLAIMS = frozenset({"room_authority_promised", "room_authority_fenced", "room_authority_superseded",
                     "host_reachable", "room_lease_active", "configuration_stale"})
CATCH_UP_PAGE = 200
RESTART_GRACE_SECONDS = 120.0
_TERMINAL_RUNS = frozenset({"completed", "failed", "cancelled", "interrupted"})
_ACTIVE_RUNS = frozenset({"queued", "running", "waiting_for_approval", "stopping"})


@dataclass
class MoveContext:
    """This computer, the caller asking it to act, and the transport to the group's other computers."""
    db_path: Path
    runs_store: Any
    service: Any = None
    actor_subject: str | None = None
    operator: bool = False
    mint_grants: Callable[[str, str, int], list[dict[str, Any]]] | None = None
    post: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None = None
    fetch_pages: Callable[..., dict[str, Any]] | None = None
    timeout: float = REQUEST_TIMEOUT_SECONDS
    workers: int = 8

    def backup_context(self):
        from gateway.hosted_room_succession_backup import BackupContext
        return BackupContext(custody_db=self.db_path, runs_store=self.runs_store, mint_grants=self.mint_grants,
                             service=self.service, post=self.post, fetch_pages=self.fetch_pages)


class RemoteRefusal(Exception):
    def __init__(self, code: str, detail: Mapping[str, Any] | None):
        super().__init__(code)
        self.code, self.detail = code, dict(detail or {})


def http_post(endpoint: str, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """JSON POST to another computer of the group; its refusals come back as ``RemoteRefusal``."""
    from gateway.hosted_room_peer import validate_room_link_url
    from tui_gateway.hosted_room_peer_http import _open_roomlink_url
    base, _ = validate_room_link_url(endpoint)
    request = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "Hermes-RoomLink/1.0"})
    try:
        with _open_roomlink_url(request, timeout=timeout, reject_redirects=True) as response:
            return json.loads(response.read(4 * 1024 * 1024))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RemoteRefusal("succession_unsupported", None) from exc
        try:
            error = json.loads(exc.read(256 * 1024)).get("error") or {}
        except (ValueError, AttributeError):
            error = {}
        raise RemoteRefusal(str(error.get("code") or f"http_{exc.code}"), error.get("detail")) from exc


def _read(ctx: MoveContext):
    return closing(rooms._read_connection(ctx.db_path))


def _save(ctx: MoveContext, room_id: str, kind: str, record: Mapping[str, Any]) -> None:
    succession.save_record(ctx.db_path, room_id, kind, record)


def named(configuration: Mapping[str, Any], install_id: str | None) -> dict[str, Any]:
    return {"install_id": install_id, "name": succession.label(configuration, install_id)}


def view(ctx: MoveContext, room_id: str) -> dict[str, Any]:
    """This computer's copy: its head, configuration, watermark and the room's original home."""
    from gateway.hosted_room_succession_backup import copy_head_locked
    with _read(ctx) as conn:
        head = copy_head_locked(conn, room_id)
        configuration = succession.configuration_locked(conn, room_id)
        return {"head": head, "configuration": configuration, "watermark": succession.watermark_locked(conn, room_id),
                "origin": succession.origin_locked(conn, room_id)
                or succession.log_origin_locked(conn, room_id, head["authority_gateway_id"]),
                "move": succession.load_record_locked(conn, room_id, "move")}


def is_owner(ctx: MoveContext, room_id: str) -> bool:
    with _read(ctx) as conn:
        return succession.is_owner_locked(conn, room_id, subject=ctx.actor_subject, operator=ctx.operator)


def _require_target(ctx: MoveContext, room_id: str, target_install_id: Any, configuration) -> None:
    """Continuing runs on the target itself, for the room's owner, while this computer still consents."""
    me = succession.local_install_id()
    if target_install_id != me:
        raise SuccessionError("ask the computer the group should continue on", reason="target_not_local",
                              detail={"target": named(configuration, target_install_id if isinstance(
                                  target_install_id, str) else None)})
    if not is_owner(ctx, room_id):
        raise SuccessionError("only the group's owner can continue it here", reason="not_owner")
    if not succession.is_eligible(configuration, me) or not succession.local_consent(ctx.db_path, room_id):
        raise SuccessionError("this computer may not continue the group", reason="target_not_ready")


# --- asking the other computers ---------------------------------------------------------------------
def _query(ctx: MoveContext, room_id: str, install_id: str, endpoint: str, *, from_epoch: int | None = None):
    unsigned = {"room_id": room_id, "requester_install_id": succession.local_install_id(), "issued_at": time.time(),
                "nonce": succession.nonce(), **({"from_epoch": from_epoch} if from_epoch is not None else {})}
    request = {**unsigned, "signature": succession.sign(succession.QUERY, unsigned)}
    answer = (ctx.post or http_post)(endpoint, "/v1/room-members/succession/query", request, ctx.timeout)
    signed = {k: v for k, v in answer.items() if k != "signature"}
    with _read(ctx) as conn:
        if (answer.get("responder_install_id") != install_id or answer.get("request_digest") != succession.digest(request)
                or not succession.verify_locked(conn, room_id, install_id, succession.ANSWER, signed,
                                                answer.get("signature"))):
            raise ProofInvalid("an answer was not signed by the computer asked")
    return answer


def survey(ctx: MoveContext, room_id: str, configuration: Mapping[str, Any], *,
           from_epoch: int | None = None) -> dict[str, Any]:
    """Ask every other computer of the group, in parallel, who it follows and how far it holds."""
    me = succession.local_install_id()
    custodians = {k: v for k, v in succession.custodians_by_id(configuration).items() if k != me}

    def one(install_id):
        endpoint = custodians[install_id].get("endpoint")
        if not endpoint:
            return install_id, {"error": "unreachable"}
        try:
            return install_id, {"answer": _query(ctx, room_id, install_id, str(endpoint), from_epoch=from_epoch)}
        except RemoteRefusal as exc:
            return install_id, {"error": exc.code}
        except Exception:
            return install_id, {"error": "unreachable"}

    if not custodians:
        return {}
    with ThreadPoolExecutor(max_workers=min(ctx.workers, len(custodians))) as pool:
        return dict(pool.map(one, sorted(custodians)))


# --- work in progress ---------------------------------------------------------------------------
def policy_profiles(room: Mapping[str, Any]) -> tuple[str, ...]:
    """The profiles a room's policy treats as local: those of the room's own original home.

    A Bot local to that home stays its Bot after the room moves: a successor validates the
    roster against it and never runs it as one of its own profiles.
    """
    return tuple(sorted({str(m.get("profile")) for m in room.get("members") or ()
                         if not isinstance(m.get("target"), Mapping) or m["target"].get("kind", "local") == "local"}))


def host_bots(members: list[Mapping[str, Any]], host_install_id: str | None, *, origin: str | None = None,
              here: str | None = None) -> list[dict[str, Any]]:
    """Members whose Bot runs on another computer than ``here`` once the group left ``host_install_id``.

    Each names the computer it runs on (``on``): a local Bot runs on the room's original home
    (``origin``, the old host when unknown) and nowhere else, and the old host's own peer members
    run on it. A Bot on ``here`` is available, so a group moved back to its home has them all again.
    """
    home = origin or host_install_id
    found = []
    for member in members or ():
        if member.get("member_id") == succession.custody_member_id():
            continue
        target = member.get("target") if isinstance(member.get("target"), Mapping) else {}
        if target.get("kind", "local") == "local":
            on = home
        elif target.get("installation_id") == host_install_id:
            on = host_install_id
        else:
            continue
        if on is None or on == here:
            continue
        found.append({"member_id": member.get("member_id"), "on": on,
                      "name": member.get("display_name") or member.get("handle") or member.get("member_id")})
    return found


def placed_bots(configuration: Mapping[str, Any], bots: list[Mapping[str, Any]],
                reachable: set[str]) -> list[dict[str, Any]]:
    """``unavailable_bots``: each Bot with the computer it runs on and whether that computer answers now,
    so a client can offer to move the group back there."""
    return [{"member_id": bot["member_id"], "name": bot["name"],
             "on": {"install_id": bot["on"], "name": succession.label(configuration, bot["on"]),
                    "reachable": bot["on"] in reachable}} for bot in bots]


def _room_events(db_path: Path, room_id: str, table: str) -> list[dict[str, Any]]:
    with closing(rooms._read_connection(db_path)) as conn:
        return [succession.event_dict(room_id, row) for row in conn.execute(
            f"""SELECT seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at FROM {table}
                WHERE room_id=? ORDER BY seq""", (room_id,))]


def pending_admissions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Admissions whose latest generation has no published outcome in this log."""
    latest: dict[str, dict[str, Any]] = {}
    settled, deferred = set(), set()
    for event in events:
        payload = event["payload"]
        if event["kind"] == "task.admitted":
            task_id = str(payload["task"]["task_id"])
            if int(payload["execution_generation"]) >= int(latest.get(task_id, {}).get("execution_generation", 0)):
                latest[task_id] = payload
        elif event["kind"] in {"turn.settled", "turn.failed", "turn.cancelled"}:
            settled.add(str(payload.get("task_id")))
        elif event["kind"] == "turn.deferred":
            deferred.add((str(payload.get("task_id")), payload.get("execution_generation")))
    return [payload for task_id, payload in latest.items()
            if task_id not in settled and (task_id, payload["execution_generation"]) not in deferred
            and payload.get("target_member_id") != succession.custody_member_id()]


def evidence_index(evidence_by_install: Mapping[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    index = {}
    for install_id, evidence in (evidence_by_install or {}).items():
        for run in (evidence or {}).get("runs", ()):
            if run.get("target_install_id") == install_id:
                index[(run["task_id"], int(run["execution_generation"]), install_id)] = run
    return index


def classify(admission: Mapping[str, Any], evidence: Mapping[tuple[str, int, str], Any], *,
             host_install_id: str | None, host_members: set[str]) -> tuple[str, dict[str, Any] | None]:
    """completed / elsewhere (still running on another computer) / unknown.

    A turn admitted for a Bot on the old host may have run there before it went offline: it is
    unknown, kept with its identity and never run again by itself (only new turns for that Bot wait
    for the host to return)."""
    target = str(admission["target_install_id"])
    if target == host_install_id or admission.get("target_member_id") in host_members:
        return "unknown", None
    run = evidence.get((admission["task"]["task_id"], int(admission["execution_generation"]), target))
    if run is None or run["status"] not in _TERMINAL_RUNS | _ACTIVE_RUNS:
        return "unknown", run
    return ("completed" if run["status"] in _TERMINAL_RUNS else "elsewhere"), run


def work_counts(entries: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"completed": 0, "elsewhere": 0, "unknown": 0, "waiting_for_host": 0}
    for entry in entries:
        counts[entry["state"]] += 1
    return counts


# --- preview ----------------------------------------------------------------------------------------
def preview(ctx: MoveContext, room_id: str, target_install_id: Any) -> dict[str, Any]:
    """What continuing on this computer would mean; changes nothing but the stored preview."""
    current = view(ctx, room_id)
    configuration, head = current["configuration"], current["head"]
    _require_target(ctx, room_id, target_install_id, configuration)
    if head["authoritative"]:
        raise SuccessionError("this computer already hosts the group", reason="host_reachable")
    snapshot = _snapshot(ctx, room_id, current)
    record = {**(current["move"] or {}), "state": (current["move"] or {}).get("state") or "previewed",
              "preview_id": snapshot["preview_id"], "preview": snapshot["result"], "previewed_at": time.time()}
    if record["state"] == "moving":
        raise SuccessionError("this group is already moving here", reason="preview_stale")
    _save(ctx, room_id, "move", {**record, "state": "previewed"})
    return snapshot["result"]


def _snapshot(ctx: MoveContext, room_id: str, current: Mapping[str, Any]) -> dict[str, Any]:
    from gateway.hosted_room_succession_status import RESTART_GRACE_SECONDS as grace
    configuration, head, mark = current["configuration"], current["head"], current["watermark"]
    me = succession.local_install_id()
    host = succession.host_entry(configuration) or {"install_id": head["authority_gateway_id"]}
    answers = survey(ctx, room_id, configuration)
    for install_id, outcome in answers.items():
        answer = outcome.get("answer") or {}
        restarting = answer.get("restarting_until")
        if install_id == host["install_id"] and (answer.get("hosting") or (
                isinstance(restarting, (int, float)) and time.time() < restarting + grace)):
            raise SuccessionError("the group's host can be reached", reason="host_reachable")
        if answer.get("hosting") and int(answer["authority"]["epoch"]) > head["authority_epoch"]:
            raise SuccessionError("the group already continues on another computer", reason="room_authority_promised",
                                  detail={"other": named(configuration, answer["responder_install_id"])})
    # The most complete copy, counted only as far as heads the host signed vouch for it (the check the
    # move's adoption makes): a computer's own word about its copy changes nothing here.
    with _read(ctx) as conn:
        vouched = succession.vouched_receipts_locked(conn, room_id, [
            {"custodian_install_id": install_id, "heads": outcome["answer"].get("heads")}
            for install_id, outcome in answers.items() if outcome.get("answer")],
            host=head["authority_gateway_id"], epoch=int(head["authority_epoch"]))
    best_id, best_seq = me, int(mark["seq"])
    for item in sorted(vouched, key=lambda item: item["install_id"]):
        if item["seq"] > best_seq:
            best_id, best_seq = item["install_id"], item["seq"]
    # What the host told this computer it had: the part no vouched copy holds is at risk.
    announced = int(head.get("announced_seq", head["latest_seq"]))
    room_members = _members(ctx, room_id)
    elsewhere = host_bots(room_members, host["install_id"], origin=current.get("origin"), here=me)
    host_members = {bot["member_id"] for bot in elsewhere}
    evidence = {k: v["answer"].get("run_evidence") for k, v in answers.items() if v.get("answer")}
    evidence[me] = ctx.runs_store.room_run_evidence(room_id, through_epoch=head["authority_epoch"],
                                                    limit=succession.MAX_RUN_EVIDENCE)
    index = evidence_index(evidence)
    entries = [{"task_id": admission["task"]["task_id"], "state": classify(
        admission, index, host_install_id=host["install_id"], host_members=host_members)[0]}
        for admission in pending_admissions(_room_events(ctx.db_path, room_id, "hosted_room_replica_events"))]
    unfenced = sorted(k for k, v in answers.items() if v.get("error") == "succession_unsupported")
    cautions: list[dict[str, Any]] = [{"code": "host_may_be_running"}]
    if unfenced:
        names = [name for name in (succession.label(configuration, k) for k in unfenced) if name]
        cautions.append({"code": "participant_not_fenced", "names": names, "count": len(unfenced)})
    from gateway.hosted_room_succession_automatic import mode_of
    silent = [voter for voter in succession.voters_of(configuration)
              if voter not in {me, host["install_id"]} and not (answers.get(voter) or {}).get("answer")]
    if mode_of(configuration) == "majority" and silent:
        # By hand in majority mode only without a majority: "only if they are really offline".
        names = [name for name in (succession.label(configuration, k) for k in silent) if name]
        cautions.append({"code": "voters_unreachable", "names": names, "count": len(silent)})
    result = {"target": {**named(configuration, me),
                         "operator_name": succession.label(configuration, me, "operator_name")},
              "owner": {"name": succession.owner_label(configuration)},
              "behind_by": max(0, best_seq - int(mark["seq"])),
              "at_risk": {"count": max(0, announced - best_seq)},
              "work": work_counts(entries),
              "unavailable_bots": placed_bots(configuration, elsewhere, {
                  install_id for install_id, outcome in answers.items() if outcome.get("answer")}),
              "cautions": cautions}
    basis = {"room_id": room_id, "watermark": mark, "best": [best_id, best_seq], "announced": announced,
             "configuration_seq": configuration.get("configuration_seq"),
             "reachable": sorted(k for k, v in answers.items() if v.get("answer")), "entries": entries,
             "unfenced": unfenced}
    preview_id = "preview_" + succession.digest(basis)[:32]
    return {"preview_id": preview_id, "result": {"preview_id": preview_id, **result}, "best": [best_id, best_seq]}


def _members(ctx: MoveContext, room_id: str) -> list[dict[str, Any]]:
    with _read(ctx) as conn:
        row = conn.execute("SELECT members_json FROM hosted_rooms WHERE room_id=? UNION ALL "
                           "SELECT members_json FROM hosted_room_replicas WHERE room_id=?", (room_id, room_id)).fetchone()
    return json.loads(row[0]) if row is not None else []


# --- continuing here ------------------------------------------------------------------------------
def continue_here(ctx: MoveContext, room_id: str, target_install_id: Any, *, preview_id: Any,
                  confirm: Any) -> dict[str, Any]:
    """Run the owner's decision to continue the group on this computer, step by step.

    The preview must still describe the evidence (else ``preview_stale``). Any reachable backup
    that already promised the epoch to another computer aborts the attempt with
    ``room_authority_promised`` naming it; a later retry by the operator moves to a later epoch.
    """
    current = view(ctx, room_id)
    record = committed_transition(ctx, room_id, dict(current["move"] or {}))
    if record.get("state") == "moving" and record.get("transition_committed"):
        return finish(ctx, room_id, record)
    configuration, head = current["configuration"], current["head"]
    if record.get("state") == "moved" and head["authoritative"] and target_install_id == succession.local_install_id():
        from gateway.hosted_room_succession_status import status
        return status(ctx, room_id)  # a repeated confirmation of a finished move changes nothing
    _require_target(ctx, room_id, target_install_id, configuration)
    if confirm is not True or not isinstance(preview_id, str):
        raise SuccessionError("confirm the preview to continue", reason="invalid_params")
    if head["authoritative"]:
        raise SuccessionError("this computer already hosts the group", reason="host_reachable")
    try:
        snapshot = _snapshot(ctx, room_id, current)
        if snapshot["preview_id"] != preview_id or record.get("preview_id") != preview_id:
            raise SuccessionError("the evidence changed since the preview", reason="preview_stale")
        record = {**record, "state": "moving", "step": "fencing", "started_at": time.time(),
                  "from_epoch": head["authority_epoch"], "previous_host": head["authority_gateway_id"],
                  "origin_install_id": current["origin"], "transition_committed": False,
                  "owner_subject": ctx.actor_subject}
        _save(ctx, room_id, "move", record)
        record = _fence_everywhere(ctx, room_id, current, record)
        record = _adopt(ctx, room_id, record)
        record = _transition(ctx, room_id, record)
    except SuccessionError as exc:
        _fail(ctx, room_id, record, exc)
        raise
    return finish(ctx, room_id, record)


def committed_transition(ctx: MoveContext, room_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """A move whose transition is already in this computer's log, though its record, saved just after,
    may not say so (a crash in between): the log decides, and the move resumes from finishing."""
    record = dict(record)
    if record.get("state") != "moving" or record.get("transition_committed") or "to_epoch" not in record:
        return record
    me = succession.local_install_id()
    with _read(ctx) as conn:
        room = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                            "AND disbanded_at IS NULL", (room_id,)).fetchone()
        mine = succession.latest_transition_locked(conn, room_id) if room is not None else None
    if room is None or room[0] != me or mine is None:
        return record
    payload = mine["event"]["payload"]
    if payload.get("successor_gateway_id") != me or int(payload.get("to_epoch") or 0) != int(record["to_epoch"]):
        return record
    record = {**record, "transition_committed": True, "proof_digest": payload.get("proof_digest"),
              "offline_since": record.get("offline_since", payload.get("offline_since"))}
    _save(ctx, room_id, "move", record)
    return record


def _fail(ctx: MoveContext, room_id: str, record: Mapping[str, Any], exc: SuccessionError) -> None:
    """Keep the reason of a failed attempt until the next one, so a reload still shows it."""
    configuration = view(ctx, room_id)["configuration"]
    failed = {**{k: v for k, v in record.items() if k not in {"step"}}, "state": "failed",
              "last_attempt": {"to": named(configuration, succession.local_install_id()), "error": exc.reason,
                               "at": time.time(), **({"detail": exc.detail} if exc.detail else {})}}
    if exc.reason == "room_authority_promised":
        failed["rival_epoch"] = max(int(record.get("rival_epoch") or 0), int(record.get("to_epoch") or 0),
                                    int(exc.detail.get("epoch") or 0))
    _save(ctx, room_id, "move", failed)


def _fence_request(room_id: str, epoch: int, watermark: Mapping[str, Any], *,
                   handover: Mapping[str, Any] | None = None, release: Mapping[str, Any] | None = None,
                   vote: Mapping[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    private, public = succession.reply_keypair()
    unsigned = {"room_id": room_id, "fence_epoch": epoch - 1, "promise_epoch": epoch,
                "candidate_install_id": succession.local_install_id(), "candidate_watermark": dict(watermark),
                "reply_key": public, "issued_at": time.time(), "nonce": succession.nonce(),
                **({"handover": dict(handover)} if handover is not None else {}),
                **({"release": dict(release)} if handover is not None and release is not None else {}),
                **({"vote": dict(vote)} if vote is not None else {})}
    return private, {**unsigned, "signature": succession.sign(succession.FENCE_REQUEST, unsigned)}


def _ask_fences(ctx: MoveContext, room_id: str, configuration: Mapping[str, Any], epoch: int,
                watermark: Mapping[str, Any], *, handover: Mapping[str, Any] | None = None,
                release: Mapping[str, Any] | None = None, vote: bool = False) -> dict[str, Any]:
    """Every configured computer's sealed receipt for ``epoch``, or the reason it gave none. With the
    host's signed ``handover`` and its ``release`` token, each computer gives back a lease the host
    asked for before it signed, before it promises. A ``vote`` (an automatic move) cites this
    computer's configuration: a voter that holds a later one refuses."""
    from gateway.hosted_room_succession_backup import answer_fence, open_fence_reply
    me = succession.local_install_id()
    private, request = _fence_request(
        room_id, epoch, watermark, handover=handover, release=release,
        vote={"configuration_seq": int(configuration.get("configuration_seq") or 0)} if vote else None)
    custodians = succession.custodians_by_id(configuration)
    host = (succession.host_entry(configuration) or {}).get("install_id")

    def heard(install_id):
        if install_id == host and install_id != me:  # the host answered: contact, whatever it said
            from gateway.hosted_room_succession_automatic import heard_from_host
            heard_from_host(ctx.db_path, room_id)

    def one(install_id):
        try:
            if install_id == me:
                reply = answer_fence(ctx.backup_context(), dict(request))
            elif not custodians[install_id].get("endpoint"):
                return install_id, RemoteRefusal("unreachable", None)
            else:
                reply = (ctx.post or http_post)(str(custodians[install_id]["endpoint"]),
                                                "/v1/room-members/succession/fence", dict(request), ctx.timeout)
                heard(install_id)
            with _read(ctx) as conn:
                return install_id, open_fence_reply(conn, room_id, reply, private_key=private, request=request,
                                                    to_epoch=epoch, successor=me)
        except SuccessionError as exc:
            return install_id, RemoteRefusal(exc.reason, exc.detail)
        except RemoteRefusal as exc:
            heard(install_id)
            return install_id, _verified_refusal(ctx, room_id, install_id, exc)
        except Exception:
            return install_id, RemoteRefusal("unreachable", None)

    with ThreadPoolExecutor(max_workers=min(ctx.workers, max(1, len(custodians)))) as pool:
        return dict(pool.map(one, sorted(custodians)))


def _verified_refusal(ctx: MoveContext, room_id: str, install_id: str, refusal: RemoteRefusal) -> RemoteRefusal:
    """Another computer's refusal, its claims kept only when it signed them with its pinned key: an
    unsigned or forged one counts as no answer, so it neither stops a move nor picks its epoch."""
    if refusal.code not in _CLAIMS:
        return refusal
    detail = refusal.detail if isinstance(refusal.detail, Mapping) else {}
    signed = {key: value for key, value in detail.items() if key != "signature"}
    with _read(ctx) as conn:
        genuine = (detail.get("custodian_install_id") == install_id and detail.get("room_id") == room_id
                   and succession.verify_locked(conn, room_id, install_id, succession.ANSWER, signed,
                                                detail.get("signature")))
    return refusal if genuine else RemoteRefusal("unverified", None)


def _refused_for_other(configuration: Mapping[str, Any], outcomes: Mapping[str, Any]) -> None:
    for outcome in outcomes.values():
        if not isinstance(outcome, RemoteRefusal):
            continue
        if outcome.code == "host_reachable":
            raise SuccessionError("the group's host can be reached", reason="host_reachable")
        if outcome.code in {"room_authority_promised", "room_authority_fenced", "room_authority_superseded"}:
            detail = outcome.detail
            promise, authority = detail.get("promise") or {}, detail.get("authority") or {}
            other = (promise.get("candidate_install_id") or authority.get("install_id")
                     or (detail.get("live_host") or {}).get("install_id"))
            epoch = max(int(promise.get("epoch") or 0), int(authority.get("epoch") or 0),
                        int(detail.get("fenced_epoch") or 0))
            raise SuccessionError("another computer holds this step of the group", reason="room_authority_promised",
                                  detail={"other": named(configuration, other), "epoch": epoch})


def next_epoch(ctx: MoveContext, room_id: str, current: Mapping[str, Any], record: Mapping[str, Any]) -> int:
    """The next epoch above every one known to be held; this computer's own earlier promise is resumed."""
    from gateway import hosted_room_fence as fence
    me = succession.local_install_id()
    own = fence.room_fence_state(ctx.runs_store.path, room_id)
    known = max(current["head"]["authority_epoch"], int(own["fenced_epoch"] or 0),
                int((own.get("authority") or {}).get("epoch") or 0))
    # A rival's signed claim moves the next epoch past it, never further than a bounded step.
    base = max(known, min(int(record.get("rival_epoch") or 0), known + MAX_EPOCH_STEP))
    mine = own["promise"] if own["promise"] and own["promise"]["candidate_install_id"] == me else None
    return mine["epoch"] if mine is not None and mine["epoch"] > base else base + 1


def record_fences(ctx: MoveContext, room_id: str, record: Mapping[str, Any], outcomes: Mapping[str, Any],
                  epoch: int) -> dict[str, Any]:
    """Keep the receipts, run evidence and continuation grants a fence round brought."""
    fenced = {k: v for k, v in outcomes.items() if not isinstance(v, RemoteRefusal)}
    record = {**record, "step": "catching_up", "to_epoch": epoch,
              "receipts": [fenced[k]["receipt"] for k in sorted(fenced)],
              "evidence": {k: v["run_evidence"] for k, v in fenced.items()},
              "grants": {k: v["continuation_grants"] for k, v in fenced.items() if v["continuation_grants"]},
              "unreachable": sorted(k for k, v in outcomes.items() if isinstance(v, RemoteRefusal)),
              "not_fenced": sorted(k for k, v in outcomes.items()
                                   if isinstance(v, RemoteRefusal) and v.code == "succession_unsupported")}
    _save(ctx, room_id, "move", record)
    return record


def _fence_everywhere(ctx: MoveContext, room_id: str, current: Mapping[str, Any],
                      record: dict[str, Any]) -> dict[str, Any]:
    """Fence the host's epoch at this computer and every reachable backup; receipts, not votes."""
    configuration, me = current["configuration"], succession.local_install_id()
    epoch = next_epoch(ctx, room_id, current, record)
    outcomes = _ask_fences(ctx, room_id, configuration, epoch, current["watermark"])
    _refused_for_other(configuration, outcomes)
    if me not in outcomes or isinstance(outcomes[me], RemoteRefusal):
        raise SuccessionError("this computer could not fence its own copy", reason="target_not_ready")
    return record_fences(ctx, room_id, record, outcomes, epoch)


def catch_up(ctx: MoveContext, room_id: str, source: str, head: Mapping[str, Any] | None = None, *,
             signed_by: str | None = None) -> dict[str, Any]:
    """Catch this copy up from ``source``, exactly as far as a head its host signed vouches (#104601);
    without ``head``, as far as the one the source relays. With ``signed_by``, only a head that host
    signed is followed: another computer's copy can't lead this one onto a lineage it didn't choose."""
    from gateway.hosted_room_custody import catch_up_from_custodian, fetch_custodian_pages
    fetch = ctx.fetch_pages or (lambda path, **request: fetch_custodian_pages(path, timeout=ctx.timeout, **request))
    if head is None and signed_by is not None:
        with _read(ctx) as conn:
            mark = succession.watermark_locked(conn, room_id)
        head = fetch(ctx.db_path, room_id=room_id, source_install_id=source, after_seq=mark["seq"], limit=1).get("head")
        if not isinstance(head, Mapping) or head.get("host") != signed_by:
            raise SuccessionError("that computer holds no history the kept host signed", reason="target_not_ready")
    return catch_up_from_custodian(ctx.db_path, room_id=room_id, source_install_id=source, head=head,
                                   page_limit=CATCH_UP_PAGE, timeout=ctx.timeout,
                                   _verify_transition=succession.verify_transition_locked, _fetch=fetch)


def _adopt(ctx: MoveContext, room_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Catch up to the most complete fenced copy its host vouches for, and only that far.

    A receipt's own watermark is the custodian's word; the heads it carries are the old host's. So
    the copy adopted is the one with the highest head the old host signed for its epoch, and catching
    up stops at that head: nobody can add history, keys or voters past what the host signed.
    """
    me = succession.local_install_id()
    with _read(ctx) as conn:
        mark = succession.watermark_locked(conn, room_id)
        vouched = succession.vouched_receipts_locked(conn, room_id, record["receipts"], host=record["previous_host"],
                                                     epoch=int(record["from_epoch"]))
    # Best first; on a tie this computer's own copy, which needs nothing fetched.
    ranked = sorted(vouched, key=lambda item: (item["seq"], item["install_id"] == me), reverse=True)
    source = me
    for item in ranked:
        if item["seq"] <= mark["seq"]:
            break
        try:
            catch_up(ctx, room_id, item["install_id"], item["head"])
        except Exception:
            continue  # the next best copy; what it can't vouch for is never adopted
        source = item["install_id"]
        break
    with _read(ctx) as conn:
        mark = succession.watermark_locked(conn, room_id)
    record = {**record, "step": "reconciling", "adopted_watermark": dict(mark), "adopted_from": source}
    _save(ctx, room_id, "move", record)
    return record


def host_last_seen(db_path: Path, room_id: str) -> float | None:
    """When this computer last heard from the group's host, from its heartbeat record."""
    heartbeat = succession.load_record(db_path, room_id, "heartbeat") or {}
    seen = heartbeat.get("last_ok")
    return float(seen) if isinstance(seen, (int, float)) and seen >= 0 else None


def _transition(ctx: MoveContext, room_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Write the marked transition with the owner's attestation; the room is unowned until finished."""
    from gateway import hosted_room_replicas as replicas
    configuration, me = view(ctx, room_id)["configuration"], succession.local_install_id()
    proof = succession.build_attestation(
        room_id=room_id, from_epoch=record["from_epoch"], to_epoch=record["to_epoch"], successor=me,
        previous_authority=record["previous_host"], origin_install_id=record["origin_install_id"],
        configuration_seq=int(configuration.get("configuration_seq") or 0),
        receipts=record["receipts"], unreachable=record["unreachable"],
        confirmed_by=ctx.actor_subject or "operator", preview_id=record["preview_id"])
    with _read(ctx) as conn:
        succession.verify_proof_locked(conn, room_id, proof_kind="attested", proof=proof,
                                       from_epoch=record["from_epoch"], to_epoch=record["to_epoch"], successor=me,
                                       fork_seq=record["adopted_watermark"]["seq"])
    offline_since = host_last_seen(ctx.db_path, room_id)
    at_risk = int(((record.get("preview") or {}).get("at_risk") or {}).get("count") or 0)
    replicas.promote_replica(
        ctx.db_path, room_id=room_id, transition=succession.transition_for(proof), to_epoch=record["to_epoch"],
        text=succession.transition_text(succession.label(configuration, me)),
        display={"from_name": succession.label(configuration, record["previous_host"]),
                 "to_name": succession.label(configuration, me), "offline_since": offline_since,
                 "reason": "manual", "at_risk": at_risk})
    record = {**record, "transition_committed": True, "proof_digest": succession.proof_digest(proof),
              "offline_since": offline_since}
    _save(ctx, room_id, "move", record)
    return record


def finish(ctx: MoveContext, room_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """After the transition: configure, reconcile, own, route, wake and announce. Each phase is idempotent."""
    from gateway import hosted_room_fence as fence
    me = succession.local_install_id()
    if not record.get("reconfigured"):
        with rooms._transaction(ctx.db_path, immediate=True) as conn:
            if (succession.host_entry(succession.configuration_locked(conn, room_id)) or {}).get("install_id") != me:
                succession.reconfigure_after_transition_locked(conn, room_id, successor=me,
                                                               previous_host=record["previous_host"])
        record = {**record, "reconfigured": True}
        _save(ctx, room_id, "move", record)
    if "reconciliation" not in record:
        record = {**record, "step": "reconciling", "reconciliation": reconcile(ctx, room_id, record)}
        _save(ctx, room_id, "move", record)
    if not record.get("owned"):
        record = {**record, "step": "finishing"}
        _take_ownership(ctx, room_id, record.get("owner_subject") or ctx.actor_subject)
        with rooms._transaction(ctx.db_path, immediate=True) as conn:
            succession.record_lineage_locked(conn, room_id, origin_install_id=record["origin_install_id"],
                                             gateway_id=me, epoch=record["to_epoch"],
                                             role=record.get("proof_kind") or "attested",
                                             proof_digest=record.get("proof_digest"))
        fence.learn_authority(ctx.runs_store.path, room_id=room_id, epoch=record["to_epoch"], install_id=me)
        record["owned"] = True
        _save(ctx, room_id, "move", record)
    if not record.get("routed"):
        record = {**record, "routes": register_routes(ctx, room_id, record.get("grants") or {}, record.get("routes")),
                  "routed": True}
        _save(ctx, room_id, "move", record)
    record = {**record, "state": "moved", "step": None, "moved_at": record.get("moved_at") or time.time(),
              "last_attempt": None}
    _save(ctx, room_id, "move", record)
    append_state(ctx.db_path, room_id, "ok")
    if ctx.service is not None:
        ctx.service.replication.wakeup()
        ctx.service.wakeup()
    announce(ctx, room_id)
    from gateway.hosted_room_succession_status import status
    return status(ctx, room_id)


# --- accepted work ------------------------------------------------------------------------------
def _reproduce(db_path: Path, room: Mapping[str, Any], admission: Mapping[str, Any]):
    """The admitted task's driver payload, reproduced and checked against its admitted identity.

    The admission event names the task, and its task id commits to the prompt and inputs, so a
    reproduction yielding the same identity is the same task; anything else stays unseeded.
    """
    from gateway.hosted_room_discussion import DiscussionPolicyError, plan_next_task
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
    identity = TaskIdentity(**{key: admission["task"][key] for key in ("room_id", "task_id", "thread_id", "turn_id")})
    try:
        snapshot = HostedRoomPolicyCheckpoint(db_path).snapshot(room_id=room["room_id"], latest_seq=room["latest_seq"])
        decision = plan_next_task(room, list(snapshot.events), local_profiles=policy_profiles(room),
                                  initial_watermarks=snapshot.watermarks, freeze_input_context=True)
    except (DiscussionPolicyError, rooms.HostedRoomError, ValueError, KeyError, RuntimeError):
        return identity, None
    if decision.task is None or decision.task.identity != identity:
        return identity, None
    return identity, dict(decision.task.payload)


def _granted_members(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    """(install_id, member_id) pairs this successor holds a continuation grant for."""
    return {(install_id, str(item.get("member_id"))) for install_id, items in (record.get("grants") or {}).items()
            for item in items or ()}


def reconcile(ctx: MoveContext, room_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify inherited work from the admissions in the log and the backups' run evidence.

    Finished or still running on a computer that granted this successor its member: this
    successor observes that exact run, whose Status and Stop passed here with the fence, and
    publishes its outcome. Work of a Bot that lives on the old host waits for that host. Everything
    else is unknown: kept with its original identity and never run again automatically.
    """
    from gateway import hosted_room_driver as driver
    from gateway.hosted_room_work_records import capture_transition_locked
    room = {**rooms.room_state(ctx.db_path, room_id=room_id),
            "authority_lineage": succession.authority_lineage(ctx.db_path, room_id)}
    host = record["previous_host"]
    configuration = view(ctx, room_id)["configuration"]
    host_name = succession.label(configuration, host)
    granted, origin = _granted_members(record), record["origin_install_id"]
    index, me, now, entries = evidence_index(record.get("evidence") or {}), succession.local_install_id(), \
        time.time(), []
    host_members = {bot["member_id"] for bot in host_bots(room["members"], host, origin=origin, here=me)}
    for admission in pending_admissions(_room_events(ctx.db_path, room_id, "hosted_room_events")):
        state, run = classify(admission, index, host_install_id=host, host_members=host_members)
        identity, payload = _reproduce(ctx.db_path, room, admission)
        resource = "bot" if state == "waiting_for_host" else None
        generation = int(admission["execution_generation"])
        member_id, target = str(admission.get("target_member_id")), str(admission["target_install_id"])
        observed = state in {"completed", "elsewhere"} and (target, member_id) in granted
        entries.append({"task_id": identity.task_id, "member_id": member_id, "execution_generation": generation,
                        "target_install_id": target, "state": state, "resource": resource,
                        "host_name": host_name if resource else None, "run_id": run["run_id"] if run else None,
                        "seeded": payload is not None})
        if payload is None:
            continue
        _, payload_json, payload_digest = driver._task_payload(payload)
        status = "running" if observed else "deferred" if state == "waiting_for_host" else "indeterminate"
        result = {"reason": "waiting_for_host", "resource": resource, "host_name": host_name, "retryable": False} \
            if status == "deferred" else None
        # Through the driver's own writer: it creates the driver's tables on a computer that never ran one.
        with driver._transaction(ctx.db_path) as conn:
            conn.execute("""INSERT OR IGNORE INTO hosted_room_driver_tasks (room_id, task_id, thread_id, turn_id,
                    source_event_seq, payload_json, payload_digest, status, execution_generation, cancel_generation,
                    run_gateway_id, run_process_generation, run_lease_generation, result_json, created_at, updated_at,
                    started_at, terminal_at, indeterminate_at) VALUES (?,?,?,?,?,?,?,?,?,0,?,?,?,?,?,?,?,?,?)""", (
                identity.room_id, identity.task_id, identity.thread_id, identity.turn_id, payload["source_event_seq"],
                payload_json, payload_digest, status, generation,
                me if status == "running" else None, f"succession:{record['to_epoch']}" if status == "running" else None,
                0 if status == "running" else None, json.dumps(result, sort_keys=True) if result else None,
                now, now, now if status == "running" else None, now if status == "deferred" else None,
                now if status == "indeterminate" else None))
            capture_transition_locked(conn, room_id)
        if status == "running":
            profile = payload["target_profile"]
            rooms.upsert_remote_run_receipt(ctx.db_path, record={
                "room_id": room_id, "home_install_id": me, "authority_gateway_id": me,
                "authority_epoch": record["to_epoch"], "member_id": member_id, "target_install_id": target,
                "target_profile": profile, "task_id": identity.task_id, "execution_generation": generation,
                "run_id": run["run_id"], "session_id": succession.member_session_for(origin, room_id, member_id,
                                                                                      profile)})
    return {"tasks": entries, "counts": work_counts(entries), "reconciled_at": now}


# --- ownership, routes, announcements and state events -------------------------------------------
def _take_ownership(ctx: MoveContext, room_id: str, subject: str | None) -> None:
    """The promoting owner owns the room here; the driver serves owned rooms only."""
    if ctx.service is None or not subject:
        return
    from hermes_state_runtime import RuntimeStoreError, _epoch
    service, key = ctx.service, "gateway.hosted.owner.v1:" + room_id

    def write(conn):
        _epoch(conn, service.authority.epoch)
        row = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
        if row is not None and row[0] != subject:
            raise RuntimeStoreError("permission_denied")
        conn.execute("INSERT OR IGNORE INTO state_meta(key, value) VALUES (?, ?)", (key, subject))
    service.authority.db._execute_write(write)


def register_routes(ctx: MoveContext, room_id: str, grants: Mapping[str, Any],
                    routes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Register each continuation grant as its member's route here, this computer's own included."""
    results = dict(routes or {})
    if ctx.service is None:
        return results
    from gateway import hosted_room_custody as custody
    from gateway.session_group_peers import register
    with _read(ctx) as conn:
        custodians = succession.custodians_by_id(succession.configuration_locked(conn, room_id))
    for install_id, items in grants.items():
        for item in items or ():
            if item["member_id"] == custody.CUSTODY_MEMBER_ID:
                # A backup's copy-only grant names no Bot: it is the custody route this host pushes the
                # history (and, to a voter, its lease requests) on. Its own copy is the room itself.
                if install_id == succession.local_install_id():
                    continue
                try:
                    custody.save_custody_route(ctx.db_path, room_id=room_id, install_id=install_id,
                                               target_url=custodians[install_id]["endpoint"],
                                               target_profile=item["target_profile"], grant=item["grant"],
                                               catalog=item["catalog"])
                    results[f"{custody.CUSTODY_MEMBER_ID}:{install_id}"] = "registered"
                except Exception as exc:
                    results[f"{custody.CUSTODY_MEMBER_ID}:{install_id}"] = getattr(exc, "reason", type(exc).__name__)
                continue
            try:
                register(ctx.service, {"room_id": room_id, "member_id": item["member_id"],
                                       "target_url": custodians[install_id]["endpoint"],
                                       "target_profile": item["target_profile"], "grant": item["grant"],
                                       "catalog": item["catalog"]})
                results[item["member_id"]] = "registered"
            except Exception as exc:  # a member without a route stays unavailable, visibly
                results[item["member_id"]] = getattr(exc, "reason", type(exc).__name__)
    return results


def append_state(db_path: Path, room_id: str, state: str, **params: Any) -> None:
    """A quiet ``succession.state`` on the room this computer hosts, so clients re-read status."""
    from gateway.hosted_room_succession_backup import serving_locked
    with closing(rooms._read_connection(db_path)) as conn:
        room = conn.execute("SELECT authority_gateway_id, authority_epoch, next_seq FROM hosted_rooms "
                            "WHERE room_id=? AND disbanded_at IS NULL", (room_id,)).fetchone()
        paused = room is not None and state != "continued_on_two" and not serving_locked(conn, room_id)
    if room is None or room[0] != succession.local_install_id() or paused:
        return  # a paused host appends nothing
    try:
        rooms.append_event(db_path, room_id=room_id, event_id=f"system:succession-state:{int(room[2])}:{state}",
                           kind="succession.state", actor={"kind": "system", "id": "succession"},
                           payload={"state": state, **params}, authority_gateway_id=room[0],
                           authority_epoch=int(room[1]))
    except rooms.HostedRoomError:
        pass  # a quarantined or full room keeps its state in status only


def restarting_until(conn, room_id: str) -> float | None:
    """When the host announced a planned restart, the time it expects to be back."""
    for table in ("hosted_room_events", "hosted_room_replica_events"):
        row = conn.execute(f"""SELECT payload_json FROM {table} WHERE room_id=? AND kind='succession.state'
            ORDER BY seq DESC LIMIT 1""", (room_id,)).fetchone()
        if row is not None:
            payload = json.loads(row[0])
            until = payload.get("until")
            if payload.get("state") == "host_restarting" and isinstance(until, (int, float)):
                return float(until)
            return None
    return None


def _verified_rival(ctx: MoveContext, room_id: str, transition: Any) -> dict[str, Any] | None:
    """A rival successor's transition from a refusal, kept only once its claim verifies."""
    try:
        event = transition["event"]
        with _read(ctx) as conn:
            mine = succession.own_transition_locked(conn, room_id, from_epoch=int(event["payload"]["from_epoch"]))
            fork = min(int(event["seq"]), int(mine["seq"]) if mine else int(event["seq"])) - 1
            succession.verify_claim_locked(conn, room_id, event["payload"], fork_seq=fork)
        return {"event": dict(event), "fork_event": transition.get("fork_event")}
    except (SuccessionError, KeyError, TypeError, ValueError):
        return None


def announce(ctx: MoveContext, room_id: str) -> dict[str, Any]:
    """Tell every other computer of the group about this host's latest transition; each verifies it,
    follows (setting aside what it holds beyond the shared history) and may grant. A computer that
    followed or became another successor at that epoch reports the conflict."""
    from gateway.hosted_room_succession_backup import open_sealed_reply
    record = succession.load_record(ctx.db_path, room_id, "move") or {}
    conflict = succession.load_record(ctx.db_path, room_id, "conflict") or {}
    me = succession.local_install_id()
    with _read(ctx) as conn:
        mine = succession.latest_transition_locked(conn, room_id)
        custodians = succession.custodians_by_id(succession.configuration_locked(conn, room_id))
        decision = conflict.get("decision") if conflict.get("kept_epoch") else None
        lineage = (succession.latest_transition_locked(conn, room_id, int(decision["epoch"]) - 1)
                   if decision is not None else None)
    if mine is None:
        return record
    announced, routes = dict(record.get("announced") or {}), record.get("routes")
    for install_id in sorted(set(custodians) - {me}):
        if announced.get(install_id) == "acknowledged" or not custodians[install_id].get("endpoint"):
            continue
        private, public = succession.reply_keypair()
        unsigned = {"room_id": room_id, "successor_install_id": me, "reply_key": public, "issued_at": time.time(),
                    "nonce": succession.nonce(), "transition": mine["event"], "fork_event": mine["fork_event"],
                    **({"decision": decision, "lineage": lineage} if decision is not None else {})}
        request = {**unsigned, "signature": succession.sign(succession.LEARN, unsigned)}
        try:
            reply = (ctx.post or http_post)(str(custodians[install_id]["endpoint"]),
                                            "/v1/room-members/succession/learn", request, ctx.timeout)
            opened = open_sealed_reply(private, reply, request)
        except RemoteRefusal as exc:
            announced[install_id] = exc.code
            if exc.code == "room_authority_conflict" and isinstance(exc.detail.get("transition"), dict):
                theirs = _verified_rival(ctx, room_id, exc.detail["transition"])
                if theirs is not None:
                    record_conflict(ctx.db_path, room_id, mine=mine, theirs=theirs)
            continue
        except Exception as exc:
            announced[install_id] = "unreachable:" + type(exc).__name__
            continue
        if opened.get("rebased"):
            announced[install_id] = "rebasing"
            continue
        if opened.get("held"):
            announced[install_id] = "held"  # the other host keeps its own history apart until the owner chooses
            continue
        announced[install_id] = "acknowledged"
        late = opened.get("continuation_grants") or []
        if late:
            routes = register_routes(ctx, room_id, {install_id: late}, routes)
    current = succession.load_record(ctx.db_path, room_id, "move") or record
    current = {**current, "announced": announced, "routes": routes}
    if announced.get(current.get("previous_host")) == "acknowledged":
        current["reconciled_old_host"] = True  # the host it replaced follows it now
    _save(ctx, room_id, "move", current)
    return current


# --- continued on two computers ----------------------------------------------------------------------
# Which proof wins a tie between two hosts at the same epoch (a handover never ties).
PROOF_RANK = {"handover": 4, "certified": 3, "evidence": 2, "attested": 1}
# ``decided_by`` of a choice the rule made, not the owner.
RULE = "rule"


def side_of(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """One side of a conflict: the host it names, since when, the epoch it holds and how it got it.

    A side is a host's own transition (``event``), or a host that never moved (``head``: the
    group's earlier host that kept writing while it was cut off).
    """
    if not entry:
        return None
    event = entry.get("event")
    if isinstance(event, Mapping):
        payload = event["payload"]
        return {"install_id": payload["successor_gateway_id"], "since": event["created_at"],
                "epoch": int(payload["to_epoch"]), "proof_kind": payload.get("proof_kind")}
    head = entry.get("head") or {}
    return {"install_id": head.get("install_id"), "since": head.get("since"), "epoch": int(head.get("epoch") or 0),
            "proof_kind": None}


def winner_of(sides: list[Mapping[str, Any]]) -> str:
    """The host the group keeps running on: the higher epoch; on a tie the stronger proof, then the
    lower installation id. Every computer reaches the same answer from the same two sides."""
    return min(sides, key=lambda side: (-int(side.get("epoch") or 0), -PROOF_RANK.get(side.get("proof_kind"), 0),
                                        str(side.get("install_id"))))["install_id"]


def record_conflict(db_path: Path, room_id: str, *, mine: Mapping[str, Any] | None,
                    theirs: Mapping[str, Any], start: float | None = None) -> dict[str, Any]:
    """Two hosts, each with history the other lacks. The winner by rule keeps serving; the other stops
    and keeps its own history apart until the owner chooses. Never merged."""
    existing = succession.load_record(db_path, room_id, "conflict") or {}
    if existing.get("state") == "active":
        return existing
    hosts = [side for side in (side_of(mine), side_of(theirs)) if side]
    record = {"state": "active", "detected_at": time.time(), "start": start, "hosts": hosts, "mine": mine,
              "theirs": dict(theirs), "decision": None, "winner": winner_of(hosts) if len(hosts) == 2 else None}
    succession.save_record(db_path, room_id, "conflict", record)
    append_state(db_path, room_id, "continued_on_two")
    return record


def conflict_loser(record: Mapping[str, Any] | None, install_id: str) -> bool:
    """Whether ``install_id`` is the side of an active conflict that stopped serving."""
    return bool(record) and record.get("state") == "active" and record.get("winner") != install_id


def resolve_conflict(db_path: Path, room_id: str, decision: Mapping[str, Any] | None) -> None:
    record = succession.load_record(db_path, room_id, "conflict")
    if record and record.get("state") == "active":
        succession.save_record(db_path, room_id, "conflict", {**record, "state": "resolved",
                                                              "decision": dict(decision or {})})


def conflict_active(db_path: Path, room_id: str) -> bool:
    record = succession.load_record(db_path, room_id, "conflict")
    return bool(record) and record.get("state") == "active"


def _sign_decision(ctx: MoveContext, room_id: str, record: Mapping[str, Any], keep_id: str, *,
                   by_rule: bool = False) -> dict[str, Any]:
    me = succession.local_install_id()
    mine, theirs = record.get("mine") or {}, record["theirs"]
    kept, discarded = (mine, theirs) if keep_id == me else (theirs, mine)
    other = next(host["install_id"] for host in record["hosts"] if host["install_id"] != keep_id)
    unsigned = {"room_id": room_id, "keep_install_id": keep_id, "discard_install_id": other,
                "epoch": max(int(host["epoch"]) for host in record["hosts"]),
                "kept": {"event": kept.get("event"), "fork_event": kept.get("fork_event"), "head": kept.get("head")},
                # The other side too, so anyone can check a choice the rule made.
                "discarded": {"event": discarded.get("event"), "head": discarded.get("head")},
                "decided_by": RULE if by_rule else ctx.actor_subject or "operator", "decided_at": time.time(),
                "signer_install_id": me, "issued_at": time.time(), "nonce": succession.nonce()}
    return {**unsigned, "signature": succession.sign(succession.DECISION, unsigned)}


def keep(ctx: MoveContext, room_id: str, install_id: Any) -> dict[str, Any]:
    """Resolve ``continued_on_two`` from either computer, without needing both reachable at once."""
    from gateway.hosted_room_succession_status import status
    record = _conflict_or_moved_in(ctx, room_id, install_id)
    configuration = view(ctx, room_id)["configuration"]
    if not record or record.get("state") != "active":
        raise SuccessionError("this group is not continued on two computers", reason="invalid_params")
    if not is_owner(ctx, room_id):
        raise SuccessionError("only the group's owner can choose", reason="not_owner")
    me = succession.local_install_id()
    hosts = [host["install_id"] for host in record["hosts"]]
    if install_id not in hosts or me not in hosts or len(set(hosts)) != 2:
        raise SuccessionError("choose on one of the two computers", reason="target_not_local",
                              detail={"target": named(configuration, install_id if isinstance(install_id, str)
                                                      else None)})
    decision = _sign_decision(ctx, room_id, record, install_id)
    succession.save_record(ctx.db_path, room_id, "conflict", {**record, "decision": decision, "delivered": False})
    if install_id == me:
        # The running host steps aside first, so the two never serve at once; then this one continues.
        deliver_decision(ctx, room_id)
        try:
            keep_continue(ctx, room_id)
        except SuccessionError:
            pass  # upkeep tries again, once the other host's lease ran out
    else:
        _discard_here(ctx, room_id, decision)
        deliver_decision(ctx, room_id)
    return status(ctx, room_id)


def _conflict_or_moved_in(ctx: MoveContext, room_id: str, install_id: Any) -> dict[str, Any] | None:
    """The active conflict; on a new host after an evidence move, "go back" to the old host before it is
    reconciled starts one from the move itself."""
    record = succession.load_record(ctx.db_path, room_id, "conflict")
    if record and record.get("state") == "active":
        return record
    move = succession.load_record(ctx.db_path, room_id, "move") or {}
    if move.get("state") != "moved" or move.get("proof_kind") != "evidence" or install_id != move.get(
            "previous_host") or move.get("reconciled_old_host"):
        return record
    with _read(ctx) as conn:
        mine = succession.latest_transition_locked(conn, room_id)
    return record_conflict(ctx.db_path, room_id, mine=mine, start=(move.get("silence") or {}).get("silent_since"),
                           theirs={"event": None, "head": {"install_id": install_id, "epoch": move["from_epoch"],
                                                           "since": move.get("offline_since")}})


def _discard_here(ctx: MoveContext, room_id: str, decision: Mapping[str, Any]) -> None:
    """This computer was not kept: step aside now when the kept side's transition replaces this
    history; otherwise (the kept host continues at a fresh epoch) stop serving until that epoch
    arrives, and step aside then."""
    from gateway.hosted_room_succession_return import pause, step_down
    kept = decision["kept"]
    stepped = None
    if isinstance(kept.get("event"), Mapping):
        stepped = step_down(ctx, room_id, kept["event"], kept.get("fork_event"), decision=decision, follow=False)
    if stepped is None and (view(ctx, room_id)["head"]["authoritative"]):
        pause(ctx.db_path, room_id, {"install_id": decision["keep_install_id"], "epoch": decision["epoch"] + 1})
    resolve_conflict(ctx.db_path, room_id, decision)


def keep_continue(ctx: MoveContext, room_id: str) -> dict[str, Any] | None:
    """The kept host continues its own group, at a fresh epoch every computer can follow when the
    other one held this epoch or a later one; otherwise its own epoch already supersedes the other.

    A fresh epoch fences the current one at every reachable computer (receipts, as for any
    continuation), then writes its own marked transition with the owner's choice. In majority mode
    a majority of the voters must promise it, so the host it replaces no longer holds its lease. The
    conflict is resolved here only then.
    """
    record = succession.load_record(ctx.db_path, room_id, "conflict") or {}
    decision, me = record.get("decision"), succession.local_install_id()
    if (record.get("state") != "active" or not decision or decision.get("keep_install_id") != me
            or decision.get("decided_by") == RULE):
        return None
    current = view(ctx, room_id)
    head = current["head"]
    if not head["authoritative"]:
        return None
    rivals = [int(host["epoch"]) for host in record["hosts"] if host["install_id"] != me]
    if rivals and head["authority_epoch"] > max(rivals):
        record = {**record, "state": "resolved", "kept_epoch": head["authority_epoch"], "resolved_at": time.time()}
        succession.save_record(ctx.db_path, room_id, "conflict", record)
        append_state(ctx.db_path, room_id, "ok")
        deliver_decision(ctx, room_id)
        return record
    epoch = _continue_fresh(ctx, room_id, current, decision, rivals)
    record = {**(succession.load_record(ctx.db_path, room_id, "conflict") or record), "state": "resolved",
              "kept_epoch": epoch, "resolved_at": time.time()}
    succession.save_record(ctx.db_path, room_id, "conflict", record)
    append_state(ctx.db_path, room_id, "ok")
    announce(ctx, room_id)
    deliver_decision(ctx, room_id)
    return record


def _continue_fresh(ctx: MoveContext, room_id: str, current: Mapping[str, Any], decision: Mapping[str, Any] | None,
                    rivals: list[int], *, stalled: Mapping[str, Any] | None = None,
                    required: frozenset[str] = frozenset()) -> int:
    """This host continues its own group at a fresh epoch above every one known: with the owner's or the
    rule's ``decision``, or, past a step promised to ``stalled`` that never took it, with the promise of
    the fresh epoch from every computer in ``required`` (each one that could have taken a later step)."""
    from gateway import hosted_room_fence as fence
    from gateway.hosted_room_succession_automatic import mode_of
    configuration, head, me = current["configuration"], current["head"], succession.local_install_id()
    own = fence.room_fence_state(ctx.runs_store.path, room_id)
    known = [head["authority_epoch"], *rivals, int(own["fenced_epoch"] or 0),
             int((own.get("authority") or {}).get("epoch") or 0), int((own.get("promise") or {}).get("epoch") or 0)]
    epoch = max(known) + 1
    outcomes = _ask_fences(ctx, room_id, configuration, epoch, current["watermark"])
    _refused_for_other(configuration, outcomes)
    if me not in outcomes or isinstance(outcomes[me], RemoteRefusal):
        raise SuccessionError("this computer could not fence its own epoch", reason="target_not_ready")
    fenced = {k: v for k, v in outcomes.items() if not isinstance(v, RemoteRefusal)}
    voters = succession.voters_of(configuration)
    if mode_of(configuration) == "majority" and len(set(fenced) & set(voters)) < succession.majority(len(voters)):
        raise SuccessionError("a majority of the voters did not promise yet", reason="no_majority")
    missing = sorted(required - set(fenced))
    if missing:
        raise SuccessionError("not every computer that could continue the group promised this step",
                              reason="target_not_ready", detail={"other": named(configuration, missing[0])})
    by_rule = stalled is not None or (decision or {}).get("decided_by") == RULE
    proof = succession.build_attestation(
        room_id=room_id, from_epoch=head["authority_epoch"], to_epoch=epoch, successor=me, previous_authority=me,
        origin_install_id=current["origin"], configuration_seq=int(configuration.get("configuration_seq") or 0),
        receipts=[fenced[k]["receipt"] for k in sorted(fenced)],
        unreachable=sorted(k for k, v in outcomes.items() if isinstance(v, RemoteRefusal)),
        confirmed_by=RULE if stalled is not None else str((decision or {}).get("decided_by") or "operator"),
        preview_id=None, decision=decision,
        statement=succession.RECOVER_TEXT if stalled is not None else None, stalled=stalled)
    advance_here(ctx, room_id, current, proof, epoch, reason="automatic" if by_rule else "manual")
    serve_again(ctx, room_id, fenced)
    return epoch


# A host paused for a later step promised to another computer continues past it itself (by rule) once
# that computer has left the step untaken this long while answering and not hosting: twice the careful
# window, the longest any takeover waits. Only when every computer that could have taken a later step
# answers that none did, and promises the fresh one; otherwise the owner decides (continue anyway).
STALLED_PROMISE_SECONDS = 2 * succession.CAREFUL_SILENCE_SECONDS


def stalled_promise(db_path: Path, room_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    """The step this host paused for, once its computer has left it untaken past ``STALLED_PROMISE_SECONDS``:
    ``{install_id, epoch, since, unconfirmed}``; ``unconfirmed`` lists the computers the host's last check
    could not hear say that nothing happened. None otherwise."""
    record = succession.load_record(db_path, room_id, "return") or {}
    stalled = record.get("fenced_by") if isinstance(record.get("fenced_by"), Mapping) else {}
    since = record.get("at")
    if (record.get("state") != "paused" or not isinstance(stalled.get("install_id"), str)
            or not isinstance(since, (int, float))
            or (time.time() if now is None else float(now)) - float(since) < STALLED_PROMISE_SECONDS):
        return None
    return {"install_id": stalled["install_id"], "epoch": int(stalled.get("epoch") or 0), "since": float(since),
            "unconfirmed": [item for item in record.get("unconfirmed") or () if isinstance(item, str)]}


def _later_step(answer: Mapping[str, Any], epoch: int, stalled: Mapping[str, Any], me: str) -> bool:
    """Whether a computer's signed answer shows a step past ``epoch`` beyond the stalled one: it hosts, it
    holds a transition or follows a later host, or it promised a later epoch to another computer."""
    lineage = answer.get("lineage") if isinstance(answer.get("lineage"), Mapping) else {}
    taken = int((((lineage.get("event") or {}).get("payload")) or {}).get("to_epoch") or 0)
    fence = answer.get("fence") if isinstance(answer.get("fence"), Mapping) else {}
    promise = fence.get("promise") if isinstance(fence.get("promise"), Mapping) else {}
    promised = int(promise.get("epoch") or 0)
    stalled_step = (promise.get("candidate_install_id") == stalled["install_id"]
                    and promised <= int(stalled["epoch"]))
    return bool(answer.get("hosting") or answer.get("transition") or taken > epoch
                or int((answer.get("authority") or {}).get("epoch") or 0) > epoch
                or int((fence.get("authority") or {}).get("epoch") or 0) > epoch
                or (promised > epoch and not stalled_step and promise.get("candidate_install_id") != me))


def _could_continue(configuration: Mapping[str, Any], stalled: Mapping[str, Any], me: str) -> frozenset[str]:
    """Every computer that could have continued the group past this host: each eligible successor, and
    the one the stalled step was promised to."""
    return frozenset({install_id for install_id in succession.custodians_by_id(configuration)
                      if install_id != me and succession.is_eligible(configuration, install_id)}
                     | {stalled["install_id"]})


def _unconfirmed(required: frozenset[str], answers: list[Mapping[str, Any]], epoch: int,
                 stalled: Mapping[str, Any], me: str) -> list[str]:
    """The computers that keep this host from continuing past a stalled step by itself: each one in
    ``required`` that did not answer, and any whose answer shows a later step. Empty: nothing in the way."""
    heard = {str(item.get("responder_install_id")): item for item in answers}
    blocking = {install_id for install_id in required if install_id not in heard}
    blocking |= {install_id for install_id, item in heard.items() if _later_step(item, epoch, stalled, me)}
    return sorted(blocking)


def recover_stalled_promise(ctx: MoveContext, room_id: str, answers: list[Mapping[str, Any]]) -> int | None:
    """This host paused because a later step of its group was promised to another computer, which never
    took it. After ``STALLED_PROMISE_SECONDS`` the host continues at a fresh step itself, but only when
    every computer that could have continued the group (each eligible successor, and the one the step was
    promised to) answers that nothing happened, and each promises the fresh step, so no step anyone took
    or might still take runs beside it. Otherwise it stays paused for the owner to decide, and its record
    names the computers it could not hear that from. Returns the fresh epoch, or None."""
    stalled, me = stalled_promise(ctx.db_path, room_id), succession.local_install_id()
    if stalled is None or stalled["install_id"] == me or getattr(ctx, "runs_store", None) is None:
        return None
    current = view(ctx, room_id)
    if not current["head"]["authoritative"]:
        return None
    configuration, epoch = current["configuration"], int(current["head"]["authority_epoch"])
    required = _could_continue(configuration, stalled, me)
    unconfirmed = _unconfirmed(required, answers, epoch, stalled, me)
    record = succession.load_record(ctx.db_path, room_id, "return") or {}
    if record.get("state") == "paused" and record.get("unconfirmed") != unconfirmed:
        succession.save_record(ctx.db_path, room_id, "return", {**record, "unconfirmed": unconfirmed})
    if unconfirmed:
        return None  # it can't hear from every computer that nothing happened: the owner decides
    fresh = _continue_fresh(ctx, room_id, current, None, [stalled["epoch"]],
                            stalled={"install_id": stalled["install_id"], "epoch": stalled["epoch"]},
                            required=required)
    announce(ctx, room_id)
    return fresh


def continue_by_rule(ctx: MoveContext, room_id: str) -> dict[str, Any] | None:
    """The host the rule keeps, tied at its epoch with the other one: continue at a fresh epoch every
    computer can follow, signed as the rule's choice. The conflict stays shown until the owner
    chooses; the other host keeps its own history apart meanwhile."""
    record = succession.load_record(ctx.db_path, room_id, "conflict") or {}
    me = succession.local_install_id()
    if record.get("state") != "active" or record.get("winner") != me or record.get("rule_epoch") or (
            record.get("decision") or {}).get("decided_by") not in {None, RULE}:
        return None
    current = view(ctx, room_id)
    head = current["head"]
    rivals = [int(host["epoch"]) for host in record["hosts"] if host["install_id"] != me]
    if not head["authoritative"] or not rivals or head["authority_epoch"] > max(rivals):
        return None  # a higher epoch already supersedes the other host
    decision = record.get("decision") or _sign_decision(ctx, room_id, record, me, by_rule=True)
    succession.save_record(ctx.db_path, room_id, "conflict", {**record, "decision": decision, "delivered": False})
    epoch = _continue_fresh(ctx, room_id, current, decision, rivals)
    record = {**(succession.load_record(ctx.db_path, room_id, "conflict") or record), "kept_epoch": epoch,
              "rule_epoch": epoch}
    succession.save_record(ctx.db_path, room_id, "conflict", record)
    announce(ctx, room_id)
    deliver_decision(ctx, room_id)
    return record


def advance_here(ctx: MoveContext, room_id: str, current: Mapping[str, Any], proof: Mapping[str, Any],
                 epoch: int, *, reason: str = "manual") -> None:
    """Write this host's own fresh epoch with the owner's attestation (or the split rule's choice,
    ``reason: automatic``) and follow it in the fence store."""
    from gateway import hosted_room_fence as fence
    configuration, me = current["configuration"], succession.local_install_id()
    advance_room(ctx.db_path, room_id=room_id, proof=proof, to_epoch=epoch,
                 text=succession.transition_text(succession.label(configuration, me)),
                 display={"from_name": succession.label(configuration, me),
                          "to_name": succession.label(configuration, me), "offline_since": None,
                          "reason": reason, "at_risk": 0})
    with rooms._transaction(ctx.db_path, immediate=True) as conn:
        succession.record_lineage_locked(conn, room_id, origin_install_id=current["origin"], gateway_id=me,
                                         epoch=epoch, role="attested", proof_digest=succession.proof_digest(proof))
        returned = succession.load_record_locked(conn, room_id, "return") or {}
        if returned.get("state") == "paused":  # it promised its old epoch away; it holds the fresh one now
            succession.save_record_locked(conn, room_id, "return", {**returned, "state": "resumed",
                                                                    "resumed_at": time.time(), "epoch": epoch})
    fence.learn_authority(ctx.runs_store.path, room_id=room_id, epoch=epoch, install_id=me)


def serve_again(ctx: MoveContext, room_id: str, fenced: Mapping[str, Any]) -> None:
    """Once the pause is over: register late continuation grants, tell clients, and wake the driver."""
    late = {k: v["continuation_grants"] for k, v in fenced.items() if v["continuation_grants"]}
    move = succession.load_record(ctx.db_path, room_id, "move") or {}
    _save(ctx, room_id, "move", {**move, "announced": {}, "grants": {**(move.get("grants") or {}), **late}})
    register_routes(ctx, room_id, late)
    append_state(ctx.db_path, room_id, "ok")
    if ctx.service is not None:
        ctx.service.replication.wakeup()
        ctx.service.wakeup()


def advance_room(db_path: Path, *, room_id: str, proof: Mapping[str, Any], to_epoch: int, text: str | None,
                 display: Mapping[str, Any]) -> dict[str, Any]:
    """Continue a room this computer hosts at a later, verified epoch, in one writer transaction.

    The same lineage event and mark ``promote_replica`` writes, for a host that keeps its group.
    """
    from gateway.hosted_room_replicas import _control_event, _transition_display, _transition_text
    from gateway.hosted_room_safety import mark_verified_transition
    transition = succession.transition_for(proof)
    me, now = succession.local_install_id(), time.time()
    notice = {**_transition_text(text, transition), **_transition_display(dict(display), transition)}
    with rooms._transaction(db_path, immediate=True) as conn:
        room = conn.execute("""SELECT next_seq, event_bytes, authority_gateway_id, authority_epoch FROM hosted_rooms
            WHERE room_id=? AND disbanded_at IS NULL""", (room_id,)).fetchone()
        if room is None or room["authority_gateway_id"] != me or int(room["authority_epoch"]) >= to_epoch:
            raise SuccessionError("this computer does not host the group at an earlier epoch",
                                  reason="room_authority_superseded")
        from_epoch, seq = int(room["authority_epoch"]), int(room["next_seq"])
        succession.verify_proof_locked(conn, room_id, proof_kind="attested", proof=dict(proof),
                                       from_epoch=from_epoch, to_epoch=to_epoch, successor=me)
        mark_verified_transition(conn, room_id=room_id, from_epoch=from_epoch, to_epoch=to_epoch,
                                 successor_gateway_id=me, proof_kind="attested",
                                 proof_digest=transition["proof_digest"])
        # Before the change is appended: this host's head for the epoch it leaves, so a copy that missed
        # that epoch's last pushes can still catch up across the change (#104601's ``heads_locked``).
        succession.keep_own_head_locked(conn, room_id)
        event = _control_event("transition", to_epoch, {"from_epoch": from_epoch, "to_epoch": to_epoch,
                                                        "successor_gateway_id": me, **transition, **notice})
        added = rooms._insert_event(conn, room, room_id, seq, event[0], event[1], event[2], to_epoch, event[3], now,
                                    allow_control=True)
        conn.execute("""UPDATE hosted_rooms SET authority_epoch=?, next_seq=?, event_bytes=event_bytes+?,
            revision=revision+1, updated_at=? WHERE room_id=? AND next_seq=?""",
                     (to_epoch, seq + 1, added, now, room_id, seq))
    return {"room_id": room_id, "authority_epoch": to_epoch, "seq": seq}


def deliver_decision(ctx: MoveContext, room_id: str) -> None:
    """Send the owner's choice to the other computer, once it can be reached."""
    record = succession.load_record(ctx.db_path, room_id, "conflict") or {}
    decision = record.get("decision")
    if not decision or record.get("delivered") or "signature" not in decision:
        return
    me = succession.local_install_id()
    other = decision["discard_install_id"] if decision["keep_install_id"] == me else decision["keep_install_id"]
    with _read(ctx) as conn:
        endpoint = succession.custodians_by_id(succession.configuration_locked(conn, room_id)).get(
            other, {}).get("endpoint")
    if not endpoint:
        return
    try:
        reply = (ctx.post or http_post)(str(endpoint), "/v1/room-members/succession/decision", dict(decision),
                                        ctx.timeout)
    except Exception:
        return
    if not (isinstance(reply, Mapping) and reply.get("applied") is True):
        return  # not applied there yet (it may not know of the conflict): sent again on the next pass
    succession.save_record(ctx.db_path, room_id, "conflict", {**record, "delivered": True})


def adopt_kept_decision(ctx: MoveContext, room_id: str, decision: Any) -> bool:
    """The owner kept this host while it was away ("Go back", or keeping it after a split): it learns
    the choice from the other computer, verified with pinned keys, and records the conflict it never
    saw, so upkeep continues its group at a fresh epoch instead of stepping it down."""
    me = succession.local_install_id()
    if not isinstance(decision, Mapping) or decision.get("keep_install_id") != me or decision.get(
            "decided_by") == RULE:
        return False
    current = view(ctx, room_id)
    head = current["head"]
    record = succession.load_record(ctx.db_path, room_id, "conflict") or {}
    with _read(ctx) as conn:
        try:
            chosen = succession.verify_decision_locked(conn, room_id, dict(decision))
        except SuccessionError:
            return False
    kept = (chosen.get("kept") or {}).get("head") or {}
    discarded = chosen.get("discarded") if isinstance(chosen.get("discarded"), Mapping) else {}
    if not head["authoritative"] or int(kept.get("epoch") or -1) != head["authority_epoch"] or not isinstance(
            discarded.get("event"), Mapping):
        return False
    if record.get("state") != "active":
        record = record_conflict(ctx.db_path, room_id, mine={"event": None, "head": dict(kept)},
                                 theirs={"event": dict(discarded["event"]), "fork_event": None})
    succession.save_record(ctx.db_path, room_id, "conflict", {**record, "decision": dict(decision),
                                                              "delivered": True})
    return True


def apply_decision(context, room_id: str, decision: Mapping[str, Any]) -> dict[str, Any]:
    """The owner's choice, signed on the other computer, reaches this one.

    Not kept: step aside now. Kept: record it (also when this host never saw the conflict, after "Go
    back" while it was away); upkeep continues the group at a fresh epoch.
    """
    record = succession.load_record(context.custody_db, room_id, "conflict") or {}
    me = succession.local_install_id()
    if record.get("state") != "active" and decision.get("keep_install_id") == me:
        applied = adopt_kept_decision(context.move_context(), room_id, decision)
        return {"applied": applied, "room_id": room_id}
    if record.get("state") != "active" or me not in {decision.get("keep_install_id"),
                                                    decision.get("discard_install_id")}:
        return {"applied": False, "room_id": room_id}
    if decision["keep_install_id"] == me or decision.get("decided_by") == RULE:
        # Kept, or the rule's choice the owner hasn't confirmed: record it. Upkeep continues a kept host;
        # a host the rule passed over keeps its history apart until the owner chooses.
        succession.save_record(context.custody_db, room_id, "conflict", {**record, "decision": dict(decision),
                                                                         "delivered": True})
        return {"applied": True, "room_id": room_id}
    ctx = context.move_context()
    _discard_here(ctx, room_id, decision)
    return {"applied": True, "room_id": room_id}


def maintain(ctx: MoveContext, room_ids) -> None:
    """Upkeep for rooms this computer hosts: finish a move, announce it, and act on the owner's choice."""
    for room_id in room_ids:
        try:
            record = committed_transition(ctx, room_id, succession.load_record(ctx.db_path, room_id, "move") or {})
            if record.get("state") == "moving" and record.get("transition_committed"):
                finish(ctx, room_id, record)
            elif record.get("state") == "moved" and any(
                    state not in {"acknowledged", "held"} for state in (record.get("announced") or {}).values()):
                announce(ctx, room_id)
            keep_continue(ctx, room_id)
            continue_by_rule(ctx, room_id)
            deliver_decision(ctx, room_id)
        except Exception:
            continue
