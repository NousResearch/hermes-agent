"""Handing a Group Chat over on purpose: the host signs its exact history over to a standby.

The owner moves a group while its host is up (``groups.succession.move``), a gateway that stops
or quits without restarting hands its groups over first, and Desktop's sleep hook asks for the
same (``groups.succession.handover_all``). In every case:

1. The host stops admitting: the group pauses here, so its history is final.
2. It signs ``{room_id, from_epoch, to_epoch, successor, last_seq, last_hash}`` with its room
   identity key (proof kind ``handover``): exactly this history, ending at ``last_seq`` whose
   custody chain hash is ``last_hash``, goes to that computer.
3. The standby catches up to ``last_seq`` from the host if it must, checks the signature with the
   host's pinned key and its own copy against ``last_seq`` and ``last_hash``, fences the host's
   epoch at every computer it reaches (each verifies the statement and gives back the host's
   lease), writes the marked transition (``reason: handover``) and finishes like any continuation.
4. The host steps down to a copy that follows the standby; nothing is set aside.

If the host vanishes before step 2, nothing moved, and the lease or the manual path applies. If it
is refused later, the host resumes. A handover is never completed without the signed statement.
"""

from __future__ import annotations

import time
from contextlib import closing
from typing import Any, Mapping

from gateway import hosted_room_succession as succession
from gateway import hosted_rooms as rooms
from gateway.hosted_room_succession import ProofInvalid, SuccessionError

HANDOVER = b"hermes.group.succession.handover.v1"
RELEASE = b"hermes.group.succession.handover-release.v1"
HANDOVER_TIMEOUT_SECONDS = 30.0
# How long a host waits for its running turns to settle before it signs: their outcomes then belong
# to the history it hands over. A turn still running after that is counted at risk and inherited.
DRAIN_SECONDS = 20.0
# The owner's "Move to..." while the host stays up waits for its running turns this long at most.
MOVE_DRAIN_SECONDS = 15 * 60.0
DRAIN_POLL_SECONDS = 0.25
_RUNNING = ("running", "stopping")


def statement_for(conn, room_id: str, *, successor: str, to_epoch: int) -> dict[str, Any]:
    """The host's statement handing exactly its current history to ``successor``."""
    room = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                        "AND disbanded_at IS NULL", (room_id,)).fetchone()
    if room is None or room[0] != succession.local_install_id():
        raise SuccessionError("this computer does not host the group", reason="room_not_found")
    mark = succession.watermark_locked(conn, room_id)
    return {"room_id": room_id, "from_epoch": int(room[1]), "to_epoch": int(to_epoch), "successor": successor,
            "last_seq": int(mark["seq"]), "last_hash": mark["event_hash"]}


_RELEASE_STEP = ("room_id", "from_epoch", "to_epoch", "successor", "last_seq")


def release_for(statement: Mapping[str, Any]) -> dict[str, Any]:
    """The host's signed word, beside the proof and never in the log, that its voters may give back a
    lease it asked for before now: its own sleep-counting clock and boot (``signed_at``, ``boot``)."""
    from gateway import hosted_room_clock as clock
    token = {**{key: statement[key] for key in _RELEASE_STEP}, "signed_at": clock.now(), "boot": clock.boot_id()}
    return {"token": token, "signature": succession.sign(RELEASE, token)}


def verify_release_locked(conn, room_id: str, release: Any, statement: Mapping[str, Any]) -> dict[str, Any] | None:
    """The release token for exactly this statement's step, signed by the group's configured host;
    ``None`` when it is missing or isn't."""
    token = release.get("token") if isinstance(release, Mapping) else None
    if not isinstance(token, Mapping) or set(token) != {*_RELEASE_STEP, "signed_at", "boot"} or any(
            token[key] != statement.get(key) for key in _RELEASE_STEP):
        return None
    signed_at, boot = token["signed_at"], token["boot"]
    if isinstance(signed_at, bool) or not isinstance(signed_at, (int, float)) or signed_at != signed_at or not (
            isinstance(boot, str) and 0 < len(boot) <= 128):
        return None
    host = succession.host_entry(succession.configuration_locked(conn, room_id))
    if host is None or not succession.verify_locked(conn, room_id, host["install_id"], RELEASE, dict(token),
                                                    release.get("signature")):
        return None
    return dict(token)


def signed_over(ctx, conn, room_id: str, *, successor: str, to_epoch: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sign the statement and its release token; from then on this host counts no lease grant asked for
    before it, so if it resumes it serves only on grants the token can't give back."""
    from gateway.hosted_room_succession_automatic import void_lease
    proof = sign(statement_for(conn, room_id, successor=successor, to_epoch=to_epoch))
    release = release_for(proof["statement"])
    void_lease(ctx.db_path, room_id, release["token"]["signed_at"])
    return proof, release


def sign(statement: Mapping[str, Any]) -> dict[str, Any]:
    return {"statement": dict(statement), "signature": succession.sign(HANDOVER, dict(statement))}


def verify_locked(conn, room_id: str, proof: Any, *, from_epoch: int, to_epoch: int, successor: str,
                  fork_seq: int | None, configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A handover proof: signed by the configured host it names, for this step, and naming exactly the
    history this computer holds up to ``last_seq`` (its chain hash)."""
    if configuration is None:
        configuration = succession.configuration_locked(conn, room_id)
    host = succession.host_entry(configuration)
    statement = proof.get("statement") if isinstance(proof, Mapping) else None
    expected = {"room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch, "successor": successor}
    if (not isinstance(statement, Mapping) or host is None
            or any(statement.get(key) != value for key, value in expected.items())
            or (fork_seq is not None and statement.get("last_seq") != fork_seq)):
        raise ProofInvalid("the handover names another group, step, successor or history")
    if successor == host["install_id"] or successor not in succession.custodians_by_id(configuration):
        raise ProofInvalid("the handover names a computer that keeps no copy of this group")
    if not succession.verify_locked(conn, room_id, host["install_id"], HANDOVER, dict(statement),
                                    proof.get("signature")):
        raise ProofInvalid("the handover is not signed by the group's host")
    if fork_seq is not None and succession.chain_hash_or_none(conn, room_id, int(fork_seq)) != statement.get(
            "last_hash"):
        raise ProofInvalid("the handover names a history this computer does not hold")
    return dict(statement)


# --- the standby -----------------------------------------------------------------------------------
def accept(ctx, room_id: str, proof: Mapping[str, Any], *, release: Mapping[str, Any] | None = None,
           finish_now: bool = True, at_risk: int = 0) -> dict[str, Any]:
    """The standby continues the group the host handed over, after checking it against its own copy.

    ``release`` is the host's token that lets each voter give its lease back at once (else the lease
    runs out by itself first). ``finish_now=False`` returns once the transition is written (the host is
    waiting on the answer); upkeep then finishes the move like any other."""
    from gateway import hosted_room_replicas as replicas
    from gateway import hosted_room_succession_move as move
    statement = proof.get("statement") if isinstance(proof, Mapping) else None
    me = succession.local_install_id()
    if not isinstance(statement, Mapping) or statement.get("successor") != me:
        raise ProofInvalid("the handover names another computer")
    current = move.view(ctx, room_id)
    host = current["head"]["authority_gateway_id"]
    # Catch up from the host to exactly the history it handed over, as far as the head it signs vouches.
    if current["watermark"]["seq"] < int(statement["last_seq"]):
        move.catch_up(ctx, room_id, host)
        current = move.view(ctx, room_id)
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        verify_locked(conn, room_id, proof, from_epoch=current["head"]["authority_epoch"],
                      to_epoch=int(statement["to_epoch"]), successor=me, fork_seq=int(current["watermark"]["seq"]))
    record = {**(current["move"] or {}), "state": "moving", "step": "fencing", "reason": "handover",
              "started_at": time.time(), "from_epoch": current["head"]["authority_epoch"],
              "previous_host": host, "origin_install_id": current["origin"], "transition_committed": False,
              "to_epoch": int(statement["to_epoch"]), "handover": dict(proof)}
    move._save(ctx, room_id, "move", record)
    outcomes = move._ask_fences(ctx, room_id, current["configuration"], int(statement["to_epoch"]),
                                current["watermark"], handover=dict(proof),
                                release=dict(release) if isinstance(release, Mapping) else None)
    # Another computer holding this step (the owner continued the paused host anyway) wins over it.
    move._refused_for_other(current["configuration"], outcomes)
    fenced = {k: v for k, v in outcomes.items() if not isinstance(v, move.RemoteRefusal)}
    if me not in fenced:
        raise SuccessionError("this computer could not fence its own copy", reason="target_not_ready")
    record = {**record, "step": "reconciling", "receipts": [fenced[k]["receipt"] for k in sorted(fenced)],
              "evidence": {k: v["run_evidence"] for k, v in fenced.items()},
              "grants": {k: v["continuation_grants"] for k, v in fenced.items() if v["continuation_grants"]},
              "unreachable": sorted(k for k, v in outcomes.items() if isinstance(v, move.RemoteRefusal)),
              "adopted_watermark": dict(current["watermark"])}
    configuration = current["configuration"]
    transition = {"proof_kind": "handover", "proof_digest": succession.proof_digest(proof), "proof": dict(proof)}
    replicas.promote_replica(
        ctx.db_path, room_id=room_id, transition=transition, to_epoch=int(statement["to_epoch"]),
        text=succession.transition_text(succession.label(configuration, me)),
        display={"from_name": succession.label(configuration, host), "to_name": succession.label(configuration, me),
                 "offline_since": None, "reason": "handover", "at_risk": max(0, int(at_risk))})
    record = {**record, "transition_committed": True, "proof_digest": transition["proof_digest"],
              "owner_subject": record.get("owner_subject") or _recorded_owner(ctx, room_id)}
    move._save(ctx, room_id, "move", record)
    if not finish_now:
        upkeep = getattr(getattr(ctx, "service", None), "succession", None)
        if upkeep is not None:
            upkeep.wakeup()
        return record
    return move.finish(ctx, room_id, record)


def _recorded_owner(ctx, room_id: str) -> str | None:
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        return succession.owner_subject_locked(conn, room_id)


# --- the host --------------------------------------------------------------------------------------
def unsettled_turns(db_path, room_id: str) -> list[str]:
    """This host's turns that started and have no outcome yet."""
    from gateway import hosted_room_driver as driver
    try:
        return sorted(str(task["identity"].task_id) for status in _RUNNING
                      for task in driver.list_tasks(db_path, room_id=room_id, status=status))
    except Exception:
        return []


def drain(ctx, room_id: str, seconds: float) -> list[str]:
    """Wait, bounded, for the host's running turns to settle, publishing their outcomes into the log the
    statement will cover; nothing new starts meanwhile (the group is paused). Returns the turns still
    running."""
    service = getattr(ctx, "service", None)
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        if service is not None:
            try:
                service.publish_settled(room_id)
            except Exception:
                pass  # the next look publishes it
        running = unsettled_turns(ctx.db_path, room_id)
        if not running or time.monotonic() >= deadline:
            return running
        time.sleep(DRAIN_POLL_SECONDS)


def _target(ctx, room_id: str, target_install_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """This host's view and the computer it hands its group to: one allowed to continue it, with an endpoint."""
    from gateway import hosted_room_succession_move as move
    current = move.view(ctx, room_id)
    configuration, head = current["configuration"], current["head"]
    if not head["authoritative"]:
        raise SuccessionError("this computer does not host the group", reason="room_not_found")
    target = succession.custodians_by_id(configuration).get(target_install_id)
    if target is None or not target.get("successor") or not target.get("endpoint"):
        raise SuccessionError("that computer can't continue this group", reason="target_not_ready",
                              detail={"target": move.named(configuration, target_install_id)})
    return current, target


def hand_over(ctx, room_id: str, target_install_id: str, *, drain_seconds: float = DRAIN_SECONDS) -> dict[str, Any]:
    """The host hands the group to ``target_install_id`` and steps down; on any refusal it resumes.

    It pauses first, then lets its running turns settle for at most ``drain_seconds``, so their
    outcomes are part of the history it signs over. Turns still running then are counted at risk on
    the new host, which inherits them as unknown: never run again by themselves."""
    current, target = _target(ctx, room_id, target_install_id)
    record = {**(succession.load_record(ctx.db_path, room_id, "move") or {}), "state": "handing_over",
              "step": "fencing", "reason": "handover", "started_at": time.time(), "to": target_install_id,
              "from_epoch": current["head"]["authority_epoch"], "signed": False}
    _pause(ctx, room_id, record)
    return _complete(ctx, room_id, target_install_id, target, record, drain_seconds)


def request_move(ctx, room_id: str, target_install_id: str) -> None:
    """``groups.succession.move``: the owner moves the group while its host is up. With no turn running it
    hands over at once; otherwise the group pauses and upkeep hands over once its turns settle, at most
    ``MOVE_DRAIN_SECONDS`` later, or at once on ``move_now``."""
    current, target = _target(ctx, room_id, target_install_id)
    if not unsettled_turns(ctx.db_path, room_id):
        hand_over(ctx, room_id, target_install_id, drain_seconds=0)
        return
    _pause(ctx, room_id, {**(succession.load_record(ctx.db_path, room_id, "move") or {}), "state": "handing_over",
                          "step": "waiting_for_turns", "reason": "handover", "started_at": time.time(),
                          "to": target_install_id, "from_epoch": current["head"]["authority_epoch"],
                          "signed": False, "drain_until": time.time() + MOVE_DRAIN_SECONDS, "now": False})
    if ctx.service is not None:
        ctx.service.wakeup()


def move_now(ctx, room_id: str) -> None:
    """The owner moves a group waiting for its turns at once: turns still running become unknown there."""
    record = succession.load_record(ctx.db_path, room_id, "move") or {}
    if record.get("state") != "handing_over" or record.get("step") != "waiting_for_turns":
        raise SuccessionError("this group isn't waiting to move", reason="invalid_params")
    succession.save_record(ctx.db_path, room_id, "move", {**record, "now": True})
    continue_move(ctx, room_id)


def continue_move(ctx, room_id: str) -> bool:
    """Upkeep for a move waiting for its turns: publish what settled, and hand over once nothing runs, at
    the deadline, or when the owner said Move now. True once the group was handed over."""
    record = succession.load_record(ctx.db_path, room_id, "move") or {}
    if record.get("state") != "handing_over" or record.get("step") != "waiting_for_turns" or (
            str(ctx.db_path), room_id) in _active:
        return False
    running = drain(ctx, room_id, 0)
    if running and not record.get("now") and time.time() < float(record.get("drain_until") or 0):
        return False
    try:
        current, target = _target(ctx, room_id, str(record.get("to")))
    except SuccessionError as exc:
        _resume(ctx, room_id, exc)
        return False
    record = {**record, "step": "fencing", "from_epoch": current["head"]["authority_epoch"]}
    _pause(ctx, room_id, record)
    _complete(ctx, room_id, str(record["to"]), target, record, 0)
    return True


def _complete(ctx, room_id: str, target_install_id: str, target: Mapping[str, Any], record: dict[str, Any],
              drain_seconds: float) -> dict[str, Any]:
    """Drain, sign exactly this history over, send it, and step down to the standby's transition."""
    from gateway import hosted_room_fence as fence
    from gateway import hosted_room_succession_return as returning
    own = fence.room_fence_state(ctx.runs_store.path, room_id)
    to_epoch = max(int(record["from_epoch"]), int(own["fenced_epoch"] or 0)) + 1
    record = {**record, "to_epoch": to_epoch}
    _pause(ctx, room_id, record)
    _active.add((str(ctx.db_path), room_id))
    try:
        if ctx.service is not None:
            ctx.service.wakeup()  # nothing new starts; a turn already running finishes into the log
        running = drain(ctx, room_id, drain_seconds)
        with closing(rooms._read_connection(ctx.db_path)) as conn:
            proof, release = signed_over(ctx, conn, room_id, successor=target_install_id, to_epoch=to_epoch)
        # From here the standby may continue: neither an unknown outcome nor a restart resumes blindly.
        _pause(ctx, room_id, {**record, "signed": True, "unsettled": running})
        demoted, stepped = _send(ctx, room_id, target_install_id, str(target["endpoint"]), proof,
                                 release=release, at_risk=len(running))
    except Exception as exc:
        if isinstance(exc, _NotContinued) or not (succession.load_record(ctx.db_path, room_id, "move") or {}).get(
                "signed"):
            _resume(ctx, room_id, exc.cause if isinstance(exc, _NotContinued) else exc)
        else:  # the standby may have continued: upkeep learns the outcome (``recover``) before resuming
            _attempt_failed(ctx, room_id, exc)
        cause = exc.cause if isinstance(exc, _NotContinued) else exc
        if isinstance(cause, SuccessionError):
            raise cause from None
        raise SuccessionError("the standby did not continue the group", reason="target_not_ready") from cause
    finally:
        _active.discard((str(ctx.db_path), room_id))
    record = succession.load_record(ctx.db_path, room_id, "move") or {}
    succession.save_record(ctx.db_path, room_id, "move", {**record, "state": "handed_over", "step": None,
                                                         "handed_over_at": time.time()})
    returning.follow_up(ctx, room_id)
    return demoted or stepped


# Handovers this process is running now, as ``(db_path, room_id)``.
_active: set[tuple[str, str]] = set()


class _NotContinued(Exception):
    """The standby certainly did not continue: it refused, or the statement never reached it."""

    def __init__(self, cause: Exception):
        super().__init__(str(cause))
        self.cause = cause


def _send(ctx, room_id: str, target_install_id: str, endpoint: str, proof: Mapping[str, Any], *,
          release: Mapping[str, Any] | None = None, at_risk: int = 0):
    """Post the signed statement (and its release token) to the standby and step down to its verified
    transition. ``at_risk`` counts this host's turns still running: their outcomes won't reach the new
    host."""
    import socket
    from gateway import hosted_room_succession_move as move
    from gateway import hosted_room_succession_return as returning
    unsigned = {"room_id": room_id, "host_install_id": succession.local_install_id(), "proof": dict(proof),
                **({"release": dict(release)} if release is not None else {}),
                "at_risk": int(at_risk), "issued_at": time.time(), "nonce": succession.nonce()}
    try:
        reply = (ctx.post or move.http_post)(endpoint, "/v1/room-members/succession/handover",
                                             {**unsigned, "signature": succession.sign(HANDOVER, unsigned)},
                                             HANDOVER_TIMEOUT_SECONDS)
    except move.RemoteRefusal as exc:
        raise _NotContinued(SuccessionError(str(exc), reason=exc.code, detail=exc.detail)) from exc
    except (TimeoutError, socket.timeout):
        raise  # it may have arrived and the standby may have continued: learn the outcome first
    except OSError as exc:
        if "timed out" in str(exc):
            raise
        raise _NotContinued(exc) from exc  # it never reached the standby
    transition = reply.get("transition") if isinstance(reply, Mapping) else None
    if not isinstance(transition, Mapping):
        raise SuccessionError("the standby did not continue the group", reason="target_not_ready")
    demoted = returning.step_down(ctx, room_id, transition["event"], transition.get("fork_event"), follow=False)
    stepped = succession.load_record(ctx.db_path, room_id, "return") or {}
    if demoted is None and not (stepped.get("state") == "stepped_down"
                                and stepped.get("successor") == target_install_id):
        raise ProofInvalid("the standby's transition could not be verified here")
    return demoted, stepped


def _attempt_failed(ctx, room_id: str, exc: Exception) -> None:
    """Keep the reason while the host stays paused, waiting to learn whether the standby continued."""
    from gateway import hosted_room_succession_move as move
    record = succession.load_record(ctx.db_path, room_id, "move") or {}
    if record.get("state") != "handing_over":
        return
    configuration = move.view(ctx, room_id)["configuration"]
    succession.save_record(ctx.db_path, room_id, "move", {**record, "last_attempt": {
        "to": move.named(configuration, record.get("to")), "at": time.time(),
        "error": getattr(exc, "reason", None) or "target_not_ready"}})


def _pause(ctx, room_id: str, record: Mapping[str, Any]) -> None:
    succession.save_record(ctx.db_path, room_id, "move", dict(record))


def recover(ctx, room_id: str) -> bool:
    """A handover this process no longer runs (it restarted mid-way): resume when nothing can have
    moved, else wait to learn the outcome. True when the host serves again.

    Before the signature nothing left this computer. After it, the host resumes only once the
    standby answers that it holds no step beyond the host's epoch; ``check`` steps it down instead
    when the standby did continue.
    """
    from gateway import hosted_room_succession_move as move
    from gateway.hosted_room_succession_return import check
    record = succession.load_record(ctx.db_path, room_id, "move") or {}
    if record.get("state") != "handing_over" or (str(ctx.db_path), room_id) in _active or record.get(
            "step") == "waiting_for_turns":
        return False  # a move waiting for its turns continues in ``continue_move``
    if not record.get("signed"):
        _resume(ctx, room_id, SuccessionError("the handover was interrupted", reason="target_not_ready"))
        return True
    if check(ctx, room_id) is not None or _stepped_down_to(ctx, room_id, record.get("to")):
        # The standby continued and this computer follows it now: the handover is done.
        from gateway.hosted_room_succession_return import follow_up
        succession.save_record(ctx.db_path, room_id, "move", {**record, "state": "handed_over", "step": None,
                                                             "handed_over_at": time.time()})
        follow_up(ctx, room_id)
        return False
    configuration = move.view(ctx, room_id)["configuration"]
    target = succession.custodians_by_id(configuration).get(record.get("to")) or {}
    if not target.get("endpoint"):
        return False
    try:
        answer = move._query(ctx, room_id, record["to"], str(target["endpoint"]))
    except Exception:
        return False
    fence_state = answer.get("fence") if isinstance(answer.get("fence"), Mapping) else {}
    promise = fence_state.get("promise") if isinstance(fence_state.get("promise"), Mapping) else None
    beyond = int(record.get("from_epoch") or 0)
    if answer.get("transition"):
        return False  # the standby continued: ``check`` steps this host down once it can verify that
    if int(fence_state.get("fenced_epoch") or 0) >= beyond or (
            promise is not None and int(promise.get("epoch") or 0) > beyond):
        # The standby began and stopped: hand the same history over again, so it completes.
        try:
            with closing(rooms._read_connection(ctx.db_path)) as conn:
                proof, release = signed_over(ctx, conn, room_id, successor=record["to"],
                                             to_epoch=int(record.get("to_epoch") or beyond + 1))
            _send(ctx, room_id, record["to"], str(target["endpoint"]), proof, release=release)
        except Exception as exc:
            _attempt_failed(ctx, room_id, exc)
        return False
    _resume(ctx, room_id, SuccessionError("the handover was interrupted", reason="target_not_ready"))
    return True


def _stepped_down_to(ctx, room_id: str, successor: Any) -> bool:
    stepped = succession.load_record(ctx.db_path, room_id, "return") or {}
    return stepped.get("state") in {"stepped_down", "rebased"} and stepped.get("successor") == successor


def _resume(ctx, room_id: str, exc: Exception) -> None:
    from gateway import hosted_room_succession_move as move
    record = succession.load_record(ctx.db_path, room_id, "move") or {}
    if record.get("state") != "handing_over":
        return
    configuration = move.view(ctx, room_id)["configuration"]
    succession.save_record(ctx.db_path, room_id, "move", {
        **{k: v for k, v in record.items() if k != "step"}, "state": "failed", "last_attempt": {
            "to": move.named(configuration, record.get("to")), "at": time.time(),
            "error": getattr(exc, "reason", None) or "target_not_ready"}})
    if ctx.service is not None:
        ctx.service.wakeup()


def answer_handover(context, request: Mapping[str, Any]) -> dict[str, Any]:
    """The standby's endpoint: a host it pins hands it the group."""
    from gateway.hosted_room_succession_backup import check_request_locked
    request = dict(request)
    with closing(rooms._read_connection(context.custody_db)) as conn:
        room_id = check_request_locked(conn, request, domain=HANDOVER, requester_field="host_install_id")
        head = conn.execute("SELECT authority_gateway_id FROM hosted_room_replicas WHERE room_id=?",
                            (room_id,)).fetchone()
    if head is None or head[0] != request["host_install_id"]:
        raise SuccessionError("this copy follows another host", reason="room_authority_superseded")
    ctx = context.move_context()
    at_risk = request.get("at_risk") if type(request.get("at_risk")) is int else 0
    accept(ctx, room_id, request["proof"], release=request.get("release"), finish_now=False, at_risk=at_risk)
    with closing(rooms._read_connection(context.custody_db)) as conn:
        return {"room_id": room_id, "transition": succession.latest_transition_locked(conn, room_id)}


def handover_all(ctx, *, reason: str, drain_seconds: float | None = None) -> dict[str, Any]:
    """Hand every group this computer hosts, and the caller owns, to its best reachable standby (stop,
    quit or sleep); the others are skipped with ``not_owner``."""
    from gateway.hosted_room_succession_status import held_rooms
    moved, skipped = [], []
    from gateway import hosted_room_succession_move as move
    for room_id in held_rooms(ctx.db_path)["hosted"]:
        if not move.is_owner(ctx, room_id):
            skipped.append({"room_id": room_id, "reason": "not_owner"})
            continue
        try:
            target = best_standby(ctx, room_id)
            if target is None:
                skipped.append({"room_id": room_id, "reason": "no_standby"})
                continue
            hand_over(ctx, room_id, target, drain_seconds=(
                drain_seconds if drain_seconds is not None else 1.0 if reason == "sleep" else DRAIN_SECONDS))
            moved.append(room_id)
        except SuccessionError as exc:
            skipped.append({"room_id": room_id, "reason": exc.reason})
        except Exception:
            skipped.append({"room_id": room_id, "reason": "target_not_ready"})
    return {"moved": moved, "skipped": skipped, "reason": reason}


def best_standby(ctx, room_id: str) -> str | None:
    """The best placed eligible computer that answers now, if any."""
    from gateway import hosted_room_succession_move as move
    from gateway.hosted_room_succession_status import backups, ranked_targets
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        configuration = succession.configuration_locked(conn, room_id)
        latest = int(conn.execute("SELECT next_seq - 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()[0])
    answers = move.survey(ctx, room_id, configuration)
    heard = {"peers": {install_id: {"ok": bool(outcome.get("answer")), "last_ok": time.time(),
                                    "last_attempt": time.time(),
                                    "watermark": (outcome.get("answer") or {}).get("watermark")}
                       for install_id, outcome in answers.items()}}
    rows = backups(configuration, succession.custody_status(ctx.db_path, room_id),
                   host_id=succession.local_install_id(), latest_seq=latest, heartbeat=heard, now=time.time())
    for install_id in ranked_targets(rows):
        if (answers.get(install_id) or {}).get("answer"):
            return install_id
    return None
