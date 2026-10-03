"""Moving a Group Chat by itself when its host goes offline: the host's lease, a standby's takeover,
careful mode for two voters, and continuing a paused host anyway.

The group's voters are its host and its always-on successors in the owner's order (#104601's
``custody.configured``); their number sets the mode (``mode_of``): ``majority`` with three or more,
``careful`` with exactly two, and ``ask`` with fewer or when the owner switched automatic moves off.
``ask`` never moves by itself: the owner continues by hand (``hosted_room_succession_move``).

**Majority.** The host asks for a lease with each push to a voter, about every five seconds, and a
voter grants ``LEASE_SECONDS`` on its sleep-counting clock: until then it promises no later epoch to
anyone (#105079). The host counts each grant from the moment it asked, less drift and a margin, and
admits, dispatches and appends only while it holds grants from a majority of every current voter
set, itself included; a gap in its heartbeats or a detected sleep voids them all. Without that
majority it pauses to stay safe (``lost_majority``). A standby that has heard nothing from the host
for the lease and ``TRIGGER_GRACE_SECONDS``, then its rank's delay and some jitter, first asks the
voters whether they would promise; only when a majority would does it ask for promises, sign a
certificate of them (proof ``certified``), catch up from the most complete promiser and continue
(``reason: automatic``). Any two majorities share a voter, and that voter's lease or promise stops
the other side, so two hosts never admit at the same time.

**Careful (two voters).** The standby moves only after hearing nothing from the host, and the host
nothing from it, for ``CAREFUL_SILENCE_SECONDS`` in both directions, while it passes its own online
check and no restart window is open. It signs that evidence (proof ``evidence``). The host pauses
(``isolated``) once it has heard nothing from the standby for half that time and fails its own
online check. A host that kept writing anyway finds ``continued_on_two`` when the two meet: the
owner keeps one history, and the other one is set aside.

**Continue anyway.** On a paused host, the owner may continue the group there regardless: a fresh
epoch at every computer it reaches, attested by the owner, served without a lease until a majority
answers again.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import random
import socket
import threading
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from gateway import hosted_room_clock as clock
from gateway import hosted_room_succession as succession
from gateway import hosted_rooms as rooms
from gateway.hosted_room_succession import SuccessionError

logger = logging.getLogger(__name__)

LEASE_SECONDS = 20.0
# The host counts a grant from when it asked, less this share for clock drift and a margin.
DRIFT = 0.01
MARGIN_SECONDS = 1.0
# A heartbeat loop that stalled this long can't vouch for the grants it held before.
GAP_SECONDS = 10.0
TRIGGER_GRACE_SECONDS = 10.0
RANK_STEP_SECONDS = 10.0
JITTER_SECONDS = 5.0
BACKOFF_SECONDS = (5.0, 60.0)
PROBE_SECONDS = 10.0
ISOLATED_AFTER_SECONDS = succession.CAREFUL_SILENCE_SECONDS / 2
ONLINE_CHECK_SECONDS = 30.0
TICK_SECONDS = 2.0
PAUSED_REASONS = frozenset({"lost_majority", "isolated", "no_lease_layer"})
AUTOMATIC_MODES = frozenset({"majority", "careful"})
# Hosts a computer already talks to, by messaging platform: the online check contacts nothing new.
_PLATFORM_HOSTS = {"telegram": "api.telegram.org", "discord": "discord.com", "slack": "slack.com",
                   "whatsapp": "graph.facebook.com"}


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _custody():
    from gateway import hosted_room_custody as custody
    return custody


def mode_of(configuration: Mapping[str, Any] | None) -> str:
    return _custody().mode_of(configuration or {})


def standbys(configuration: Mapping[str, Any]) -> list[str]:
    """The voters that may take over, in the owner's order: every voter but the host."""
    host = (succession.host_entry(configuration) or {}).get("install_id")
    return [voter for voter in succession.voters_of(configuration)
            if voter != host and succession.is_eligible(configuration, voter)]


# --- the host's lease ----------------------------------------------------------------------------
class HostLease:
    """The host's view of its voters' grants, in memory only: a restarted host starts without one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._grants: dict[str, dict[str, Any]] = {}
        self._asked: dict[str, float] = {}
        self._void_before: dict[str, float] = {}
        self._extended: dict[str, float] = {}
        self._detector = clock.SuspendDetector()

    def _slept_locked(self, now: float) -> None:
        if self._detector.check():
            for room_id in set(self._grants) | set(self._asked):
                self._void_locked(room_id, now)

    def _void_locked(self, room_id: str, now: float) -> None:
        self._grants.pop(room_id, None)
        self._void_before[room_id] = now

    def request(self, room_id: str, epoch: int) -> dict[str, Any]:
        """One push's lease request; a heartbeat loop that stalled first voids what it held."""
        now = clock.now()
        with self._lock:
            self._slept_locked(now)
            last = self._asked.get(room_id)
            if last is not None and now - last > GAP_SECONDS:
                self._void_locked(room_id, now)
            self._asked[room_id] = now
            until = self._extended.get(room_id)
        request = {"epoch": int(epoch), "duration_s": LEASE_SECONDS, "sent_at": now, "boot": clock.boot_id()}
        if until is not None and until > time.time():
            request["until"] = until
        return request

    def void(self, room_id: str, before: float) -> None:
        """Count no grant asked for before ``before`` (this clock): the host signed a handover then,
        and a voter may give such a grant back for it."""
        with self._lock:
            self._grants.pop(room_id, None)
            self._void_before[room_id] = max(self._void_before.get(room_id, -math.inf), float(before))

    def acknowledged(self, room_id: str, voter: str, grant: Any, sent_at: Any) -> bool:
        """Count a voter's grant from the moment it was asked for, less drift and a margin."""
        granted = grant.get("granted_until_s") if isinstance(grant, Mapping) else None
        epoch = grant.get("epoch") if isinstance(grant, Mapping) else None
        if not (_finite(sent_at) and _finite(granted) and granted > 0 and type(epoch) is int):
            self.lost(room_id, voter)
            return False
        until = float(sent_at) + float(granted) * (1.0 - DRIFT) - MARGIN_SECONDS
        with self._lock:
            if float(sent_at) < self._void_before.get(room_id, -math.inf):
                return False
            entry = self._grants.get(room_id)
            if entry is not None and entry["epoch"] > epoch:
                return False
            if entry is None or entry["epoch"] < epoch:
                entry = self._grants[room_id] = {"epoch": epoch, "until": {}}
            entry["until"][voter] = max(entry["until"].get(voter, -math.inf), until)
        return True

    def lost(self, room_id: str, voter: str) -> None:
        with self._lock:
            entry = self._grants.get(room_id)
            if entry is not None:
                entry["until"].pop(voter, None)

    def remaining(self, room_id: str, epoch: int, voter_sets: list[list[str]], me: str) -> float:
        """Seconds the host still holds grants from a majority of every voter set, itself counted."""
        now = clock.now()
        with self._lock:
            self._slept_locked(now)
            entry = self._grants.get(room_id)
            untils = dict(entry["until"]) if entry is not None and entry["epoch"] == epoch else {}
        left = math.inf
        for voters in voter_sets or [[me]]:
            need = succession.majority(len(voters)) - (1 if me in voters else 0)
            if need <= 0:
                continue
            held = sorted((untils.get(voter, -math.inf) for voter in voters if voter != me), reverse=True)
            if len(held) < need:
                return 0.0
            left = min(left, held[need - 1] - now)
        return max(0.0, left)

    def holders(self, room_id: str, epoch: int) -> set[str]:
        """Voters whose grant still runs, as this host counts it."""
        now = clock.now()
        with self._lock:
            entry = self._grants.get(room_id)
            if entry is None or entry["epoch"] != epoch:
                return set()
            return {voter for voter, until in entry["until"].items() if until > now}

    def extend(self, room_id: str, until: float | None) -> None:
        """Ask the voters to keep the grant until ``until`` (wall clock): a planned restart."""
        with self._lock:
            if until is None:
                self._extended.pop(room_id, None)
            else:
                self._extended[room_id] = float(until)



# --- one computer's automatic moves --------------------------------------------------------------
@dataclass
class _Attempt:
    next_at: float = 0.0
    failures: int = 0
    jitter: float = field(default_factory=lambda: random.uniform(0.0, JITTER_SECONDS))


class Automatic:
    """The lease, the contact both ways, and the takeover checks of one gateway."""

    def __init__(self, context_factory: Callable[[], Any], *, online_check: Callable[[], bool] | None = None,
                 wake: Callable[[], None] | None = None):
        self._context_factory = context_factory
        self._online_check = online_check or online_check_default
        self._wake = wake
        self.lease = HostLease()
        self._lock = threading.Lock()
        self._started = clock.now()
        self._heard_host: dict[str, tuple[float, float]] = {}
        self._following: dict[str, tuple[str, int]] = {}
        self._heard_voter: dict[str, dict[str, float]] = {}
        self._probed: dict[str, float] = {}
        self._attempts: dict[str, _Attempt] = {}
        self._news: dict[str, list[dict[str, Any]]] = {}
        self._paused: dict[str, dict[str, Any]] = {}
        self._restored: set[str] = set()
        self._online: tuple[float, bool] | None = None
        self._db: str | None = None

    def db_path(self) -> str | None:
        """The room store this lease layer holds the groups of (its hosted service's)."""
        if self._db is None:
            ctx = self.context()
            self._db = _resolved(ctx.db_path) if ctx is not None else None
        return self._db

    def fence_path(self):
        """This computer's fence store (its Runs store), when it has a durable one."""
        ctx = self.context()
        store = getattr(ctx, "runs_store", None) if ctx is not None else None
        return getattr(store, "path", None)

    def hosts(self, room_id: str) -> bool:
        ctx = self.context()
        if ctx is None:
            return False
        with closing(rooms._read_connection(ctx.db_path)) as conn:
            row = conn.execute("SELECT authority_gateway_id FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL",
                               (room_id,)).fetchone()
        return row is not None and row[0] == succession.local_install_id()

    def keeps_copy(self, room_id: str) -> bool:
        ctx = self.context()
        if ctx is None:
            return False
        with closing(rooms._read_connection(ctx.db_path)) as conn:
            return succession.table_exists(conn, "hosted_room_replicas") and conn.execute(
                "SELECT 1 FROM hosted_room_replicas WHERE room_id=? AND disbanded_at IS NULL", (room_id,)).fetchone(
            ) is not None

    # contact, both ways
    def heard_from_host(self, room_id: str) -> None:
        with self._lock:
            self._heard_host[room_id] = (clock.now(), time.time())

    def follows(self, room_id: str, host: str, epoch: int) -> None:
        """A copy that starts following a host (after a move, a step down or a going back) has just heard
        of it: its silence counts from now, never from before it followed."""
        with self._lock:
            if self._following.get(room_id) == (host, int(epoch)):
                return
            self._following[room_id] = (host, int(epoch))
            self._heard_host[room_id] = (clock.now(), time.time())

    def heard_from_voter(self, room_id: str, voter: str) -> None:
        with self._lock:
            self._heard_voter.setdefault(room_id, {})[voter] = clock.now()

    def host_silence(self, room_id: str, *, reported_at: float | None = None) -> tuple[float, float]:
        """``(seconds without contact with the host, wall time of the last contact)``: a lease request, a
        probe answered, or the copy's own record of the host's last verified push (``reported_at``, every
        push and heartbeat, with or without a lease). A restart counts as contact, so silence is measured
        only from this process's start."""
        now, wall = clock.now(), time.time()
        with self._lock:
            at, last = self._heard_host.get(room_id, (self._started, wall - (now - self._started)))
        silent = now - at
        if reported_at is not None and wall - reported_at < silent:
            silent, last = max(0.0, wall - reported_at), reported_at
        return silent, last

    def voter_silence(self, room_id: str, voter: str, *, acknowledged_at: float | None = None) -> float:
        """Seconds without contact with ``voter``: its answers to this host's pushes (and their durable
        record, ``acknowledged_at``) and its signed requests."""
        with self._lock:
            at = self._heard_voter.get(room_id, {}).get(voter, self._started)
        silent = clock.now() - at
        if acknowledged_at is not None:
            silent = min(silent, max(0.0, time.time() - acknowledged_at))
        return silent

    def online(self) -> bool:
        now = clock.now()
        if self._online is not None and now - self._online[0] < ONLINE_CHECK_SECONDS:
            return self._online[1]
        try:
            result = bool(self._online_check())
        except Exception:
            result = False
        self._online = (now, result)
        return result

    def context(self):
        try:
            return self._context_factory()
        except Exception:
            return None

    # the hooks #104601 calls
    def request(self, room_id: str) -> dict[str, Any] | None:
        """``lease_request_provider``: on the host, for each push to a voter."""
        ctx = self.context()
        if ctx is None:
            return None
        with closing(rooms._read_connection(ctx.db_path)) as conn:
            room = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                                "AND disbanded_at IS NULL", (room_id,)).fetchone()
            configuration = succession.configuration_locked(conn, room_id) if room is not None else None
        if room is None or room[0] != succession.local_install_id() or mode_of(configuration) == "ask":
            return None
        if succession.paused_reason(ctx.db_path, room_id) in {"room_authority_conflict", "room_authority_promised"}:
            return None  # stepping aside: let the lease run out so the kept host can gather its promises
        return self.lease.request(room_id, int(room[1]))

    def grant(self, room_id: str, epoch: int, authority: str, request: Any) -> dict[str, Any] | None:
        """``lease_grant_hook``: on a voter, after it stored a push from the host it follows."""
        from gateway import hosted_room_fence as fence
        self.heard_from_host(room_id)
        ctx = self.context()
        duration = request.get("duration_s") if isinstance(request, Mapping) else None
        if ctx is None or getattr(ctx, "runs_store", None) is None or not _finite(duration) or duration <= 0:
            return {"refused": "lease_unavailable"}  # no fence store here: this computer can't hold a lease
        until, sent_at, boot = request.get("until"), request.get("sent_at"), request.get("boot")
        try:
            granted = fence.grant_lease(ctx.runs_store.path, room_id=room_id, epoch=int(epoch),
                                        authority_install_id=str(authority),
                                        duration_s=min(float(duration), fence.MAX_LEASE_SECONDS),
                                        until=float(until) if _finite(until) else None,
                                        host_sent_at=float(sent_at) if _finite(sent_at) else None,
                                        host_boot=boot if isinstance(boot, str) else None)
        except fence.RoomFenceError as exc:
            # Refusing an older epoch's host, a voter hands it the chain that superseded it.
            with closing(rooms._read_connection(ctx.db_path)) as conn:
                events = succession.chain_after_locked(conn, room_id, int(epoch))
            return {"refused": exc.code, **({"events": events} if events else {})}
        except (TypeError, ValueError):
            return {"refused": "lease_unavailable"}
        return {"granted_until_s": float(granted["duration_s"]), "epoch": int(epoch)}

    def acknowledged(self, room_id: str, voter: str, grant: Any, sent_at: Any) -> None:
        """``lease_ack_hook``: on the host, a voter's answer to its request."""
        self.heard_from_voter(room_id, voter)
        if isinstance(grant, Mapping) and grant.get("refused"):
            self.lease.lost(room_id, voter)
            events = grant.get("events")
            if isinstance(events, list) and events:
                with self._lock:
                    self._news[room_id] = list(events)
                if self._wake is not None:
                    self._wake()
            return
        self.lease.acknowledged(room_id, voter, grant, sent_at)

    def remaining(self, room_id: str) -> float | None:
        """``lease_remaining_provider``: how long this host may still wait for a send, in majority mode."""
        ctx = self.context()
        if ctx is None:
            return None
        info = room_view(ctx, room_id)
        if info is None or not info["hosts"] or info["mode"] != "majority":
            return None
        if anyway_epoch(ctx.db_path, room_id) == info["epoch"]:
            return None
        left = self.lease.remaining(room_id, info["epoch"], info["voter_sets"], info["me"])
        return None if math.isinf(left) else left

    def serving(self, room_id: str) -> bool | None:
        """``serving_provider``: whether this host may append for the room now."""
        ctx = self.context()
        if ctx is None:
            return None
        from gateway.hosted_room_succession_backup import serving_locked
        with closing(rooms._read_connection(ctx.db_path)) as conn:
            hosts = conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=? AND authority_gateway_id=? "
                                 "AND disbanded_at IS NULL", (room_id, succession.local_install_id())).fetchone()
            return serving_locked(conn, room_id) if hosts else None

    # why the host is paused
    def paused_reason(self, room_id: str, ctx=None) -> str | None:
        """``lost_majority`` or ``isolated`` while this host may not serve the room; else None. ``ctx`` is the
        caller's own context when it has one (status), so a lease layer still starting never reads as serving."""
        with self._lock:
            isolated = self._paused.get(room_id, {}).get("reason") == "isolated"
        ctx = ctx if ctx is not None else self.context()
        if isolated:
            info = room_view(ctx, room_id) if ctx is not None else None
            return None if info is not None and anyway_epoch(ctx.db_path, room_id) == info["epoch"] else "isolated"
        info = room_view(ctx, room_id) if ctx is not None else None
        if info is None or not info["hosts"] or info["mode"] != "majority":
            return None
        if anyway_epoch(ctx.db_path, room_id) == info["epoch"]:
            return None
        if self.lease.remaining(room_id, info["epoch"], info["voter_sets"], info["me"]) > 0:
            return None
        return "lost_majority"

    def paused_view(self, ctx, room_id: str) -> dict[str, Any] | None:
        """``status.paused``: why, since when, and the voters this host is waiting for."""
        reason = self.paused_reason(room_id, ctx=ctx)
        if reason is None:
            return None
        info = room_view(ctx, room_id)
        with self._lock:
            since = self._paused.get(room_id, {}).get("since")
        holding = self.lease.holders(room_id, info["epoch"]) if info else set()
        waiting = [voter for voter in (info or {}).get("voters", ()) if voter != info["me"] and voter not in holding]
        configuration = (info or {}).get("configuration")
        return {"reason": reason, "since": since,
                "waiting_for": [{"install_id": voter, "name": succession.label(configuration, voter)}
                                for voter in waiting]}

    # the upkeep's fast checks
    def tick(self, ctx, held: Mapping[str, list[str]], *, standby=None) -> list[dict[str, Any]]:
        """Learn what refusals brought, keep the hosts' pause state, and take over where due: a standby
        needs ``standby``, this computer's full succession context (its fence store and its endpoint)."""
        results = []
        with self._lock:
            news, self._news = self._news, {}
        for room_id, events in news.items():
            try:
                from gateway.hosted_room_succession_return import learn
                learn(ctx, room_id, events)
            except Exception:
                logger.debug("group %s: a refusal's chain could not be learned", room_id, exc_info=True)
        for room_id in held.get("hosted", ()):
            try:
                self._host_tick(ctx, room_id)
            except Exception:
                logger.debug("group %s: host check failed", room_id, exc_info=True)
        for room_id in held.get("copies", ()) if standby is not None else ():
            try:
                result = self._standby_tick(standby, room_id)
            except Exception:
                logger.debug("group %s: standby check failed", room_id, exc_info=True)
                continue
            if result is not None:
                results.append(result)
        return results

    def _host_tick(self, ctx, room_id: str) -> None:
        info = room_view(ctx, room_id)
        if info is None or not info["hosts"]:
            return
        with self._lock:
            self._following.pop(room_id, None)  # if it steps aside later, it starts following afresh
        moving = succession.load_record(ctx.db_path, room_id, "move") or {}
        if moving.get("state") == "handing_over":
            from gateway.hosted_room_succession_handover import continue_move, recover
            if moving.get("step") == "waiting_for_turns":
                continue_move(ctx, room_id)  # the owner's move, once its running turns settled
            elif clock.now() - self._probed.get(room_id, -math.inf) >= PROBE_SECONDS:
                self._probed[room_id] = clock.now()
                recover(ctx, room_id)  # a handover a restart interrupted
            return
        with self._lock:
            paused = room_id in self._paused
        if paused and clock.now() - self._probed.get(room_id, -math.inf) >= PROBE_SECONDS:
            # A paused host asks the group whether it was replaced, so it steps down on first contact.
            self._probed[room_id] = clock.now()
            from gateway.hosted_room_succession_return import check
            if check(ctx, room_id) is not None:
                return
        reason = None
        anyway = anyway_epoch(ctx.db_path, room_id) == info["epoch"]
        if info["mode"] == "careful":
            other = next((voter for voter in info["voters"] if voter != info["me"]), None)
            with self._lock:
                isolated = self._paused.get(room_id, {}).get("reason") == "isolated"
            silent = self.voter_silence(room_id, other, acknowledged_at=_last_ack(ctx.db_path, room_id, other)) \
                if other else 0.0
            with self._lock:
                heard = other is not None and other in self._heard_voter.get(room_id, {})
                restored = room_id in self._restored and not heard
                if heard:
                    self._restored.discard(room_id)
            if anyway:
                if heard and silent < ISOLATED_AFTER_SECONDS:
                    clear_anyway(ctx.db_path, room_id)  # the standby answers again
            elif isolated and restored:
                reason = "isolated"  # restarted while cut off: stays paused until it hears the standby
            elif silent >= ISOLATED_AFTER_SECONDS and (isolated or not self.online()):
                reason = "isolated"
        elif info["mode"] == "majority":
            if anyway:
                if self.lease.remaining(room_id, info["epoch"], info["voter_sets"], info["me"]) > 0:
                    clear_anyway(ctx.db_path, room_id)  # a majority answers again: back to the lease
            elif self.lease.remaining(room_id, info["epoch"], info["voter_sets"], info["me"]) <= 0:
                reason = "lost_majority"
        self._set_paused(ctx, room_id, reason)

    def _set_paused(self, ctx, room_id: str, reason: str | None) -> None:
        with self._lock:
            before = self._paused.get(room_id, {}).get("reason")
            if before == reason:
                return
            if reason is None:
                self._paused.pop(room_id, None)
            else:
                self._paused[room_id] = {"reason": reason, "since": time.time()}
        record = succession.load_record(ctx.db_path, room_id, "automatic") or {}
        succession.save_record(ctx.db_path, room_id, "automatic", {
            **record, "paused": self._paused.get(room_id)})
        service = getattr(ctx, "service", None)
        if service is not None:
            service.wakeup()
        if reason is None:
            from gateway.hosted_room_succession_move import append_state
            append_state(ctx.db_path, room_id, "ok")

    def restore(self, ctx, room_ids) -> None:
        """After a restart, a host that paused itself as isolated stays paused until it hears the standby."""
        for room_id in room_ids:
            paused = (succession.load_record(ctx.db_path, room_id, "automatic") or {}).get("paused") or {}
            if paused.get("reason") == "isolated":
                with self._lock:
                    self._paused[room_id] = dict(paused)
                    self._restored.add(room_id)

    def _standby_tick(self, ctx, room_id: str) -> dict[str, Any] | None:
        info = copy_view(ctx, room_id)
        if info is None:
            with self._lock:
                self._following.pop(room_id, None)
            return None
        self.follows(room_id, info["host"], info["epoch"])
        if info["mode"] == "ask" or info["me"] not in info["standbys"]:
            return None
        now = clock.now()
        if info["restarting"]:
            self.heard_from_host(room_id)  # silence counts only after the host's restart window
            return None
        attempt = self._attempts.setdefault(room_id, _Attempt())
        if now < attempt.next_at:
            return None
        reported = _last_report(ctx.db_path, room_id)
        if info["mode"] == "careful":
            self._probe(ctx, room_id, info, now)
            silent, last = self.host_silence(room_id, reported_at=reported)
            if silent < succession.CAREFUL_SILENCE_SECONDS or not self.online():
                return None
            return self._attempt(room_id, attempt, lambda: careful_move(
                ctx, room_id, silent_since=last, silent_for=silent))
        rank = info["standbys"].index(info["me"])
        silent, last = self.host_silence(room_id, reported_at=reported)
        if silent < LEASE_SECONDS + TRIGGER_GRACE_SECONDS + rank * RANK_STEP_SECONDS + attempt.jitter:
            return None
        return self._attempt(room_id, attempt, lambda: majority_move(ctx, room_id, offline_since=last))

    def _probe(self, ctx, room_id: str, info: Mapping[str, Any], now: float) -> None:
        """Careful mode: the standby asks the host itself too, so silence means both directions."""
        if now - self._probed.get(room_id, -math.inf) < PROBE_SECONDS:
            return
        self._probed[room_id] = now
        host = succession.host_entry(info["configuration"]) or {}
        if not host.get("endpoint"):
            return
        from gateway.hosted_room_succession_move import _query
        try:
            _query(ctx, room_id, host["install_id"], str(host["endpoint"]))
        except Exception:
            return
        self.heard_from_host(room_id)

    def _attempt(self, room_id: str, attempt: _Attempt, run: Callable[[], dict[str, Any] | None]):
        try:
            result = run()
        except Exception as exc:
            attempt.failures += 1
            low, high = BACKOFF_SECONDS
            attempt.next_at = clock.now() + min(high, low * 2 ** (attempt.failures - 1)) + random.uniform(
                0.0, JITTER_SECONDS)
            if isinstance(exc, SuccessionError):
                logger.info("group %s: automatic move not made (%s)", room_id, exc.reason)
            else:  # the move's steps are recorded, so the next attempt resumes where this one stopped
                logger.warning("group %s: automatic move failed", room_id, exc_info=True)
            return None
        self._attempts.pop(room_id, None)
        return result


# --- status ----------------------------------------------------------------------------------------
def majority_reachable(configuration: Mapping[str, Any], rows: list[Mapping[str, Any]], *,
                       host_id: str | None) -> bool:
    """In majority mode, whether enough voters besides the host look reachable to move by themselves."""
    if mode_of(configuration) != "majority":
        return False
    from gateway.hosted_room_succession_status import UNAVAILABLE
    voters = succession.voters_of(configuration)
    offline = {row["install_id"] for row in rows if row["readiness"] in UNAVAILABLE}
    reachable = [voter for voter in voters if voter != host_id and voter not in offline]
    return len(reachable) >= succession.majority(len(voters))


def automatic_view(configuration: Mapping[str, Any], rows: list[Mapping[str, Any]], *,
                   host_id: str | None, pending: bool | None = None) -> dict[str, Any]:
    """``status.automatic``: whether the group moves by itself if its host goes offline, and to which computer.

    ``enabled`` is the owner's switch as the group's configuration holds it; ``pending`` the value the
    owner asked for while that change is not yet in force with the voters (else ``None``)."""
    voters = succession.voters_of(configuration)
    mode = mode_of(configuration)
    from gateway.hosted_room_succession_status import UNAVAILABLE
    readiness = {row["install_id"]: row["readiness"] for row in rows}
    # Voters that can't take part now: offline, or needing their grant renewed (their rows say which).
    offline = [voter for voter in voters if voter != host_id and readiness.get(voter) in UNAVAILABLE]
    ranked = sorted(standbys(configuration), key=lambda voter: (voter in offline, voters.index(voter)))
    enabled = configuration.get("automatic") is not False
    view: dict[str, Any] = {
        "mode": mode, "state": "ready",
        "standby": {"install_id": ranked[0], "name": succession.label(configuration, ranked[0])} if ranked else None,
        "voters": [{"install_id": voter, "name": succession.label(configuration, voter)} for voter in voters],
        "enabled": enabled, "pending": pending}
    if configuration.get("automatic") is False:
        view["state"] = "off"
    elif len(voters) < 2 or not ranked:
        view.update(state="unavailable", reason="needs_computers", needed=max(1, 2 - len(voters)))
    elif (mode == "majority" and not majority_reachable(configuration, rows, host_id=host_id)) or (
            mode == "careful" and offline):
        view.update(state="not_ready", reason="voters_offline",
                    offline=[{"install_id": voter, "name": succession.label(configuration, voter)}
                             for voter in offline])
    return view


# --- what one computer holds ---------------------------------------------------------------------
def room_view(ctx, room_id: str) -> dict[str, Any] | None:
    """The room this computer hosts: epoch, configuration, mode and the voter sets its lease needs."""
    custody = _custody()
    me = succession.local_install_id()
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        room = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                            "AND disbanded_at IS NULL", (room_id,)).fetchone()
        if room is None:
            return None
        protection = custody.protection_locked(conn, room_id, me) if room[0] == me else None
    configuration = protection["configuration"] if protection else {"voters": []}
    return {"hosts": room[0] == me, "epoch": int(room[1]), "me": me, "configuration": configuration,
            "mode": mode_of(configuration), "voters": list(configuration.get("voters") or ()),
            "voter_sets": protection["voter_sets"] if protection else []}


def copy_view(ctx, room_id: str) -> dict[str, Any] | None:
    """The copy this computer keeps: mode, standbys, and whether the host announced a restart."""
    from gateway.hosted_room_succession_move import restarting_until
    from gateway.hosted_room_succession_status import RESTART_GRACE_SECONDS
    me = succession.local_install_id()
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        copy = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_room_replicas WHERE room_id=? "
                            "AND disbanded_at IS NULL AND quarantine_reason IS NULL", (room_id,)).fetchone()
        if copy is None:
            return None
        configuration = succession.configuration_locked(conn, room_id)
        until = restarting_until(conn, room_id)
    if not succession.local_consent(ctx.db_path, room_id):
        return None
    return {"me": me, "epoch": int(copy[1]), "host": copy[0], "configuration": configuration,
            "mode": mode_of(configuration), "standbys": standbys(configuration),
            "restarting": until is not None and time.time() < until + RESTART_GRACE_SECONDS}


def anyway_epoch(db_path, room_id: str) -> int | None:
    """The epoch the owner continued this paused host at anyway, served without a lease."""
    record = succession.load_record(db_path, room_id, "automatic") or {}
    epoch = record.get("anyway_epoch")
    return epoch if type(epoch) is int else None


def clear_anyway(db_path, room_id: str) -> None:
    record = succession.load_record(db_path, room_id, "automatic") or {}
    if "anyway_epoch" in record:
        succession.save_record(db_path, room_id, "automatic", {k: v for k, v in record.items() if k != "anyway_epoch"})


# --- the takeovers -------------------------------------------------------------------------------
def _start(ctx, room_id: str, current: Mapping[str, Any], *, proof_kind: str) -> dict[str, Any]:
    from gateway.hosted_room_succession_handover import _recorded_owner
    head = current["head"]
    if head["authoritative"]:
        raise SuccessionError("this computer already hosts the group", reason="host_reachable")
    record = {**(current["move"] or {}), "state": "moving", "step": "fencing", "reason": "automatic",
              "proof_kind": proof_kind, "started_at": time.time(), "from_epoch": head["authority_epoch"],
              "previous_host": head["authority_gateway_id"], "origin_install_id": current["origin"],
              "transition_committed": False, "owner_subject": _recorded_owner(ctx, room_id)}
    record.pop("last_attempt", None)
    return record


def _would_promise(answer: Mapping[str, Any], epoch: int, me: str, configuration_seq: int) -> bool:
    """Whether a voter's signed answer leaves room for its promise of ``epoch`` to this computer."""
    fence = answer.get("fence") if isinstance(answer.get("fence"), Mapping) else {}
    promise = fence.get("promise") if isinstance(fence.get("promise"), Mapping) else None
    return not (answer.get("hosting") or answer.get("lease_active")
                or int(fence.get("fenced_epoch") or 0) >= epoch
                or (promise is not None and int(promise.get("epoch") or 0) >= epoch
                    and promise.get("candidate_install_id") != me)
                or int(answer.get("configuration_seq") or 0) > configuration_seq)


def _own_answer(ctx, room_id: str) -> dict[str, Any]:
    from gateway import hosted_room_fence as fence
    return {"fence": fence.room_fence_state(ctx.runs_store.path, room_id),
            "lease_active": fence.room_lease_state(ctx.runs_store.path, room_id) is not None}


def majority_move(ctx, room_id: str, *, offline_since: float | None = None) -> dict[str, Any] | None:
    """A standby takes over with promises from a majority of the voters (``certified``)."""
    from gateway import hosted_room_succession_move as move
    current = move.view(ctx, room_id)
    record = move.committed_transition(ctx, room_id, dict(current["move"] or {}))
    if record.get("state") == "moving" and record.get("transition_committed"):
        return move.finish(ctx, room_id, record)
    configuration, me = current["configuration"], succession.local_install_id()
    voters = succession.voters_of(configuration)
    if mode_of(configuration) != "majority" or me not in standbys(configuration):
        raise SuccessionError("this computer may not take this group over by itself", reason="target_not_ready")
    epoch = move.next_epoch(ctx, room_id, current, record)
    seq = int(configuration.get("configuration_seq") or 0)
    # Ask first, changing nothing: a voter still holding the host's lease would refuse anyway.
    answers = move.survey(ctx, room_id, configuration)
    for install_id, outcome in answers.items():
        answer = outcome.get("answer") or {}
        if answer.get("hosting"):
            raise SuccessionError("the group's host can be reached", reason="host_reachable",
                                  detail={"other": move.named(configuration, install_id)})
    willing = {install_id for install_id, outcome in answers.items()
               if outcome.get("answer") and _would_promise(outcome["answer"], epoch, me, seq)}
    if _would_promise(_own_answer(ctx, room_id), epoch, me, seq):
        willing.add(me)
    needed = succession.majority(len(voters))
    if me not in willing or len(willing & set(voters)) < needed:
        raise SuccessionError("a majority of the voters can't promise yet", reason="no_majority",
                              detail={"willing": len(willing & set(voters)), "needed": needed})
    record = _start(ctx, room_id, current, proof_kind="certified")
    move._save(ctx, room_id, "move", record)
    try:
        outcomes = move._ask_fences(ctx, room_id, configuration, epoch, current["watermark"], vote=True)
        move._refused_for_other(configuration, outcomes)
        promised = [voter for voter in voters
                    if voter in outcomes and not isinstance(outcomes[voter], move.RemoteRefusal)]
        if me not in promised or len(promised) < needed:
            raise SuccessionError("a majority of the voters did not promise", reason="no_majority",
                                  detail={"willing": len(promised), "needed": needed})
        record = move.record_fences(ctx, room_id, record, outcomes, epoch)
        record = move._adopt(ctx, room_id, record)
        certificate = succession.build_certificate(
            room_id=room_id, from_epoch=record["from_epoch"], to_epoch=epoch, successor=me,
            previous_authority=record["previous_host"], origin_install_id=record["origin_install_id"],
            configuration=configuration, receipts=record["receipts"])
        record = _promote(ctx, room_id, record, certificate, "certified", offline_since=offline_since,
                          at_risk=at_risk(ctx, room_id, current, record))
    except SuccessionError as exc:
        move._fail(ctx, room_id, record, exc)
        raise
    return move.finish(ctx, room_id, record)


def careful_move(ctx, room_id: str, *, silent_since: float, silent_for: float) -> dict[str, Any] | None:
    """The standby of a two-voter group takes over after the careful silence (``evidence``)."""
    from gateway import hosted_room_succession_move as move
    current = move.view(ctx, room_id)
    record = move.committed_transition(ctx, room_id, dict(current["move"] or {}))
    if record.get("state") == "moving" and record.get("transition_committed"):
        return move.finish(ctx, room_id, record)
    configuration, me = current["configuration"], succession.local_install_id()
    host = (succession.host_entry(configuration) or {}).get("install_id")
    if mode_of(configuration) != "careful" or succession.voters_of(configuration) != [host, me]:
        raise SuccessionError("this computer may not take this group over by itself", reason="target_not_ready")
    epoch = move.next_epoch(ctx, room_id, current, record)
    record = {**_start(ctx, room_id, current, proof_kind="evidence"),
              "silence": {"silent_since": float(silent_since), "silent_for_s": float(silent_for)}}
    move._save(ctx, room_id, "move", record)
    try:
        outcomes = move._ask_fences(ctx, room_id, configuration, epoch, current["watermark"])
        move._refused_for_other(configuration, outcomes)
        if me not in outcomes or isinstance(outcomes[me], move.RemoteRefusal):
            raise SuccessionError("this computer could not fence its own copy", reason="target_not_ready")
        record = move.record_fences(ctx, room_id, record, outcomes, epoch)
        record = move._adopt(ctx, room_id, record)
        with closing(rooms._read_connection(ctx.db_path)) as conn:
            statement = succession.evidence_statement(
                conn, room_id, from_epoch=record["from_epoch"], to_epoch=epoch, successor=me,
                silent_since=silent_since, silent_for_s=silent_for)
        proof = {"statement": statement, "signature": succession.sign(succession.EVIDENCE, statement)}
        record = _promote(ctx, room_id, {**record, "silence": dict(statement)}, proof, "evidence",
                          offline_since=silent_since,
                          at_risk=at_risk(ctx, room_id, current, record))
    except SuccessionError as exc:
        move._fail(ctx, room_id, record, exc)
        raise
    return move.finish(ctx, room_id, record)


def at_risk(ctx, room_id: str, current: Mapping[str, Any], record: Mapping[str, Any]) -> int:
    """Events the old host said it had, before or during the catch-up, that the copy this move adopted
    doesn't hold."""
    from gateway import hosted_room_succession_move as move
    heads = (current["head"], move.view(ctx, room_id)["head"])
    announced = max(int(head.get("announced_seq", head["latest_seq"])) for head in heads)
    return max(0, announced - int(record["adopted_watermark"]["seq"]))


def _promote(ctx, room_id: str, record: dict[str, Any], proof: Mapping[str, Any], kind: str, *,
             offline_since: float | None, at_risk: int) -> dict[str, Any]:
    """Verify the proof against this copy, then write the marked transition (``reason: automatic``)."""
    from gateway import hosted_room_replicas as replicas
    from gateway import hosted_room_succession_move as move
    configuration, me = move.view(ctx, room_id)["configuration"], succession.local_install_id()
    with closing(rooms._read_connection(ctx.db_path)) as conn:
        succession.verify_proof_locked(conn, room_id, proof_kind=kind, proof=dict(proof),
                                       from_epoch=record["from_epoch"], to_epoch=record["to_epoch"], successor=me,
                                       fork_seq=record["adopted_watermark"]["seq"])
    replicas.promote_replica(
        ctx.db_path, room_id=room_id, transition=succession.transition_for(proof, kind), to_epoch=record["to_epoch"],
        text=succession.transition_text(succession.label(configuration, me)),
        display={"from_name": succession.label(configuration, record["previous_host"]),
                 "to_name": succession.label(configuration, me), "offline_since": offline_since,
                 "reason": "automatic", "at_risk": int(at_risk)})
    record = {**record, "transition_committed": True, "proof_digest": succession.proof_digest(proof),
              "offline_since": offline_since}
    move._save(ctx, room_id, "move", record)
    return record


# --- continuing a paused host anyway -------------------------------------------------------------
def continue_anyway(ctx, room_id: str) -> dict[str, Any]:
    """The owner continues the paused host anyway: a fresh epoch at every computer it reaches. A host
    paused for a step promised to another computer that never took it continues past that step."""
    from gateway import hosted_room_succession_move as move
    from gateway.hosted_room_succession_status import status
    current = move.view(ctx, room_id)
    configuration, head, me = current["configuration"], current["head"], succession.local_install_id()
    if not head["authoritative"]:
        raise SuccessionError("this computer does not host the group", reason="room_not_found")
    if not move.is_owner(ctx, room_id):
        raise SuccessionError("only the group's owner can continue it here", reason="not_owner")
    automatic = instance_for(ctx.db_path)
    reason = host_paused_reason(ctx.db_path, room_id)
    stalled = move.stalled_promise(ctx.db_path, room_id)
    if reason not in PAUSED_REASONS and stalled is None:
        raise SuccessionError("this group is not paused to stay safe", reason="invalid_params")
    if reason == "no_lease_layer" and stalled is None:
        # It holds no lease, so no standby may move the group by itself beside it: continuing here turns
        # automatic moves off for the group (ask mode) until the owner turns them back on. Its voters
        # still hear it; it serves at its epoch.
        record = succession.load_record(ctx.db_path, room_id, "automatic") or {}
        succession.save_record(ctx.db_path, room_id, "automatic", {**record, "anyway_epoch": head["authority_epoch"],
                                                                   "paused": None})
        _serve_without_leases(ctx.db_path, room_id)
        _ask_first(ctx, room_id, configuration)
        move.append_state(ctx.db_path, room_id, "ok")
        return status(ctx, room_id)
    if getattr(ctx, "runs_store", None) is None:
        raise SuccessionError("this computer has no fence store to continue from", reason="target_not_ready")
    epoch = move.next_epoch(ctx, room_id, current, {"rival_epoch": stalled["epoch"]} if stalled else {})
    outcomes = move._ask_fences(ctx, room_id, configuration, epoch, current["watermark"])
    # A computer that already promised the next step to another one means that one is taking over.
    move._refused_for_other(configuration, {k: v for k, v in outcomes.items()
                                            if not (isinstance(v, move.RemoteRefusal) and v.code == "host_reachable")})
    if me not in outcomes or isinstance(outcomes[me], move.RemoteRefusal):
        raise SuccessionError("this computer could not fence its own epoch", reason="target_not_ready")
    fenced = {k: v for k, v in outcomes.items() if not isinstance(v, move.RemoteRefusal)}
    proof = succession.build_attestation(
        room_id=room_id, from_epoch=head["authority_epoch"], to_epoch=epoch, successor=me, previous_authority=me,
        origin_install_id=current["origin"], configuration_seq=int(configuration.get("configuration_seq") or 0),
        receipts=[fenced[k]["receipt"] for k in sorted(fenced)],
        unreachable=sorted(k for k, v in outcomes.items() if isinstance(v, move.RemoteRefusal)),
        confirmed_by=ctx.actor_subject or "operator", preview_id=None, statement=succession.ANYWAY_TEXT)
    move.advance_here(ctx, room_id, current, proof, epoch)
    record = succession.load_record(ctx.db_path, room_id, "automatic") or {}
    succession.save_record(ctx.db_path, room_id, "automatic", {**record, "anyway_epoch": epoch, "paused": None})
    if automatic is not None:
        with automatic._lock:
            automatic._paused.pop(room_id, None)
    if reason == "no_lease_layer":  # past a stalled step on a host without its lease layer: as above
        _serve_without_leases(ctx.db_path, room_id)
        _ask_first(ctx, room_id, configuration)
    move.serve_again(ctx, room_id, fenced)
    move.announce(ctx, room_id)
    return status(ctx, room_id)


# --- a planned restart ---------------------------------------------------------------------------
def extend_for_restart(ctx, room_ids, until: float) -> list[str]:
    """Before a planned restart in majority mode, ask the voters to keep the lease until ``until``.

    The next pushes carry the request; nothing waits for them, so the restart is never delayed. A
    voter that granted it promises no takeover before ``until`` (at most five minutes); one that
    didn't hear in time simply lets the lease run out, and the restart window announced beside it
    keeps the standbys waiting.
    """
    automatic = instance_for(ctx.db_path)
    if automatic is None:
        return []
    extended = []
    for room_id in room_ids:
        info = room_view(ctx, room_id)
        if info is not None and info["hosts"] and info["mode"] == "majority":
            automatic.lease.extend(room_id, until)
            extended.append(room_id)
    return extended


# --- the registry and the hooks ------------------------------------------------------------------
# One lease layer per hosted service of this installation (one per served profile), each for the groups
# in its own room store; #104601's hooks name only the room, so each call finds the store that holds it.
_instances: dict[str, list[Automatic]] = {}
_registry_lock = threading.Lock()


def _resolved(path) -> str:
    return str(Path(path).resolve())


def _all() -> list[Automatic]:
    try:
        return list(_instances.get(succession.local_install_id(), ()))
    except Exception:
        return []


def instance_for(db_path) -> Automatic | None:
    """The lease layer of the hosted service whose room store is ``db_path``, while it runs."""
    if db_path is None:
        return None
    wanted = _resolved(db_path)
    return next((automatic for automatic in _all() if automatic.db_path() == wanted), None)


def db_file(conn) -> str | None:
    """The file of a connection's main database."""
    row = next((row for row in conn.execute("PRAGMA database_list").fetchall() if row[1] == "main"), None)
    return row[2] if row is not None and row[2] else None


def host_paused_reason(db_path, room_id: str) -> str | None:
    """Why the group this store hosts may not run now, as far as automatic moves go: ``lost_majority`` or
    ``isolated`` from its lease layer, ``no_lease_layer`` when an automatic mode has none to keep it
    safe (it fails closed until its lease layer is up, or the owner continues it anyway); else None."""
    automatic = instance_for(db_path)
    if automatic is not None:
        return automatic.paused_reason(room_id)
    with closing(rooms._read_connection(Path(db_path))) as conn:
        room = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                            "AND disbanded_at IS NULL", (room_id,)).fetchone()
        if room is None or room[0] != succession.local_install_id():
            return None
        mode = mode_of(succession.configuration_locked(conn, room_id))
    if mode not in AUTOMATIC_MODES or anyway_epoch(db_path, room_id) == int(room[1]):
        return None
    return "no_lease_layer"


def paused_view(ctx, room_id: str) -> dict[str, Any] | None:
    """``status.paused`` for a room this computer hosts: why, since when, and whom it waits for."""
    automatic = instance_for(ctx.db_path)
    if automatic is not None:
        return automatic.paused_view(ctx, room_id)
    if host_paused_reason(ctx.db_path, room_id) == "no_lease_layer":
        return {"reason": "no_lease_layer", "since": None, "waiting_for": []}
    return None


def may_follow(db_path, room_id: str, to_epoch: int, successor: str) -> bool:
    """Whether a copy here may follow ``successor`` into ``to_epoch``. An epoch this computer fenced is
    over for it: it follows a host into it only on the lineage it committed to there, the computer it
    promised a later epoch to or the host it learned, never another host of that epoch."""
    from gateway import hosted_room_fence as fence
    automatic = instance_for(db_path)
    path = automatic.fence_path() if automatic is not None else None
    if path is None:
        return True
    try:
        state = fence.room_fence_state(path, room_id)
    except Exception:
        return True  # nothing recorded here to hold it to
    if int(to_epoch) > int(state["fenced_epoch"]):
        return True
    committed = {(state.get("promise") or {}).get("candidate_install_id"),
                 (state.get("authority") or {}).get("install_id")}
    return successor in committed


def fenced_here(db_path, room_id: str, epoch: int) -> bool:
    """Whether this computer's own fence store has fenced ``epoch`` (it promised a later one, or learned
    a later host): then it never serves that epoch again, whatever leases it gets back."""
    from gateway import hosted_room_fence as fence
    automatic = instance_for(db_path)
    path = automatic.fence_path() if automatic is not None else None
    if path is None:
        return False
    try:
        return int(fence.room_fence_state(path, room_id)["fenced_epoch"]) >= int(epoch)
    except Exception:
        return True  # an unreadable fence store can't vouch for this epoch


def install(automatic: Automatic) -> None:
    """Register a hosted service's lease layer with #104601's lease hooks."""
    with _registry_lock:
        listed = _instances.setdefault(succession.local_install_id(), [])
        if automatic not in listed:
            listed.append(automatic)
        _register_hooks()


def uninstall(automatic: Automatic) -> None:
    with _registry_lock:
        for key, listed in list(_instances.items()):
            if automatic in listed:
                listed.remove(automatic)
            if not listed:
                del _instances[key]
        if not _instances and not _without_leases:
            _custody().register_lease_hooks(
                lease_request_provider=None, lease_grant_hook=None, lease_ack_hook=None,
                lease_remaining_provider=None, serving_provider=None)


# Rooms whose owner continued them anyway on a host whose lease layer isn't running: room id -> store.
_without_leases: dict[str, str] = {}


def _serve_without_leases(db_path, room_id: str) -> None:
    """#104601 asks the lease layer whether this host may append; with none running here it would never
    say so. Until the group's configuration says "ask first", the owner's choice answers for it."""
    with _registry_lock:
        _without_leases[room_id] = _resolved(db_path)
        if not _instances:
            _register_hooks()


def _ask_first(ctx, room_id: str, configuration: Mapping[str, Any]) -> None:
    """Turn automatic moves off for the group and write the configuration that says so now (the copies
    carry it to every standby). With a change of voters still settling, the next configuration does."""
    from gateway import hosted_room_identity as identity
    custody = _custody()
    custody.set_automatic(ctx.db_path, room_id=room_id, enabled=False)
    host = succession.host_entry(configuration) or {}
    try:
        custody.maintain_configuration(
            ctx.db_path, room_id=room_id, local_gateway_id=succession.local_install_id(),
            public_key=host.get("public_key") or identity.local_public_key(), endpoint=host.get("endpoint"),
            name=host.get("name"), owner_name=configuration.get("owner_name"), always_on=host.get("always_on"))
    except Exception:
        logger.warning("could not write the configuration that turns automatic moves off", exc_info=True)
    service = getattr(ctx, "service", None)
    if service is not None and getattr(service, "replication", None) is not None:
        service.replication.wakeup()


def _register_hooks() -> None:
    _custody().register_lease_hooks(
        lease_request_provider=_request, lease_grant_hook=_grant, lease_ack_hook=_ack,
        lease_remaining_provider=_remaining, serving_provider=_serving)


def _hosting(room_id) -> Automatic | None:
    return next((automatic for automatic in _all() if automatic.hosts(room_id)), None)


def _request(room_id):
    automatic = _hosting(room_id)
    return automatic.request(room_id) if automatic is not None else None


def _grant(room_id, epoch, authority_install_id, request):
    automatic = next((automatic for automatic in _all() if automatic.keeps_copy(room_id)), None)
    return automatic.grant(room_id, epoch, authority_install_id, request) if automatic is not None else None


def _ack(room_id, voter_install_id, lease_grant, sent_at):
    automatic = _hosting(room_id)
    if automatic is not None:
        automatic.acknowledged(room_id, voter_install_id, lease_grant, sent_at)


def _remaining(room_id):
    automatic = _hosting(room_id)
    return automatic.remaining(room_id) if automatic is not None else None


def _serving(room_id):
    """``serving_provider``: False while the store that hosts the room may not append for it."""
    for automatic in _all():
        if automatic.hosts(room_id):
            return automatic.serving(room_id)
    store = _without_leases.get(room_id)
    if store is not None:
        # Continued anyway without a lease layer: it serves at the epoch the owner continued it at.
        try:
            with closing(rooms._read_connection(Path(store))) as conn:
                room = conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=? "
                                    "AND disbanded_at IS NULL", (room_id,)).fetchone()
                asks_first = mode_of(succession.configuration_locked(conn, room_id)) == "ask"
        except Exception:
            return None
        if asks_first or room is None:
            with _registry_lock:
                _without_leases.pop(room_id, None)  # "ask first" needs no lease: #104601 lets it serve
            return None
        if room[0] == succession.local_install_id() and anyway_epoch(store, room_id) == int(room[1]):
            return True
    return None


def void_lease(db_path, room_id: str, before: float) -> None:
    """The host signed its group over at ``before``: the grants it holds from requests sent earlier no
    longer count, so if it resumes it serves only on grants that statement can't release."""
    automatic = instance_for(db_path)
    if automatic is not None:
        automatic.lease.void(room_id, before)


def heard_from_host(db_path, room_id: str) -> None:
    """This copy's host answered a fence request, or asked for one: that is contact too."""
    automatic = instance_for(db_path)
    if automatic is not None:
        automatic.heard_from_host(room_id)


def started_following(db_path, room_id: str) -> None:
    """This computer just stepped down to follow another host: it heard of that host now."""
    automatic = instance_for(db_path)
    if automatic is not None:
        with automatic._lock:
            automatic._following.pop(room_id, None)
        automatic.heard_from_host(room_id)


def contact(room_id: str, install_id: str) -> None:
    """A voter's signed request reached this host: in careful mode that counts as contact."""
    for automatic in _all():
        automatic.heard_from_voter(room_id, install_id)


def _last_report(db_path, room_id: str) -> float | None:
    """When this copy last stored a verified push from the host it follows (#104601's custody report)."""
    custody = _custody()
    with closing(rooms._read_connection(Path(db_path))) as conn:
        if not succession.table_exists(conn, custody.REPORTS_TABLE):
            return None
        row = conn.execute(f"SELECT reported_at FROM {custody.REPORTS_TABLE} WHERE room_id=?", (room_id,)).fetchone()
    return float(row[0]) if row is not None and row[0] is not None else None


def _last_ack(db_path, room_id: str, install_id: str) -> float | None:
    """When this host last recorded ``install_id``'s acknowledgment of a push."""
    custody = _custody()
    with closing(rooms._read_connection(Path(db_path))) as conn:
        if not succession.table_exists(conn, custody.WATERMARKS_TABLE):
            return None
        row = conn.execute(f"SELECT acknowledged_at FROM {custody.WATERMARKS_TABLE} WHERE room_id=? AND install_id=?",
                           (room_id, install_id)).fetchone()
    return float(row[0]) if row is not None and row[0] is not None else None


# --- the online check ----------------------------------------------------------------------------
_endpoints: tuple[float, list[tuple[str, int]]] | None = None


def _known_endpoints() -> list[tuple[str, int]]:
    """Hosts this computer already uses: its model provider, and its messaging platforms."""
    global _endpoints
    if _endpoints is not None and time.monotonic() - _endpoints[0] < 600:
        return _endpoints[1]
    found: list[tuple[str, int]] = []
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        base = resolve_runtime_provider().get("base_url")
        parts = urlsplit(str(base)) if base else None
        if parts is not None and parts.hostname:
            found.append((parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)))
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config
        platforms = (load_config() or {}).get("platforms") or {}
        for name, host in _PLATFORM_HOSTS.items():
            if isinstance(platforms.get(name), Mapping) and platforms[name].get("enabled", True):
                found.append((host, 443))
    except Exception:
        pass
    _endpoints = (time.monotonic(), found)
    return found


_SHARED_SPACE = ipaddress.ip_network("100.64.0.0/10")


def public_address(value: str) -> bool:
    """A globally reachable address: not loopback, link-local, private (RFC 1918, ULA), shared
    (100.64.0.0/10, as overlay networks use) or otherwise reserved."""
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_loopback or address.is_link_local or address.is_private or address.is_multicast:
        return False
    if address.version == 4 and address in _SHARED_SPACE:
        return False
    return address.is_global


def online_check_default(timeout: float = 3.0) -> bool:
    """Whether this computer reaches a public endpoint it already uses; a TCP connection, nothing sent.

    Each endpoint is resolved first, and only public addresses count: a model on this computer or on
    the local network (Ollama on localhost, say) is not the internet, so with no public endpoint left
    the check fails and the standby never moves carefully.
    """
    for host, port in _known_endpoints():
        try:
            resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError:
            continue
        for _family, _type, _proto, _name, sockaddr in resolved:
            if not public_address(str(sockaddr[0])):
                continue
            try:
                with socket.create_connection((sockaddr[0], port), timeout=timeout):
                    return True
            except OSError:
                continue
    return False
