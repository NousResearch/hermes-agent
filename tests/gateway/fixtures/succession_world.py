"""A small world of simulated gateways on a virtual clock, for automatic Group Chat moves.

They all run in this one process, and a request calls the receiving gateway's real handler directly (no
HTTP). Each gateway has its own root, room store, Runs store, room identity key and ``Automatic``
instance, registered with #104601's real lease hooks under its own installation id. Time is
virtual: ``hosted_room_clock`` and the wall clock advance only with ``World.advance``. Every
second each running gateway runs its automatic upkeep; every five seconds each host pushes its log
to its voters through the real custody calls the publisher makes (the lease request, the voter's
ingest and grant, the acknowledgment), and every minute to the other custodians. ``Network`` cuts
links, stops gateways and takes a computer offline (its online check fails). An observer checks
after every step that at most one epoch admits work.
"""

from __future__ import annotations

import time
from contextlib import closing

from gateway import hosted_room_clock as clock
from gateway import hosted_room_custody as custody
from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_automatic as automatic
from gateway import hosted_room_succession_backup as backup
from gateway import hosted_room_succession_move as move
from gateway import hosted_rooms as rooms
from gateway.hosted_room_succession_status import SuccessionUpkeep
from tests.gateway.fixtures.succession import (
    MEMBERS, OWNER, ROOM, copy_to, fetcher, head, make_gateways, message)

PUSH_SECONDS = 5.0
KEEPALIVE_SECONDS = 60.0
UPKEEP_SECONDS = 15.0


class Network:
    """Which gateways run, which links work (both ways), and which computers are online."""

    def __init__(self, names):
        self.stopped: set[str] = set()
        self.cut: set[frozenset[str]] = set()
        self.offline: set[str] = set()
        self.names = set(names)

    def reachable(self, a: str, b: str) -> bool:
        return a == b or (a not in self.stopped and b not in self.stopped and frozenset((a, b)) not in self.cut)

    def isolate(self, name: str) -> None:
        self.cut |= {frozenset((name, other)) for other in self.names - {name}}

    def split(self, *sides) -> None:
        """Cut every link between the given groups of names."""
        for index, side in enumerate(sides):
            for other in sides[index + 1:]:
                self.cut |= {frozenset((a, b)) for a in side for b in other}

    def heal(self) -> None:
        self.cut.clear()


class World:
    def __init__(self, tmp_path, monkeypatch, names, *, voters, others=(), automatic_on=True, peers=()):
        self.t = 1000.0
        self.wall0 = 1_790_000_000.0
        self.slept: dict[str, float] = {}
        monkeypatch.setattr(clock, "_NOW", lambda: self.t)
        monkeypatch.setattr(clock, "_AWAKE", lambda: self.t - self.slept.get(self.acting_name(), 0.0))
        monkeypatch.setattr(clock, "EXACT", True)
        monkeypatch.setattr(clock, "boot_id", lambda: "boot-7")
        monkeypatch.setattr(time, "time", lambda: self.wall0 + self.t)
        monkeypatch.setattr(automatic, "JITTER_SECONDS", 0.0)
        monkeypatch.setattr(custody, "local_always_on", lambda refresh=False: True)
        self.gateways = make_gateways(tmp_path, *names)
        self.by_id = {gateway.install_id: name for name, gateway in self.gateways.items()}
        self.network = Network(names)
        self.automatics: dict[str, automatic.Automatic] = {}
        self.upkeeps: dict[str, SuccessionUpkeep] = {}
        self._last_full: dict[str, float] = {}
        self.log: list[tuple[float, str, int]] = []
        self.admitted: list[tuple[float, set[tuple[str, int]]]] = []
        self.voters, self.others, self.peers = list(voters), list(others), list(peers)
        self._last_keepalive = -1e9
        self._last_push = -1e9
        for name, gateway in self.gateways.items():
            with gateway.acting():
                upkeep = SuccessionUpkeep(lambda gateway=gateway: self.context(gateway))
                upkeep.automatic._online_check = lambda name=name: name not in self.network.offline
                automatic.install(upkeep.automatic)
            self.upkeeps[name], self.automatics[name] = upkeep, upkeep.automatic
        self._home(voters[0], automatic_on=automatic_on)

    # --- plumbing ------------------------------------------------------------------------------------
    def acting_name(self) -> str | None:
        try:
            return self.by_id.get(succession.local_install_id())
        except Exception:
            return None

    def router(self):
        """``post``: the caller is whichever gateway is acting; the network decides if it gets through."""
        inner = {}

        def post(endpoint, path, body, timeout):
            caller = self.acting_name()
            target = next(name for name, gateway in self.gateways.items() if gateway.endpoint == endpoint)
            if not self.network.reachable(caller, target):
                raise OSError("unreachable")
            gateway = self.gateways[target]
            name = path.rsplit("/", 1)[-1]
            with gateway.acting():
                context = backup.BackupContext(custody_db=gateway.db, runs_store=gateway.runs, post=post,
                                               fetch_pages=self.fetcher(), workers=1)
                try:
                    if name == "fence":
                        return backup.answer_fence(context, body)
                    if name == "learn":
                        return backup.answer_learn(context, body)
                    if name == "query":
                        return backup.answer_query(context, body)
                    if name == "decision":
                        return backup.answer_decision(context, body)
                    if name == "handover":
                        from gateway.hosted_room_succession_handover import answer_handover
                        return answer_handover(context, body)
                    return backup.answer_report(gateway.db, body)
                except succession.SuccessionError as exc:
                    raise move.RemoteRefusal(exc.reason, exc.detail) from exc
        inner["post"] = post
        return post

    def fetcher(self):
        base = fetcher(self.gateways)

        def fetch(db_path, *, room_id, source_install_id, after_seq, limit):
            caller, source = self.acting_name(), self.by_id[source_install_id]
            if not self.network.reachable(caller, source):
                raise OSError("unreachable")
            return base(db_path, room_id=room_id, source_install_id=source_install_id, after_seq=after_seq,
                        limit=limit)
        return fetch

    def context(self, gateway, *, subject=OWNER, operator=False):
        return move.MoveContext(db_path=gateway.db, runs_store=gateway.runs, actor_subject=subject, operator=operator,
                                post=self.router(), fetch_pages=self.fetcher(), workers=1)

    # --- the group -----------------------------------------------------------------------------------
    def _home(self, host_name: str, *, automatic_on: bool) -> None:
        host = self.gateways[host_name]
        from tests.gateway.fixtures.passive_copy import member
        # ``peers`` each run a Bot of the group on their own computer (a participant).
        members = [*MEMBERS, *(member(f"bot-{name}", target=self.gateways[name].install_id) for name in self.peers)]
        with host.acting():
            rooms.create_room(host.db, room_id=ROOM, name="Room", members=members, authority_gateway_id=host.install_id)
            with rooms._transaction(host.db, immediate=True) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO state_meta(key, value) VALUES (?, ?)",
                             ("gateway.hosted.owner.v1:" + ROOM, OWNER))
            for order, name in enumerate([*self.voters[1:], *self.others]):
                other = self.gateways[name]
                successor = name in self.voters
                # Designated one after the other: that is the owner's order of standbys.
                custody.enroll_custodian(host.db, room_id=ROOM, install_id=other.install_id,
                                         public_key=other.public_key, endpoint=other.endpoint, name=name.title(),
                                         operator_name="Dana", role="custodian", active=True, allowed=successor,
                                         designated=successor, always_on=successor, now=time.time() + order)
            if not automatic_on:
                custody.set_automatic(host.db, room_id=ROOM, enabled=False)
        for name in self.voters[1:]:
            with self.gateways[name].acting():
                custody.set_local_consent(self.gateways[name].db, room_id=ROOM, allowed=True)
                succession.record_owner_subject(self.gateways[name].db, ROOM, OWNER)
        self.settle(host_name)
        with host.acting():
            for number in range(3):
                message(host.db, f"user:{number}", gateway=host.install_id)
        self.push_all(host_name, everyone=True)

    def configure(self, host_name: str):
        host = self.gateways[host_name]
        with host.acting():
            return custody.maintain_configuration(host.db, room_id=ROOM, local_gateway_id=host.install_id,
                                                  public_key=host.public_key, endpoint=host.endpoint,
                                                  name=host_name.title(), owner_name="Dana", always_on=True)

    def settle(self, host_name: str) -> None:
        """Configure step by step, each custodian storing every step, until the voters are settled."""
        for _ in range(16):
            if self.configure(host_name) is None:
                return
            self.push_all(host_name, everyone=True)
        raise AssertionError("the voters never settled")

    def configuration(self, name: str) -> dict:
        gateway = self.gateways[name]
        with gateway.acting(), closing(rooms._read_connection(gateway.db)) as conn:
            return succession.configuration_locked(conn, ROOM)

    # --- the publisher, as #104601 runs it -----------------------------------------------------------
    def push(self, host_name: str, name: str) -> bool:
        """One push from a host to one custodian: the page, the lease request, the grant and the ack."""
        host, target = self.gateways[host_name], self.gateways[name]
        if not self.network.reachable(host_name, name):
            return False
        try:
            with host.acting():
                current = head(host)
                configuration = self.configuration(host_name)
                voter = target.install_id in succession.voters_of(configuration)
                with closing(rooms._read_connection(host.db)) as conn:
                    report = custody.report_locked(conn, ROOM, target.install_id, head_seq=current["latest_seq"])
                request = custody.lease_request(ROOM) if voter else None
                if request is not None:
                    report["lease_request"] = request
            copy_to(host, target, report=report)
            with target.acting():
                follows = head(target)
                grant = None
                # Every push stored from the host it follows reaches the lease layer, lease request or not.
                if (follows["authority_gateway_id"], follows["authority_epoch"]) == (
                        current["authority_gateway_id"], current["authority_epoch"]):
                    grant = custody.lease_grant(ROOM, current["authority_epoch"], current["authority_gateway_id"],
                                                request)
                with closing(rooms._read_connection(target.db)) as conn:
                    mark = succession.watermark_locked(conn, ROOM)
        except Exception:
            return False  # the copy refused the page: no acknowledgment
        with host.acting():
            try:
                custody.record_acknowledgment(host.db, room_id=ROOM, install_id=target.install_id, watermark=mark)
            except custody.CustodyError:
                pass
            if request is not None:
                custody.lease_acknowledged(ROOM, target.install_id, grant, request)
        return True

    def hosts(self) -> list[str]:
        found = []
        for name, gateway in self.gateways.items():
            if name in self.network.stopped:
                continue
            with gateway.acting():
                try:
                    if head(gateway)["authoritative"]:
                        found.append(name)
                except Exception:
                    continue
        return found

    def push_all(self, host_name: str, *, everyone: bool) -> None:
        configuration = self.configuration(host_name)
        voters = set(succession.voters_of(configuration))
        for name, gateway in self.gateways.items():
            if name == host_name or gateway.install_id not in succession.custodians_by_id(configuration):
                continue
            if everyone or gateway.install_id in voters:
                self.push(host_name, name)

    # --- time ----------------------------------------------------------------------------------------
    def tick(self, name: str) -> list:
        """One second of this gateway's upkeep: the automatic checks, and the full upkeep every 15 s."""
        gateway, upkeep = self.gateways[name], self.upkeeps[name]
        with gateway.acting():
            if self.t - self._last_full.get(name, -1e9) >= UPKEEP_SECONDS:
                self._last_full[name] = self.t
                upkeep.run_once()
            return upkeep.run_automatic()

    def admitting(self) -> set[tuple[str, int]]:
        """``(name, epoch)`` of every running gateway that would admit work for the room now."""
        found = set()
        for name, gateway in self.gateways.items():
            if name in self.network.stopped:
                continue
            with gateway.acting(), closing(rooms._read_connection(gateway.db)) as conn:
                current = backup.copy_head_locked(conn, ROOM) if conn.execute(
                    "SELECT 1 FROM hosted_rooms WHERE room_id=?", (ROOM,)).fetchone() else None
                if current and current["serving"]:
                    found.add((name, current["authority_epoch"]))
        return found

    def observe(self) -> None:
        admitting = self.admitting()
        self.admitted.append((self.t, admitting))
        assert len({epoch for _, epoch in admitting}) <= 1 and len(admitting) <= 1, (self.t, admitting)

    def advance(self, seconds: float, *, observe: bool = True) -> None:
        end = self.t + seconds
        while self.t < end:
            self.t += 1.0
            if self.t - self._last_push >= PUSH_SECONDS:
                self._last_push = self.t
                everyone = self.t - self._last_keepalive >= KEEPALIVE_SECONDS
                if everyone:
                    self._last_keepalive = self.t
                for host_name in self.hosts():
                    self.push_all(host_name, everyone=everyone)
            for name in sorted(self.gateways):
                if name not in self.network.stopped:
                    self.tick(name)
            if observe:
                self.observe()

    def until(self, predicate, *, limit: float, observe: bool = True) -> float:
        """Advance second by second until ``predicate()``; returns the virtual seconds it took."""
        start = self.t
        while not predicate():
            if self.t - start >= limit:
                raise AssertionError(f"not within {limit} s")
            self.advance(1.0, observe=observe)
        return self.t - start

    def send(self, name: str, text: str) -> bool:
        """A user message on ``name`` if it admits work now (the hosted service's send gate)."""
        gateway = self.gateways[name]
        with gateway.acting():
            if succession.paused_reason(gateway.db, ROOM) is not None or not head(gateway)["serving"]:
                return False
            message(gateway.db, text, gateway=gateway.install_id, epoch=head(gateway)["authority_epoch"])
        return True

    def status(self, name: str) -> dict:
        from gateway.hosted_room_succession_status import status
        gateway = self.gateways[name]
        with gateway.acting():
            return status(self.context(gateway), ROOM)

    def head(self, name: str) -> dict:
        with self.gateways[name].acting():
            return head(self.gateways[name])

    def texts(self, name: str) -> list[str]:
        from tests.gateway.fixtures.succession import events
        table = "hosted_room_events" if self.head(name)["authoritative"] else "hosted_room_replica_events"
        return [event["payload"].get("text") for event in events(self.gateways[name], table)
                if event["kind"] == "message.user"]

    def close(self) -> None:
        for name, gateway in self.gateways.items():
            with gateway.acting():
                automatic.uninstall(self.automatics[name])
            gateway.close()
