"""Several gateway identities in one process, each with its own root, room store and Runs store.

``Gateway.acting()`` switches the process to that gateway's HERMES_HOME, so its RoomLink secret,
installation id and room identity key are the real ones derived from that root. Rooms, copies,
custody and fences use the production stores; the network between gateways is a direct call to
the real handler, made as the gateway that answers.
"""

from __future__ import annotations

import os
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from gateway import hosted_room_custody as custody
from gateway import hosted_room_identity as identity
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_backup as backup
from gateway import hosted_room_succession_move as move
from gateway import hosted_rooms as rooms
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

ROOM = "room"
OWNER = "uid:501"
MEMBERS = [{"member_id": "writer", "handle": "writer", "profile": "default"},
           {"member_id": "reviewer", "handle": "reviewer", "profile": "reviewer"}]


@dataclass
class Gateway:
    name: str
    home: Path
    install_id: str = ""
    public_key: str = ""
    endpoint: str = ""
    stores: dict = field(default_factory=dict)

    @property
    def db(self) -> Path:
        return self.home / "state.db"

    @property
    def runs(self) -> RunIdempotencyStore:
        if "runs" not in self.stores:
            self.stores["runs"] = RunIdempotencyStore(str(self.home / "runs_idempotency.db"))
        return self.stores["runs"]

    @contextmanager
    def acting(self):
        previous = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.home)
        try:
            yield self
        finally:
            if previous is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = previous

    def backup_context(self, gateways=None, *, down=()) -> backup.BackupContext:
        # One request at a time: these gateways share one process, and ``acting`` switches its home.
        return backup.BackupContext(custody_db=self.db, runs_store=self.runs,
                                    post=router(gateways, down=down) if gateways else None,
                                    fetch_pages=fetcher(gateways, down=down) if gateways else None, workers=1)

    def close(self):
        for store in self.stores.values():
            store.close()


def make_gateways(tmp_path: Path, *names: str) -> dict[str, Gateway]:
    gateways = {}
    for name in names:
        home = tmp_path / name
        home.mkdir()
        gateway = Gateway(name, home, endpoint=f"https://{name}.example.test")
        with gateway.acting():
            gateway.install_id = rooms.local_authority_gateway_id()
            gateway.public_key = identity.local_public_key()
        gateways[name] = gateway
    return gateways


def message(db, event_id, *, gateway, epoch=1, text=None):
    return rooms.append_event(db, room_id=ROOM, event_id=event_id, kind="message.user",
                              actor={"kind": "user", "id": "owner"},
                              payload={"text": text or event_id, "thread_id": "main"},
                              authority_gateway_id=gateway, authority_epoch=epoch)


def configure(gateways, host: str, *, successors=(), custodians=None, owner_name="Dana") -> dict:
    """The host enrolls the other gateways as custodians (``successors`` may continue) and records them."""
    home = gateways[host]
    names = custodians if custodians is not None else [name for name in gateways if name != host]
    with home.acting():
        for name in names:
            other = gateways[name]
            custody.enroll_custodian(home.db, room_id=ROOM, install_id=other.install_id,
                                     public_key=other.public_key, endpoint=other.endpoint,
                                     name=name.title(), operator_name=f"{name.title()} Operator",
                                     role="custodian", active=True, allowed=name in successors,
                                     designated=name in successors)
        return custody.maintain_configuration(home.db, room_id=ROOM, local_gateway_id=home.install_id,
                                              public_key=home.public_key, endpoint=home.endpoint,
                                              name=host.title(), owner_name=owner_name)


def home_room(gateways, host: str, *, messages=3, successors=(), owner=OWNER) -> Gateway:
    """The host creates the room, records its owner and custodians, and writes a few messages."""
    gateway = gateways[host]
    with gateway.acting():
        rooms.create_room(gateway.db, room_id=ROOM, name="Room", members=MEMBERS,
                          authority_gateway_id=gateway.install_id)
        with rooms._transaction(gateway.db, immediate=True) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO state_meta(key, value) VALUES (?, ?)", ("gateway.hosted.owner.v1:" + ROOM, owner))
    configure(gateways, host, successors=successors)
    with gateway.acting():
        for number in range(messages):
            message(gateway.db, f"user:{number}", gateway=gateway.install_id)
    for name in successors:
        with gateways[name].acting():
            custody.set_local_consent(gateways[name].db, room_id=ROOM, allowed=True)
            succession.record_owner_subject(gateways[name].db, ROOM, owner)
    return gateway


def copy_to(source: Gateway, target: Gateway, *, through: int | None = None, report: dict | None = None):
    """Replicate the source's history into the target's copy, verifying transitions on the way; ``report``
    is the host's custody report a push carries (#104601), which the copy keeps."""
    with source.acting(), closing(rooms._read_connection(source.db)) as conn:
        page = custody.read_copy_page(conn, ROOM, after_seq=0, limit=rooms.MAX_LOG_LIMIT)
    with target.acting(), closing(rooms._read_connection(target.db)) as conn:
        mark = succession.watermark_locked(conn, ROOM)
    held = mark["seq"] if mark else 0
    events = [event for event in page["page"]["events"] if event["seq"] > held and (
        through is None or event["seq"] <= through)]
    if not events and report is None:
        return None
    page["page"].update(events=events, cursor=events[-1]["seq"] if events else held)
    page["page"]["has_more"] = page["page"]["cursor"] < page["page"]["latest_seq"]
    with source.acting(), closing(rooms._read_connection(source.db)) as conn:
        # The head the source vouches with for the page's end (#104601): the host signs one, a copy relays
        # the one it keeps.
        source_head = head(source)
        signed = (custody.sign_head_locked(conn, ROOM, seq=min(page["page"]["cursor"], source_head["latest_seq"]))
                  if source_head["authoritative"] else custody.vouched_head_locked(conn, ROOM))
    with target.acting():
        result = replicas.ingest_page(target.db, room_id=ROOM, room_name=page["room_name"], members=page["members"],
                                      page=page["page"], _verify_transition=succession.verify_transition_locked,
                                      _from_custodian=not head(source)["authoritative"], custody_report=report)
        if signed is not None:
            custody.record_head(target.db, ROOM, signed)
        return result


def router(gateways, *, down=(), before=None):
    """Deliver a request to another gateway by calling its real handler as that gateway."""
    by_endpoint = {gateway.endpoint: gateway for gateway in gateways.values()}

    def post(endpoint, path, body, timeout):
        gateway = by_endpoint[endpoint]
        if gateway.name in down:
            raise OSError("unreachable")
        name = path.rsplit("/", 1)[-1]
        if before is not None:
            before(gateway, name, body)
        with gateway.acting():
            context = gateway.backup_context(gateways, down=down)
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
    return post


def fetcher(gateways, *, down=()):
    """Custodian catch-up straight from the source gateway's own history."""
    by_id = {gateway.install_id: gateway for gateway in gateways.values()}

    def fetch(db_path, *, room_id, source_install_id, after_seq, limit):
        source = by_id[source_install_id]
        if source.name in down:
            raise OSError("unreachable")
        with source.acting(), closing(rooms._read_connection(source.db)) as conn:
            page = custody.read_copy_page(conn, room_id, after_seq=after_seq, limit=limit)
            vouched = custody.vouched_head_locked(conn, room_id)
        return {"room_id": room_id, **page, "head": vouched, "source_install_id": source_install_id}
    return fetch


def minter(gateway):
    """``mint_grants`` without an API server: one stand-in grant per consent this gateway recorded."""
    def mint(room_id, successor, epoch):
        with gateway.acting(), closing(rooms._read_connection(gateway.db)) as conn:
            consents = succession.consents_locked(conn, room_id)
        return [{"member_id": consent["member_id"], "target_profile": consent["target_profile"],
                 "grant": f"grant:{gateway.name}:{consent['member_id']}:{successor}:{epoch}",
                 "catalog": {"installation_id": gateway.install_id}} for consent in consents]
    return mint


def context(gateway, gateways, *, down=(), before=None, subject=OWNER, operator=False, service=None, grants=False):
    return move.MoveContext(
        db_path=gateway.db, runs_store=gateway.runs, service=service, actor_subject=subject, operator=operator,
        mint_grants=minter(gateway) if grants else None,
        post=router(gateways, down=down, before=before), fetch_pages=fetcher(gateways, down=down), workers=1)


def head(gateway):
    with gateway.acting(), closing(rooms._read_connection(gateway.db)) as conn:
        return backup.copy_head_locked(conn, ROOM)


def events(gateway, table="hosted_room_events"):
    with gateway.acting():
        return move._room_events(gateway.db, ROOM, table)
