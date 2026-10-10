"""``hermes group`` — relay user messages into Group Chats from any session.

Acting-as-user: the caller (typically an agent in a Discord/CLI/Desktop
session) writes into the room ON THE USER'S BEHALF. It is not a room member
and needs no membership — the actor is a ``user`` whose ``profile`` and
``display_name`` record who relayed. Turns run wherever the room's
coordinator lives; ``send --wait`` follows the room until the discussion
settles and streams member replies, so a background ``terminal`` call brings
the deliberation back to the originating session as a completion
notification (the same shape as ``hermes -p <agent> chat -q`` handoffs).

Transports are pluggable via :func:`register_transport`. This module ships the
gateway-hosted room transport (:mod:`gateway.hosted_rooms`), whose rooms are
driven headlessly by the gateway's hosted-room worker.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

from gateway import hosted_rooms

# ``thread_id`` grammar in gateway.hosted_room_discussion: [A-Za-z0-9][A-Za-z0-9._:-]*
_THREAD_SAFE_RE = re.compile(r"[^A-Za-z0-9._:-]+")
DEFAULT_THREAD = "cli"
DEFAULT_WAIT_TIMEOUT_SECONDS = 1800.0
_WORKER_WARN_AFTER_SECONDS = 30.0

# Exit codes (documented in --help; the relaying agent branches on them).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_TIMEOUT = 3
EXIT_SUPERSEDED = 4


class GroupCLIError(Exception):
    """User-facing failure; ``cmd_group`` prints it and exits 1."""


@dataclass(frozen=True)
class RoomRef:
    kind: str
    room_id: str
    name: str
    members: tuple[str, ...]
    managed_here: bool
    raw: dict = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class SentMessage:
    ref: RoomRef
    message_id: str
    seq: int | None
    thread: str
    raw: dict = field(default_factory=dict, compare=False, repr=False)


OnReply = Callable[[str, str], None]


class Transport(Protocol):
    kind: str

    def list_rooms(self) -> list[RoomRef]: ...

    def send(
        self,
        ref: RoomRef,
        *,
        text: str,
        thread: str,
        label: str,
        event_key: str | None,
    ) -> SentMessage: ...

    def wait(
        self,
        sent: SentMessage,
        *,
        timeout: float,
        poll_seconds: float,
        on_reply: OnReply,
    ) -> tuple[int, dict[str, Any]]: ...

    def log(self, ref: RoomRef, *, since: int) -> Iterator[dict[str, Any]]: ...


_TRANSPORTS: list[Transport] = []

# ── session → thread continuity ──────────────────────────────────────────────
#
# A relaying session (Discord thread, CLI chat) that sends twice into the same
# room should land in the SAME room thread, while a different session starts a
# new one. The agent cannot be trusted to remember a minted thread id across
# turns, so the CLI binds (room, session) → thread on the first send and reuses
# it. The session id comes from HERMES_SESSION_ID (set per turn by the agent
# runtime for tools it spawns) or --session; --new-thread breaks the binding.

_THREAD_BINDINGS_FILE = "thread-bindings.json"
_THREAD_BINDINGS_MAX = 500


def _bindings_path() -> Path:
    # Machine root (profile-independent), beside state.db.
    return hosted_rooms.default_db_path().parent / "group_relay" / _THREAD_BINDINGS_FILE


def _read_bindings(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@contextlib.contextmanager
def _bindings_lock(path: Path) -> Iterator[None]:
    """Serialize read-modify-write across relays. flock where available;
    a no-op elsewhere (the write below is still atomic per file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    try:
        import fcntl
    except ImportError:  # pragma: no cover — Windows
        yield
        return
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _mutate_bindings(mutate: Callable[[dict[str, dict[str, Any]]], None]) -> None:
    """Locked, atomic read-modify-write of the bindings file. Never raises."""
    try:
        path = _bindings_path()
        with _bindings_lock(path):
            data = _read_bindings(path)
            mutate(data)
            # Unique temp per writer: a shared name let two writers clobber
            # each other's temp before os.replace.
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".bindings-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.replace(tmp, path)
    except Exception:
        pass


def _binding_key(ref: RoomRef, session_id: str) -> str:
    return f"{ref.kind}:{ref.room_id}:{session_id}"


def bound_thread(ref: RoomRef, session_id: str) -> str | None:
    """The room thread this session already writes to, if any."""
    entry = _read_bindings(_bindings_path()).get(_binding_key(ref, session_id))
    thread = entry.get("thread") if isinstance(entry, dict) else None
    return str(thread) if thread else None


def bind_thread(ref: RoomRef, session_id: str, thread: str) -> None:
    """Remember that ``session_id`` continues ``thread`` in ``ref``. Never raises."""
    if not session_id or not thread:
        return

    def mutate(data: dict[str, dict[str, Any]]) -> None:
        data[_binding_key(ref, session_id)] = {"thread": thread, "at": int(time.time())}
        if len(data) > _THREAD_BINDINGS_MAX:
            oldest = sorted(
                data.items(),
                key=lambda kv: kv[1].get("at", 0) if isinstance(kv[1], dict) else 0,
            )
            for key, _ in oldest[: len(data) - _THREAD_BINDINGS_MAX]:
                data.pop(key, None)

    _mutate_bindings(mutate)


def forget_thread(ref: RoomRef, session_id: str) -> None:
    _mutate_bindings(lambda data: data.pop(_binding_key(ref, session_id), None))


def _session_id(args) -> str:
    explicit = str(getattr(args, "session", None) or "").strip()
    return explicit or (os.getenv("HERMES_SESSION_ID") or "").strip()


def register_transport(transport: Transport) -> None:
    if all(existing.kind != transport.kind for existing in _TRANSPORTS):
        _TRANSPORTS.append(transport)


def transports() -> list[Transport]:
    return list(_TRANSPORTS)


def _transport_for(kind: str) -> Transport:
    for transport in _TRANSPORTS:
        if transport.kind == kind:
            return transport
    raise GroupCLIError(f"no transport registered for group kind {kind!r}")


# ── helpers ──────────────────────────────────────────────────────────────────


def _profile() -> str:
    return (os.getenv("HERMES_PROFILE") or "default").strip() or "default"


def thread_id(raw: str | None) -> str:
    """Coerce a caller-supplied thread label into the room-log identifier grammar."""
    text = _THREAD_SAFE_RE.sub("-", (raw or DEFAULT_THREAD).strip()).strip("-")
    if not text:
        return DEFAULT_THREAD
    return text if text[0].isalnum() else f"t-{text}"


def resolve_room(ref: str, *, kind: str | None = None) -> RoomRef:
    """Exact name → case-insensitive unique name → room_id, across transports."""
    want = (ref or "").strip()
    if not want:
        raise GroupCLIError("group name or room_id is required")
    rooms = [
        room
        for transport in _TRANSPORTS
        if kind in (None, transport.kind)
        for room in transport.list_rooms()
    ]
    matchers: tuple[Callable[[RoomRef], bool], ...] = (
        lambda room: room.name == want,
        lambda room: room.name.lower() == want.lower(),
        lambda room: room.room_id == want,
    )
    for matcher in matchers:
        hits = [room for room in rooms if matcher(room)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            listed = ", ".join(f"{room.kind}:{room.room_id}" for room in hits)
            raise GroupCLIError(
                f"Group {want!r} is ambiguous: {listed}. Pass the room_id, or --kind <kind>."
            )
    names = ", ".join(sorted({f"{room.name} ({room.kind})" for room in rooms})) or "(none)"
    raise GroupCLIError(f"No group named {want!r}. Groups on this machine: {names}")


# ── hosted transport ─────────────────────────────────────────────────────────


class HostedTransport:
    """Rooms in the gateway's ``state.db``, driven by the hosted-room worker."""

    kind = "hosted"

    @staticmethod
    def _db():
        return hosted_rooms.default_db_path()

    def list_rooms(self) -> list[RoomRef]:
        local = hosted_rooms.local_authority_gateway_id()
        out: list[RoomRef] = []
        for room in hosted_rooms.list_rooms(self._db()):
            members_raw = room.get("members")
            members: list[Any] = members_raw if isinstance(members_raw, list) else []
            out.append(
                RoomRef(
                    kind=self.kind,
                    room_id=str(room["room_id"]),
                    name=str(room["name"]),
                    members=tuple(
                        str(m.get("handle") or m.get("profile") or m.get("member_id") or "?")
                        for m in members
                        if isinstance(m, dict)
                    ),
                    managed_here=str(room["authority_gateway_id"]) == local,
                    raw=room,
                )
            )
        return out

    def send(self, ref, *, text, thread, label, event_key):
        from gateway import hosted_room_discussion as discussion

        if not ref.managed_here:
            raise GroupCLIError(
                f"Group {ref.name!r} is managed by another gateway "
                f"({ref.raw.get('authority_gateway_id')}); send from that machine."
            )
        try:
            payload = discussion.validate_user_payload({"text": text, "thread_id": thread})
        except discussion.DiscussionValidationError as exc:
            raise GroupCLIError(str(exc)) from exc
        actor = {
            "kind": "user",
            "id": "cli",
            "profile": _profile(),
            "display_name": label,
        }
        event = hosted_rooms.append_event(
            self._db(),
            room_id=ref.room_id,
            event_id=hosted_rooms.user_event_id(event_key or f"cli:{uuid.uuid4().hex}"),
            kind="message.user",
            actor=actor,
            payload=payload,
            authority_gateway_id=str(ref.raw["authority_gateway_id"]),
            authority_epoch=int(ref.raw["authority_epoch"]),
        )
        return SentMessage(
            ref=ref,
            message_id=str(event["event_id"]),
            seq=int(event["seq"]),
            thread=str(payload["thread_id"]),
            raw=event,
        )

    def _handles(self, ref: RoomRef) -> dict[str, str]:
        members_raw = ref.raw.get("members")
        members: list[Any] = members_raw if isinstance(members_raw, list) else []
        return {
            str(m.get("member_id")): str(m.get("handle") or m.get("member_id"))
            for m in members
            if isinstance(m, dict) and m.get("member_id") is not None
        }

    def _page(self, room_id: str, since_seq: int) -> dict[str, Any]:
        return hosted_rooms.read_events(
            self._db(),
            room_id=room_id,
            since_seq=since_seq,
            limit=hosted_rooms.MAX_LOG_LIMIT,
        )

    def wait(self, sent, *, timeout, poll_seconds, on_reply):
        if not (float(poll_seconds) > 0):
            raise GroupCLIError("--poll must be a positive number of seconds")
        room_id, my_id, my_seq = sent.ref.room_id, sent.message_id, int(sent.seq or 0)
        handles = self._handles(sent.ref)
        cursor = my_seq
        started = time.monotonic()
        deadline = started + max(0.0, float(timeout))
        warned = False
        saw_driver_activity = False
        pending: dict[str, dict[str, Any]] = {}  # message.member awaiting turn.settled
        replies: list[dict[str, Any]] = []
        while True:
            page = self._page(room_id, cursor)
            for event in page.get("events", []):
                cursor = int(event["seq"])
                kind = str(event.get("kind") or "")
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                if kind.startswith("turn.") or kind == "room.activity":
                    saw_driver_activity = True
                if kind == "message.member":
                    # Only OUR discussion's replies: other relays/threads in the
                    # same room interleave in the log and must not stream here.
                    if (
                        str(payload.get("discussion_event_id") or "") == my_id
                        and str(payload.get("thread_id") or "") == sent.thread
                    ):
                        pending[str(event["event_id"])] = event
                elif kind == "turn.settled":
                    committed = pending.pop(str(payload.get("message_event_id") or ""), None)
                    if (
                        committed is not None
                        and str(payload.get("discussion_event_id") or "") == my_id
                    ):
                        member_id = str(committed["payload"].get("member_id") or "?")
                        text = str(committed["payload"].get("text") or "")
                        replies.append(
                            {
                                "member_id": member_id,
                                "handle": handles.get(member_id, member_id),
                                "text": text,
                                "seq": int(committed["seq"]),
                            }
                        )
                        on_reply(f"@{handles.get(member_id, member_id)}", text)
                elif (
                    kind == "room.activity"
                    and str(payload.get("discussion_event_id") or "") == my_id
                    and payload.get("status") in ("settled", "bounded")
                ):
                    return EXIT_OK, {
                        "status": str(payload["status"]),
                        "reason": str(payload.get("reason_code") or ""),
                        "replies": replies,
                    }
                elif kind == "room.stop_requested" and cursor > my_seq:
                    # The stop fence is ROOM-WIDE by design: the policy's
                    # stopped_through_seq supersedes every earlier user turn in
                    # every thread (gateway/hosted_room_discussion.py
                    # plan_next_task). A stop issued after our send therefore
                    # cancels our discussion too, so exit 4 is the truth even
                    # when the stop was aimed at another thread.
                    return EXIT_SUPERSEDED, {
                        "status": "stopped",
                        "reason": "room.stop_requested",
                        "replies": replies,
                    }
            if page.get("has_more"):
                continue
            now = time.monotonic()
            if now >= deadline:
                return EXIT_TIMEOUT, {"status": "timeout", "reason": "timeout", "replies": replies}
            if (
                not warned
                and not saw_driver_activity
                and now - started >= _WORKER_WARN_AFTER_SECONDS
            ):
                print(
                    "hermes group: no gateway worker has picked this up yet — is "
                    "`hermes gateway` (or the Desktop backend) running?",
                    file=sys.stderr,
                    flush=True,
                )
                warned = True
            time.sleep(float(poll_seconds))

    def log(self, ref, *, since):
        handles = self._handles(ref)
        cursor = int(since)
        pending: dict[str, dict[str, Any]] = {}
        while True:
            page = self._page(ref.room_id, cursor)
            for event in page.get("events", []):
                cursor = int(event["seq"])
                kind = str(event.get("kind") or "")
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                if kind == "message.user":
                    actor = event.get("actor") if isinstance(event.get("actor"), dict) else {}
                    who = str(actor.get("display_name") or "user")
                    yield {
                        "seq": cursor,
                        "speaker": f"User ({who})",
                        "text": str(payload.get("text") or ""),
                        "thread": str(payload.get("thread_id") or ""),
                    }
                elif kind == "message.member":
                    pending[str(event["event_id"])] = event
                elif kind == "turn.settled":
                    committed = pending.pop(str(payload.get("message_event_id") or ""), None)
                    if committed is not None:
                        member_id = str(committed["payload"].get("member_id") or "?")
                        yield {
                            "seq": int(committed["seq"]),
                            "speaker": f"@{handles.get(member_id, member_id)}",
                            "text": str(committed["payload"].get("text") or ""),
                            "thread": str(committed["payload"].get("thread_id") or ""),
                        }
            if not page.get("has_more"):
                return


register_transport(HostedTransport())


# ── commands ─────────────────────────────────────────────────────────────────


def _cmd_list(args) -> int:
    rows = [room for transport in _TRANSPORTS for room in transport.list_rooms()]
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "kind": room.kind,
                        "name": room.name,
                        "room_id": room.room_id,
                        "members": list(room.members),
                        "managed_here": room.managed_here,
                    }
                    for room in rows
                ],
                indent=2,
            )
        )
        return EXIT_OK
    if not rows:
        print("No groups found. Create a hosted one: hermes group create <name> --member a --member b")
        return EXIT_OK
    for room in rows:
        tail = "" if room.managed_here else "  [managed elsewhere]"
        print(f"{room.name}  [{room.kind}] ({room.room_id})  members: {', '.join(room.members)}{tail}")
    return EXIT_OK


def _local_profiles(root) -> tuple[str, ...]:
    names = {"default"}
    profiles_dir = root / "profiles"
    if profiles_dir.is_dir():
        names.update(p.name for p in profiles_dir.iterdir() if p.is_dir())
    return tuple(sorted(names))


def _cmd_create(args) -> int:
    from gateway import hosted_room_discussion as discussion

    def handle(profile: str) -> str:
        return "hermes" if profile == "default" else profile

    members = [
        {"member_id": profile, "profile": profile, "handle": handle(profile)}
        for profile in args.member
    ]
    db = hosted_rooms.default_db_path()
    try:
        discussion.validate_roster(members, local_profiles=_local_profiles(db.parent))
    except discussion.DiscussionValidationError as exc:
        raise GroupCLIError(str(exc)) from exc
    room = hosted_rooms.create_room(
        db,
        room_id=f"grp-{uuid.uuid4().hex[:12]}",
        name=args.name,
        members=members,
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )
    if args.json:
        print(json.dumps(room, indent=2))
    else:
        print(f"created {room['name']} ({room['room_id']}) members: {', '.join(args.member)}")
    return EXIT_OK


def _print_reply(speaker: str, text: str) -> None:
    print(f"{speaker}: {text}\n", flush=True)


def _cmd_send(args) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    text = (text or "").strip()
    if not text:
        raise GroupCLIError("message text is required (argument or stdin)")
    ref = resolve_room(args.group, kind=getattr(args, "kind", None))
    transport = _transport_for(ref.kind)
    label = (args.as_label or f"{_profile()} relay").strip()
    session = _session_id(args)
    if args.thread:
        thread = thread_id(args.thread)
    elif session and not args.new_thread and bound_thread(ref, session):
        thread = str(bound_thread(ref, session))
    elif session:
        # New thread for this session: a session-derived label, distinct
        # from other sessions' threads. Transports that mint their thread on
        # delivery bind it after --wait instead (see below).
        thread = thread_id(f"s-{session}") if ref.kind == "hosted" else DEFAULT_THREAD
        forget_thread(ref, session)
    else:
        thread = DEFAULT_THREAD
    sent = transport.send(
        ref,
        text=text,
        thread=thread,
        label=label,
        event_key=args.event_id,
    )
    if session and ref.kind == "hosted" and not args.thread:
        bind_thread(ref, session, sent.thread)
    if not args.wait:
        if args.json:
            print(
                json.dumps(
                    {
                        "kind": ref.kind,
                        "room_id": ref.room_id,
                        "name": ref.name,
                        "message_id": sent.message_id,
                        "seq": sent.seq,
                        "thread": sent.thread,
                        "raw": sent.raw,
                    },
                    indent=2,
                )
            )
        else:
            print(f"sent to {ref.name} [{ref.kind}] id={sent.message_id} thread={sent.thread}")
        return EXIT_OK
    rc, summary = transport.wait(
        sent,
        timeout=float(args.timeout),
        poll_seconds=float(args.poll),
        on_reply=(lambda *_: None) if args.json else _print_reply,
    )
    if session and summary.get("thread") and not args.thread:
        # An explicit --thread is a one-off; it must not redirect the
        # session's implicit continuity.
        bind_thread(ref, session, str(summary["thread"]))
    if args.json:
        print(json.dumps({"kind": ref.kind, "room_id": ref.room_id, "name": ref.name, **summary}, indent=2))
    else:
        count = len(summary.get("replies") or [])
        line = f"[group {ref.name}: {summary.get('status')} ({summary.get('reason')}), {count} replies]"
        print(line, file=sys.stderr if rc else sys.stdout, flush=True)
    return rc


def _cmd_log(args) -> int:
    ref = resolve_room(args.group, kind=getattr(args, "kind", None))
    rows = list(_transport_for(ref.kind).log(ref, since=int(args.since)))
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    for row in rows:
        print(f"{row['speaker']}: {row['text']}")
    return EXIT_OK


_ACTIONS = {
    "list": _cmd_list,
    "ls": _cmd_list,
    "create": _cmd_create,
    "send": _cmd_send,
    "log": _cmd_log,
}


def cmd_group(args) -> int:
    action = getattr(args, "group_action", None)
    if action not in _ACTIONS:
        print("usage: hermes group {list,create,send,log} ...", file=sys.stderr)
        return EXIT_USAGE
    try:
        return _ACTIONS[action](args)
    except (GroupCLIError, hosted_rooms.HostedRoomError) as exc:
        print(f"hermes group: {exc}", file=sys.stderr)
        return EXIT_ERROR


def build_args(argv: list[str]) -> argparse.Namespace:
    """Test helper: parse ``argv`` exactly as the top-level CLI would."""
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers()
    build_group_parser(subparsers)
    return parser.parse_args(["group", *argv])


def build_group_parser(subparsers) -> None:
    """Attach the ``group`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "group",
        help="Relay user messages into Group Chats (from any session)",
        description=(
            "Write into a Group Chat AS THE USER and follow the deliberation. The "
            "caller is a relay, not a member: the message lands attributed to the "
            "user with your profile/label recorded as who relayed it. Run "
            "'hermes group send <group> \"...\" --wait' in the background from any "
            "agent session; its output carries the members' replies."
        ),
        epilog=(
            "Threads follow the relaying SESSION: the first send from a session "
            "starts a new room thread; later sends from the same session "
            "($HERMES_SESSION_ID, or --session) continue it; a different session "
            "gets its own thread. --new-thread starts over; --thread <id|label> "
            "overrides for that send only (it does not change the session's "
            "thread). Without any session id the shared 'cli' thread is used. "
            "A room keeps only the LATEST pending user message per thread.\n"
            "\n"
            "Exit codes: 0 settled/bounded, 1 error, 2 usage, 3 timeout "
            "(partial replies already printed), 4 superseded by a room stop. A "
            "room stop is room-wide: it cancels every pending user turn in every "
            "thread, including this relay's.\n"
            "\n"
            "Examples:\n"
            "  hermes group list\n"
            "  hermes group create DevTeam --member archie --member mason\n"
            '  hermes group send DevTeam "Plan the release checklist" --as "Pax via Discord" --wait\n'
            "  hermes group log DevTeam --since 0\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = parser.add_subparsers(dest="group_action")

    ls = actions.add_parser("list", aliases=["ls"], help="List groups on this machine")
    ls.add_argument("--json", action="store_true", default=False)

    create = actions.add_parser("create", help="Create a gateway-hosted group (2-6 local profiles)")
    create.add_argument("name")
    create.add_argument(
        "--member", action="append", required=True, metavar="PROFILE", help="Repeat per member"
    )
    create.add_argument("--json", action="store_true", default=False)

    send = actions.add_parser("send", help="Relay a message into a group as the user")
    send.add_argument("group", help="Group name or room_id")
    send.add_argument("text", nargs="?", default=None, help="Message text (or stdin)")
    send.add_argument("--thread", default=None, help="Explicit thread (see epilog)")
    send.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Relaying session id for thread continuity (default: $HERMES_SESSION_ID)",
    )
    send.add_argument(
        "--new-thread",
        action="store_true",
        default=False,
        help="Start a new room thread even if this session already has one",
    )
    send.add_argument(
        "--as",
        dest="as_label",
        default=None,
        metavar="LABEL",
        help="Who relayed, e.g. 'Pax via Discord' (default '<profile> relay')",
    )
    send.add_argument(
        "--event-id",
        default=None,
        metavar="KEY",
        help="Idempotent retry key; resending the same key+text is a no-op",
    )
    send.add_argument("--wait", action="store_true", default=False, help="Stream replies until settled")
    send.add_argument("--timeout", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS, metavar="SECONDS")
    send.add_argument("--poll", type=float, default=1.0, help=argparse.SUPPRESS)
    send.add_argument("--json", action="store_true", default=False)

    log = actions.add_parser("log", help="Print the group's committed transcript")
    log.add_argument("group", help="Group name or room_id")
    log.add_argument("--since", type=int, default=0, metavar="SEQ")
    log.add_argument("--json", action="store_true", default=False)

    parser.set_defaults(func=cmd_group)
