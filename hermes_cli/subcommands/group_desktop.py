"""Desktop-room transport for ``hermes group`` — reaches Group Chats that the
Hermes Desktop coordinates (plugin storage + renderer-driven rounds).

Interim bridge until the Desktop client adopts gateway-hosted rooms: the CLI
cannot start a Desktop round itself, so ``send`` queues an envelope in
``<root>/group_relay/`` (``tools/group_relay.py``) and the open Desktop drains
it, calls ``sendToGroupChat`` on the user's behalf, and streams progress
lines back that ``--wait`` tails. Rooms are discovered from the compact
projection the Desktop mirrors into the default profile's
``ui_meta['hermes-bots-groups']`` (``<root>/profile.yaml``).

Registered through :func:`hermes_cli.subcommands.group.register_transport`
at import; deleting this module (and its one-line import in ``group.py``)
removes the transport cleanly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Iterator

from tools import group_relay

from . import group as base

GROUP_CHAT_SYNC_META_KEY = "hermes-bots-groups"
_DESKTOP_WARN_AFTER_SECONDS = 30.0
# Fail fast when the outbox is visibly not being drained (Desktop closed).
_STALE_OUTBOX_MIN_AGE_SECONDS = 60.0
_STALE_OUTBOX_MAX_PENDING = 20


def _root() -> Path:
    return group_relay.gateway_root()


def read_projection(root: Path | None = None) -> dict[str, Any]:
    """The Desktop's ``hermes-bots-groups`` mirror, or ``{}``. Never raises."""
    path = (root or _root()) / "profile.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    ui_meta = data.get("ui_meta") if isinstance(data, dict) else None
    snapshot = ui_meta.get(GROUP_CHAT_SYNC_META_KEY) if isinstance(ui_meta, dict) else None
    return snapshot if isinstance(snapshot, dict) else {}


def _entries(room: dict[str, Any]) -> list[dict[str, Any]]:
    log = room.get("log")
    return [e for e in log if isinstance(e, dict)] if isinstance(log, list) else []


def _author(entry: dict[str, Any]) -> dict[str, Any]:
    author = entry.get("from")
    return author if isinstance(author, dict) else {}


def _member_handles(room: dict[str, Any]) -> tuple[str, ...]:
    members = room.get("members")
    names: list[str] = []
    if isinstance(members, list):
        for member in members:
            if isinstance(member, dict) and member.get("name"):
                names.append(str(member["name"]))
    if not names:
        seen: dict[str, None] = {}
        for entry in _entries(room):
            author = _author(entry)
            if author.get("kind") == "member" and author.get("name"):
                seen.setdefault(str(author["name"]), None)
        names = list(seen)
    return tuple(names)


class DesktopTransport:
    """Desktop-coordinated rooms, reached through the group_relay outbox."""

    kind = "desktop"

    def list_rooms(self) -> list[base.RoomRef]:
        rooms = read_projection().get("rooms")
        out: list[base.RoomRef] = []
        if not isinstance(rooms, dict):
            return out
        for key, room in rooms.items():
            if not isinstance(room, dict):
                continue
            room_id = str(room.get("roomId") or (key[3:] if str(key).startswith("id:") else "") or "")
            name = str(room.get("name") or (key[5:] if str(key).startswith("name:") else key) or "")
            if not name:
                continue
            out.append(
                base.RoomRef(
                    kind=self.kind,
                    room_id=room_id or name,
                    name=name,
                    members=_member_handles(room),
                    managed_here=True,
                    raw=room,
                )
            )
        return out

    def send(self, ref, *, text, thread, label, event_key):
        stale = group_relay.pending_count(_root(), older_than_seconds=_STALE_OUTBOX_MIN_AGE_SECONDS)
        if stale >= _STALE_OUTBOX_MAX_PENDING:
            raise base.GroupCLIError(
                f"{stale} relay envelopes have sat undrained for over "
                f"{int(_STALE_OUTBOX_MIN_AGE_SECONDS)}s — the Hermes Desktop does not "
                "appear to be open. Open it (Bots pane) and retry."
            )
        # The hosted default thread means "no explicit thread" here → the
        # Desktop mints a new one; a bound/explicit Desktop thread id continues.
        requested = None if thread == base.DEFAULT_THREAD else thread
        try:
            envelope = group_relay.enqueue(
                _root(),
                room_id=ref.room_id if ref.room_id != ref.name else "",
                room_name=ref.name,
                text=text,
                from_profile=base._profile(),
                label=label,
                thread=requested,
                event_key=event_key,
            )
        except group_relay.GroupRelayError as exc:
            raise base.GroupCLIError(str(exc)) from exc
        return base.SentMessage(
            ref=ref,
            message_id=str(envelope["id"]),
            seq=None,
            thread=requested or "(new thread — assigned by the Desktop)",
            raw=envelope,
        )

    def wait(self, sent, *, timeout, poll_seconds, on_reply):
        if not (float(poll_seconds) > 0):
            raise base.GroupCLIError("--poll must be a positive number of seconds")
        offset = 0
        started = time.monotonic()
        deadline = started + max(0.0, float(timeout))
        accepted_thread: str | None = None
        warned = False
        replies: list[dict[str, Any]] = []
        while True:
            lines, offset = group_relay.read_reply_lines(_root(), sent.message_id, offset=offset)
            for line in lines:
                kind = str(line.get("kind") or "")
                if kind == "accepted":
                    accepted_thread = str(line.get("thread") or "")
                elif kind == "reply":
                    member = str(line.get("member") or "bot")
                    text = str(line.get("text") or "")
                    replies.append({"handle": member, "member_id": member, "text": text})
                    on_reply(f"@{member}", text)
                elif kind == "done":
                    status = str(line.get("status") or "")
                    summary = {"status": status, "reason": status, "thread": accepted_thread, "replies": replies}
                    if status in ("settled", "capped"):
                        return base.EXIT_OK, summary
                    if status == "cancelled":
                        return base.EXIT_SUPERSEDED, summary
                    return base.EXIT_TIMEOUT, summary
                elif kind == "error":
                    raise base.GroupCLIError(
                        f"{line.get('error') or 'relay failed'}"
                        + (f" [reason: {line['reason']}]" if line.get("reason") else "")
                    )
            now = time.monotonic()
            if now >= deadline:
                return base.EXIT_TIMEOUT, {
                    "status": "timeout",
                    "reason": "timeout",
                    "thread": accepted_thread,
                    "replies": replies,
                }
            if not warned and accepted_thread is None and now - started >= _DESKTOP_WARN_AFTER_SECONDS:
                print(
                    "hermes group: the Desktop hasn't picked this up yet — is the Hermes "
                    "app open with the Bots pane loaded?",
                    file=sys.stderr,
                    flush=True,
                )
                warned = True
            time.sleep(float(poll_seconds))

    def log(self, ref, *, since) -> Iterator[dict[str, Any]]:
        for index, entry in enumerate(_entries(ref.raw)):
            if index < int(since):
                continue
            author = _author(entry)
            if author.get("kind") == "member":
                speaker = f"@{author.get('name') or 'bot'}"
            else:
                via = str(author.get("via") or "").strip()
                speaker = f"User ({via})" if via else "User (You)"
            yield {
                "seq": index,
                "speaker": speaker,
                "text": str(entry.get("text") or ""),
                "thread": str(entry.get("thread") or ""),
            }


base.register_transport(DesktopTransport())
