"""Cron delivery into a Bot Mode Group Chat (``deliver: room:<room_id>[@<profile>]``).

The job's output becomes one message in the room's log, labelled with the job's name. It starts a round
only for the members it @mentions (``@all`` addresses everyone); without a mention it is only a message in
the room, so a feed of scheduled results never makes every Bot answer it. Failure notices never start a
round. The post is appended to the hosting gateway's room store directly; the room worker picks it up on
its next poll, so delivery does not need the Desktop open. A room store is a home and a placement: the home's
``state.db`` (``'profile'``) or its install's ``shared-state.db`` (``'shared'``), reached through a storage
handle on that home; the probes create and migrate nothing.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from persistence_assembly import Persistence

# A room store: the home whose storage handle reaches it, and the placement that names its file.
RoomStore = tuple[Path, str]

logger = logging.getLogger(__name__)

ROOM_PLATFORM = "room"
_PREFIX = ROOM_PLATFORM + ":"
_ROOM_TOKEN_RE = re.compile(r"^(?P<room>[A-Za-z0-9][A-Za-z0-9._:-]*)(?:@(?P<profile>[A-Za-z0-9_-]+))?$")


def parse_room_deliver_token(part: str) -> Optional[str]:
    """``room:<id>[@<profile>]`` → ``"<id>[@<profile>]"``, ``""`` for a malformed room token, or None
    when ``part`` is not a room token at all."""
    raw = (part or "").strip()
    if not raw.lower().startswith(_PREFIX):
        return None
    rest = raw[len(_PREFIX):].strip()
    return rest if _ROOM_TOKEN_RE.fullmatch(rest) else ""


def resolve_room_target(job: dict, address: str) -> Optional[dict]:
    if not address:
        logger.warning("Job '%s': malformed room delivery target (expected room:<room_id>[@<profile>])",
                       job.get("id", "?"))
        return None
    return {"platform": ROOM_PLATFORM, "chat_id": address, "thread_id": None}


def _room_stores(profile: Optional[str]) -> tuple[RoomStore, RoomStore]:
    """Where a room can live: the hosting profile's session store (rooms its gateway owns), then the
    install's shared room store (rooms the dashboard backend owns)."""
    from hermes_constants import get_hermes_home
    if profile:
        from hermes_cli.profiles import get_profile_dir
        home = get_profile_dir(profile)
    else:
        home = get_hermes_home()
    return (home, "profile"), (get_hermes_home(), "shared")


@contextmanager
def _handle(store: RoomStore) -> Iterator[Persistence]:
    """A storage handle on the store's home, closed on exit."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope(home=store[0])) as handle:
        yield handle


def _find_room_store(room_id: str, profile: Optional[str]) -> Optional[RoomStore]:
    """The store whose worker serves ``room_id``. The session store counts only when its gateway holds
    the room's owner entry: the shared-store import copies rooms out of ``state.db`` without deleting
    them, and nothing serves that leftover copy, so a post there would never reach the visible room."""
    from gateway.hosted_rooms import default_db_path, probe_hosted_room
    session_store, shared_store = _room_stores(profile)
    if _serves(session_store, room_id):
        return session_store
    with _handle(shared_store) as handle:
        if not probe_hosted_room(handle, placement=shared_store[1], room_id=room_id):
            return None
    # A shared room the root gateway imported is fenced here and lives on in the ROOT home's state.db
    # (only that gateway imports), whichever profile's view advertised the target.
    from gateway.hosted_rooms_gateway_import import handed_room_ids
    shared_file = default_db_path()
    root_store = (shared_file.parent, "profile")
    if room_id in handed_room_ids(shared_file) and _serves(root_store, room_id):
        return root_store
    return shared_store


def _serves(store: RoomStore, room_id: str) -> bool:
    from gateway.hosted_rooms import probe_hosted_room
    from gateway.session_hosted_service import probe_native_room_owner
    with _handle(store) as handle:
        return (probe_hosted_room(handle, placement=store[1], room_id=room_id)
                and probe_native_room_owner(handle, room_id=room_id))


def deliver_to_room(job: dict, content: str, address: str, *, for_failure: bool = False) -> Optional[str]:
    """Post one cron output into a hosted room. None on success, else an error string."""
    from cron.scheduler_delivery import _redact_cron_payload
    from gateway.hosted_room_context import post_scheduled_message
    from gateway.hosted_rooms import AuthorityConflictError
    from persistence import PersistenceError

    target = f"room:{address}"
    match = _ROOM_TOKEN_RE.fullmatch(address)
    if match is None:
        return f"{target}: malformed room delivery target"
    room_id, profile = match.group("room"), match.group("profile")
    from hermes_cli.profiles import profile_exists
    if profile and not profile_exists(profile):
        return f"{target}: no profile named {profile!r} on this machine (Room settings shows the exact target)"
    try:
        store = _find_room_store(room_id, profile)
        if store is None:
            return f"{target}: no live Group Chat with that id on this machine"
        job_id = str(job.get("id") or "job")
        # The durable execution id makes a re-delivered run idempotent; direct helper calls get a fresh key.
        run_key = str(job.get("execution_id") or job.setdefault("_room_run_key", uuid.uuid4().hex))
        with _handle(store) as handle:
            event = post_scheduled_message(
                handle, placement=store[1], room_id=room_id, job_id=job_id,
                job_name=_redact_cron_payload(str(job.get("name") or job_id), "job name"),
                run_key=run_key, text=_redact_cron_payload(content, "room payload"), notice=for_failure)
        logger.info("Job '%s': %s Group Chat %s (seq %s)", job_id,
                    "already posted this run to" if event.get("idempotent") else "posted to", room_id, event.get("seq"))
        return None
    except AuthorityConflictError as exc:
        return f"{target} delivery failed: {exc} {_profile_hint(room_id, profile)}"
    except (ValueError, OSError, PersistenceError) as exc:  # room/validation errors are ValueErrors
        return f"{target} delivery failed: {exc}"


def _profile_hint(room_id: str, profile: Optional[str]) -> str:
    """A room a profile's own gateway took over leaves only a fenced shared copy here, which answers "managed
    by another gateway"; the remedy is naming the hosting profile so its session store is the one posted to."""
    lead = f"@{profile} does not host it. " if profile else ""
    return (f"{lead}If a profile's gateway hosts this room, deliver to room:{room_id}@<that profile> "
            "(Room settings shows the exact target).")
