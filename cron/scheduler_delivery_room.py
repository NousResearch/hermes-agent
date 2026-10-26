"""Cron delivery into a Bot Mode Group Chat (``deliver: room:<room_id>[@<profile>]``).

The job's output becomes one message in the room's log, labelled with the job's name. It starts a round
only for the members it @mentions (``@all`` addresses everyone); without a mention it is only a message in
the room, so a feed of scheduled results never makes every Bot answer it. Failure notices never start a
round. The post is appended to the hosting gateway's room store directly; the room worker picks it up on
its next poll, so delivery does not need the Desktop open.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

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


def _room_stores(profile: Optional[str]) -> list[Path]:
    """Where a room can live: the hosting profile's session store (rooms the gateway owns), then the
    install's shared room store (rooms the dashboard backend owns)."""
    from gateway.hosted_rooms import default_db_path
    from hermes_constants import get_hermes_home
    if profile:
        from hermes_cli.profiles import get_profile_dir
        home = get_profile_dir(profile)
    else:
        home = get_hermes_home()
    return [home / "state.db", default_db_path()]


def _find_room_store(room_id: str, profile: Optional[str]) -> Optional[Path]:
    from gateway.hosted_rooms import probe_hosted_room
    return next((path for path in _room_stores(profile) if probe_hosted_room(path, room_id=room_id)), None)


def deliver_to_room(job: dict, content: str, address: str, *, for_failure: bool = False) -> Optional[str]:
    """Post one cron output into a hosted room. None on success, else an error string."""
    from cron.scheduler_delivery import _redact_cron_payload
    from gateway.hosted_room_context import post_scheduled_message

    target = f"room:{address}"
    match = _ROOM_TOKEN_RE.fullmatch(address)
    if match is None:
        return f"{target}: malformed room delivery target"
    room_id, profile = match.group("room"), match.group("profile")
    try:
        store = _find_room_store(room_id, profile)
        if store is None:
            return f"{target}: no live Group Chat with that id on this machine"
        job_id = str(job.get("id") or "job")
        # The durable execution id makes a re-delivered run idempotent; direct helper calls get a fresh key.
        run_key = str(job.get("execution_id") or job.setdefault("_room_run_key", uuid.uuid4().hex))
        event = post_scheduled_message(
            store, room_id=room_id, job_id=job_id,
            job_name=_redact_cron_payload(str(job.get("name") or job_id), "job name"),
            run_key=run_key, text=_redact_cron_payload(content, "room payload"), notice=for_failure)
        logger.info("Job '%s': posted to Group Chat %s (seq %s)", job_id, room_id, event.get("seq"))
        return None
    except (ValueError, OSError, sqlite3.Error) as exc:  # room/validation errors are ValueErrors
        return f"{target} delivery failed: {exc}"
