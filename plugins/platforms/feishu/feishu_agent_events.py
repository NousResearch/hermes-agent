"""Preserve subscribed non-chat events for local automation consumers.

Consumers tail ``feishu-user/agent-events.jsonl`` under the receiving profile's
Hermes home. An optional ``agent-events.sock`` receives the same JSON line as a
best-effort wakeup; consumers should use header.event_id to deduplicate retries.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# The SDK rejects unregistered subscriptions before consumers can see them.
# Include tenant/all-meeting variants: registering only user/participant keys
# leaves task changes and completed meetings silently invisible to users.
AGENT_EVENT_TYPES = (
    "minutes.minute.generated_v1",
    "vc.meeting.participant_meeting_ended_v1",
    "vc.meeting.participant_meeting_started_v1",
    "vc.meeting.participant_meeting_joined_v1",
    "task.task.update_user_access_v2",
    "task.task.update_tenant_v1",
    "approval.task.status_changed_v4",
    "approval.instance.status_changed_v4",
    "vc.recording.recording_started_v1",
    "vc.recording.recording_ended_v1",
    "vc.recording.recording_transcript_generated_v1",
    "vc.meeting.all_meeting_started_v1",
    "vc.meeting.all_meeting_ended_v1",
)


def forward_agent_event(event_key: str, data: Any) -> None:
    """Append before notifying so an offline consumer can catch up from the journal."""
    try:
        # CustomizedEvent.event is already a dict; webhook ingress supplies the
        # original mapping to preserve nested payloads without a lossy repr().
        if isinstance(data, dict):
            payload, header = data.get("event"), data.get("header") or {}
        else:
            payload = data.event
            header = vars(data.header) if data.header is not None else {}
        record = {
            "ts": int(time.time()),
            "event_key": event_key,
            # Verification tokens and tenant/app identifiers are not needed by consumers.
            "header": {key: header.get(key) for key in ("event_id", "event_type", "create_time")},
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        base = get_hermes_home() / "feishu-user"
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(base / "agent-events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            stream.write(line)
    except (OSError, TypeError, ValueError):
        logger.warning("[Feishu] Failed to persist agent event %s", event_key, exc_info=True)
        return

    try:
        sock_path = base / "agent-events.sock"
        if hasattr(socket, "AF_UNIX") and sock_path.exists():
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.5)
                client.connect(str(sock_path))
                client.sendall(line.encode("utf-8"))
    except OSError:
        logger.warning(
            "[Feishu] Agent event %s persisted, but socket notification failed",
            event_key, exc_info=True,
        )
