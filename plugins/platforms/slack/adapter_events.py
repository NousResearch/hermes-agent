"""Slack inbound event classification before nested-message normalization."""

from typing import Any


_SLACK_VISIBLE_MESSAGE_FIELDS = ("text", "blocks", "attachments", "files")
_SLACK_REPLY_METADATA_FIELDS = (
    "reply_count",
    "reply_users_count",
    "reply_users",
    "latest_reply",
    "replies",
)


def _slack_visible_message_value(message: dict, field: str) -> Any:
    """Normalize absent and empty Slack display fields to the same value."""
    value = message.get(field)
    return None if value in (None, "", [], {}) else value


def _is_hidden_thread_parent_metadata_update(event: dict, message: dict) -> bool:
    """Return whether Slack changed only reply bookkeeping on a thread parent.

    Slack emits hidden ``message_changed`` events when a thread receives a
    reply. Replaying the nested parent as user input is unsafe after restart,
    when the process-local processed-message cache is empty. Preserve real
    edits whenever Slack supplies a visible before/after delta. If the prior
    snapshot is malformed or absent, accept only an explicit edit whose edit
    timestamp identifies this event; ambiguous hidden parent updates fail
    closed.
    """
    if not event.get("hidden"):
        return False
    previous = event.get("previous_message")
    previous_for_metadata = previous if isinstance(previous, dict) else {}
    if not any(
        field in message or field in previous_for_metadata
        for field in _SLACK_REPLY_METADATA_FIELDS
    ):
        return False

    # Slack snapshots omit absent display fields; compare their normalized union
    # so additions and removals both count. Metadata-only partial snapshots do
    # not establish a visible delta and use the explicit-edit fallback below.
    if isinstance(previous, dict) and any(
        field in previous for field in _SLACK_VISIBLE_MESSAGE_FIELDS
    ):
        return not any(
            _slack_visible_message_value(previous, field)
            != _slack_visible_message_value(message, field)
            for field in _SLACK_VISIBLE_MESSAGE_FIELDS
        )

    edited = message.get("edited")
    edited_ts = str(edited.get("ts") or "") if isinstance(edited, dict) else ""
    changed_ts = str(event.get("event_ts") or event.get("ts") or "")
    return not (edited_ts and changed_ts and edited_ts == changed_ts)
