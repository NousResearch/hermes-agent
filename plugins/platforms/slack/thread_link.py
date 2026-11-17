"""Slack links rebuilt from ids alone: the ``archives/<channel>/p<ts>`` permalink tail and the
``thread_url`` the session lists open. Kept apart from adapter.py, which is over its size cap."""

from urllib.parse import quote


def _slack_permalink_path(channel_id: str | None, message_ts: str | None) -> str:
    """Workspace-independent tail (``archives/<channel>/p<ts>``) of a permalink.
    Only the tail can be rebuilt from a payload, so dedupe compares on it."""
    if not channel_id or not message_ts:
        return ""
    return f"archives/{channel_id}/p{str(message_ts).replace('.', '')}"


def slack_thread_url(source) -> str | None:
    """The session's channel or thread: in the workspace's web client when the team id is known,
    else through slack.com, which forwards to whichever workspace the reader is signed in to."""
    channel = quote(str(source.chat_id or ""), safe="")
    if not channel:
        return None
    team = quote(str(source.scope_id or ""), safe="")
    thread_ts = str(source.thread_id or "")
    if team:
        tail = f"/thread/{channel}-{quote(thread_ts, safe='.')}" if thread_ts else ""
        return f"https://app.slack.com/client/{team}/{channel}{tail}"
    if thread_ts:
        return f"https://slack.com/{_slack_permalink_path(channel, thread_ts)}"
    return f"https://slack.com/app_redirect?channel={channel}"
