"""``/branch`` destination on thread-capable platforms (#66023).

On Discord, Telegram, Slack and Matrix a plain ``/branch`` opens a NEW sibling thread for the
clone and leaves the current chat bound to the original session, so the user can keep both paths
alive. ``/branch --here`` keeps the legacy in-place switch; platforms without threads (and the CLI)
always branch in place. Pure helpers only — the handler lives in ``slash_commands_session.py``.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Optional

from gateway.config import Platform
from gateway.session import SessionSource

# Adapters that override ``BasePlatformAdapter.create_handoff_thread`` (the base returns None).
BRANCH_THREAD_PLATFORMS = frozenset({Platform.DISCORD, Platform.TELEGRAM, Platform.SLACK, Platform.MATRIX})
BRANCH_HERE_FLAG = "--here"


def parse_branch_args(raw: str) -> tuple[bool, str]:
    """``(stay_here, title)`` from the text after ``/branch``.

    Only a leading ``--here`` is a flag; everything else is the optional title. The CLI strips the
    flag through the same parser so ``/branch --here` never becomes a session titled ``--here``.
    """
    text = (raw or "").strip()
    head, _, rest = text.partition(" ")
    if head.lower() == BRANCH_HERE_FLAG:
        return True, rest.strip()
    return False, text


def branch_thread_parent(source: SessionSource) -> Optional[str]:
    """Chat that can host a sibling thread for *source*, or None when nothing can (Discord DMs,
    a Discord thread whose parent channel is unknown)."""
    if source.platform not in BRANCH_THREAD_PLATFORMS:
        return None
    if source.platform == Platform.DISCORD:
        if source.chat_type == "dm":
            return None
        # Inbound Discord threads carry ``chat_id == thread_id``; the sibling goes under the real
        # text channel, which only ``parent_chat_id`` names.
        if source.thread_id or source.chat_type == "thread":
            return str(source.parent_chat_id) if source.parent_chat_id else None
    # Telegram forum topics, Slack threads and Matrix threads all key on the parent chat/room id.
    return str(source.chat_id) if source.chat_id else None


def branch_dest_source(source: SessionSource, *, parent_id: str, thread_id: str, title: str) -> SessionSource:
    """The ``SessionSource`` a follow-up typed in the new thread will arrive on — the shape must
    match each adapter's inbound source or the clone is bound to a key nobody ever reads (the
    CLI→platform handoff in ``run_startup.py::_handoff_resolve_destination`` mirrors the same
    rules). Per-message fields are dropped; identity/scope fields travel with the copy."""
    common = dict(thread_id=str(thread_id), message_id=None, prospective_thread_id=None,
                  auto_thread_created=False, auto_thread_initial_name=None)
    if source.platform == Platform.DISCORD:
        # Discord keys an in-thread message on the thread's OWN id as chat_id.
        return dataclasses.replace(source, chat_id=str(thread_id), chat_name=title or source.chat_name,
                                   chat_type="thread", parent_chat_id=str(parent_id), **common)
    # Telegram (``group:<chat>:<topic>``, private-chat topics stay ``dm``), Slack (parent channel's
    # dm/group + workspace scope) and Matrix (room type) key a thread reply on the PARENT chat's type.
    chat_type = "group" if source.chat_type == "thread" else source.chat_type
    return dataclasses.replace(source, chat_id=str(parent_id), chat_type=chat_type, **common)


def format_thread_ref(platform: Optional[Platform], thread_id: str) -> str:
    """Clickable pointer where the platform has one (Discord ``<#id>`` mentions); id otherwise."""
    if platform == Platform.DISCORD:
        return f"<#{thread_id}>"
    return f"`{thread_id}`"


# ── Slack permalink targets (/branch <message link>) ──────────────────────────
# Workspace-independent tail of a Slack permalink: ``archives/<channel>/p<ts>`` where the message
# ts is the dotless form (10-digit seconds + 6-digit microseconds, e.g. ``p1793128064260549``).
_SLACK_ARCHIVE_LINK_RE = re.compile(r"archives/([A-Z0-9]+)/p(\d{16})")
_SLACK_THREAD_TS_RE = re.compile(r"[?&]thread_ts=(\d{6,14}\.\d{1,9})")


@dataclasses.dataclass(frozen=True)
class SlackLinkTarget:
    """One parsed Slack thread permalink: the thread a session is keyed on, plus the message.

    ``thread_ts`` names the session (a thread message's permalink always carries it); the
    ``p<ts>`` path names the branch-point message."""

    channel_id: str
    thread_ts: str
    message_ts: str


def _link_message_ts(digits: str) -> str:
    """``p1793128064260549`` → ``1793128064.260549`` — the dotted form rows store."""
    return f"{digits[:-6]}.{digits[-6:]}"


def parse_slack_archive_link(text: str) -> Optional[SlackLinkTarget]:
    """First Slack thread permalink in *text*, or None when there is none.

    Works on bare URLs and on Slack's mrkdwn autolink form (``<https://…?thread_ts=…|https://…>``
    — the label repeats the URL, so both halves carry the same path). A permalink without
    ``thread_ts`` points at a channel-root message, whose session key cannot be reconstructed
    from the link alone (the participant slot is unknown), so only thread-message links parse."""

    if not text or "archives/" not in text:
        return None
    path = _SLACK_ARCHIVE_LINK_RE.search(text)
    if path is None:
        return None
    thread = _SLACK_THREAD_TS_RE.search(text)
    if thread is None:
        return None
    return SlackLinkTarget(channel_id=path.group(1), thread_ts=thread.group(1),
                           message_ts=_link_message_ts(path.group(2)))


def parse_branch_link_args(raw: str) -> tuple[bool, Optional[SlackLinkTarget], str, bool]:
    """``(stay_here, link, title, link_forced)`` from the text after ``/branch``.

    ``--at <link>`` is THE branch-point form: whatever follows the flag must carry a Slack
    thread permalink, and the flag holds the same branch-point slot the CLI surface proposes
    for ``--at <message_id>`` in #32105. Without the flag nothing link-shaped is interpreted —
    the text is just the title, exactly as before this feature. ``--here`` keeps its legacy
    leading-flag handling in either order, and the CLI keeps using :func:`parse_branch_args`
    unchanged."""

    stay_here, text = parse_branch_args(raw)
    forced = False
    head, _, rest = text.partition(" ")
    if head == "--at":
        forced = True
        # Tolerate ``--at --here <link>``: re-strip a leading --here so the flag order
        # never decides whether the combination is rejected.
        inner_here, inner_text = parse_branch_args(rest)
        stay_here, text = stay_here or inner_here, inner_text
    if not forced:
        return stay_here, None, text, False
    link = parse_slack_archive_link(text)
    if link is None:
        return stay_here, None, text, True
    cleaned = re.sub(r"<[^<>]*archives/[A-Z0-9]+/p\d+[^<>]*>", " ", text)  # the mrkdwn autolink…
    cleaned = re.sub(r"https?://\S+", " ", cleaned)                        # …and any bare URL left over
    # A stray second ``--at`` is a title word nobody meant; drop it.
    cleaned = " ".join(token for token in cleaned.split() if token != "--at")
    return stay_here, link, re.sub(r"\s+", " ", cleaned).strip(), True
