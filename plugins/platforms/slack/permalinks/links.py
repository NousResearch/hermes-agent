"""Parse pasted Slack message permalinks and neutralize linked text."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

MAX_PERMALINKS = 3

#: Strict on purpose: canonical workspace host, typed conversation id, and the 16-digit
#: packed ts as named groups. The adapter's loose ``_SLACK_PERMALINK_RE`` canonicalizes
#: text; this one is a security boundary.
_RESOLVER_PERMALINK_RE = re.compile(
    r"https://(?P<workspace>[A-Za-z0-9-]+)\.slack\.com/archives/"
    r"(?P<channel>[CGD][A-Z0-9]{8,})/p(?P<packed_ts>\d{16})"
    r"(?:\?[^\s>|]*)?(?=$|[\s>|)])",
    re.IGNORECASE,
)
_TS_RE = re.compile(r"\d+\.\d{6}")


@dataclass(frozen=True)
class PermalinkTarget:
    """A message target parsed from a Slack ``/archives/.../p...`` URL."""

    channel_id: str
    message_ts: str
    thread_ts: Optional[str] = None

    @property
    def label(self) -> str:
        """Canonical provenance; never echoes attacker-controlled URL text."""
        return f"channel={self.channel_id} ts={self.message_ts}"

    @property
    def root_ts(self) -> str:
        return self.thread_ts or self.message_ts


def parse_permalinks(text: str, *, limit: int = MAX_PERMALINKS) -> list[PermalinkTarget]:
    """Bounded, de-duplicated Slack message permalinks in first-seen order.

    Slack renders links as ``<url|label>`` and escapes ``&`` as ``&amp;``. The path ts drops
    the decimal point of Slack's ``seconds.microseconds`` id, so it is restored here.
    """
    hard_limit = min(max(int(limit), 0), MAX_PERMALINKS)
    targets: list[PermalinkTarget] = []
    seen: set[tuple[str, str]] = set()
    for match in _RESOLVER_PERMALINK_RE.finditer(html.unescape(text or "")):
        if len(targets) >= hard_limit:
            break
        packed = match.group("packed_ts")
        message_ts = f"{packed[:-6]}.{packed[-6:]}"
        channel_id = match.group("channel").upper()
        if (channel_id, message_ts) in seen:
            continue
        seen.add((channel_id, message_ts))
        thread_ts = (parse_qs(urlsplit(match.group(0)).query).get("thread_ts") or [None])[0]
        if thread_ts and not _TS_RE.fullmatch(thread_ts):
            thread_ts = None
        targets.append(PermalinkTarget(channel_id, message_ts, thread_ts))
    return targets


def sanitize_context_text(value: Any) -> str:
    """Prevent linked content from composing or forging framing markers."""
    return (
        str(value or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("[", "［")
        .replace("]", "］")
    )


def unreadable(target: PermalinkTarget, reason: str) -> str:
    """The one-line note attached instead of a link's content."""
    return (
        f"[Unable to read linked Slack message {target.label}: "
        f"{sanitize_context_text(reason)}]"
    )


def payload(response: Any) -> dict:
    """A Slack response (or ``SlackApiError.response``) as a plain dict."""
    data = response if isinstance(response, dict) else getattr(response, "data", None)
    return data if isinstance(data, dict) else {}
