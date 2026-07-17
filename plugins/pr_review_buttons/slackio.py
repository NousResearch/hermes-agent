"""Minimal Slack Web API client for the pr_review_buttons plugin.

Both entry points that post to Slack — the ``hermes prreview publish`` CLI and
the in-gateway button-click handler — go through here. Deliberately built on
the Python standard library (``urllib``) rather than ``slack_sdk``: the cron
spawns the *installed* ``hermes`` binary, whose interpreter does not carry
``slack_sdk``, so a hard dependency on it made ``publish`` fail with
``No module named 'slack_sdk'``. Slack's Web API is a plain HTTPS POST with a
Bearer token, so stdlib is enough and works in every environment.

``hermes`` loads ``~/.hermes/.env`` at import time, so ``SLACK_BOT_TOKEN`` is
already in the environment. The client exposes the two async methods the rest
of the plugin uses — ``chat_postMessage`` and ``chat_update`` — matching the
slack_sdk surface so callers and tests are unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

_SLACK_API = "https://slack.com/api"


def bot_token() -> Optional[str]:
    """First configured Slack bot token (comma-separated list supported)."""
    raw = os.environ.get("SLACK_BOT_TOKEN", "")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    return tokens[0] if tokens else None


def home_channel() -> Optional[str]:
    """The configured Slack home channel id, if any."""
    return os.environ.get("SLACK_HOME_CHANNEL") or None


class SlackError(RuntimeError):
    """A Slack Web API call returned ``ok: false`` or failed at the transport."""


class _StdlibSlackClient:
    """Tiny async Slack Web API client backed by ``urllib`` in a worker thread.

    Only the two methods the plugin needs are implemented. Each returns the
    parsed Slack response dict and raises :class:`SlackError` on ``ok: false``
    or a transport failure, so callers can rely on a raised exception the same
    way they would with slack_sdk's ``SlackApiError``.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def _call_sync(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{_SLACK_API}/{method}",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise SlackError(f"{method}: transport error: {exc}") from exc
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            raise SlackError(f"{method}: non-JSON response: {body[:200]}") from exc
        if not parsed.get("ok"):
            raise SlackError(f"{method}: {parsed.get('error', 'unknown error')}")
        return parsed

    async def _call(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Run the blocking HTTP call off the event loop so the in-gateway click
        # handler never stalls the loop.
        return await asyncio.to_thread(self._call_sync, method, payload)

    async def chat_postMessage(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._call("chat.postMessage", kwargs)

    async def chat_update(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._call("chat.update", kwargs)


def make_client(token: Optional[str] = None) -> _StdlibSlackClient:
    """Build a stdlib-backed Slack client. Raises if the token is missing."""
    tok = token or bot_token()
    if not tok:
        raise RuntimeError("SLACK_BOT_TOKEN is not set")
    return _StdlibSlackClient(tok)


async def post_message(
    client: Any,
    channel: str,
    text: str,
    blocks: List[dict],
    thread_ts: Optional[str] = None,
) -> str:
    """Post a Block Kit message; return its ts. ``text`` is the a11y fallback."""
    kwargs = {"channel": channel, "text": text, "blocks": blocks}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    resp = await client.chat_postMessage(**kwargs)
    return resp.get("ts", "")


async def update_message(
    client: Any,
    channel: str,
    ts: str,
    text: str,
    blocks: List[dict],
) -> None:
    """Replace an existing message's blocks (used to strip buttons post-click)."""
    await client.chat_update(channel=channel, ts=ts, text=text, blocks=blocks)
