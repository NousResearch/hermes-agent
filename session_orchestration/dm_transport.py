"""
session_orchestration/dm_transport.py — Discord DM transport helper.

Provides two primitives:
- ``get_dm_channel_id`` — opens (or returns existing) DM channel for a user.
- ``send_dm`` — sends a message to a user via DM.

Both accept an injectable ``http_post(url, headers, json) -> _Response`` callable
so tests never touch the network.  The default implementation uses ``urllib``
(same library as ``tools/discord_tool.py`` and ``feed.py``).

No callers are wired at this level; ``feed.py`` and ``watcher.py`` will import
``send_dm`` in level 3 tasks (criterion #6 / #13).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, NamedTuple, Optional

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"


# ---------------------------------------------------------------------------
# Lightweight response object shared by the default and injectable transports
# ---------------------------------------------------------------------------


class _Response(NamedTuple):
    """Minimal response interface for the http_post seam."""

    status_code: int
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


# ---------------------------------------------------------------------------
# Default http_post implementation (urllib — matches discord_tool.py idiom)
# ---------------------------------------------------------------------------


def _default_http_post(
    url: str,
    headers: Dict[str, str],
    json_body: Any,
) -> _Response:
    """POST *json_body* to *url* with *headers*.  Returns a ``_Response``.

    Uses ``urllib`` so no third-party deps are required, matching the pattern
    in ``tools/discord_tool.py:_discord_request``.  HTTP 4xx/5xx responses are
    returned as ``_Response`` objects (not raised) so callers can inspect the
    status code uniformly.
    """
    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return _Response(status_code=resp.status, body=body)
    except urllib.error.HTTPError as exc:
        body = b""
        try:
            body = exc.read()
        except Exception:
            pass
        return _Response(status_code=exc.code, body=body)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Type alias for the injectable POST callable.
HttpPostFn = Callable[[str, Dict[str, str], Any], _Response]


def get_dm_channel_id(
    user_id: str,
    token: str,
    *,
    http_post: Optional[HttpPostFn] = None,
) -> str:
    """Open (or retrieve) the DM channel for *user_id*.  Returns the channel id.

    POSTs to ``/api/v10/users/@me/channels`` with ``{"recipient_id": user_id}``.

    Parameters
    ----------
    user_id:
        Discord user snowflake id.
    token:
        Discord bot token (without the ``Bot `` prefix — this function adds it).
    http_post:
        Injectable HTTP callable ``(url, headers, json) -> _Response``.
        Defaults to the urllib-based ``_default_http_post``.

    Raises
    ------
    RuntimeError
        If the Discord API returns a non-2xx status.
    """
    _post = http_post if http_post is not None else _default_http_post
    url = f"{DISCORD_API_BASE}/users/@me/channels"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    resp = _post(url, headers, {"recipient_id": user_id})
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"dm_transport.get_dm_channel_id: Discord API returned {resp.status_code} "
            f"for user_id={user_id}"
        )
    data = resp.json()
    return str(data["id"])


def send_dm(
    user_id: str,
    message: str,
    token: str,
    *,
    http_post: Optional[HttpPostFn] = None,
) -> bool:
    """Send *message* to *user_id* via Discord DM.

    Calls ``get_dm_channel_id`` then POSTs to ``/channels/{channel_id}/messages``.
    Returns True on success.  Returns False (without raising) on HTTP 4xx/5xx and
    logs a warning so the caller can continue without crashing.

    Parameters
    ----------
    user_id:
        Discord user snowflake id.
    message:
        Text content to send.
    token:
        Discord bot token.
    http_post:
        Injectable HTTP callable for both the channel-open and message-post
        requests.  Defaults to the urllib-based ``_default_http_post``.
    """
    _post = http_post if http_post is not None else _default_http_post

    # Step 1 — open/resolve the DM channel.
    try:
        channel_id = get_dm_channel_id(user_id, token, http_post=_post)
    except RuntimeError as exc:
        logger.warning("dm_transport.send_dm: failed to open DM channel: %s", exc)
        return False

    # Step 2 — send the message.
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    resp = _post(url, headers, {"content": message})
    if resp.status_code < 200 or resp.status_code >= 300:
        logger.warning(
            "dm_transport.send_dm: message post failed with status=%s user_id=%s channel_id=%s",
            resp.status_code,
            user_id,
            channel_id,
        )
        return False

    logger.debug(
        "dm_transport.send_dm: sent DM to user_id=%s via channel_id=%s",
        user_id,
        channel_id,
    )
    return True
