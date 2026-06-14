"""Time (Т‑Банк) platform adapter.

Time is a **Mattermost-based** corporate messenger (its REST API is the
Mattermost v4 API at ``/api/v4/`` and it streams events over the Mattermost
WebSocket). This adapter is therefore a thin subclass of ``MattermostAdapter``
that maps Time-specific env vars onto the Mattermost transport and carries its
own ``Platform("time")`` identity — load-bearing for the corporate security
profile (corp_safe toolset + CORP_RESTRICTED_PLATFORMS hard-deny), which is
independent of the wire transport.

Env vars:
    TIME_BOT_TOKEN       — Mattermost bot/personal-access token
    TIME_API_BASE_URL    — server base URL (e.g. https://time.tbank.ru)
    TIME_ALLOWED_USERS   — comma-separated allowed user IDs
    TIME_ALLOW_ALL_USERS — truthy = allow everyone (dev only)
    TIME_HOME_CHANNEL    — channel ID for cron/notification delivery
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, Optional

from gateway.config import Platform
from plugins.platforms.mattermost.adapter import MattermostAdapter

logger = logging.getLogger(__name__)


class TimeAdapter(MattermostAdapter):
    """Mattermost-compatible adapter for Т‑Банк's Time messenger."""

    def __init__(self, config):
        # Built-in platforms get url/token populated from a hardcoded env
        # mapping in gateway/config.py; plugin platforms don't. Seed the
        # Mattermost transport config from TIME_* env vars BEFORE the base
        # adapter reads them (config.yaml values still win).
        if config.extra is None:
            config.extra = {}
        if not config.extra.get("url"):
            config.extra["url"] = (
                os.getenv("TIME_API_BASE_URL")
                or os.getenv("MATTERMOST_URL")
                or ""
            )
        if not getattr(config, "token", None):
            config.token = os.getenv("TIME_BOT_TOKEN") or config.token

        super().__init__(config)

        # Override the Mattermost identity so Time sessions are scoped to the
        # corporate-safe security policy (corp_safe + hard-deny key on "time").
        self.platform = Platform("time")

    def _ws_handshake_headers(self):
        # Time fronts Mattermost with an auth-enforcing gateway that rejects
        # the WebSocket upgrade (HTTP 403) unless the bearer token is present
        # on the handshake itself — vanilla Mattermost only challenges after
        # the socket opens.
        return {"Authorization": f"Bearer {self._token}"}

    # ------------------------------------------------------------------
    # Inbound transport: REST long-polling (Time's proxy blocks WebSocket).
    # ------------------------------------------------------------------

    def _start_receive_loop(self) -> "asyncio.Task":
        return asyncio.create_task(self._poll_loop())

    async def _discover_channels(self) -> Dict[str, str]:
        """Return ``{channel_id: channel_type}`` for every channel the bot is in."""
        channels: Dict[str, str] = {}
        teams = await self._api_get(f"users/{self._bot_user_id}/teams")
        for team in teams if isinstance(teams, list) else []:
            tid = team.get("id")
            if not tid:
                continue
            chans = await self._api_get(
                f"users/{self._bot_user_id}/teams/{tid}/channels"
            )
            for chan in chans if isinstance(chans, list) else []:
                cid = chan.get("id")
                if cid:
                    channels[cid] = chan.get("type", "O")
        return channels

    async def _poll_loop(self) -> None:
        """Poll Mattermost REST for new posts and dispatch them like WS events.

        Time blocks the Mattermost WebSocket, so we long-poll
        ``GET /channels/{id}/posts?since=<ms>`` per channel and synthesize the
        same ``posted`` event shape the WS handler consumes — reusing all of
        its dedup, mention-gating, and dispatch logic.
        """
        interval = float(os.getenv("TIME_POLL_INTERVAL", "3"))
        # Baseline at startup so we don't replay history on first poll.
        since = int(time.time() * 1000)
        channels = await self._discover_channels()
        logger.info(
            "Time: REST polling %d channel(s) every %.0fs (WebSocket blocked by proxy)",
            len(channels), interval,
        )
        ticks = 0
        while not self._closing:
            try:
                newest = since
                for cid, ctype in list(channels.items()):
                    data = await self._api_get(f"channels/{cid}/posts?since={since}")
                    if not isinstance(data, dict):
                        continue
                    order = data.get("order", []) or []
                    posts = data.get("posts", {}) or {}
                    # ``order`` is newest-first; dispatch oldest-first.
                    for pid in reversed(order):
                        post = posts.get(pid) or {}
                        created = int(post.get("create_at", 0))
                        if created <= since:
                            continue
                        newest = max(newest, created)
                        synthetic = {
                            "event": "posted",
                            "data": {
                                "post": json.dumps(post),
                                "channel_type": ctype,
                            },
                        }
                        await self._handle_ws_event(synthetic)
                since = newest
                # Periodically re-discover channels (e.g. new DMs).
                ticks += 1
                if ticks % 10 == 0:
                    channels = await self._discover_channels()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Time: poll error: %s", exc)
            await asyncio.sleep(interval)


def _check_requirements() -> bool:
    try:
        import aiohttp  # noqa: F401  (Mattermost transport dependency)
        return True
    except Exception:
        return False


def _validate_config(cfg) -> bool:
    extra = getattr(cfg, "extra", {}) or {}
    token = os.getenv("TIME_BOT_TOKEN") or getattr(cfg, "token", None)
    base_url = (
        os.getenv("TIME_API_BASE_URL")
        or extra.get("url")
        or os.getenv("MATTERMOST_URL")
    )
    return bool(token) and bool(base_url)


def _env_enablement():
    if not os.getenv("TIME_BOT_TOKEN"):
        return None
    # Mattermost transport reads config.extra["url"]; seed it (flat dict —
    # the registry pops "home_channel" and merges the rest into extra).
    seed = {"url": os.getenv("TIME_API_BASE_URL", "")}
    home = os.getenv("TIME_HOME_CHANNEL")
    if home:
        seed["home_channel"] = {"chat_id": home}
    return seed


def register(ctx):
    """Plugin entry point."""
    ctx.register_platform(
        name="time",
        label="Time",
        adapter_factory=lambda cfg: TimeAdapter(cfg),
        check_fn=_check_requirements,
        validate_config=_validate_config,
        required_env=["TIME_BOT_TOKEN", "TIME_API_BASE_URL"],
        install_hint="No extra packages needed (uses aiohttp, a core dependency)",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="TIME_HOME_CHANNEL",
        allowed_users_env="TIME_ALLOWED_USERS",
        allow_all_env="TIME_ALLOW_ALL_USERS",
        max_message_length=16000,
        emoji="🕐",
        platform_hint=(
            "You are chatting via Time (Т‑Банк corporate messenger, a "
            "Mattermost-based platform). Keep replies professional."
        ),
    )
