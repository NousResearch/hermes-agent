"""
Social relay platform adapter.

Connects to an Ed25519-signed social relay and polls for notifications
(mentions, replies). When another agent mentions or replies to this agent,
it triggers the message handler so the agent can respond.

Config (config.yaml):
    platforms:
      social:
        enabled: true

    social:
      relay: "https://agentnet-relay.0xbyt4.workers.dev"
      poll_interval: 30  # seconds between notification polls
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from identity import (
    get_identity,
    identity_exists,
    create_post_event,
    create_profile_event,
)

logger = logging.getLogger(__name__)


def _load_social_relay_config() -> Dict[str, Any]:
    """Load social relay config from config.yaml."""
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    config_path = hermes_home / "config.yaml"

    defaults = {"relay": "", "poll_interval": 30}

    if not config_path.is_file():
        return defaults

    try:
        import yaml
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        social = config.get("social", {})
        if isinstance(social, dict):
            return {**defaults, **social}
    except Exception as e:
        logger.warning("Failed to load social config: %s", e)

    return defaults


def check_social_adapter_requirements() -> bool:
    """Check if social adapter can run."""
    config = _load_social_relay_config()
    return bool(config.get("relay")) and config.get("enabled", False)


class SocialAdapter(BasePlatformAdapter):
    """Platform adapter for Ed25519-signed social relays."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SOCIAL)
        social_config = _load_social_relay_config()
        self._relay_url = social_config.get("relay", "").rstrip("/")
        self._poll_interval = social_config.get("poll_interval", 30)
        self._poll_task: Optional[asyncio.Task] = None
        self._last_seen_at: int = int(time.time())
        self._identity = None
        self._client = httpx.AsyncClient(timeout=15)
        self._seen_event_ids: set = set()  # dedup notifications

    async def connect(self) -> bool:
        """Connect: verify relay is reachable, initialize identity, start polling."""
        if not self._relay_url:
            logger.error("Social adapter: no relay URL configured")
            self._set_fatal_error("NO_RELAY", "Social relay URL not configured", retryable=False)
            await self._notify_fatal_error()
            return False

        if not identity_exists():
            logger.info("Social adapter: creating agent identity...")

        try:
            self._identity = get_identity()
        except Exception as e:
            logger.error("Social adapter: failed to load identity: %s", e)
            self._set_fatal_error("IDENTITY_ERROR", str(e), retryable=False)
            await self._notify_fatal_error()
            return False

        # Verify relay is reachable
        try:
            r = await self._client.get(f"{self._relay_url}/api/health")
            health = r.json()
            if not health.get("ok"):
                raise ValueError("Relay health check failed")
            logger.info(
                "Social adapter: connected to %s (relay: %s)",
                self._relay_url,
                health.get("data", {}).get("relay", "unknown"),
            )
        except Exception as e:
            logger.error("Social adapter: relay unreachable at %s: %s", self._relay_url, e)
            self._set_fatal_error("RELAY_UNREACHABLE", f"Cannot reach relay: {e}", retryable=True)
            await self._notify_fatal_error()
            return False

        # Publish/update agent profile
        await self._publish_profile()

        self._mark_connected()
        self._last_seen_at = int(time.time())

        # Start notification polling
        self._poll_task = asyncio.create_task(self._poll_notifications())
        return True

    async def disconnect(self) -> None:
        """Stop polling and clean up."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        await self._client.aclose()
        self._mark_disconnected()
        logger.info("Social adapter: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a reply as a signed post on the relay."""
        if not self._identity:
            return SendResult(success=False, error="No identity loaded")

        # Secret filter: prevent leaking API keys, tokens, etc.
        from tools.social_tools import _check_outgoing_content
        secret_check = _check_outgoing_content(content)
        if secret_check:
            logger.warning("Social adapter: blocked outgoing content with secret: %s", secret_check)
            return SendResult(success=False, error=secret_check)

        try:
            tags = []
            mentions = None

            if reply_to:
                tags.append(["e", reply_to])
                # chat_id is the mentioning agent's pubkey
                if chat_id and chat_id != self._identity.pubkey_hex:
                    mentions = [chat_id]

            event = create_post_event(
                self._identity,
                content,
                mentions=mentions,
                reply_to=reply_to,
            )

            r = await self._client.post(
                f"{self._relay_url}/api/events",
                json=event,
            )
            result = r.json()

            if result.get("ok"):
                return SendResult(success=True, message_id=event["id"])
            else:
                return SendResult(success=False, error=result.get("error", "Unknown error"))

        except Exception as e:
            logger.error("Social adapter: send failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get agent info from the relay."""
        try:
            r = await self._client.get(f"{self._relay_url}/api/agents/{chat_id}")
            data = r.json()
            if data.get("ok"):
                agent = data["data"]
                return {
                    "id": chat_id,
                    "name": agent.get("display_name") or chat_id[:12] + "...",
                    "type": "agent",
                }
        except Exception:
            pass

        return {"id": chat_id, "name": chat_id[:12] + "...", "type": "agent"}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _publish_profile(self) -> None:
        """Publish or update agent profile on the relay."""
        if not self._identity:
            return

        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        config_path = hermes_home / "config.yaml"

        display_name = "Hermes Agent"
        bio = ""
        model = ""
        hermes_version = ""

        try:
            import yaml
            if config_path.is_file():
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                social = config.get("social", {})
                profile = social.get("profile", {})
                display_name = profile.get("display_name", display_name)
                bio = profile.get("bio", bio)
                model = config.get("model", model)

                # Try to get hermes version
                try:
                    import importlib.metadata
                    hermes_version = importlib.metadata.version("hermes-agent")
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Social adapter: could not load profile config: %s", e)

        # Auto-detect Tempo wallet address
        tempo_address = ""
        try:
            import subprocess, shutil
            tempo_bin = shutil.which("tempo") or os.path.expanduser("~/.tempo/bin/tempo")
            if os.path.isfile(tempo_bin):
                r = subprocess.run(
                    [tempo_bin, "wallet", "-t", "whoami"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0:
                    for line in r.stdout.split("\n"):
                        if "wallet:" in line.lower() and "0x" in line:
                            addr = line.split('"')[1] if '"' in line else line.split()[-1]
                            if re.match(r'^0x[0-9a-fA-F]{40}$', addr):
                                tempo_address = addr
                            break
        except Exception:
            pass

        event = create_profile_event(
            self._identity,
            display_name=display_name,
            bio=bio,
            model=model,
            hermes_version=hermes_version,
            tempo_address=tempo_address,
        )

        try:
            r = await self._client.post(f"{self._relay_url}/api/events", json=event)
            result = r.json()
            if result.get("ok"):
                logger.info("Social adapter: profile published (%s)", display_name)
            else:
                logger.warning("Social adapter: profile publish failed: %s", result.get("error"))
        except Exception as e:
            logger.warning("Social adapter: profile publish error: %s", e)

    async def _poll_notifications(self) -> None:
        """Poll relay for new mentions/replies and dispatch as messages."""
        logger.info(
            "Social adapter: polling notifications every %ds",
            self._poll_interval,
        )

        consecutive_errors = 0

        while True:
            try:
                sleep_time = self._poll_interval
                if consecutive_errors >= 10:
                    sleep_time = 60
                await asyncio.sleep(sleep_time)

                if not self._identity:
                    continue

                r = await self._client.get(
                    f"{self._relay_url}/api/notifications/{self._identity.pubkey_hex}",
                    params={"since": str(self._last_seen_at), "limit": "20"},
                )
                data = r.json()

                if not data.get("ok"):
                    consecutive_errors += 1
                    if consecutive_errors == 5:
                        logger.warning(
                            "Social adapter: %d consecutive poll failures",
                            consecutive_errors,
                        )
                    continue

                consecutive_errors = 0
                notifications = data.get("data", [])
                if not notifications:
                    continue

                for notif in notifications:
                    event = notif.get("event", {})
                    notif_type = notif.get("type", "")

                    # Only process mentions and replies
                    if notif_type not in ("mention", "reply"):
                        continue

                    # Skip own events
                    if event.get("pubkey") == self._identity.pubkey_hex:
                        continue

                    # Dedup: skip already-processed notifications
                    event_id = event.get("id", "")
                    if event_id in self._seen_event_ids:
                        continue
                    self._seen_event_ids.add(event_id)
                    # Prune to prevent memory leak — keep most recent 500
                    # (evict oldest half instead of clearing all to avoid
                    # reprocessing events during the next poll cycle)
                    if len(self._seen_event_ids) > 1000:
                        # Convert to list, keep last 500 by insertion order (Python 3.7+ sets are not ordered,
                        # but we only need rough recency — the _last_seen_at timestamp prevents true duplicates)
                        excess = len(self._seen_event_ids) - 500
                        it = iter(self._seen_event_ids)
                        to_remove = [next(it) for _ in range(excess)]
                        self._seen_event_ids -= set(to_remove)

                    # Verify event signature
                    sig_valid = False
                    try:
                        from identity.events import compute_event_id
                        from nacl.signing import VerifyKey
                        expected_id = compute_event_id(
                            event.get("pubkey", ""), event.get("created_at", 0),
                            event.get("kind", 0), event.get("tags", []), event.get("content", "")
                        )
                        if expected_id == event.get("id"):
                            vk = VerifyKey(bytes.fromhex(event["pubkey"]))
                            vk.verify(bytes.fromhex(event["id"]), bytes.fromhex(event.get("sig", "")))
                            sig_valid = True
                    except Exception:
                        pass

                    # Sanitize content
                    from tools.social_tools import _sanitize_relay_content
                    source_pubkey = event.get("pubkey", "")
                    raw_content = event.get("content", "")
                    if sig_valid:
                        content_text = _sanitize_relay_content(raw_content)
                    else:
                        content_text = f"[UNVERIFIED - INVALID SIGNATURE] {_sanitize_relay_content(raw_content)}"
                    msg = MessageEvent(
                        text=content_text,
                        message_type=MessageType.TEXT,
                        source=self._build_source(source_pubkey, event.get("id", "")),
                        raw_message=event,
                        message_id=event.get("id", ""),
                        timestamp=event.get("created_at"),
                    )

                    # Track the latest timestamp
                    ts = event.get("created_at", 0)
                    if ts > self._last_seen_at:
                        self._last_seen_at = ts

                    # Dispatch to message handler
                    if self._message_handler:
                        try:
                            await self._message_handler(msg)
                        except Exception as e:
                            logger.error("Social adapter: handler error: %s", e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors == 5:
                    logger.warning(
                        "Social adapter: %d consecutive poll errors: %s",
                        consecutive_errors, e,
                    )
                else:
                    logger.error("Social adapter: poll error: %s", e)
                backoff = 60 if consecutive_errors >= 10 else 5
                await asyncio.sleep(backoff)

    def _build_source(self, pubkey: str, event_id: str) -> Dict[str, Any]:
        """Build session source dict for a social notification."""
        from gateway.session import SessionSource

        return SessionSource(
            platform="social",
            chat_id=pubkey,
            user_id=pubkey,
            display_name=pubkey[:12] + "...",
            is_dm=True,
            reply_to_message_id=event_id,
        )
