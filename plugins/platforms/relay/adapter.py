"""
Agent Relay Platform Adapter for Hermes Agent.

A plugin-based gateway adapter that receives and sends inter-agent messages
via HTTP. Runs inside the Hermes gateway process as an aiohttp server.

Other agents (cc-connect sidecar, custom agents) POST messages to this
adapter, which routes them through the standard gateway pipeline. Outbound
messages are POSTed to the target agent's HTTP endpoint.

Configuration in config.yaml::

    platforms:
      relay:
        enabled: true
        extra:
          port: 8766
          host: "127.0.0.1"
          sidecar_url: "http://127.0.0.1:8767"

Or via environment variables:
    RELAY_PORT, RELAY_HOST, RELAY_SIDECAR_URL
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — BasePlatformAdapter lives in the main repo.
# ---------------------------------------------------------------------------

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    MessageEvent,
    MessageType,
)
from gateway.config import Platform


# ---------------------------------------------------------------------------
# Relay Adapter
# ---------------------------------------------------------------------------

class RelayAdapter(BasePlatformAdapter):
    """HTTP-based adapter for agent-to-agent relay communication.

    Listens on a local HTTP port for incoming messages from other agents,
    and sends outbound messages via HTTP POST to target agent endpoints.
    """

    def __init__(self, config, **kwargs):
        platform = Platform("relay")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self.port = int(
            os.getenv("RELAY_PORT") or extra.get("port", 8766)
        )
        self.host = os.getenv("RELAY_HOST") or extra.get("host", "127.0.0.1")
        self.sidecar_url = (
            os.getenv("RELAY_SIDECAR_URL")
            or extra.get("sidecar_url", "http://127.0.0.1:8767")
        )
        self._runner = None
        self._agent_endpoints: Dict[str, str] = {}
        self._load_endpoints(extra)

    def _load_endpoints(self, extra: dict) -> None:
        """Load agent endpoint mappings from config.

        Format in config.yaml:
            extra:
              endpoints:
                zhizhiruo: "http://127.0.0.1:8767"
                zhongwuyan: "http://127.0.0.1:8768"
        """
        endpoints = extra.get("endpoints", {})
        if isinstance(endpoints, dict):
            self._agent_endpoints.update(endpoints)
        # Default: sidecar_url for zhizhiruo
        if "zhizhiruo" not in self._agent_endpoints:
            self._agent_endpoints["zhizhiruo"] = self.sidecar_url

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the HTTP server to receive messages from other agents."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("[relay] aiohttp not installed — cannot start HTTP server")
            return False

        # Port conflict detection
        import socket as _socket
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect((self.host, self.port))
            logger.error(
                "[relay] Port %d already in use. Set a different port: "
                "platforms.relay.extra.port",
                self.port,
            )
            return False
        except (ConnectionRefusedError, OSError):
            pass  # port is free

        app = web.Application()
        app.router.add_post("/message", self._handle_incoming)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/register", self._handle_register)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self._mark_connected()

        logger.info(
            "[relay] Listening on %s:%d — endpoints: %s",
            self.host,
            self.port,
            list(self._agent_endpoints.keys()),
        )
        return True

    async def disconnect(self) -> None:
        """Stop the HTTP server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[relay] Disconnected")

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_incoming(self, request):
        """Handle incoming POST /message from external agents."""
        from aiohttp import web

        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"status": "error", "detail": "Invalid JSON"}, status=400
            )

        text = data.get("content", "")
        from_agent = data.get("from", "unknown")
        msg_id = data.get("id", f"relay_{int(time.time())}")

        if not text:
            return web.json_response(
                {"status": "error", "detail": "Empty content"}, status=400
            )

        logger.info(
            "[relay] Incoming from %s: %s", from_agent, text[:100]
        )

        # Build source — use relay:<from_agent> as chat_id
        source = self.build_source(
            chat_id=f"relay:{from_agent}",
            chat_name=f"relay/{from_agent}",
            chat_type="dm",
            user_id=from_agent,
            user_name=from_agent,
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=data,
            message_id=msg_id,
        )

        # Process through gateway pipeline (non-blocking)
        await self.handle_message(event)

        return web.json_response({"status": "ok", "message_id": msg_id})

    async def _handle_health(self, request):
        """Health check endpoint."""
        from aiohttp import web
        return web.json_response({
            "status": "ok",
            "component": "relay-adapter",
            "port": self.port,
            "endpoints": list(self._agent_endpoints.keys()),
        })

    async def _handle_register(self, request):
        """Register an agent endpoint dynamically.

        POST /register {"agent": "name", "url": "http://..."}
        """
        from aiohttp import web
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"status": "error", "detail": "Invalid JSON"}, status=400
            )

        agent = data.get("agent", "")
        url = data.get("url", "")
        if agent and url:
            self._agent_endpoints[agent] = url
            logger.info("[relay] Registered endpoint: %s → %s", agent, url)
            return web.json_response({"status": "ok"})
        return web.json_response(
            {"status": "error", "detail": "Missing agent or url"}, status=400
        )

    # ------------------------------------------------------------------
    # Send — route messages to target agent
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to an external agent via HTTP POST.

        chat_id format: "relay:<target_agent>"
        e.g. "relay:zhizhiruo" → POST to zhizhiruo's endpoint
        """
        # Extract target agent from chat_id
        if chat_id.startswith("relay:"):
            target_agent = chat_id[6:]
        else:
            target_agent = chat_id

        # Resolve endpoint
        endpoint_url = None
        if metadata and metadata.get("sidecar_url"):
            endpoint_url = metadata["sidecar_url"]
        else:
            endpoint_url = self._agent_endpoints.get(target_agent)

        if not endpoint_url:
            # No endpoint — message was processed, just can't send response back.
            # This is normal for relay: the agent processes the message and
            # the response goes to the user's current chat, not back through relay.
            logger.info(
                "[relay] No endpoint for agent '%s' — response logged (not sent back)",
                target_agent,
            )
            return SendResult(success=True, message_id=None)

        # POST message to target agent
        message_url = endpoint_url.rstrip("/") + "/message"
        payload = {
            "from": "zhangwuji",
            "to": target_agent,
            "content": content,
            "id": reply_to or f"msg_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    message_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    success = resp.status == 200
                    if not success:
                        text = await resp.text()
                        logger.warning(
                            "[relay] Send to %s failed (%d): %s",
                            target_agent,
                            resp.status,
                            text[:200],
                        )
                    return SendResult(
                        success=success,
                        message_id=payload["id"],
                    )
        except asyncio.TimeoutError:
            logger.warning("[relay] Send to %s timed out", target_agent)
            return SendResult(success=False, error="Timeout")
        except Exception as e:
            logger.error("[relay] Send to %s failed: %s", target_agent, e)
            return SendResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Typing indicators (no-op for relay)
    # ------------------------------------------------------------------

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Relay doesn't support typing indicators."""
        pass

    async def stop_typing(self, chat_id: str) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return chat info for a relay chat_id."""
        agent = chat_id.replace("relay:", "") if chat_id.startswith("relay:") else chat_id
        return {
            "name": f"relay/{agent}",
            "type": "dm",
            "chat_id": chat_id,
        }


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Check if aiohttp is available."""
    try:
        import aiohttp
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    """Validate relay platform configuration."""
    return True  # Minimal config needed


def is_connected(config) -> bool:
    """Check if relay platform is connected."""
    return config.enabled if hasattr(config, "enabled") else True


def _env_enablement():
    """Seed PlatformConfig.extra from env vars."""
    port = os.getenv("RELAY_PORT")
    host = os.getenv("RELAY_HOST")
    sidecar_url = os.getenv("RELAY_SIDECAR_URL")
    extra = {}
    if port:
        extra["port"] = int(port)
    if host:
        extra["host"] = host
    if sidecar_url:
        extra["sidecar_url"] = sidecar_url
    return extra or None


# ---------------------------------------------------------------------------
# YAML → env config bridge (apply_yaml_config_fn)
# ---------------------------------------------------------------------------


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> dict | None:
    """Translate ``config.yaml`` ``relay:`` keys into env vars.

    Implements the ``apply_yaml_config_fn`` contract (#24836 / #25443).

    The RelayAdapter reads its runtime configuration via ``os.getenv()``
    for ``RELAY_PORT``, ``RELAY_HOST``, and ``RELAY_SIDECAR_URL``.
    Rather than rewrite those call sites to read from
    ``PlatformConfig.extra``, this hook keeps the env-driven model and
    owns the YAML→env translation here, next to the adapter that
    consumes it.

    Env vars take precedence over YAML — every assignment is guarded
    by ``not os.getenv(...)`` so an explicit env var survives a
    config.yaml update.  Returns ``None`` because no extras are seeded
    into ``PlatformConfig.extra`` directly (everything flows through
    env).
    """
    if "port" in yaml_cfg and not os.getenv("RELAY_PORT"):
        os.environ["RELAY_PORT"] = str(yaml_cfg["port"])
    if "host" in yaml_cfg and not os.getenv("RELAY_HOST"):
        os.environ["RELAY_HOST"] = str(yaml_cfg["host"])
    if "sidecar_url" in yaml_cfg and not os.getenv("RELAY_SIDECAR_URL"):
        os.environ["RELAY_SIDECAR_URL"] = str(yaml_cfg["sidecar_url"])
    return None  # all settings flow through env; nothing to merge into extras


# ---------------------------------------------------------------------------
# Standalone sender (for cron jobs / out-of-process delivery)
# ---------------------------------------------------------------------------


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Send a message to an external agent via HTTP POST without a live adapter.

    Used by ``tools/send_message_tool._send_via_adapter`` and the cron
    scheduler when the gateway runner is not in this process (e.g.
    ``hermes cron`` running standalone). Without this hook,
    ``deliver=relay`` cron jobs fail with "No live adapter for platform".

    ``chat_id`` format: ``relay:<target_agent>``
    e.g. ``relay:zhizhiruo`` → POST to zhizhiruo's endpoint.

    ``thread_id`` and ``media_files`` are accepted for signature parity
    only — relay has no thread or attachment primitive.
    """
    try:
        import aiohttp as _aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    extra = getattr(pconfig, "extra", {}) or {}

    # Resolve target agent from chat_id
    if chat_id.startswith("relay:"):
        target_agent = chat_id[6:]
    else:
        target_agent = chat_id

    if not target_agent:
        return {"error": "relay standalone send: chat_id must be relay:<agent>"}

    # Resolve endpoint URL from config extra or env vars
    endpoints = extra.get("endpoints", {})
    sidecar_url = (
        extra.get("sidecar_url")
        or os.getenv("RELAY_SIDECAR_URL", "http://127.0.0.1:8767")
    )
    # Default: sidecar_url for zhizhiruo
    if isinstance(endpoints, dict):
        endpoint_url = endpoints.get(target_agent)
    else:
        endpoint_url = None
    if not endpoint_url:
        if target_agent == "zhizhiruo":
            endpoint_url = sidecar_url
        else:
            return {
                "error": (
                    f"relay standalone send: no endpoint configured for "
                    f"agent '{target_agent}'. Add it to "
                    f"platforms.relay.extra.endpoints in config.yaml "
                    f"or set RELAY_SIDECAR_URL."
                )
            }

    message_url = endpoint_url.rstrip("/") + "/message"
    msg_id = f"relay_cron_{int(time.time())}"
    payload = {
        "from": "zhangwuji",
        "to": target_agent,
        "content": message,
        "id": msg_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    try:
        async with _aiohttp.ClientSession(
            timeout=_aiohttp.ClientTimeout(total=30),
        ) as session:
            async with session.post(
                message_url, json=payload,
            ) as resp:
                success = resp.status == 200
                if not success:
                    body = await resp.text()
                    return {
                        "error": (
                            f"relay HTTP {resp.status} from "
                            f"{target_agent}: {body[:200]}"
                        )
                    }
                return {
                    "success": True,
                    "platform": "relay",
                    "chat_id": chat_id,
                    "message_id": msg_id,
                }
    except _aiohttp.ClientError as exc:
        return {"error": f"relay standalone send failed: {exc}"}
    except Exception as exc:
        return {"error": f"relay standalone send failed: {exc}"}


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="relay",
        label="Agent Relay",
        adapter_factory=lambda cfg: RelayAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="No extra packages needed (aiohttp already included)",
        env_enablement_fn=_env_enablement,
        # YAML→env config bridge — owns the translation of
        # ``config.yaml`` ``relay:`` keys (port, host, sidecar_url) into
        # ``RELAY_*`` env vars that the adapter reads via ``os.getenv()``.
        # Hook contract: #24836 / #25443.
        apply_yaml_config_fn=_apply_yaml_config,
        # Auth — relay doesn't have user auth (agent-to-agent only)
        allowed_users_env="",
        allow_all_env="RELAY_ALLOW_ALL_AGENTS",
        # Cron home-channel delivery — `deliver=relay` cron jobs
        # route to RELAY_HOME_CHANNEL when set.
        cron_deliver_env_var="RELAY_HOME_CHANNEL",
        # Out-of-process cron delivery. Without this hook, deliver=relay
        # cron jobs fail with "No live adapter" when cron runs separately
        # from the gateway.
        standalone_sender_fn=_standalone_send,
        # Display
        emoji="🔗",
        pii_safe=False,
        allow_update_command=False,
        # LLM guidance
        platform_hint=(
            "You are connected to an agent relay network. You can send "
            "messages to other agents (e.g. 周芷若) by writing to the outbox "
            "file or using the relay_send tool. Incoming messages from other "
            "agents appear as user messages in this session."
        ),
    )
