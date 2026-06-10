"""Voice-call platform adapter (Hermes plugin).

Makes and receives real phone calls through carrier APIs — Telnyx
(default), Twilio, Plivo, or a credential-free mock provider. Unlike thin
messaging adapters, this adapter owns gateway-lifetime infrastructure: an
aiohttp webhook server for carrier callbacks, an optional tunnel for public
exposure, a call state machine with JSONL persistence, and (optionally) a
realtime media-stream bridge to speech-to-speech models.

The adapter itself stays thin: ``connect()``/``disconnect()`` delegate to
the runtime singleton in ``runtime.py``, and ``send()`` maps gateway replies
onto live calls. All heavy lifting lives in the runtime's components so the
``voice_call`` tool, CLI, and slash command can drive calls without an
adapter reference.

Configuration in config.yaml::

    gateway:
      platforms:
        voice_call:
          enabled: true
          extra:
            provider: telnyx          # mock | telnyx | twilio | plivo
            from_number: "+15555550000"
            inbound_policy: allowlist # disabled | allowlist | open
            allow_from: ["+15555550001"]
            serve: { bind: "127.0.0.1", port: 3334, path: "/voice/webhook" }
            tunnel: { provider: ngrok }   # or public_url: https://...

Carrier credentials live in ``~/.hermes/.env`` (TELNYX_API_KEY,
TELNYX_CONNECTION_ID, TELNYX_PUBLIC_KEY, TWILIO_*, PLIVO_*). See
``config.py`` for the full schema and validation rules.
"""

import logging
import os
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

from .config import VoiceCallConfig

logger = logging.getLogger(__name__)


def _aiohttp_available(lazy_install: bool = False) -> bool:
    """Check for aiohttp, optionally lazy-installing it (discord pattern)."""
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        pass
    if not lazy_install:
        return False
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("platform.voice_call", prompt=False)
        import aiohttp  # noqa: F401
        return True
    except Exception:
        return False


def check_requirements() -> bool:
    """Plugin gate: the webhook server needs aiohttp (lazy-installable)."""
    return _aiohttp_available(lazy_install=False)


def validate_config(config) -> bool:
    """Is the platform fully configured for its selected provider?"""
    try:
        cfg = VoiceCallConfig.from_platform_config(config)
        return not cfg.validate()
    except Exception:
        logger.debug("voice_call validate_config failed", exc_info=True)
        return False


def is_connected(config) -> bool:
    """Surface in ``hermes status`` before the adapter is instantiated."""
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Unlike pure-messaging platforms, voice_call binds a local port and may
    spawn a tunnel at startup, so mere credential presence must not
    auto-enable it. Users opt in explicitly with ``VOICE_CALL_ENABLED=true``.
    """
    if os.getenv("VOICE_CALL_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        return None
    seed: Dict[str, Any] = {}
    provider = os.getenv("VOICE_CALL_PROVIDER", "").strip().lower()
    if provider:
        seed["provider"] = provider
    from_number = os.getenv("VOICE_CALL_FROM_NUMBER", "").strip()
    if from_number:
        seed["from_number"] = from_number
    allowed = os.getenv("VOICE_CALL_ALLOWED_NUMBERS", "").strip()
    if allowed:
        seed["allow_from"] = [n.strip() for n in allowed.split(",") if n.strip()]
    public_url = os.getenv("VOICE_CALL_PUBLIC_URL", "").strip()
    if public_url:
        seed["public_url"] = public_url
    home = os.getenv("VOICE_CALL_HOME_NUMBER", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": "Voice home"}
    return seed


class VoiceCallAdapter(BasePlatformAdapter):
    """Gateway-lifetime shell around the voice-call runtime singleton."""

    def __init__(self, config: PlatformConfig):
        platform = Platform("voice_call")
        super().__init__(config=config, platform=platform)
        # Parsed lazily in connect(): __init__ must stay side-effect-free
        # and tolerate synthetic configs (platform interface tests construct
        # adapters with MagicMock configs).
        self._vc_config: Optional[VoiceCallConfig] = None

    @property
    def enforces_own_access_policy(self) -> bool:
        """Inbound access is gated by the webhook's inbound_policy/allowlist.

        Without this, the gateway's env-allowlist default-deny would
        silently drop callers that the configured allowlist already accepted.
        """
        return True

    # -- Connection lifecycle -------------------------------------------------

    async def connect(self) -> bool:
        """Validate config and boot the voice-call runtime."""
        if not _aiohttp_available(lazy_install=True):
            logger.warning(
                "[%s] aiohttp not installed and lazy install failed. "
                "Run: pip install aiohttp",
                self.name,
            )
            return False

        try:
            cfg = VoiceCallConfig.from_platform_config(self.config)
        except Exception as e:
            logger.error("[%s] Failed to parse config: %s", self.name, e)
            return False

        errors = cfg.validate()
        if errors:
            for err in errors:
                logger.warning("[%s] Config error: %s", self.name, err)
            self._set_fatal_error(
                "voice_call_config",
                "voice_call config invalid: " + "; ".join(errors),
                retryable=False,
            )
            return False

        self._vc_config = cfg
        try:
            from .runtime import ensure_runtime

            await ensure_runtime(cfg, adapter=self)
        except OSError as e:
            # Port bind / tunnel network failures are environmental and may
            # clear up — let the gateway's reconnect watcher retry.
            logger.error("[%s] Runtime start failed: %s", self.name, e)
            self._set_fatal_error(
                "voice_call_bind",
                f"voice_call runtime failed to start: {e}",
                retryable=True,
            )
            return False
        except Exception as e:
            logger.error("[%s] Runtime start failed: %s", self.name, e, exc_info=True)
            return False

        self._mark_connected()
        logger.info(
            "[%s] Connected — provider=%s, webhook %s:%s%s",
            self.name, cfg.provider, cfg.serve.bind, cfg.serve.port, cfg.serve.path,
        )
        return True

    async def disconnect(self) -> None:
        """Stop the runtime (idempotent; tolerates partial initialization)."""
        self._running = False
        self._mark_disconnected()
        try:
            from .runtime import stop_runtime

            await stop_runtime()
        except Exception as e:
            logger.warning("[%s] Error stopping runtime: %s", self.name, e)
        logger.info("[%s] Disconnected", self.name)

    # -- Outbound (gateway → caller) -------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Speak an agent reply on the live call associated with ``chat_id``.

        ``chat_id`` is the remote party's E.164 number; with per-call session
        scope the call id travels as the source ``thread_id`` and arrives in
        ``metadata``.
        """
        from .runtime import get_runtime

        runtime = get_runtime()
        if runtime is None:
            return SendResult(success=False, error="voice_call runtime not running")
        thread_id = (metadata or {}).get("thread_id")
        ok, detail = await runtime.speak_for_chat(chat_id, content, thread_id=thread_id)
        if ok:
            return SendResult(success=True, message_id=detail)
        return SendResult(success=False, error=detail)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """No typing indicator on a phone call."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """A chat is one remote phone number; calls are always 1:1."""
        return {"name": chat_id, "type": "dm"}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="voice_call",
        label="Voice Calls",
        adapter_factory=lambda cfg: VoiceCallAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="pip install aiohttp",
        env_enablement_fn=_env_enablement,
        # Cron delivery: `deliver=voice_call` jobs call VOICE_CALL_HOME_NUMBER.
        cron_deliver_env_var="VOICE_CALL_HOME_NUMBER",
        # The gateway-side allowlist seed; the webhook's inbound_policy is the
        # actual enforcement point (enforces_own_access_policy = True).
        allowed_users_env="VOICE_CALL_ALLOWED_NUMBERS",
        allow_all_env="VOICE_CALL_ALLOW_ALL",
        max_message_length=1000,
        emoji="📞",
        # Phone numbers are PII — keep them out of session descriptions.
        pii_safe=True,
        platform_hint=(
            "You are speaking on a live phone call. Reply in one to three "
            "short, natural, spoken-style sentences of plain text — no "
            "markdown, URLs, code, emoji, or lists. Never read secrets, "
            "tokens, or credentials aloud."
        ),
    )
