"""Configuration for the voice_call platform.

Parsed from ``gateway.platforms.voice_call.extra`` in config.yaml with
environment-variable fallbacks. Secrets (carrier credentials) live only in
``~/.hermes/.env`` — validation checks their *presence*, never logs values.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

E164_RE = re.compile(r"^\+[1-9]\d{1,14}$")

PROVIDERS = ("mock", "telnyx", "twilio", "plivo")
DEFAULT_PROVIDER = "telnyx"

SESSION_SCOPES = ("per-phone", "per-call")
INBOUND_POLICIES = ("disabled", "allowlist", "open")
OUTBOUND_MODES = ("notify", "conversation")
TUNNEL_PROVIDERS = ("none", "ngrok", "tailscale-serve", "tailscale-funnel")
REALTIME_PROVIDERS = ("openai", "gemini")
# Carriers whose media streams the realtime bridge supports.
REALTIME_CALL_PROVIDERS = ("telnyx", "twilio")

DEFAULT_SERVE_BIND = "127.0.0.1"
DEFAULT_SERVE_PORT = 3334
DEFAULT_SERVE_PATH = "/voice/webhook"
DEFAULT_STREAM_PATH = "/voice/stream"

# Env vars whose presence is required per provider (checked at validation
# time only when that provider is selected).
PROVIDER_REQUIRED_ENV = {
    "mock": [],
    "telnyx": ["TELNYX_API_KEY", "TELNYX_CONNECTION_ID"],
    "twilio": ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"],
    "plivo": ["PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN"],
}


def normalize_e164(number: str) -> str:
    """Normalize a phone number for comparison: strip separators, keep +digits."""
    cleaned = re.sub(r"[\s\-().]", "", str(number or ""))
    if cleaned and not cleaned.startswith("+") and cleaned.isdigit():
        cleaned = f"+{cleaned}"
    return cleaned


def is_e164(number: str) -> bool:
    return bool(E164_RE.match(normalize_e164(number)))


def _env_list(name: str) -> List[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ServeConfig:
    bind: str = DEFAULT_SERVE_BIND
    port: int = DEFAULT_SERVE_PORT
    path: str = DEFAULT_SERVE_PATH
    stream_path: str = DEFAULT_STREAM_PATH


@dataclass
class TunnelConfig:
    provider: str = "none"
    ngrok_domain: Optional[str] = None


@dataclass
class OutboundConfig:
    default_mode: str = "notify"
    notify_hangup_delay_s: int = 3


@dataclass
class TimeoutsConfig:
    max_call_s: int = 600
    ring_s: int = 45
    silence_s: int = 30
    transcript_wait_s: int = 60


@dataclass
class ResponderConfig:
    thinking_phrase: str = "One moment."
    response_timeout_s: int = 60


@dataclass
class SecurityConfig:
    skip_signature_verification: bool = False
    max_body_bytes: int = 1_000_000
    body_read_timeout_s: int = 30
    max_inflight_per_ip: int = 8
    replay_ttl_s: int = 600


@dataclass
class StreamingConfig:
    enabled: bool = False


@dataclass
class RealtimeConfig:
    enabled: bool = False
    provider: str = "openai"
    model: Optional[str] = None
    voice: Optional[str] = None
    instructions: Optional[str] = None


@dataclass
class VoiceCallConfig:
    provider: str = DEFAULT_PROVIDER
    from_number: Optional[str] = None
    session_scope: str = "per-phone"
    inbound_policy: str = "allowlist"
    allow_from: List[str] = field(default_factory=list)
    inbound_greeting: str = "Hello, this is Hermes. How can I help?"
    serve: ServeConfig = field(default_factory=ServeConfig)
    public_url: Optional[str] = None
    tunnel: TunnelConfig = field(default_factory=TunnelConfig)
    outbound: OutboundConfig = field(default_factory=OutboundConfig)
    timeouts: TimeoutsConfig = field(default_factory=TimeoutsConfig)
    responder: ResponderConfig = field(default_factory=ResponderConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    # Raw provider-specific sub-dicts from extra (e.g. extra["telnyx"]).
    # Providers read credentials env-first, then from here.
    provider_extra: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # -- parsing --------------------------------------------------------------

    @classmethod
    def from_extra(cls, extra: Optional[Dict[str, Any]]) -> "VoiceCallConfig":
        """Build config from ``PlatformConfig.extra`` with env fallbacks."""
        extra = dict(extra or {})

        serve_raw = extra.get("serve") or {}
        tunnel_raw = extra.get("tunnel") or {}
        outbound_raw = extra.get("outbound") or {}
        timeouts_raw = extra.get("timeouts") or {}
        responder_raw = extra.get("responder") or {}
        security_raw = extra.get("security") or {}
        streaming_raw = extra.get("streaming") or {}
        realtime_raw = extra.get("realtime") or {}

        allow_from = extra.get("allow_from")
        if allow_from is None:
            allow_from = _env_list("VOICE_CALL_ALLOWED_NUMBERS")
        if isinstance(allow_from, str):
            allow_from = [n.strip() for n in allow_from.split(",") if n.strip()]

        provider = str(
            extra.get("provider") or os.getenv("VOICE_CALL_PROVIDER") or DEFAULT_PROVIDER
        ).strip().lower()

        return cls(
            provider=provider,
            from_number=(
                extra.get("from_number")
                or os.getenv("VOICE_CALL_FROM_NUMBER")
                or None
            ),
            session_scope=str(extra.get("session_scope", "per-phone")).strip().lower(),
            inbound_policy=str(extra.get("inbound_policy", "allowlist")).strip().lower(),
            allow_from=[normalize_e164(n) for n in allow_from],
            inbound_greeting=str(
                extra.get("inbound_greeting", cls.inbound_greeting)
            ),
            serve=ServeConfig(
                bind=str(serve_raw.get("bind", DEFAULT_SERVE_BIND)),
                port=_as_int(serve_raw.get("port"), DEFAULT_SERVE_PORT),
                path=str(serve_raw.get("path", DEFAULT_SERVE_PATH)),
                stream_path=str(serve_raw.get("stream_path", DEFAULT_STREAM_PATH)),
            ),
            public_url=(extra.get("public_url") or os.getenv("VOICE_CALL_PUBLIC_URL") or None),
            tunnel=TunnelConfig(
                provider=str(tunnel_raw.get("provider", "none")).strip().lower(),
                ngrok_domain=tunnel_raw.get("ngrok_domain") or os.getenv("NGROK_DOMAIN") or None,
            ),
            outbound=OutboundConfig(
                default_mode=str(outbound_raw.get("default_mode", "notify")).strip().lower(),
                notify_hangup_delay_s=_as_int(outbound_raw.get("notify_hangup_delay_s"), 3),
            ),
            timeouts=TimeoutsConfig(
                max_call_s=_as_int(timeouts_raw.get("max_call_s"), 600),
                ring_s=_as_int(timeouts_raw.get("ring_s"), 45),
                silence_s=_as_int(timeouts_raw.get("silence_s"), 30),
                transcript_wait_s=_as_int(timeouts_raw.get("transcript_wait_s"), 60),
            ),
            responder=ResponderConfig(
                thinking_phrase=str(responder_raw.get("thinking_phrase", "One moment.")),
                response_timeout_s=_as_int(responder_raw.get("response_timeout_s"), 60),
            ),
            security=SecurityConfig(
                skip_signature_verification=_as_bool(
                    security_raw.get("skip_signature_verification"), False
                ),
                max_body_bytes=_as_int(security_raw.get("max_body_bytes"), 1_000_000),
                body_read_timeout_s=_as_int(security_raw.get("body_read_timeout_s"), 30),
                max_inflight_per_ip=_as_int(security_raw.get("max_inflight_per_ip"), 8),
                replay_ttl_s=_as_int(security_raw.get("replay_ttl_s"), 600),
            ),
            streaming=StreamingConfig(enabled=_as_bool(streaming_raw.get("enabled"), False)),
            realtime=RealtimeConfig(
                enabled=_as_bool(realtime_raw.get("enabled"), False),
                provider=str(realtime_raw.get("provider", "openai")).strip().lower(),
                model=realtime_raw.get("model") or None,
                voice=realtime_raw.get("voice") or None,
                instructions=realtime_raw.get("instructions") or None,
            ),
            provider_extra={
                name: dict(extra.get(name) or {})
                for name in PROVIDERS
                if isinstance(extra.get(name), dict)
            },
        )

    @classmethod
    def from_platform_config(cls, config: Any) -> "VoiceCallConfig":
        return cls.from_extra(getattr(config, "extra", None) or {})

    # -- validation -----------------------------------------------------------

    @property
    def requires_public_webhook(self) -> bool:
        return self.provider != "mock"

    def provider_credential(self, key: str, env_var: str) -> Optional[str]:
        """Resolve a provider credential env-first, then from provider_extra."""
        value = os.getenv(env_var, "").strip()
        if value:
            return value
        sub = self.provider_extra.get(self.provider, {})
        value = str(sub.get(key, "") or "").strip()
        return value or None

    def validate(self) -> List[str]:
        """Return a list of human-readable config errors (empty = valid)."""
        errors: List[str] = []

        if self.provider not in PROVIDERS:
            errors.append(
                f"provider must be one of {', '.join(PROVIDERS)} (got {self.provider!r})"
            )
            return errors  # everything else is provider-relative

        if self.session_scope not in SESSION_SCOPES:
            errors.append(
                f"session_scope must be one of {', '.join(SESSION_SCOPES)}"
            )
        if self.inbound_policy not in INBOUND_POLICIES:
            errors.append(
                f"inbound_policy must be one of {', '.join(INBOUND_POLICIES)}"
            )
        if self.outbound.default_mode not in OUTBOUND_MODES:
            errors.append(
                f"outbound.default_mode must be one of {', '.join(OUTBOUND_MODES)}"
            )
        if self.tunnel.provider not in TUNNEL_PROVIDERS:
            errors.append(
                f"tunnel.provider must be one of {', '.join(TUNNEL_PROVIDERS)}"
            )

        if self.from_number and not is_e164(self.from_number):
            errors.append("from_number must be E.164 (+15555550000 style)")
        for number in self.allow_from:
            if not is_e164(number):
                errors.append(f"allow_from entry {number!r} is not E.164")

        if not (1 <= self.serve.port <= 65535):
            errors.append("serve.port must be 1-65535")
        if not self.serve.path.startswith("/"):
            errors.append("serve.path must start with /")
        # Note: inbound_policy "allowlist" with an empty allow_from is valid
        # config for outbound-only setups — it simply rejects all inbound
        # calls (the webhook logs a warning when that happens).

        # Provider credentials (presence only — values are never logged).
        missing = [
            env
            for env in PROVIDER_REQUIRED_ENV[self.provider]
            if not self.provider_credential(env.split("_", 1)[1].lower(), env)
        ]
        if missing:
            errors.append(
                f"provider '{self.provider}' requires env vars: {', '.join(missing)}"
            )

        if (
            self.provider == "telnyx"
            and not self.security.skip_signature_verification
            and not self.provider_credential("public_key", "TELNYX_PUBLIC_KEY")
        ):
            errors.append(
                "telnyx requires TELNYX_PUBLIC_KEY for webhook signature "
                "verification (or security.skip_signature_verification: true — "
                "not recommended)"
            )

        if (
            self.requires_public_webhook
            and not self.public_url
            and self.tunnel.provider == "none"
        ):
            errors.append(
                f"provider '{self.provider}' needs a publicly reachable webhook: "
                "set public_url or tunnel.provider (ngrok / tailscale-serve / "
                "tailscale-funnel)"
            )

        if self.realtime.enabled:
            if self.realtime.provider not in REALTIME_PROVIDERS:
                errors.append(
                    f"realtime.provider must be one of {', '.join(REALTIME_PROVIDERS)}"
                )
            if self.provider not in REALTIME_CALL_PROVIDERS:
                errors.append(
                    "realtime.enabled requires provider telnyx or twilio "
                    "(media streams are not supported on "
                    f"{self.provider!r})"
                )

        return errors
