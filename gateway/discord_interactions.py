"""Local Discord interaction route helpers for the API server.

This module is intentionally small and fail-closed. It only builds the
server-side shape needed to receive Discord interaction callbacks safely:
read raw bytes, verify timestamp/signature headers first, reject replayed
requests, then build a dry-run ACK preview. It does not register a Discord
endpoint, send network messages, or apply approval decisions.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, cast


def _aiohttp_web() -> Any:
    return importlib.import_module("aiohttp.web")


DISCORD_INTERACTION_ROUTE = "/discord/interactions/soma"
DISCORD_SIGNATURE_HEADER = "X-Signature-Ed25519"
DISCORD_TIMESTAMP_HEADER = "X-Signature-Timestamp"
DISCORD_TIMESTAMP_MAX_AGE_SECONDS = 300
DISCORD_TIMESTAMP_MAX_FUTURE_SKEW_SECONDS = 5
DISCORD_REPLAY_CACHE_TTL_SECONDS = DISCORD_TIMESTAMP_MAX_AGE_SECONDS + DISCORD_TIMESTAMP_MAX_FUTURE_SKEW_SECONDS

_ALLOWED_PUBLIC_KEY_ENV_NAMES = {"DISCORD_APPLICATION_PUBLIC_KEY"}
_BLOCKED_PUBLIC_KEY_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "WEBHOOK",
    "TELEGRAM",
    "AUTHORIZATION",
    "BEARER",
)
_CUSTOM_ID_RE = re.compile(r"^mim:soma-review:v1:(approve|reject|defer):([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})$")
_ASCII_DECIMAL_RE = re.compile(r"^[0-9]{1,20}$")


@dataclass(frozen=True)
class DiscordInteractionConfig:
    """Resolved local route config.

    `enabled=False` is the safe default. A route becomes active only when the
    API server config explicitly enables it and supplies a Discord application
    public key by value or by the allowlisted public-key environment variable.
    """

    enabled: bool = False
    public_key: str = ""
    route_path: str = DISCORD_INTERACTION_ROUTE


@dataclass
class DiscordInteractionReplayCache:
    """Bounded in-memory replay guard for local Discord interaction callbacks.

    The first live-ready safety layer should not write to runtime SQLite or any
    external store. This cache only remembers request fingerprints in the API
    server process, prunes by TTL, and fails closed when full.
    """

    max_entries: int = 4096
    ttl_seconds: int = DISCORD_REPLAY_CACHE_TTL_SECONDS
    _entries: dict[str, float] = field(default_factory=dict)

    def _fingerprint(self, *, timestamp: str, signature: str, body: bytes) -> str:
        h = hashlib.sha256()
        h.update(timestamp.encode("ascii", errors="ignore"))
        h.update(b":")
        h.update(signature.encode("ascii", errors="ignore"))
        h.update(b":")
        h.update(body)
        return h.hexdigest()

    def _prune(self, now: float) -> None:
        expired = [key for key, expires_at in self._entries.items() if expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)

    def check_and_remember(self, *, timestamp: str, signature: str, body: bytes, now: float | None = None) -> tuple[bool, str]:
        """Remember a verified request fingerprint or reject a duplicate.

        Returns `(True, "ok")` for the first sighting. Duplicate requests inside
        the TTL are rejected. If the cache is full after pruning, fail closed
        rather than evicting a still-valid fingerprint.
        """

        current = time.time() if now is None else float(now)
        self._prune(current)
        key = self._fingerprint(timestamp=timestamp, signature=signature, body=body)
        if key in self._entries:
            return False, "replay_detected"
        if len(self._entries) >= max(1, int(self.max_entries)):
            return False, "replay_cache_full"
        self._entries[key] = current + max(1, int(self.ttl_seconds))
        return True, "ok"


def _looks_like_secret_env(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _BLOCKED_PUBLIC_KEY_ENV_MARKERS)


def _safe_route_path(value: Any) -> str:
    path = str(value or DISCORD_INTERACTION_ROUTE).strip()
    if not path.startswith("/") or "//" in path or any(ch in path for ch in "\r\n\x00"):
        return DISCORD_INTERACTION_ROUTE
    return path


def resolve_discord_interaction_config(extra: dict[str, Any] | None) -> DiscordInteractionConfig:
    """Resolve the API-server Discord interaction route config.

    The resolver deliberately ignores token/secret/webhook-looking env names.
    Discord interaction verification needs the application public key, not a
    bot token, client secret, webhook URL, or Telegram credential.
    """

    section = (extra or {}).get("discord_interactions") or {}
    if not isinstance(section, dict) or section.get("enabled") is not True:
        return DiscordInteractionConfig()

    public_key = str(section.get("public_key") or "").strip()
    public_key_env = str(section.get("public_key_env") or "").strip()
    if not public_key and public_key_env:
        if public_key_env not in _ALLOWED_PUBLIC_KEY_ENV_NAMES or _looks_like_secret_env(public_key_env):
            return DiscordInteractionConfig(route_path=_safe_route_path(section.get("route") or section.get("route_path")))
        public_key = os.getenv(public_key_env, "").strip()

    route_path = _safe_route_path(section.get("route") or section.get("route_path"))
    if not public_key:
        return DiscordInteractionConfig(route_path=route_path)

    return DiscordInteractionConfig(enabled=True, public_key=public_key, route_path=route_path)


def validate_discord_interaction_timestamp(
    timestamp: str,
    *,
    now: float | None = None,
    max_age_seconds: int = DISCORD_TIMESTAMP_MAX_AGE_SECONDS,
    max_future_skew_seconds: int = DISCORD_TIMESTAMP_MAX_FUTURE_SKEW_SECONDS,
) -> tuple[bool, str]:
    """Validate Discord's timestamp before signature and JSON handling.

    Discord signs `timestamp + raw_body`. A good signature can still be replayed
    shortly after capture, so stale and far-future timestamps fail before any
    dry-run handler can run. The parser accepts ASCII decimal epoch seconds only
    to avoid Unicode digit or formatting surprises.
    """

    if not isinstance(timestamp, str) or not _ASCII_DECIMAL_RE.fullmatch(timestamp):
        return False, "invalid_timestamp"
    request_time = int(timestamp)
    current = time.time() if now is None else float(now)
    if request_time > current + max_future_skew_seconds:
        return False, "future_timestamp"
    if current - request_time > max_age_seconds:
        return False, "stale_timestamp"
    return True, "ok"


def default_discord_signature_verifier(*, public_key: str, timestamp: str, signature: str, body: bytes) -> bool:
    """Verify Discord's Ed25519 signature over `timestamp + raw_body`.

    PyNaCl is already present in Hermes' locked dependency graph for Discord
    voice support. If it is unavailable in a minimal runtime, the verifier fails
    closed instead of accepting unsigned interaction callbacks.
    """

    try:
        signing = importlib.import_module("nacl.signing")
        exceptions = importlib.import_module("nacl.exceptions")
        verify_key = signing.VerifyKey(bytes.fromhex(public_key))
        verify_key.verify(timestamp.encode("ascii") + body, bytes.fromhex(signature))
        return True
    except Exception:
        return False


def validate_discord_dry_run_result(result: Any) -> tuple[bool, str]:
    """Reject handler results that claim live or runtime side effects.

    The interaction route is still a preview path. Even an injected handler must
    not smuggle an "applied", "runtime_write", or "live_send" success claim into
    the ACK response during the local safety gate.
    """

    if not isinstance(result, dict):
        return True, "ok"
    for field_name in ("live_send", "runtime_write", "db_write", "applied", "apply"):
        if result.get(field_name) is True:
            return False, "unsafe_dry_run_result"
    return True, "ok"


def _current_time_from(now: Callable[[], float] | float | None) -> float | None:
    """Resolve an injected clock while keeping static type checkers happy."""

    if now is None:
        return None
    if isinstance(now, (int, float)):
        return float(now)
    return float(cast(Callable[[], float], now)())


def _error_response(code: str, status: int):
    return _aiohttp_web().json_response({"error": code}, status=status)


def _parse_component(payload: dict[str, Any]) -> tuple[str, str] | None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    custom_id = data.get("custom_id")
    if not isinstance(custom_id, str) or len(custom_id) > 180:
        return None
    match = _CUSTOM_ID_RE.match(custom_id)
    if not match:
        return None
    return match.group(1), match.group(2)


def build_discord_ack_preview(payload: dict[str, Any], dry_run_result: Any = None) -> dict[str, Any] | None:
    """Build a Discord interaction ACK preview after signature verification.

    Type 1 is Discord PING and must return PONG (`{"type": 1}`). Type 3 is a
    component interaction; this helper returns an ephemeral dry-run message. It
    never applies approval decisions, writes runtime state, or echoes handler
    output back to Discord.
    """

    _ = dry_run_result
    if payload.get("type") == 1:
        return {"type": 1}

    if payload.get("type") != 3:
        return None

    parsed = _parse_component(payload)
    if parsed is None:
        return None
    action, review_id = parsed
    return {
        "type": 4,
        "data": {
            "flags": 64,
            "content": f"MIM dry-run: {action} / {review_id}",
        },
    }


async def handle_discord_interaction_request(
    request,
    *,
    config: DiscordInteractionConfig,
    verifier: Callable[..., bool] | None = None,
    dry_run_handler: Callable[[dict[str, Any]], Any] | None = None,
    replay_cache: DiscordInteractionReplayCache | None = None,
    now: Callable[[], float] | float | None = None,
):
    """Handle a Discord interaction callback in local/dry-run mode.

    Safety order:
    1. Read raw bytes.
    2. Validate timestamp freshness.
    3. Verify signature headers against the raw body.
    4. Reject replays.
    5. Only then parse JSON and build ACK previews.
    """

    if not config.enabled or not config.public_key:
        return _error_response("discord_interactions_disabled", 404)

    body = await request.read()
    timestamp = request.headers.get(DISCORD_TIMESTAMP_HEADER, "")
    signature = request.headers.get(DISCORD_SIGNATURE_HEADER, "")
    if not timestamp or not signature:
        return _error_response("missing_signature", 401)

    current_time = _current_time_from(now)
    timestamp_ok, timestamp_code = validate_discord_interaction_timestamp(timestamp, now=current_time)
    if not timestamp_ok:
        return _error_response(timestamp_code, 401)

    verify = verifier or default_discord_signature_verifier
    if not verify(public_key=config.public_key, timestamp=timestamp, signature=signature, body=body):
        return _error_response("invalid_signature", 401)

    if replay_cache is None:
        return _error_response("replay_cache_required", 500)

    if replay_cache is not None:
        replay_ok, replay_code = replay_cache.check_and_remember(
            timestamp=timestamp,
            signature=signature,
            body=body,
            now=current_time,
        )
        if not replay_ok:
            return _error_response(replay_code, 409)

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error_response("invalid_json", 400)
    if not isinstance(payload, dict):
        return _error_response("invalid_payload", 400)

    if payload.get("type") == 3 and _parse_component(payload) is None:
        return _error_response("invalid_component", 400)

    # Keep this route preview-only. It deliberately does not invoke arbitrary
    # dry-run/apply handlers; even "dry-run" callables can have side effects
    # before returning. A later live apply gate must introduce a side-effect-safe
    # executor contract under separate review.
    _ = dry_run_handler
    dry_run_result = None

    ack = build_discord_ack_preview(payload, dry_run_result)
    if ack is None:
        return _error_response("unsupported_interaction", 400)
    return _aiohttp_web().json_response(ack)
