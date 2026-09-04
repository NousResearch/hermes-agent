"""Typed contracts for autonomous cross-gateway hosted-room members.

The Desktop may bootstrap an invitation but is never the issuer or runtime courier: the target gateway
verifies a scoped grant and the full task coordinates before admitting any model or tool work.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import stat
import urllib.parse
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from gateway.hosted_room_execution_policy import RoomExecutionPolicy, execution_policy_mapping
from gateway.hosted_rooms_common import bounded_int, clock, compact_json, exact_fields, identifier, text


# v2 adds authority/member lineage to scoped grants and is deliberately not wire-compatible with the
# unpublished v1 draft; mixed gateways fall back to Desktop-driven rooms rather than accept a weaker token.
PROTOCOL_VERSION = 2
MAX_TOKEN_BYTES = 16 * 1024
MAX_PROMPT_BYTES = 256 * 1024
MAX_ROOM_LINK_ATTACHMENTS = 16
MAX_ROOM_LINK_ATTACHMENT_BYTES = 15_000_000
MAX_ROOM_LINK_ATTACHMENT_BATCH_BYTES = 50_000_000
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_LINK_MODES = frozenset({"direct", "overlay", "relay", "pull", "desktop"})
LinkMode = Literal["direct", "overlay", "relay", "pull", "desktop"]
TransportSecurity = Literal["tls", "loopback"]


class HostedRoomPeerError(ValueError): """Base error for malformed or unauthorized peer-room input."""

class HostedRoomGrantError(HostedRoomPeerError): """Raised when a room-scoped grant is invalid or expired."""


_ROOM_GRANT_SECRET_FILE = ".room-link-grant-secret"


@lru_cache(maxsize=32)
def _gateway_room_grant_secret_for_home(home_value: str) -> bytes:
    """Load one restart-scoped grant secret for an exact installation root."""
    from hermes_cli.install_identity import _fsync_directory
    (home := Path(home_value)).mkdir(parents=True, exist_ok=True)
    path = home / _ROOM_GRANT_SECRET_FILE
    def _read() -> bytes:
        if len(data := path.read_bytes()) != 32:
            raise HostedRoomGrantError("gateway RoomLink secret is invalid")
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            path.chmod(0o600)
        return data
    try:
        material = _read()
    except FileNotFoundError:
        # Atomic create: O_EXCL temp file, hard-link into place (loser re-reads the winner's secret),
        # fsync the directory, always drop the temp name.
        material = os.urandom(32)
        temporary = home / f".{_ROOM_GRANT_SECRET_FILE}.{os.getpid()}.{os.urandom(8).hex()}"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(material)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                material = _read()
            else:
                _fsync_directory(home)
        finally:
            temporary.unlink(missing_ok=True)
    return hmac.new(material, b"hermes-hosted-room-installation-grant-v1", hashlib.sha256).digest()


def gateway_room_grant_secret(root: Path | str | None = None) -> bytes:
    """Load or atomically mint the gateway-only RoomLink signing secret.

    API keys are client-known, possibly profile-scoped bearer credentials and must never become
    grant-signing authority; this secret lives in the installation root, is never exposed by config
    or capability RPCs, and is shared only by this installation's gateway processes.
    """
    if root is None:
        from hermes_constants import get_hermes_home
        # Profile routing uses a context-local HERMES_HOME override; the process environment
        # retains the installation root and is the authority here.
        root = os.environ.get("HERMES_HOME") or get_hermes_home()
    return _gateway_room_grant_secret_for_home(str(Path(root).expanduser().resolve()))


def derive_room_grant_secret(api_key: str) -> bytes:
    """Domain-separate room grants from the configured API key.

    The 8-char floor is for contract tests; production key strength is enforced by the API-server startup guard.
    """
    if not isinstance(api_key, str) or len(api_key) < 8:
        raise HostedRoomGrantError("room grants require a strong gateway API key")
    return hmac.new(api_key.encode("utf-8"), b"hermes-hosted-room-grant-v1", hashlib.sha256).digest()


def _identifier(value: Any, *, field: str) -> str:
    return identifier(
        value, label=field, error=HostedRoomPeerError, max_chars=256, pattern=_IDENTIFIER_RE,
        invalid=f"{field} is invalid")


def _positive_int(value: Any, *, field: str) -> int:
    return bounded_int(value, error=HostedRoomPeerError, message=f"{field} must be a positive integer", low=1)


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise HostedRoomPeerError(f"{field} must be a sha256 digest")
    return value


_exact_fields = partial(
    exact_fields, error=HostedRoomPeerError, missing_fmt="{label} missing fields: {fields}",
    unknown_fmt="{label} unknown fields: {fields}")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return compact_json(value).encode("ascii")


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as exc:
        raise HostedRoomGrantError("room grant encoding is invalid") from exc


def _split_token(token: str) -> tuple[bytes, bytes]:
    """Return the decoded ``(payload, signature)`` halves of a grant token."""
    encoded_token, separator, signature_token = token.partition(".")
    if not separator:
        raise HostedRoomGrantError("room grant is invalid")
    return _b64decode(encoded_token), _b64decode(signature_token)


def _non_empty_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise HostedRoomPeerError(f"{field} must be a non-empty list")
    return value


def _parse_link_modes(links_raw: Any) -> tuple[LinkMode, ...]:
    if any(item not in _LINK_MODES for item in _non_empty_list(links_raw, field="link_modes")):
        raise HostedRoomPeerError("link_modes contains an unsupported mode")
    return tuple(dict.fromkeys(links_raw))


def _protocol_versions(versions_raw: Any) -> list[int]:
    return sorted({_positive_int(item, field="protocol_version") for item in versions_raw})


def _catalog_digest(value: Mapping[str, Any]) -> str:
    """Digest of the catalog mapping without its ``catalog_digest`` field."""
    return hashlib.sha256(_canonical_json({k: v for k, v in value.items() if k != "catalog_digest"})).hexdigest()


def _parse_endpoint(endpoint: Any) -> tuple[str | None, str | None, TransportSecurity | None]:
    """Return ``(url, reason, transport_security)`` for an advertised endpoint capability."""
    if not isinstance(endpoint, Mapping) or not isinstance(endpoint.get("available"), bool):
        raise HostedRoomPeerError("endpoint capability is invalid")
    if not endpoint["available"]:
        _exact_fields(endpoint, required={"available", "reason"}, label="endpoint capability")
        return None, _identifier(endpoint["reason"], field="endpoint.reason"), None
    _exact_fields(endpoint, required={"available", "url", "transport_security"}, label="endpoint capability")
    url, transport_security = validate_room_link_url(endpoint["url"])
    if endpoint["transport_security"] != transport_security:
        raise HostedRoomPeerError("endpoint transport_security does not match its URL")
    return url, None, transport_security


_CATALOG_FIELDS = {
    "installation_id", "protocol_versions", "link_modes", "persistent_process", "text", "attachments",
    "execution_policy", "catalog_digest"}


@dataclass(frozen=True)
class GatewayRoomCatalog:
    """Authenticated gateway capabilities inherited by its Bots."""
    installation_id: str
    protocol_versions: tuple[int, ...]
    link_modes: tuple[LinkMode, ...]
    persistent_process: bool
    text: bool
    attachments: bool
    execution_policy: RoomExecutionPolicy
    catalog_digest: str
    endpoint_url: str | None = None
    endpoint_reason: str | None = None
    transport_security: TransportSecurity | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GatewayRoomCatalog":
        _exact_fields(value, required=_CATALOG_FIELDS, optional={"endpoint"}, label="capability catalog")
        installation_id = _identifier(value["installation_id"], field="installation_id")
        versions = tuple(_protocol_versions(_non_empty_list(value["protocol_versions"], field="protocol_versions")))
        links = _parse_link_modes(value["link_modes"])
        for field in ("persistent_process", "text", "attachments"):
            if not isinstance(value[field], bool):
                raise HostedRoomPeerError(f"{field} must be a boolean")
        policy = RoomExecutionPolicy.from_mapping(value["execution_policy"])
        endpoint = _parse_endpoint(value["endpoint"]) if "endpoint" in value else (None, None, None)
        catalog = cls(
            installation_id, versions, links, value["persistent_process"], value["text"], value["attachments"], policy,
            _digest(value["catalog_digest"], field="catalog_digest"), *endpoint)
        if not hmac.compare_digest(_catalog_digest(catalog.as_mapping()), catalog.catalog_digest):
            raise HostedRoomPeerError("catalog_digest does not match the catalog")
        return catalog

    def as_mapping(self) -> dict[str, Any]:
        """Canonical catalog mapping; ``endpoint`` appears only when advertised."""
        value = {
            "installation_id": self.installation_id, "protocol_versions": list(self.protocol_versions),
            "link_modes": list(self.link_modes), "persistent_process": self.persistent_process, "text": self.text,
            "attachments": self.attachments, "execution_policy": self.execution_policy.as_mapping(),
            "catalog_digest": self.catalog_digest}
        if self.endpoint_url is not None or self.endpoint_reason is not None:
            value["endpoint"] = self.endpoint_mapping()
        return value

    def endpoint_mapping(self) -> dict[str, Any]:
        """Return the normalized self-advertised endpoint capability."""
        if self.endpoint_url is None:
            return {"available": False, "reason": self.endpoint_reason or "not_configured"}
        return {"available": True, "url": self.endpoint_url, "transport_security": self.transport_security}


def catalog_mapping(
    *, installation_id: str, protocol_versions: Iterable[int] = (PROTOCOL_VERSION,),
    link_modes: Iterable[LinkMode] = ("direct", "pull"), persistent_process: bool, text: bool = True,
    attachments: bool = False, endpoint: Mapping[str, Any] | None = None, target_profile: str | None = None,
    execution_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a canonical catalog mapping with its digest."""
    # A Desktop-managed gateway exits with the app: the caller's flag is only an upper bound.
    persistent_process = bool(persistent_process and os.getenv("HERMES_DESKTOP") != "1")
    profile = str(target_profile or "").strip() or (os.getenv("HERMES_PROFILE") or "default").strip() or "default"
    checked_policy = RoomExecutionPolicy.from_mapping(
        execution_policy or execution_policy_mapping(target_profile=profile))
    # A RoomLink run is initiated by another installation. Process-wide YOLO mode bypasses the scoped
    # approval ContextVar, so rewriting the advertised policy cannot make it safe: refuse.
    if checked_policy.approval_mode == "off":
        raise HostedRoomPeerError("remote room execution requires manual or smart approvals")
    value = {
        "installation_id": _identifier(installation_id, field="installation_id"),
        "protocol_versions": _protocol_versions(protocol_versions),
        # Direct HTTPS/loopback is the only implemented RoomLink transport; never advertise pull/relay.
        "link_modes": [mode for mode in dict.fromkeys(link_modes) if mode == "direct"],
        "persistent_process": persistent_process, "text": bool(text), "attachments": bool(attachments),
        "execution_policy": checked_policy.as_mapping(),
        "endpoint": dict(local_room_link_endpoint() if endpoint is None else endpoint)}
    value["catalog_digest"] = _catalog_digest(value)
    GatewayRoomCatalog.from_mapping(value)
    return value


# The one truthful catalog advertised by this local process (keyword arguments as for catalog_mapping).
local_catalog_mapping = partial(catalog_mapping, persistent_process=True)


def local_room_link_endpoint(value: Any | None = None) -> dict[str, Any]:
    """Return the validated endpoint this gateway explicitly advertises."""
    if not str((configured := _configured_room_link_url() if value is None else value) or "").strip():
        return {"available": False, "reason": "not_configured"}
    try:
        url, transport_security = validate_room_link_url(configured)
    except HostedRoomPeerError:
        return {"available": False, "reason": "invalid_configuration"}
    return {"available": True, "url": url, "transport_security": transport_security}


@lru_cache(maxsize=16)
def _room_link_url_from_config(home: str) -> str | None:
    """Read the restart-scoped user setting without polling config on probes."""
    from gateway.config import load_gateway_config
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    if str(get_hermes_home()) == home:
        value = load_gateway_config().room_link_url
    else:
        token = set_hermes_home_override(home)
        try:
            value = load_gateway_config().room_link_url
        finally:
            reset_hermes_home_override(token)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _configured_room_link_url() -> str | None:
    """Resolve the explicit endpoint: env override > profile config > root config."""
    if (override := os.getenv("HERMES_ROOM_LINK_URL")) is not None:
        return override
    from hermes_constants import get_default_hermes_root, get_hermes_home
    home = get_hermes_home()
    if configured := _room_link_url_from_config(str(home)):
        return configured
    # RoomLink is a gateway reachability property, not a Bot personality setting: named profiles may
    # override it but otherwise inherit the process gateway's root endpoint.
    root = get_default_hermes_root()
    return _room_link_url_from_config(str(root)) if root != home else None


def validate_room_link_url(value: Any) -> tuple[str, TransportSecurity]:
    """Validate a RoomLink endpoint and classify its transport: plaintext HTTP only toward loopback."""
    raw = str(value or "").strip().rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(raw)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        # Force urllib to validate a malformed/out-of-range port.
        parsed.port
    except ValueError as exc:
        raise HostedRoomPeerError("target_url is invalid") from exc
    if not hostname or parsed.username is not None or parsed.password is not None:
        raise HostedRoomPeerError("target_url is invalid")
    if parsed.query or parsed.fragment:
        raise HostedRoomPeerError("target_url must not include query or fragment")
    if parsed.scheme.lower() == "https":
        return raw, "tls"
    if parsed.scheme.lower() != "http":
        raise HostedRoomPeerError("target_url must use https")
    try:
        if hostname == "localhost" or hostname.endswith(".localhost") or ipaddress.ip_address(hostname).is_loopback:
            return raw, "loopback"
    except ValueError:
        pass
    raise HostedRoomPeerError("target_url must use https outside the local machine")


# Validator per dispatch field, in validation order (prompt/prompt_digest are cross-checked in from_mapping).
_DISPATCH_FIELDS: dict[str, Callable[..., Any]] = dict(
    protocol_version=_positive_int, room_id=_identifier, home_install_id=_identifier, authority_gateway_id=_identifier,
    authority_epoch=_positive_int, member_id=_identifier, target_install_id=_identifier, target_profile=_identifier,
    task_id=_identifier, execution_generation=_positive_int, source_event_seq=_positive_int,
    cancellation_scope_id=_identifier, capability_digest=_digest, execution_policy_digest=_digest, trace_id=_identifier)


@dataclass(frozen=True)
class HostedMemberDispatch:
    """Recipient-validated identity for one remote room member attempt."""
    protocol_version: int
    room_id: str
    home_install_id: str
    authority_gateway_id: str
    authority_epoch: int
    member_id: str
    target_install_id: str
    target_profile: str
    task_id: str
    execution_generation: int
    source_event_seq: int
    cancellation_scope_id: str
    prompt: str
    prompt_digest: str
    capability_digest: str
    execution_policy_digest: str
    trace_id: str
    attachment_manifest_digest: str | None = None

    def as_mapping(self) -> dict[str, Any]:
        """Return the canonical wire mapping used for fingerprinting."""
        mapping = asdict(self)
        if self.attachment_manifest_digest is None:
            mapping.pop("attachment_manifest_digest")
        return mapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HostedMemberDispatch":
        _exact_fields(value, required=set(_DISPATCH_FIELDS) | {"prompt", "prompt_digest"},
                      optional={"attachment_manifest_digest"}, label="dispatch")
        if not isinstance(prompt := value["prompt"], str) or not prompt.strip():
            raise HostedRoomPeerError("prompt must be a non-empty string")
        text(prompt, error=HostedRoomPeerError, label="prompt", max_bytes=MAX_PROMPT_BYTES, strip=False)
        prompt_digest = _digest(value["prompt_digest"], field="prompt_digest")
        if not hmac.compare_digest(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), prompt_digest):
            raise HostedRoomPeerError("prompt_digest does not match prompt")
        return cls(
            prompt=prompt, prompt_digest=prompt_digest,
            attachment_manifest_digest=(
                _digest(value["attachment_manifest_digest"], field="attachment_manifest_digest")
                if "attachment_manifest_digest" in value else None),
            **{name: check(value[name], field=name) for name, check in _DISPATCH_FIELDS.items()})


# Grant scope fields in issue-time validation order; each uses the matching dispatch checker (grant_id: identifier).
_GRANT_SCOPE = (
    "grant_id", "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id",
    "target_install_id", "target_profile")
_GRANT_FIELDS = frozenset({
    "version", *_GRANT_SCOPE, "execution_policy_digest", "permissions", "issued_at", "expires_at"})
_GRANT_REFRESH_FIELDS = _GRANT_FIELDS | {"status_expires_at"}
_GRANT_PERMISSIONS = {"approve", "attachment.stage", "dispatch", "status", "stop"}
MAX_DISPATCH_GRANT_TTL_SECONDS = 24 * 60 * 60
MAX_STATUS_GRANT_TTL_SECONDS = 30 * 24 * 60 * 60


def issue_room_grant(
    secret: bytes, *, grant_id: str, room_id: str, home_install_id: str, authority_gateway_id: str,
    authority_epoch: int, member_id: str, target_install_id: str, target_profile: str,
    execution_policy_digest: str | None = None,
    permissions: Iterable[str] = ("approve", "attachment.stage", "dispatch", "status", "stop"),
    issued_at: float | None = None, ttl_seconds: float = 3600, status_ttl_seconds: float | None = None,
    status_expires_at: float | None = None) -> str:
    """Issue a target-verifiable bearer grant scoped to one room member."""
    if len(secret) < 32:
        raise HostedRoomGrantError("room grant secret must be at least 32 bytes")
    now = clock(issued_at)
    bounded_status_expiry = (
        now + float(ttl_seconds if status_ttl_seconds is None else status_ttl_seconds)
        if status_expires_at is None
        else float(status_expires_at))
    if (
        not math.isfinite(now) or ttl_seconds <= 0 or ttl_seconds > MAX_DISPATCH_GRANT_TTL_SECONDS
        or not math.isfinite(bounded_status_expiry) or bounded_status_expiry < now + float(ttl_seconds)
        or bounded_status_expiry > now + MAX_STATUS_GRANT_TTL_SECONDS):
        raise HostedRoomGrantError("room grant lifetime is invalid")
    allowed = tuple(sorted(set(permissions)))
    if not allowed or not set(allowed) <= _GRANT_PERMISSIONS:
        raise HostedRoomGrantError("room grant permissions are invalid")
    scope = locals()
    payload = {
        "version": PROTOCOL_VERSION,
        **{name: _DISPATCH_FIELDS.get(name, _identifier)(scope[name], field=name) for name in _GRANT_SCOPE},
        "execution_policy_digest": _digest(
            execution_policy_digest
            or execution_policy_mapping(target_profile=target_profile)["policy_digest"], field="execution_policy_digest"
        ), "permissions": list(allowed), "issued_at": now, "expires_at": now + float(ttl_seconds),
        "status_expires_at": bounded_status_expiry}
    encoded = _canonical_json(payload)
    token = f"{_b64encode(encoded)}.{_b64encode(hmac.new(secret, encoded, hashlib.sha256).digest())}"
    if len(token.encode("ascii")) > MAX_TOKEN_BYTES:
        raise HostedRoomGrantError("room grant is too large")
    return token


def verify_room_grant(
    secret: bytes, token: str, dispatch: HostedMemberDispatch, *, permission: str = "dispatch", now: float | None = None
) -> dict[str, Any]:
    """Verify one room grant against exact recipient dispatch coordinates."""
    payload = decode_room_grant(secret, token, permission=permission, now=now)
    if payload["version"] != dispatch.protocol_version:
        raise HostedRoomGrantError("room grant protocol does not match dispatch")
    if any(payload.get(f) != getattr(dispatch, f) for f in (*_GRANT_SCOPE[1:], "execution_policy_digest")):
        raise HostedRoomGrantError("room grant scope does not match dispatch")
    return payload


def decode_room_grant(secret: bytes, token: str, *, permission: str, now: float | None = None,
                      allow_expired_for_revocation: bool = False) -> dict[str, Any]:
    """Verify a signed grant without restoring expired operational authority."""
    if allow_expired_for_revocation and permission != "status":
        raise HostedRoomGrantError("expired grants are valid only for revocation")
    if not isinstance(token, str) or len(token.encode("utf-8")) > MAX_TOKEN_BYTES:
        raise HostedRoomGrantError("room grant is invalid")
    encoded, supplied_signature = _split_token(token)
    if not hmac.compare_digest(hmac.new(secret, encoded, hashlib.sha256).digest(), supplied_signature):
        raise HostedRoomGrantError("room grant signature is invalid")
    try:
        payload = json.loads(encoded.decode("ascii"))
    except Exception as exc:
        raise HostedRoomGrantError("room grant payload is invalid") from exc
    if not isinstance(payload, dict) or frozenset(payload) not in {_GRANT_FIELDS, _GRANT_REFRESH_FIELDS}:
        raise HostedRoomGrantError("room grant fields are invalid")
    if not math.isfinite(checked_now := clock(now)):
        raise HostedRoomGrantError("room grant clock is invalid")
    try:
        issued_at = float(payload["issued_at"])
        expires_at = float(payload["expires_at"])
        status_expires_at = float(payload.get("status_expires_at", expires_at))
    except (TypeError, ValueError) as exc:
        raise HostedRoomGrantError("room grant lifetime is invalid") from exc
    lifetimes = (issued_at, expires_at, status_expires_at)
    if not (all(map(math.isfinite, lifetimes)) and issued_at < expires_at <= status_expires_at):
        raise HostedRoomGrantError("room grant lifetime is invalid")
    operation_expires_at = status_expires_at if permission in {"approve", "status", "stop"} else expires_at
    if not allow_expired_for_revocation and (checked_now < issued_at - 30 or checked_now >= operation_expires_at):
        raise HostedRoomGrantError("room grant is expired or not active")
    if not isinstance(permissions := payload.get("permissions"), list) or permission not in permissions:
        raise HostedRoomGrantError("room grant does not allow this operation")
    return payload


def room_grant_needs_dispatch_refresh(token: str, *, now: float | None = None, leeway_seconds: float = 5 * 60) -> bool:
    """Read only grant timing to schedule refresh; trust is established by the target, not here."""
    try:
        expires_at = float(json.loads(_split_token(token)[0].decode("ascii"))["expires_at"])
        return clock(now) + max(0.0, float(leeway_seconds)) >= expires_at
    except Exception:
        return True


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import time  # noqa: F401,E402

@dataclass(frozen=True)
class RoomLinkProbe:
    """One gateway-verified route candidate."""

    mode: LinkMode
    verified: bool
    encrypted: bool
    latency_ms: float

_LINK_PRIORITY = {
    "direct": 0,
    "overlay": 1,
    "relay": 2,
    "pull": 3,
    "desktop": 4,
}

def select_room_link(
    probes: Iterable[RoomLinkProbe],
    *,
    desktop_available: bool,
) -> RoomLinkProbe | None:
    """Choose the fastest safe route without weakening encryption."""
    candidates = [
        probe
        for probe in probes
        if probe.verified
        and probe.encrypted
        and probe.mode != "desktop"
        and math.isfinite(probe.latency_ms)
        and probe.latency_ms >= 0
    ]
    if candidates:
        return min(
            candidates,
            key=lambda item: (_LINK_PRIORITY[item.mode], item.latency_ms),
        )
    if desktop_available:
        return RoomLinkProbe(
            mode="desktop",
            verified=True,
            encrypted=True,
            latency_ms=0,
        )
    return None
# ---- END PLUGIN-COMPAT ----


def attachment_manifest_digest(value: Any) -> str:
    """Return the canonical SHA-256 identity for a validated manifest."""
    manifest = canonical_attachment_manifest(value)
    return hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def canonical_attachment_manifest(value: Any) -> list[dict[str, Any]]:
    """Validate and normalize one bounded RoomLink attachment manifest.

    The manifest contains metadata and content digests only. Bytes, local paths,
    and bearer material are deliberately outside this durable wire contract.
    Entry order is preserved because it is part of the user-visible message.
    """
    if not isinstance(value, list) or not value:
        raise HostedRoomPeerError("attachment manifest must be a non-empty list")
    if len(value) > MAX_ROOM_LINK_ATTACHMENTS:
        raise HostedRoomPeerError("attachment manifest has too many entries")
    required = {"attachment_id", "kind", "name", "size", "mime", "sha256"}
    normalized: list[dict[str, Any]] = []
    total = 0
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise HostedRoomPeerError("attachment manifest entries must be objects")
        _exact_fields(raw, required=required, label="attachment manifest entry")
        attachment_id = _identifier(raw["attachment_id"], field="attachment_id")
        if attachment_id in seen:
            raise HostedRoomPeerError("attachment ids must be unique")
        seen.add(attachment_id)
        kind = raw["kind"]
        if kind not in {"image", "pdf", "file"}:
            raise HostedRoomPeerError("attachment kind is unsupported")
        name = raw["name"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 255
            or name.strip() in {".", ".."}
            or any(token in name for token in ("/", "\\", "\x00", "\n", "\r"))
        ):
            raise HostedRoomPeerError("attachment name must be a bounded basename")
        mime = raw["mime"]
        if (
            not isinstance(mime, str)
            or not mime.strip()
            or len(mime) > 127
            or "/" not in mime
        ):
            raise HostedRoomPeerError("attachment mime is invalid")
        size = raw["size"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= MAX_ROOM_LINK_ATTACHMENT_BYTES
        ):
            raise HostedRoomPeerError("attachment size is outside the RoomLink limit")
        total += size
        if total > MAX_ROOM_LINK_ATTACHMENT_BATCH_BYTES:
            raise HostedRoomPeerError("attachment batch is too large")
        normalized.append(
            {
                "attachment_id": attachment_id,
                "kind": kind,
                "name": name.strip(),
                "size": size,
                "mime": mime.strip().lower(),
                "sha256": _digest(raw["sha256"], field="attachment sha256"),
            }
        )
    return normalized
