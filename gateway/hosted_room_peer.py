"""Typed contracts for autonomous cross-gateway hosted-room members.

The Desktop may bootstrap an invitation, but it is never the issuer or the
runtime courier. The target gateway verifies a scoped grant and the full task
coordinates before it admits any model or tool work.
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
import time
import urllib.parse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from gateway.hosted_room_execution_policy import (
    RoomExecutionPolicy,
    execution_policy_mapping,
)


# Version 2 adds authority/member lineage to scoped grants. It is intentionally
# not wire-compatible with the unpublished v1 draft; mixed gateways must fall
# back to Desktop-driven rooms instead of accepting a weaker token shape.
PROTOCOL_VERSION = 2
MAX_TOKEN_BYTES = 16 * 1024
MAX_PROMPT_BYTES = 256 * 1024
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_LINK_PRIORITY = {
    "direct": 0,
    "overlay": 1,
    "relay": 2,
    "pull": 3,
    "desktop": 4,
}
LinkMode = Literal["direct", "overlay", "relay", "pull", "desktop"]
TransportSecurity = Literal["tls", "loopback"]


class HostedRoomPeerError(ValueError):
    """Base error for malformed or unauthorized peer-room input."""


class HostedRoomGrantError(HostedRoomPeerError):
    """Raised when a room-scoped grant is invalid or expired."""


_ROOM_GRANT_SECRET_FILE = ".room-link-grant-secret"


@lru_cache(maxsize=32)
def _gateway_room_grant_secret_for_home(home_value: str) -> bytes:
    """Load one restart-scoped grant secret for an exact installation root."""

    home = Path(home_value)
    home.mkdir(parents=True, exist_ok=True)
    path = home / _ROOM_GRANT_SECRET_FILE

    def _read() -> bytes:
        data = path.read_bytes()
        if len(data) != 32:
            raise HostedRoomGrantError("gateway RoomLink secret is invalid")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            path.chmod(0o600)
        return data

    try:
        material = _read()
    except FileNotFoundError:
        material = os.urandom(32)
        temporary = home / (
            f".{_ROOM_GRANT_SECRET_FILE}.{os.getpid()}.{os.urandom(8).hex()}"
        )
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
                try:
                    parent_fd = os.open(home, os.O_RDONLY)
                    try:
                        os.fsync(parent_fd)
                    finally:
                        os.close(parent_fd)
                except OSError:
                    pass
        finally:
            temporary.unlink(missing_ok=True)
    return hmac.new(
        material,
        b"hermes-hosted-room-installation-grant-v1",
        hashlib.sha256,
    ).digest()


def gateway_room_grant_secret(root: Path | str | None = None) -> bytes:
    """Load or atomically mint the gateway-only RoomLink signing secret.

    API keys are bearer credentials known to clients and may be profile scoped;
    they must never become grant-signing authority. This secret lives in the
    installation root, is not exposed by configuration or capability RPCs, and
    is shared only by the gateway processes that serve this installation.
    """

    if root is None:
        from hermes_constants import get_hermes_home

        # Profile routing uses a context-local HERMES_HOME override. The process
        # environment retains the installation root and is the authority here.
        root = os.environ.get("HERMES_HOME") or get_hermes_home()
    home = Path(root).expanduser().resolve()
    return _gateway_room_grant_secret_for_home(str(home))


def derive_room_grant_secret(api_key: str) -> bytes:
    """Domain-separate room grants from the configured API key.

    The API-server startup guard enforces the production key-strength policy;
    this helper keeps a lower structural floor for isolated contract tests.
    """
    if not isinstance(api_key, str) or len(api_key) < 8:
        raise HostedRoomGrantError("room grants require a strong gateway API key")
    return hmac.new(
        api_key.encode("utf-8"),
        b"hermes-hosted-room-grant-v1",
        hashlib.sha256,
    ).digest()


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise HostedRoomPeerError(f"{field} must be a string")
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise HostedRoomPeerError(f"{field} is invalid")
    return normalized


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HostedRoomPeerError(f"{field} must be a positive integer")
    return value


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise HostedRoomPeerError(f"{field} must be a sha256 digest")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    fields = set(value)
    missing = required - fields
    unknown = fields - required - optional
    if missing:
        raise HostedRoomPeerError(
            f"{label} missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise HostedRoomPeerError(
            f"{label} unknown fields: {', '.join(sorted(unknown))}"
        )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except Exception as exc:
        raise HostedRoomGrantError("room grant encoding is invalid") from exc


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
        _exact_fields(
            value,
            required={
                "installation_id",
                "protocol_versions",
                "link_modes",
                "persistent_process",
                "text",
                "attachments",
                "execution_policy",
                "catalog_digest",
            },
            optional={"endpoint"},
            label="capability catalog",
        )
        installation_id = _identifier(value["installation_id"], field="installation_id")
        versions_raw = value["protocol_versions"]
        if not isinstance(versions_raw, list) or not versions_raw:
            raise HostedRoomPeerError("protocol_versions must be a non-empty list")
        versions = tuple(
            sorted({
                _positive_int(item, field="protocol_version") for item in versions_raw
            })
        )
        links_raw = value["link_modes"]
        if not isinstance(links_raw, list) or not links_raw:
            raise HostedRoomPeerError("link_modes must be a non-empty list")
        links: list[LinkMode] = []
        for item in links_raw:
            if item not in _LINK_PRIORITY:
                raise HostedRoomPeerError("link_modes contains an unsupported mode")
            if item not in links:
                links.append(item)
        for field in ("persistent_process", "text", "attachments"):
            if not isinstance(value[field], bool):
                raise HostedRoomPeerError(f"{field} must be a boolean")

        unsigned = {
            "installation_id": installation_id,
            "protocol_versions": list(versions),
            "link_modes": links,
            "persistent_process": value["persistent_process"],
            "text": value["text"],
            "attachments": value["attachments"],
            "execution_policy": RoomExecutionPolicy.from_mapping(
                value["execution_policy"]
            ).as_mapping(),
        }
        endpoint_url = None
        endpoint_reason = None
        transport_security = None
        if "endpoint" in value:
            endpoint = value["endpoint"]
            if not isinstance(endpoint, Mapping) or not isinstance(
                endpoint.get("available"), bool
            ):
                raise HostedRoomPeerError("endpoint capability is invalid")
            if endpoint["available"]:
                _exact_fields(
                    endpoint,
                    required={"available", "url", "transport_security"},
                    label="endpoint capability",
                )
                endpoint_url, transport_security = validate_room_link_url(
                    endpoint["url"]
                )
                if endpoint["transport_security"] != transport_security:
                    raise HostedRoomPeerError(
                        "endpoint transport_security does not match its URL"
                    )
                normalized_endpoint = {
                    "available": True,
                    "url": endpoint_url,
                    "transport_security": transport_security,
                }
            else:
                _exact_fields(
                    endpoint,
                    required={"available", "reason"},
                    label="endpoint capability",
                )
                endpoint_reason = _identifier(
                    endpoint["reason"], field="endpoint.reason"
                )
                normalized_endpoint = {
                    "available": False,
                    "reason": endpoint_reason,
                }
            unsigned["endpoint"] = normalized_endpoint
        expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        supplied = _digest(value["catalog_digest"], field="catalog_digest")
        if not hmac.compare_digest(expected, supplied):
            raise HostedRoomPeerError("catalog_digest does not match the catalog")
        return cls(
            installation_id=installation_id,
            protocol_versions=versions,
            link_modes=tuple(links),
            persistent_process=value["persistent_process"],
            text=value["text"],
            attachments=value["attachments"],
            execution_policy=RoomExecutionPolicy.from_mapping(
                value["execution_policy"]
            ),
            catalog_digest=supplied,
            endpoint_url=endpoint_url,
            endpoint_reason=endpoint_reason,
            transport_security=transport_security,
        )

    def endpoint_mapping(self) -> dict[str, Any]:
        """Return the normalized self-advertised endpoint capability."""
        if self.endpoint_url is not None:
            return {
                "available": True,
                "url": self.endpoint_url,
                "transport_security": self.transport_security,
            }
        return {
            "available": False,
            "reason": self.endpoint_reason or "not_configured",
        }


def catalog_mapping(
    *,
    installation_id: str,
    protocol_versions: Iterable[int] = (PROTOCOL_VERSION,),
    link_modes: Iterable[LinkMode] = ("direct", "pull"),
    persistent_process: bool,
    text: bool = True,
    attachments: bool = False,
    endpoint: Mapping[str, Any] | None = None,
    target_profile: str | None = None,
    execution_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical catalog mapping with its digest."""
    # A Desktop-managed gateway exits with the app.  Treat the caller's flag
    # as an upper bound so every local catalog construction site stays honest,
    # including older call sites that still pass ``True`` explicitly.
    persistent_process = bool(persistent_process and os.getenv("HERMES_DESKTOP") != "1")
    checked_policy = RoomExecutionPolicy.from_mapping(
        execution_policy
        or execution_policy_mapping(
            target_profile=(
                str(target_profile or "").strip()
                or (os.getenv("HERMES_PROFILE") or "default").strip()
                or "default"
            )
        )
    )
    # A RoomLink run is initiated by another installation. Process-wide YOLO
    # mode bypasses the scoped approval ContextVar, so it cannot be made safe by
    # rewriting the advertised policy. Refuse to advertise or accept remote
    # room execution until the target enables manual or smart approvals.
    if checked_policy.approval_mode == "off":
        raise HostedRoomPeerError(
            "remote room execution requires manual or smart approvals"
        )
    value = {
        "installation_id": _identifier(installation_id, field="installation_id"),
        "protocol_versions": sorted({
            _positive_int(item, field="protocol_version") for item in protocol_versions
        }),
        # Direct HTTPS/loopback is the only RoomLink transport implemented by
        # this backend slice. Do not advertise pull/relay placeholders.
        "link_modes": [mode for mode in dict.fromkeys(link_modes) if mode == "direct"],
        "persistent_process": bool(persistent_process),
        "text": bool(text),
        "attachments": bool(attachments),
        "execution_policy": checked_policy.as_mapping(),
    }
    value["endpoint"] = dict(
        local_room_link_endpoint() if endpoint is None else endpoint
    )
    value["catalog_digest"] = hashlib.sha256(_canonical_json(value)).hexdigest()
    GatewayRoomCatalog.from_mapping(value)
    return value


def local_catalog_mapping(
    *,
    installation_id: str,
    protocol_versions: Iterable[int] = (PROTOCOL_VERSION,),
    link_modes: Iterable[LinkMode] = ("direct", "pull"),
    text: bool = True,
    attachments: bool = False,
    target_profile: str | None = None,
    execution_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one truthful catalog advertised by this local process."""
    return catalog_mapping(
        installation_id=installation_id,
        protocol_versions=protocol_versions,
        link_modes=link_modes,
        persistent_process=True,
        text=text,
        attachments=attachments,
        endpoint=local_room_link_endpoint(),
        target_profile=target_profile,
        execution_policy=execution_policy,
    )


def local_room_link_endpoint(value: Any | None = None) -> dict[str, Any]:
    """Return the validated endpoint this gateway explicitly advertises."""
    configured = _configured_room_link_url() if value is None else value
    if not str(configured or "").strip():
        return {"available": False, "reason": "not_configured"}
    try:
        url, transport_security = validate_room_link_url(configured)
    except HostedRoomPeerError:
        return {"available": False, "reason": "invalid_configuration"}
    return {
        "available": True,
        "url": url,
        "transport_security": transport_security,
    }


@lru_cache(maxsize=16)
def _room_link_url_from_config(home: str) -> str | None:
    """Read the restart-scoped user setting without polling config on probes."""
    from gateway.config import load_gateway_config
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

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
    """Resolve the explicit endpoint with environment override precedence."""
    override = os.getenv("HERMES_ROOM_LINK_URL")
    if override is not None:
        return override
    from hermes_constants import get_default_hermes_root, get_hermes_home

    home = get_hermes_home()
    configured = _room_link_url_from_config(str(home))
    if configured:
        return configured

    # RoomLink is a gateway reachability property, not a Bot personality
    # setting. Named profiles may override it, but otherwise inherit the
    # process gateway's root endpoint so adding a Bot does not require
    # repeating network configuration in every profile.
    root = get_default_hermes_root()
    if root != home:
        return _room_link_url_from_config(str(root))
    return None


def validate_room_link_url(value: Any) -> tuple[str, TransportSecurity]:
    """Validate a RoomLink endpoint and classify its transport protection.

    Scoped grants and room prompts may use plaintext HTTP only when the peer
    is reached through the local loopback interface.  Every non-loopback
    endpoint must use HTTPS.
    """
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

    loopback = hostname == "localhost" or hostname.endswith(".localhost")
    if not loopback:
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
    if not loopback:
        raise HostedRoomPeerError("target_url must use https outside the local machine")
    return raw, "loopback"


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

    def as_mapping(self) -> dict[str, Any]:
        """Return the canonical wire mapping used for fingerprinting."""
        return {
            "protocol_version": self.protocol_version,
            "room_id": self.room_id,
            "home_install_id": self.home_install_id,
            "authority_gateway_id": self.authority_gateway_id,
            "authority_epoch": self.authority_epoch,
            "member_id": self.member_id,
            "target_install_id": self.target_install_id,
            "target_profile": self.target_profile,
            "task_id": self.task_id,
            "execution_generation": self.execution_generation,
            "source_event_seq": self.source_event_seq,
            "cancellation_scope_id": self.cancellation_scope_id,
            "prompt": self.prompt,
            "prompt_digest": self.prompt_digest,
            "capability_digest": self.capability_digest,
            "execution_policy_digest": self.execution_policy_digest,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HostedMemberDispatch":
        required = {
            "protocol_version",
            "room_id",
            "home_install_id",
            "authority_gateway_id",
            "authority_epoch",
            "member_id",
            "target_install_id",
            "target_profile",
            "task_id",
            "execution_generation",
            "source_event_seq",
            "cancellation_scope_id",
            "prompt",
            "prompt_digest",
            "capability_digest",
            "execution_policy_digest",
            "trace_id",
        }
        _exact_fields(value, required=required, label="dispatch")
        prompt = value["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise HostedRoomPeerError("prompt must be a non-empty string")
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise HostedRoomPeerError("prompt is too large")
        expected_prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_digest = _digest(value["prompt_digest"], field="prompt_digest")
        if not hmac.compare_digest(expected_prompt_digest, prompt_digest):
            raise HostedRoomPeerError("prompt_digest does not match prompt")
        return cls(
            protocol_version=_positive_int(
                value["protocol_version"], field="protocol_version"
            ),
            room_id=_identifier(value["room_id"], field="room_id"),
            home_install_id=_identifier(
                value["home_install_id"], field="home_install_id"
            ),
            authority_gateway_id=_identifier(
                value["authority_gateway_id"], field="authority_gateway_id"
            ),
            authority_epoch=_positive_int(
                value["authority_epoch"], field="authority_epoch"
            ),
            member_id=_identifier(value["member_id"], field="member_id"),
            target_install_id=_identifier(
                value["target_install_id"], field="target_install_id"
            ),
            target_profile=_identifier(value["target_profile"], field="target_profile"),
            task_id=_identifier(value["task_id"], field="task_id"),
            execution_generation=_positive_int(
                value["execution_generation"], field="execution_generation"
            ),
            source_event_seq=_positive_int(
                value["source_event_seq"], field="source_event_seq"
            ),
            cancellation_scope_id=_identifier(
                value["cancellation_scope_id"], field="cancellation_scope_id"
            ),
            prompt=prompt,
            prompt_digest=prompt_digest,
            capability_digest=_digest(
                value["capability_digest"], field="capability_digest"
            ),
            execution_policy_digest=_digest(
                value["execution_policy_digest"],
                field="execution_policy_digest",
            ),
            trace_id=_identifier(value["trace_id"], field="trace_id"),
        )


@dataclass(frozen=True)
class RoomLinkProbe:
    """One gateway-verified route candidate."""

    mode: LinkMode
    verified: bool
    encrypted: bool
    latency_ms: float


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


_GRANT_FIELDS = {
    "version",
    "grant_id",
    "room_id",
    "home_install_id",
    "authority_gateway_id",
    "authority_epoch",
    "member_id",
    "target_install_id",
    "target_profile",
    "execution_policy_digest",
    "permissions",
    "issued_at",
    "expires_at",
}
_GRANT_REFRESH_FIELDS = _GRANT_FIELDS | {"status_expires_at"}
MAX_DISPATCH_GRANT_TTL_SECONDS = 24 * 60 * 60
MAX_STATUS_GRANT_TTL_SECONDS = 30 * 24 * 60 * 60


def issue_room_grant(
    secret: bytes,
    *,
    grant_id: str,
    room_id: str,
    home_install_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    member_id: str,
    target_install_id: str,
    target_profile: str,
    execution_policy_digest: str | None = None,
    permissions: Iterable[str] = ("approve", "dispatch", "status", "stop"),
    issued_at: float | None = None,
    ttl_seconds: float = 3600,
    status_ttl_seconds: float | None = None,
    status_expires_at: float | None = None,
) -> str:
    """Issue a target-verifiable bearer grant scoped to one room member."""
    if len(secret) < 32:
        raise HostedRoomGrantError("room grant secret must be at least 32 bytes")
    now = time.time() if issued_at is None else float(issued_at)
    bounded_status_expiry = (
        now + float(ttl_seconds if status_ttl_seconds is None else status_ttl_seconds)
        if status_expires_at is None
        else float(status_expires_at)
    )
    if (
        not math.isfinite(now)
        or ttl_seconds <= 0
        or ttl_seconds > MAX_DISPATCH_GRANT_TTL_SECONDS
        or not math.isfinite(bounded_status_expiry)
        or bounded_status_expiry < now + float(ttl_seconds)
        or bounded_status_expiry > now + MAX_STATUS_GRANT_TTL_SECONDS
    ):
        raise HostedRoomGrantError("room grant lifetime is invalid")
    allowed = tuple(sorted(set(permissions)))
    if not allowed or not set(allowed) <= {
        "approve",
        "dispatch",
        "status",
        "stop",
    }:
        raise HostedRoomGrantError("room grant permissions are invalid")
    payload = {
        "version": PROTOCOL_VERSION,
        "grant_id": _identifier(grant_id, field="grant_id"),
        "room_id": _identifier(room_id, field="room_id"),
        "home_install_id": _identifier(home_install_id, field="home_install_id"),
        "authority_gateway_id": _identifier(
            authority_gateway_id, field="authority_gateway_id"
        ),
        "authority_epoch": _positive_int(
            authority_epoch, field="authority_epoch"
        ),
        "member_id": _identifier(member_id, field="member_id"),
        "target_install_id": _identifier(target_install_id, field="target_install_id"),
        "target_profile": _identifier(target_profile, field="target_profile"),
        "execution_policy_digest": _digest(
            execution_policy_digest
            or execution_policy_mapping(target_profile=target_profile)["policy_digest"],
            field="execution_policy_digest",
        ),
        "permissions": list(allowed),
        "issued_at": now,
        "expires_at": now + float(ttl_seconds),
        "status_expires_at": bounded_status_expiry,
    }
    encoded = _canonical_json(payload)
    signature = hmac.new(secret, encoded, hashlib.sha256).digest()
    token = f"{_b64encode(encoded)}.{_b64encode(signature)}"
    if len(token.encode("ascii")) > MAX_TOKEN_BYTES:
        raise HostedRoomGrantError("room grant is too large")
    return token


def verify_room_grant(
    secret: bytes,
    token: str,
    dispatch: HostedMemberDispatch,
    *,
    permission: str = "dispatch",
    now: float | None = None,
) -> dict[str, Any]:
    """Verify one room grant against exact recipient dispatch coordinates."""
    payload = decode_room_grant(
        secret,
        token,
        permission=permission,
        now=now,
    )
    if payload["version"] != dispatch.protocol_version:
        raise HostedRoomGrantError("room grant protocol does not match dispatch")
    expected = {
        "room_id": dispatch.room_id,
        "home_install_id": dispatch.home_install_id,
        "authority_gateway_id": dispatch.authority_gateway_id,
        "authority_epoch": dispatch.authority_epoch,
        "member_id": dispatch.member_id,
        "target_install_id": dispatch.target_install_id,
        "target_profile": dispatch.target_profile,
        "execution_policy_digest": dispatch.execution_policy_digest,
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise HostedRoomGrantError("room grant scope does not match dispatch")
    return payload


def decode_room_grant(
    secret: bytes,
    token: str,
    *,
    permission: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify grant signature, lifetime and operation without a dispatch."""
    if not isinstance(token, str) or len(token.encode("utf-8")) > MAX_TOKEN_BYTES:
        raise HostedRoomGrantError("room grant is invalid")
    encoded_token, separator, signature_token = token.partition(".")
    if not separator:
        raise HostedRoomGrantError("room grant is invalid")
    encoded = _b64decode(encoded_token)
    supplied_signature = _b64decode(signature_token)
    expected_signature = hmac.new(secret, encoded, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise HostedRoomGrantError("room grant signature is invalid")
    try:
        payload = json.loads(encoded.decode("ascii"))
    except Exception as exc:
        raise HostedRoomGrantError("room grant payload is invalid") from exc
    if not isinstance(payload, dict) or frozenset(payload) not in {
        frozenset(_GRANT_FIELDS),
        frozenset(_GRANT_REFRESH_FIELDS),
    }:
        raise HostedRoomGrantError("room grant fields are invalid")
    checked_now = time.time() if now is None else float(now)
    if not math.isfinite(checked_now):
        raise HostedRoomGrantError("room grant clock is invalid")
    try:
        issued_at = float(payload["issued_at"])
        expires_at = float(payload["expires_at"])
        status_expires_at = float(payload.get("status_expires_at", expires_at))
    except (TypeError, ValueError) as exc:
        raise HostedRoomGrantError("room grant lifetime is invalid") from exc
    if not (
        math.isfinite(issued_at)
        and math.isfinite(expires_at)
        and math.isfinite(status_expires_at)
        and issued_at < expires_at <= status_expires_at
    ):
        raise HostedRoomGrantError("room grant lifetime is invalid")
    operation_expires_at = (
        status_expires_at
        if permission in {"approve", "status", "stop"}
        else expires_at
    )
    if checked_now < issued_at - 30 or checked_now >= operation_expires_at:
        raise HostedRoomGrantError("room grant is expired or not active")
    permissions = payload.get("permissions")
    if not isinstance(permissions, list) or permission not in permissions:
        raise HostedRoomGrantError("room grant does not allow this operation")
    return payload


def room_grant_needs_dispatch_refresh(
    token: str,
    *,
    now: float | None = None,
    leeway_seconds: float = 5 * 60,
) -> bool:
    """Read only grant timing to schedule target-validated refresh.

    This deliberately does not establish trust; the target validates the
    signature and immutable scope before issuing a replacement.
    """
    try:
        encoded_token, separator, _signature = token.partition(".")
        if not separator:
            return True
        payload = json.loads(_b64decode(encoded_token).decode("ascii"))
        expires_at = float(payload["expires_at"])
        checked_now = time.time() if now is None else float(now)
        return checked_now + max(0.0, float(leeway_seconds)) >= expires_at
    except Exception:
        return True
