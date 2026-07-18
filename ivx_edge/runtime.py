"""Fail-closed IVX Edge command verifier and deterministic safe adapters."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
from functools import wraps
from types import MappingProxyType
from typing import Any, Callable, Mapping

MAX_CLOCK_SKEW_SECONDS = 30
MAX_COMMAND_LIFETIME_SECONDS = 300
CONTENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
MAX_CANONICAL_COMMAND_BYTES = 16 * 1024
MAX_JSON_DEPTH = 8
MAX_JSON_ITEMS = 128


class EnrollmentError(ValueError):
    """Enrollment request failed closed."""


class SecurityError(ValueError):
    """A command violated identity, signature, replay, or policy boundaries."""


@dataclass(frozen=True)
class DeviceIdentity:
    device_id: str
    workspace_id: str
    app_id: str
    key: bytes = field(repr=False)


@dataclass(frozen=True)
class Command:
    command_id: str
    device_id: str
    workspace_id: str
    app_id: str
    capability: str
    payload: Mapping[str, Any]
    issued_at: int
    expires_at: int
    nonce: str
    signature: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_json(self.payload))


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    status: str
    detail: str
    memory_event: Mapping[str, str] | None = None


@dataclass(frozen=True)
class AuditEvent:
    event: str
    command_id: str
    workspace_id: str
    app_id: str
    device_id: str
    capability: str
    status: str
    timestamp: int
    payload_digest: str


@dataclass(frozen=True)
class RuntimePolicy:
    capabilities: frozenset[str]
    require_confirmation: frozenset[str] = frozenset({"screen.content.set"})


@dataclass
class _EnrollmentCode:
    digest: str
    workspace_id: str
    app_id: str
    expires_at: int
    consumed: bool = False


class EnrollmentAuthority:
    """Issues one-time, short-lived enrollment codes for the pilot contract."""

    def __init__(self, now: Callable[[], float] = time.time) -> None:
        self._now = now
        self._codes: dict[str, _EnrollmentCode] = {}

    def issue_code(self, workspace_id: str, app_id: str, ttl_seconds: int = 300) -> str:
        if not workspace_id or not app_id:
            raise EnrollmentError("workspace_id and app_id are required")
        if ttl_seconds <= 0 or ttl_seconds > 600:
            raise EnrollmentError("enrollment code lifetime must be between 1 and 600 seconds")

        code = secrets.token_urlsafe(18)
        code_id = secrets.token_hex(8)
        self._codes[code_id] = _EnrollmentCode(
            digest=_secret_digest(code),
            workspace_id=workspace_id,
            app_id=app_id,
            expires_at=int(self._now()) + ttl_seconds,
        )
        return f"{code_id}.{code}"

    def enroll(self, presented_code: str, device_id: str) -> DeviceIdentity:
        if not device_id or not CONTENT_ID_PATTERN.fullmatch(device_id):
            raise EnrollmentError("device_id must be a bounded lowercase identifier")
        try:
            code_id, secret = presented_code.split(".", 1)
        except ValueError as exc:
            raise EnrollmentError("invalid enrollment code") from exc

        record = self._codes.get(code_id)
        if record is None or not hmac.compare_digest(record.digest, _secret_digest(secret)):
            raise EnrollmentError("invalid enrollment code")
        if record.consumed:
            raise EnrollmentError("enrollment code was already used")
        if int(self._now()) > record.expires_at:
            raise EnrollmentError("enrollment code expired")

        record.consumed = True
        return DeviceIdentity(
            device_id=device_id,
            workspace_id=record.workspace_id,
            app_id=record.app_id,
            key=secrets.token_bytes(32),
        )


def _secret_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _freeze_json(value: Any, depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise SecurityError("payload nesting exceeds policy")
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        raise SecurityError("floating-point payload values are not allowed")
    if isinstance(value, MappingABC):
        if len(value) > MAX_JSON_ITEMS:
            raise SecurityError("payload contains too many fields")
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise SecurityError("payload keys must be bounded strings")
            frozen[key] = _freeze_json(item, depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_JSON_ITEMS:
            raise SecurityError("payload contains too many items")
        return tuple(_freeze_json(item, depth + 1) for item in value)
    raise SecurityError("payload contains a non-JSON value")


def _plain_json(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _canonical_command(command: Command) -> bytes:
    body = {
        "app_id": command.app_id,
        "capability": command.capability,
        "command_id": command.command_id,
        "device_id": command.device_id,
        "expires_at": command.expires_at,
        "issued_at": command.issued_at,
        "nonce": command.nonce,
        "payload": _plain_json(command.payload),
        "workspace_id": command.workspace_id,
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    if len(canonical) > MAX_CANONICAL_COMMAND_BYTES:
        raise SecurityError("command exceeds maximum canonical size")
    return canonical


def sign_command(command: Command, key: bytes) -> Command:
    signature = hmac.new(key, _canonical_command(command), hashlib.sha256).hexdigest()
    return replace(command, signature=signature)


def _synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _audit_security_failures(method):
    @wraps(method)
    def wrapper(self, command: Command, *args, **kwargs):
        try:
            return method(self, command, *args, **kwargs)
        except SecurityError as exc:
            if not getattr(exc, "_ivx_edge_audited", False):
                self._audit(command, "failed")
                exc._ivx_edge_audited = True
            raise

    return wrapper


class DeviceRuntime:
    """Executes two bounded simulator adapters after strict verification."""

    def __init__(
        self,
        identity: DeviceIdentity,
        policy: RuntimePolicy,
        *,
        confirm: Callable[[Command], bool] | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.identity = identity
        self.policy = policy
        self.confirm = confirm or (lambda _command: False)
        self._now = now
        self._lock = threading.RLock()
        self.online = True
        self.revoked = False
        self._nonces: set[str] = set()
        self._results: dict[str, CommandResult] = {}
        self._result_digests: dict[str, str] = {}
        self._queue: list[Command] = []
        self.audit: list[AuditEvent] = []

    @_synchronized
    def set_online(self, online: bool) -> list[CommandResult]:
        self.online = online
        if not online:
            return []
        results: list[CommandResult] = []
        for command in list(self._queue):
            try:
                result = self.execute(command)
            except SecurityError as exc:
                result = CommandResult(command.command_id, "failed", str(exc))
            results.append(result)
            self._queue.remove(command)
        return results

    @_synchronized
    def revoke(self) -> None:
        self.revoked = True
        self._queue.clear()

    @_synchronized
    @_audit_security_failures
    def submit(self, command: Command) -> CommandResult:
        self._verify(command)
        digest = hashlib.sha256(_canonical_command(command)).hexdigest()
        if command.command_id in self._results:
            if self._result_digests[command.command_id] != digest:
                raise SecurityError("command_id was reused with different signed content")
            return self._results[command.command_id]
        if not self.online:
            queued = next(
                (item for item in self._queue if item.command_id == command.command_id),
                None,
            )
            if queued is not None:
                if not hmac.compare_digest(_canonical_command(queued), _canonical_command(command)):
                    raise SecurityError("command_id was reused with different signed content")
                return CommandResult(command.command_id, "queued", "Queued while offline.")
            if any(item.nonce == command.nonce for item in self._queue):
                raise SecurityError("nonce replay detected")
            if not any(item.command_id == command.command_id for item in self._queue):
                self._queue.append(command)
            result = CommandResult(command.command_id, "queued", "Queued while offline.")
            self._audit(command, result.status)
            return result
        return self.execute(command)

    @_synchronized
    @_audit_security_failures
    def execute(self, command: Command) -> CommandResult:
        self._verify(command)
        prior = self._results.get(command.command_id)
        if prior is not None:
            digest = hashlib.sha256(_canonical_command(command)).hexdigest()
            if self._result_digests[command.command_id] != digest:
                raise SecurityError("command_id was reused with different signed content")
            return prior

        verified_digest = hashlib.sha256(_canonical_command(command)).hexdigest()

        content_id: str | None = None
        if command.capability == "sensor.temperature.read":
            if command.payload:
                raise SecurityError("sensor read does not accept payload fields")
        elif command.capability == "screen.content.set":
            content_id = command.payload.get("content_id")
            if not isinstance(content_id, str) or not CONTENT_ID_PATTERN.fullmatch(content_id):
                raise SecurityError("screen adapter accepts only a bounded content_id")
            if set(command.payload) != {"content_id"}:
                raise SecurityError("screen adapter payload contains unapproved fields")
        else:
            raise SecurityError("capability has no registered adapter")

        try:
            confirmed = (
                command.capability not in self.policy.require_confirmation
                or self.confirm(command)
            )
        except Exception as exc:
            raise SecurityError("confirmation handler failed closed") from exc

        self._nonces.add(command.nonce)
        if not confirmed:
            result = CommandResult(
                command.command_id,
                "rejected",
                "Explicit confirmation was not granted.",
            )
        elif command.capability == "sensor.temperature.read":
            result = CommandResult(
                command.command_id,
                "completed",
                "Simulated temperature: 21.5 C.",
                {
                    "scope": "workspace/app/device",
                    "event": "mock_sensor_read_completed",
                    "retention": "summary-only",
                },
            )
        elif command.capability == "screen.content.set":
            assert content_id is not None
            result = CommandResult(
                command.command_id,
                "completed",
                f"Simulated screen content set to {content_id}.",
                {
                    "scope": "workspace/app/device",
                    "event": "screen_content_changed",
                    "retention": "summary-only",
                },
            )
        self._results[command.command_id] = result
        if not hmac.compare_digest(
            verified_digest, hashlib.sha256(_canonical_command(command)).hexdigest()
        ):
            raise SecurityError("command changed after verification")
        self._result_digests[command.command_id] = verified_digest
        self._audit(command, result.status)
        return result

    def _verify(self, command: Command) -> None:
        now = int(self._now())
        if self.revoked:
            raise SecurityError("device identity is revoked")
        for value, label in (
            (command.device_id, "device_id"),
            (command.workspace_id, "workspace_id"),
            (command.app_id, "app_id"),
            (command.signature, "signature"),
        ):
            if not isinstance(value, str):
                raise SecurityError(f"{label} must be a string")
        for value, label in (
            (command.command_id, "command_id"),
            (command.nonce, "nonce"),
            (command.capability, "capability"),
        ):
            if not isinstance(value, str):
                raise SecurityError(f"{label} must be a string")
            if not IDENTIFIER_PATTERN.fullmatch(value):
                raise SecurityError(f"{label} must be a bounded identifier")
        if (
            not isinstance(command.issued_at, int)
            or isinstance(command.issued_at, bool)
            or not isinstance(command.expires_at, int)
            or isinstance(command.expires_at, bool)
        ):
            raise SecurityError("command timestamps must be integer seconds")
        if (
            command.device_id != self.identity.device_id
            or command.workspace_id != self.identity.workspace_id
            or command.app_id != self.identity.app_id
        ):
            raise SecurityError("command scope does not match device identity")
        if command.capability not in self.policy.capabilities:
            raise SecurityError("capability is not granted")
        if command.expires_at <= command.issued_at:
            raise SecurityError("invalid command lifetime")
        if command.expires_at - command.issued_at > MAX_COMMAND_LIFETIME_SECONDS:
            raise SecurityError("command lifetime exceeds policy")
        if command.issued_at > now + MAX_CLOCK_SKEW_SECONDS:
            raise SecurityError("command issued too far in the future")
        if now > command.expires_at:
            raise SecurityError("command expired")
        expected = hmac.new(
            self.identity.key, _canonical_command(command), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, command.signature):
            raise SecurityError("invalid command signature")
        if command.command_id not in self._results and command.nonce in self._nonces:
            raise SecurityError("nonce replay detected")

    def _audit(self, command: Command, status: str) -> None:
        try:
            canonical = _canonical_command(command)
        except (SecurityError, TypeError, ValueError):
            canonical = json.dumps(
                {
                    "app_id": command.app_id,
                    "capability": command.capability,
                    "command_id": command.command_id,
                    "device_id": command.device_id,
                    "expires_at": command.expires_at,
                    "issued_at": command.issued_at,
                    "nonce": command.nonce,
                    "payload": _plain_json(command.payload),
                    "workspace_id": command.workspace_id,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=lambda value: f"<{type(value).__name__}>",
            ).encode()
        self.audit.append(
            AuditEvent(
                event="device_command",
                command_id=command.command_id,
                workspace_id=self.identity.workspace_id,
                app_id=self.identity.app_id,
                device_id=self.identity.device_id,
                capability=command.capability,
                status=status,
                timestamp=int(self._now()),
                payload_digest=hmac.new(
                    self.identity.key, canonical, hashlib.sha256
                ).hexdigest(),
            )
        )
