"""Typed command invocation, policy, binding, and settlement contracts.

This module is transport-free.  It is the bounded PR-2 narrow waist between a
versioned command catalog and every surface-specific renderer.  Catalog
construction remains a separate authority; this module accepts one immutable
snapshot, enforces its revision and policy settlement, invokes exactly one
binding, and returns one structured result.

The catalog adapter is deliberately duck-typed so the PR-1 schema can land
independently.  A snapshot must expose ``revision`` plus either
``resolve_command_id(command_id)`` or a ``commands`` collection.  A command
row must expose ``command_id`` and may expose ``execution_owner`` and
``handler_id`` for binding-attestation checks.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

COMMAND_RESULT_STATUSES = frozenset(
    {
        "ok",
        "error",
        "unknown_command",
        "unavailable",
        "confirmation_required",
        "deferred",
        "catalog_stale",
        "indeterminate",
        "send_to_agent",
        "client_action",
    }
)


def _require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be blank")
    return text


def _json_copy(value: Any, field_name: str) -> Any:
    """Return a detached JSON-compatible copy or fail the contract."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, Any] | None, field_name: str) -> Mapping[str, Any]:
    copied = _json_copy(dict(value or {}), field_name)
    return _freeze_json(copied)


def _freeze_mapping_sequence(
    value: Iterable[Mapping[str, Any]] | None,
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(_freeze_mapping(item, field_name) for item in (value or ()))


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    """One normalized command attempt bound to an exact catalog revision."""

    request_id: str
    catalog_revision: str
    command_id: str
    entered_name: str
    raw_input: str
    raw_arguments: str = ""
    parsed_arguments: Mapping[str, Any] = field(default_factory=dict)
    surface: str = ""
    platform: str = ""
    actor: Mapping[str, Any] = field(default_factory=dict)
    profile_home: str = ""
    cwd: str = ""
    session_id: str = ""
    chat_id: str = ""
    channel_id: str = ""
    thread_id: str = ""
    source_id: str = ""
    locale: str = ""
    attachments: tuple[Mapping[str, Any], ...] = ()
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    retry_of: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_text(self.request_id, "request_id"))
        object.__setattr__(
            self,
            "catalog_revision",
            _require_text(self.catalog_revision, "catalog_revision"),
        )
        object.__setattr__(self, "command_id", _require_text(self.command_id, "command_id"))
        object.__setattr__(self, "entered_name", _require_text(self.entered_name, "entered_name"))
        object.__setattr__(self, "surface", _require_text(self.surface, "surface"))
        object.__setattr__(
            self,
            "parsed_arguments",
            _freeze_mapping(self.parsed_arguments, "parsed_arguments"),
        )
        object.__setattr__(self, "actor", _freeze_mapping(self.actor, "actor"))
        object.__setattr__(
            self,
            "attachments",
            _freeze_mapping_sequence(self.attachments, "attachments"),
        )
        object.__setattr__(
            self,
            "capabilities",
            _freeze_mapping(self.capabilities, "capabilities"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CommandInvocation":
        if not isinstance(value, Mapping):
            raise ValueError("invocation must be a mapping")
        return cls(
            request_id=value.get("request_id", ""),
            catalog_revision=value.get("catalog_revision", ""),
            command_id=value.get("command_id", ""),
            entered_name=value.get("entered_name", ""),
            raw_input=str(value.get("raw_input") or ""),
            raw_arguments=str(value.get("raw_arguments") or ""),
            parsed_arguments=value.get("parsed_arguments") or {},
            surface=value.get("surface", ""),
            platform=str(value.get("platform") or ""),
            actor=value.get("actor") or {},
            profile_home=str(value.get("profile_home") or ""),
            cwd=str(value.get("cwd") or ""),
            session_id=str(value.get("session_id") or ""),
            chat_id=str(value.get("chat_id") or ""),
            channel_id=str(value.get("channel_id") or ""),
            thread_id=str(value.get("thread_id") or ""),
            source_id=str(value.get("source_id") or ""),
            locale=str(value.get("locale") or ""),
            attachments=tuple(value.get("attachments") or ()),
            capabilities=value.get("capabilities") or {},
            idempotency_key=str(value.get("idempotency_key") or ""),
            retry_of=str(value.get("retry_of") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "catalog_revision": self.catalog_revision,
            "command_id": self.command_id,
            "entered_name": self.entered_name,
            "raw_input": self.raw_input,
            "raw_arguments": self.raw_arguments,
            "parsed_arguments": _thaw_json(self.parsed_arguments),
            "surface": self.surface,
            "platform": self.platform,
            "actor": _thaw_json(self.actor),
            "profile_home": self.profile_home,
            "cwd": self.cwd,
            "session_id": self.session_id,
            "chat_id": self.chat_id,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "source_id": self.source_id,
            "locale": self.locale,
            "attachments": [_thaw_json(item) for item in self.attachments],
            "capabilities": _thaw_json(self.capabilities),
            "idempotency_key": self.idempotency_key,
            "retry_of": self.retry_of,
        }

    def effect_fingerprint(self) -> str:
        """Bind an idempotency key to the effect, not the transport request id."""
        payload = self.to_dict()
        payload.pop("request_id", None)
        payload.pop("idempotency_key", None)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Typed terminal or continuation settlement returned by a command binding."""

    status: str
    command_id: str
    content: tuple[Mapping[str, Any], ...] = ()
    error: Mapping[str, Any] | None = None
    unavailable_reason: str = ""
    client_action: Mapping[str, Any] | None = None
    session_mutations: Mapping[str, Any] = field(default_factory=dict)
    receipts: tuple[Mapping[str, Any], ...] = ()
    catalog_revision: str = ""
    current_catalog_revision: str = ""
    catalog_invalidation: bool = False
    ephemeral: bool | None = None
    visibility: str = ""

    def __post_init__(self) -> None:
        status = _require_text(self.status, "status")
        if status not in COMMAND_RESULT_STATUSES:
            raise ValueError(f"unsupported command result status: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "command_id", _require_text(self.command_id, "command_id"))
        object.__setattr__(self, "content", _freeze_mapping_sequence(self.content, "content"))
        object.__setattr__(
            self,
            "error",
            None if self.error is None else _freeze_mapping(self.error, "error"),
        )
        object.__setattr__(
            self,
            "client_action",
            None
            if self.client_action is None
            else _freeze_mapping(self.client_action, "client_action"),
        )
        object.__setattr__(
            self,
            "session_mutations",
            _freeze_mapping(self.session_mutations, "session_mutations"),
        )
        object.__setattr__(
            self,
            "receipts",
            _freeze_mapping_sequence(self.receipts, "receipts"),
        )

    @classmethod
    def error_result(
        cls,
        command_id: str,
        code: str,
        message: str,
        *,
        status: str = "error",
        catalog_revision: str = "",
    ) -> "CommandResult":
        return cls(
            status=status,
            command_id=command_id,
            error={"code": code, "message": message},
            catalog_revision=catalog_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "command_id": self.command_id,
            "content": [_thaw_json(item) for item in self.content],
            "error": None if self.error is None else _thaw_json(self.error),
            "unavailable_reason": self.unavailable_reason or None,
            "client_action": (
                None
                if self.client_action is None
                else _thaw_json(self.client_action)
            ),
            "session_mutations": _thaw_json(self.session_mutations),
            "receipts": [_thaw_json(item) for item in self.receipts],
            "catalog_revision": self.catalog_revision or None,
            "current_catalog_revision": self.current_catalog_revision or None,
            "catalog_invalidation": self.catalog_invalidation,
            "ephemeral": self.ephemeral,
            "visibility": self.visibility or None,
        }


@dataclass(frozen=True, slots=True)
class CommandBinding:
    """One executable owner for one stable command identity."""

    command_id: str
    execution_owner: str
    handler_id: str
    handler: Callable[[CommandInvocation, Any], CommandResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _require_text(self.command_id, "command_id"))
        object.__setattr__(
            self,
            "execution_owner",
            _require_text(self.execution_owner, "execution_owner"),
        )
        object.__setattr__(self, "handler_id", _require_text(self.handler_id, "handler_id"))
        if not callable(self.handler):
            raise ValueError("handler must be callable")


@dataclass(frozen=True, slots=True)
class CommandPolicyState:
    """Fully resolved policy facts enforced in one deterministic order."""

    available: bool = True
    unavailable_reason: str = ""
    authorized: bool = True
    authorization_reason: str = ""
    confirmation_required: bool = False
    confirmation: Mapping[str, Any] = field(default_factory=dict)
    mutation_allowed: bool = True
    mutation_reason: str = ""
    busy_allowed: bool = True
    busy_reason: str = ""
    live_session_satisfied: bool = True
    live_session_reason: str = ""
    idempotency_allowed: bool = True
    idempotency_reason: str = ""
    retry_allowed: bool = True
    retry_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confirmation",
            _freeze_mapping(self.confirmation, "confirmation"),
        )


class CatalogSnapshot(Protocol):
    revision: str


CatalogProvider: TypeAlias = Callable[[CommandInvocation], CatalogSnapshot]
PolicyResolver: TypeAlias = Callable[[CommandInvocation, Any], CommandPolicyState]


def catalog_revision(snapshot: Any) -> str:
    return _require_text(getattr(snapshot, "revision", ""), "catalog revision")


def catalog_to_dict(snapshot: Any) -> dict[str, Any]:
    to_dict = getattr(snapshot, "to_dict", None)
    if not callable(to_dict):
        raise ValueError("catalog snapshot must expose to_dict()")
    value = to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("catalog to_dict() must return a mapping")
    return _json_copy(dict(value), "catalog")


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def resolve_catalog_command_id(snapshot: Any, command_id: str) -> Any | None:
    resolver = getattr(snapshot, "resolve_command_id", None)
    if callable(resolver):
        return resolver(command_id)

    commands = getattr(snapshot, "commands", ())
    if isinstance(commands, Mapping):
        direct = commands.get(command_id)
        if direct is not None:
            return direct
        for key, row in commands.items():
            if str(key).casefold() == command_id.casefold():
                return row
        return None

    for row in commands or ():
        value = str(_row_value(row, "command_id", "") or "")
        if value.casefold() == command_id.casefold():
            return row
    return None


class CommandReceiptStore(Protocol):
    def begin(
        self,
        key: str,
        fingerprint: str,
    ) -> tuple[str, CommandResult | None]: ...

    def settle(self, key: str, fingerprint: str, result: CommandResult) -> None: ...


@dataclass(slots=True)
class _ReceiptRecord:
    fingerprint: str
    result: CommandResult | None = None


class InMemoryCommandReceiptStore:
    """Process-local idempotency fence with typed replay/conflict outcomes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, _ReceiptRecord] = {}

    def begin(
        self,
        key: str,
        fingerprint: str,
    ) -> tuple[str, CommandResult | None]:
        normalized = _require_text(key, "idempotency_key")
        with self._lock:
            record = self._records.get(normalized)
            if record is None:
                self._records[normalized] = _ReceiptRecord(fingerprint=fingerprint)
                return "new", None
            if record.fingerprint != fingerprint:
                return "conflict", None
            if record.result is None:
                return "pending", None
            return "replay", record.result

    def settle(self, key: str, fingerprint: str, result: CommandResult) -> None:
        with self._lock:
            record = self._records.get(key)
            if record is None or record.fingerprint != fingerprint:
                raise RuntimeError("idempotency reservation changed before settlement")
            record.result = result


class CommandDispatcher:
    """Resolve, gate, invoke, and settle one command attempt."""

    def __init__(
        self,
        *,
        catalog_provider: CatalogProvider,
        policy_resolver: PolicyResolver,
        bindings: Iterable[CommandBinding],
        receipt_store: CommandReceiptStore | None = None,
    ) -> None:
        if not callable(catalog_provider):
            raise ValueError("catalog_provider must be callable")
        if not callable(policy_resolver):
            raise ValueError("policy_resolver must be callable")
        self._catalog_provider = catalog_provider
        self._policy_resolver = policy_resolver
        self._receipt_store = receipt_store
        self._bindings: dict[str, CommandBinding] = {}
        for binding in bindings:
            key = binding.command_id.casefold()
            if key in self._bindings:
                raise ValueError(f"duplicate command binding: {binding.command_id}")
            self._bindings[key] = binding

    def dispatch(self, invocation: CommandInvocation) -> CommandResult:
        try:
            snapshot = self._catalog_provider(invocation)
            current_revision = catalog_revision(snapshot)
        except Exception as exc:
            return CommandResult.error_result(
                invocation.command_id,
                "catalog_unavailable",
                str(exc)[:500] or type(exc).__name__,
            )

        if invocation.catalog_revision != current_revision:
            return CommandResult(
                status="catalog_stale",
                command_id=invocation.command_id,
                catalog_revision=invocation.catalog_revision,
                current_catalog_revision=current_revision,
                catalog_invalidation=True,
            )

        command = resolve_catalog_command_id(snapshot, invocation.command_id)
        if command is None:
            return CommandResult(
                status="unknown_command",
                command_id=invocation.command_id,
                catalog_revision=current_revision,
            )

        resolved_id = str(_row_value(command, "command_id", "") or "").strip()
        if not resolved_id or resolved_id.casefold() != invocation.command_id.casefold():
            return CommandResult.error_result(
                invocation.command_id,
                "catalog_identity_mismatch",
                "catalog resolved a different command identity",
                catalog_revision=current_revision,
            )

        try:
            policy = self._policy_resolver(invocation, command)
        except Exception as exc:
            return CommandResult.error_result(
                invocation.command_id,
                "policy_resolution_failed",
                str(exc)[:500] or type(exc).__name__,
                catalog_revision=current_revision,
            )
        if not isinstance(policy, CommandPolicyState):
            return CommandResult.error_result(
                invocation.command_id,
                "invalid_policy_state",
                "policy resolver must return CommandPolicyState",
                catalog_revision=current_revision,
            )

        refusal = self._enforce_policy(invocation, policy, current_revision)
        if refusal is not None:
            return refusal

        binding = self._bindings.get(invocation.command_id.casefold())
        if binding is None:
            return CommandResult(
                status="unavailable",
                command_id=invocation.command_id,
                unavailable_reason="binding_unavailable",
                catalog_revision=current_revision,
            )

        owner = str(_row_value(command, "execution_owner", "") or "").strip()
        handler_id = str(_row_value(command, "handler_id", "") or "").strip()
        if owner and owner != binding.execution_owner:
            return CommandResult.error_result(
                invocation.command_id,
                "execution_owner_mismatch",
                "catalog and binding execution owners differ",
                catalog_revision=current_revision,
            )
        if handler_id and handler_id != binding.handler_id:
            return CommandResult.error_result(
                invocation.command_id,
                "handler_binding_mismatch",
                "catalog and binding handler ids differ",
                catalog_revision=current_revision,
            )

        reservation: tuple[str, str] | None = None
        if invocation.idempotency_key:
            if self._receipt_store is None:
                return CommandResult.error_result(
                    invocation.command_id,
                    "idempotency_store_unavailable",
                    "idempotent invocation cannot execute without a receipt store",
                    catalog_revision=current_revision,
                )
            fingerprint = invocation.effect_fingerprint()
            verdict, prior = self._receipt_store.begin(
                invocation.idempotency_key,
                fingerprint,
            )
            if verdict == "conflict":
                return CommandResult.error_result(
                    invocation.command_id,
                    "idempotency_conflict",
                    "idempotency key is already bound to a different effect",
                    catalog_revision=current_revision,
                )
            if verdict == "pending":
                return CommandResult(
                    status="deferred",
                    command_id=invocation.command_id,
                    error={
                        "code": "idempotency_pending",
                        "message": "an equivalent invocation is still executing",
                    },
                    catalog_revision=current_revision,
                )
            if verdict == "replay" and prior is not None:
                return prior
            reservation = (invocation.idempotency_key, fingerprint)

        try:
            result = binding.handler(invocation, command)
        except Exception as exc:
            result = CommandResult.error_result(
                invocation.command_id,
                "handler_error",
                str(exc)[:500] or type(exc).__name__,
                catalog_revision=current_revision,
            )

        if not isinstance(result, CommandResult):
            result = CommandResult.error_result(
                invocation.command_id,
                "invalid_handler_result",
                "command handlers must return CommandResult",
                catalog_revision=current_revision,
            )
        elif result.command_id.casefold() != invocation.command_id.casefold():
            result = CommandResult.error_result(
                invocation.command_id,
                "result_identity_mismatch",
                "handler settled a different command identity",
                catalog_revision=current_revision,
            )
        elif not result.catalog_revision:
            result = replace(result, catalog_revision=current_revision)

        if reservation is not None and self._receipt_store is not None:
            try:
                self._receipt_store.settle(*reservation, result)
            except Exception as exc:
                return CommandResult.error_result(
                    invocation.command_id,
                    "receipt_settlement_failed",
                    str(exc)[:500] or type(exc).__name__,
                    status="indeterminate",
                    catalog_revision=current_revision,
                )
        return result

    @staticmethod
    def _enforce_policy(
        invocation: CommandInvocation,
        policy: CommandPolicyState,
        revision: str,
    ) -> CommandResult | None:
        if not policy.available:
            return CommandResult(
                status="unavailable",
                command_id=invocation.command_id,
                unavailable_reason=policy.unavailable_reason or "unavailable",
                catalog_revision=revision,
            )
        if not policy.authorized:
            return CommandResult.error_result(
                invocation.command_id,
                "unauthorized",
                policy.authorization_reason or "command is not authorized",
                catalog_revision=revision,
            )
        if policy.confirmation_required:
            return CommandResult(
                status="confirmation_required",
                command_id=invocation.command_id,
                client_action=dict(policy.confirmation),
                catalog_revision=revision,
            )
        if not policy.mutation_allowed:
            return CommandResult.error_result(
                invocation.command_id,
                "mutation_refused",
                policy.mutation_reason or "command mutation is not authorized",
                catalog_revision=revision,
            )
        if not policy.busy_allowed:
            return CommandResult(
                status="unavailable",
                command_id=invocation.command_id,
                unavailable_reason=policy.busy_reason or "session_busy",
                catalog_revision=revision,
            )
        if not policy.live_session_satisfied:
            return CommandResult(
                status="unavailable",
                command_id=invocation.command_id,
                unavailable_reason=(
                    policy.live_session_reason or "live_session_required"
                ),
                catalog_revision=revision,
            )
        if not policy.idempotency_allowed:
            return CommandResult.error_result(
                invocation.command_id,
                "idempotency_refused",
                policy.idempotency_reason or "idempotency policy refused invocation",
                catalog_revision=revision,
            )
        if invocation.retry_of and not policy.retry_allowed:
            return CommandResult.error_result(
                invocation.command_id,
                "retry_refused",
                policy.retry_reason or "command is not retry-safe",
                catalog_revision=revision,
            )
        return None
