"""Server-side Cockpit send foundation (BIF-513 phase 2).

This module intentionally does not expose a web route and does not dispatch to
real gateway/platform adapters.  It provides a pure, process-local send/audit
schema plus a fake dry-run gateway mutation adapter behind default-off gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Mapping, Optional
from uuid import uuid4

from hermes_cli.cockpit import COCKPIT_SCHEMA_VERSION, _redact_text

_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{8,200}$")
_SECRET_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]+\b")
_ALLOWED_STATUSES = {
    "feature_disabled",
    "kill_switch_active",
    "dry_run_recorded",
    "dispatched",
    "dispatch_failed",
    "idempotency_conflict",
    "validation_failed",
    "ambiguous_lane",
    "unknown_lane",
}


def _iter_sensitive_strings(value: Any):
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_sensitive_strings(nested)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for nested in value:
            yield from _iter_sensitive_strings(nested)
        return
    text = str(value).strip()
    if text:
        yield text


def _sensitive_values_for_send(
    *,
    actor_user_id: Optional[str] = None,
    approver_user_id: Optional[str] = None,
    lane: Optional["CockpitSendLane"] = None,
    idempotency_key: Optional[str] = None,
    extra_values: Any = None,
) -> tuple[str, ...]:
    """Collect exact server-only values that regex redaction may not catch.

    Canonical lane targets can contain arbitrary platform IDs, so their values
    must be removed by exact-match redaction before generic pattern redaction.
    """

    values: list[str] = []
    for item in (actor_user_id, approver_user_id, idempotency_key):
        values.extend(_iter_sensitive_strings(item) or ())
    if lane is not None:
        canonical_target = dict(lane.canonical_target or {})
        canonical_target.pop("platform", None)
        values.extend(_iter_sensitive_strings(canonical_target) or ())
    values.extend(_iter_sensitive_strings(extra_values) or ())

    unique: dict[str, None] = {}
    for value in values:
        if value:
            unique[value] = None
    return tuple(sorted(unique, key=len, reverse=True))


def _redact_known_sensitive_text(text: str, known_sensitive_values: tuple[str, ...] = ()) -> str:
    redacted = str(text)
    for value in known_sensitive_values:
        if value:
            redacted = redacted.replace(value, "[id]")
    return redacted


def _redact_send_text(text: str, known_sensitive_values: tuple[str, ...] = ()) -> str:
    return _redact_text(_redact_known_sensitive_text(str(text), known_sensitive_values))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _display_hash(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return "sha256:" + _sha256_text(text)


def _normalize_message_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _safe_preview(normalized_message: str, *, limit: int = 120, known_sensitive_values: tuple[str, ...] = ()) -> str:
    redacted = _redact_known_sensitive_text(normalized_message, known_sensitive_values)
    redacted = _SECRET_TOKEN_RE.sub("[redacted]", redacted)
    redacted = _redact_text(redacted)
    redacted = " ".join(redacted.split())
    if len(redacted) > limit:
        return redacted[: limit - 1].rstrip() + "…"
    return redacted


def _safe_dispatch_result(result: Any, known_sensitive_values: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {"ok": False, "error_code": "invalid_dispatch_result"}
    if result.get("success"):
        message_ref = str(result.get("message_id") or result.get("id") or "")
        payload: dict[str, Any] = {"ok": True}
        if message_ref:
            payload["message_ref_hash"] = _display_hash(_redact_known_sensitive_text(message_ref, known_sensitive_values))
        warnings = result.get("warnings")
        if isinstance(warnings, list):
            payload["warnings"] = [_safe_preview(str(item), known_sensitive_values=known_sensitive_values) for item in warnings[:3]]
        return payload
    error = str(result.get("error") or "dispatch_failed")
    return {"ok": False, "error_code": _safe_preview(error, limit=80, known_sensitive_values=known_sensitive_values)}


@dataclass(frozen=True)
class CockpitSendConfig:
    """Default-off safety gates for the Cockpit send foundation."""

    send_enabled: bool = False
    fake_gateway_enabled: bool = False
    real_gateway_enabled: bool = False
    kill_switch: bool = True
    attachments_enabled: bool = False
    voice_enabled: bool = False


@dataclass(frozen=True)
class CockpitSendLane:
    """Server-only lane descriptor.

    ``canonical_target`` may contain raw platform IDs and is intentionally never
    serialized by the response/audit helpers below.
    """

    alias: str
    label: str
    platform_label: str
    canonical_target: Mapping[str, Any] = field(default_factory=dict)

    def display(self) -> dict[str, str]:
        known_sensitive_values = _sensitive_values_for_send(lane=self)
        return {
            "lane_alias": self.alias,
            "lane_label": _redact_send_text(self.label, known_sensitive_values),
            "platform": _redact_send_text(self.platform_label, known_sensitive_values),
        }


@dataclass
class CockpitSendRecord:
    send_record_id: str
    actor_user_id: str
    lane_alias: str
    lane_label: str
    platform: str
    idempotency_key_hash: str
    message_digest: str
    content_preview_redacted: str
    status: str
    approval_policy: str
    dispatch_count: int
    gateway_request_id: Optional[str]
    dispatch_result: Optional[dict[str, Any]]
    error_code: Optional[str]
    created_at: str
    updated_at: str

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COCKPIT_SCHEMA_VERSION,
            "send_record_id": self.send_record_id,
            "lane_alias": self.lane_alias,
            "lane_label": self.lane_label,
            "platform": self.platform,
            "status": self.status,
            "approval_policy": self.approval_policy,
            "dispatch_count": self.dispatch_count,
            "gateway_request_id": self.gateway_request_id,
            "dispatch_result": self.dispatch_result,
            "error_code": self.error_code,
            "content_preview_redacted": self.content_preview_redacted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class CockpitSendAuditEvent:
    event_type: str
    send_record_id: str
    idempotency_key_hash: str
    actor_user_id: str
    approver_user_id: Optional[str]
    lane_alias: str
    platform: str
    message_digest: str
    content_preview_redacted: str
    approval_policy: str
    status: str
    gateway_request_id: Optional[str]
    dispatch_result: Optional[dict[str, Any]]
    error_code: Optional[str]
    created_at: str
    updated_at: str

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "send_record_id": self.send_record_id,
            "idempotency_key_hash": self.idempotency_key_hash,
            "actor_user_id_hash": _display_hash(self.actor_user_id),
            "approver_user_id_hash": _display_hash(self.approver_user_id),
            "lane_alias": self.lane_alias,
            "platform": self.platform,
            "message_digest": self.message_digest,
            "content_preview_redacted": self.content_preview_redacted,
            "approval_policy": self.approval_policy,
            "status": self.status,
            "gateway_request_id": self.gateway_request_id,
            "dispatch_result": self.dispatch_result,
            "error_code": self.error_code,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class FakeGatewayMutationAdapter:
    """Dry-run gateway mutation adapter that never calls platform dispatch."""

    def __init__(self):
        self.mutations: list[dict[str, Any]] = []

    def record_dry_run_mutation(
        self,
        *,
        send_record_id: str,
        lane: CockpitSendLane,
        message_digest: str,
        known_sensitive_values: tuple[str, ...] = (),
    ) -> dict[str, str]:
        request_id = "fake_gateway_" + _sha256_text(f"{send_record_id}\0{lane.alias}\0{message_digest}")[:16]
        self.mutations.append(
            {
                "gateway_request_id": request_id,
                "send_record_id": send_record_id,
                "lane_alias": lane.alias,
                "lane_label": _redact_send_text(lane.label, known_sensitive_values),
                "platform": _redact_send_text(lane.platform_label, known_sensitive_values),
                "message_digest": message_digest,
                "dry_run": True,
            }
        )
        return {"gateway_request_id": request_id, "status": "dry_run_recorded"}


@dataclass(frozen=True)
class CockpitLaneResolution:
    ok: bool
    status: str
    lane: Optional[CockpitSendLane] = None
    error_code: Optional[str] = None

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "lane": self.lane.display() if self.lane is not None else None,
            "error_code": self.error_code,
        }


class CockpitLaneResolver:
    """Explicit Cockpit send lane resolver; ambiguity fails closed."""

    def __init__(self, *, directory_loader=None):
        self.directory_loader = directory_loader or self._default_directory_loader

    def resolve(self, alias: str) -> CockpitLaneResolution:
        raw_alias = str(alias or "").strip()
        if not raw_alias:
            return CockpitLaneResolution(False, "unknown_lane", error_code="missing_lane_alias")
        candidates = self._matching_discord_hermes_lanes(raw_alias)
        if len(candidates) == 1:
            return CockpitLaneResolution(True, "resolved", lane=candidates[0])
        if len(candidates) > 1:
            return CockpitLaneResolution(False, "ambiguous_lane", error_code="ambiguous_lane_alias")
        return CockpitLaneResolution(False, "unknown_lane", error_code="unknown_lane_alias")

    def list_allowed_lanes(self) -> list[dict[str, str]]:
        lanes = self._matching_discord_hermes_lanes("%discord/hermes")
        if len(lanes) != 1:
            return []
        return [lanes[0].display()]

    def _matching_discord_hermes_lanes(self, alias: str) -> list[CockpitSendLane]:
        normalized = _normalize_lane_alias(alias)
        allowed_aliases = {"%discord/hermes", "discord:#hermes", "discord:hermes", "#hermes", "hermes"}
        if normalized not in allowed_aliases:
            return []
        directory = self.directory_loader() or {}
        channels = (directory.get("platforms") or {}).get("discord") or []
        matches: list[CockpitSendLane] = []
        for channel in channels:
            name = str(channel.get("name") or "").strip().lower().lstrip("#")
            if name != "hermes":
                continue
            chat_id = str(channel.get("id") or "").strip()
            if not chat_id:
                continue
            canonical_target: dict[str, Any] = {"platform": "discord", "chat_id": chat_id}
            thread_id = str(channel.get("thread_id") or "").strip()
            if thread_id:
                canonical_target["thread_id"] = thread_id
            matches.append(
                CockpitSendLane(
                    alias="%discord/hermes",
                    label="Discord #hermes",
                    platform_label="discord",
                    canonical_target=canonical_target,
                )
            )
        unique: dict[tuple[str, Optional[str]], CockpitSendLane] = {}
        for lane in matches:
            unique[(str(lane.canonical_target.get("chat_id")), lane.canonical_target.get("thread_id"))] = lane
        return list(unique.values())

    @staticmethod
    def _default_directory_loader() -> dict[str, Any]:
        from gateway.channel_directory import load_directory

        return load_directory()


def _normalize_lane_alias(alias: str) -> str:
    text = str(alias or "").strip().lower()
    if text.startswith("discord:"):
        prefix, rest = text.split(":", 1)
        return f"{prefix}:{rest.strip()}"
    return text


class CockpitGatewayDeliveryAdapter:
    """Production delivery adapter using the configured gateway direct send path."""

    def dispatch(self, *, lane: CockpitSendLane, message_text: str, idempotency_key: str) -> dict[str, Any]:
        target = lane.canonical_target
        if str(target.get("platform") or "").lower() != "discord":
            return {"error": "unsupported_platform"}
        chat_id = str(target.get("chat_id") or "").strip()
        if not chat_id:
            return {"error": "missing_chat_id"}
        thread_id = str(target.get("thread_id") or "").strip() or None
        try:
            from gateway.config import Platform, load_gateway_config
            from model_tools import _run_async
            from tools.send_message_tool import _send_to_platform

            config = load_gateway_config()
            platform = Platform("discord")
            pconfig = config.platforms.get(platform)
            if not pconfig:
                from gateway.config import PlatformConfig
                pconfig = PlatformConfig(enabled=True)
            if not (pconfig.token or "").strip():
                import subprocess
                try:
                    pconfig.token = subprocess.check_output(
                        ["bash", "/Users/marco/.local/bin/get_credential.sh", "discord_bot_token"],
                        text=True,
                        timeout=10,
                    ).strip()
                except Exception:
                    pconfig.token = None
            if not pconfig.enabled:
                pconfig.enabled = bool((pconfig.token or "").strip())
            if not pconfig.enabled or not (pconfig.token or "").strip():
                return {"error": "platform_not_configured"}
            return _run_async(_send_to_platform(platform, pconfig, chat_id, message_text, thread_id=thread_id))
        except Exception as exc:
            return {"error": _redact_send_text(str(exc), _sensitive_values_for_send(lane=lane, idempotency_key=idempotency_key))}


class CockpitSendStore:
    """In-memory idempotency/audit store for server-side send records."""

    def __init__(self):
        self.records_by_id: dict[str, CockpitSendRecord] = {}
        self.audit_events: list[CockpitSendAuditEvent] = []
        self._idempotency_index: dict[tuple[str, str, str], str] = {}

    def lookup(self, *, actor_user_id: str, lane_alias: str, idempotency_key_hash: str) -> Optional[CockpitSendRecord]:
        record_id = self._idempotency_index.get((actor_user_id, lane_alias, idempotency_key_hash))
        if record_id is None:
            return None
        return self.records_by_id[record_id]

    def add_record(self, record: CockpitSendRecord) -> None:
        self.records_by_id[record.send_record_id] = record
        self._idempotency_index[(record.actor_user_id, record.lane_alias, record.idempotency_key_hash)] = record.send_record_id

    def add_audit(self, event: CockpitSendAuditEvent) -> None:
        self.audit_events.append(event)


class CockpitSendService:
    """Server-side send foundation with default-off fake-only dispatch intent."""

    def __init__(
        self,
        *,
        config: Optional[CockpitSendConfig] = None,
        store: Optional[CockpitSendStore] = None,
        gateway_adapter: Optional[FakeGatewayMutationAdapter] = None,
    ):
        self.config = config or CockpitSendConfig()
        self.store = store or CockpitSendStore()
        self.gateway_adapter = gateway_adapter or FakeGatewayMutationAdapter()

    def submit_send_intent(
        self,
        *,
        actor_user_id: str,
        lane: CockpitSendLane,
        idempotency_key: str,
        message_text: str,
    ) -> dict[str, Any]:
        if not self.config.send_enabled:
            return self._disabled_response("feature_disabled")
        if self.config.kill_switch:
            return self._disabled_response("kill_switch_active")
        if not self.config.fake_gateway_enabled and not self.config.real_gateway_enabled:
            return self._disabled_response("feature_disabled")

        normalized_message = _normalize_message_text(message_text)
        validation_error = self._validation_error(actor_user_id, lane, idempotency_key, normalized_message)
        if validation_error is not None:
            return validation_error

        known_sensitive_values = _sensitive_values_for_send(
            actor_user_id=str(actor_user_id),
            lane=lane,
            idempotency_key=str(idempotency_key),
        )
        idempotency_key_hash = _sha256_text(idempotency_key)
        message_digest = _sha256_text(normalized_message)
        existing = self.store.lookup(
            actor_user_id=actor_user_id,
            lane_alias=lane.alias,
            idempotency_key_hash=idempotency_key_hash,
        )
        if existing is not None:
            if existing.message_digest == message_digest:
                return {"ok": True, "status": existing.status, "record": existing.to_display_dict(), "idempotent_replay": True}
            return {
                "ok": False,
                "status": "idempotency_conflict",
                "error_code": "idempotency_key_conflict",
                "record": existing.to_display_dict(),
                "idempotent_replay": False,
            }

        now = _utc_now()
        record = CockpitSendRecord(
            send_record_id="cockpit_send_" + uuid4().hex,
            actor_user_id=str(actor_user_id),
            lane_alias=lane.alias,
            lane_label=_redact_send_text(lane.label, known_sensitive_values),
            platform=_redact_send_text(lane.platform_label, known_sensitive_values),
            idempotency_key_hash=idempotency_key_hash,
            message_digest=message_digest,
            content_preview_redacted=_safe_preview(normalized_message, known_sensitive_values=known_sensitive_values),
            status="dry_run_recorded",
            approval_policy="required",
            dispatch_count=0,
            gateway_request_id=None,
            dispatch_result=None,
            error_code=None,
            created_at=now,
            updated_at=now,
        )
        self.store.add_audit(self._audit_event(record, "cockpit_send_requested"))
        if self.config.real_gateway_enabled:
            dispatch = self.gateway_adapter.dispatch(
                lane=lane,
                message_text=normalized_message,
                idempotency_key=str(idempotency_key),
            )
            record.dispatch_count = 1
            record.dispatch_result = _safe_dispatch_result(dispatch, known_sensitive_values)
            if isinstance(dispatch, Mapping) and dispatch.get("success"):
                record.status = "dispatched"
                message_ref = str(dispatch.get("message_id") or dispatch.get("id") or "")
                record.gateway_request_id = _display_hash(message_ref)
                record.error_code = None
            else:
                record.status = "dispatch_failed"
                record.error_code = _redact_send_text(
                    str((dispatch or {}).get("error") if isinstance(dispatch, Mapping) else "dispatch_failed"),
                    known_sensitive_values,
                )
            record.updated_at = _utc_now()
        else:
            mutation = self.gateway_adapter.record_dry_run_mutation(
                send_record_id=record.send_record_id,
                lane=lane,
                message_digest=message_digest,
                known_sensitive_values=known_sensitive_values,
            )
            record.dispatch_count = 1
            record.gateway_request_id = mutation["gateway_request_id"]
            record.dispatch_result = {"ok": True, "mode": "dry_run"}
            record.updated_at = _utc_now()

        self.store.add_record(record)
        self.store.add_audit(self._audit_event(record, "cockpit_send_dispatched"))
        return {"ok": record.status == "dispatched" or record.status == "dry_run_recorded", "status": record.status, "record": record.to_display_dict(), "idempotent_replay": False}

    def _disabled_response(self, status: str) -> dict[str, Any]:
        if status not in _ALLOWED_STATUSES:
            status = "feature_disabled"
        return {"ok": False, "status": status, "error_code": status, "record": None}

    def _validation_error(self, actor_user_id: str, lane: CockpitSendLane, idempotency_key: str, normalized_message: str) -> Optional[dict[str, Any]]:
        if not str(actor_user_id or "").strip():
            return {"ok": False, "status": "validation_failed", "error_code": "missing_actor_user_id", "record": None}
        if not str(lane.alias or "").strip() or not str(lane.label or "").strip():
            return {"ok": False, "status": "validation_failed", "error_code": "missing_lane_alias", "record": None}
        if not _IDEMPOTENCY_KEY_RE.match(str(idempotency_key or "")):
            return {"ok": False, "status": "validation_failed", "error_code": "invalid_idempotency_key", "record": None}
        if not normalized_message:
            return {"ok": False, "status": "validation_failed", "error_code": "empty_message", "record": None}
        if "MEDIA:" in normalized_message and not self.config.attachments_enabled:
            return {"ok": False, "status": "validation_failed", "error_code": "attachments_disabled", "record": None}
        if "VOICE:" in normalized_message and not self.config.voice_enabled:
            return {"ok": False, "status": "validation_failed", "error_code": "voice_disabled", "record": None}
        return None

    def _audit_event(self, record: CockpitSendRecord, event_type: str) -> CockpitSendAuditEvent:
        return CockpitSendAuditEvent(
            event_type=event_type,
            send_record_id=record.send_record_id,
            idempotency_key_hash=record.idempotency_key_hash,
            actor_user_id=record.actor_user_id,
            approver_user_id=None,
            lane_alias=record.lane_alias,
            platform=record.platform,
            message_digest=record.message_digest,
            content_preview_redacted=record.content_preview_redacted,
            approval_policy=record.approval_policy,
            status=record.status,
            gateway_request_id=record.gateway_request_id,
            dispatch_result=record.dispatch_result,
            error_code=record.error_code,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


__all__ = [
    "CockpitSendAuditEvent",
    "CockpitSendConfig",
    "CockpitGatewayDeliveryAdapter",
    "CockpitLaneResolution",
    "CockpitLaneResolver",
    "CockpitSendLane",
    "CockpitSendRecord",
    "CockpitSendService",
    "CockpitSendStore",
    "FakeGatewayMutationAdapter",
]
