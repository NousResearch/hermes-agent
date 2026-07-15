"""Passive Canonical Brain audit bridge for gateway events.

The bridge is intentionally fail-soft and metadata-only.  It observes the
gateway hot path and appends canonical event envelopes without changing
delivery, routing, or model behavior.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

AUDIT_EVENT_TYPES = {
    "discord.inbound.received",
    "assistant.status",
    "outbound.intent.recorded",
    "outbound.receipt.recorded",
    "outbound.delivery_error.recorded",
    "runtime.lease.check.allowed",
    "runtime.lease.check.failed",
}

_AUDIT_NAMESPACE = uuid.UUID("77dd7ec4-80ed-4fb9-97b3-2f9c7b13c8d5")
_METADATA_KEY = "_canonical_brain_audit"
_BRIDGE_CACHE: Optional["CanonicalBrainAuditBridge"] = None
_BRIDGE_CACHE_LOADED = False


@dataclass(frozen=True)
class CanonicalBrainAuditConfig:
    enabled: bool = False
    backend: str = "jsonl"
    jsonl_path: Optional[Path] = None
    runtime_id: str = "hermes-gateway"
    role: str = "gateway-runtime"
    host: str = socket.gethostname()
    fail_soft: bool = True
    metadata_only: bool = True
    hash_refs: bool = True
    runtime_lease_enforcement_enabled: bool = False
    runtime_lease_send_path_blocking_enabled: bool = False


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _config_yaml() -> dict[str, Any]:
    path = get_hermes_home() / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("canonical brain audit config load failed: %s", exc)
        return {}


def load_audit_config() -> CanonicalBrainAuditConfig:
    root = _config_yaml()
    section = root.get("canonical_brain", {})
    if not isinstance(section, dict):
        section = {}
    bridge = section.get("audit_bridge", {})
    if not isinstance(bridge, dict):
        bridge = {}
    runtime_lease = section.get("runtime_lease", {})
    if not isinstance(runtime_lease, dict):
        runtime_lease = {}

    raw_path = bridge.get("jsonl_path")
    jsonl_path = Path(raw_path).expanduser() if raw_path else None

    return CanonicalBrainAuditConfig(
        enabled=_coerce_bool(bridge.get("enabled"), False),
        backend=str(bridge.get("backend") or "jsonl").strip().lower(),
        jsonl_path=jsonl_path,
        runtime_id=str(bridge.get("runtime_id") or "hermes-gateway"),
        role=str(bridge.get("role") or "gateway-runtime"),
        host=str(bridge.get("host") or socket.gethostname()),
        fail_soft=_coerce_bool(bridge.get("fail_soft"), True),
        metadata_only=_coerce_bool(bridge.get("metadata_only"), True),
        hash_refs=_coerce_bool(bridge.get("hash_refs"), True),
        runtime_lease_enforcement_enabled=_coerce_bool(runtime_lease.get("enforcement_enabled"), False),
        runtime_lease_send_path_blocking_enabled=_coerce_bool(runtime_lease.get("send_path_blocking_enabled"), False),
    )


def reset_audit_bridge_cache() -> None:
    global _BRIDGE_CACHE, _BRIDGE_CACHE_LOADED
    _BRIDGE_CACHE = None
    _BRIDGE_CACHE_LOADED = False


def get_audit_bridge() -> Optional["CanonicalBrainAuditBridge"]:
    global _BRIDGE_CACHE, _BRIDGE_CACHE_LOADED
    if _BRIDGE_CACHE_LOADED:
        return _BRIDGE_CACHE
    _BRIDGE_CACHE_LOADED = True
    try:
        cfg = load_audit_config()
        if not cfg.enabled:
            return None
        _BRIDGE_CACHE = CanonicalBrainAuditBridge(cfg)
    except Exception as exc:
        logger.debug("canonical brain audit bridge disabled after init failure: %s", exc)
        _BRIDGE_CACHE = None
    return _BRIDGE_CACHE


def _stable_hash(value: Any, prefix: str) -> str:
    raw = str(value or "")
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}:sha256:{digest}"


def _platform_name(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _event_id(event_type: str, idempotency_key: str) -> str:
    return str(uuid.uuid5(_AUDIT_NAMESPACE, f"{event_type}:{idempotency_key}"))


def _schedule(coro: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(coro)

    def _done(done: asyncio.Task) -> None:
        try:
            done.result()
        except Exception as exc:
            logger.debug("canonical brain audit write failed: %s", exc)

    task.add_done_callback(_done)


class CanonicalBrainAuditBridge:
    def __init__(self, config: CanonicalBrainAuditConfig):
        self.config = config

    @property
    def jsonl_path(self) -> Path:
        if self.config.jsonl_path is not None:
            return self.config.jsonl_path
        return get_hermes_home() / "canonical-brain-audit-events.jsonl"

    async def append(self, envelope: dict[str, Any]) -> None:
        if self.config.backend != "jsonl":
            if self.config.fail_soft:
                logger.debug("canonical brain audit backend %r is not enabled", self.config.backend)
                return
            raise RuntimeError(f"Unsupported canonical brain audit backend: {self.config.backend}")
        await asyncio.to_thread(self._append_jsonl, envelope)

    def _append_jsonl(self, envelope: dict[str, Any]) -> None:
        path = self.jsonl_path
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(envelope, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def ref(self, value: Any, prefix: str) -> str:
        if not self.config.hash_refs:
            return f"{prefix}:{value}"
        return _stable_hash(value, prefix)

    def _source_ref(self, source: Any) -> dict[str, Any]:
        platform = _platform_name(getattr(source, "platform", None)) or "unknown"
        chat_id = getattr(source, "chat_id", None)
        thread_id = getattr(source, "thread_id", None) or chat_id
        return {
            "system": "hermes-gateway",
            "channel": platform if platform in {"discord", "operator_note", "gcp_runtime", "manual_evidence", "synthetic"} else "operator_note",
            "thread_ref": self.ref(thread_id, "thread") if thread_id else None,
        }

    def _subject(self, source: Any) -> dict[str, list[str]]:
        people: list[str] = []
        threads: list[str] = []
        user_id = getattr(source, "user_id", None)
        if user_id:
            people.append(self.ref(user_id, "person"))
        chat_id = getattr(source, "chat_id", None)
        thread_id = getattr(source, "thread_id", None)
        if chat_id:
            threads.append(self.ref(chat_id, "channel"))
        if thread_id and thread_id != chat_id:
            threads.append(self.ref(thread_id, "thread"))
        return {"people": people, "threads": threads}

    def _case(self, session_key: Optional[str]) -> dict[str, str]:
        session_ref = self.ref(session_key or "unknown", "session")
        return {
            "case_id": f"case:gateway-audit:{session_ref.rsplit(':', 1)[-1]}",
            "title": "Hermes gateway audit trail",
            "business_domain": "ops",
        }

    def _runtime_payload(self) -> dict[str, str]:
        return {
            "runtime_id": self.config.runtime_id,
            "role": self.config.role,
            "host": self.config.host,
        }

    def envelope(
        self,
        *,
        event_type: str,
        idempotency_key: str,
        source: Any,
        session_key: Optional[str],
        actor_kind: str,
        actor_ref: str,
        status_state: str,
        summary: str,
        payload: dict[str, Any],
        occurred_at: Optional[str] = None,
        case_ref: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        safe_payload = dict(payload)
        safe_payload["metadata_only"] = True
        safe_payload["source_runtime"] = self._runtime_payload()
        return {
            "schema_version": "canonical_event.v1",
            "event_id": _event_id(event_type, idempotency_key),
            "event_type": event_type,
            "occurred_at": occurred_at or _now_iso(),
            "source": self._source_ref(source),
            "actor": {"kind": actor_kind, "ref": actor_ref},
            "case": case_ref or self._case(session_key),
            "subject": self._subject(source),
            "evidence": [],
            "decision": {"decision_id": None, "state": "none", "summary": summary},
            "status": {"state": status_state, "blocker": None, "owner": self.config.runtime_id},
            "next_action": {
                "action_id": None,
                "kind": "none",
                "summary": "audit-only bridge recorded metadata; no action executed",
                "allowed_without_approval": True,
            },
            "safety": {
                "data_class": "non_secret_operational",
                "outbound_allowed": False,
                "mutation_allowed": False,
                "customer_data_used": False,
            },
            "payload": safe_payload,
        }

    async def record_inbound(self, event: Any, session_key: str) -> None:
        source = getattr(event, "source", None)
        platform = _platform_name(getattr(source, "platform", None))
        if platform != "discord":
            return
        message_id = getattr(event, "message_id", None)
        message_text = getattr(event, "text", "") or ""
        idempotency = f"inbound:{platform}:{getattr(source, 'chat_id', '')}:{getattr(source, 'thread_id', '')}:{message_id or getattr(event, 'timestamp', '')}"
        actor_ref = self.ref(getattr(source, "user_id", None) or getattr(source, "user_name", None) or "unknown", "person")
        envelope = self.envelope(
            event_type="discord.inbound.received",
            idempotency_key=idempotency,
            source=source,
            session_key=session_key,
            actor_kind="operator",
            actor_ref=actor_ref,
            status_state="new",
            summary="Discord inbound metadata recorded",
            payload={
                "status": "received",
                "platform": platform,
                "message_ref": self.ref(message_id, "message") if message_id else None,
                "message_length": len(message_text),
                "message_type": str(getattr(getattr(event, "message_type", None), "value", getattr(event, "message_type", ""))),
                "has_media": bool(getattr(event, "media_urls", None)),
                "media_count": len(getattr(event, "media_urls", None) or []),
            },
        )
        await self.append(envelope)

    async def record_assistant_status(
        self,
        *,
        source: Any,
        session_key: str,
        session_id: Optional[str],
        inbound_message_id: Optional[str],
        run_generation: Optional[int],
        status: str,
        response_chars: Optional[int] = None,
        api_calls: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
        error_type: Optional[str] = None,
    ) -> None:
        platform = _platform_name(getattr(source, "platform", None))
        if platform != "discord":
            return
        idempotency = f"assistant:{status}:{session_key}:{session_id}:{inbound_message_id}:{run_generation}"
        state = "in_progress" if status == "started" else ("failed" if status == "failed" else "pass")
        envelope = self.envelope(
            event_type="assistant.status",
            idempotency_key=idempotency,
            source=source,
            session_key=session_key,
            actor_kind="assistant",
            actor_ref=self.ref(self.config.runtime_id, "runtime"),
            status_state=state,
            summary=f"Assistant status {status}",
            payload={
                "status": status,
                "session_ref": self.ref(session_id, "session-id") if session_id else None,
                "inbound_message_ref": self.ref(inbound_message_id, "message") if inbound_message_id else None,
                "run_generation": run_generation,
                "response_chars": response_chars,
                "api_calls": api_calls,
                "elapsed_seconds": elapsed_seconds,
                "error_type": error_type,
            },
        )
        await self.append(envelope)

    def intent_marker(
        self,
        *,
        source: Any,
        session_key: Optional[str],
        inbound_message_id: Optional[str],
        intent_kind: str,
    ) -> dict[str, Any]:
        platform = _platform_name(getattr(source, "platform", None))
        base = f"outbound-intent:{platform}:{session_key}:{inbound_message_id}:{intent_kind}"
        intent_event_id = _event_id("outbound.intent.recorded", base)
        enforcement_enabled = bool(self.config.runtime_lease_enforcement_enabled)
        send_path_blocking_enabled = bool(self.config.runtime_lease_send_path_blocking_enabled)
        return {
            "intent_event_id": intent_event_id,
            "idempotency_key": base,
            "intent_kind": intent_kind,
            "case": self._case(session_key),
            "session_key_ref": self.ref(session_key, "session") if session_key else None,
            "inbound_message_ref": self.ref(inbound_message_id, "message") if inbound_message_id else None,
            "runtime_lease_enforcement": {
                "config_path": "canonical_brain.runtime_lease",
                "enforcement_enabled": enforcement_enabled,
                "send_path_blocking_enabled": send_path_blocking_enabled,
                "blocking_effective": enforcement_enabled and send_path_blocking_enabled,
                "preflight_read_at": _now_iso(),
            },
        }

    async def record_outbound_intent(
        self,
        *,
        source: Any,
        session_key: Optional[str],
        marker: dict[str, Any],
        content: Optional[str],
    ) -> None:
        platform = _platform_name(getattr(source, "platform", None))
        if platform != "discord":
            return
        envelope = self.envelope(
            event_type="outbound.intent.recorded",
            idempotency_key=str(marker.get("idempotency_key")),
            source=source,
            session_key=session_key,
            actor_kind="assistant",
            actor_ref=self.ref(self.config.runtime_id, "runtime"),
            status_state="in_progress",
            summary="Outbound Discord send intent recorded",
            payload={
                "status": "intent_created",
                "intent_event_id": marker.get("intent_event_id"),
                "intent_kind": marker.get("intent_kind"),
                "content_length": len(content or ""),
            },
        )
        await self.append(envelope)

    async def record_send_path_lease_shadow(
        self,
        *,
        source: Any,
        session_key: Optional[str],
        marker: dict[str, Any],
    ) -> None:
        platform = _platform_name(getattr(source, "platform", None))
        if platform != "discord":
            return
        await asyncio.to_thread(self._record_send_path_lease_shadow_sync, source, session_key, marker)

    def _record_send_path_lease_shadow_sync(self, source: Any, session_key: Optional[str], marker: dict[str, Any]) -> None:
        from gateway.canonical_writer_boundary import canonical_writer_call
        from gateway.canonical_writer_protocol import CanonicalWriterOperation

        result = canonical_writer_call(
            CanonicalWriterOperation.LEASE_SHADOW_RECORD.value,
            {
                "intent_event_id": str(marker.get("intent_event_id") or ""),
                "intent_kind": str(marker.get("intent_kind") or ""),
                "case": marker.get("case") or {},
                "runtime_lease_enforcement": (
                    marker.get("runtime_lease_enforcement") or {}
                ),
                "enforcement_enabled": bool(
                    self.config.runtime_lease_enforcement_enabled
                ),
                "send_path_blocking_enabled": bool(
                    self.config.runtime_lease_send_path_blocking_enabled
                ),
                "audit_runtime_id": self.config.runtime_id,
                "source_platform": _platform_name(
                    getattr(source, "platform", None)
                ),
                "session_key_ref": self.ref(session_key or "", "session"),
            },
            idempotency_key=(
                f"send-path-lease-shadow:{marker.get('intent_event_id')}"
            ),
        )
        envelope = result.get("envelope")
        if isinstance(envelope, dict):
            self._append_jsonl(envelope)
        logger.debug(
            "canonical brain send-path lease shadow persisted: %s",
            result.get("status") or result.get("event_id") or "ok",
        )

    async def record_outbound_receipt(
        self,
        *,
        chat_id: Any,
        metadata: Optional[dict[str, Any]],
        result: Any,
        content: Optional[str],
    ) -> None:
        marker = (metadata or {}).get(_METADATA_KEY)
        if not marker:
            return
        source_stub = _SourceStub(
            platform="discord",
            chat_id=chat_id,
            thread_id=(metadata or {}).get("thread_id"),
            user_id=None,
        )
        success = bool(getattr(result, "success", False))
        message_id = getattr(result, "message_id", None)
        raw_response = getattr(result, "raw_response", None) or {}
        message_ids = raw_response.get("message_ids") if isinstance(raw_response, dict) else None
        event_type = "outbound.receipt.recorded" if success else "outbound.delivery_error.recorded"
        receipt_key = f"{marker.get('intent_event_id')}:{event_type}:{message_id}:{getattr(result, 'error', '')}"
        envelope = self.envelope(
            event_type=event_type,
            idempotency_key=receipt_key,
            source=source_stub,
            session_key=marker.get("session_key_ref"),
            actor_kind="service",
            actor_ref=self.ref(self.config.runtime_id, "runtime"),
            status_state="pass" if success else "failed",
            summary="Discord outbound receipt metadata recorded",
            case_ref=marker.get("case"),
            payload={
                "status": "sent" if success else "failed",
                "intent_event_id": marker.get("intent_event_id"),
                "intent_kind": marker.get("intent_kind"),
                "message_ref": self.ref(message_id, "message") if message_id else None,
                "message_count": len(message_ids or []) if isinstance(message_ids, list) else (1 if message_id else 0),
                "content_length": len(content or ""),
                "error_type": type(getattr(result, "error", None)).__name__ if getattr(result, "error", None) else None,
                "error_present": bool(getattr(result, "error", None)),
            },
        )
        await self.append(envelope)


@dataclass
class _SourceStub:
    platform: str
    chat_id: Any
    thread_id: Any
    user_id: Any = None
    chat_type: str = "thread"


def schedule_inbound_event(event: Any, session_key: str) -> None:
    bridge = get_audit_bridge()
    if bridge is None:
        return
    _schedule(bridge.record_inbound(event, session_key))


def schedule_assistant_status(
    *,
    source: Any,
    session_key: str,
    session_id: Optional[str],
    inbound_message_id: Optional[str],
    run_generation: Optional[int],
    status: str,
    response_chars: Optional[int] = None,
    api_calls: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
    error_type: Optional[str] = None,
) -> None:
    bridge = get_audit_bridge()
    if bridge is None:
        return
    _schedule(
        bridge.record_assistant_status(
            source=source,
            session_key=session_key,
            session_id=session_id,
            inbound_message_id=inbound_message_id,
            run_generation=run_generation,
            status=status,
            response_chars=response_chars,
            api_calls=api_calls,
            elapsed_seconds=elapsed_seconds,
            error_type=error_type,
        )
    )


def prepare_outbound_intent_metadata(
    metadata: Optional[dict[str, Any]],
    *,
    source: Any,
    session_key: Optional[str],
    inbound_message_id: Optional[str],
    intent_kind: str,
    content: Optional[str],
) -> Optional[dict[str, Any]]:
    bridge = get_audit_bridge()
    if bridge is None:
        return metadata
    new_metadata = dict(metadata or {})
    marker = bridge.intent_marker(
        source=source,
        session_key=session_key,
        inbound_message_id=inbound_message_id,
        intent_kind=intent_kind,
    )
    new_metadata[_METADATA_KEY] = marker
    _schedule(
        bridge.record_outbound_intent(
            source=source,
            session_key=session_key,
            marker=marker,
            content=content,
        )
    )
    _schedule(
        bridge.record_send_path_lease_shadow(
            source=source,
            session_key=session_key,
            marker=marker,
        )
    )
    return new_metadata


def schedule_outbound_receipt(
    *,
    chat_id: Any,
    metadata: Optional[dict[str, Any]],
    result: Any,
    content: Optional[str],
) -> None:
    bridge = get_audit_bridge()
    if bridge is None:
        return
    _schedule(
        bridge.record_outbound_receipt(
            chat_id=chat_id,
            metadata=metadata,
            result=result,
            content=content,
        )
    )
