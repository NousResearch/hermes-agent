"""Atomic native-session ownership receipts for gateway rollover."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from typing import Any, Optional


_ROLLOVER_META_PREFIX = "gateway_rollover:"
_ROLLOVER_PHASES = frozenset({
    "authoring", "prepared", "committed", "bootstrapping", "admitting", "complete", "blocked",
    "invalidated",
})


def _rollover_meta_key(session_key: str) -> str:
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return f"{_ROLLOVER_META_PREFIX}{digest}"


def _decode_receipt(raw: Optional[str], *, session_key: Optional[str] = None) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    try:
        receipt = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Malformed native session rollover receipt") from exc
    if not isinstance(receipt, dict) or not isinstance(receipt.get("session_key"), str):
        raise RuntimeError("Malformed native session rollover receipt")
    if session_key is not None and receipt["session_key"] != session_key:
        raise RuntimeError("Native session rollover receipt key mismatch")
    if receipt.get("phase") not in _ROLLOVER_PHASES:
        raise RuntimeError("Native session rollover receipt has an invalid phase")
    return receipt


class SessionRolloverMixin:
    """SessionDB receipt reads and transactional rollover ownership commits."""

    @staticmethod
    def _rollover_receipt_on_conn(conn, session_key: str) -> Optional[dict[str, Any]]:
        row = conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (_rollover_meta_key(session_key),)
        ).fetchone()
        return _decode_receipt(row[0] if row else None, session_key=session_key)

    @staticmethod
    def _write_rollover_receipt_on_conn(conn, receipt: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_rollover_meta_key(receipt["session_key"]), json.dumps(receipt, ensure_ascii=False)),
        )

    @staticmethod
    def _rollover_route_row(conn, receipt: dict[str, Any]):
        row = conn.execute(
            "SELECT id, source, user_id, session_key, chat_id, chat_type, thread_id, "
            "profile_name, transport_profile, origin_json, display_name, ended_at, end_reason "
            "FROM sessions WHERE id = ?",
            (receipt["old_session_id"],),
        ).fetchone()
        if row is None or row["session_key"] != receipt["session_key"]:
            raise ValueError("Rollover predecessor is not the current durable route")
        return row

    @staticmethod
    def _validate_route_identity(row, route_identity: dict[str, Any]) -> dict[str, Any]:
        origin = route_identity.get("origin")
        if not isinstance(origin, dict):
            raise ValueError("Rollover route identity is missing its source origin")
        try:
            stored_origin = json.loads(row["origin_json"] or "")
        except (TypeError, ValueError) as exc:
            raise ValueError("Rollover predecessor has no valid durable route origin") from exc
        identity_fields = (
            "platform", "chat_id", "chat_type", "user_id", "thread_id", "scope_id", "profile",
        )
        if (
            not isinstance(stored_origin, dict)
            or any(stored_origin.get(name) != origin.get(name) for name in identity_fields)
            or row["source"] != route_identity.get("platform")
            or str(row["user_id"] or "") != str(stored_origin.get("user_id") or "")
            or str(row["chat_id"] or "") != str(stored_origin.get("chat_id") or "")
            or str(row["chat_type"] or "") != str(stored_origin.get("chat_type") or "")
            or str(row["thread_id"] or "") != str(stored_origin.get("thread_id") or "")
            or (route_identity.get("transport_profile") is not None
                and row["transport_profile"] != route_identity["transport_profile"])
        ):
            raise ValueError("Rollover predecessor route identity changed")
        return stored_origin

    def get_gateway_rollover(self, session_key: str) -> Optional[dict[str, Any]]:
        if not session_key:
            return None
        return _decode_receipt(self.get_meta(_rollover_meta_key(session_key)), session_key=session_key)

    def list_gateway_rollovers(self) -> list[dict[str, Any]]:
        receipts = []
        for _key, raw in self.list_meta_prefix(_ROLLOVER_META_PREFIX):
            receipt = _decode_receipt(raw)
            if receipt is not None:
                receipts.append(receipt)
        return receipts

    def begin_gateway_rollover(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Persist authoring ownership before model packet-writing and allocate one successor ID."""
        required = ("session_key", "old_session_id", "profile_home", "packet_path")
        if any(not receipt.get(field) for field in required) or receipt.get("generation") is None:
            raise ValueError("Rollover authoring receipt is missing a required binding")
        if receipt.get("phase") != "authoring" or receipt.get("owner_committed") is not False:
            raise ValueError("Rollover authoring receipt has an invalid initial state")

        def _do(conn):
            existing = self._rollover_receipt_on_conn(conn, receipt["session_key"])
            if existing is not None:
                same_attempt = all(existing.get(name) == receipt.get(name) for name in (
                    "session_key", "old_session_id", "generation", "profile_home", "packet_path",
                ))
                if same_attempt and existing["phase"] == "authoring" and not existing.get("owner_committed"):
                    return existing
                if existing["phase"] not in {"complete", "invalidated"}:
                    prior_owner = (
                        existing.get("new_session_id")
                        if existing.get("owner_committed") else existing.get("old_session_id")
                    )
                    prior_row = conn.execute(
                        "SELECT session_key, end_reason FROM sessions WHERE id = ?", (prior_owner,),
                    ).fetchone()
                    superseded_by_reset = bool(
                        prior_owner != receipt["old_session_id"]
                        and prior_row is not None
                        and prior_row["session_key"] == receipt["session_key"]
                        and prior_row["end_reason"] == "session_reset"
                    )
                    if not superseded_by_reset:
                        raise ValueError("A different native session rollover is already pending")
                    invalidated = dict(existing)
                    invalidated["phase"] = "invalidated"
                    invalidated["phase_reason"] = "An explicit session reset superseded the rollover."
                    self._write_rollover_receipt_on_conn(conn, invalidated)
            row = self._rollover_route_row(conn, receipt)
            if row["ended_at"] is not None:
                raise ValueError("Rollover predecessor is not the live durable route")
            stored_origin = self._validate_route_identity(row, receipt["route_identity"])
            from hermes_state_ids import new_session_id

            allocated = dict(receipt)
            allocated["new_session_id"] = new_session_id(datetime.now(), hex_len=8)
            allocated["profile_name"] = row["profile_name"]
            allocated["transport_profile"] = row["transport_profile"]
            allocated["origin_json"] = row["origin_json"]
            allocated["source"] = row["source"]
            allocated["user_id"] = row["user_id"]
            allocated["chat_id"] = row["chat_id"]
            allocated["chat_type"] = row["chat_type"]
            allocated["thread_id"] = row["thread_id"]
            allocated["display_name"] = row["display_name"]
            canonical_route = dict(allocated["route_identity"])
            canonical_route["origin"] = stored_origin
            canonical_route["transport_profile"] = row["transport_profile"]
            route_entry = dict(canonical_route["route_entry"])
            route_entry.update({
                "origin": stored_origin,
                "transport_profile": row["transport_profile"],
                "display_name": row["display_name"],
            })
            canonical_route["route_entry"] = route_entry
            allocated["route_identity"] = canonical_route
            self._write_rollover_receipt_on_conn(conn, allocated)
            return allocated

        return self._execute_write(_do)

    def prepare_gateway_rollover(
        self, binding: dict[str, Any], *, expected_new_session_id: str,
    ) -> dict[str, Any]:
        """Seal the exact authoring receipt; never allocate a second successor."""
        required = (
            "session_key", "old_session_id", "generation", "profile_home", "packet_path",
            "packet_sha256", "restart_message", "bootstrap_prompt", "reason", "transcript_watermark",
        )
        if any(name not in binding for name in required):
            raise ValueError("Rollover handoff binding is incomplete")
        watermark = binding["transcript_watermark"]
        if type(watermark) is not int or watermark < 0:
            raise ValueError("Rollover requires the authoring turn's transcript watermark")

        def _do(conn):
            receipt = self._rollover_receipt_on_conn(conn, binding["session_key"])
            if receipt is None:
                raise ValueError("No authoring receipt exists for this rollover")
            if receipt.get("new_session_id") != expected_new_session_id:
                raise ValueError("Rollover attempt changed before preparation")
            if any(receipt.get(name) != binding.get(name) for name in (
                "session_key", "old_session_id", "generation", "profile_home", "packet_path",
            )):
                raise ValueError("Rollover authoring binding changed")
            if receipt["phase"] == "prepared":
                if any(receipt.get(name) != binding.get(name) for name in (
                    "packet_sha256", "restart_message", "bootstrap_prompt", "reason", "transcript_watermark",
                )):
                    raise ValueError("Rollover handoff binding changed after preparation")
                return receipt
            if receipt["phase"] != "authoring" or receipt.get("owner_committed"):
                raise ValueError("Rollover authoring receipt is no longer preparable")
            row = self._rollover_route_row(conn, receipt)
            self._validate_route_identity(row, receipt["route_identity"])
            if row["ended_at"] is not None:
                raise ValueError("Rollover predecessor ended before packet preparation")
            prepared = dict(receipt)
            prepared.update({
                "packet_sha256": binding["packet_sha256"],
                "restart_message": binding["restart_message"],
                "bootstrap_prompt": binding["bootstrap_prompt"],
                "reason": binding["reason"],
                "transcript_watermark": watermark,
                "phase": "prepared",
            })
            self._write_rollover_receipt_on_conn(conn, prepared)
            return prepared

        return self._execute_write(_do)

    def commit_gateway_rollover(
        self, session_key: str, expected_session_id: str, generation: Any, *,
        expected_new_session_id: str,
    ) -> dict[str, Any]:
        """Atomically end the predecessor, insert its fresh reset child and commit the receipt."""
        if not session_key or not expected_session_id:
            raise ValueError("Rollover commit requires a session key and predecessor id")

        def _do(conn):
            receipt = self._rollover_receipt_on_conn(conn, session_key)
            if receipt is None:
                raise ValueError("No prepared native session rollover exists")
            if receipt.get("old_session_id") != expected_session_id:
                raise ValueError("Rollover predecessor changed")
            if receipt.get("new_session_id") != expected_new_session_id:
                raise ValueError("Rollover attempt changed before commit")
            if receipt.get("generation") != generation:
                raise ValueError("Rollover generation changed")
            if receipt["phase"] == "committed" and receipt.get("owner_committed") is True:
                return receipt
            if receipt["phase"] != "prepared" or receipt.get("owner_committed") is not False:
                raise ValueError("Rollover receipt is not prepared for commit")

            row = self._rollover_route_row(conn, receipt)
            self._validate_route_identity(row, receipt["route_identity"])
            if row["ended_at"] is not None:
                raise ValueError("Rollover predecessor ended before cutover")
            self._check_transcript_write_guards(
                conn, expected_session_id, compression_lock_holder=None,
                reject_active_turn_lease=True, reject_active_compression_lock=True,
            )
            watermark = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ? AND active = 1",
                (expected_session_id,),
            ).fetchone()[0]
            if watermark != receipt["transcript_watermark"]:
                raise ValueError("Rollover transcript changed after handoff authoring; refresh the handoff")
            successor_id = receipt["new_session_id"]
            if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (successor_id,)).fetchone():
                raise ValueError("Rollover successor id is already present")

            changed = self._end_and_bump(
                conn,
                "UPDATE sessions SET ended_at = ?, end_reason = 'session_reset' "
                "WHERE id = ? AND session_key = ? AND ended_at IS NULL",
                (time.time(), expected_session_id, session_key), expected_session_id, "session_reset",
            )
            if changed != 1:
                raise ValueError("Rollover predecessor changed during cutover")
            from hermes_state_worker_lifecycle import insert_session_row_in_transaction

            insert_session_row_in_transaction(
                self, conn, session_id=successor_id, source=receipt["source"],
                model_config={"_reset_from": expected_session_id},
                user_id=receipt.get("user_id"), session_key=session_key,
                chat_id=receipt.get("chat_id"), chat_type=receipt.get("chat_type"),
                thread_id=receipt.get("thread_id"), parent_session_id=expected_session_id,
                profile_name=receipt.get("profile_name"),
                transport_profile=receipt.get("transport_profile"),
                origin_json=receipt.get("origin_json"), display_name=receipt.get("display_name"),
            )
            committed = dict(receipt)
            committed.update({"phase": "committed", "owner_committed": True, "committed_at": time.time()})
            self._write_rollover_receipt_on_conn(conn, committed)
            return committed

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def set_gateway_rollover_phase(
        self, session_key: str, new_session_id: str, expected_phase: str, phase: str, reason: str = "",
    ) -> dict[str, Any]:
        if phase not in _ROLLOVER_PHASES or expected_phase not in _ROLLOVER_PHASES:
            raise ValueError("Invalid native session rollover phase")

        def _do(conn):
            receipt = self._rollover_receipt_on_conn(conn, session_key)
            if receipt is None or receipt.get("new_session_id") != new_session_id:
                raise ValueError("Native session rollover identity changed")
            if receipt["phase"] != expected_phase:
                raise ValueError("Native session rollover phase changed")
            allowed = {
                "authoring": {"blocked", "invalidated"},
                "prepared": {"blocked", "invalidated"},
                "committed": {"bootstrapping", "blocked", "invalidated"},
                "bootstrapping": {"admitting", "complete", "blocked", "invalidated"},
                "admitting": {"complete", "blocked", "invalidated"},
                "blocked": {"invalidated"},
                "invalidated": set(),
                "complete": set(),
            }
            if phase not in allowed[expected_phase]:
                raise ValueError(f"Invalid rollover phase transition: {expected_phase} -> {phase}")
            if phase in {"bootstrapping", "admitting", "complete"} and not receipt.get("owner_committed"):
                raise ValueError("An uncommitted rollover cannot admit or complete a successor")
            updated = dict(receipt)
            updated["phase"] = phase
            updated["phase_reason"] = reason
            self._write_rollover_receipt_on_conn(conn, updated)
            return updated

        return self._execute_write(_do)
