from __future__ import annotations

import json
import os
import time
import uuid

from pathlib import Path
from typing import Any, Dict, Optional

try:
    from hermes_plugins.truth_ledger.schemas import validate_document
except Exception:  # pragma: no cover - normal package import path
    from .schemas import validate_document


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _write_private_json_atomic(path: Path, payload: Dict[str, Any]) -> Path:
    _mkdir_private(path.parent)
    tmp = path.parent / f".tmp-{uuid.uuid4().hex}.json"
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    try:
        with tmp.open("wb") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    try:
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
    return path


class TruthSpool:
    def __init__(self, root: Path, soft_count: int = 5000, hard_count: int = 8000) -> None:
        self.root = Path(root)
        self.soft_count = soft_count
        self.hard_count = hard_count
        self.spool_dir = self.root / "spool"
        self.pending_dir = self.spool_dir / "pending"
        self.processing_dir = self.spool_dir / "processing"
        self.dead_letter_dir = self.spool_dir / "dead-letter"
        self.payloads_dir = self.spool_dir / "payloads"
        self.errors_dir = self.root / "errors"
        for d in (
            self.spool_dir,
            self.pending_dir,
            self.processing_dir,
            self.dead_letter_dir,
            self.payloads_dir,
            self.errors_dir,
        ):
            _mkdir_private(d)

    def _envelope_id(self) -> str:
        return f"env_{uuid.uuid4().hex}"

    def _payload_path_from_record(self, record: Dict[str, Any]) -> Optional[Path]:
        raw = record.get("payload_path")
        if not isinstance(raw, str) or not raw:
            return None
        return Path(raw)

    def _is_owned_payload_path(self, payload_path: Path) -> bool:
        try:
            payload_real = payload_path.resolve(strict=False)
            root_real = self.payloads_dir.resolve(strict=False)
            payload_real.relative_to(root_real)
            return True
        except Exception:
            return False

    def _unlink_payload_if_owned(self, record: Dict[str, Any]) -> None:
        payload_path = self._payload_path_from_record(record)
        if payload_path is None:
            return
        if not self._is_owned_payload_path(payload_path):
            return
        try:
            payload_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _idempotency_key(self, envelope: Dict[str, Any]) -> str:
        profile = str(envelope.get("profile") or "")
        session_id = str(envelope.get("session_id") or "")
        turn_id = str(envelope.get("turn_id") or "")
        if profile and session_id and turn_id:
            return f"{profile}:{session_id}:{turn_id}"
        return f"fallback:{uuid.uuid4().hex}"

    def _source_ref(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "profile": str(envelope.get("profile") or "unknown"),
            "session_id": str(envelope.get("session_id") or "unknown"),
            "turn_id": str(envelope.get("turn_id") or ""),
        }

    def _new_record_name(self) -> str:
        return f"{time.time_ns()}-{uuid.uuid4().hex}.json"

    def _record_from_envelope(self, *, envelope: Dict[str, Any], payload_path: Path, state: str) -> Dict[str, Any]:
        return {
            "schema_name": "truth-ledger.spool-record.v1",
            "schema_version": 1,
            "envelope_id": self._envelope_id(),
            "state": state,
            "captured_at": str(envelope.get("captured_at") or ""),
            "attempt_count": 0,
            "idempotency_key": self._idempotency_key(envelope),
            "source_ref": self._source_ref(envelope),
            "payload_path": str(payload_path),
            "flow": {},
        }

    def _load_record(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _dead_letter_record_for_quarantine(self, record: Dict[str, Any], reason: str) -> Dict[str, Any]:
        now = time.time()

        def _safe_str(value: Any, fallback: str, max_len: int) -> str:
            text = str(value) if value is not None else fallback
            text = text.strip() or fallback
            return text[:max_len]

        source_candidate: Any = record.get("source_ref")
        raw_source: Dict[str, Any] = source_candidate if isinstance(source_candidate, dict) else {}
        flow_candidate: Any = record.get("flow")
        flow_in: Dict[str, Any] = flow_candidate if isinstance(flow_candidate, dict) else {}
        flow: Dict[str, Any] = {}
        for key in (
            "processing_owner",
            "claimed_at",
            "last_error_code",
            "next_retry_at",
            "recovered_at",
            "decode_error",
        ):
            if key in flow_in:
                flow[key] = flow_in[key]
        flow["dead_letter_reason"] = str(reason)[:256]
        flow["dead_letter_at"] = now

        attempt_raw = record.get("attempt_count", 0)
        try:
            attempt_count = int(attempt_raw)
        except Exception:
            attempt_count = 0
        attempt_count = max(0, min(100, attempt_count))

        envelope_id = str(record.get("envelope_id") or "")
        if not envelope_id.startswith("env_"):
            envelope_id = self._envelope_id()

        payload_path = _safe_str(
            record.get("payload_path"),
            str(self.payloads_dir / f"missing-{uuid.uuid4().hex}.json"),
            4096,
        )

        dead_record = {
            "schema_name": "truth-ledger.spool-record.v1",
            "schema_version": 1,
            "envelope_id": envelope_id[:128],
            "state": "dead_lettered",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "attempt_count": attempt_count,
            "idempotency_key": _safe_str(record.get("idempotency_key"), f"quarantine:{uuid.uuid4().hex}", 512),
            "source_ref": {
                "profile": _safe_str(raw_source.get("profile"), "unknown", 128),
                "session_id": _safe_str(raw_source.get("session_id"), "unknown", 256),
                "turn_id": _safe_str(raw_source.get("turn_id"), "", 256),
            },
            "payload_path": payload_path,
            "flow": flow,
        }
        try:
            validate_document("spool-record.v1", dead_record)
        except Exception:
            dead_record = {
                "schema_name": "truth-ledger.spool-record.v1",
                "schema_version": 1,
                "envelope_id": self._envelope_id(),
                "state": "dead_lettered",
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "attempt_count": 0,
                "idempotency_key": f"quarantine:{uuid.uuid4().hex}",
                "source_ref": {"profile": "unknown", "session_id": "unknown", "turn_id": ""},
                "payload_path": str(self.payloads_dir / f"missing-{uuid.uuid4().hex}.json"),
                "flow": {"dead_letter_reason": str(reason)[:256], "dead_letter_at": now},
            }
        return dead_record

    def _quarantine_record(self, path: Path, reason: str) -> None:
        original_record: Dict[str, Any] = {}
        try:
            loaded = self._load_record(path)
            if isinstance(loaded, dict):
                original_record = loaded
        except Exception:
            original_record = {}

        dead_record = self._dead_letter_record_for_quarantine(original_record, reason)
        _write_private_json_atomic(self.dead_letter_dir / self._new_record_name(), dead_record)
        self._unlink_payload_if_owned(original_record)
        if path.exists():
            path.unlink()

    def _load_envelope_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        payload_path = self._payload_path_from_record(record)
        if payload_path is None:
            raise ValueError("missing_payload")
        if not self._is_owned_payload_path(payload_path):
            raise ValueError("payload_path_out_of_root")
        if not payload_path.exists():
            raise ValueError("missing_payload")
        try:
            envelope = json.loads(payload_path.read_text(encoding="utf-8"))
            validate_document("source-envelope.v1", envelope)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid_source_envelope") from exc
        flow = record.get("flow") if isinstance(record.get("flow"), dict) else {}
        enriched = dict(envelope)
        enriched["attempt_count"] = int(record.get("attempt_count", 0))
        if isinstance(flow, dict):
            if "last_error_code" in flow:
                enriched["last_error_code"] = flow.get("last_error_code")
            if "next_retry_at" in flow:
                enriched["next_retry_at"] = flow.get("next_retry_at")
        return enriched

    def _queue_counts(self) -> tuple[int, int]:
        pending = sum(1 for _ in self.pending_dir.glob("*.json"))
        processing = sum(1 for _ in self.processing_dir.glob("*.json"))
        return pending, processing

    def enqueue(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        pending, processing = self._queue_counts()
        total = pending + processing
        if total >= self.hard_count:
            self._write_error({"code": "queue_hard_cap", "at": time.time()})
            return {"ok": False, "reason": "queue_hard_cap", "path": None}

        payload_path = self.payloads_dir / f"{self._envelope_id()}.json"
        _write_private_json_atomic(payload_path, dict(envelope))
        try:
            record = self._record_from_envelope(
                envelope=dict(envelope),
                payload_path=payload_path,
                state="pending",
            )
            out = _write_private_json_atomic(self.pending_dir / self._new_record_name(), record)
        except Exception:
            try:
                payload_path.unlink()
            except FileNotFoundError:
                pass
            raise

        self._shed_soft_overflow_if_needed()
        return {"ok": True, "reason": None, "path": str(out)}

    def claim_next(self, owner: str = "") -> Optional[Dict[str, Any]]:
        for src in sorted(self.pending_dir.glob("*.json")):
            dst = self.processing_dir / src.name
            try:
                os.replace(src, dst)
            except FileNotFoundError:
                continue
            except OSError:
                continue

            try:
                record = self._load_record(dst)
                validate_document("spool-record.v1", record)
            except Exception:
                self._quarantine_record(dst, "invalid_spool_record")
                continue

            try:
                envelope = self._load_envelope_from_record(record)
            except ValueError as exc:
                reason = str(exc)
                if reason not in {"invalid_source_envelope", "missing_payload", "payload_path_out_of_root"}:
                    reason = "invalid_source_envelope"
                self._quarantine_record(dst, reason)
                continue

            flow = dict(record.get("flow") or {})
            flow["processing_owner"] = owner
            flow["claimed_at"] = time.time()
            record["state"] = "processing"
            record["flow"] = flow
            _write_private_json_atomic(dst, record)
            return {
                "path": str(dst),
                "envelope": envelope,
                "record": record,
            }
        return None

    def ack_processing(self, processing_path: Path) -> Dict[str, Any]:
        path = Path(processing_path)
        record: Dict[str, Any] = {}
        if path.exists():
            try:
                record = self._load_record(path)
            except Exception:
                record = {}
        if path.exists():
            path.unlink()
        self._unlink_payload_if_owned(record)
        return {"ok": True}

    def retry_processing(self, processing_path: Path, error_code: str) -> Dict[str, Any]:
        src = Path(processing_path)
        record = self._load_record(src)
        flow = dict(record.get("flow") or {})
        flow["last_error_code"] = str(error_code)
        flow["next_retry_at"] = time.time()
        record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
        record["state"] = "pending"
        record["flow"] = flow

        dst = _write_private_json_atomic(self.pending_dir / self._new_record_name(), record)
        if src.exists():
            src.unlink()
        return {"ok": True, "path": str(dst)}

    def dead_letter(self, processing_path: Path, reason: str) -> Dict[str, Any]:
        src = Path(processing_path)
        record = self._load_record(src)
        flow = dict(record.get("flow") or {})
        flow["dead_letter_reason"] = str(reason)
        flow["dead_letter_at"] = time.time()
        record["state"] = "dead_lettered"
        record["flow"] = flow
        dst = _write_private_json_atomic(
            self.dead_letter_dir / self._new_record_name(),
            record,
        )
        if src.exists():
            src.unlink()
        self._unlink_payload_if_owned(record)
        return {"ok": True, "path": str(dst)}

    def recover_stale_processing(self, stale_seconds: int = 900) -> int:
        now = time.time()
        moved = 0
        for src in sorted(self.processing_dir.glob("*.json")):
            try:
                age = now - src.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < stale_seconds:
                continue
            try:
                record = self._load_record(src)
                validate_document("spool-record.v1", record)
                record["state"] = "pending"
                flow = dict(record.get("flow") or {})
                flow["recovered_at"] = now
                record["flow"] = flow
                _write_private_json_atomic(self.pending_dir / self._new_record_name(), record)
                src.unlink()
                moved += 1
            except Exception:
                if not src.exists():
                    # Benign race: another lifecycle path (ack/close) removed the record.
                    continue
                try:
                    self._quarantine_record(src, "invalid_spool_record")
                    moved += 1
                except Exception:
                    continue
        return moved

    def _shed_soft_overflow_if_needed(self) -> None:
        pending_files = sorted(self.pending_dir.glob("*.json"))
        while len(pending_files) > self.soft_count:
            src = pending_files.pop(0)
            try:
                record = self._load_record(src)
            except Exception:
                record = {
                    "schema_name": "truth-ledger.spool-record.v1",
                    "schema_version": 1,
                    "envelope_id": self._envelope_id(),
                    "state": "dead_lettered",
                    "captured_at": "1970-01-01T00:00:00Z",
                    "attempt_count": 0,
                    "idempotency_key": f"decode-error:{uuid.uuid4().hex}",
                    "source_ref": {"profile": "unknown", "session_id": "unknown", "turn_id": ""},
                    "payload_path": str(self.payloads_dir / f"missing-{uuid.uuid4().hex}.json"),
                    "flow": {"decode_error": True},
                }
            flow = dict(record.get("flow") or {})
            flow["dead_letter_reason"] = "queue_overflow"
            flow["dead_letter_at"] = time.time()
            record["state"] = "dead_lettered"
            record["flow"] = flow
            _write_private_json_atomic(
                self.dead_letter_dir / self._new_record_name(),
                record,
            )
            self._unlink_payload_if_owned(record)
            try:
                src.unlink()
            except FileNotFoundError:
                pass

    def _write_error(self, payload: Dict[str, Any]) -> None:
        path = self.errors_dir / "errors.jsonl"
        _mkdir_private(path.parent)
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
