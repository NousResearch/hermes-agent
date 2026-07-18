from __future__ import annotations

import json
import os
import time
import uuid

from pathlib import Path
from typing import Any, Dict, Optional


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
    with tmp.open("wb") as fh:
        fh.write(encoded)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
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
        self.errors_dir = self.root / "errors"
        for d in (self.spool_dir, self.pending_dir, self.processing_dir, self.dead_letter_dir, self.errors_dir):
            _mkdir_private(d)

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

        stamped = dict(envelope)
        stamped.setdefault("attempt_count", 0)
        stamped.setdefault("first_seen_at", time.time())
        stamped.setdefault("next_retry_at", 0.0)

        name = f"{time.time_ns()}-{uuid.uuid4().hex}.json"
        out = _write_private_json_atomic(self.pending_dir / name, stamped)

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

            payload = json.loads(dst.read_text(encoding="utf-8"))
            payload["processing_owner"] = owner
            payload["claimed_at"] = time.time()
            _write_private_json_atomic(dst, payload)
            return {"path": str(dst), "envelope": payload}
        return None

    def ack_processing(self, processing_path: Path) -> Dict[str, Any]:
        path = Path(processing_path)
        if path.exists():
            path.unlink()
        return {"ok": True}

    def retry_processing(self, processing_path: Path, error_code: str) -> Dict[str, Any]:
        src = Path(processing_path)
        payload = json.loads(src.read_text(encoding="utf-8"))
        payload["attempt_count"] = int(payload.get("attempt_count", 0)) + 1
        payload["last_error_code"] = error_code
        payload["next_retry_at"] = time.time()

        new_name = f"{time.time_ns()}-{uuid.uuid4().hex}.json"
        dst = _write_private_json_atomic(self.pending_dir / new_name, payload)
        if src.exists():
            src.unlink()
        return {"ok": True, "path": str(dst)}

    def dead_letter(self, processing_path: Path, reason: str) -> Dict[str, Any]:
        src = Path(processing_path)
        payload = json.loads(src.read_text(encoding="utf-8"))
        payload["dead_letter_reason"] = reason
        payload["dead_letter_at"] = time.time()
        dst = _write_private_json_atomic(
            self.dead_letter_dir / f"{time.time_ns()}-{uuid.uuid4().hex}.json",
            payload,
        )
        if src.exists():
            src.unlink()
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
            dst = self.pending_dir / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
            try:
                os.replace(src, dst)
                moved += 1
            except OSError:
                continue
        return moved

    def _shed_soft_overflow_if_needed(self) -> None:
        pending_files = sorted(self.pending_dir.glob("*.json"))
        while len(pending_files) > self.soft_count:
            src = pending_files.pop(0)
            try:
                payload = json.loads(src.read_text(encoding="utf-8"))
            except Exception:
                payload = {"decode_error": True}
            payload["dead_letter_reason"] = "queue_overflow"
            _write_private_json_atomic(
                self.dead_letter_dir / f"{time.time_ns()}-{uuid.uuid4().hex}.json",
                payload,
            )
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
