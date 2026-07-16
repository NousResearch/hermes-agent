from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .models import WorkerRecord


def canonical_receipt_bytes(value: dict) -> bytes:
    """Canonical receipt encoding: sorted keys, compact JSON, literal UTF-8, no newline."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def receipt_hash(value: dict) -> str:
    body = {key: item for key, item in value.items() if key != "receipt_hash"}
    return hashlib.sha256(canonical_receipt_bytes(body)).hexdigest()


def receipt_from_launch(launch: dict, *, status: str, usage: dict, **extra) -> dict:
    """Build a matching receipt from the launch JSON passed to an external worker."""
    required = (
        "worker_id",
        "campaign_id",
        "step_index",
        "role",
        "cwd",
        "tmux_session",
        "provider",
        "model",
        "prompt_hash",
        "mutable_manifest",
        "nonce",
    )
    missing = [key for key in required if key not in launch]
    if missing:
        raise ValueError(f"launch spec missing receipt fields: {', '.join(missing)}")
    value = {
        "schema": 1,
        **{key: launch[key] for key in required},
        "status": status,
        "usage": usage,
        "worker_turns": int(extra.pop("worker_turns", 1)),
        "model_fallback": bool(extra.pop("model_fallback", False)),
        **extra,
    }
    value["receipt_hash"] = receipt_hash(value)
    return value


def build_receipt(worker: WorkerRecord, *, status: str, usage: dict, **extra) -> dict:
    return receipt_from_launch(worker.__dict__, status=status, usage=usage, **extra)


def write_receipt(path: Path, value: dict) -> None:
    value = dict(value)
    value["receipt_hash"] = receipt_hash(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_receipt_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def verify_receipt(
    path: Path, worker: WorkerRecord, max_worker_turns: int
) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "receipt is missing or malformed"
    expected = {
        "worker_id": worker.worker_id,
        "campaign_id": worker.campaign_id,
        "step_index": worker.step_index,
        "role": worker.role,
        "cwd": worker.cwd,
        "tmux_session": worker.tmux_session,
        "provider": worker.provider,
        "model": worker.model,
        "prompt_hash": worker.prompt_hash,
        "mutable_manifest": worker.mutable_manifest,
        "nonce": worker.nonce,
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        return None, "receipt metadata does not match launch record"
    if value.get("receipt_hash") != receipt_hash(value):
        return None, "receipt hash mismatch"
    if value.get("status") != "COMPLETE" or not isinstance(value.get("usage"), dict):
        return None, "receipt is not terminal-complete"
    if int(value.get("worker_turns", 0)) > max_worker_turns:
        return None, "worker turn budget exceeded"
    if worker.role == "reviewer" and value.get("model_fallback"):
        return None, "reviewer fallback is advisory only"
    return value, None
