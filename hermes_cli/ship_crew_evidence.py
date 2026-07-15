"""Deterministic evidence records and safe artifact persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

_SECRET_KEY = re.compile(r"(token|secret|password|api[_-]?key|credential|authorization)", re.I)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): "[REDACTED]" if _SECRET_KEY.search(str(k)) else _safe(v)
            for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def evidence_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_evidence(
    *,
    task_id: str,
    contract_version: str,
    outcome: str,
    role: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "contract_version": contract_version,
        "task_id": task_id,
        "outcome": outcome,
        "role": role,
        "inputs": dict(inputs),
        "outputs": dict(outputs),
        "provenance": dict(provenance),
    }
    safe_body = _safe(body)
    safe_body["evidence_sha256"] = evidence_sha256(safe_body)
    return safe_body


def write_evidence_artifact(root: str | Path, record: Mapping[str, Any], filename: str) -> Path:
    name = Path(filename).name
    if not name or name != filename or name in {".", ".."}:
        raise ValueError("filename must be a simple artifact filename")
    destination = Path(root).resolve() / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(record) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", dir=str(destination.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination
