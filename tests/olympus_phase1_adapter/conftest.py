"""Hermetic fixtures for the source-checkout-only Phase 1 adapter."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from olympus_phase1_adapter.adapter import _PHASE0_FILES
from olympus_phase1_adapter.contracts import (
    ENGINE_CONTRACT_DIGEST,
    OPERATION_SCHEMA_ID,
    canonical_bytes,
)

PHASE0_ROOT = Path("/Users/macmini/Hermes-Handoff/olympus-engine")
PHASE0_SRC = PHASE0_ROOT / "src"

# The scoped test process may expose the frozen Phase 0 source only after every
# file in the Gate 2 runtime binding has independently matched.
for _relative, _mode, _size, _digest, _blob in _PHASE0_FILES:
    _path = PHASE0_ROOT / _relative
    _entry = _path.lstat()
    assert _path.is_file() and not _path.is_symlink()
    assert _entry.st_size == _size
    assert hashlib.sha256(_path.read_bytes()).hexdigest() == _digest
sys.path.insert(0, str(PHASE0_SRC))

from olympus_engine.authorization import PHASE0_AUTHORITY_SCOPE, authorize
from olympus_engine.contracts import (
    REQUEST_CONTRACT,
    REVIEWER_RESULT_CONTRACT,
    WORKER_RESULT_CONTRACT,
    Request,
    WorkerResult,
)
from olympus_engine.packetizer import build_packet
from olympus_engine.transports import FakePairATransport, FakePairBTransport


@pytest.fixture(autouse=True)
def _owner_only_umask() -> None:
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def phase1_roots(tmp_path: Path) -> dict[str, Path]:
    roots = {
        "repository": tmp_path / "repository",
        "receipt": tmp_path / "receipt",
        "evidence": tmp_path / "evidence",
    }
    for root in roots.values():
        root.mkdir(mode=0o700)
        root.chmod(0o700)
    (roots["repository"] / "src").mkdir(mode=0o700)
    (roots["repository"] / "README.md").write_text(
        "# Synthetic fixture\n\nOffline review content only.\n",
        encoding="utf-8",
        newline="",
    )
    (roots["repository"] / "src" / "example.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n",
        encoding="utf-8",
        newline="",
    )
    return roots


def request_value(*, request_id: str = "req-phase1-test") -> dict[str, Any]:
    return {
        "contract_id": REQUEST_CONTRACT,
        "request_id": request_id,
        "repository_id": "synthetic-phase1-test",
        "repository_kind": "SYNTHETIC",
        "repository_path": ".",
        "requested_paths": ["README.md", "src/example.py"],
        "purpose": "bounded offline Phase 1 repository review",
        "authority": {
            "scope": PHASE0_AUTHORITY_SCOPE,
            "granted": True,
        },
        "limits": {
            "max_files": 96,
            "max_file_bytes": 32768,
            "max_total_bytes": 196608,
            "max_findings": 64,
            "max_model_json_bytes": 65536,
        },
        "controls": {
            "offline": True,
            "fake_transports_only": True,
            "tools": [],
            "max_pair_a_attempts": 1,
            "max_pair_b_attempts": 1,
            "allow_retries": False,
            "allow_response_repair": False,
            "allow_cloud_fallback": False,
        },
    }


def operation_value(
    *,
    idempotency_key: str = "idem-phase1-test",
    correlation_id: str = "corr-phase1-test",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": OPERATION_SCHEMA_ID,
        "correlation_id": correlation_id,
        "idempotency_key": idempotency_key,
        "engine_contract_digest": ENGINE_CONTRACT_DIGEST,
        "payload": request_value() if payload is None else payload,
    }


def operation_bytes(**kwargs: Any) -> bytes:
    return canonical_bytes(operation_value(**kwargs))


def valid_transports(
    repository: Path,
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[FakePairATransport, FakePairBTransport]:
    value = request_value() if payload is None else payload
    request = Request.from_value(value)
    authorization = authorize(request)
    assert authorization.decision == "ALLOW"
    packet = build_packet(repository, request, authorization)
    worker_value = {
        "contract_id": WORKER_RESULT_CONTRACT,
        "request_id": request.request_id,
        "packet_digest": packet.digest,
        "status": "COMPLETE",
        "summary": "Synthetic Pair A analysis.",
        "findings": [
            {
                "id": "F-001",
                "severity": "MEDIUM",
                "path": "src/example.py",
                "line": 1,
                "category": "CORRECTNESS",
                "message": "Synthetic addition behavior remains explicit.",
            }
        ],
    }
    worker = WorkerResult.from_value(worker_value)
    reviewer_value = {
        "contract_id": REVIEWER_RESULT_CONTRACT,
        "request_id": request.request_id,
        "packet_digest": packet.digest,
        "worker_result_digest": worker.digest,
        "status": "COMPLETE",
        "summary": "Synthetic Pair B review.",
        "dispositions": [
            {
                "finding_id": "F-001",
                "disposition": "CONFIRM",
                "rationale": "Independent synthetic review completed.",
            }
        ],
    }
    return (
        FakePairATransport(worker_value),
        FakePairBTransport(reviewer_value),
    )
