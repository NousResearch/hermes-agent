from __future__ import annotations

import json
import os
import time
from pathlib import Path


def test_enqueue_creates_pending_file_with_restrictive_mode(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)

    result = spool.enqueue({"event": "assert", "fact_id": "f1"})

    assert result["ok"] is True
    path = Path(result["path"])
    assert path.parent.name == "pending"
    assert path.exists()
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event"] == "assert"
    assert payload["fact_id"] == "f1"


def test_claim_retry_and_dead_letter_flow(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path, soft_count=10, hard_count=20)
    spool.enqueue({"event": "assert", "fact_id": "f1"})

    claim = spool.claim_next(owner="worker-1")
    assert claim is not None
    processing_path = Path(claim["path"])
    assert processing_path.parent.name == "processing"

    retry = spool.retry_processing(processing_path, error_code="SQLITE_BUSY")
    assert retry["ok"] is True
    pending_path = Path(retry["path"])
    assert pending_path.parent.name == "pending"

    claim2 = spool.claim_next(owner="worker-2")
    assert claim2 is not None
    processing_path2 = Path(claim2["path"])

    dead = spool.dead_letter(processing_path2, reason="permanent")
    assert dead["ok"] is True
    dead_path = Path(dead["path"])
    assert dead_path.parent.name == "dead-letter"
    dead_payload = json.loads(dead_path.read_text(encoding="utf-8"))
    assert dead_payload["dead_letter_reason"] == "permanent"


def test_soft_and_hard_caps_are_enforced_without_throwing(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path, soft_count=2, hard_count=3)

    r1 = spool.enqueue({"fact_id": "a"})
    r2 = spool.enqueue({"fact_id": "b"})
    r3 = spool.enqueue({"fact_id": "c"})
    assert r1["ok"] and r2["ok"] and r3["ok"]

    # soft cap sheds oldest pending to dead-letter
    dead_files = list((tmp_path / "spool" / "dead-letter").glob("*.json"))
    assert len(dead_files) >= 1

    # hard cap should fail-open: no exception, explicit rejection.
    strict = spool_mod.TruthSpool(tmp_path / "strict", soft_count=99, hard_count=3)
    assert strict.enqueue({"fact_id": "a"})["ok"]
    assert strict.enqueue({"fact_id": "b"})["ok"]
    assert strict.enqueue({"fact_id": "c"})["ok"]
    r4 = strict.enqueue({"fact_id": "d"})
    assert r4["ok"] is False
    assert r4["reason"] == "queue_hard_cap"


def test_recover_stale_processing_moves_back_to_pending(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)
    spool.enqueue({"fact_id": "stale"})
    claim = spool.claim_next(owner="worker")
    assert claim is not None

    processing_path = Path(claim["path"])
    old = time.time() - 600
    os.utime(processing_path, (old, old))

    moved = spool.recover_stale_processing(stale_seconds=60)
    assert moved == 1
    assert not processing_path.exists()
    assert len(list((tmp_path / "spool" / "pending").glob("*.json"))) == 1
