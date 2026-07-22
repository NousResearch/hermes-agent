from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from conftest import _load_truth_module


def _envelope(
    *,
    turn_id: str = "turn-1",
    speaker_id: str | None = "user-1",
    user_message: str = "Keep replies concise.",
) -> dict:
    return {
        "schema_name": "truth-ledger.source-envelope.v1",
        "schema_version": 1,
        "captured_at": "2026-07-20T23:30:00Z",
        "profile": "default",
        "session_id": "session-1",
        "turn_id": turn_id,
        "origin": {
            "platform": "cli",
            "conversation_id": "conversation-1",
            "chat_id": None,
            "thread_id": "thread-1",
            "chat_type": "private",
            "speaker_id": speaker_id,
        },
        "input": {"user_message": user_message},
        "output": {"assistant_response": "Understood."},
    }


class _FakeLlm:
    def __init__(self, parsed: dict | None = None, error: Exception | None = None):
        self.parsed = parsed
        self.error = error
        self.calls = 0

    def complete_structured(self, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            parsed=self.parsed,
            provider="openai-codex",
            model="gpt-5.6-sol",
        )


class _UpstreamError(RuntimeError):
    status_code = 503


def _fact_document() -> dict:
    return {
        "schema_name": "truth-ledger.fact-candidates.v1",
        "facts": [
            {
                "schema_version": 1,
                "operation": "assert",
                "fact": {
                    "scope": "user",
                    "kind": "preference",
                    "subject": "user",
                    "key": "response.style",
                    "value": "concise",
                },
                "evidence": {
                    "type": "user_stated",
                    "speaker_id": "user-1",
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "conversation_id": "conversation-1",
                    "thread_id": "thread-1",
                },
                "confidence": 1.0,
            }
        ],
    }


def test_process_dry_run_is_non_mutating_and_does_not_call_llm(tmp_path):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope())["ok"] is True
    llm = _FakeLlm(parsed=_fact_document())

    out = asyncio.run(processor.process_pending(root=root, ctx=SimpleNamespace(llm=llm), limit=3, apply=False))

    assert out["ok"] is True
    assert out["dry_run"] is True
    assert out["would_process"] == 1
    assert llm.calls == 0
    assert len(list(spool.pending_dir.glob("*.json"))) == 1
    assert len(list(spool.processing_dir.glob("*.json"))) == 0
    assert not (root / "ledger").exists()


def test_process_apply_extracts_admits_appends_projects_and_acks(tmp_path):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope())["ok"] is True
    llm = _FakeLlm(parsed=_fact_document())

    out = asyncio.run(processor.process_pending(root=root, ctx=SimpleNamespace(llm=llm), limit=3, apply=True))

    assert out["ok"] is True
    assert out["dry_run"] is False
    assert out["claimed"] == 1
    assert out["acked"] == 1, out
    assert out["appended"] == 1
    assert out["active_facts"] == 1
    assert llm.calls == 1
    assert list(spool.pending_dir.glob("*.json")) == []
    assert list(spool.processing_dir.glob("*.json")) == []
    assert list(spool.payloads_dir.glob("*.json")) == []

    ledger_files = list((root / "ledger").glob("*.jsonl"))
    assert len(ledger_files) == 1
    event = json.loads(ledger_files[0].read_text(encoding="utf-8").strip())
    assert event["operation"] == "assert"
    assert event["fact"]["subject"] == "platform-user:cli:user-1"
    assert event["fact"]["key"] == "response.style"
    assert event["fact"]["value"] == "concise"
    assert event["extraction"]["provider"] == "openai-codex"

    current = (root / "views" / "current.jsonl").read_text(encoding="utf-8")
    assert event["fact_id"] in current
    assert not (root / "USER.md").exists()
    assert not (root / "MEMORY.md").exists()
    assert not (root / "gbrain").exists()


def test_process_rejects_user_fact_without_trusted_speaker_and_acks(tmp_path):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope(speaker_id=None))["ok"] is True

    out = asyncio.run(
        processor.process_pending(
            root=root,
            ctx=SimpleNamespace(llm=_FakeLlm(parsed=_fact_document())),
            limit=1,
            apply=True,
        )
    )

    assert out["ok"] is True
    assert out["rejected"] == 1, out
    assert out["appended"] == 0
    assert out["acked"] == 1
    assert list((root / "ledger").glob("*.jsonl")) == []
    assert list(spool.pending_dir.glob("*.json")) == []


def test_process_requeues_transient_extraction_failure_once_per_run(tmp_path):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope())["ok"] is True
    llm = _FakeLlm(error=_UpstreamError("temporary upstream failure"))

    out = asyncio.run(processor.process_pending(root=root, ctx=SimpleNamespace(llm=llm), limit=3, apply=True))

    assert out["ok"] is True
    assert out["claimed"] == 1
    assert out["retried"] == 1
    assert out["acked"] == 0
    assert llm.calls == 1
    assert len(list(spool.pending_dir.glob("*.json"))) == 1
    assert list(spool.processing_dir.glob("*.json")) == []
    assert list((root / "ledger").glob("*.jsonl")) == []


def test_process_enforces_persisted_retry_budget_across_operator_runs(tmp_path):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope())["ok"] is True

    for attempt in range(6):
        claim = spool.claim_next(owner=f"seed-{attempt}")
        assert claim is not None
        spool.retry_processing(
            Path(claim["path"]),
            error_code="upstream_5xx",
            claim_token=claim["claim_token"],
        )

    llm = _FakeLlm(error=_UpstreamError("still unavailable"))
    out = asyncio.run(processor.process_pending(root=root, ctx=SimpleNamespace(llm=llm), limit=1, apply=True))

    assert out["ok"] is True
    assert out["dead_lettered"] == 1
    assert out["retried"] == 0
    assert len(list(spool.dead_letter_dir.glob("*.json"))) == 1
    assert list(spool.pending_dir.glob("*.json")) == []


def test_process_rejects_invalid_limit_and_requires_runtime_context_for_apply(tmp_path):
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"

    too_large = asyncio.run(processor.process_pending(root=root, ctx=None, limit=4, apply=False))
    assert too_large["ok"] is False
    assert too_large["reason"] == "invalid_limit"

    no_ctx = asyncio.run(processor.process_pending(root=root, ctx=None, limit=1, apply=True))
    assert no_ctx["ok"] is False
    assert no_ctx["reason"] == "runtime_context_required"


def test_process_dry_run_on_absent_root_creates_nothing_and_defaults_to_one(tmp_path):
    processor = _load_truth_module("processor")
    root = tmp_path / "absent"

    out = asyncio.run(processor.process_pending(root=root, ctx=None, apply=False))

    assert out["ok"] is True
    assert out["limit"] == 1
    assert out["would_process"] == 0
    assert root.exists() is False


def test_process_apply_only_claims_pending_snapshot_taken_at_start(tmp_path, monkeypatch):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope(turn_id="turn-1"))["ok"] is True

    async def _extract_and_enqueue(**_kwargs):
        assert spool.enqueue(_envelope(turn_id="turn-2"))["ok"] is True
        return {"status": "none"}

    monkeypatch.setattr(processor, "extract_candidates", _extract_and_enqueue)
    out = asyncio.run(
        processor.process_pending(root=root, ctx=SimpleNamespace(llm=object()), limit=3, apply=True)
    )

    assert out["claimed"] == 1
    assert out["pending_after"] == 1


def test_process_reports_stale_ack_as_failure_without_incrementing_acked(
    tmp_path, monkeypatch
):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope())["ok"] is True

    async def _extract_none(**_kwargs):
        return {"status": "none"}

    monkeypatch.setattr(processor, "extract_candidates", _extract_none)
    monkeypatch.setattr(
        processor.TruthSpool,
        "ack_processing",
        lambda self, processing_path, *, claim_token: {
            "ok": False,
            "reason": "stale_claim",
        },
    )

    out = asyncio.run(
        processor.process_pending(
            root=root,
            ctx=SimpleNamespace(llm=object()),
            limit=1,
            apply=True,
        )
    )

    assert out["ok"] is False, out
    assert out["reason"] == "stale_claim"
    assert out["acked"] == 0


def test_process_reports_stale_retry_as_failure_without_incrementing_retried(
    tmp_path, monkeypatch
):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope())["ok"] is True

    async def _extract_retry(**_kwargs):
        return {"status": "retry", "reason": "timeout", "retry_delay_ms": 10}

    monkeypatch.setattr(processor, "extract_candidates", _extract_retry)
    monkeypatch.setattr(
        processor.TruthSpool,
        "retry_processing",
        lambda self, processing_path, error_code, delay_ms=0, *, claim_token: {
            "ok": False,
            "reason": "stale_claim",
        },
    )

    out = asyncio.run(
        processor.process_pending(
            root=root,
            ctx=SimpleNamespace(llm=object()),
            limit=1,
            apply=True,
        )
    )

    assert out["ok"] is False, out
    assert out["reason"] == "stale_claim"
    assert out["retried"] == 0


def test_process_reports_stale_dead_letter_as_failure_without_incrementing_dead_lettered(
    tmp_path, monkeypatch
):
    spool_mod = _load_truth_module("spool")
    processor = _load_truth_module("processor")
    root = tmp_path / "truth-ledger"
    spool = spool_mod.TruthSpool(root)
    assert spool.enqueue(_envelope())["ok"] is True

    async def _extract_dead_letter(**_kwargs):
        return {"status": "dead_letter", "reason": "schema_mismatch"}

    monkeypatch.setattr(processor, "extract_candidates", _extract_dead_letter)
    monkeypatch.setattr(
        processor.TruthSpool,
        "dead_letter",
        lambda self, processing_path, reason, *, claim_token: {
            "ok": False,
            "reason": "stale_claim",
        },
    )

    out = asyncio.run(
        processor.process_pending(
            root=root,
            ctx=SimpleNamespace(llm=object()),
            limit=1,
            apply=True,
        )
    )

    assert out["ok"] is False, out
    assert out["reason"] == "stale_claim"
    assert out["dead_lettered"] == 0
