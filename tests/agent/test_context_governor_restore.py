"""Regression coverage for the Ares Context Governor host binding.

The configured external engine must be discoverable.  A silent fallback to
Hermes' built-in LLM ContextCompressor changes the Ares deterministic-first,
hash-preserving contract and is therefore a conformance failure.
"""

import copy
import hashlib
import json
import os
import shutil
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_engine import ContextEngine
from agent.manual_compression_feedback import summarize_manual_compression
from hermes_state import SessionDB
from plugins.context_engine import load_context_engine
from plugins.context_engine._context_governor import (
    DEFAULT_MAX_PROVENANCE_BYTES,
    ContextGovernorEngine,
    _SummaryLLMResult,
    _SummaryLLMRoute,
)
from plugins.context_engine._context_governor.key_state import ContextGovernorKeyState
from plugins.context_engine._context_governor.protocol import ContextGovernorCommandError
from tools.todo_tool import TODO_INJECTION_HEADER


def _bind_fixture(engine):
    """Route certified fixture calls without opening real secret descriptors."""
    command_args = [
        "--governed-key-fd",
        "71",
        "--governed-snapshot-fd",
        "72",
    ]

    def run_certified(args, payload):
        return engine._run_json([*args, *command_args], payload)

    engine._run_certified_json = run_certified


def test_ares_governor_is_discoverable_as_a_context_engine():
    """Configured Ares ownership must not silently resolve to built-in LLM compression."""
    engine = load_context_engine("ri-context-governor")

    assert engine is not None
    assert isinstance(engine, ContextEngine)
    assert engine.name == "ri-context-governor"
    # Discovery is intentionally separate from activation. A default local
    # install without a certified binary/key pair must not be selected.
    assert engine.is_available() is False


def test_ares_governor_rejects_an_arbitrary_configured_key_path():
    with (
        TemporaryDirectory() as store_dir,
        patch(
            "hermes_cli.config.load_config",
            return_value={
                "context": {
                    "governor": {"receipt_hmac_key_path": "/tmp/not-canonical.key"}
                }
            },
        ),
    ):
        engine = ContextGovernorEngine(
            binary="/tmp/context-governor", store_dir=store_dir
        )
    with pytest.raises(Exception, match="ConfigurationPathOutsideCanonicalState"):
        engine.probe_activation()


def test_protocol_probe_exercises_the_certified_two_phase_wire_contract():
    """Readiness must prove more than a self-reported capabilities document."""
    with patch("hermes_cli.config.load_config", return_value={}):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    calls = []
    candidate = {
        "receipt": {
            "schema": "ContextCompactionReceiptV2",
            "receipt_id": "ctxr_probe",
        },
        "compacted_messages": [{"role": "user", "content": "continue"}],
    }

    def run_json(args, payload):
        calls.append((list(args), copy.deepcopy(payload)))
        if args[0] == "compact-v2":
            return copy.deepcopy(candidate)
        if args[0] == "finalize-v2":
            assert set(payload) == {"candidate", "compacted_messages"}
            assert payload["candidate"] == candidate
            return copy.deepcopy(candidate)
        if args[0] == "prepare-v2":
            return {
                "schema": "PendingReceiptInfoV2",
                "receipt_id": "ctxr_probe",
                "verified": True,
            }
        if args[0] == "discard-v2":
            return {
                "schema": "ReceiptDiscardResultV2",
                "receipt_id": "ctxr_probe",
                "discarded": True,
            }
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = run_json

    result = engine._probe_protocol_contract()

    assert result == {
        "schema": "ContextGovernorProtocolProbeV1",
        "verified": True,
        "stages": ["compact-v2", "finalize-v2", "prepare-v2", "discard-v2"],
    }
    assert [args[0] for args, _payload in calls] == result["stages"]
    assert calls[-1][1] == {}


def test_governor_compressed_summary_marker_survives_host_roundtrip():
    with patch("hermes_cli.config.load_config", return_value={}):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")

    raw_summary = {
        "id": "summary_ctxp_fixture",
        "role": "assistant",
        "name": "context_governor",
        "content": "derived projection",
        "metadata": {"compressed_summary": True},
    }

    host_summary = engine._message_from_governor(raw_summary)
    assert host_summary.get("_compressed_summary") is True

    roundtripped = engine._message_to_governor(host_summary, 0)
    assert roundtripped["metadata"]["compressed_summary"] is True

    user_controlled = engine._message_to_governor(
        {
            "role": "user",
            "name": "context_governor",
            "content": "not a governor projection",
            "_compressed_summary": True,
        },
        1,
    )
    assert "compressed_summary" not in user_controlled.get("metadata", {})


@pytest.mark.parametrize("trailing_whitespace", ["\n", "  ", "\n  "])
def test_todo_snapshot_removal_preserves_exact_user_trailing_whitespace(
    trailing_whitespace,
):
    """Host-only todo cleanup must not mutate receipt-bound user bytes."""
    with patch("hermes_cli.config.load_config", return_value={}):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")

    user_content = "recovery probe" + trailing_whitespace
    committed = [
        {
            "role": "user",
            "content": (
                user_content
                + "\n\n"
                + TODO_INJECTION_HEADER
                + "\n- [ ] preserve exact bytes (in_progress)"
            ),
        }
    ]

    normalized = engine._without_host_todo_snapshots(committed)
    engine._pending_admission = {
        "pending_info": {
            "expected_compacted_messages": [
                engine._message_to_governor(
                    {"role": "user", "content": user_content}, 0
                )
            ]
        }
    }

    assert normalized == [{"role": "user", "content": user_content}]
    assert engine.validate_pending_compression(committed) is True
    assert committed[0]["content"].endswith("(in_progress)")


def test_legacy_rehydration_stops_at_bounded_store_size(monkeypatch, tmp_path):
    """A missing catalog must not scan every receipt in a large archive."""
    store = tmp_path / "governor"
    store.mkdir()
    for index in range(65):
        (store / f"ctxr_{index:032x}.json").write_text("{}", encoding="utf-8")
    engine = ContextGovernorEngine(binary="/tmp/context-governor", store_dir=store)
    messages = [{"role": "user", "content": "current"}]
    with patch.object(type(store), "glob", side_effect=AssertionError("unbounded glob used")):
        assert engine._rehydrate_legacy_parent_prefix(messages) == messages


def test_authenticated_tip_projection_avoids_python_receipt_discovery(
    monkeypatch, tmp_path
):
    """A Rust-authenticated tip must bypass Python catalog/receipt selection."""
    session_id = "authenticated-tip-only"
    engine = ContextGovernorEngine(binary="/tmp/context-governor", store_dir=tmp_path)
    engine.session_id = session_id
    engine._capabilities = {"supports_lineage_tip_projection": True}
    receipt_prefix = [
        {
            "role": "assistant",
            "id": "summary_authenticated",
            "name": "context_governor",
            "content": "authenticated canonical summary",
        },
        {"role": "user", "content": "active task"},
    ]
    incoming = [
        engine._message_to_governor(
            {"role": "assistant", "content": "authenticated canonical summary"}, 0
        ),
        engine._message_to_governor({"role": "user", "content": "active task"}, 1),
        engine._message_to_governor({"role": "user", "content": "new work"}, 2),
    ]
    calls = []

    def run_certified(args, payload):
        calls.append((args, payload))
        return {
            "schema": "LineageTipProjectionV1",
            "session_id": session_id,
            "receipt_id": "ctxr_authenticated",
            "generation": 1,
            "lineage_epoch": 0,
            "compacted_messages": receipt_prefix,
            "verified": True,
        }

    engine._run_certified_json = run_certified
    monkeypatch.setattr(
        type(tmp_path),
        "iterdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Python receipt discovery must not run")
        ),
    )

    assert engine._rehydrate_legacy_parent_prefix(incoming) == receipt_prefix + [
        {"role": "user", "content": "new work"}
    ]
    assert len(calls) == 1
    assert calls[0][0][0] == "lineage-tip-v2"


def test_authenticated_tip_restores_sessiondb_sanitized_trailing_whitespace(tmp_path):
    """A verified tip restores bytes lost only by the durable replay sanitizer."""
    session_id = "authenticated-whitespace-tip"
    engine = ContextGovernorEngine(binary="/tmp/context-governor", store_dir=tmp_path)
    engine.session_id = session_id
    engine._lineage_session_id = session_id
    engine._capabilities = {"supports_lineage_tip_projection": True}
    receipt_prefix = [
        {
            "role": "assistant",
            "id": "summary_authenticated",
            "name": "context_governor",
            "content": "authenticated canonical summary",
        },
        {"role": "user", "content": "active task\n"},
    ]
    suffix = {"role": "assistant", "content": "post-compaction answer"}
    incoming = [
        engine._message_to_governor(
            {"role": "assistant", "content": "authenticated canonical summary"}, 0
        ),
        engine._message_to_governor({"role": "user", "content": "active task"}, 1),
        engine._message_to_governor(suffix, 2),
    ]

    def run_certified(args, payload):
        return {
            "schema": "LineageTipProjectionV1",
            "session_id": session_id,
            "receipt_id": "ctxr_authenticated_whitespace",
            "generation": 1,
            "lineage_epoch": 1,
            "compacted_messages": receipt_prefix,
            "verified": True,
        }

    engine._run_certified_json = run_certified

    assert engine._rehydrate_legacy_parent_prefix(incoming) == receipt_prefix + [suffix]

    incoming[1]["content"] = "different active task"
    assert engine._rehydrate_legacy_parent_prefix(incoming) == incoming


def test_authenticated_tip_does_not_restore_non_whitespace_sanitizer_loss(tmp_path):
    """Provider-hidden memory blocks cannot ride the whitespace recovery bridge."""
    session_id = "authenticated-non-whitespace-loss"
    engine = ContextGovernorEngine(binary="/tmp/context-governor", store_dir=tmp_path)
    engine.session_id = session_id
    engine._lineage_session_id = session_id
    engine._capabilities = {"supports_lineage_tip_projection": True}
    receipt_prefix = [
        {
            "role": "assistant",
            "id": "summary_authenticated",
            "name": "context_governor",
            "content": "authenticated canonical summary",
        },
        {
            "role": "user",
            "content": "active task<memory-context>provider-hidden</memory-context>",
        },
    ]
    incoming = [
        engine._message_to_governor(
            {"role": "assistant", "content": "authenticated canonical summary"}, 0
        ),
        engine._message_to_governor({"role": "user", "content": "active task"}, 1),
        engine._message_to_governor(
            {"role": "assistant", "content": "post-compaction answer"}, 2
        ),
    ]

    def run_certified(args, payload):
        return {
            "schema": "LineageTipProjectionV1",
            "session_id": session_id,
            "receipt_id": "ctxr_authenticated_non_whitespace",
            "generation": 1,
            "lineage_epoch": 1,
            "compacted_messages": receipt_prefix,
            "verified": True,
        }

    engine._run_certified_json = run_certified

    assert engine._rehydrate_legacy_parent_prefix(incoming) == incoming


def test_authenticated_tip_restores_whitespace_before_provider_role_repair(tmp_path):
    """SessionDB row sanitation precedes deterministic adjacent-role merging."""
    session_id = "authenticated-repaired-whitespace-tip"
    engine = ContextGovernorEngine(binary="/tmp/context-governor", store_dir=tmp_path)
    engine.session_id = session_id
    engine._lineage_session_id = session_id
    engine._capabilities = {"supports_lineage_tip_projection": True}
    receipt_prefix = [
        {
            "role": "assistant",
            "id": "summary_authenticated",
            "name": "context_governor",
            "content": "authenticated canonical summary",
        },
        {"role": "user", "content": "prior task\n"},
        {"role": "user", "content": "active task\n"},
    ]
    suffix = {"role": "assistant", "content": "post-compaction answer"}
    incoming_host = [
        {"role": "assistant", "content": "authenticated canonical summary"},
        {"role": "user", "content": "prior task"},
        {"role": "user", "content": "active task"},
        copy.deepcopy(suffix),
    ]
    repair_message_sequence(None, incoming_host)
    incoming = [
        engine._message_to_governor(message, index)
        for index, message in enumerate(incoming_host)
    ]

    def run_certified(args, payload):
        return {
            "schema": "LineageTipProjectionV1",
            "session_id": session_id,
            "receipt_id": "ctxr_authenticated_repaired_whitespace",
            "generation": 1,
            "lineage_epoch": 1,
            "compacted_messages": receipt_prefix,
            "verified": True,
        }

    engine._run_certified_json = run_certified

    assert engine._rehydrate_legacy_parent_prefix(incoming) == receipt_prefix + [suffix]


def _valid_llm_summary(body: str = "checkpoint") -> str:
    return (
        "=== ACTIVE TASK ===\nfinal\n\n"
        "=== ACCEPTANCE GATES ===\nNone\n\n"
        "=== EXACT FALLBACK REFS ===\nNone\n\n"
        "=== SUMMARY LOSSES ===\nNone\n\n"
        f"=== PRIOR CONTEXT SUMMARY ===\n{body}"
    )


def _receipt_id(ordinal: int) -> str:
    return f"ctxr_{ordinal:032x}"


def _summary_result(
    content: str,
    *,
    provider: str = "openai-codex",
    model: str = "gpt-5.6-luna",
) -> _SummaryLLMResult:
    return _SummaryLLMResult(content, _SummaryLLMRoute(provider, model))


def _checkpoint_response(
    generation: int,
    *,
    session_id: str,
    compacted_prefix: list[dict] | None = None,
    summarized_item_ids: list[str] | None = None,
) -> dict:
    """Return a receipt-valid V2 response with one exact raw summary identity."""
    plan_id = f"ctxp_checkpoint_{generation}"
    item_ids = ["ctxi_old"] if summarized_item_ids is None else summarized_item_ids
    return {
        "receipt": {
            "schema": "ContextCompactionReceiptV2",
            "receipt_id": _receipt_id(generation),
            "session_id": session_id,
            "allocation_plan_id": plan_id,
            "original_transcript_blake3": "a" * 64,
            "compacted_transcript_blake3": "b" * 64,
            "original_transcript_sha256": "c" * 64,
            "compacted_transcript_sha256": "f" * 64,
            "lineage_blake3": "d" * 64,
            "lineage_sha256": "e" * 64,
            "original_approx_tokens": 1000,
            "compacted_approx_tokens": 950,
            "token_savings_estimate": 50,
            "generation": generation,
            "covered_original_sources": [
                {
                    "source_id": "ctxs_" + "f" * 64,
                    "content_blake3": "1" * 64,
                    "content_sha256": "2" * 64,
                }
            ],
        },
        "allocation_plan": {
            "plan_id": plan_id,
            "summarized_item_ids": item_ids,
            "items": [
                {"item_id": item_id, "start_index": index}
                for index, item_id in enumerate(item_ids)
            ],
        },
        "compacted_messages": [
            *copy.deepcopy(compacted_prefix or []),
            {
                "id": f"summary_{plan_id}",
                "role": "assistant",
                "name": "context_governor",
                "content": "deterministic extractive summary",
                "metadata": {"compressed_summary": True},
            },
            {"role": "user", "content": "final"},
        ],
    }


def _checkpoint_engine(
    *,
    target_tokens: int,
    llm_output: str,
    checkpoint_strategy: str = "ineffective_only",
    first_generation: int = 1,
    durable_checkpoint_count: int = 0,
    max_checkpoints: int | None = 10,
    session_id: str = "checkpoint-session",
    unsafe_summary_policy: str = "fallback_extract",
    boundary_safe: bool = True,
    compacted_prefix: list[dict] | None = None,
    candidate_finalize_error: Exception | None = None,
    summary_max_chars: int = 8000,
):
    """Return a deterministic-core fixture with a saturated 950/1000 receipt."""
    config = {
        "context": {
            "governor": {
                "summary_mode": "llm",
                "checkpoint_strategy": checkpoint_strategy,
                "max_checkpoints": max_checkpoints,
                "unsafe_summary_policy": unsafe_summary_policy,
                "summary_max_chars": summary_max_chars,
            }
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    engine.session_id = session_id
    engine._lineage_session_id = session_id
    engine._target_tokens = lambda _current: target_tokens
    llm = MagicMock(return_value=_summary_result(llm_output))
    engine._call_summary_llm = llm
    generation = first_generation - 1
    candidate_finalize_failed = False
    compact_requests = []
    search_requests = []
    boundary_requests = []

    def run_json(args, payload):
        nonlocal candidate_finalize_failed, generation
        if args[:3] == ["compact-v2", "--dir", str(engine.store_dir)]:
            current = copy.deepcopy(payload)
            same_generation_retry = bool(
                compact_requests
                and compact_requests[-1].get("messages") == current.get("messages")
                and int(
                    (current.get("policy") or {}).get(
                        "post_finalize_reserve_tokens", 0
                    )
                )
                > int(
                    (compact_requests[-1].get("policy") or {}).get(
                        "post_finalize_reserve_tokens", 0
                    )
                )
            )
            compact_requests.append(current)
            if not same_generation_retry:
                generation += 1
            return _checkpoint_response(
                generation,
                session_id=session_id,
                compacted_prefix=compacted_prefix,
            )
        if args[:3] == ["search", "--dir", str(engine.store_dir)]:
            search_requests.append(list(args))
            query = args[args.index("--query") + 1]
            return [
                {
                    "receipt_id": _receipt_id(10_000 + index),
                    "hit": {
                        "snippet": (
                            f"{query}\n"
                            f"llm_checkpoint_receipt={_receipt_id(10_000 + index)}"
                        )
                    },
                }
                for index in range(durable_checkpoint_count)
            ]
        if args == ["render-prompt-v2"]:
            return {"system": "system", "user": "prompt"}
        if args == ["boundary-audit"]:
            boundary_requests.append(copy.deepcopy(payload))
            return {"safe_to_reinject": boundary_safe}
        if args and args[0] == "finalize-v2":
            candidate = copy.deepcopy(payload["candidate"])
            candidate["compacted_messages"] = copy.deepcopy(
                payload["compacted_messages"]
            )
            is_llm_candidate = any(
                bool(
                    ((message.get("metadata") or {}).get("hermes_metadata") or {}).get(
                        "llm_checkpoint"
                    )
                )
                for message in candidate["compacted_messages"]
            )
            if (
                is_llm_candidate
                and candidate_finalize_error is not None
                and not candidate_finalize_failed
            ):
                candidate_finalize_failed = True
                raise candidate_finalize_error
            response = candidate
            tokens = sum(
                max(1, len(str(message.get("content", ""))) // 4)
                for message in response["compacted_messages"]
            )
            response["receipt"]["compacted_approx_tokens"] = tokens
            response["receipt"]["token_savings_estimate"] = 1000 - tokens
            return response
        if args[:3] == ["prepare-v2", "--dir", str(engine.store_dir)]:
            receipt = payload["receipt"]
            return {
                "schema": "PendingReceiptInfoV2",
                "receipt_id": receipt["receipt_id"],
                "session_id": receipt["session_id"],
                "generation": receipt["generation"],
                "created_utc": "2026-08-16T00:00:00Z",
                "pending_path": f"/tmp/{receipt['receipt_id']}.json",
                "expected_compacted_message_count": len(payload["compacted_messages"]),
                "expected_compacted_transcript_blake3": receipt[
                    "compacted_transcript_blake3"
                ],
                "expected_compacted_transcript_sha256": receipt[
                    "compacted_transcript_sha256"
                ],
                "expected_compacted_messages": copy.deepcopy(
                    payload["compacted_messages"]
                ),
                "verified": True,
            }
        if args[:3] == ["activate-v2", "--dir", str(engine.store_dir)]:
            return {
                "schema": "ReceiptActivationResultV2",
                "receipt_id": payload["receipt_id"],
                "path": f"/tmp/{payload['receipt_id']}.json",
                "activated": True,
                "verified": True,
                "already_activated": False,
            }
        if args[:3] == ["discard-v2", "--dir", str(engine.store_dir)]:
            receipt_id = args[args.index("--receipt") + 1]
            return {
                "schema": "ReceiptDiscardResultV2",
                "receipt_id": receipt_id,
                "discarded": True,
            }
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = run_json
    engine._fixture_compact_requests = compact_requests
    engine._fixture_search_requests = search_requests
    engine._fixture_boundary_requests = boundary_requests
    raw_compress = engine.compress

    def compress_and_commit(messages, *args, **kwargs):
        compacted = raw_compress(messages, *args, **kwargs)
        if engine._pending_admission is not None:
            engine.commit_pending_compression(compacted)
        return compacted

    engine.compress = compress_and_commit
    return engine, llm


def test_generation_limit_invokes_one_bounded_continuation_with_projection_retry():
    engine, _llm = _checkpoint_engine(
        target_tokens=900,
        llm_output=_valid_llm_summary(),
        checkpoint_strategy="after_n:999",
    )
    engine._capabilities = {"supports_lineage_continuation": True}
    original_run_json = engine._run_json
    commands = []

    def run_json(args, payload):
        command = args[0]
        commands.append(command)
        if command == "compact-v2":
            raise ContextGovernorCommandError(
                "compact-v2",
                "lineage_generation_limit",
                {"generation": 33, "maximum_generation": 32},
            )
        if command == "compact-continue-v2":
            response = _checkpoint_response(
                1,
                session_id=engine._governor_session_id(),
            )
            response["receipt"]["lineage_epoch"] = 1
            response["receipt"]["parent_receipt"] = {
                "receipt_id": _receipt_id(32),
                "generation": 32,
                "lineage_epoch": 0,
            }
            response["receipt"]["supersedes_receipt_id"] = _receipt_id(32)
            return response
        return original_run_json(args, payload)

    engine._run_json = run_json
    compacted = engine._compress_once(
        [
            {"role": "assistant", "content": "old context " * 400},
            {"role": "user", "content": "continue after the epoch boundary"},
        ],
        current_tokens=100,
    )

    assert commands.count("compact-v2") == 2
    assert commands.count("compact-continue-v2") == 2
    assert engine.last_compaction_metrics is not None
    assert engine.last_compaction_metrics["host_projection_reserve_retry"] is True
    assert engine._pending_admission is not None
    assert engine.last_outcome["kind"] == "compacted_pending_host_commit"
    assert engine.last_outcome["lineage_epoch"] == 1
    assert engine.validate_pending_compression(compacted) is True


def test_below_threshold_does_not_schedule_governor_compaction():
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    engine.update_model("fixture", context_length=1_000)

    assert engine.should_compress(499) is False


def test_generation_limit_rejects_invalid_continuation_transition():
    engine, _llm = _checkpoint_engine(
        target_tokens=900,
        llm_output=_valid_llm_summary(),
        checkpoint_strategy="after_n:999",
    )
    engine._capabilities = {"supports_lineage_continuation": True}
    original_run_json = engine._run_json

    def run_json(args, payload):
        if args[0] == "compact-v2":
            raise ContextGovernorCommandError(
                "compact-v2",
                "lineage_generation_limit",
                {"generation": 33, "maximum_generation": 32},
            )
        if args[0] == "compact-continue-v2":
            response = _checkpoint_response(
                1,
                session_id=engine._governor_session_id(),
            )
            response["receipt"]["lineage_epoch"] = 0
            response["receipt"]["parent_receipt"] = {
                "receipt_id": _receipt_id(32),
                "generation": 32,
                "lineage_epoch": 0,
            }
            response["receipt"]["supersedes_receipt_id"] = _receipt_id(32)
            return response
        return original_run_json(args, payload)

    engine._run_json = MagicMock(side_effect=run_json)
    messages = [
        {"role": "assistant", "content": "old context " * 400},
        {"role": "user", "content": "continue after the epoch boundary"},
    ]
    compacted = engine._compress_once(messages, current_tokens=100)

    assert compacted == messages
    assert engine._pending_admission is None
    assert engine.last_outcome is not None
    assert engine.last_outcome["kind"] == "compaction_failed_closed"
    assert "invalid authenticated epoch transition" in str(engine.last_error)


def test_governor_failure_is_reported_as_abort_not_successful_noop():
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    engine._run_json = MagicMock(
        side_effect=RuntimeError("parent compacted transcript is not exact prefix")
    )
    messages = [
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "final"},
    ]

    compacted = engine.compress(messages, current_tokens=100)
    feedback = summarize_manual_compression(
        messages,
        compacted,
        100,
        100,
        compression_state=engine,
    )

    assert compacted == messages
    assert engine._last_compress_aborted is True
    assert feedback["aborted"] is True
    assert feedback["headline"].startswith("Compression aborted:")
    assert "No changes" not in feedback["headline"]
    assert "parent compacted transcript" in feedback["note"]


def test_live_engine_retries_pending_activation_before_new_compaction():
    """A post-commit activation outage must not wedge the same process forever."""
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    messages = [{"role": "user", "content": "durable compacted transcript"}]
    expected = [engine._message_to_governor(messages[0], 0)]
    receipt_id = _receipt_id(77)
    engine._pending_admission = {
        "receipt_id": receipt_id,
        "pending_info": {
            "receipt_id": receipt_id,
            "generation": 1,
            "expected_compacted_messages": expected,
        },
        "llm_checkpoint_applied": False,
        "savings_pct": 50.0,
        "exact_fallback_available": True,
        "host_boundary_accepted": True,
    }
    calls = []
    activation_attempts = 0

    def run_json(args, payload):
        nonlocal activation_attempts
        calls.append(args[0])
        if args[0] == "activate-v2":
            activation_attempts += 1
            if activation_attempts == 1:
                raise TimeoutError("synthetic activation outage")
            return {
                "schema": "ReceiptActivationResultV2",
                "receipt_id": payload["receipt_id"],
                "activated": True,
                "verified": True,
                "already_activated": False,
            }
        if args[0] == "compact-v2":
            raise RuntimeError("synthetic next compaction reached")
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = run_json

    first = engine.compress(messages, current_tokens=100)
    assert first == messages
    assert engine._pending_admission is not None
    assert isinstance(engine._last_summary_error, str)
    assert "activation retry failed" in engine._last_summary_error

    second = engine.compress(messages, current_tokens=100)

    assert second == messages
    assert calls == ["activate-v2", "activate-v2", "compact-v2"]
    assert engine._pending_admission is None
    assert engine.compression_count == 1
    assert engine.last_error == "synthetic next compaction reached"


def test_mismatched_pending_projection_is_discarded_before_new_compaction():
    """An inert receipt that cannot match host truth must not block progress."""
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    messages = [{"role": "user", "content": "authoritative current transcript"}]
    receipt_id = _receipt_id(78)
    engine._pending_admission = {
        "receipt_id": receipt_id,
        "pending_info": {
            "receipt_id": receipt_id,
            "generation": 1,
            "expected_compacted_messages": [
                {"role": "user", "content": "different stale projection"}
            ],
        },
        "llm_checkpoint_applied": False,
        "savings_pct": 50.0,
        "exact_fallback_available": True,
        "host_boundary_accepted": False,
    }
    calls = []

    def run_json(args, payload):
        calls.append(args[0])
        if args[0] == "discard-v2":
            return {
                "schema": "ReceiptDiscardResultV2",
                "receipt_id": receipt_id,
                "discarded": True,
            }
        if args[0] == "compact-v2":
            raise RuntimeError("synthetic next compaction reached")
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = run_json

    compacted = engine.compress(messages, current_tokens=100)

    assert compacted == messages
    assert calls == ["discard-v2", "compact-v2"]
    assert engine._pending_admission is None
    assert isinstance(engine.last_warning, str)
    assert "stale pending receipt" in engine.last_warning
    assert engine.last_error == "synthetic next compaction reached"


def test_restart_reconciliation_failure_stays_bound_for_next_turn_retry():
    """One failed bind-time activation must retry on an ordinary later turn."""
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    engine.session_id = "restart-lineage"
    engine._lineage_session_id = "restart-lineage"
    messages = [{"role": "user", "content": "durable restarted transcript"}]
    expected = [engine._message_to_governor(messages[0], 0)]
    receipt_id = _receipt_id(79)
    info = {
        "schema": "PendingReceiptInfoV2",
        "receipt_id": receipt_id,
        "session_id": "restart-lineage",
        "generation": 1,
        "expected_compacted_messages": expected,
        "verified": True,
    }
    calls = []
    activation_attempts = 0

    class RestartedSessionDB:
        def get_messages_as_conversation(
            self,
            session_id,
            repair_alternation=False,
            include_summary_markers=False,
        ):
            assert session_id == "restart-lineage"
            assert repair_alternation is False
            assert include_summary_markers is True
            return copy.deepcopy(messages)

        def get_compression_tip(self, session_id):
            assert session_id == "restart-lineage"
            return ""

    def run_json(args, payload):
        nonlocal activation_attempts
        calls.append(args[0])
        if args[0] == "pending-v2":
            return [copy.deepcopy(info)]
        if args[0] == "activate-v2":
            activation_attempts += 1
            if activation_attempts == 1:
                raise TimeoutError("synthetic restart activation outage")
            return {
                "schema": "ReceiptActivationResultV2",
                "receipt_id": payload["receipt_id"],
                "activated": True,
                "verified": True,
                "already_activated": False,
            }
        if args[0] == "compact-v2":
            raise RuntimeError("synthetic next compaction reached")
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = MagicMock(side_effect=run_json)
    engine._reconcile_pending_receipts(RestartedSessionDB(), "restart-lineage")

    assert engine._pending_admission is not None
    assert isinstance(engine.last_warning, str)
    assert "remains staged and bound" in engine.last_warning

    compacted = engine.compress(messages, current_tokens=100)

    assert compacted == messages
    assert calls == ["pending-v2", "activate-v2", "activate-v2", "compact-v2"]
    assert engine._pending_admission is None
    assert engine.compression_count == 1
    assert engine.last_error == "synthetic next compaction reached"


def test_restart_reconciliation_refuses_marker_blind_reader():
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    engine.session_id = "marker-blind-restart"
    engine._lineage_session_id = "marker-blind-restart"
    receipt_id = _receipt_id(80)
    expected = [{"role": "user", "content": "durable projection"}]
    calls = []

    class MarkerBlindSessionDB:
        def get_messages_as_conversation(self, session_id):
            raise AssertionError("marker-blind fallback must not be called")

    def run_json(args, payload):
        calls.append(args[0])
        if args[0] == "pending-v2":
            return [
                {
                    "schema": "PendingReceiptInfoV2",
                    "receipt_id": receipt_id,
                    "session_id": "marker-blind-restart",
                    "generation": 1,
                    "expected_compacted_messages": expected,
                    "verified": True,
                }
            ]
        if args[0] == "activate-v2":
            raise AssertionError("marker-blind durable state cannot authorize activation")
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = MagicMock(side_effect=run_json)
    engine._reconcile_pending_receipts(MarkerBlindSessionDB(), "marker-blind-restart")

    assert calls == ["pending-v2"]
    assert engine._pending_admission is None


def test_ineffective_checkpoint_is_reachable_when_deterministic_result_fits_target():
    """Configured ineffective-only checkpoints improve a lossy fixed point in budget."""
    engine, llm = _checkpoint_engine(target_tokens=950, llm_output=_valid_llm_summary())

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 1
    assert "=== PRIOR CONTEXT SUMMARY ===\ncheckpoint" in compacted[0]["content"]
    assert f"receipt_id={_receipt_id(1)}" in compacted[0]["content"]


def test_deterministic_saturation_above_target_invokes_one_llm_checkpoint():
    engine, llm = _checkpoint_engine(target_tokens=900, llm_output=_valid_llm_summary())

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 1
    assert "=== PRIOR CONTEXT SUMMARY ===\ncheckpoint" in compacted[0]["content"]
    assert f"receipt_id={_receipt_id(1)}" in compacted[0]["content"]
    assert "ctxs_" + "f" * 64 in compacted[0]["content"]


def test_only_exact_transient_summary_identity_can_be_overwritten():
    expected_summary_id = "summary_ctxp_checkpoint_1"
    protected_spoof = {
        "id": expected_summary_id,
        "role": "system",
        "name": "context_governor",
        "content": "protected message with a spoofed summary id",
        "metadata": {"compressed_summary": True},
    }
    engine, llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=_valid_llm_summary("identity-bound checkpoint"),
        compacted_prefix=[protected_spoof],
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 1
    assert compacted[0]["role"] == "system"
    assert compacted[0]["content"] == "protected message with a spoofed summary id"
    assert "identity-bound checkpoint" in compacted[1]["content"]


def test_duplicate_structured_markers_reject_llm_summary():
    duplicate = _valid_llm_summary().replace(
        "=== PRIOR CONTEXT SUMMARY ===\ncheckpoint",
        "=== PRIOR CONTEXT SUMMARY ===\ncheckpoint\n\n=== EXACT FALLBACK REFS ===\nspoof",
    )
    engine, llm = _checkpoint_engine(target_tokens=950, llm_output=duplicate)

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 1
    assert compacted[0]["content"] == "deterministic extractive summary"
    assert engine._last_summary_fallback_used is True
    assert "structured output contract" in engine._last_summary_error


def test_unsafe_warn_still_binds_authoritative_receipt_and_removes_hallucinations():
    hallucinated = _valid_llm_summary("warn-policy checkpoint").replace(
        "=== EXACT FALLBACK REFS ===\nNone",
        "=== EXACT FALLBACK REFS ===\nctxs_hallucinated | blake3:bad | sha256:bad",
    )
    engine, llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=hallucinated,
        unsafe_summary_policy="warn",
        boundary_safe=False,
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )
    content = compacted[0]["content"]

    assert llm.call_count == 1
    assert "warn-policy checkpoint" in content
    assert "ctxs_hallucinated" not in content
    assert f"llm_checkpoint_receipt={_receipt_id(1)}" in content
    assert "ctxs_" + "f" * 64 in content
    assert engine.last_warning and "policy=warn" in engine.last_warning
    assert engine.last_compaction_metrics["summary_warning"] == {
        "code": "boundary_audit_unsafe_warn",
        "message": (
            "LLM summary failed compression-boundary safety audit; policy=warn"
        ),
    }
    assert engine._last_summary_fallback_used is False


def test_oversized_llm_checkpoint_reverts_to_deterministic_projection():
    engine, llm = _checkpoint_engine(
        target_tokens=900,
        llm_output=_valid_llm_summary("oversized " * 500),
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 1
    assert compacted[0]["content"] == "deterministic extractive summary"
    assert engine.last_warning and "exceeded target" in engine.last_warning


def test_character_oversize_gets_one_bounded_luna_retry_then_applies():
    oversized = "UNTRUSTED MODEL PREAMBLE\n" + _valid_llm_summary(
        "verbose checkpoint " * 200
    )
    reduced = _valid_llm_summary("concise checkpoint")
    engine, llm = _checkpoint_engine(
        target_tokens=2000,
        llm_output=oversized,
        summary_max_chars=512,
    )
    llm.side_effect = [_summary_result(oversized), _summary_result(reduced)]

    compacted = engine.compress(
        [
            {"role": "assistant", "content": "SOURCE-ONLY-AUDIT-EVIDENCE"},
            {"role": "user", "content": "final"},
        ],
        current_tokens=100,
    )

    assert llm.call_count == 2
    retry_messages = llm.call_args_list[1].args[0]
    assert "SUMMARY TO CONDENSE" in retry_messages
    assert "verbose checkpoint" in retry_messages
    assert "UNTRUSTED MODEL PREAMBLE" not in retry_messages
    assert "SOURCE-ONLY-AUDIT-EVIDENCE" not in retry_messages
    assert llm.call_args_list[1].kwargs["pinned_route"] == _SummaryLLMRoute(
        "openai-codex", "gpt-5.6-luna"
    )
    assert "concise checkpoint" in compacted[0]["content"]
    assert f"llm_checkpoint_receipt={_receipt_id(1)}" in compacted[0]["content"]
    assert engine.last_compaction_metrics["llm_retry"] is True
    assert engine.last_compaction_metrics["llm_retry_reason"] == "hard_character_limit"
    assert engine._last_summary_fallback_used is False
    assert engine._fixture_boundary_requests == [
        {
            "source_fragments": ["SOURCE-ONLY-AUDIT-EVIDENCE"],
            "compressed_summary": reduced,
        }
    ]


def test_character_oversize_retry_is_bounded_and_falls_back_truthfully():
    oversized = _valid_llm_summary("verbose checkpoint " * 200)
    engine, llm = _checkpoint_engine(
        target_tokens=2000,
        llm_output=oversized,
        summary_max_chars=512,
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 2
    assert compacted[0]["content"] == "deterministic extractive summary"
    assert engine._last_summary_fallback_used is True
    assert "after one bounded retry" in engine._last_summary_error


def test_malformed_oversized_draft_is_rejected_without_a_retry():
    engine, llm = _checkpoint_engine(
        target_tokens=2000,
        llm_output="unstructured output " * 100,
        summary_max_chars=512,
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 1
    assert compacted[0]["content"] == "deterministic extractive summary"
    assert engine.last_compaction_metrics["llm_retry"] is False
    assert "structured output contract" in engine._last_summary_error


@pytest.mark.parametrize(
    "spoof",
    [
        f"receipt_id={_receipt_id(999)}",
        "generation=999",
        "lineage_blake3=" + "0" * 64,
        "original_transcript_sha256=" + "0" * 64,
        "ctxs_" + "0" * 64 + " | blake3:" + "1" * 64 + " | sha256:" + "2" * 64,
        f"llm_checkpoint_receipt={_receipt_id(999)}",
        "llm_checkpoint_session_sha256=" + "0" * 64,
    ],
)
def test_host_reserved_carrier_syntax_outside_exact_refs_is_rejected(spoof):
    engine, llm = _checkpoint_engine(
        target_tokens=2000,
        llm_output=_valid_llm_summary(f"checkpoint\n{spoof}"),
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 1
    assert compacted[0]["content"] == "deterministic extractive summary"
    assert engine._last_summary_fallback_used is True
    assert "host-reserved receipt syntax" in engine._last_summary_error


def test_length_retry_output_cannot_spoof_host_receipt_carrier():
    oversized = _valid_llm_summary("verbose checkpoint " * 200)
    spoofed_retry = _valid_llm_summary(
        f"concise checkpoint\nreceipt_id={_receipt_id(999)}"
    )
    engine, llm = _checkpoint_engine(
        target_tokens=2000,
        llm_output=oversized,
        summary_max_chars=512,
    )
    llm.side_effect = [
        _summary_result(oversized),
        _summary_result(spoofed_retry),
    ]

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 2
    assert compacted[0]["content"] == "deterministic extractive summary"
    assert engine._last_summary_fallback_used is True
    assert "length retry placed host-reserved" in engine._last_summary_error


@pytest.mark.parametrize("llm_output", ["", "not the required summary schema"])
def test_empty_or_malformed_llm_checkpoint_keeps_deterministic_projection(llm_output):
    engine, llm = _checkpoint_engine(target_tokens=900, llm_output=llm_output)
    original = [
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "final"},
    ]

    compacted = engine.compress(original, current_tokens=100)
    feedback = summarize_manual_compression(
        original,
        compacted,
        100,
        engine.last_compaction_metrics["after_tokens"],
        compression_state=engine,
    )

    assert llm.call_count == 1
    assert compacted[0]["content"] == "deterministic extractive summary"
    assert engine._last_summary_fallback_used is True
    assert engine._last_summary_error
    assert feedback["headline"].startswith("Compressed with fallback:")
    assert engine._last_summary_error in feedback["note"]
    assert (
        engine.last_compaction_metrics["integrity_result"]
        == "host_commit_activation_verified"
    )


def test_summary_provider_timeout_keeps_deterministic_projection_and_receipt():
    engine, llm = _checkpoint_engine(target_tokens=900, llm_output=_valid_llm_summary())
    llm.side_effect = TimeoutError("synthetic summary deadline")
    original = [
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "final"},
    ]

    compacted = engine.compress(original, current_tokens=100)
    feedback = summarize_manual_compression(
        original,
        compacted,
        100,
        engine.last_compaction_metrics["after_tokens"],
        compression_state=engine,
    )

    assert llm.call_count == 1
    assert compacted[0]["content"] == "deterministic extractive summary"
    metrics = engine.last_compaction_metrics
    assert metrics["summary_id"] == _receipt_id(1)
    assert metrics["llm_call_reason"].endswith(":fallback_extract")
    assert metrics["integrity_result"] == "host_commit_activation_verified"
    assert engine._last_summary_fallback_used is True
    assert "synthetic summary deadline" in engine._last_summary_error
    assert feedback["headline"].startswith("Compressed with fallback:")
    assert engine._last_summary_error in feedback["note"]


def test_empty_compress_clears_stale_summary_fallback_state():
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    engine._last_summary_fallback_used = True
    engine._last_summary_error = "stale fallback"
    engine._last_compress_aborted = True

    assert engine.compress([], current_tokens=0) == []
    assert engine._last_summary_fallback_used is False
    assert engine._last_summary_error is None
    assert engine._last_compress_aborted is False


def test_candidate_finalize_failure_falls_back_to_deterministic_projection():
    engine, llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=_valid_llm_summary("candidate"),
        candidate_finalize_error=RuntimeError("candidate rejected"),
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == 1
    assert compacted[0]["content"] == "deterministic extractive summary"
    assert engine._last_compress_aborted is False
    assert engine._last_summary_fallback_used is True
    assert "candidate rejected" in engine._last_summary_error
    assert engine.last_compaction_metrics["integrity_result"] == (
        "host_commit_activation_verified"
    )


def test_hybrid_checkpoint_requests_zero_minimum_net_savings():
    engine, _llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=_valid_llm_summary(),
    )

    engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert engine._fixture_compact_requests[0]["policy"]["min_net_savings_tokens"] == 0


def test_after_n_checkpoint_uses_epoch_aware_durable_ordinal():
    engine, _llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=_valid_llm_summary(),
        checkpoint_strategy="after_n:3",
    )
    engine._policy["max_lineage_generation"] = 32
    response = _checkpoint_response(1, session_id=engine._governor_session_id())
    response["receipt"]["lineage_epoch"] = 1

    due, reason = engine._llm_checkpoint_decision(response, target_tokens=950)

    assert due is True
    assert reason == "after_n:3:ordinal:33"


def test_governor_config_reaches_rust_policy_owner():
    config = {
        "context": {
            "governor": {
                "unsafe_summary_policy": "fail_closed",
                "checkpoint_strategy": "after_n:3",
                "max_checkpoints": 4,
                "token_budget": 1234,
                "protect_first_n": 2,
                "protect_last_n": 5,
            }
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")

    assert engine._policy["unsafe_summary_policy"] == "fail_closed"
    assert engine._checkpoint_strategy_json() == {"after_n": 3}
    assert engine._max_checkpoints() == 4
    assert engine._target_tokens(99_999) == 1234
    assert engine.protect_first_n == 2
    assert engine.protect_last_n == 5


def test_secondary_summary_uses_auxiliary_compression_route_by_default():
    config = {
        "auxiliary": {
            "compression": {"provider": "ollama-launch", "model": "glm-5.2:cloud"}
        },
        "context": {"governor": {"summary_mode": "llm"}},
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")
    engine.update_model(
        "primary-model",
        context_length=100_000,
        provider="primary-provider",
        base_url="https://primary.invalid/v1",
        api_key="primary-secret",
        api_mode="responses",
    )

    def fake_call_llm(**kwargs):
        kwargs["route_info"].update({
            "provider": "ollama-launch",
            "model": "glm-5.2:cloud",
        })
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
        )

    engine.last_compaction_metrics = {}
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm) as call,
    ):
        result = engine._call_summary_llm("prompt", 100, system_prompt="system")

    assert result == _summary_result(
        "summary", provider="ollama-launch", model="glm-5.2:cloud"
    )
    kwargs = call.call_args.kwargs
    assert "provider" not in kwargs
    assert "model" not in kwargs
    assert kwargs["task"] == "compression"
    assert kwargs["main_runtime"]["provider"] == "primary-provider"
    assert engine.last_compaction_metrics["summarizer_provider"] == "ollama-launch"
    assert engine.last_compaction_metrics["summarizer_model"] == "glm-5.2:cloud"


def test_secondary_summary_rejects_fallback_route_mismatch():
    config = {
        "auxiliary": {
            "compression": {"provider": "ollama-launch", "model": "glm-5.2:cloud"}
        },
        "context": {"governor": {"summary_mode": "llm"}},
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")

    def fake_call_llm(**kwargs):
        kwargs["route_info"].update({
            "provider": "openai-codex",
            "model": "gpt-5.4",
        })
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
        )

    engine.last_compaction_metrics = {}
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm),
        pytest.raises(RuntimeError, match="route changed during fallback"),
    ):
        engine._call_summary_llm("prompt", 100, system_prompt="system")

    assert engine.last_compaction_metrics["summarizer_provider"] == "openai-codex"
    assert engine.last_compaction_metrics["summarizer_model"] == "gpt-5.4"


def test_summary_retry_pins_first_actual_route_across_config_mutation():
    with patch("hermes_cli.config.load_config", return_value={}):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")
    route_config = {"provider": "openai-codex", "model": "gpt-5.6-luna"}

    def fake_resolve(_task, provider, model, base_url, api_key):
        return (
            provider or route_config["provider"],
            model or route_config["model"],
            base_url,
            api_key,
            None,
        )

    def fake_call_llm(**kwargs):
        provider = kwargs.get("provider") or route_config["provider"]
        model = kwargs.get("model") or route_config["model"]
        kwargs["route_info"].update({"provider": provider, "model": model})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
        )

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            side_effect=fake_resolve,
        ),
        patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm) as call,
    ):
        first = engine._call_summary_llm("first", 100, system_prompt="system")
        assert first is not None
        route_config.update({"provider": "other", "model": "changed"})
        second = engine._call_summary_llm(
            "retry", 100, system_prompt="system", pinned_route=first.route
        )

    assert second is not None
    assert (
        first.route == second.route == _SummaryLLMRoute("openai-codex", "gpt-5.6-luna")
    )
    assert call.call_args_list[1].kwargs["provider"] == "openai-codex"
    assert call.call_args_list[1].kwargs["model"] == "gpt-5.6-luna"


def test_default_provenance_budget_covers_a_tool_heavy_recursive_suffix():
    """A normal 256-message suffix must fit before checkpoint evaluation."""
    with patch("hermes_cli.config.load_config", return_value={}):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")

    conservative_reference_bytes = 1024

    assert engine._policy["max_provenance_bytes"] >= (
        256 * conservative_reference_bytes
    )


def test_default_provenance_budget_is_large_but_still_bounded():
    """Large initial transcripts must reach receipt/LLM admission before failing."""
    with patch("hermes_cli.config.load_config", return_value={}):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")

    assert engine._policy["max_provenance_bytes"] == DEFAULT_MAX_PROVENANCE_BYTES
    assert DEFAULT_MAX_PROVENANCE_BYTES == 16 * 1024 * 1024
    assert DEFAULT_MAX_PROVENANCE_BYTES < 64 * 1024 * 1024


def test_after_n_checkpoint_policy_calls_llm_only_on_intended_boundary():
    engine, llm = _checkpoint_engine(
        target_tokens=900,
        llm_output=_valid_llm_summary(),
        checkpoint_strategy="after_n:2",
    )
    messages = [
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "final"},
    ]

    first = engine.compress(messages, current_tokens=100)
    second = engine.compress(messages, current_tokens=100)

    assert first[0]["content"] == "deterministic extractive summary"
    assert "=== PRIOR CONTEXT SUMMARY ===\ncheckpoint" in second[0]["content"]
    assert f"receipt_id={_receipt_id(2)}" in second[0]["content"]
    assert llm.call_count == 1
    assert engine.compression_count == 2
    assert engine._llm_checkpoint_count == 1


def test_after_n_uses_receipt_generation_after_desktop_restart():
    """A fresh process must still honor the persisted generation-2 boundary."""
    engine, llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=_valid_llm_summary("restart-safe checkpoint"),
        checkpoint_strategy="after_n:2",
        first_generation=2,
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert engine.compression_count == 2
    assert llm.call_count == 1
    assert "restart-safe checkpoint" in compacted[0]["content"]
    assert (
        "after_n:2:ordinal:2:applied"
        == (engine.last_compaction_metrics["llm_call_reason"])
    )


@pytest.mark.parametrize(
    ("durable_checkpoint_count", "expected_llm_calls", "expected_reason"),
    [
        (0, 1, "after_n:2:ordinal:22:applied"),
        (10, 0, "checkpoint_limit_reached"),
    ],
)
def test_checkpoint_maximum_uses_applied_durable_count_not_generation(
    durable_checkpoint_count,
    expected_llm_calls,
    expected_reason,
):
    engine, llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=_valid_llm_summary(),
        checkpoint_strategy="after_n:2",
        first_generation=22,
        durable_checkpoint_count=durable_checkpoint_count,
        max_checkpoints=10,
    )

    engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    assert llm.call_count == expected_llm_calls
    assert engine.last_compaction_metrics["llm_call_reason"] == expected_reason
    assert len(engine._fixture_search_requests) == 1


def test_durable_checkpoint_count_is_session_scoped_and_marker_bound():
    session_id = "session-alpha"
    engine, _llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=_valid_llm_summary(),
        session_id=session_id,
    )
    session_marker = (
        "llm_checkpoint_session_sha256="
        + hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    )
    valid_receipt = _receipt_id(101)
    wrong_session_receipt = _receipt_id(102)
    mismatched_receipt = _receipt_id(103)
    engine._run_json = MagicMock(
        return_value=[
            {
                "receipt_id": valid_receipt,
                "hit": {
                    "snippet": (
                        f"{session_marker}\nllm_checkpoint_receipt={valid_receipt}"
                    )
                },
            },
            {
                "receipt_id": valid_receipt,
                "hit": {
                    "snippet": (
                        f"{session_marker}\nllm_checkpoint_receipt={valid_receipt}"
                    )
                },
            },
            {
                "receipt_id": wrong_session_receipt,
                "hit": {
                    "snippet": (
                        "llm_checkpoint_session_sha256="
                        + "0" * 64
                        + f"\nllm_checkpoint_receipt={wrong_session_receipt}"
                    )
                },
            },
            {
                "receipt_id": mismatched_receipt,
                "hit": {
                    "snippet": (
                        f"{session_marker}\nllm_checkpoint_receipt={valid_receipt}"
                    )
                },
            },
            {
                "receipt_id": "ctxr_not_a_verified_id",
                "hit": {
                    "snippet": (
                        f"{session_marker}\n"
                        "llm_checkpoint_receipt=ctxr_not_a_verified_id"
                    )
                },
            },
        ]
    )

    assert engine._durable_llm_checkpoint_count(10) == 1
    search_args = engine._run_json.call_args.args[0]
    assert search_args[search_args.index("--query") + 1] == session_marker


def test_checkpoint_history_failure_is_structured_visible_and_redacted():
    engine, _llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=_valid_llm_summary(),
    )
    secret = "sk-supersecretcredential123456"
    engine._run_json = MagicMock(
        side_effect=RuntimeError(f"provider failed api_key={secret}")
    )
    engine.last_compaction_metrics = {}

    assert engine._durable_llm_checkpoint_count(10) is None

    warning = engine.last_compaction_metrics["summary_warning"]
    assert warning["code"] == "checkpoint_history_unavailable"
    assert warning["message"] == engine.last_warning
    assert secret not in warning["message"]
    assert "api_key=***" in warning["message"]


def test_llm_fallback_refs_are_replaced_by_authoritative_receipt_carrier():
    hallucinated = _valid_llm_summary().replace(
        "=== EXACT FALLBACK REFS ===\nNone",
        "=== EXACT FALLBACK REFS ===\nctxs_hallucinated | blake3:bad | sha256:bad",
    )
    engine, llm = _checkpoint_engine(
        target_tokens=950,
        llm_output=hallucinated,
        checkpoint_strategy="after_n:1",
    )

    compacted = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )
    content = compacted[0]["content"]

    assert llm.call_count == 1
    assert "ctxs_hallucinated" not in content
    assert f"receipt_id={_receipt_id(1)}" in content
    assert "lineage_blake3=" + "d" * 64 in content
    assert "ctxs_" + "f" * 64 in content
    assert "blake3:" + "1" * 64 in content
    assert "sha256:" + "2" * 64 in content
    assert content.splitlines().count(f"receipt_id={_receipt_id(1)}") == 1
    assert content.splitlines().count("generation=1") == 1
    assert content.splitlines().count("lineage_blake3=" + "d" * 64) == 1
    assert content.splitlines().count("original_transcript_sha256=" + "c" * 64) == 1


def test_compaction_metrics_expose_deterministic_llm_and_receipt_boundaries():
    engine, _llm = _checkpoint_engine(
        target_tokens=900,
        llm_output=_valid_llm_summary(),
    )

    engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )

    metrics = engine.get_status()["last_compaction_metrics"]
    assert metrics["passes"] == 2
    assert metrics["llm_call"] is True
    assert metrics["llm_call_reason"].endswith(":applied")
    assert metrics["llm_latency_ms"] is not None
    assert metrics["summary_id"] == _receipt_id(1)
    assert metrics["deterministic_reduction_tokens"] is not None
    assert metrics["after_tokens"] is not None
    assert metrics["integrity_result"] == "host_commit_activation_verified"


def test_fixed_point_without_new_summary_items_still_calls_secondary_llm():
    """A fixed point enhances the governed projection, not only newly omitted turns."""
    engine, llm = _checkpoint_engine(
        target_tokens=90,
        llm_output=_valid_llm_summary("fixed-point checkpoint"),
    )
    response = _checkpoint_response(
        1,
        session_id=engine.session_id,
        summarized_item_ids=[],
    )
    raw_summary = response["compacted_messages"][0]
    host_summary = engine._message_from_governor(raw_summary)
    host_summary["_context_governor_summary_id"] = raw_summary["id"]
    engine._run_json = MagicMock(
        side_effect=[
            {"system": "system", "user": "prompt"},
            {"safe_to_reinject": True},
        ]
    )
    engine.last_compaction_metrics = {"llm_call": False}

    compacted = engine._enhance_with_llm_summary(
        [
            host_summary,
            {"role": "user", "content": "final"},
        ],
        [{"role": "user", "content": "final"}],
        response,
        None,
    )

    assert llm.call_count == 1
    assert engine.last_compaction_metrics["llm_call"] is True
    assert (
        "=== PRIOR CONTEXT SUMMARY ===\nfixed-point checkpoint"
        in compacted[0]["content"]
    )
    assert f"receipt_id={_receipt_id(1)}" in compacted[0]["content"]


def test_missing_summary_projection_does_not_claim_an_llm_call():
    engine, llm = _checkpoint_engine(
        target_tokens=90,
        llm_output=_valid_llm_summary(),
    )
    engine.last_compaction_metrics = {
        "llm_call": False,
        "llm_call_reason": "checkpoint_ready",
    }

    compacted = engine._enhance_with_llm_summary(
        [{"role": "user", "content": "final"}],
        [{"role": "user", "content": "final"}],
        {"receipt": {}, "allocation_plan": {}},
        None,
    )

    assert compacted == [{"role": "user", "content": "final"}]
    assert llm.call_count == 0
    assert engine.last_compaction_metrics["llm_call"] is False
    assert (
        engine.last_compaction_metrics["llm_call_reason"]
        == "summary_projection_unavailable"
    )
    assert engine.last_compaction_metrics["summary_fallback_reason"] == (
        "deterministic summary carrier was unavailable; retaining the "
        "receipt-backed deterministic projection"
    )
    assert engine._last_summary_fallback_used is True


def test_host_todo_snapshot_does_not_block_recursive_llm_checkpoint():
    """Host-only todo state must not invalidate the receipt-backed parent prefix."""
    config = {
        "context": {
            "governor": {
                "summary_mode": "llm",
                "checkpoint_strategy": "after_n:2",
            }
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    engine._target_tokens = lambda current_tokens: 900
    llm = MagicMock(
        return_value=_summary_result(_valid_llm_summary("recursive checkpoint"))
    )
    engine._call_summary_llm = llm
    parent_projection = None
    generation = 0
    last_compact_payload = None

    def run_json(args, payload):
        nonlocal generation, parent_projection, last_compact_payload
        if args[:3] == ["compact-v2", "--dir", str(engine.store_dir)]:
            incoming = payload["messages"]
            if (
                parent_projection is not None
                and incoming[: len(parent_projection)] != parent_projection
            ):
                raise RuntimeError(
                    "parent compacted transcript is not the exact child-input prefix"
                )
            same_generation_retry = bool(
                last_compact_payload
                and last_compact_payload.get("messages") == payload.get("messages")
                and int(
                    (payload.get("policy") or {}).get(
                        "post_finalize_reserve_tokens", 0
                    )
                )
                > int(
                    (last_compact_payload.get("policy") or {}).get(
                        "post_finalize_reserve_tokens", 0
                    )
                )
            )
            if not same_generation_retry:
                generation += 1
            last_compact_payload = copy.deepcopy(payload)
            return _checkpoint_response(generation, session_id="hermes-session")
        if args == ["render-prompt-v2"]:
            return {"system": "system", "user": "prompt"}
        if args == ["boundary-audit"]:
            return {"safe_to_reinject": True}
        if args[:3] == ["search", "--dir", str(engine.store_dir)]:
            return []
        if args and args[0] == "finalize-v2":
            response = copy.deepcopy(payload["candidate"])
            response["compacted_messages"] = copy.deepcopy(
                payload["compacted_messages"]
            )
            parent_projection = response["compacted_messages"]
            tokens = sum(
                max(1, len(str(message.get("content", ""))) // 4)
                for message in parent_projection
            )
            response["receipt"]["compacted_approx_tokens"] = tokens
            response["receipt"]["token_savings_estimate"] = 1000 - tokens
            return response
        if args[:3] == ["prepare-v2", "--dir", str(engine.store_dir)]:
            receipt = payload["receipt"]
            return {
                "schema": "PendingReceiptInfoV2",
                "receipt_id": receipt["receipt_id"],
                "session_id": receipt["session_id"],
                "generation": receipt["generation"],
                "created_utc": "2026-08-16T00:00:00Z",
                "pending_path": f"/tmp/{receipt['receipt_id']}.json",
                "expected_compacted_message_count": len(payload["compacted_messages"]),
                "expected_compacted_transcript_blake3": receipt[
                    "compacted_transcript_blake3"
                ],
                "expected_compacted_transcript_sha256": receipt[
                    "compacted_transcript_sha256"
                ],
                "expected_compacted_messages": copy.deepcopy(
                    payload["compacted_messages"]
                ),
                "verified": True,
            }
        if args[:3] == ["activate-v2", "--dir", str(engine.store_dir)]:
            return {
                "schema": "ReceiptActivationResultV2",
                "receipt_id": payload["receipt_id"],
                "activated": True,
                "verified": True,
                "already_activated": False,
            }
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = run_json
    first = engine.compress(
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "final"}],
        current_tokens=100,
    )
    engine.commit_pending_compression(first)
    first[-1]["content"] += f"\n\n{TODO_INJECTION_HEADER}\n- [>] reproduce"

    second = engine.compress(first, current_tokens=100)
    engine.commit_pending_compression(second)

    # A third generation must remain on the authenticated parent chain. The
    # generation-2 LLM checkpoint is already applied; generation 3 is a normal
    # deterministic pass that must not strand the session or lose exact refs.
    third = engine.compress(
        second
        + [
            {"role": "assistant", "content": "third-generation work"},
            {"role": "user", "content": "third-generation active task"},
        ],
        current_tokens=100,
    )
    engine.commit_pending_compression(third)

    assert llm.call_count == 1
    assert "=== PRIOR CONTEXT SUMMARY ===\nrecursive checkpoint" in second[0]["content"]
    assert f"receipt_id={_receipt_id(2)}" in second[0]["content"]
    assert engine.compression_count == 3
    assert engine.last_receipt_id == _receipt_id(3)
    assert engine.last_compaction_metrics["integrity_result"] == (
        "host_commit_activation_verified"
    )
    assert engine.last_compaction_metrics["exact_fallback_available"] is True
    assert engine.last_error is None


def test_host_projection_growth_retries_once_with_rust_enforced_reserve():
    compacted_prefix = [
        {
            "role": "assistant",
            "content": "",
            "metadata": {
                "tool_calls": [
                    {
                        "id": "call_missing_result",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ]
            },
        }
    ]
    engine, _llm = _checkpoint_engine(
        target_tokens=128_000,
        llm_output=_valid_llm_summary(),
        checkpoint_strategy="after_n:999",
        compacted_prefix=compacted_prefix,
    )
    messages = [
        {"role": "user", "content": "continue the verified task"},
        {"role": "assistant", "content": "historical work " * 400},
        {"role": "user", "content": "current task"},
    ]

    engine.compress(messages, current_tokens=200_000)

    requests = getattr(engine, "_fixture_compact_requests")
    assert len(requests) == 2
    assert requests[0]["policy"]["post_finalize_reserve_tokens"] == 0
    assert requests[1]["policy"]["post_finalize_reserve_tokens"] > 0
    assert engine.last_compaction_metrics is not None
    assert engine.last_compaction_metrics["host_projection_reserve_retry"] is True
    assert engine.last_compaction_metrics["post_finalize_reserve_tokens"] == requests[
        1
    ]["policy"]["post_finalize_reserve_tokens"]


def test_synthetic_notification_does_not_reserve_anchor_already_in_candidate():
    engine, _llm = _checkpoint_engine(
        target_tokens=128_000,
        llm_output=_valid_llm_summary(),
        checkpoint_strategy="after_n:999",
    )
    messages = [
        {"role": "user", "content": "human intent " * 20},
        {"role": "assistant", "content": "historical work " * 400},
        {
            "role": "user",
            "content": (
                "[IMPORTANT: Background process proc_fixture completed normally "
                "(exit code 0).]"
            ),
        },
    ]

    compacted = engine.compress(messages, current_tokens=200_000)

    requests = getattr(engine, "_fixture_compact_requests")
    anchor_tokens = engine._estimate_message_tokens(messages[0])
    assert requests[0]["policy"]["post_finalize_reserve_tokens"] == 0
    assert all(
        request["policy"]["post_finalize_reserve_tokens"] < anchor_tokens
        for request in requests
    )
    assert not any(
        messages[0]["content"] in str(message.get("content") or "")
        for message in compacted
        if isinstance(message, dict)
    )
    assert engine.last_compaction_metrics is not None


def test_synthetic_notification_reserves_missing_real_user_for_host_finalization():
    engine, _llm = _checkpoint_engine(
        target_tokens=128_000,
        llm_output=_valid_llm_summary(),
        checkpoint_strategy="after_n:999",
    )
    human_anchor = "human intent " * 20
    synthetic_notification = (
        "[IMPORTANT: Background process proc_fixture completed normally "
        "(exit code 0).]"
    )
    original_run_json = engine._run_json

    def run_json(args, payload, *, pass_fds=()):
        assert not pass_fds
        response = original_run_json(args, payload)
        if args and args[0] in {"compact-v2", "compact-continue-v2"}:
            response = copy.deepcopy(response)
            response["compacted_messages"][-1] = {
                "role": "user",
                "content": synthetic_notification,
            }
        return response

    engine._run_json = run_json
    messages = [
        {"role": "user", "content": human_anchor},
        {"role": "assistant", "content": "historical work " * 400},
        {"role": "user", "content": synthetic_notification},
    ]

    compacted = engine.compress(messages, current_tokens=200_000)

    requests = getattr(engine, "_fixture_compact_requests")
    assert len(requests) == 2
    assert requests[0]["policy"]["post_finalize_reserve_tokens"] == 0
    assert requests[1]["policy"]["post_finalize_reserve_tokens"] > 0
    assert any(
        human_anchor in str(message.get("content") or "")
        for message in compacted
        if isinstance(message, dict)
    )
    assert engine.last_compaction_metrics is not None
    assert engine.last_compaction_metrics["host_projection_reserve_retry"] is True
    assert engine.last_compaction_metrics["post_finalize_reserve_tokens"] == requests[
        1
    ]["policy"]["post_finalize_reserve_tokens"]


def test_background_notification_compaction_binds_host_real_user_anchor():
    """A synthetic current turn must not make the host reject the prepared receipt."""
    config = {
        "context": {
            "governor": {
                "summary_mode": "extractive",
                "checkpoint_strategy": "off",
            }
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    engine._target_tokens = lambda current_tokens: 900
    synthetic_notification = (
        "[IMPORTANT: Background process proc_deadbeef completed normally "
        "(exit code 0).]"
    )

    def run_json(args, payload):
        if args[:3] == ["compact-v2", "--dir", str(engine.store_dir)]:
            response = _checkpoint_response(1, session_id="background-session")
            response["compacted_messages"][-1] = {
                "role": "user",
                "content": synthetic_notification,
            }
            return response
        if args and args[0] == "finalize-v2":
            response = copy.deepcopy(payload["candidate"])
            response["compacted_messages"] = copy.deepcopy(
                payload["compacted_messages"]
            )
            response["receipt"]["compacted_approx_tokens"] = 100
            response["receipt"]["token_savings_estimate"] = 900
            return response
        if args[:3] == ["prepare-v2", "--dir", str(engine.store_dir)]:
            receipt = payload["receipt"]
            return {
                "schema": "PendingReceiptInfoV2",
                "receipt_id": receipt["receipt_id"],
                "session_id": receipt["session_id"],
                "generation": receipt["generation"],
                "created_utc": "2026-08-26T00:00:00Z",
                "pending_path": f"/tmp/{receipt['receipt_id']}.json",
                "expected_compacted_message_count": len(payload["compacted_messages"]),
                "expected_compacted_transcript_blake3": receipt[
                    "compacted_transcript_blake3"
                ],
                "expected_compacted_transcript_sha256": receipt[
                    "compacted_transcript_sha256"
                ],
                "expected_compacted_messages": copy.deepcopy(
                    payload["compacted_messages"]
                ),
                "verified": True,
            }
        if args[:3] == ["activate-v2", "--dir", str(engine.store_dir)]:
            return {
                "schema": "ReceiptActivationResultV2",
                "receipt_id": payload["receipt_id"],
                "activated": True,
                "verified": True,
                "already_activated": False,
            }
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = run_json
    original = [
        {"role": "user", "content": "Completely fix context compaction."},
        {"role": "assistant", "content": "Working on the fix."},
        {"role": "user", "content": synthetic_notification},
    ]

    compacted = engine.compress(original, current_tokens=100)
    before_host_boundary = copy.deepcopy(compacted)
    from agent.conversation_compression import _ensure_compressed_has_user_turn

    _ensure_compressed_has_user_turn(original, compacted)

    assert compacted == before_host_boundary
    assert any(
        message.get("role") == "user"
        and message.get("content") == "Completely fix context compaction."
        for message in compacted
    )
    assert engine.validate_pending_compression(compacted) is True
    assert engine.commit_pending_compression(compacted) is True


@pytest.mark.skip(
    reason="Encodes the Ares staged-receipt boundary semantics (containment-recognized "
           "anchors in _ensure_compressed_has_user_turn) that upstream hermes-agent does "
           "not have; the adapter degrades to the upstream direct-commit flow. Revisit if "
           "upstream adopts staged receipt activation."
)
def test_zero_user_continuation_anchor_is_idempotent_at_host_boundary():
    """The host must not append another continuation after adapter-side repair."""
    from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT
    from agent.conversation_compression import _ensure_compressed_has_user_turn

    notification = "[IMPORTANT: Background process proc_zero completed normally.]"
    original = [
        {"role": "assistant", "content": "runtime-only session"},
        {"role": "user", "content": notification},
    ]
    compacted = [
        {"role": "assistant", "content": "deterministic extractive summary"},
        {
            "role": "user",
            "content": (f"{notification}\n\n{COMPRESSION_CONTINUATION_USER_CONTENT}"),
        },
    ]
    before_host_boundary = copy.deepcopy(compacted)

    _ensure_compressed_has_user_turn(original, compacted)

    assert compacted == before_host_boundary


def test_host_alternation_repair_is_bound_into_the_receipt_projection():
    """The finalized receipt must describe the host-repaired transcript.

    Regression: compact-v2 emits compacted tool-result notes as consecutive
    assistant messages (inline notes, not provider tool results), which
    Hermes' conversation loop merges in memory immediately after
    ``compress()`` returns. If the receipt bound the un-repaired transcript,
    the next compaction's child input no longer exactly prefixes the stored
    parent and the Rust recursive-lineage check rejects it — deterministic
    compaction became one-shot per session (every later attempt failed and
    the LLM checkpoint could never reach its generation boundary).
    """
    config = {
        "context": {
            "governor": {
                "summary_mode": "llm",
                "checkpoint_strategy": "off",
            }
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary="/tmp/context-governor")
    _bind_fixture(engine)
    engine._target_tokens = lambda current_tokens: 900
    parent_projection = None
    generation = 0
    last_compact_payload = None

    def compact_response(gen: int) -> dict:
        return {
            "receipt": {
                "schema": "ContextCompactionReceiptV2",
                "receipt_id": f"ctxr_{gen:032x}",
                "session_id": "hermes-session",
                "generation": gen,
                "original_approx_tokens": 1000,
                "compacted_approx_tokens": 500,
                "token_savings_estimate": 500,
                "original_transcript_blake3": "a" * 64,
                "compacted_transcript_blake3": "b" * 64,
                "original_transcript_sha256": "c" * 64,
                "compacted_transcript_sha256": "d" * 64,
            },
            "allocation_plan": {
                "summary_id": None,
                "items": [],
                "summarized_item_ids": [],
            },
            "compacted_messages": [
                {"role": "user", "content": "keep me"},
                {"role": "assistant", "content": "[Tool result call_a]: first"},
                {"role": "assistant", "content": "[Tool result call_b]: second"},
                {"role": "assistant", "content": "[Tool result call_c]: third"},
                {"role": "user", "content": "start"},
            ],
        }

    def run_json(args, payload):
        nonlocal generation, parent_projection, last_compact_payload
        if args[:3] == ["compact-v2", "--dir", str(engine.store_dir)]:
            incoming = payload["messages"]
            if (
                parent_projection is not None
                and incoming[: len(parent_projection)] != parent_projection
            ):
                raise RuntimeError(
                    "parent compacted transcript is not the exact child-input prefix"
                )
            same_generation_retry = bool(
                last_compact_payload
                and last_compact_payload.get("messages") == payload.get("messages")
                and int(
                    (payload.get("policy") or {}).get(
                        "post_finalize_reserve_tokens", 0
                    )
                )
                > int(
                    (last_compact_payload.get("policy") or {}).get(
                        "post_finalize_reserve_tokens", 0
                    )
                )
            )
            if not same_generation_retry:
                generation += 1
            last_compact_payload = copy.deepcopy(payload)
            return compact_response(generation)
        if args[:3] == ["search", "--dir", str(engine.store_dir)]:
            return []
        if args and args[0] == "finalize-v2":
            response = copy.deepcopy(payload["candidate"])
            response["compacted_messages"] = copy.deepcopy(
                payload["compacted_messages"]
            )
            parent_projection = response["compacted_messages"]
            tokens = sum(
                max(1, len(str(message.get("content", ""))) // 4)
                for message in parent_projection
            )
            response["receipt"]["compacted_approx_tokens"] = tokens
            response["receipt"]["token_savings_estimate"] = 1000 - tokens
            return response
        if args[:3] == ["prepare-v2", "--dir", str(engine.store_dir)]:
            receipt = payload["receipt"]
            return {
                "schema": "PendingReceiptInfoV2",
                "receipt_id": receipt["receipt_id"],
                "session_id": receipt["session_id"],
                "generation": receipt["generation"],
                "created_utc": "2026-08-16T00:00:00Z",
                "pending_path": f"/tmp/{receipt['receipt_id']}.json",
                "expected_compacted_message_count": len(payload["compacted_messages"]),
                "expected_compacted_transcript_blake3": receipt[
                    "compacted_transcript_blake3"
                ],
                "expected_compacted_transcript_sha256": receipt[
                    "compacted_transcript_sha256"
                ],
                "expected_compacted_messages": copy.deepcopy(
                    payload["compacted_messages"]
                ),
                "verified": True,
            }
        if args[:3] == ["activate-v2", "--dir", str(engine.store_dir)]:
            return {
                "schema": "ReceiptActivationResultV2",
                "receipt_id": payload["receipt_id"],
                "activated": True,
                "verified": True,
                "already_activated": False,
            }
        raise AssertionError(f"unexpected command: {args}")

    engine._run_json = run_json
    first = engine.compress([{"role": "user", "content": "start"}], current_tokens=100)
    engine.commit_pending_compression(first)

    # The finalized projection must already be the host-repaired form: the
    # receipt carries no consecutive-assistant run, so the host's own repair
    # after compress() is a no-op and the live transcript matches the parent.
    roles = [m.get("role") for m in parent_projection]
    assert all(
        roles[i] != "assistant" or roles[i + 1] != "assistant"
        for i in range(len(roles) - 1)
    ), f"receipt projection still has consecutive assistants: {roles}"
    assert repair_message_sequence(None, first) == 0

    # Grow the session exactly as the host does (its repair is now a no-op)
    # and compact again: the child must chain onto the stored parent.
    second = engine.compress(
        first + [{"role": "user", "content": "next"}], current_tokens=100
    )
    engine.commit_pending_compression(second)

    assert engine.last_error is None
    assert generation == 2


@pytest.mark.integration
def test_real_binary_background_notification_compacts_across_generations(
    tmp_path, monkeypatch
):
    """The live protocol must bind the host anchor and keep recursive continuity."""
    binary = os.environ.get("CONTEXT_GOVERNOR_BINARY") or shutil.which(
        "context-governor"
    )
    if binary is None:
        pytest.skip("context-governor binary is not installed")

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    binding = ContextGovernorKeyState(home, binary).initialize_first_install()
    binding.close()

    store = home / "context-governor"
    session_id = "background-notification-e2e"
    config = {
        "context": {
            "governor": {
                "summary_mode": "llm",
                "checkpoint_strategy": "after_n:999",
                "token_budget": 64,
                "min_net_savings_tokens": 128,
                "allocator": "deterministic_v1",
                "budget_mode": "hard_cascade",
                "protect_first_n": 0,
                "protect_last_n": 1,
                "max_lineage_generation": 8,
            }
        }
    }
    from agent.conversation_compression import _ensure_compressed_has_user_turn

    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary=binary, store_dir=store)
        engine.on_session_start(session_id)
        engine.update_model("fixture", context_length=1_000, provider="openai-codex")
        first_input = [
            {"role": "user", "content": "Preserve this human intent."},
            {"role": "assistant", "content": "old context " * 800},
            {
                "role": "user",
                "content": (
                    "[IMPORTANT: Background process proc_first completed normally "
                    "(exit code 0).]"
                ),
            },
        ]
        first = engine.compress(first_input, current_tokens=10)
        first_before_host = copy.deepcopy(first)
        _ensure_compressed_has_user_turn(first_input, first)
        assert first == first_before_host
        assert engine.validate_pending_compression(first) is True
        engine.commit_pending_compression(first)

        second_input = first + [
            {"role": "assistant", "content": "new context " * 800},
            {"role": "user", "content": "Please continue the verified task."},
            {
                "role": "user",
                "content": (
                    "[IMPORTANT: Background process proc_second completed normally "
                    "(exit code 0).]"
                ),
            },
        ]
        second = engine.compress(second_input, current_tokens=10)
        second_before_host = copy.deepcopy(second)
        _ensure_compressed_has_user_turn(second_input, second)
        assert second == second_before_host
        assert engine.validate_pending_compression(second) is True
        engine.commit_pending_compression(second)

    assert engine.compression_count == 2
    assert engine.last_error is None
    assert not list((store / ".pending").glob("*.json"))


@pytest.mark.integration
def test_real_binary_continuation_epoch_survives_restart_and_next_generation(
    tmp_path, monkeypatch
):
    binary = os.environ.get("CONTEXT_GOVERNOR_BINARY") or shutil.which(
        "context-governor"
    )
    if binary is None:
        pytest.skip("context-governor binary is not installed")

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    binding = ContextGovernorKeyState(home, binary).initialize_first_install()
    binding.close()

    store = home / "context-governor"
    session_id = "continuation-epoch-e2e"
    config = {
        "context": {
            "governor": {
                "summary_mode": "llm",
                "checkpoint_strategy": "after_n:999",
                "token_budget": 1024,
                "min_net_savings_tokens": 0,
                "allocator": "deterministic_v1",
                "budget_mode": "hard_cascade",
                "protect_first_n": 0,
                "protect_last_n": 1,
                "telemetry_max_additional_protected_messages": 0,
                "max_lineage_generation": 2,
            }
        }
    }
    messages = [
        {"role": "user", "content": "Preserve the root human intent."},
        {"role": "assistant", "content": "initial context " * 300},
        {"role": "user", "content": "continue generation zero"},
    ]
    coordinates = []
    engine = None

    for ordinal in range(4):
        with patch("hermes_cli.config.load_config", return_value=config):
            engine = ContextGovernorEngine(binary=binary, store_dir=store)
            engine.on_session_start(session_id)
            engine.update_model(
                "fixture", context_length=16_000, provider="openai-codex"
            )
            compacted = engine.compress(messages, current_tokens=100)
            assert engine._pending_admission is not None, (
                f"ordinal={ordinal} outcome={engine.last_outcome!r} error={engine.last_error!r}"
            )
            receipt = engine._pending_admission["response"]["receipt"]
            coordinates.append(
                (receipt.get("lineage_epoch", 0), receipt.get("generation"))
            )
            assert engine.validate_pending_compression(compacted) is True
            engine.commit_pending_compression(compacted)
            assert engine.last_error is None
            assert engine._pending_admission is None

        if ordinal < 3:
            messages = compacted + [
                {
                    "role": "assistant",
                    "content": f"new context {ordinal + 1} " * 300,
                },
                {
                    "role": "user",
                    "content": f"continue generation {ordinal + 1}",
                },
            ]

    assert coordinates == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert engine is not None
    assert engine.last_outcome["kind"] == "compacted_pending_host_commit"
    assert engine._run_certified_json(
        [
            "search",
            "--dir",
            str(store),
            "--query",
            "definitely-absent-checkpoint-marker",
            "--scope",
            "summary",
            "--top-k",
            "1",
        ],
        {},
    ) == []
    assert not list((store / ".pending").glob("*.json"))


def test_orphaned_tool_result_is_not_promoted_to_assistant_text():
    """Compaction must repair a broken tool pair without retaining raw output.

    The old adapter converted the orphan to ``[Tool result <id>]: <payload>``
    assistant text, which subsequently became durable and user-visible.
    """
    secret = "SYNTHETIC_TOOL_PAYLOAD_MUST_NOT_BE_VISIBLE"
    engine, _llm = _checkpoint_engine(
        target_tokens=900,
        llm_output=_valid_llm_summary(),
        compacted_prefix=[
            {
                "role": "tool",
                "id": "call_orphan",
                "name": "terminal",
                "content": secret,
                "metadata": {"tool_call_id": "call_orphan"},
            }
        ],
    )

    compacted = engine.compress(
        [
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "final"},
        ],
        current_tokens=100,
    )

    assert all(message.get("role") != "tool" for message in compacted)
    assert all(secret not in str(message.get("content", "")) for message in compacted)
    assert all(
        not str(message.get("content", "")).startswith("[Tool result")
        for message in compacted
    )


def test_tool_pair_repair_drops_a_result_that_precedes_its_later_call():
    """A later call must not retroactively legitimize a raw earlier result."""
    secret = "SYNTHETIC_OUT_OF_ORDER_TOOL_PAYLOAD_MUST_NOT_BE_VISIBLE"
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    messages = [
        {"role": "tool", "tool_call_id": "call_late", "content": secret},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_late", "type": "function"}],
        },
        {"role": "user", "content": "protected latest ask"},
    ]

    repaired = engine._sanitize_tool_pairs(messages)

    assert [message["role"] for message in repaired] == ["assistant", "tool", "user"]
    assert repaired[1]["tool_call_id"] == "call_late"
    assert repaired[1]["content"] == "[Result from earlier conversation — see context summary]"
    assert all(secret not in str(message.get("content", "")) for message in repaired)
    assert repaired[-1]["content"] == "protected latest ask"


def test_tool_pair_repair_preserves_results_keyed_by_provider_alias():
    """Provider-native secondary IDs must match without replacing valid output."""
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    result = {
        "role": "tool",
        "tool_call_id": "response_item_42",
        "content": "valid provider result",
    }
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_42",
                    "call_id": "call_alias_42",
                    "response_item_id": "response_item_42",
                    "type": "function",
                }
            ],
        },
        result,
        {"role": "user", "content": "protected latest ask"},
    ]

    repaired = engine._sanitize_tool_pairs(messages)

    assert repaired[1] is result
    assert repaired[1]["content"] == "valid provider result"
    assert [message["role"] for message in repaired] == ["assistant", "tool", "user"]


@pytest.mark.integration
def test_real_binary_recovers_prepared_receipt_after_host_commit_and_restart(
    tmp_path, monkeypatch
):
    """A crash after host durability but before activation must recover safely."""
    binary = os.environ.get("CONTEXT_GOVERNOR_BINARY") or shutil.which(
        "context-governor"
    )
    if binary is None:
        pytest.skip("context-governor binary is not installed")

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    binding = ContextGovernorKeyState(home, binary).initialize_first_install()
    binding.close()

    store = home / "context-governor"
    session_id = "crash-recovery-e2e"
    session_db = SessionDB(db_path=home / "state.db")
    session_db.create_session(session_id, source="cli")
    config = {
        "context": {
            "governor": {
                "summary_mode": "llm",
                "checkpoint_strategy": "after_n:999",
                "token_budget": 64,
                "min_net_savings_tokens": 128,
                "allocator": "deterministic_v1",
                "budget_mode": "hard_cascade",
                "protect_first_n": 0,
                "protect_last_n": 1,
                "max_lineage_generation": 8,
            }
        }
    }

    with patch("hermes_cli.config.load_config", return_value=config):
        first_process = ContextGovernorEngine(binary=binary, store_dir=store)
        first_process.on_session_start(session_id, session_db=session_db)
        first_process.update_model(
            "fixture", context_length=1_000, provider="openai-codex"
        )
        compacted = first_process.compress(
            [{"role": "user", "content": "continue"}], current_tokens=10
        )
        assert first_process._pending_admission is not None
        receipt_id = first_process._pending_admission["receipt_id"]

        # Simulate the exact crash window: SessionDB committed the authenticated
        # projection, but the process died before commit_pending_compression().
        session_db.archive_and_compact(session_id, compacted)
        assert not (store / f"{receipt_id}.json").exists()

        restarted = ContextGovernorEngine(binary=binary, store_dir=store)
        restarted.on_session_start(session_id, session_db=session_db)

    assert (store / f"{receipt_id}.json").is_file()
    assert not list((store / ".pending").glob("*.json"))
    assert restarted.last_receipt_id == receipt_id


@pytest.mark.integration
def test_real_binary_retries_activation_in_same_process_after_host_commit(
    tmp_path, monkeypatch
):
    """A transient activate-v2 outage must not require restart or rebinding."""
    binary = os.environ.get("CONTEXT_GOVERNOR_BINARY") or shutil.which(
        "context-governor"
    )
    if binary is None:
        pytest.skip("context-governor binary is not installed")

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    binding = ContextGovernorKeyState(home, binary).initialize_first_install()
    binding.close()

    store = home / "context-governor"
    session_id = "same-process-activation-recovery-e2e"
    session_db = SessionDB(db_path=home / "state.db")
    session_db.create_session(session_id, source="cli")
    config = {
        "context": {
            "governor": {
                "summary_mode": "llm",
                "checkpoint_strategy": "after_n:999",
                "token_budget": 64,
                "min_net_savings_tokens": 128,
                "allocator": "deterministic_v1",
                "budget_mode": "hard_cascade",
                "protect_first_n": 0,
                "protect_last_n": 1,
                "max_lineage_generation": 8,
            }
        }
    }

    with patch("hermes_cli.config.load_config", return_value=config):
        engine = ContextGovernorEngine(binary=binary, store_dir=store)
        engine.on_session_start(session_id, session_db=session_db)
        engine.update_model("fixture", context_length=1_000, provider="openai-codex")
        first = engine.compress(
            [{"role": "user", "content": "continue"}], current_tokens=10
        )
        assert engine._pending_admission is not None
        first_receipt_id = engine._pending_admission["receipt_id"]
        session_db.archive_and_compact(session_id, first)

        raw_run_json = engine._run_json
        activation_calls = 0

        def fail_first_activation(args, payload, *, pass_fds=()):
            nonlocal activation_calls
            if args[0] == "activate-v2":
                activation_calls += 1
                if activation_calls == 1:
                    raise TimeoutError("synthetic post-commit activation outage")
            return raw_run_json(args, payload, pass_fds=pass_fds)

        engine._run_json = fail_first_activation
        with pytest.raises(TimeoutError, match="post-commit activation outage"):
            engine.commit_pending_compression(first)
        assert engine._pending_admission is not None

        second_input = first + [
            {"role": "assistant", "content": "continued work"},
            {"role": "user", "content": "continue without restarting"},
        ]
        second = engine.compress(second_input, current_tokens=10)
        assert engine._pending_admission is not None
        second_receipt_id = engine._pending_admission["receipt_id"]
        assert second_receipt_id != first_receipt_id
        session_db.archive_and_compact(session_id, second)
        engine.commit_pending_compression(second)

    assert activation_calls == 3
    assert (store / f"{first_receipt_id}.json").is_file()
    assert (store / f"{second_receipt_id}.json").is_file()
    assert not list((store / ".pending").glob("*.json"))
    assert engine.compression_count == 2
    assert engine.last_receipt_id == second_receipt_id


def test_governor_projection_roundtrips_through_session_store(tmp_path):
    """Receipt projection fields must survive Hermes' durable in-place rewrite."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "governor-roundtrip"
    db.create_session(session_id, source="cli")
    engine = ContextGovernorEngine(binary="/tmp/context-governor")
    governor_messages = [
        {
            "role": "tool",
            "id": "call_123",
            "name": "skill_view",
            "content": "tool result",
            "metadata": {"tool_call_id": "call_123"},
        },
        {
            "role": "assistant",
            "id": "summary_random_id",
            "name": "context_governor",
            "content": "deterministic extractive summary",
        },
    ]
    host_messages = [
        engine._message_from_governor(message) for message in governor_messages
    ]

    db.archive_and_compact(session_id, host_messages)
    reloaded = db.get_messages_as_conversation(session_id)
    roundtripped = [
        engine._message_to_governor(message, index)
        for index, message in enumerate(reloaded)
    ]

    assert roundtripped == [
        governor_messages[0],
        {
            "role": "assistant",
            "name": "context_governor",
            "content": "deterministic extractive summary",
        },
    ]


def test_legacy_receipt_prefix_rehydrates_only_lost_durable_fields(tmp_path):
    """A pre-fix archive can resume with names absent but no other drift."""
    session_id = "legacy-prefix"
    engine = ContextGovernorEngine(binary="/tmp/context-governor", store_dir=tmp_path)
    engine.session_id = session_id
    receipt_prefix = [
        {
            "role": "tool",
            "id": "call_123",
            "name": "skill_view",
            "content": "tool result",
            "metadata": {
                "hermes_metadata": {"interim": True},
                "tool_call_id": "call_123",
            },
        },
        {
            "role": "assistant",
            "id": "summary_123",
            "name": "context_governor",
            "content": "deterministic extractive summary",
            "metadata": {
                "hermes_metadata": {
                    "llm_checkpoint": True,
                    "receipt_id": "ctxr_legacy",
                }
            },
        },
    ]
    (tmp_path / "ctxr_legacy.json").write_text(
        json.dumps({
            "receipt": {
                "session_id": session_id,
                "generation": 1,
                "created_utc": "2026-08-16T00:00:00Z",
            },
            "compacted_messages": receipt_prefix,
        }),
        encoding="utf-8",
    )

    # Simulate the historical SessionDB projection: tool-call id survives,
    # while provider-facing names, assistant summary ids, and generic host
    # metadata do not.
    resumed = [
        {
            "role": "tool",
            "content": "tool result",
            "tool_call_id": "call_123",
        },
        {"role": "assistant", "content": "deterministic extractive summary"},
        {"role": "user", "content": "new work"},
    ]
    incoming = [
        engine._message_to_governor(message, index)
        for index, message in enumerate(resumed)
    ]

    assert engine._rehydrate_legacy_parent_prefix(incoming) == receipt_prefix + [
        {"role": "user", "content": "new work"}
    ]

    # Same legacy field loss is not sufficient if content has drifted.
    incoming[0]["content"] = "different result"
    assert engine._rehydrate_legacy_parent_prefix(incoming) == incoming

    # Governor-owned durable metadata remains comparison-significant even
    # beside the ignored host envelope.
    incoming[0]["content"] = "tool result"
    incoming[0]["metadata"]["tool_call_id"] = "call_drifted"
    assert engine._rehydrate_legacy_parent_prefix(incoming) == incoming


def test_legacy_rehydrate_uses_catalog_session_rows(tmp_path, monkeypatch):
    """Large stores must not rescan unrelated receipt payloads on resume."""
    session_id = "catalog-bounded"
    engine = ContextGovernorEngine(binary="/tmp/context-governor", store_dir=tmp_path)
    engine.session_id = session_id
    receipt_prefix = [
        {
            "role": "assistant",
            "id": "summary_123",
            "name": "context_governor",
            "content": "deterministic extractive summary",
        },
        {"role": "user", "content": "active task"},
    ]
    target = tmp_path / "ctxr_target.json"
    target.write_text(
        json.dumps(
            {
                "receipt": {
                    "session_id": session_id,
                    "generation": 4,
                    "created_utc": "2026-08-16T00:00:00Z",
                },
                "compacted_messages": receipt_prefix,
            }
        ),
        encoding="utf-8",
    )
    for index in range(128):
        (tmp_path / f"ctxr_noise_{index}.json").write_text("not-read", encoding="utf-8")

    with sqlite3.connect(tmp_path / ".receipt-index.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE receipts ("
            "receipt_id TEXT, generation INTEGER, created_utc TEXT, session_id TEXT)"
        )
        connection.execute(
            "INSERT INTO receipts VALUES (?, ?, ?, ?)",
            ("ctxr_target", 4, "2026-08-16T00:00:00Z", session_id),
        )
        connection.commit()

    original_read_text = type(target).read_text
    read_names = []

    def tracked_read_text(path, *args, **kwargs):
        read_names.append(path.name)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(target), "read_text", tracked_read_text)
    incoming = [
        engine._message_to_governor(
            {"role": "assistant", "content": "deterministic extractive summary"}, 0
        ),
        engine._message_to_governor({"role": "user", "content": "active task"}, 1),
        engine._message_to_governor({"role": "user", "content": "new work"}, 2),
    ]

    assert engine._rehydrate_legacy_parent_prefix(incoming) == receipt_prefix + [
        {"role": "user", "content": "new work"}
    ]
    assert read_names == ["ctxr_target.json"]


def test_resume_alternation_repair_rehydrates_authenticated_parent(tmp_path):
    """Provider replay repair must not replace receipt-owned lineage identity."""
    session_id = "repaired-prefix"
    engine = ContextGovernorEngine(binary="/tmp/context-governor", store_dir=tmp_path)
    engine.session_id = session_id
    receipt_prefix = [
        {"role": "assistant", "content": "preserved answer"},
        {
            "role": "assistant",
            "id": "summary_123",
            "name": "context_governor",
            "content": "deterministic extractive summary",
        },
        {"role": "assistant", "content": "preserved decision"},
        {"role": "user", "content": "active task"},
    ]
    (tmp_path / "ctxr_repaired.json").write_text(
        json.dumps({
            "receipt": {
                "session_id": session_id,
                "generation": 1,
                "created_utc": "2026-08-16T00:00:00Z",
            },
            "compacted_messages": receipt_prefix,
        }),
        encoding="utf-8",
    )
    resumed = [engine._message_from_governor(message) for message in receipt_prefix]
    assert repair_message_sequence(None, resumed) == 2
    resumed.append({"role": "assistant", "content": "new result"})
    incoming = [
        engine._message_to_governor(message, index)
        for index, message in enumerate(resumed)
    ]

    assert engine._rehydrate_legacy_parent_prefix(incoming) == receipt_prefix + [
        {"role": "assistant", "content": "new result"}
    ]

    # A provider-repaired shape with any changed byte is not a receipt match.
    incoming[0]["content"] += " drift"
    assert engine._rehydrate_legacy_parent_prefix(incoming) == incoming
