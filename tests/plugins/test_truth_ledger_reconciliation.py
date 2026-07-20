import copy
import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def _load_reconciliation_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "plugins" / "truth-ledger" / "reconciliation.py"
    spec = importlib.util.spec_from_file_location("truth_ledger_reconciliation_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _obs(**overrides):
    payload = {
        "profile": "default",
        "platform": "cli",
        "session_id": "s1",
        "turn_id": "t1",
        "task_id": "task-a",
        "speaker_id": "user-1",
        "conversation_id": "conv-1",
        "thread_id": "thread-1",
    }
    payload.update(overrides)
    return payload


def _candidate(value, **overrides):
    payload = {
        "scope": "user",
        "kind": "preference",
        "subject": "platform-user:cli:user-1",
        "key": "response.style",
        "value": value,
        "proposed_operation": "assert",
        "source_text": "use concise responses",
    }
    payload.update(overrides)
    return payload


def test_identical_claim_twice_is_assert_then_confirm_with_different_event_ids():
    mod = _load_reconciliation_module()

    first = mod.reconcile_candidate(
        history=[],
        observation=_obs(turn_id="t1"),
        candidate=_candidate("concise"),
        occurred_at="2026-07-17T22:00:00Z",
    )
    assert first["decision"] == "append"
    assert first["event"]["operation"] == "assert"

    second = mod.reconcile_candidate(
        history=[first["event"]],
        observation=_obs(turn_id="t2"),
        candidate=_candidate("concise", source_text="please stay concise"),
        occurred_at="2026-07-17T22:05:00Z",
    )
    assert second["decision"] == "append"
    assert second["event"]["operation"] == "confirm"
    assert second["event"]["event_id"] != first["event"]["event_id"]
    assert second["event"]["fact_id"] == first["event"]["fact_id"]


def test_duplicate_callback_for_same_turn_is_idempotent():
    mod = _load_reconciliation_module()

    result1 = mod.reconcile_candidate(
        history=[],
        observation=_obs(turn_id="t1"),
        candidate=_candidate("concise"),
        occurred_at="2026-07-17T22:00:00Z",
    )

    result2 = mod.reconcile_candidate(
        history=[result1["event"]],
        observation=_obs(turn_id="t1"),
        candidate=_candidate("concise", source_text="same words, retry callback"),
        occurred_at="2026-07-17T22:00:00Z",
    )

    assert result1["decision"] == "append"
    assert result2["decision"] == "duplicate"
    assert result2["event"]["event_id"] == result1["event"]["event_id"]


def test_correction_supersedes_previous_fact_and_links_edge():
    mod = _load_reconciliation_module()

    first = mod.reconcile_candidate(
        history=[],
        observation=_obs(turn_id="t1"),
        candidate=_candidate("concise"),
        occurred_at="2026-07-17T22:00:00Z",
    )

    correction = mod.reconcile_candidate(
        history=[first["event"]],
        observation=_obs(turn_id="t2"),
        candidate=_candidate("detailed", source_text="actually use detailed responses for engineering"),
        occurred_at="2026-07-17T22:10:00Z",
    )

    assert correction["decision"] == "append"
    assert correction["event"]["operation"] == "supersede"
    assert correction["event"]["supersedes"] == first["event"]["fact_id"]
    assert correction["event"]["fact_id"] != first["event"]["fact_id"]


def test_retract_creates_tombstone_and_allows_new_assert_later():
    mod = _load_reconciliation_module()

    asserted = mod.reconcile_candidate(
        history=[],
        observation=_obs(turn_id="t1"),
        candidate=_candidate("concise"),
        occurred_at="2026-07-17T22:00:00Z",
    )
    retracted = mod.reconcile_candidate(
        history=[asserted["event"]],
        observation=_obs(turn_id="t2"),
        candidate=_candidate(None, proposed_operation="retract", source_text="do not remember this preference"),
        occurred_at="2026-07-17T22:20:00Z",
    )
    reasserted = mod.reconcile_candidate(
        history=[asserted["event"], retracted["event"]],
        observation=_obs(turn_id="t3"),
        candidate=_candidate("concise", source_text="you can remember concise again"),
        occurred_at="2026-07-17T22:40:00Z",
    )

    assert retracted["decision"] == "append"
    assert retracted["event"]["operation"] == "retract"
    assert retracted["event"]["supersedes"] == asserted["event"]["fact_id"]
    assert reasserted["decision"] == "append"
    assert reasserted["event"]["operation"] == "assert"


def test_out_of_order_retract_before_any_assert_is_none():
    mod = _load_reconciliation_module()

    result = mod.reconcile_candidate(
        history=[],
        observation=_obs(turn_id="t1"),
        candidate=_candidate(None, proposed_operation="retract"),
        occurred_at="2026-07-17T22:00:00Z",
    )

    assert result["decision"] == "none"
    assert result["event"] is None
    assert result["reason"] == "out_of_order_retract"


def test_none_operation_never_emits_event():
    mod = _load_reconciliation_module()

    result = mod.reconcile_candidate(
        history=[],
        observation=_obs(turn_id="t1"),
        candidate=_candidate("ignored", proposed_operation="NONE", source_text="running tests now"),
        occurred_at="2026-07-17T22:00:00Z",
    )

    assert result["decision"] == "none"
    assert result["event"] is None
    assert result["reason"] == "none_operation"


def test_history_is_not_mutated_by_reconciliation():
    mod = _load_reconciliation_module()

    first = mod.reconcile_candidate(
        history=[],
        observation=_obs(turn_id="t1"),
        candidate=_candidate("concise"),
        occurred_at="2026-07-17T22:00:00Z",
    )

    history = [first["event"]]
    original = copy.deepcopy(history)

    _ = mod.reconcile_candidate(
        history=history,
        observation=_obs(turn_id="t2"),
        candidate=_candidate("detailed"),
        occurred_at="2026-07-17T22:10:00Z",
    )

    assert history == original


def test_runtime_reconciliation_event_conforms_to_frozen_ledger_schema():
    mod = _load_reconciliation_module()
    result = mod.reconcile_candidate(
        history=[],
        observation=_obs(turn_id="t-schema"),
        candidate=_candidate("concise"),
        occurred_at="2026-07-17T22:00:00Z",
        extraction={
            "schema_name": "truth-ledger.fact-candidates.v1",
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "prompt_version": 2,
        },
    )
    event = result["event"]
    schema_path = Path(__file__).resolve().parents[2] / "plugins" / "truth-ledger" / "schemas" / "ledger-event-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(event), key=lambda err: list(err.path))
    assert errors == []
    assert set(event) == {
        "schema_version", "event_id", "occurred_at", "operation", "fact_id",
        "supersedes", "fact", "evidence", "extraction",
    }
