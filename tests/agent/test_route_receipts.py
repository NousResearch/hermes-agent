from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


NOW = datetime(2026, 9, 17, 19, 0, tzinfo=timezone.utc)


def _prepare(store, **overrides):
    values = {
        "stable_call_id": "call-gemini-1",
        "parent_session_id": "parent-1",
        "parent_turn_id": "turn-1",
        "child_session_id": "child-1",
        "task_index": 0,
        "run_kind": "production",
        "route_requested": "auto",
        "route_decision": "gemini",
        "route_reason": "eligible",
        "data_classification": "standard",
        "output_contract": "text",
        "provider": "antigravity-subscription",
        "model": "gemini-fixture",
        "started_at": NOW,
    }
    values.update(overrides)
    return store.prepare_attempt(**values)


def test_route_receipts_require_typed_run_population_and_closed_route(tmp_path: Path):
    from agent.route_receipts import RouteReceiptStore

    store = RouteReceiptStore(tmp_path / "routing.sqlite3")

    with pytest.raises(ValueError, match="run_kind"):
        _prepare(store, run_kind="unknown")
    with pytest.raises(ValueError, match="route_decision"):
        _prepare(store, route_decision="sol")

    stable = _prepare(store)
    assert stable == "call-gemini-1"
    assert store.get_attempt(stable)["run_kind"] == "production"


@pytest.mark.parametrize("run_kind", ["production", "canary", "synthetic", "evaluation"])
@pytest.mark.parametrize("route_decision", ["gemini", "frontier"])
def test_route_receipts_support_every_typed_population_and_both_routes(
    tmp_path: Path, run_kind: str, route_decision: str
):
    from agent.route_receipts import RouteReceiptStore

    store = RouteReceiptStore(tmp_path / f"{run_kind}-{route_decision}.sqlite3")
    stable = _prepare(
        store,
        stable_call_id=f"call-{run_kind}-{route_decision}",
        run_kind=run_kind,
        route_decision=route_decision,
    )
    row = store.get_attempt(stable)
    assert row["run_kind"] == run_kind
    assert row["route_decision"] == route_decision
    assert row["status"] == "prepared"


def test_route_receipt_completion_binds_usage_terminal_route_and_fallback(tmp_path: Path):
    from agent.route_receipts import RouteReceiptStore

    store = RouteReceiptStore(tmp_path / "routing.sqlite3")
    gemini = _prepare(store)
    frontier = _prepare(
        store,
        stable_call_id="call-frontier-1",
        route_decision="frontier",
        provider="openai-codex",
        model="frontier-fixture",
        fallback_from_call_id=gemini,
    )
    store.complete_attempt(
        gemini,
        status="failed",
        terminal_route="gemini",
        terminal_provider="antigravity-subscription",
        terminal_model="gemini-fixture",
        usage=None,
        usage_status="unavailable",
        fallback_used=True,
        fallback_call_id=frontier,
        error_code="worker_failed",
        completed_at=NOW,
    )
    store.complete_attempt(
        frontier,
        status="completed",
        terminal_route="frontier",
        terminal_provider="openai-codex",
        terminal_model="frontier-fixture",
        usage={
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_tokens": 3,
            "cache_write_tokens": 4,
            "reasoning_tokens": 5,
            "total_tokens": 24,
        },
        usage_status="complete",
        fallback_used=False,
        completed_at=NOW,
    )

    gemini_row = store.get_attempt(gemini)
    frontier_row = store.get_attempt(frontier)
    assert gemini_row["fallback_used"] == 1
    assert gemini_row["fallback_call_id"] == frontier
    assert frontier_row["fallback_from_call_id"] == gemini
    assert frontier_row["terminal_route"] == "frontier"
    assert frontier_row["total_tokens"] == 24


def test_usage_must_be_complete_and_self_consistent_or_unavailable(tmp_path: Path):
    from agent.route_receipts import RouteReceiptStore

    store = RouteReceiptStore(tmp_path / "routing.sqlite3")
    stable = _prepare(store)
    with pytest.raises(ValueError, match="usage"):
        store.complete_attempt(
            stable,
            status="completed",
            terminal_route="gemini",
            terminal_provider="provider",
            terminal_model="model",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 999},
            usage_status="complete",
            completed_at=NOW,
        )


def test_parent_disposition_receipt_is_joinable_closed_and_append_only(tmp_path: Path):
    from agent.route_receipts import RouteReceiptStore

    store = RouteReceiptStore(tmp_path / "routing.sqlite3")
    stable = _prepare(store)
    disposition_id = store.record_parent_disposition(
        stable_call_id=stable,
        parent_session_id="parent-1",
        parent_turn_id="turn-2",
        disposition="materially_revised",
        proof_kind="parent_explicit",
        recorded_at=NOW,
        disposition_receipt_id="disp-1",
    )
    row = store.get_parent_disposition(disposition_id)
    assert row == {
        "disposition_receipt_id": "disp-1",
        "stable_call_id": stable,
        "parent_session_id": "parent-1",
        "parent_turn_id": "turn-2",
        "disposition": "materially_revised",
        "proof_kind": "parent_explicit",
        "recorded_at_utc": NOW.isoformat(),
    }
    with pytest.raises(ValueError, match="disposition"):
        store.record_parent_disposition(
            stable_call_id=stable,
            parent_session_id="parent-1",
            parent_turn_id="turn-3",
            disposition="edited_a_bit",
            proof_kind="parent_explicit",
        )
    with pytest.raises(ValueError, match="parent session"):
        store.record_parent_disposition(
            stable_call_id=stable,
            parent_session_id="different-parent",
            parent_turn_id="turn-2",
            disposition="accepted_as_is",
            proof_kind="parent_explicit",
        )
    with pytest.raises(Exception):
        store.record_parent_disposition(
            stable_call_id=stable,
            parent_session_id="parent-1",
            parent_turn_id="turn-2",
            disposition="materially_revised",
            proof_kind="parent_explicit",
            disposition_receipt_id="disp-1",
        )


class FakeFrontierChild:
    def __init__(self):
        self.session_id = "child-frontier"
        self.provider = "openai-codex"
        self.model = "frontier-fixture"
        self.session_prompt_tokens = 10
        self.session_completion_tokens = 2
        self.session_cache_read_tokens = 3
        self.session_cache_write_tokens = 4
        self.session_reasoning_tokens = 5
        self.session_total_tokens = 24
        self.calls = 0
        self.closed = False

    def run_conversation(self, **_kwargs):
        self.calls += 1
        return {
            "completed": True,
            "final_response": "fake frontier answer",
            "messages": [],
            "api_calls": 1,
        }

    def close(self):
        self.closed = True


def test_frontier_child_wrapper_records_fake_call_without_transport(tmp_path: Path):
    from agent.route_receipts import RouteReceiptStore, RouteReceiptChild

    store = RouteReceiptStore(tmp_path / "routing.sqlite3")
    fake = FakeFrontierChild()
    child = RouteReceiptChild(
        child=fake,
        store=store,
        parent_session_id="parent-1",
        parent_turn_id="turn-1",
        task_index=0,
        run_kind="synthetic",
        route_requested="sol",
        route_reason="explicit",
        data_classification="standard",
        output_contract="text",
    )

    result = child.run_conversation(user_message="fixture", task_id="fake-task")

    assert result["final_response"] == "fake frontier answer"
    assert fake.calls == 1
    row = store.get_attempt(child.route_receipt_id)
    assert row["route_decision"] == "frontier"
    assert row["run_kind"] == "synthetic"
    assert row["status"] == "completed"
    assert row["usage_status"] == "complete"
    assert row["total_tokens"] == 24
    assert result["route_receipt_id"] == child.route_receipt_id
    child.close()
    assert fake.closed


def test_frontier_child_wrapper_keeps_release_callback_local(tmp_path: Path):
    from agent.route_receipts import RouteReceiptStore, RouteReceiptChild

    fake = FakeFrontierChild()
    child = RouteReceiptChild(
        child=fake,
        store=RouteReceiptStore(tmp_path / "routing.sqlite3"),
        parent_session_id="parent-1",
        parent_turn_id="turn-1",
        task_index=0,
        run_kind="synthetic",
        route_requested="sol",
        route_reason="explicit",
        data_classification="standard",
        output_contract="text",
    )
    release = lambda: True

    child._delegate_release_ownership = release

    assert vars(child)["_delegate_release_ownership"] is release
    assert "_delegate_release_ownership" not in vars(fake)
