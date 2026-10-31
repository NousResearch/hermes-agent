"""Effect-time authority and malformed-input regressions for Phase 2 storage."""

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from agent.phase2_authority import (
    AuthorityError,
    Phase2AuthorityStore,
    ResultRejection,
    current_sealed_envelope,
    validate_sealed_envelope,
)
from agent.phase2_idempotency import MutationClaimStore
from tests.agent.test_phase2_authority import _node, _plan
from tests.agent.test_phase2_idempotency import _envelope


@pytest.fixture
def live_authority(tmp_path):
    now = datetime(2026, 7, 30, tzinfo=UTC)
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    node = _node("n-tool")
    node["network_policy"] = "allowlist"
    node["network_allowlist"] = ["example.com"]
    node["permissions"]["spawn"] = ["worker"]
    store.seal_plan(_plan(node), policy_hash="d" * 64, now=now)
    envelope = store.grant_node(
        "g-authority", "n-tool", holder="hermes:test", ttl_s=30,
        attempt_id="at-1", now=now,
    )
    return store, envelope, now


@pytest.mark.parametrize("state", [
    "cancelled_before_effect", "failed_proven_no_effect", "indeterminate",
])
def test_settlement_ends_effect_authority_durably(live_authority, state):
    store, envelope, now = live_authority
    with store.bind_current(envelope, now=now):
        bound = current_sealed_envelope()
        store.settle_attempt(envelope, state=state, now=now)
        reopened = Phase2AuthorityStore(store.db_path)
        assert "attempt_settled" in reopened.validate_current(bound, now=now)
        for reserve in (
            lambda: reopened.reserve_budget(bound, tokens=1, usd=0.01, now=now),
            lambda: reopened.reserve_budget_allocations(
                bound, [{"tokens": 1, "usd": 0.01}], now=now,
            ),
        ):
            with pytest.raises(AuthorityError, match="settled"):
                reserve()
        with pytest.raises(AuthorityError, match="settled"):
            reopened.renew_node(bound, ttl_s=60, now=now)
        with pytest.raises(AuthorityError, match="attempt_settled"):
            with reopened.bind_current(bound, now=now):
                pytest.fail("settled attempt bound")
        with pytest.raises(ResultRejection) as rejection:
            reopened.complete_node(bound, result_hash="e" * 64, now=now)
        assert rejection.value.reason == "attempt_settled"
        assert reopened.settle_attempt(bound, state=state, now=now)["idempotent"]
    events = store.read_events("g-authority")
    assert [event["kind"] for event in events] == [
        "PLAN_SEALED", "LEASE_GRANTED", "ATTEMPT_SETTLED", "RESULT_REJECTED",
    ]


def test_bound_envelope_survives_effect_time_validation(live_authority):
    store, envelope, now = live_authority
    with store.bind_current(envelope, now=now):
        bound = current_sealed_envelope()
        assert isinstance(bound["roots"], tuple)
        assert validate_sealed_envelope(bound) == []
        assert store.validate_current(bound, now=now) == []
        store.reserve_budget(bound, tokens=1, usd=0.01, now=now)
        store.complete_node(bound, result_hash="e" * 64, now=now)
    assert current_sealed_envelope() is None


@pytest.mark.parametrize("field", [
    "execution_surface", "lane", "network_policy", "exec_policy", "destructive_policy",
])
@pytest.mark.parametrize("value", [[], {}])
def test_malformed_enum_fields_return_errors_and_typed_seal_rejection(
    live_authority, field, value,
):
    store, envelope, now = live_authority
    malformed = deepcopy(envelope)
    malformed[field] = value
    malformed["retry_policy"] = {
        "max_attempts": 2, "retry_eligible": True, "backoff_s": [0],
    }
    assert field in validate_sealed_envelope(malformed)
    assert field in store.validate_current(malformed, now=now)
    node = _node("malformed")
    node[field] = value
    with pytest.raises(AuthorityError, match=field):
        store.seal_plan(_plan(node), policy_hash="d" * 64, now=now)


def test_tilde_authority_path_preserves_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = Phase2AuthorityStore("~/authority.db")
    store.seal_plan(_plan(), policy_hash="d" * 64)
    assert store.db_path == tmp_path / "authority.db"
    assert store.read_events("g-authority")[0]["kind"] == "PLAN_SEALED"
    assert store.migration_report()["exists"] is True


def test_tilde_mutation_path_preserves_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = MutationClaimStore("~/mutation.db")
    store.try_claim(_envelope(), tool_name="write_file", args={"path": "one"})
    assert store.db_path == tmp_path / "mutation.db"
    assert len(store.read_attempts("d" * 64)) == 1
    assert len(store.read_all()) == 1


def test_settled_attempt_requires_readmission_before_new_live_grant(live_authority):
    store, envelope, now = live_authority
    store.settle_attempt(envelope, state="cancelled_before_effect", now=now)
    with pytest.raises(AuthorityError, match="explicit readmission"):
        store.grant_node(
            "g-authority", "n-tool", holder="hermes:test", ttl_s=30,
            attempt_id="at-2", now=now,
        )
    store.record_readmission(
        "g-authority", "n-tool", prior_attempt_id="at-1",
        readmission_id="operator-1", reason="verified non-effect", now=now,
    )
    next_envelope = store.grant_node(
        "g-authority", "n-tool", holder="hermes:test", ttl_s=30,
        attempt_id="at-2", readmission_id="operator-1", now=now,
    )
    assert next_envelope["fence"] == 2
    assert store.validate_current(next_envelope, now=now) == []
    assert "stale_fence" in store.validate_current(envelope, now=now)


def test_recovery_reconciles_settled_attempt_reservation(live_authority):
    store, envelope, now = live_authority
    reservation = store.reserve_budget(envelope, tokens=1, usd=0.01, now=now)
    store.settle_attempt(envelope, state="indeterminate", now=now)
    store.recover(now=now)
    store.recover(now=now)
    reconciled = [
        event for event in store.read_events("g-authority")
        if event["kind"] == "BUDGET_RECONCILED"
    ]
    assert len(reconciled) == 1
    assert reconciled[0]["payload"]["reservation_id"] == reservation
    assert reconciled[0]["payload"]["actual_usd"] is None
