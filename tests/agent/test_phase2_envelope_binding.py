"""Node-scoped binding contracts for sealed Phase 2 task envelopes.

Tests ``validate_sealed_envelope``, ``bind_sealed_envelope``,
``current_sealed_envelope``, and ``current_authoritative_fence`` as exported
from phase2_authority. No dependency on phase2_enforcement or any runtime
session wiring.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.phase2_authority import (
    AuthorityError,
    bind_sealed_envelope,
    current_authoritative_fence,
    current_sealed_envelope,
    validate_sealed_envelope,
)


def _envelope(
    *,
    node_id: str = "n-bound-node",
    surface: str = "local_tool",
    fence: int = 1,
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "envelope_version": 2,
        "graph_id": "g-bound-graph",
        "node_id": node_id,
        "attempt_id": f"at-{node_id}",
        "idempotency_key": "d" * 64,
        "objective": "bind one sealed graph node to one execution surface",
        "execution_surface": surface,
        "lane": "hermes",
        "roots": ["/tmp"],
        "permissions": {"read": ["/tmp"], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60},
        "deadline_utc": (now + timedelta(minutes=5)).isoformat(),
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["tool_exec"], "min_receipts": 1},
        "verifier_policy": {
            "required": False,
            "verifier_lane_must_differ": False,
            "clean_workspace_required": False,
        },
        "planner_hash": "a" * 64,
        "policy_hash": "b" * 64,
        "input_hash": "c" * 64,
        "lease": {
            "holder": "hermes",
            "granted_utc": now.isoformat(),
            "ttl_s": 300,
            "renewable": True,
        },
        "fence": fence,
    }


# ── validate_sealed_envelope ──────────────────────────────────────────────────


def test_valid_envelope_passes_validation():
    assert validate_sealed_envelope(_envelope()) == []


def test_missing_required_field_is_reported():
    bad = _envelope()
    del bad["graph_id"]
    errors = validate_sealed_envelope(bad)
    assert "graph_id" in errors


def test_wrong_envelope_version_is_reported():
    bad = _envelope()
    bad["envelope_version"] = 1
    assert "envelope_version" in validate_sealed_envelope(bad)


def test_invalid_execution_surface_is_reported():
    bad = _envelope()
    bad["execution_surface"] = "unknown_surface"
    assert "execution_surface" in validate_sealed_envelope(bad)


def test_non_finite_budget_is_reported():
    bad = _envelope()
    bad["budgets"] = {
        "usd_max": float("inf"),
        "tokens_max": 1000,
        "wall_clock_s_max": 60,
    }
    assert "budgets.usd_max" in validate_sealed_envelope(bad)


def test_sha256_fields_must_be_lowercase_64_hex_chars():
    bad = _envelope()
    bad["idempotency_key"] = "UPPERCASE" + "a" * 55
    assert "idempotency_key" in validate_sealed_envelope(bad)
    bad2 = _envelope()
    bad2["planner_hash"] = "short"
    assert "planner_hash" in validate_sealed_envelope(bad2)


def test_fence_must_be_positive_integer():
    bad = _envelope()
    bad["fence"] = 0
    assert "fence" in validate_sealed_envelope(bad)
    bad2 = _envelope()
    bad2["fence"] = True  # bool subclasses int but is not a valid fence
    assert "fence" in validate_sealed_envelope(bad2)
    bad3 = _envelope()
    bad3["fence"] = "1"
    assert "fence" in validate_sealed_envelope(bad3)
    bad4 = _envelope()
    bad4["fence"] = 2**63
    assert "fence" in validate_sealed_envelope(bad4)


def test_malformed_retry_attempt_count_is_reported_without_raising():
    bad = _envelope()
    bad["retry_policy"]["max_attempts"] = "two"

    assert "retry_policy.max_attempts" in validate_sealed_envelope(bad)


def test_model_and_tool_envelopes_are_independently_valid():
    model = _envelope(node_id="n-model", surface="direct_model")
    tool = _envelope(node_id="n-tool", surface="local_tool")
    assert validate_sealed_envelope(model) == []
    assert validate_sealed_envelope(tool) == []
    assert model["node_id"] != tool["node_id"]
    assert model["execution_surface"] == "direct_model"
    assert tool["execution_surface"] == "local_tool"


# ── bind_sealed_envelope ──────────────────────────────────────────────────────


def test_binding_is_node_scoped_and_resets_after_success():
    with bind_sealed_envelope(_envelope(), current_fence=1):
        current = current_sealed_envelope()
        assert current is not None
        assert current["graph_id"] == "g-bound-graph"
        assert current["node_id"] == "n-bound-node"
        assert current["execution_surface"] == "local_tool"
        assert current_authoritative_fence() == 1

    assert current_sealed_envelope() is None
    assert current_authoritative_fence() is None


def test_binding_resets_after_exception():
    try:
        with bind_sealed_envelope(_envelope(), current_fence=1):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert current_sealed_envelope() is None
    assert current_authoritative_fence() is None


def test_bound_envelope_is_immutable_proxy():
    env = _envelope()
    with bind_sealed_envelope(env, current_fence=1):
        bound = current_sealed_envelope()
        with pytest.raises((TypeError, AttributeError)):
            bound["fence"] = 999  # type: ignore[index]


def test_bound_envelope_is_recursively_immutable_and_detached():
    env = _envelope()
    with bind_sealed_envelope(env, current_fence=1):
        bound = current_sealed_envelope()
        assert bound is not None
        env["permissions"]["read"].append("/outside")
        env["budgets"]["tokens_max"] = 0
        assert bound["permissions"]["read"] == ("/tmp",)
        assert bound["budgets"]["tokens_max"] == 1000
        with pytest.raises(TypeError):
            bound["permissions"]["read"] += ("/outside",)  # type: ignore[index]
        with pytest.raises(TypeError):
            bound["budgets"]["tokens_max"] = 0  # type: ignore[index]
        with pytest.raises(TypeError):
            bound["retry_policy"]["backoff_s"] += (1,)  # type: ignore[index]
        with pytest.raises(AttributeError):
            bound["roots"].append("/outside")  # type: ignore[union-attr]


def test_binding_rejects_non_json_mutable_values():
    env = _envelope()
    env["permissions"]["read"] = {"/tmp"}

    with pytest.raises(AuthorityError, match="unsupported value type"):
        with bind_sealed_envelope(env, current_fence=1):
            pass


def test_binding_without_fence_publishes_none():
    with bind_sealed_envelope(_envelope()):
        assert current_authoritative_fence() is None
        assert current_sealed_envelope() is not None


def test_nested_binding_is_scoped_to_innermost_context():
    outer = _envelope(node_id="n-outer", fence=1)
    inner = _envelope(node_id="n-inner", fence=2)
    with bind_sealed_envelope(outer, current_fence=1):
        assert current_sealed_envelope()["node_id"] == "n-outer"
        assert current_authoritative_fence() == 1
        with bind_sealed_envelope(inner, current_fence=2):
            assert current_sealed_envelope()["node_id"] == "n-inner"
            assert current_authoritative_fence() == 2
        assert current_sealed_envelope()["node_id"] == "n-outer"
        assert current_authoritative_fence() == 1
    assert current_sealed_envelope() is None
