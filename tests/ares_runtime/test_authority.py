"""DR Task 3.2 — deterministic attenuation and effect settlement tests.

Covers the exact RED fixtures named in the DR finish plan: one-field
widening, sibling max-use duplication, stale generation, crash-after-effect
(indeterminate), serializer field-loss, out-of-parent issuance, and valid
controls remaining executable.
"""

from __future__ import annotations

import pytest

from ares_runtime.authority import (
    AuthorityScopeV1,
    ContractError,
    is_subset_scope,
    normalize_scope,
    scope_fingerprint,
)


def make_parent():
    return AuthorityScopeV1(
        scope={
            "tool": "write_file",
            "target": "path:/tmp/x",
            "use_count": 3,
            "time": {"not_before": "2026-01-01T00:00:00Z", "not_after": "2026-12-31T23:59:59Z"},
        },
        generation=5,
        holder="holder:root",
    )


# --- normalize ---------------------------------------------------------------


def test_normalize_rejects_unknown_and_empty() -> None:
    with pytest.raises(ContractError) as e:
        normalize_scope({"tool": "x", "extra": 1})
    assert e.value.code == "UNKNOWN_FIELD"
    with pytest.raises(ContractError) as e:
        normalize_scope({})
    assert e.value.code == "EMPTY_SCOPE"
    with pytest.raises(ContractError):
        normalize_scope("nope")


def test_normalize_time_bounds() -> None:
    with pytest.raises(ContractError):
        normalize_scope({"time": {}})
    with pytest.raises(ContractError):
        normalize_scope({"time": {"not_before": "b", "not_after": "a"}})
    with pytest.raises(ContractError):
        normalize_scope({"time": {"not_before": 7}})
    assert normalize_scope({"time": {"not_before": "a", "not_after": "b"}}) == {
        "time": {"not_before": "a", "not_after": "b"}
    }


def test_normalize_use_count_strict() -> None:
    with pytest.raises(ContractError):
        normalize_scope({"use_count": 0})
    with pytest.raises(ContractError):
        normalize_scope({"use_count": True})
    assert normalize_scope({"use_count": 2}) == {"use_count": 2}


def test_field_loss_rejected() -> None:
    """Serializer field-loss: a truncated record must not silently widen."""
    original = {"tool": "write_file", "target": "path:/tmp/x"}
    partial = {"tool": "write_file"}
    assert not is_subset_scope(partial, original)


# --- is_subset ---------------------------------------------------------------


def test_subset_field_matrix() -> None:
    parent_scope = {
        "tool": "t",
        "target": "p",
        "use_count": 5,
        "time": {"not_before": "a", "not_after": "z"},
    }
    assert is_subset_scope({}, parent_scope)
    assert is_subset_scope(parent_scope, parent_scope)
    # Strict semantics: dropping any parent restriction is widening.
    assert not is_subset_scope({"tool": "t"}, parent_scope)
    assert not is_subset_scope({"tool": "t", "use_count": 5}, parent_scope)
    assert not is_subset_scope(
        {"tool": "t", "target": "p", "time": {"not_before": "a", "not_after": "z"}}, parent_scope
    )
    # Fully restated child with narrowed bounds is a subset.
    assert is_subset_scope(
        {"tool": "t", "target": "p", "use_count": 5, "time": {"not_before": "b", "not_after": "y"}},
        parent_scope,
    )
    assert not is_subset_scope({"tool": "u", "target": "p", "use_count": 5, "time": {"not_before": "a", "not_after": "z"}}, parent_scope)
    assert not is_subset_scope({"tool": "t", "target": "p", "use_count": 6, "time": {"not_before": "a", "not_after": "z"}}, parent_scope)
    assert not is_subset_scope({"tool": "t", "target": "p", "use_count": 5, "time": {"not_before": "9", "not_after": "z"}}, parent_scope)
    # Child-only restriction against a less-restricted parent is narrower;
    # a child claiming a larger count than the parent allows is not.
    assert is_subset_scope({"tool": "t", "use_count": 1}, {"tool": "t", "use_count": 2})
    assert not is_subset_scope({"tool": "t", "use_count": 3}, {"tool": "t", "use_count": 2})


# --- attenuation -------------------------------------------------------------


def test_out_of_parent_issuance_rejected() -> None:
    p = make_parent()
    with pytest.raises(ContractError) as e:
        p.attenuate({"tool": "other_tool"}, child_generation=6)
    assert e.value.code == "ATTENUATION_ESCALATION"


def test_stale_generation_rejected() -> None:
    p = make_parent()
    with pytest.raises(ContractError) as e:
        p.attenuate({"tool": "write_file"}, child_generation=5)
    assert e.value.code == "NON_MONOTONE_GENERATION"
    with pytest.raises(ContractError) as e2:
        p.attenuate({"tool": "write_file"}, child_generation=4)
    assert e2.value.code == "NON_MONOTONE_GENERATION"


def test_valid_attenuation_and_witness() -> None:
    p = make_parent()
    child = p.attenuate({"tool": "write_file", "use_count": 1}, child_generation=6, child_holder="holder:child")
    assert child.holder == "holder:child"
    assert child.generation == 6
    witness = child.subset_witness(p)
    assert witness["contained"] is True
    assert witness["witness_digest"].startswith("sha256:")
    with pytest.raises(ContractError) as e:
        p.subset_witness(child)
    assert e.value.code == "ATTENUATION_ESCALATION"


def test_scope_fingerprint_deterministic() -> None:
    a = scope_fingerprint({"tool": "t", "use_count": 1})
    b = scope_fingerprint({"use_count": 1, "tool": "t"})
    assert a == b
    assert a.startswith("sha256:")


# --- reservation / settlement lifecycle --------------------------------------


def test_crash_after_effect_is_indeterminate() -> None:
    s = make_parent()
    s.reserve(consumption_ref="r1", args_digest="sha256:a", target_ref="path:/tmp/x")
    receipt = s.mark_indeterminate("r1", reason="crash_after_effect")
    assert receipt["record"]["state"] == "indeterminate"
    with pytest.raises(ContractError) as e:
        s.commit("r1", effect_receipt_digest="sha256:e1")
    assert e.value.code == "ALREADY_SETTLED"


def test_commit_release_and_double_settlement() -> None:
    s = make_parent()
    s.reserve(consumption_ref="r1", args_digest="sha256:a")
    r = s.commit("r1", effect_receipt_digest="sha256:e1")
    assert r["record"]["state"] == "committed"
    assert r["consumed_total"] == 1
    with pytest.raises(ContractError) as e:
        s.release("r1")
    assert e.value.code == "ALREADY_SETTLED"
    s.reserve(consumption_ref="r2", args_digest="sha256:b")
    assert s.release("r2")["record"]["state"] == "released"


def test_unknown_and_duplicate_refs_fail_closed() -> None:
    s = make_parent()
    with pytest.raises(ContractError) as e:
        s.commit("ghost", effect_receipt_digest="sha256:x")
    assert e.value.code == "UNKNOWN_CONSUMPTION_REF"
    s.reserve(consumption_ref="r1", args_digest="sha256:a")
    with pytest.raises(ContractError) as e:
        s.reserve(consumption_ref="r1", args_digest="sha256:a")
    assert e.value.code == "DUPLICATE_CONSUMPTION_REF"


def test_target_binding_enforced() -> None:
    s = make_parent()
    with pytest.raises(ContractError) as e:
        s.reserve(consumption_ref="r1", args_digest="sha256:a", target_ref="path:/elsewhere")
    assert e.value.code == "TARGET_OUTSIDE_SCOPE"
    ok = s.reserve(consumption_ref="r1", args_digest="sha256:a", target_ref="path:/tmp/x")
    assert ok["record"]["state"] == "reserved"


def test_finite_allocation_exhaustion() -> None:
    s = AuthorityScopeV1(scope={"use_count": 2}, generation=0)
    s.reserve(consumption_ref="r1", args_digest="sha256:a")
    s.reserve(consumption_ref="r2", args_digest="sha256:b")
    with pytest.raises(ContractError) as e:
        s.reserve(consumption_ref="r3", args_digest="sha256:c")
    assert e.value.code == "USE_COUNT_EXHAUSTED"
    s.release("r1")
    s.reserve(consumption_ref="r3", args_digest="sha256:c")
    with pytest.raises(ContractError) as e2:
        s.reserve(consumption_ref="r4", args_digest="sha256:d")
    assert e2.value.code == "USE_COUNT_EXHAUSTED"


def test_valid_controls_remain_executable() -> None:
    """GREEN control: legitimate flow end-to-end."""
    p = make_parent()
    child = p.attenuate({"tool": "write_file", "target": "path:/tmp/x", "use_count": 1}, child_generation=6)
    r = child.reserve(consumption_ref="r1", args_digest="sha256:a", target_ref="path:/tmp/x")
    assert r["scope_fingerprint"] == child.fingerprint()
    r2 = child.commit("r1", effect_receipt_digest="sha256:e1")
    assert r2["record"]["state"] == "committed"
    assert child.settlement("r1")["state"] == "committed"
