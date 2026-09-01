"""Public-review regression cases for AuthorityScopeV1.

These are the concrete gaps identified by the public evidence review of PR #20.
"""

from __future__ import annotations

import pytest

from ares_runtime.authority import AuthorityScopeV1, ContractError, is_subset_scope, normalize_scope


def _scope(*, uses: int = 2) -> AuthorityScopeV1:
    return AuthorityScopeV1(
        scope={
            "tool": "write_file",
            "target": "path:/tmp/target",
            "use_count": uses,
            "time": {"not_before": "2026-01-01T00:00:00Z", "not_after": "2026-12-31T00:00:00Z"},
        },
        generation=1,
        holder="holder:parent",
    )


def test_commit_and_indeterminate_remain_charged() -> None:
    s = _scope(uses=2)
    s.reserve(consumption_ref="r1", args_digest="sha256:1", target_ref="path:/tmp/target")
    s.commit("r1", effect_receipt_digest="sha256:effect")
    s.reserve(consumption_ref="r2", args_digest="sha256:2", target_ref="path:/tmp/target")
    s.mark_indeterminate("r2", reason="crash_after_effect")
    with pytest.raises(ContractError) as exc:
        s.reserve(consumption_ref="r3", args_digest="sha256:3", target_ref="path:/tmp/target")
    assert exc.value.code == "USE_COUNT_EXHAUSTED"


def test_release_returns_one_charged_use() -> None:
    s = _scope(uses=1)
    s.reserve(consumption_ref="r1", args_digest="sha256:1", target_ref="path:/tmp/target")
    s.release("r1")
    assert s.reserve(consumption_ref="r2", args_digest="sha256:2", target_ref="path:/tmp/target")["charged_total"] == 1


def test_sibling_attenuation_cannot_duplicate_parent_budget() -> None:
    parent = _scope(uses=2)
    parent.attenuate({"tool": "write_file", "target": "path:/tmp/target", "use_count": 1, "time": {"not_before": "2026-01-01T00:00:00Z", "not_after": "2026-12-31T00:00:00Z"}}, child_generation=2)
    parent.attenuate({"tool": "write_file", "target": "path:/tmp/target", "use_count": 1, "time": {"not_before": "2026-01-01T00:00:00Z", "not_after": "2026-12-31T00:00:00Z"}}, child_generation=3)
    with pytest.raises(ContractError) as exc:
        parent.attenuate({"tool": "write_file", "target": "path:/tmp/target", "use_count": 1, "time": {"not_before": "2026-01-01T00:00:00Z", "not_after": "2026-12-31T00:00:00Z"}}, child_generation=4)
    assert exc.value.code == "USE_COUNT_EXHAUSTED"


def test_target_scope_requires_target_ref_and_scope_copy_is_deep() -> None:
    s = _scope()
    with pytest.raises(ContractError) as exc:
        s.reserve(consumption_ref="r1", args_digest="sha256:1")
    assert exc.value.code == "TARGET_REQUIRED"
    copy = s.scope
    copy["time"]["not_after"] = "2099-01-01T00:00:00Z"
    assert s.scope["time"]["not_after"] == "2026-12-31T00:00:00Z"


def test_time_scope_is_instant_not_lexical_and_witness_binds_holders() -> None:
    normalized = normalize_scope({"time": {"not_before": "2026-01-01T01:00:00+01:00", "not_after": "2026-01-01T02:00:00+01:00"}})
    assert normalized["time"]["not_before"] == "2026-01-01T00:00:00Z"
    parent = _scope(uses=1)
    child = parent.attenuate({"tool": "write_file", "target": "path:/tmp/target", "use_count": 1, "time": {"not_before": "2026-01-01T00:00:00Z", "not_after": "2026-12-31T00:00:00Z"}}, child_generation=2, child_holder="holder:child")
    witness = child.subset_witness(parent)
    assert witness["child_holder"] == "holder:child"
    assert witness["parent_holder"] == "holder:parent"
    stale = AuthorityScopeV1(scope=child.scope, generation=1, holder="holder:child")
    with pytest.raises(ContractError) as exc:
        stale.subset_witness(parent)
    assert exc.value.code == "NON_MONOTONE_GENERATION"
