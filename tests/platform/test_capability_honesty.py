"""Claims that must match what the runtime actually does.

Two audit findings: a capability flag that promised more than the runtime delivers, and two
limit classifications that promised less. Both are the same failure — a statement about
enforcement that nobody re-checked after the code underneath it moved — and both are
rendered to customers through ``/budget`` and the capability table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nova.policy.limits import ENFORCING_CLASSES
from nova.runtime.hermes.limits import LIMIT_FACTS
from nova.runtime.hermes.readiness import check
from nova.spec import load_bundle
from nova.supervisor.submit import build_work_items

from .conftest import EXAMPLE_BUNDLE


def fact(key: str):
    return next(item for item in LIMIT_FACTS if item.key == key)


# -- finding 15: classifications that understated NOVA ----------------------


@pytest.mark.parametrize("key", ["max_task_runtime_seconds", "max_retries"])
def test_the_task_limits_are_classified_as_enforcing(key):
    """They were RECORDED_ONLY on the grounds that NOVA "does not create tasks". Phase 5
    made it create them, so the grounds stopped being true."""
    assert fact(key).enforcement in ENFORCING_CLASSES


@pytest.mark.parametrize("key", ["max_task_runtime_seconds", "max_retries"])
def test_their_summaries_no_longer_claim_nova_cannot_set_them(key):
    summary = fact(key).summary + fact(key).compiles_to
    assert "cannot set" not in summary
    assert "not settable" not in summary


@pytest.mark.parametrize("key", ["max_task_runtime_seconds", "max_retries"])
def test_they_are_scoped_to_work_nova_submits(key):
    """The qualification is real and must survive: a task a human creates carries whatever
    that command was given."""
    assert "submitted work" in fact(key).compiles_to


def test_every_enforcing_limit_still_names_a_call_site():
    """A control may not claim an enforcing class on the strength of a docstring."""
    for item in LIMIT_FACTS:
        if item.enforcement in ENFORCING_CLASSES:
            assert item.verified_at, f"{item.key} claims enforcement with no call site"


# -- finding 14: the limit must actually be applied -------------------------


def test_a_step_inherits_its_assignees_declared_runtime_cap():
    """Otherwise the classification above would be a lie: the limit is enforceable and was
    being dropped unless a step author happened to restate the number."""
    bundle = load_bundle(EXAMPLE_BUNDLE)
    items = {
        item.key.split(":")[1]: item
        for item in build_work_items(bundle.objectives[0], bundle.agents)
    }
    # handbook-thresholds states no cap and goes to customer-support, which declares 900.
    assert bundle.agent("customer-support").limits.max_task_runtime_seconds == 900
    assert items["handbook-thresholds"].max_runtime_seconds == 900


def test_a_step_that_states_a_cap_overrides_its_assignees():
    """The step is the narrower statement, made by whoever wrote this plan."""
    from dataclasses import replace

    bundle = load_bundle(EXAMPLE_BUNDLE)
    objective = bundle.objectives[0]
    narrowed = replace(
        objective,
        steps=tuple(
            replace(step, max_runtime_seconds=60) if step.id == "pull-ledger" else step
            for step in objective.steps
        ),
    )
    items = {
        item.key.split(":")[1]: item for item in build_work_items(narrowed, bundle.agents)
    }
    assert items["pull-ledger"].max_runtime_seconds == 60


def test_retries_are_inherited_the_same_way():
    bundle = load_bundle(EXAMPLE_BUNDLE)
    items = {
        item.key.split(":")[1]: item
        for item in build_work_items(bundle.objectives[0], bundle.agents)
    }
    assert bundle.agent("customer-support").limits.max_retries == 2
    assert items["handbook-thresholds"].max_retries == 2


def test_an_unknown_assignee_inherits_nothing_rather_than_failing():
    """Routing already refuses an unknown assignee; building items must not crash first."""
    bundle = load_bundle(EXAMPLE_BUNDLE)
    items = build_work_items(bundle.objectives[0], [])
    assert all(item.max_runtime_seconds in (None, 1800) for item in items)


# -- finding 6: credential isolation, stated precisely ----------------------


def test_the_capability_docstring_states_the_boundary():
    """The flag means "per-agent store", not "agents cannot see each other's credentials".
    A reader of the capability table must not have to discover that empirically."""
    from nova.runtime.base import RuntimeCapabilities

    doc = RuntimeCapabilities.__doc__ or ""
    annotations = Path("nova/runtime/base.py").read_text(encoding="utf-8")
    marker = annotations[annotations.index("credential_isolation")::]
    context = annotations[: annotations.index("credential_isolation")][-1200:]
    assert "process environment" in context
    assert "does not mean" in context.lower()


def test_readiness_names_credentials_shared_across_the_host(tmp_path):
    """The actionable half: an operator can see which of their credentials are host-wide
    and therefore not isolated, rather than trusting a boolean."""
    result = check("a", ["SHARED"], profile_dir=tmp_path, environ={"SHARED": "v"})
    assert result.ready
    assert result.host_wide == ("SHARED",)


def test_a_per_agent_credential_is_not_reported_as_host_wide(tmp_path):
    (tmp_path / ".env").write_text("OWN=v\n", encoding="utf-8")
    result = check("a", ["OWN"], profile_dir=tmp_path, environ={})
    assert result.host_wide == ()


def test_host_wide_survives_the_serialised_form(tmp_path):
    """It has to reach the CLI and the control plane, not just the dataclass."""
    result = check("a", ["SHARED"], profile_dir=tmp_path, environ={"SHARED": "v"})
    assert result.to_dict()["host_wide"] == ["SHARED"]
