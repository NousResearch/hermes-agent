"""Per-channel approval: a channel may tighten what needs a human, and never loosen it.

The mechanism is unusual enough to state up front. The runtime's policy hook is never told
which channel a turn arrived on — it receives ``tool_name``, ``args``, ``task_id``,
``session_id``, ``turn_id`` and ``tool_call_id``, and the session *id* (unlike the session
*key*) carries no platform. So a per-channel rule evaluated inside the hook would be a rule
that never fires. What the runtime *does* enforce is one compiled policy per profile, and a
profile is what the channel layer already routes to — so a channel that tightens approval
derives an agent variant with its own policy.

The tests that matter most are the ones about widening. An approval requirement that
*granted* reach would be the opposite of a control, and ``decide()`` checks approval actions
before the allow-list, which makes that failure one line of code away at all times.
"""

from __future__ import annotations

import pytest

from nova.channels import parse_channels
from nova.channels.derive import (
    DerivedAgent,
    additional_approvals,
    derived_id,
    plan_derivations,
    reachable_approvals,
    route_target,
)
from nova.errors import SpecError
from nova.policy.decide import ALLOW, DENY, REQUIRE_APPROVAL, decide
from nova.spec import load_bundle

from .conftest import EXAMPLE_BUNDLE


@pytest.fixture
def example():
    return load_bundle(EXAMPLE_BUNDLE)


# -- the mechanism -----------------------------------------------------------


def test_a_channel_that_tightens_approval_derives_a_variant(example):
    derivations = plan_derivations(example)
    assert derivations, "the example channel escalates an action operations can perform"
    derived = derivations[0]
    assert derived.base_agent == "operations"
    assert derived.channel_id == "acme-support-telegram"
    assert "send_external_email" in derived.added_approvals


def test_the_variant_differs_from_its_base_only_in_approval(example):
    """A variant that could drift from its base would be a second agent wearing the first
    one's name, and the drift would show as different answers on different channels."""
    from dataclasses import replace

    from nova.channels.derive import derive_specs

    base = next(a for a in example.agents if a.id == "operations")
    variant = derive_specs(example)[0]

    assert variant.id != base.id
    assert set(base.approval.required_for) < set(variant.approval.required_for)
    # Everything else is identical: compare with the two known differences normalised away.
    assert replace(variant, id=base.id, approval=base.approval) == base


def test_routes_point_at_the_variant_not_the_base(example):
    from nova.runtime.hermes.channels import plan

    compiled = plan(example.channels, plan_derivations(example))
    targets = {route["profile"] for route in compiled.routes}
    assert "operations__acme-support-telegram" in targets
    assert "operations" not in targets, (
        "the base agent must not be reachable over a channel that tightened its approvals"
    )


def test_the_grant_serves_the_variant_so_the_runtime_enforces_it(example):
    """NOVA's routing is the first gate; the runtime refusing to serve an unlisted profile
    is the one that holds when somebody edits configuration by hand."""
    from nova.runtime.hermes.channels import plan

    compiled = plan(example.channels, plan_derivations(example))
    assert "operations__acme-support-telegram" in compiled.served_agents
    for route in compiled.routes:
        assert route["profile"] in compiled.served_agents


def test_no_variant_when_the_channel_asks_for_nothing_new(example):
    """`customer-support` already escalates the action the channel requires, so deriving a
    profile would double an agent to change nothing."""
    assert not any(d.base_agent == "customer-support" for d in plan_derivations(example))


def test_additional_approvals_subtracts_what_is_already_required():
    assert additional_approvals(["refund", "x"], ["refund"], []) == ("x",)
    assert additional_approvals(["refund"], [], ["refund"]) == ()


# -- widening: the failure this must never have -------------------------------


class _Action:
    def __init__(self, tools):
        self.tools = tools


def test_an_approval_for_a_tool_the_agent_cannot_reach_is_dropped():
    """``decide()`` tests approval actions BEFORE the allow-list, so an approval naming a
    tool outside the allow-list resolves to require_approval rather than deny. Compiling one
    would *grant* the tool behind a human gate — a channel declaration widening an agent's
    reach, which is precisely what must be impossible."""
    reachable, dropped = reachable_approvals(
        ["refund", "email"],
        {"refund": _Action(["crm_refund"]), "email": _Action(["email_send"])},
        granted=["email_send"],
    )
    assert reachable == ("email",)
    assert dropped == ("refund",)


def test_a_channel_cannot_widen_an_agents_reach(example):
    """End to end, on the example: the variant's allow-list is byte-identical to its base."""
    from nova.policy import compile_policy
    from nova.channels.derive import derive_specs

    base = next(a for a in example.agents if a.id == "operations")
    variant = derive_specs(example)[0]

    base_doc = compile_policy(base, example.policy).document
    variant_doc = compile_policy(variant, example.policy).document
    assert variant_doc["allow"] == base_doc["allow"], "the channel granted a tool"
    assert variant_doc["deny"] == base_doc["deny"], "the channel revoked a denial"


def test_the_variant_escalates_where_the_base_allows(example):
    """The whole point, asserted through the real decision function rather than by reading
    the compiled document: same agent, same tool, two postures."""
    from nova.policy import compile_policy
    from nova.channels.derive import derive_specs

    base = compile_policy(
        next(a for a in example.agents if a.id == "operations"), example.policy
    ).document
    variant = compile_policy(derive_specs(example)[0], example.policy).document

    assert decide(base, "email_send", calls_used=0).effect == ALLOW
    assert decide(variant, "email_send", calls_used=0).effect == REQUIRE_APPROVAL


def test_a_tool_denied_to_the_base_stays_denied_on_the_variant(example):
    from nova.policy import compile_policy
    from nova.channels.derive import derive_specs

    variant = compile_policy(derive_specs(example)[0], example.policy).document
    assert decide(variant, "execute_code", calls_used=0).effect == DENY


# -- declaration errors -------------------------------------------------------


def test_approval_for_an_action_the_policy_does_not_define_is_refused(tmp_path):
    import shutil

    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    path = root / "channels.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("send_external_email", "teleportation"),
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="does not define as an action"):
        plan_derivations(load_bundle(root))


def test_approval_with_no_policy_at_all_is_refused(tmp_path):
    import shutil

    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    (root / "policy.yaml").unlink()
    with pytest.raises(SpecError, match="declares no policy"):
        plan_derivations(load_bundle(root))


def test_a_derived_name_too_long_for_a_profile_is_refused():
    """The runtime's profile grammar caps a name at 64 characters. Catching it here names
    the cause; catching it at materialization names a directory."""
    with pytest.raises(SpecError, match="not a valid profile id"):
        derived_id("a" * 40, "b" * 40)


def test_route_target_falls_back_to_the_base_agent():
    derivations = (DerivedAgent(id="a__c", base_agent="a", channel_id="c", added_approvals=("x",)),)
    assert route_target(derivations, "c", "a") == "a__c"
    assert route_target(derivations, "other", "a") == "a"
    assert route_target(derivations, "c", "b") == "b"


def test_a_disabled_channel_derives_nothing(tmp_path):
    import shutil

    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    path = root / "channels.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    provider: telegram", "    provider: telegram\n    enabled: false"
        ),
        encoding="utf-8",
    )
    assert plan_derivations(load_bundle(root)) == ()


def test_approval_is_carried_in_the_bundle_digest(example):
    """Tightening what needs a human must move the provenance, or nobody can prove when the
    requirement started applying."""
    from dataclasses import replace

    from nova.channels.spec import ChannelApproval

    loosened = replace(
        example,
        channels=tuple(
            replace(c, approval=ChannelApproval()) for c in example.channels
        ),
    )
    assert example.digest() != loosened.digest()


def test_readiness_checks_the_profile_that_actually_runs(example, tmp_path):
    """Found on a live deployment: the dashboard reported the BASE agent as missing a
    credential while the variant was the profile the adapter would read. Reporting the wrong
    profile tells an operator the credential is in place while every message fails."""
    from nova.runtime.hermes.channels import readiness

    derivations = plan_derivations(example)
    variant = derivations[0].id

    for profile in ("customer-support", variant):
        target = tmp_path / "profiles" / profile
        target.mkdir(parents=True)
        (target / ".env").write_text("TELEGRAM_BOT_TOKEN=present\n", encoding="utf-8")

    rows = readiness(example.channels, home=tmp_path, derivations=derivations)
    assert rows[0]["ready"] is True, rows[0]["missing_by_agent"]
    assert variant in rows[0]["missing_by_agent"]
    assert "operations" not in rows[0]["missing_by_agent"], (
        "readiness named the base agent, whose .env the adapter will never read"
    )
