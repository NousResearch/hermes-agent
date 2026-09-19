"""Behavior contracts for the typed probabilistic decision runtime (#113008)."""

from __future__ import annotations

import contextvars
import json
import time
from pathlib import Path

import pytest

from agent.decision_eval import ReplayCase, replay_decisions
from agent.decision_provider import (
    BinaryQuestion,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionProvider,
    DecisionStatus,
    OrdinalQuestion,
    ProviderDecision,
)
from agent.decision_registry import _reset_for_tests
from agent.secret_scope import (
    get_secret,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture(autouse=True)
def _clean_decision_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


class _FixtureProvider(DecisionProvider):
    def __init__(self, name="fixture", *, invert=False, behavior="normal"):
        self.name = name
        self.invert = invert
        self.behavior = behavior
        self.requests = []

    def evaluate(self, request):
        self.requests.append(request)
        if self.behavior == "timeout":
            time.sleep(0.05)
        if self.behavior == "error":
            raise RuntimeError("backend failed")
        if self.behavior == "value-error":
            raise ValueError("backend input rejected")
        if self.behavior == "abstain":
            return ProviderDecision({}, abstained=True, abstention_reason="insufficient evidence")

        answers = {}
        for key, question in request.questions.items():
            labels = question.labels
            winning = labels[-1]
            if self.invert:
                winning = labels[0]
            probabilities = {label: 0.1 / (len(labels) - 1) for label in labels}
            probabilities[winning] = 0.9
            if self.behavior == "malformed":
                probabilities[winning] = 0.8
            answers[key] = DecisionAnswer(probabilities)
        return ProviderDecision(
            answers,
            model="fixture-model",
            version="v1",
            usage={"cost": 0.001, "requests": 1},
        )


def _context(tmp_path: Path):
    manager = PluginManager(scope_key=str(tmp_path))
    return manager, PluginContext(PluginManifest(name="decision-consumer"), manager)


def test_plugin_facade_preserves_typed_probabilities_and_explicit_state_boundary(tmp_path):
    manager, ctx = _context(tmp_path)
    provider = _FixtureProvider()
    tools_before = set(manager._plugin_tool_names)
    prompts_before = dict(manager._system_prompt_sections)
    handle = ctx.register_decision_provider(provider)
    caller_state = {"request": "private raw input", "nested": {"candidate": "alpha"}}

    result = ctx.decision.evaluate(
        task="route",
        state=caller_state,
        questions={
            "needed": BinaryQuestion("Is a specialist useful?"),
            "choice": ChoiceQuestion("Choose one", ["alpha", "beta"]),
            "strength": OrdinalQuestion("Rate the match", ["low", "medium", "high"]),
        },
        mode="shadow",
    )

    assert result.status is DecisionStatus.AVAILABLE
    assert result.answers["needed"].probabilities == {False: 0.1, True: 0.9}
    assert result.answers["choice"].selected == "beta"
    assert result.answers["strength"].probabilities == {"low": 0.05, "medium": 0.05, "high": 0.9}
    assert provider.requests[0].state == caller_state
    provider.requests[0].state["nested"]["candidate"] = "mutated"
    assert caller_state["nested"]["candidate"] == "alpha"
    assert manager._plugin_tool_names == tools_before
    assert manager._system_prompt_sections == prompts_before

    event = json.loads((tmp_path / "logs" / "decisions.jsonl").read_text(encoding="utf-8"))
    assert event["mode"] == "shadow"
    assert event["provider"] == "fixture"
    assert event["model"] == "fixture-model"
    assert event["version"] == "v1"
    assert event["answers"]["choice"]["probabilities"] == {"alpha": 0.1, "beta": 0.9}
    assert "private raw input" not in json.dumps(event)

    assert handle is not None
    handle.dispose()
    unavailable = ctx.decision.evaluate(
        task="route", state={}, questions={"needed": BinaryQuestion("Still useful?")},
        provider="fixture",
    )
    assert unavailable.status is DecisionStatus.UNAVAILABLE


def test_provider_receives_only_explicit_state_and_typed_questions(tmp_path):
    ambient = contextvars.ContextVar("decision_test_ambient", default="default")

    class InspectingProvider(_FixtureProvider):
        def evaluate(self, request):
            self.ambient = ambient.get()
            return super().evaluate(request)

    _, ctx = _context(tmp_path)
    provider = InspectingProvider()
    ctx.register_decision_provider(provider)
    token = ambient.set("session-private")
    try:
        result = ctx.decision.evaluate(
            task="isolation",
            state={"projected": "visible"},
            questions={"gate": BinaryQuestion("Continue?")},
            provider="fixture",
        )
    finally:
        ambient.reset(token)

    assert result.status is DecisionStatus.AVAILABLE
    assert provider.ambient == "default"
    assert provider.requests[0].state == {"projected": "visible"}

    class LabelsOnly:
        labels = (False, True)

    with pytest.raises(TypeError, match="unsupported type"):
        ctx.decision.evaluate(
            task="isolation", state={}, questions={"gate": LabelsOnly()}, provider="fixture",
        )


def test_provider_keeps_owning_profile_capabilities_without_ambient_context(tmp_path):
    ambient = contextvars.ContextVar("decision_test_profile_ambient", default="default")
    profile_home = tmp_path / "profiles" / "secondary"

    class ProfileProvider(_FixtureProvider):
        def is_available(self):
            self.availability_home = get_hermes_home()
            self.availability_secret = get_secret("PROBE_KEY")
            self.availability_ambient = ambient.get()
            return True

        def evaluate(self, request):
            self.home = get_hermes_home()
            self.secret = get_secret("PROBE_KEY")
            self.ambient = ambient.get()
            return super().evaluate(request)

    _, ctx = _context(profile_home)
    provider = ProfileProvider()
    ctx.register_decision_provider(provider)
    prior_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(profile_home)
    secret_token = set_secret_scope({"PROBE_KEY": "secondary-secret"})
    ambient_token = ambient.set("session-private")
    try:
        result = ctx.decision.evaluate(
            task="profile-isolation",
            state={"projected": "visible"},
            questions={"gate": BinaryQuestion("Continue?")},
            provider="fixture",
        )
    finally:
        ambient.reset(ambient_token)
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(prior_multiplex)

    assert result.status is DecisionStatus.AVAILABLE
    assert provider.availability_home == profile_home
    assert provider.availability_secret == "secondary-secret"
    assert provider.availability_ambient == "default"
    assert provider.home == profile_home
    assert provider.secret == "secondary-secret"
    assert provider.ambient == "default"


def test_failures_abstention_staging_and_replay_are_explicit(tmp_path):
    _, ctx = _context(tmp_path)
    good = _FixtureProvider("good")
    bad = _FixtureProvider("bad", invert=True)
    malformed = _FixtureProvider("malformed", behavior="malformed")
    slow = _FixtureProvider("slow", behavior="timeout")
    abstaining = _FixtureProvider("abstaining", behavior="abstain")
    failing = _FixtureProvider("failing", behavior="error")
    value_error = _FixtureProvider("value-error", behavior="value-error")
    for provider in (good, bad, malformed, slow, abstaining, failing, value_error):
        ctx.register_decision_provider(provider)

    ambiguous = ctx.decision.evaluate(
        task="stage-one", state={}, questions={"gate": BinaryQuestion("Continue?")},
    )
    assert ambiguous.status is DecisionStatus.UNAVAILABLE

    stage_one = ctx.decision.evaluate(
        task="stage-one",
        state={"candidates": ["a", "b"]},
        questions={
            "gate": BinaryQuestion("Continue?"),
            "rank": ChoiceQuestion("Choose", ["a", "b"]),
        },
        provider="good",
    )
    ctx.decision.evaluate(
        task="stage-two",
        state={"shortlist": [stage_one.answers["rank"].selected]},
        questions={"inspect": BinaryQuestion("Accept?")},
        provider="good",
    )
    assert [set(request.questions) for request in good.requests[:2]] == [
        {"gate", "rank"}, {"inspect"},
    ]

    question = {"gate": BinaryQuestion("Continue?")}
    assert ctx.decision.evaluate(
        task="failure", state={}, questions=question, provider="malformed",
    ).status is DecisionStatus.MALFORMED
    assert ctx.decision.evaluate(
        task="failure", state={}, questions=question, provider="slow", timeout=0.001,
    ).status is DecisionStatus.TIMEOUT
    assert ctx.decision.evaluate(
        task="failure", state={}, questions=question, provider="failing",
    ).status is DecisionStatus.PROVIDER_ERROR
    assert ctx.decision.evaluate(
        task="failure", state={}, questions=question, provider="value-error",
    ).status is DecisionStatus.PROVIDER_ERROR
    malformed_usage = _FixtureProvider("malformed-usage")
    malformed_usage.evaluate = lambda request: ProviderDecision(
        {"gate": DecisionAnswer({False: 0.1, True: 0.9})}, usage=None,
    )
    ctx.register_decision_provider(malformed_usage)
    assert ctx.decision.evaluate(
        task="failure", state={}, questions=question, provider="malformed-usage",
    ).status is DecisionStatus.MALFORMED
    abstained = ctx.decision.evaluate(
        task="failure", state={}, questions=question, provider="abstaining",
    )
    assert abstained.status is DecisionStatus.ABSTAINED
    assert abstained.fallback_reason == "insufficient evidence"

    private_abstention = _FixtureProvider("private-abstention", behavior="abstain")
    private_abstention.evaluate = lambda request: ProviderDecision(
        {}, abstained=True, abstention_reason=request.state["secret"],
    )
    ctx.register_decision_provider(private_abstention)
    private_result = ctx.decision.evaluate(
        task="private", state={"secret": "raw-private-state"}, questions=question,
        provider="private-abstention",
    )
    assert private_result.fallback_reason == "raw-private-state"
    telemetry = (tmp_path / "logs" / "decisions.jsonl").read_text(encoding="utf-8")
    assert "raw-private-state" not in telemetry

    corpus = [
        ReplayCase("routing", {"row": 1}, question, {"gate": True}),
        ReplayCase("routing", {"row": 2}, question, {"gate": True}),
    ]
    report = replay_decisions(ctx.decision, corpus, ["good", "bad"], acceptance_threshold=0.8)
    assert report["good"]["routing"].coverage == 1.0
    assert report["good"]["routing"].top1_error == 0.0
    assert report["bad"]["routing"].top1_error == 1.0
    assert report["good"]["routing"].brier_score < report["bad"]["routing"].brier_score
    assert report["good"]["routing"].usage == {"cost": 0.002, "requests": 2.0}

    with pytest.raises(ValueError, match="outside the question domain"):
        replay_decisions(
            ctx.decision,
            [ReplayCase("routing", {}, question, {"gate": "outside-domain"})],
            ["abstaining"],
        )
