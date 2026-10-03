"""Advisory model-capability pre-flight for the turn loop (part of #5437).

These assert the RELATION the runtime depends on, never a model's capability
value: a capability the assembled request actually carries must draw a warning
when metadata positively declares it unsupported, and a capability metadata
cannot determine must never draw one. Both halves move together because a
pre-flight that fires on "unknown" would warn on every unrecognised model.

Every signal is read off the request that is about to be sent — the tool list the
agent is configured with, the outgoing messages, and the assembled ``api_kwargs``
the reasoning check needs — so "the request carries it" is evidence, not intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from agent import turn_preflight_capabilities as cap
from agent.models_dev import ModelCapabilities


class _Agent:
    """Minimal agent stand-in: only the attributes the pre-flight reads."""

    def __init__(self, **kw):
        self.provider = "openai"
        self.model = "some-model"
        self.tools = [{"type": "function", "function": {"name": "terminal"}}]
        self.reasoning_config = {"effort": "high"}
        self.warnings: list[str] = []
        for k, v in kw.items():
            setattr(self, k, v)

    def _emit_diagnostic_status(self, message):
        self.warnings.append(message)


def _request(**kw):
    """An assembled request payload: only what the pre-flight is allowed to read."""
    return {"model": "some-model", "messages": [{"role": "user", "content": "hi"}], **kw}


def _image_message():
    return [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
    ]}]


def _reasoning_warnings(agent):
    return [w for w in agent.warnings if "reasoning" in w]


def test_declared_unsupported_capability_actually_sent_warns_once(monkeypatch):
    """The relation that matters: metadata declares the capability unsupported AND
    the request really carries it -> exactly one advisory warning, and no repeat."""
    caps = ModelCapabilities(supports_tools=True, supports_reasoning=False)
    monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: caps)
    agent = _Agent()
    wire = _request(extra_body={"reasoning": {"effort": "high"}})

    cap.warn_capability_mismatches(agent, [], wire)
    cap.warn_capability_mismatches(agent, [], wire)  # second API attempt of the same turn

    assert _reasoning_warnings(agent), agent.warnings
    assert len(agent.warnings) == 1, agent.warnings


def test_undeterminable_capability_never_warns(monkeypatch):
    """A capability the catalog cannot determine is not a negative verdict: an
    unresolvable model (None) and an explicitly-unknown capability both stay silent."""
    monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: None)
    unresolved = _Agent()
    cap.warn_capability_mismatches(unresolved, _image_message(), _request())
    assert unresolved.warnings == []

    # Resolvable model, but each capability is unknown (Optional[bool] == None).
    unknown = ModelCapabilities(
        supports_tools=True, supports_vision=None, supports_reasoning=None,
    )
    monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: unknown)
    agent = _Agent()
    cap.warn_capability_mismatches(agent, _image_message(), _request())
    assert agent.warnings == []


def test_preflight_never_blocks_the_turn(monkeypatch):
    """Advisory only: the helper has no verdict to return, so a turn cannot be
    stopped by it, and a capability probe failure degrades to silence."""
    monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe boom")))
    assert cap.warn_capability_mismatches(_Agent(), _image_message(), _request()) is None


@pytest.mark.parametrize(
    "sent", [
        pytest.param({"tools": True, "reasoning": True, "vision": True}, id="all-sent"),
        pytest.param({"tools": False, "reasoning": False, "vision": False}, id="none-sent"),
    ],
)
def test_warning_set_tracks_what_the_request_sends(monkeypatch, sent):
    """Only capabilities actually on the wire are reported; a capability the
    request does not use is not a mismatch however the metadata reads."""
    caps = ModelCapabilities(supports_tools=False, supports_vision=False, supports_reasoning=False)
    monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: caps)
    agent = _Agent(tools=[{"type": "function"}] if sent["tools"] else [])
    messages = _image_message() if sent["vision"] else []
    wire = _request(**({"extra_body": {"reasoning": {"effort": "high"}}} if sent["reasoning"] else {}))

    cap.warn_capability_mismatches(agent, messages, wire)

    reported = " ".join(agent.warnings)
    assert ("tool" in reported) is sent["tools"]
    assert ("image" in reported or "vision" in reported) is sent["vision"]
    assert ("reasoning" in reported) is sent["reasoning"]


def test_reasoning_config_set_but_nothing_reasoning_on_the_wire_never_warns(monkeypatch):
    """The config is intent, the assembled request is evidence.

    A route that already dropped the reasoning fields for this model must not be
    reported as "may reject the request": the unsupported field is not on the wire.
    The same config with reasoning really on the wire is still a mismatch — both
    halves, so the check cannot be satisfied by going silent about everything.
    """
    caps = ModelCapabilities(supports_tools=True, supports_reasoning=False)
    monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: caps)
    agent = _Agent(reasoning_config={"effort": "high"})

    cap.warn_capability_mismatches(agent, [], _request())
    assert _reasoning_warnings(agent) == []

    cap.warn_capability_mismatches(agent, [], _request(extra_body={"reasoning": {"effort": "high"}}))
    assert _reasoning_warnings(agent), agent.warnings


@pytest.mark.parametrize(
    "payload, expected", [
        pytest.param({"reasoning_effort": "high"}, True, id="top-level-effort"),
        pytest.param({"extra_body": {"reasoning": {"effort": "high"}}}, True, id="extra-body-reasoning"),
        pytest.param({"thinking": {"type": "enabled"}}, True, id="anthropic-thinking"),
        pytest.param(
            {"extra_body": {"extra_body": {"google": {"thinking_config": {"thinkingBudget": 0}}}}},
            True, id="nested-gemini-thinking-config",
        ),
        # Structural keys are not reasoning controls, however the model is named: a scan
        # that walked ``messages``/``tools`` recursively would warn on these.
        pytest.param(
            {"messages": [{"role": "user", "content": [{"type": "reasoning"}]}],
             "tools": [{"type": "function", "function": {"name": "think"}}]},
            False, id="message-content-only",
        ),
    ],
)
def test_reasoning_controls_are_recognised_in_their_real_shapes(monkeypatch, payload, expected):
    caps = ModelCapabilities(supports_tools=True, supports_reasoning=False)
    monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: caps)
    agent = _Agent()

    cap.warn_capability_mismatches(agent, [], _request(**payload))

    assert bool(_reasoning_warnings(agent)) is expected, agent.warnings


@dataclass
class _Verdict:
    """What a loop phase returns: ``_run_phase`` reads ``.action`` and copies the fields."""

    action: str = "fallthrough"
    result: Any = None


@dataclass
class _CallVerdict:
    action: str = "break"
    result: Any = None


def test_check_runs_in_the_attempt_loop_on_the_assembled_request(monkeypatch):
    """Locks in WHERE it runs and WHAT it reads: inside the API-attempt loop, on the
    payload ``build_api_request`` just assembled, and without ending the turn.

    The reasoning warning is only reachable from that payload, so a loop that checked
    the agent's intent instead — or that never called the check — goes red here.
    """
    from agent import conversation_loop as loop
    from agent.conversation_loop import _run_api_retry_loop
    from agent.turn_api_request import ApiRequestBuild

    caps = ModelCapabilities(supports_tools=False, supports_vision=False, supports_reasoning=False)
    monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: caps)
    agent = _Agent()
    assembled = _request(tools=[{"type": "function"}], extra_body={"reasoning": {"effort": "high"}})
    built = ApiRequestBuild("fallthrough", _image_message(), None, agent.tools, assembled, {}, [])

    monkeypatch.setattr(loop, "nous_rate_limit_guard", lambda agent: _Verdict())
    monkeypatch.setattr(loop, "build_api_request", lambda agent: built)
    monkeypatch.setattr(loop, "perform_api_call", lambda agent: _CallVerdict())
    state = SimpleNamespace(retry_count=0, max_retries=1, messages=_image_message())

    assert _run_api_retry_loop(agent, state) is None

    reported = " ".join(agent.warnings)
    assert "tool" in reported and "image" in reported and "reasoning" in reported, agent.warnings


class TestAssembledRequestFidelity:
    """A real agent over a real builder: what the pre-flight reports is what the wire carries."""

    def _agent(self, monkeypatch, model, *, route_supports_reasoning):
        import hermes_cli.models as models_mod

        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_failed_at", None)
        monkeypatch.setattr(models_mod, "_openrouter_reasoning_caps_cache", {
            model: {"supports_reasoning": route_supports_reasoning},
        })
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model=model,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
        agent.reasoning_config = {"effort": "high"}
        agent.provider = "openrouter"  # the route the base_url implies; the pre-flight needs it
        agent.warnings = []
        agent._emit_diagnostic_status = agent.warnings.append
        return agent

    @staticmethod
    def _caps(monkeypatch):
        caps = ModelCapabilities(supports_tools=True, supports_reasoning=False)
        monkeypatch.setattr(cap, "_model_capabilities", lambda *a, **k: caps)

    def test_reasoning_config_set_on_a_route_that_drops_it_never_warns(self, monkeypatch):
        """The review's false positive, end to end: ``reasoning_config`` is set but the
        route refuses reasoning controls, so the request carries none and the pre-flight
        must not claim the provider may reject them."""
        agent = self._agent(monkeypatch, "openai/gpt-4o-mini", route_supports_reasoning=False)
        self._caps(monkeypatch)
        messages = [{"role": "user", "content": "hi"}]

        api_kwargs = agent._build_api_kwargs(messages)

        assert "reasoning" not in (api_kwargs.get("extra_body") or {}), api_kwargs
        cap.warn_capability_mismatches(agent, messages, api_kwargs)
        assert agent.warnings == []

        # Anti-vacuity: the same wiring does warn once reasoning is really on the request,
        # so the line above is silence about this payload and not a dead call path.
        cap.warn_capability_mismatches(agent, messages, {**api_kwargs, "reasoning_effort": "high"})
        assert _reasoning_warnings(agent), agent.warnings

    def test_the_same_config_on_a_route_that_carries_it_still_warns(self, monkeypatch):
        agent = self._agent(monkeypatch, "deepseek/deepseek-chat", route_supports_reasoning=True)
        self._caps(monkeypatch)
        messages = [{"role": "user", "content": "hi"}]

        api_kwargs = agent._build_api_kwargs(messages)

        assert (api_kwargs.get("extra_body") or {}).get("reasoning"), api_kwargs
        cap.warn_capability_mismatches(agent, messages, api_kwargs)
        assert _reasoning_warnings(agent), agent.warnings
