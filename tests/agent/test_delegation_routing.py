"""Behavioral delegation-routing tests for the primary (orchestrator) agent.

Why: SOUL.md gained a "Delegation Routing" section so the orchestrator
delegates domain work to specialized profiles instead of answering from
weights (which caused hallucinated service states, fabricated timing, etc).
Two things must hold for that fix to work, and this file locks both down so
a future SOUL.md edit or a refactor of the delegate dispatch path can't
silently regress them:

  1. PROMPT CONTRACT — the routing rules SOUL.md carries must actually reach
     the assembled system prompt the model sees. If the rules don't reach the
     prompt, the real model can't act on them. (TestSoulRoutingReachesPrompt)

  2. DISPATCH CONTRACT — when the model decides to emit a delegate_task call
     with a chosen profile, the agent loop must route it to delegate_task with
     that exact profile (not drop it, not answer inline). This is the plumbing
     that turns the model's choice into an actual subagent run.
     (TestDelegationDispatch)

What this file does NOT do: it cannot prove a *mocked* LLM "chooses" to
delegate — with the network mocked, the model's choice is whatever the fake
returns. Proving the real model now delegates is the job of the live smoke
test (run_agent CLI against the gateway), not a hermetic unit test. Faking the
LLM and then asserting it "chose" to delegate would be testing the mock, not
the system (see tests/.../testing-anti-patterns). So here we assert the two
deterministic halves the live behavior depends on.

Test: run `pytest tests/agent/test_delegation_routing.py -v`. Prompt tests
assert routing keywords + profile names appear in the built prompt; dispatch
tests drive a full run_conversation turn with a fake client that emits one
delegate_task tool call and assert _dispatch_delegate_task received the
expected profile.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Fake LLM plumbing — mirrors tests/run_agent/test_dict_tool_call_args.py so we
# never touch the network. The fake returns a delegate_task tool call on the
# first API call, then a plain "stop" on the second so the loop terminates.
# ---------------------------------------------------------------------------

def _delegate_tool_call(profile: str, goal: str, call_id: str = "call_1"):
    """A single OpenAI-shaped tool_call selecting delegate_task with a profile."""
    args = json.dumps({"profile": profile, "goal": goal})
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="delegate_task", arguments=args),
    )


def _response_with_delegate(profile: str, goal: str):
    assistant = SimpleNamespace(
        content=None,
        reasoning=None,
        tool_calls=[_delegate_tool_call(profile, goal)],
    )
    choice = SimpleNamespace(message=assistant, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


def _stop_response(text: str = "done"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, reasoning=None, tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def _answer_inline_response(text: str):
    """A turn where the model answers directly with no tool call at all."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, reasoning=None, tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


class _FakeChatCompletions:
    """Scripted completions endpoint. ``script`` is a list of response builders."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        idx = min(self.calls - 1, len(self._script) - 1)
        return self._script[idx]()


class _FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(script))


def _make_orchestrator(monkeypatch, script):
    """Build a primary AIAgent wired to a scripted fake LLM client.

    delegate_task is in valid_tool_names so the dispatch branch is reachable.
    Streaming is disabled so the fake's non-stream `create()` is used.
    """
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: _FakeClient(script))
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *a, **k: [
            {"type": "function", "function": {"name": "delegate_task"}},
        ],
    )
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda *a, **k: {})

    agent = AIAgent(
        model="openai/gpt-oss-120b",
        api_key="test-key-1234567890",
        base_url="https://openrouter.ai/api/v1",
        platform="cli",
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._disable_streaming = True
    return agent


def _run_and_capture_delegations(monkeypatch, model_profile: str, goal: str):
    """Drive one turn; return the list of (profile, goal) dispatched.

    The model is scripted to emit exactly one delegate_task(profile=...) call,
    then stop. We patch _dispatch_delegate_task so no real subagent spawns —
    we only record what the agent loop forwarded.
    """
    script = [lambda: _response_with_delegate(model_profile, goal), _stop_response]
    agent = _make_orchestrator(monkeypatch, script)

    captured = []

    def _fake_dispatch(self, function_args):
        captured.append(
            (function_args.get("profile"), function_args.get("goal"))
        )
        return json.dumps({"status": "completed", "summary": "ok"})

    monkeypatch.setattr(AIAgent, "_dispatch_delegate_task", _fake_dispatch)
    agent.run_conversation(goal)
    return captured


# ---------------------------------------------------------------------------
# 1. PROMPT CONTRACT — SOUL.md routing rules reach the assembled system prompt.
#
# The test suite is hermetic: an autouse conftest fixture redirects HERMES_HOME
# to a per-test tempdir with NO SOUL.md, so load_soul_md() would fall back to
# DEFAULT_AGENT_IDENTITY. To test the *wiring* (SOUL.md on disk -> loader ->
# build_system_prompt) deterministically, we seed the test HERMES_HOME with the
# canonical routing SOUL.md (the exact text shipped to the real HERMES_HOME) and
# assert it survives the round-trip. A separate test reads the real deployed
# file to confirm the shipped artifact matches.
# ---------------------------------------------------------------------------

# Canonical routing SOUL.md — kept in lockstep with the deployed
# $HERMES_HOME/SOUL.md. CANONICAL_PROFILES are the agent_profiles a domain
# query must route to (all exist in config.yaml).
CANONICAL_PROFILES = [
    "homelab", "mail", "calendar", "engineer", "debugger", "kb", "search", "think",
]

ROUTING_SOUL = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research.\n\n"
    "## Delegation Routing\n\n"
    "You are an orchestrator. For most non-trivial tasks, delegate to a "
    "specialized profile using delegate_task(profile=\"...\", goal=\"...\").\n\n"
    "ALWAYS delegate (never answer from weights):\n"
    "- Infrastructure, containers, services, server status -> profile=\"homelab\"\n"
    "- Email -> profile=\"mail\"\n"
    "- Calendar -> profile=\"calendar\"\n"
    "- Code/engineering -> profile=\"engineer\"\n"
    "- Fault diagnosis -> profile=\"debugger\"\n"
    "- Knowledge base -> profile=\"kb\"\n"
    "- Web research -> profile=\"search\"\n"
    "- Deep reasoning -> profile=\"think\"\n\n"
    "When uncertain whether to delegate: DELEGATE. If you cannot answer with "
    "certainty from direct knowledge, delegate -- do not fabricate."
)


@pytest.fixture
def seeded_hermes_home(monkeypatch):
    """Write the canonical routing SOUL.md into the per-test HERMES_HOME.

    Uses the conftest-provided tempdir HERMES_HOME (already isolated) so the
    production loader resolves to our seeded file. Returns nothing — callers
    then exercise the real load_soul_md()/build_system_prompt() path.
    """
    import os
    from pathlib import Path

    home = Path(os.environ["HERMES_HOME"])
    (home / "SOUL.md").write_text(ROUTING_SOUL, encoding="utf-8")
    # load_soul_md caches nothing, but get_hermes_home() may — re-resolve.
    return home


class TestSoulRoutingReachesPrompt:
    """The routing guidance must reach the prompt the model is sent.

    If SOUL.md is reverted to a bare identity prompt, or the prompt builder
    stops embedding SOUL.md, these fail — exactly the regression that re-opened
    the hallucination bug.
    """

    def _soul(self):
        from agent.prompt_builder import load_soul_md
        return load_soul_md() or ""

    def test_soul_declares_orchestrator_and_delegation(self, seeded_hermes_home):
        soul = self._soul()
        assert "Delegation Routing" in soul, "SOUL.md missing Delegation Routing section"
        assert "delegate_task(" in soul, "SOUL.md must reference the delegate_task tool"
        assert "orchestrator" in soul.lower()

    def test_soul_has_no_fabrication_rule(self, seeded_hermes_home):
        soul = self._soul().lower()
        # The core anti-hallucination instruction must be present and tie
        # uncertainty to delegation rather than guessing.
        assert "delegate" in soul and "fabricate" in soul
        assert "from weights" in soul

    @pytest.mark.parametrize("profile", CANONICAL_PROFILES)
    def test_soul_maps_core_domains_to_real_profiles(self, seeded_hermes_home, profile):
        # Each routed profile name must actually appear so the model is told
        # which profile to pick. These names exist in config.yaml agent_profiles.
        assert f'profile="{profile}"' in self._soul()

    def test_routing_present_in_built_system_prompt(self, seeded_hermes_home):
        """End-to-end: the prompt builder must embed SOUL.md (incl. routing).

        Guards against a future change to system_prompt assembly that drops or
        replaces SOUL.md before the model ever sees the routing rules.
        """
        from agent.system_prompt import build_system_prompt

        agent = SimpleNamespace(
            load_soul_identity=True,
            skip_context_files=True,
            valid_tool_names=["delegate_task"],
            _task_completion_guidance=False,
            _tool_use_enforcement=False,
            _environment_probe=False,
            _kanban_worker_guidance="",
            _memory_store=None,
            _memory_manager=None,
            _memory_enabled=False,
            _user_profile_enabled=False,
            model="openai/gpt-oss-120b",
            provider="openrouter",
            platform="cli",
            pass_session_id=False,
            session_id="",
        )
        prompt = build_system_prompt(agent)
        assert "Delegation Routing" in prompt
        assert 'profile="homelab"' in prompt

    def test_default_identity_without_routing_is_caught(self):
        """Sanity: with an empty HERMES_HOME (no SOUL.md), the fallback identity
        carries NO routing. This documents the pre-fix state and proves the
        seeded tests above are actually exercising the routing SOUL.md, not a
        baked-in default."""
        # No seeded_hermes_home fixture here -> conftest tempdir has no SOUL.md.
        assert "Delegation Routing" not in self._soul()


class TestDeployedSoulArtifact:
    """Verify the SHIPPED SOUL.md (the real file at the deployment HERMES_HOME)
    contains the routing rules. Skipped on machines where the deployment file
    isn't present, so the suite stays portable/CI-safe."""

    def _deployed_soul_path(self):
        # The gateway/CLI runs with HERMES_HOME=/opt/hermes/home/.hermes on this
        # deployment. conftest blanks HERMES_HOME, so resolve the deployment
        # path directly rather than via the (test-redirected) env var.
        from pathlib import Path
        return Path("/opt/hermes/home/.hermes/SOUL.md")

    def test_deployed_soul_has_routing(self):
        path = self._deployed_soul_path()
        if not path.exists():
            pytest.skip(f"deployed SOUL.md not present at {path}")
        text = path.read_text(encoding="utf-8")
        assert "Delegation Routing" in text
        assert 'delegate_task(' in text
        assert "fabricate" in text.lower()
        for profile in CANONICAL_PROFILES:
            assert f'profile="{profile}"' in text, f"deployed SOUL.md missing {profile} route"


# ---------------------------------------------------------------------------
# 2. DISPATCH CONTRACT — a model-chosen delegate_task reaches delegate_task
#    with the chosen profile, end to end through run_conversation.
# ---------------------------------------------------------------------------

class TestDelegationDispatch:
    """When the model emits delegate_task(profile=X), the agent loop must
    forward profile=X to the delegation machinery (not drop it, not inline)."""

    def test_infrastructure_query_delegates_to_homelab(self, monkeypatch):
        captured = _run_and_capture_delegations(
            monkeypatch, "homelab", "is plex running?"
        )
        assert captured, "agent did not delegate; it answered without a tool call"
        profile, goal = captured[0]
        assert profile == "homelab"
        assert "plex" in goal

    def test_service_status_query_delegates(self, monkeypatch):
        captured = _run_and_capture_delegations(
            monkeypatch, "homelab", "is the hermes gateway running?"
        )
        assert captured, "service-status query was not delegated"
        assert captured[0][0] == "homelab"

    def test_check_plex_status_delegates_to_homelab(self, monkeypatch):
        captured = _run_and_capture_delegations(
            monkeypatch, "homelab", "check plex status"
        )
        assert captured and captured[0][0] == "homelab"

    @pytest.mark.parametrize("eng_profile", ["engineer", "debugger"])
    def test_code_query_delegates_to_engineering_profile(self, monkeypatch, eng_profile):
        captured = _run_and_capture_delegations(
            monkeypatch, eng_profile, "find and fix the race condition in the gateway"
        )
        assert captured and captured[0][0] == eng_profile

    def test_dispatched_profile_is_not_silently_rewritten(self, monkeypatch):
        """The agent must forward the model's chosen profile verbatim."""
        captured = _run_and_capture_delegations(monkeypatch, "mail", "search my inbox")
        assert captured and captured[0][0] == "mail"

    def test_simple_math_does_not_force_delegation(self, monkeypatch):
        """A turn the model answers inline (no tool call) must NOT be coerced
        into a delegation. SOUL.md explicitly allows direct answers for simple
        certain facts; the loop must honor a tool-call-free response."""
        script = [lambda: _answer_inline_response("4")]
        agent = _make_orchestrator(monkeypatch, script)

        captured = []

        def _fake_dispatch(self, function_args):  # pragma: no cover - must not run
            captured.append(function_args.get("profile"))
            return "{}"

        monkeypatch.setattr(AIAgent, "_dispatch_delegate_task", _fake_dispatch)
        result = agent.run_conversation("what is 2+2")

        assert captured == [], "simple inline answer must not trigger delegation"
        assert "4" in result["final_response"]
