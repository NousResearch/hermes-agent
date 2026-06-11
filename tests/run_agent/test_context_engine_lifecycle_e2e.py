"""End-to-end tests for ContextEngine lifecycle hooks through run_conversation.

Drives the REAL conversation loop (mocked provider call only, following the
harness pattern of tests/run_agent/test_context_token_tracking.py) with an
OpenViking-style full-capability engine, and verifies the functional goals:

1. Request assembly shapes what the provider receives (request-only view);
   the working message list / session persistence never see it.
2. Turn observation fires after the loop with the finalized message list.
3. Legacy path (built-in compressor) is unchanged: outbound equals input.
4. Assembly failure fails open: original outbound, turn completes.
5. Lineage kwargs reach the engine (boundary_reason / lineage_root_id).
"""

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.context_engine import ContextEngine, ContextEngineCapabilities

SUMMARY_MARK = "[ARCHIVE SUMMARY] previously discussed the Lisbon trip"


# ── OpenViking-style full-capability engine ──────────────────────────────────


class OVStyleEngine(ContextEngine):
    """Archive-summary + recall-injection engine shaped like the OpenViking
    OpenClaw plugin: observation ingest + request assembly."""

    def __init__(self, fail_assembly=False):
        self.observed = []
        self.prepared = []
        self.session_starts = []
        self.fail_assembly = fail_assembly
        self.context_length = 200_000
        self.threshold_tokens = 100_000

    @property
    def name(self):
        return "ov-style"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def capabilities(self):
        return ContextEngineCapabilities(observation=True, request_assembly=True)

    def prepare_request_messages(self, messages, ctx):
        self.prepared.append((messages, ctx))
        if self.fail_assembly:
            raise RuntimeError("assembly backend down")
        # Request-only view: archive summary merged into the user message
        # (replacement-style, keeps role alternation valid).
        view = [dict(m) for m in messages]
        for m in view:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = SUMMARY_MARK + "\n\n" + m["content"]
        return view

    def on_turn_complete(self, messages, turn):
        self.observed.append((messages, turn))

    def on_session_start(self, session_id, **kwargs):
        self.session_starts.append((session_id, kwargs))


# ── harness (pattern from test_context_token_tracking.py) ────────────────────


def _patch_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [{
        "type": "function",
        "function": {"name": "t", "description": "t",
                     "parameters": {"type": "object", "properties": {}}},
    }])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _text_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace(
            role="assistant", content="Sure — done.", tool_calls=None,
            reasoning_content=None,
        ), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=500, completion_tokens=20,
                              total_tokens=520),
        model="test-model",
    )


def _make_agent(monkeypatch, engine=None, captured=None):
    _patch_bootstrap(monkeypatch)

    class _A(run_agent.AIAgent):
        def __init__(self, *a, **kw):
            kw.update(skip_context_files=True, skip_memory=True, max_iterations=4)
            super().__init__(*a, **kw)
            self._cleanup_task_resources = self._persist_session = lambda *a, **k: None
            self._save_trajectory = lambda *a, **k: None

        def run_conversation(self, msg, conversation_history=None, task_id=None):
            def _call(kw):
                if captured is not None:
                    captured.append(kw)
                return _text_response()

            self._interruptible_api_call = _call
            self._disable_streaming = True
            return super().run_conversation(
                msg, conversation_history=conversation_history, task_id=task_id
            )

    agent = _A(model="test-model", api_key="test-key",
               base_url="http://localhost:1234/v1",
               provider="openrouter", api_mode="chat_completions")
    if engine is not None:
        agent.context_compressor = engine
        agent._context_engine_caps = engine.capabilities()
    return agent


def _outbound_user_contents(api_kwargs):
    return [m.get("content") for m in api_kwargs["messages"]
            if m.get("role") == "user"]


# ── E2E: request assembly ────────────────────────────────────────────────────


class TestAssemblyE2E:
    def test_provider_sees_assembled_view(self, monkeypatch):
        captured = []
        engine = OVStyleEngine()
        agent = _make_agent(monkeypatch, engine=engine, captured=captured)
        agent.run_conversation("plan the next step")

        assert engine.prepared, "prepare_request_messages was never called"
        assert captured, "provider was never called"
        user_contents = _outbound_user_contents(captured[0])
        assert any(SUMMARY_MARK in (c or "") for c in user_contents), (
            "assembled summary must reach the provider request"
        )

    def test_working_messages_never_see_the_view(self, monkeypatch):
        captured = []
        engine = OVStyleEngine()
        agent = _make_agent(monkeypatch, engine=engine, captured=captured)
        result = agent.run_conversation("plan the next step")

        for m in result["messages"]:
            content = m.get("content") or ""
            assert SUMMARY_MARK not in content, (
                "request-only view leaked into the working message list"
            )

    def test_request_context_fields(self, monkeypatch):
        engine = OVStyleEngine()
        agent = _make_agent(monkeypatch, engine=engine, captured=[])
        agent.run_conversation("hello there")

        messages_arg, ctx = engine.prepared[0]
        assert ctx.model == "test-model"
        assert ctx.provider == "openrouter"
        assert ctx.api_mode == "chat_completions"
        assert ctx.session_id == (agent.session_id or "")
        assert ctx.platform == "cli"
        assert ctx.tools
        assert ctx.budget_tokens == engine.threshold_tokens
        assert ctx.incoming_message is not None
        assert ctx.incoming_message.get("role") == "user"

    def test_assembly_takes_over_plugin_context_injection(self, monkeypatch):
        captured = []
        engine = OVStyleEngine()
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda name, **kwargs: [{"context": "PLUGIN_CONTEXT_MARK"}]
            if name == "pre_llm_call" else [],
        )
        agent = _make_agent(monkeypatch, engine=engine, captured=captured)
        agent.run_conversation("plan the next step")

        assert engine.prepared[0][1].plugin_user_context == "PLUGIN_CONTEXT_MARK"
        user_contents = _outbound_user_contents(captured[0])
        assert all("PLUGIN_CONTEXT_MARK" not in (c or "") for c in user_contents)

    def test_assembly_failure_fails_open(self, monkeypatch):
        captured = []
        engine = OVStyleEngine(fail_assembly=True)
        agent = _make_agent(monkeypatch, engine=engine, captured=captured)
        result = agent.run_conversation("plan the next step")

        assert result["final_response"] == "Sure — done."
        user_contents = _outbound_user_contents(captured[0])
        assert all(SUMMARY_MARK not in (c or "") for c in user_contents)
        # Observation still works after an assembly failure.
        assert len(engine.observed) == 1


# ── E2E: turn observation ────────────────────────────────────────────────────


class TestObservationE2E:
    def test_fires_once_with_finalized_messages(self, monkeypatch):
        engine = OVStyleEngine()
        agent = _make_agent(monkeypatch, engine=engine, captured=[])
        result = agent.run_conversation("hello")

        assert len(engine.observed) == 1
        observed_messages, turn = engine.observed[0]
        assert observed_messages is result["messages"]
        assert any(m.get("role") == "assistant" for m in observed_messages)
        assert turn.completed is True
        assert turn.interrupted is False
        assert turn.compressed_during_turn is False
        assert turn.session_id == (agent.session_id or "")
        assert turn.turn_index == agent._user_turn_count
        # usage flows from update_from_response into TurnInfo
        assert turn.usage["prompt_tokens"] == 500


# ── E2E: legacy path unchanged ───────────────────────────────────────────────


class TestLegacyBaselineE2E:
    def test_builtin_compressor_outbound_equals_input(self, monkeypatch):
        captured = []
        agent = _make_agent(monkeypatch, engine=None, captured=captured)
        agent.run_conversation("plain question")

        user_contents = _outbound_user_contents(captured[0])
        assert "plain question" in (user_contents[-1] or "")
        assert all(SUMMARY_MARK not in (c or "") for c in user_contents)
        # Default capabilities declare nothing.
        caps = agent._context_engine_caps
        assert caps.observation is False and caps.request_assembly is False


# ── E2E: lineage kwargs ──────────────────────────────────────────────────────


# NOTE: lineage E2E tests (boundary_reason / lineage_root_id propagation and
# reset-root regeneration) ship with the follow-up lineage PR alongside the
# code they exercise.
