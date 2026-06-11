"""Loop-path coverage for ContextEngine lifecycle hooks (test plan A2/A7/A8).

Drives the REAL run_conversation with scripted provider responses:
- A2: tool loop — request assembly runs on EVERY dispatch (idempotent engine)
- A7: anthropic_messages api_mode shares the outbound build (assembly works)
- A8: codex_app_server hands the turn to an external runtime — hooks must
      not fire (documented limitation, asserted here)
"""

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.context_engine import ContextEngine, ContextEngineCapabilities

MARK = "[CTX-VIEW]"


class AssemblingEngine(ContextEngine):
    """Minimal observation+assembly engine that tags the outbound view."""

    def __init__(self):
        self.prepare_calls = 0
        self.turns = []

    @property
    def name(self):
        return "assembling-test"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def capabilities(self):
        return ContextEngineCapabilities(observation=True, request_assembly=True)

    def prepare_request_messages(self, messages, ctx):
        self.prepare_calls += 1
        view = [dict(m) for m in messages]
        for m in view:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = MARK + " " + m["content"]
        return view

    def on_turn_complete(self, messages, turn):
        self.turns.append(turn)


def _patch_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [{
        "type": "function",
        "function": {"name": "t", "description": "t",
                     "parameters": {"type": "object", "properties": {}}},
    }])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _text_resp(text="done"):
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace(
            role="assistant", content=text, tool_calls=None,
            reasoning_content=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5,
                              total_tokens=105),
        model="test-model",
    )


def _tool_resp():
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace(
            role="assistant", content=None,
            tool_calls=[SimpleNamespace(
                id="call-1", type="function",
                function=SimpleNamespace(name="t", arguments="{}"))],
            reasoning_content=None), finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5,
                              total_tokens=105),
        model="test-model",
    )


def _anthropic_text_resp(text="done"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=100, output_tokens=5),
        model="claude-test",
    )


class _FakeAnthropicClient:
    def close(self):
        pass


def _make_agent(monkeypatch, responses, engine, api_mode="chat_completions",
                provider="openrouter", captured=None):
    _patch_bootstrap(monkeypatch)
    if api_mode == "anthropic_messages":
        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client",
            lambda k, b=None, **kw: _FakeAnthropicClient(),
        )
    queue = list(responses)

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
                return queue.pop(0) if len(queue) > 1 else queue[0]

            self._interruptible_api_call = _call
            self._disable_streaming = True
            return super().run_conversation(
                msg, conversation_history=conversation_history, task_id=task_id
            )

    agent = _A(model="test-model", api_key="test-key",
               base_url="http://localhost:1234/v1",
               provider=provider, api_mode=api_mode)
    agent.context_compressor = engine
    agent._context_engine_caps = engine.capabilities()
    return agent


# ── A2: tool loop — assembly on every dispatch ───────────────────────────────


class TestToolLoopAssembly:
    def test_assembly_runs_per_dispatch(self, monkeypatch):
        engine = AssemblingEngine()
        captured = []
        agent = _make_agent(
            monkeypatch, [_tool_resp(), _text_resp("after tool")],
            engine, captured=captured,
        )
        result = agent.run_conversation("use the tool then answer")

        assert result["final_response"] == "after tool"
        # Two provider dispatches -> two assembly invocations.
        assert engine.prepare_calls == len(captured) == 2
        # The engine view reached the provider on BOTH dispatches.
        for kw in captured:
            user_contents = [m.get("content") for m in kw["messages"]
                             if m.get("role") == "user"]
            assert any(MARK in (c or "") for c in user_contents)
        # Observation still fires exactly once at turn end.
        assert len(engine.turns) == 1

    def test_working_messages_clean_after_tool_loop(self, monkeypatch):
        engine = AssemblingEngine()
        agent = _make_agent(
            monkeypatch, [_tool_resp(), _text_resp()], engine,
        )
        result = agent.run_conversation("go")
        for m in result["messages"]:
            assert MARK not in str(m.get("content") or "")


# ── A7: anthropic_messages shares the outbound build ─────────────────────────


class TestAnthropicAssembly:
    def test_assembly_applies_on_anthropic_path(self, monkeypatch):
        engine = AssemblingEngine()
        captured = []
        agent = _make_agent(
            monkeypatch, [_anthropic_text_resp("ok")], engine,
            api_mode="anthropic_messages", provider="anthropic",
            captured=captured,
        )
        result = agent.run_conversation("hello")
        assert result["final_response"] == "ok"
        assert engine.prepare_calls >= 1
        assert len(engine.turns) == 1
        # The view must reach the anthropic request payload.
        flat = str(captured[0])
        assert MARK in flat

    def test_anthropic_working_messages_clean(self, monkeypatch):
        engine = AssemblingEngine()
        agent = _make_agent(
            monkeypatch, [_anthropic_text_resp()], engine,
            api_mode="anthropic_messages", provider="anthropic",
        )
        result = agent.run_conversation("hello")
        for m in result["messages"]:
            assert MARK not in str(m.get("content") or "")


# ── A8: codex_app_server bypasses the loop entirely ──────────────────────────


class TestCodexAppServerLimitation:
    def test_hooks_do_not_fire(self, monkeypatch):
        engine = AssemblingEngine()
        agent = _make_agent(monkeypatch, [_text_resp()], engine)
        agent.api_mode = "codex_app_server"
        agent._run_codex_app_server_turn = lambda **kw: {
            "final_response": "external runtime", "messages": [],
            "api_calls": 1, "completed": True,
        }
        result = agent.run_conversation("hi")
        assert result["final_response"] == "external runtime"
        # Documented limitation: the external runtime owns the turn —
        # no assembly, no observation.
        assert engine.prepare_calls == 0
        assert engine.turns == []
