"""E2E tests: pre_llm_call runtime_override through the REAL conversation loop.

These drive ``AIAgent.run_conversation`` end-to-end against in-process fake
wire clients (no network), with hooks registered on the real plugin manager.
They prove the model/provider/api_mode override contract (issue #23739) on the
production path:

1. P1-1 — the overridden route is authoritative for the wire request built for
   the turn, and the pre-override identity is restored afterwards.

Issue: #23739.  The override no longer accepts endpoint/credential keys, so
the route override here changes only model/provider/api_mode.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── fake wire clients ───────────────────────────────────────────────────────


def _chat_response(content: str = "ok"):
    """A valid chat.completions response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=[], reasoning=None),
            finish_reason="stop",
        )],
        usage=None,
    )


class _WireRecorder:
    """Records every fake client construction and wire request."""

    def __init__(self) -> None:
        self.openai_clients: list = []      # kwargs per OpenAI client construction
        self.openai_requests: list = []     # kwargs per chat.completions.create

    def make_openai(self, **kwargs):
        self.openai_clients.append(kwargs)
        return _FakeChatClient(self.openai_requests)

    def make_anthropic(self, *args, **kwargs):
        return _FakeAnthropicClient()


class _FakeCompletions:
    def __init__(self, handler, sink) -> None:
        self._handler = handler
        self._sink = sink

    def create(self, **kwargs):
        self._sink.append(kwargs)
        return self._handler(kwargs)


class _FakeChatClient:
    def __init__(self, sink) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(lambda k: _chat_response(), sink)
        )

    def close(self) -> None:
        pass


class _FakeAnthropicClient:
    def close(self) -> None:
        pass


# ── plugin registration on the real manager ─────────────────────────────────


class _BundledManifest:
    name = "e2e-runtime-override"
    key = "e2e-runtime-override"
    source = "bundled"


def _register_pre_llm_call_hook(manager, callback):
    """Register a ``pre_llm_call`` callback through the real PluginContext path
    (bundled => trusted => runtime_override survives the trust gate)."""
    from hermes_cli.plugins import PluginContext

    ctx = object.__new__(PluginContext)
    ctx.manifest = _BundledManifest()
    ctx._manager = manager
    return ctx.register_hook("pre_llm_call", callback)


# ── agent + turn helpers ────────────────────────────────────────────────────


def _build_agent(monkeypatch, recorder):
    """Build a real AIAgent whose wire traffic lands on the fake clients."""
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", recorder.make_openai)
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client", recorder.make_anthropic
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    # The agent is built without explicit credentials; feed the router an empty
    # client-kwargs dict so init skips the real provider/credential resolution
    # and constructs the (fake) OpenAI-wire client with no network lookup.
    monkeypatch.setattr(
        "agent.agent_init._routed_client_kwargs",
        lambda agent, fallback_model, timeout: {},
    )

    from run_agent import AIAgent

    agent = AIAgent(
        model="base-model",
        provider="openai",
        platform="cli",
        max_iterations=3,
        quiet_mode=True,
        skip_memory=True,
    )
    agent._disable_streaming = True
    return agent


def _run_turn(agent):
    """One real ``run_conversation`` turn."""
    return agent.run_conversation("hello")


# ── P1-1: the overridden route is authoritative for the wire ────────────────


def test_override_model_and_provider_are_authoritative(monkeypatch):
    """The wire request built for the turn uses the overridden model, and the
    pre-override identity is restored after the turn."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    recorder = _WireRecorder()

    def pre_llm_call(**kw):
        return {"runtime_override": {
            "model": "override-model",
            "provider": "openai",
            "api_mode": "chat_completions",
        }}

    handle = _register_pre_llm_call_hook(manager, pre_llm_call)
    try:
        agent = _build_agent(monkeypatch, recorder)
        pre = (agent.model, agent.provider, agent.api_mode)
        result = _run_turn(agent)

        assert "ok" in (result.get("final_response") or "")

        # The wire was built from the overridden route.
        assert recorder.openai_requests, "no OpenAI-wire request recorded"
        wire_req = recorder.openai_requests[0]
        assert wire_req.get("model") == "override-model"

        # The turn restored the pre-override identity.
        assert (agent.model, agent.provider, agent.api_mode) == pre
    finally:
        handle.dispose()
