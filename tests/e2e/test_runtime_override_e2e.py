"""E2E tests: pre_llm_call runtime_override through the REAL conversation loop.

These drive ``AIAgent.run_conversation`` end-to-end against in-process fake
wire clients (no network), with hooks registered on the real plugin manager.
They prove the model-only override contract (issue #23739) on the production
path:

1. P1-1 — the overridden model is authoritative for the wire request built for
   the turn, and the pre-override identity is restored afterwards.
2. Request-assembly parity — the override is applied BEFORE request assembly /
   preflight, so the ``llm_request`` middleware and the ``pre_api_request``
   hook observe the same overridden model as the wire request.
3. Turn scoping — an override returned for turn 1 does not leak into turn 2:
   the second turn's wire request returns to the base model.
4. Fallback handoff — when the overridden route fails and the fallback chain
   activates, ``consume_runtime_override`` consumes the failed override, so no
   later retry re-issues a request for the overridden model.

Issue: #23739.  The override no longer accepts endpoint/credential keys, so
the route override here changes only the model.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── fake wire clients ───────────────────────────────────────────────────────


class _SimulatedRateLimit(Exception):
    """A 429-shaped wire failure the recovery path classifies as rate_limit."""

    status_code = 429


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
    """Records every fake client construction and wire request.

    ``fail_models``: models whose wire request raises ``_SimulatedRateLimit``
    (used to force the fallback chain onto the remaining route).
    """

    def __init__(self, fail_models=()) -> None:
        self.fail_models = set(fail_models)
        self.openai_clients: list = []      # kwargs per OpenAI client construction
        self.openai_requests: list = []     # kwargs per chat.completions.create

    def make_openai(self, **kwargs):
        self.openai_clients.append(kwargs)
        return _FakeChatClient(self.openai_requests, fail_models=self.fail_models)

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
    def __init__(self, sink, fail_models=()) -> None:
        def _handle(kwargs):
            if kwargs.get("model") in set(fail_models):
                raise _SimulatedRateLimit(
                    f"simulated rate limit for {kwargs.get('model')}"
                )
            return _chat_response()

        self.chat = SimpleNamespace(
            completions=_FakeCompletions(_handle, sink)
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


def _register_plugin_callback(manager, kind, name, callback):
    """Register ``callback`` (hook or middleware) through the real PluginContext
    path (bundled => trusted => the callback survives the trust gate)."""
    from hermes_cli.plugins import PluginContext

    ctx = object.__new__(PluginContext)
    ctx.manifest = _BundledManifest()
    ctx._manager = manager
    if kind == "hook":
        return ctx.register_hook(name, callback)
    if kind == "middleware":
        return ctx.register_middleware(name, callback)
    raise ValueError(f"unknown plugin registration kind: {kind!r}")


def _register_pre_llm_call_hook(manager, callback):
    """Register a ``pre_llm_call`` callback through the real PluginContext path
    (bundled => trusted => runtime_override survives the trust gate)."""
    return _register_plugin_callback(manager, "hook", "pre_llm_call", callback)


def _register_llm_request_middleware(manager, callback):
    """Register an ``llm_request`` middleware through the real PluginContext path."""
    return _register_plugin_callback(manager, "middleware", "llm_request", callback)


def _register_pre_api_request_hook(manager, callback):
    """Register a ``pre_api_request`` hook through the real PluginContext path."""
    return _register_plugin_callback(manager, "hook", "pre_api_request", callback)


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


# ── P1-1: the overridden model is authoritative for the wire ───────────────


def test_override_model_is_authoritative(monkeypatch):
    """The wire request built for the turn uses the overridden model, and the
    pre-override identity is restored after the turn."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    recorder = _WireRecorder()

    def pre_llm_call(**kw):
        return {"runtime_override": {"model": "override-model"}}

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
        # P1-1: the turn-scoped projection left no model-owned state behind — the
        # session compressor still describes the base route (no context-length
        # leak from the override).
        assert agent.context_compressor.model == "base-model"
    finally:
        handle.dispose()


# ── request-assembly parity: middleware + pre_api_request see the override ──


def test_middleware_and_pre_api_request_observe_override_model(monkeypatch):
    """The override is applied before request assembly/preflight, so the
    ``llm_request`` middleware and the ``pre_api_request`` hook record the same
    overridden model the wire request carries."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    recorder = _WireRecorder()
    observed = {"middleware": [], "pre_api_request": []}

    def pre_llm_call(**kw):
        return {"runtime_override": {"model": "override-model"}}

    def llm_request_middleware(**kw):
        observed["middleware"].append(kw.get("model"))

    def pre_api_request(**kw):
        observed["pre_api_request"].append(kw.get("model"))

    handles = [
        _register_pre_llm_call_hook(manager, pre_llm_call),
        _register_llm_request_middleware(manager, llm_request_middleware),
        _register_pre_api_request_hook(manager, pre_api_request),
    ]
    try:
        agent = _build_agent(monkeypatch, recorder)
        result = _run_turn(agent)

        assert "ok" in (result.get("final_response") or "")
        assert recorder.openai_requests, "no OpenAI-wire request recorded"
        assert recorder.openai_requests[0].get("model") == "override-model"

        assert observed["middleware"], "llm_request middleware never fired"
        assert all(m == "override-model" for m in observed["middleware"]), (
            f"middleware observed non-override model(s): {observed['middleware']}"
        )
        assert observed["pre_api_request"], "pre_api_request hook never fired"
        assert all(m == "override-model" for m in observed["pre_api_request"]), (
            f"pre_api_request observed non-override model(s): "
            f"{observed['pre_api_request']}"
        )
    finally:
        for handle in handles:
            handle.dispose()


# ── turn scoping: turn 2 restores the base route ────────────────────────────


def test_override_is_restored_on_next_turn(monkeypatch):
    """An override returned for turn 1 does not leak into turn 2: the second
    turn's wire request uses the base model again."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    recorder = _WireRecorder()
    state = {"calls": 0}

    def pre_llm_call(**kw):
        state["calls"] += 1
        if state["calls"] == 1:
            return {"runtime_override": {"model": "override-model"}}
        return {}

    handle = _register_pre_llm_call_hook(manager, pre_llm_call)
    try:
        agent = _build_agent(monkeypatch, recorder)
        first = _run_turn(agent)
        second = _run_turn(agent)

        assert "ok" in (first.get("final_response") or "")
        assert "ok" in (second.get("final_response") or "")
        assert [r.get("model") for r in recorder.openai_requests] == [
            "override-model", "base-model",
        ]
    finally:
        handle.dispose()


# ── fallback handoff: a failed override is consumed, never re-applied ───────


def test_failed_override_falls_back_and_is_not_reapplied(monkeypatch):
    """A wire failure on the overridden route activates the fallback chain; the
    failed override is consumed (``consume_runtime_override``), so no later
    retry re-issues a request for the overridden model."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    recorder = _WireRecorder(fail_models={"override-model"})

    def pre_llm_call(**kw):
        return {"runtime_override": {"model": "override-model"}}

    handle = _register_pre_llm_call_hook(manager, pre_llm_call)
    try:
        agent = _build_agent(monkeypatch, recorder)
        # The overridden route is the primary; the fallback route must stay on
        # the chat-completions wire so its per-request client is rebuilt through
        # the recorder (process_bootstrap.OpenAI is patched). api_mode is pinned
        # because the fallback's re-detection would otherwise flip the stub's
        # api.openai.com base_url to codex_responses, and the base_url host must
        # stay a known provider endpoint so context-length resolution does not
        # probe a live /models route.
        agent._fallback_chain = [
            {"provider": "openai", "model": "fallback-model", "api_mode": "chat_completions"}
        ]
        agent._fallback_index = 0
        # Provider/credential resolution is the one remaining network boundary
        # on the fallback path; swap in a stub whose wire requests still land on
        # the fake recorder.
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client",
            lambda provider, model=None, **kwargs: (
                SimpleNamespace(api_key="test-key", base_url="https://api.openai.com/v1"),
                model,
            ),
        )

        result = _run_turn(agent)

        assert "ok" in (result.get("final_response") or "")
        models = [r.get("model") for r in recorder.openai_requests]
        assert models, "no OpenAI-wire request recorded"
        assert models[0] == "override-model"
        # The retry built after the override's wire failure goes to the fallback
        # route — the failed override must never re-enter the wire afterwards
        # (consume_runtime_override supersedes the open scope when the fallback
        # activates, so no later request is rebuilt for the overridden model).
        assert models[1] == "fallback-model", (
            f"retry not on the fallback route; wire models: {models}"
        )
        assert "override-model" not in models[1:], (
            f"failed override re-applied after the fallback activated: {models}"
        )
        # P1-1 supersession: the fallback-owned compressor state survives the
        # override scope's exit (the pre-override route is NOT restored over the
        # freshly activated fallback).
        assert agent.context_compressor.model == "fallback-model"
    finally:
        handle.dispose()
