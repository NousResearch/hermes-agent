"""Coverage for the turn-bound invocation primitive (``ctx.llm`` inheritance, #109499).

A plugin that has to reason about the user's own conversation must reach the model that
conversation is already using — same provider, same model, same route — exactly once, and
must be refused rather than quietly land somewhere else. Every way of getting this wrong
is silent: a call that falls back to the configured auxiliary route still returns a
plausible answer, and nothing in the result says it went to a different provider than the
caller asked for. Hence the contract is pinned here, not left to the happy path.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import agent.auxiliary_client as aux
from agent.plugin_llm import (
    PluginInvocation,
    PluginLlmInvocationError,
    PluginLlmTextInput,
    _TrustPolicy,
    make_plugin_llm_for_test,
)

# One settled invocation: provider + model + the route fields that name where it goes.
ROUTE = dict(provider="custom", model="locked-model", base_url="https://locked.example/v1",
             api_key="turn-secret", api_mode="chat_completions", session_id="s-1")


def _response(text: str = "ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        model=ROUTE["model"],
    )


class _RecordingClient:
    """Counts physical requests, so "exactly once" is an assertion rather than a hope."""

    def __init__(self, *, raises: BaseException | None = None):
        self.calls: list[dict] = []
        self._raises = raises
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer._raises is not None:
                    raise outer._raises
                return _response()

        self.chat = SimpleNamespace(completions=_Completions())
        self.base_url = ROUTE["base_url"]


def _llm(**kwargs):
    return make_plugin_llm_for_test(
        plugin_id="refine", policy=_TrustPolicy(plugin_id="refine"), **kwargs)


@pytest.fixture
def unbound():
    """No turn binding. ``clear_runtime_main`` also resets the legacy compat mirrors."""
    aux.clear_runtime_main()
    yield
    aux.clear_runtime_main()


@pytest.fixture
def bound():
    """A live turn, as the turn prologue publishes it (``_publish_runtime_main``)."""
    token = aux.set_runtime_main(**ROUTE)
    yield
    aux.reset_runtime_main(token)
    aux.clear_runtime_main()


# ── What a plugin can ask for ────────────────────────────────────────────────────


def test_the_turn_binding_is_readable_as_a_value(bound):
    invocation = PluginInvocation.current()
    assert invocation == PluginInvocation(
        provider="custom", model="locked-model", base_url=ROUTE["base_url"],
        api_mode="chat_completions", session_id="s-1")
    assert _llm().current_invocation() == invocation


def test_the_snapshot_carries_no_credentials(bound):
    """The plugin journals this; the host's key must not ride along."""
    invocation = PluginInvocation.current()
    assert "turn-secret" not in repr(invocation)
    assert "turn-secret" not in str(asdict(invocation))
    with pytest.raises(Exception):
        invocation.model = "something-else"  # type: ignore[misc]  # frozen: a value, not a handle


def test_without_a_turn_binding_there_is_nothing_to_inherit(unbound):
    assert PluginInvocation.current() is None
    assert _llm().current_invocation() is None


@pytest.mark.parametrize("provider,model", [("custom", ""), ("", "m"), ("auto", "m")])
def test_a_binding_that_cannot_name_its_route_is_not_an_invocation(unbound, provider, model):
    token = aux.set_runtime_main(provider, model, api_mode="chat_completions")
    try:
        assert PluginInvocation.current() is None
    finally:
        aux.reset_runtime_main(token)
        aux.clear_runtime_main()


def test_a_worker_thread_does_not_inherit_the_turn(bound):
    """Context-local, not global: the same reason a process-global route is unsafe."""
    seen: dict = {}
    thread = threading.Thread(target=lambda: seen.update(current=PluginInvocation.current()))
    thread.start()
    thread.join(timeout=5)
    assert seen["current"] is None


def test_the_turn_prologue_publishes_what_plugins_read(unbound):
    """The value a plugin reads is the one ``build_turn_context`` publishes at turn start."""
    from agent.turn_context import _publish_runtime_main

    _publish_runtime_main(SimpleNamespace(
        provider="openrouter", model="vendor/model", base_url="https://openrouter.ai/api/v1",
        api_key="k", api_mode="chat_completions", auth_mode="", session_id="s-7",
        requested_provider="openrouter",
    ))
    try:
        invocation = PluginInvocation.current()
        assert (invocation.provider, invocation.model, invocation.session_id) == (
            "openrouter", "vendor/model", "s-7")
    finally:
        aux.clear_runtime_main()


def test_two_concurrent_turns_do_not_see_each_others_route():
    seen: dict = {}
    barrier = threading.Barrier(2)

    def _worker(name: str) -> None:
        def _run() -> None:
            token = aux.set_runtime_main(
                "custom", f"{name}-model", base_url=f"https://{name}.example/v1",
                api_mode="chat_completions")
            try:
                barrier.wait(timeout=5)
                seen[name] = PluginInvocation.current().model
            finally:
                aux.reset_runtime_main(token)

        contextvars.copy_context().run(_run)

    threads = [threading.Thread(target=_worker, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    aux.clear_runtime_main()
    assert seen == {"a": "a-model", "b": "b-model"}


# ── Inheritance through ctx.llm ──────────────────────────────────────────────────


def test_ctx_llm_inherits_the_turn_invocation(monkeypatch, bound):
    seen: dict = {}
    monkeypatch.setattr(aux, "call_llm_route_locked",
                        lambda **kw: seen.update(kw) or _response("inherited"))

    result = _llm().complete([{"role": "user", "content": "hi"}],
                             inherit_turn_invocation=True, purpose="probe")

    assert (seen["provider"], seen["model"], seen["base_url"], seen["api_mode"]) == (
        "custom", "locked-model", ROUTE["base_url"], "chat_completions")
    assert "api_key" not in seen  # auth is resolved host-side, never by the plugin
    assert result.text == "inherited"
    assert (result.provider, result.model) == ("custom", "locked-model")
    assert result.audit["inherited_turn_invocation"] is True


def test_inherit_without_a_turn_binding_fails_closed(monkeypatch, unbound):
    """No binding is a refusal, not a licence to use the ambient-config route."""
    calls: list = []
    monkeypatch.setattr(aux, "call_llm_route_locked", lambda **kw: calls.append(kw) or _response())
    monkeypatch.setattr(aux, "call_llm", lambda **kw: calls.append(kw) or _response())

    with pytest.raises(PluginLlmInvocationError) as excinfo:
        _llm().complete([{"role": "user", "content": "hi"}], inherit_turn_invocation=True)
    assert excinfo.value.code == "no_turn_invocation"
    assert calls == []


@pytest.mark.parametrize("override", [
    {"provider": "other"}, {"model": "other-model"}, {"agent_id": "other"},
    {"profile": "work"}, {"task": "compression"},
])
def test_inherit_refuses_an_override(monkeypatch, bound, override):
    """Refused, not ignored: an override contradicts pinning to the turn's own route."""
    monkeypatch.setattr(aux, "call_llm_route_locked",
                        lambda **kw: pytest.fail("an override must be refused before any request"))
    with pytest.raises(PluginLlmInvocationError) as excinfo:
        _llm().complete([{"role": "user", "content": "hi"}], inherit_turn_invocation=True, **override)
    assert excinfo.value.code == "override_conflict"


def test_the_locked_call_goes_out_once_on_the_bound_route(monkeypatch, bound):
    client = _RecordingClient()
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **kw: (client, ROUTE["model"]))
    monkeypatch.setattr(aux, "call_llm",
                        lambda **kw: pytest.fail("an inherited call must not reach the fallback ladder"))

    result = _llm().complete([{"role": "user", "content": "hi"}],
                             inherit_turn_invocation=True, max_tokens=64)

    assert len(client.calls) == 1, "a locked route must issue exactly one request"
    assert client.calls[0]["model"] == "locked-model"
    assert [m["role"] for m in client.calls[0]["messages"]] == ["user"]
    assert "stream" not in client.calls[0], "a locked route goes out as captured"
    assert result.text == "ok"


def test_a_transport_failure_is_not_retried_or_rerouted(monkeypatch, bound):
    client = _RecordingClient(raises=ConnectionError(f"boom {ROUTE['base_url']}?key=turn-secret"))
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **kw: (client, ROUTE["model"]))

    with pytest.raises(PluginLlmInvocationError) as excinfo:
        _llm().complete([{"role": "user", "content": "hi"}], inherit_turn_invocation=True)
    assert len(client.calls) == 1, "the locked route retried or fell back"
    assert excinfo.value.code == "transport_error"
    assert "turn-secret" not in str(excinfo.value)  # the code is journaled, the message is not


def test_structured_inheritance_uses_the_same_locked_route(monkeypatch, bound):
    client = _RecordingClient()
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **kw: (client, ROUTE["model"]))

    result = _llm().complete_structured(
        instructions="say ok", input=[PluginLlmTextInput(text="payload")],
        inherit_turn_invocation=True)

    assert len(client.calls) == 1
    assert result.text == "ok"
    assert (result.provider, result.model) == ("custom", "locked-model")


def test_the_async_lane_holds_the_same_contract(monkeypatch, bound):
    client = _RecordingClient()
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **kw: (client, ROUTE["model"]))

    result = asyncio.run(_llm().acomplete([{"role": "user", "content": "hi"}],
                                          inherit_turn_invocation=True))

    assert len(client.calls) == 1
    assert result.text == "ok"
    assert result.audit["inherited_turn_invocation"] is True


def test_the_ordinary_lane_is_untouched(unbound):
    """Inheritance is opt-in: the default call still goes through the host's own path."""
    llm = _llm(sync_caller=lambda **kw: ("openai", "gpt-5", _response("plain")))
    result = llm.complete([{"role": "user", "content": "hi"}])
    assert result.text == "plain"
    assert result.audit["inherited_turn_invocation"] is False


# ── The locked entry point itself ────────────────────────────────────────────────


def test_a_locked_call_with_no_binding_fails_closed(unbound):
    with pytest.raises(aux.RouteLockedCallError) as excinfo:
        aux.call_llm_route_locked(messages=[{"role": "user", "content": "hi"}])
    assert excinfo.value.code == "no_turn_invocation"


def test_a_route_that_cannot_name_its_endpoint_is_refused(monkeypatch, unbound):
    """A ``custom`` route with no base URL would resolve from OPENAI_BASE_URL instead."""
    monkeypatch.setattr(aux, "resolve_provider_client",
                        lambda *a, **kw: pytest.fail("an incomplete route must not be resolved"))
    token = aux.set_runtime_main("custom", "m", api_mode="chat_completions")
    try:
        with pytest.raises(aux.RouteLockedCallError) as excinfo:
            aux.call_llm_route_locked(messages=[{"role": "user", "content": "hi"}])
        assert excinfo.value.code == "incomplete_route"
    finally:
        aux.reset_runtime_main(token)
        aux.clear_runtime_main()


def test_a_locked_call_with_no_messages_is_refused(bound):
    with pytest.raises(aux.RouteLockedCallError) as excinfo:
        aux.call_llm_route_locked(messages=[])
    assert excinfo.value.code == "empty_request"


def test_error_codes_are_coarse_and_leak_free():
    assert aux.classify_route_locked_error(ConnectionError("https://x.example/v1?key=abc")) == "transport_error"
    assert aux.classify_route_locked_error(RuntimeError("https://x.example/v1?key=abc")) == "call_failed"
    assert "abc" not in aux.classify_route_locked_error(RuntimeError("https://x.example/v1?key=abc"))
