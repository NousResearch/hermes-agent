"""ACP session switch transaction tests and independent dashboard validation.

ACP and dashboard now resolve through distinct application coordinators
over the same canonical model and provider domains. These transaction tests
inject ACP resolution outcomes independently; dashboard unknown-provider
rejection runs through the actual dashboard selection application.
"""

from __future__ import annotations

import types

import pytest



def _patch_acp_switch(monkeypatch, legacy_factory):
    """Keep transaction tests focused on ACP ownership, not live validation I/O."""
    from acp_adapter.model_switch_resolution import AcpModelRoute

    def resolve(**kwargs):
        result = legacy_factory(
            raw_input=kwargs["raw_model"], explicit_provider="",
            current_provider=kwargs["current_provider"], current_model=kwargs["current_model"],
        )
        if not result.success:
            raise ValueError(result.error_message or "Invalid model")
        base = kwargs["current_base_url"] if kwargs.get("keep_endpoint") else "https://api.anthropic.com"
        return AcpModelRoute(
            model=result.new_model, provider=result.target_provider,
            base_url=base, api_mode="anthropic_messages",
            runtime={"provider": result.target_provider, "base_url": base,
                     "api_mode": "anthropic_messages", "api_key": "fixture-key"},
        )

    monkeypatch.setattr("acp_adapter.model_switch_resolution.resolve_acp_model_switch", resolve)


def _acp_agent():
    from acp_adapter.server import HermesACPAgent
    made: dict = {}

    class _SM:
        def _make_agent(self, **kw):
            made.update(kw)
            return types.SimpleNamespace(provider=kw.get("requested_provider"), model=kw.get("model"))

        def save_session(self, sid, *, strict=False):
            assert strict

    return HermesACPAgent(session_manager=_SM()), made


def _state(**agent_attrs):
    import threading
    return types.SimpleNamespace(
        session_id="s1", cwd=".", model="claude-sonnet-5",
        is_running=False, command_op=False, queued_prompts=[], runtime_lock=threading.Lock(),
        agent=types.SimpleNamespace(
            provider="anthropic", base_url="https://api.anthropic.com", api_key="k", **agent_attrs))


def test_acp_and_dashboard_reject_what_switch_model_rejects(monkeypatch):
    rejected = types.SimpleNamespace(success=False, error_message="Unknown provider 'notaprovider'.")
    _patch_acp_switch(monkeypatch, lambda **_kw: rejected)

    agent, made = _acp_agent()
    state = _state()
    with pytest.raises(ValueError, match="Unknown provider"):
        agent._switch_model(state, "notaprovider:whatever")
    assert made == {} and state.model == "claude-sonnet-5"  # session untouched

    from fastapi import HTTPException
    from hermes_cli.web_server_config import _apply_model_assignment_sync
    with pytest.raises(HTTPException) as exc:
        _apply_model_assignment_sync("main", "notaprovider", "whatever", "", "")
    assert exc.value.status_code == 400 and "Unknown provider" in exc.value.detail


def test_acp_forwards_qualified_choice_to_acp_resolution(monkeypatch):
    seen: dict = {}

    def _switch(**kw):
        seen.update(kw)
        return types.SimpleNamespace(
            success=True,
            new_model="claude-sonnet-5",
            target_provider="anthropic",
        )

    _patch_acp_switch(monkeypatch, _switch)
    agent, made = _acp_agent()
    old, new_provider, model = agent._switch_model(
        _state(), "anthropic:claude-sonnet-5", keep_endpoint=True
    )
    assert (seen["explicit_provider"], seen["raw_input"]) == (
        "",
        "anthropic:claude-sonnet-5",
    )
    assert (old, new_provider, model) == ("anthropic", "anthropic", "claude-sonnet-5")
    assert made["requested_provider"] == "anthropic" and made["base_url"] == "https://api.anthropic.com"


def test_acp_set_session_model_runs_switch_model_off_the_event_loop(monkeypatch):
    """``switch_model`` does ~10 s of sync network I/O on a cold cache; ACP must run it on a
    worker thread (like the gateway) or every session in the process stalls."""
    import asyncio
    import threading

    seen: dict = {}

    def _switch(**kw):
        seen["thread"] = threading.current_thread()
        return types.SimpleNamespace(
            success=True, new_model="claude-sonnet-5", target_provider="anthropic"
        )

    _patch_acp_switch(monkeypatch, _switch)
    agent, _made = _acp_agent()
    state = _state()
    agent.session_manager.get_session = lambda sid: state

    async def _run():
        loop_thread = threading.current_thread()
        resp = await agent.set_session_model("anthropic:claude-sonnet-5", "s1")
        return resp, loop_thread

    resp, loop_thread = asyncio.run(_run())
    assert resp is not None and state.model == "claude-sonnet-5"
    assert seen["thread"] is not loop_thread


def test_acp_set_session_model_rejected_while_turn_running(monkeypatch):
    """The picker swaps state.agent wholesale; mid-turn that strands the running agent and
    makes _finish_turn emit a spurious compression-rotation update."""
    import acp
    import asyncio

    called = {}
    _patch_acp_switch(monkeypatch,
        lambda **kw: called.setdefault("hit", kw))
    agent, made = _acp_agent()
    state = _state()
    state.is_running = True
    agent.session_manager.get_session = lambda sid: state

    with pytest.raises(acp.RequestError):
        asyncio.run(agent.set_session_model("anthropic:claude-sonnet-5", "s1"))

    assert called == {} and made == {}  # no resolution, no rebuild
    assert state.model == "claude-sonnet-5"
    assert state.command_op is False


def test_acp_switch_model_carries_the_live_agent_toolsets_into_the_rebuild(monkeypatch):
    """Regression for #42719: ACP-provided MCP servers live only on the running agent's toolsets
    (``_register_session_mcp_servers``); a rebuild that re-derived them from config.yaml dropped
    every session MCP tool after ``session/set_model`` or ``/model``."""
    _patch_acp_switch(monkeypatch,
        lambda **_kw: types.SimpleNamespace(success=True, target_provider="anthropic", new_model="claude-sonnet-5"))

    agent, made = _acp_agent()
    agent._switch_model(_state(enabled_toolsets=["hermes-acp", "mcp-demo-search"], disabled_toolsets=["browser"]),
                        "claude-sonnet-5")

    assert made["enabled_toolsets"] == ["hermes-acp", "mcp-demo-search"]
    assert made["disabled_toolsets"] == ["browser"]


def test_acp_set_session_model_rejection_is_invalid_params_and_leaves_session_untouched(monkeypatch):
    """#72439: a ``modelId`` no provider can serve is a bad param (-32602 with the switch_model
    reason), not a -32603 internal error; and a rebuild that blows up after switch_model accepted
    the model must not leave ``state.model`` pointing at a model the live agent does not run."""
    import asyncio

    from acp.exceptions import RequestError

    _patch_acp_switch(monkeypatch,
                        lambda **_kw: types.SimpleNamespace(success=False, error_message="`nope` is not a model"))
    agent, _made = _acp_agent()
    state = _state()
    agent.session_manager.get_session = lambda sid: state
    with pytest.raises(RequestError) as exc:
        asyncio.run(agent.set_session_model("nope", "s1"))
    assert exc.value.code == -32602 and exc.value.data == {"details": "`nope` is not a model"}

    _patch_acp_switch(monkeypatch,
                        lambda **_kw: types.SimpleNamespace(success=True, new_model="other", target_provider="anthropic"))

    def _boom(**_kw):
        raise RuntimeError("No Codex credentials stored")

    agent.session_manager._make_agent = _boom
    old_agent = state.agent
    with pytest.raises(RuntimeError, match="No Codex credentials"):
        agent._switch_model(state, "other")
    assert state.model == "claude-sonnet-5" and state.agent is old_agent

    # A ValueError raised by the rebuild itself (disabled provider, context floor) is not a bad
    # ``modelId``: it must escape as-is so acp maps it to -32603, not be relabelled -32602.
    def _rebuild_value_error(**_kw):
        raise ValueError("provider 'anthropic' is disabled in config")

    agent.session_manager._make_agent = _rebuild_value_error
    with pytest.raises(ValueError, match="disabled in config") as rebuild_exc:
        asyncio.run(agent.set_session_model("other", "s1"))
    assert not isinstance(rebuild_exc.value, RequestError)
    assert state.model == "claude-sonnet-5" and state.agent is old_agent


def test_acp_set_session_model_does_not_run_queued_prompts_inside_the_rpc(monkeypatch):
    """A prompt that arrives while ``switch_model`` is off-loop is queued behind ``command_op``;
    it must run AFTER the set_model response (error or success) is queued, never inside the RPC —
    otherwise a failed switch is reported only after a whole agent turn."""
    import asyncio

    from acp.exceptions import RequestError

    _patch_acp_switch(monkeypatch,
                        lambda **_kw: types.SimpleNamespace(success=False, error_message="`nope` is not a model"))
    agent, _made = _acp_agent()
    state = _state()
    state.queued_prompts = ["hello, queued mid-switch"]
    agent.session_manager.get_session = lambda sid: state

    class _Conn:
        async def session_update(self, *_a, **_k):
            pass

    agent._conn = _Conn()
    ran: list = []

    async def _prompt(*, prompt, session_id):
        ran.append(prompt[0].text)

    agent.prompt = _prompt

    async def _run():
        with pytest.raises(RequestError):
            await agent.set_session_model("nope", "s1")
        assert ran == [], "the queued prompt ran inside the set_model RPC"
        await asyncio.sleep(0)  # let the scheduled drain run once the response is out
        await asyncio.sleep(0)
        return list(ran)

    assert asyncio.run(_run()) == ["hello, queued mid-switch"]
    assert state.command_op is False