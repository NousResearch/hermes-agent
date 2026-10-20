"""`select_turn_model`: a plugin can move one turn to another model, through the /model resolve+swap path."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.turn_context import _apply_turn_model_selection


def _agent(**kw):
    return SimpleNamespace(session_id="s1", model="claude-opus-5-5", provider="anthropic", platform="discord",
                           base_url="https://api.anthropic.com", api_key="k", _parent_session_id=None,
                           _user_id="u1", _turn_origin=None, _persist_disabled=False, switch_model=Mock(), **kw)


def _resolved(**kw):
    return SimpleNamespace(success=True, new_model="gpt-6.1-sol", target_provider="openai-codex", api_key="tok",
                           base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses",
                           runtime_capabilities={"tools": True}, error_message="", **kw)


def _run(agent, hook_results, resolved=None):
    with patch("hermes_cli.lifecycle.invoke_hook", return_value=hook_results) as hook, \
         patch("hermes_cli.model_switch.switch_model", return_value=resolved or _resolved()) as resolve:
        _apply_turn_model_selection(agent, turn_id="t1", original_user_message="refactor the parser",
                                    conversation_history=[{"role": "user", "content": "hi"}])
    return hook, resolve


def test_a_plugin_choice_is_resolved_like_slash_model_and_swapped_in():
    agent = _agent()
    hook, resolve = _run(agent, [None, {"provider": "openai-codex", "model": "gpt-6.1-sol", "reason": "coding"}])
    assert hook.call_args.args[0] == "select_turn_model"
    assert hook.call_args.kwargs["provider"] == "anthropic" and hook.call_args.kwargs["is_first_turn"] is False
    assert resolve.call_args.kwargs["explicit_provider"] == "openai-codex"
    assert resolve.call_args.kwargs["raw_input"] == "gpt-6.1-sol"
    agent.switch_model.assert_called_once_with(
        new_model="gpt-6.1-sol", new_provider="openai-codex", api_key="tok",
        base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses", capabilities={"tools": True})


def test_no_switch_for_same_model_invalid_results_or_fork_turns():
    agent = _agent()
    _run(agent, [{"provider": "anthropic", "model": "claude-opus-5-5"}])            # already there
    _run(agent, ["gpt-6.1-sol", {"model": "x"}, {"provider": "", "model": "y"}])    # not a valid choice
    agent.switch_model.assert_not_called()
    fork = _agent(); fork._turn_origin = "background_review"
    hook, _ = _run(fork, [{"provider": "openai-codex", "model": "gpt-6.1-sol"}])
    hook.assert_not_called()
    fork.switch_model.assert_not_called()


def test_fail_open_on_rejected_route_failed_swap_or_hook_error():
    agent = _agent()
    _run(agent, [{"provider": "openai-codex", "model": "nope"}], resolved=SimpleNamespace(success=False, error_message="unknown model"))
    agent.switch_model.assert_not_called()
    agent.switch_model.side_effect = RuntimeError("rebuild failed")
    _run(agent, [{"provider": "openai-codex", "model": "gpt-6.1-sol"}])            # must not raise
    with patch("hermes_cli.lifecycle.invoke_hook", side_effect=RuntimeError("plugin crashed")):
        _apply_turn_model_selection(agent, turn_id="t2", original_user_message="x", conversation_history=None)


def test_the_switch_lands_before_the_turns_system_prompt_is_built(tmp_path):
    """Through the real build_turn_context: the swap must precede the prompt build, so the turn is
    prepared for the model that serves it (switch_model clears the cached prompt)."""
    import types

    from agent.turn_context import build_turn_context
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="s-route", source="cli", model="test-model")
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, session_db=db, session_id="s-route")
    agent._session_db_created = True
    agent._cached_system_prompt = "SYSTEM"
    agent._skip_mcp_refresh = True
    events = []

    def fake_switch(**kw):
        events.append(("switch", kw["new_model"]))
        agent._cached_system_prompt = None        # what the real switch_model does

    def build_prompt(a, *_args, **_kw):
        events.append(("build_prompt", a.model))
        a._cached_system_prompt = "SYSTEM v2"

    agent.switch_model = fake_switch
    hook = lambda name, **kw: [{"provider": "openai-codex", "model": "gpt-6.1-sol"}] if name == "select_turn_model" else []
    try:
        with patch("hermes_cli.lifecycle.invoke_hook", side_effect=hook), \
             patch("hermes_cli.model_switch.switch_model", return_value=_resolved()), \
             patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
            build_turn_context(
                agent=agent, user_message="refactor the parser", system_message=None, conversation_history=None,
                task_id=None, stream_callback=None, persist_user_message=None,
                restore_or_build_system_prompt=build_prompt, install_safe_stdio=lambda: None,
                sanitize_surrogates=lambda s: s, summarize_user_message_for_log=lambda s: s,
                set_session_context=lambda _sid: None, set_current_write_origin=lambda _o: None,
                ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None))
    finally:
        db.close()
    assert [e[0] for e in events] == ["switch", "build_prompt"]
