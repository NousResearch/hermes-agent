"""``/model --once`` is one turn; a session ``/model`` is the session's route.

``--once`` swaps the agent and restores it after one turn, and never writes the session
row. A session switch writes that row through the owner's fenced ``session.mutate model``.
Switching onto the route the session already runs commits nothing.
"""
from types import SimpleNamespace

import pytest

from hermes_cli.model_switch import ModelSwitchResult
from tests.hermes_cli.test_cli_gateway_model_route import RecordingConnection, _agent, _model_routes

_CONFIRM_MESSAGE = "Large context: this switch re-reads the whole session."


class GuardedConnection(RecordingConnection):
    """The owner's selection guard: the first model mutate answers ``confirmation_required``."""

    def call(self, method, *, timeout=None, **params):
        if method == "session.mutate" and params.get("operation") == "model" \
                and params["payload"].get("confirm") != "tok":
            self.calls.append((method, params))
            return {"status": "confirmation_required", "confirm_required": True, "confirm": "tok",
                    "confirm_message": _CONFIRM_MESSAGE}
        return super().call(method, timeout=timeout, **params)


def _cli(monkeypatch, conn):
    import cli as cli_mod
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from tests.hermes_cli.test_cli_model_once import _StubCLI
    stub = _StubCLI()
    stub.agent = _agent(conn, model="old/model", provider="openrouter")
    stub._snapshot_model_runtime = cli_mod.HermesCLI._snapshot_model_runtime.__get__(stub)
    stub._session_db = GatewaySessionStore(conn)
    stub.session_id = "s"
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: printed.append(str(s)))
    monkeypatch.setattr(cli_mod, "save_config_value", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: SimpleNamespace(
        user_providers=None, custom_providers=None,
        with_overrides=lambda **_: SimpleNamespace(user_providers=None, custom_providers=None)))
    _pick(monkeypatch, "claude-sonnet-4.6", "anthropic", "https://api.anthropic.com", "anthropic_messages")
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)
    return cli_mod, stub, printed


def _pick(monkeypatch, model, provider, base_url, api_mode):
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_: ModelSwitchResult(
        success=True, new_model=model, target_provider=provider, api_key=f"sk-{provider}",
        base_url=base_url, api_mode=api_mode, provider_label=provider.title()))


def _submits(conn):
    return [p for m, p in conn.calls if m == "prompt.submit"]


def _prepare_for_chat(monkeypatch, cli_mod, stub):
    """Enough of HermesCLI for the real ``chat``: its retire, ``_init_agent``, the agent thread
    and the one-turn restore all run. Only the terminal UI around the turn is stubbed."""
    for name in ("chat", "_chat_run_agent", "_chat_release_turn_audio", "_init_agent",
                 "_resolve_turn_agent_config", "_restore_model_runtime_snapshot"):
        setattr(stub, name, getattr(cli_mod.HermesCLI, name).__get__(stub))
    monkeypatch.setattr(cli_mod, "ChatConsole", lambda *a, **k: SimpleNamespace(print=lambda *a, **k: None))
    stub._ensure_runtime_credentials = lambda: True
    stub.finalize_preloaded_skills = lambda: None
    stub._install_tool_callbacks = lambda: None
    stub._ensure_gateway_session = lambda: None
    stub._sync_fallback_chain_with_config = lambda agent: None
    stub._chat_route_images = lambda message, images: message

    def _stage(agent, message):
        stub.conversation_history.append({"role": "user", "content": message})
        agent._pending_cli_user_message = {"role": "user", "content": message}

    stub._chat_stage_user_message = _stage
    stub._reset_stream_state = lambda: None
    stub._chat_setup_turn_audio = lambda turn, message, voice_input: None
    stub._chat_monitor_agent_thread = lambda turn, thread: thread.join()
    stub._chat_settle_turn = lambda turn: None
    stub._chat_render_turn = lambda turn, thread, interrupt: (turn.result or {}).get("final_response")
    stub._flush_credit_notices = lambda: None
    for callback in ("_secret_capture_callback", "_sudo_password_callback", "_approval_callback",
                     "_vault_unlock_callback", "_vault_save_login_callback", "_vault_code_callback"):
        setattr(stub, callback, None)
    stub._pending_moa_config = None
    stub._resumed = False
    stub.conversation_history = []
    stub._active_agent_route_signature = None
    stub.acp_command = None
    stub.acp_args = []
    stub.max_turns = 8
    stub.enabled_toolsets = None
    stub.disabled_toolsets = None
    stub.verbose = False
    stub.tool_progress_mode = "off"
    stub.system_prompt = ""
    stub.prefill_messages = None
    stub.reasoning_config = None
    stub.service_tier = None
    stub._reasoning_user_set = False
    stub._providers_only = None
    stub._providers_ignore = None
    stub._providers_order = None
    stub._provider_sort = None
    stub._provider_require_params = None
    stub._provider_data_collection = None
    stub._openrouter_min_coding_score = None
    stub._single_query_mode = False
    stub._clarify_callback = None
    stub._connection_callback = None
    stub._confirm_owner_model_switch = lambda refusal: False
    stub._withdraw_gateway_modal = lambda method: None
    stub._current_reasoning_callback = lambda: None
    stub._fallback_model = None
    stub._on_thinking = None
    stub.checkpoints_enabled = False
    stub.checkpoint_max_snapshots = 0
    stub.checkpoint_max_total_size_mb = 0
    stub.checkpoint_max_file_size_mb = 0
    stub.pass_session_id = False
    stub.ignore_rules = False
    stub._inline_diffs_enabled = False
    stub.streaming_enabled = False
    stub._on_tool_progress = None
    stub._on_tool_start = None
    stub._on_tool_complete = None
    stub._stream_delta = None
    stub._on_tool_gen_start = None
    stub._on_notice = None
    stub._on_notice_clear = None
    stub._on_reaction = None
    stub._agent_status_print = lambda *_a, **_k: None
    stub._pending_title = None
    stub._auto_load_skills_result = None


def _chat_cli(monkeypatch, conn):
    cli_mod, stub, printed = _cli(monkeypatch, conn)
    _prepare_for_chat(monkeypatch, cli_mod, stub)
    return cli_mod, stub, printed


def test_model_once_rides_one_turn_and_never_touches_the_session(monkeypatch):
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, printed = _chat_cli(monkeypatch, conn)
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic --once")
    assert not [line for line in printed if "fail" in line.lower()], printed
    assert stub.chat("the once turn") == "ok"
    assert stub.chat("the turn after") == "ok"
    mutates, _ = _model_routes(conn)
    assert not mutates, ("--once wrote the shared session model", mutates)
    once, after = _submits(conn)
    assert once.get("model_override") == {"model": "claude-sonnet-4.6", "provider": "anthropic"}, once
    assert "model_override" not in after, after
    assert conn.state("s").info["model"] == "old/model"
    assert (stub.model, stub.agent.model) == ("old/model", "old/model")


def test_a_second_once_before_the_turn_restores_the_route_before_the_first(monkeypatch):
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, _printed = _chat_cli(monkeypatch, conn)
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic --once")
    _pick(monkeypatch, "gpt-5.4", "openai", "https://api.openai.com/v1", "codex_responses")
    cli_mod.HermesCLI._handle_model_switch(stub, "/model gpt-5.4 --provider openai --once")
    stub.chat("the once turn")
    stub.chat("the turn after")
    once, after = _submits(conn)
    assert once.get("model_override") == {"model": "gpt-5.4", "provider": "openai"}, once
    assert "model_override" not in after, after
    assert not _model_routes(conn)[0]
    assert (stub.model, stub.provider, stub.base_url) == ("old/model", "openrouter", "https://openrouter.ai/api/v1")


def test_session_switch_of_the_one_turn_target_commits_on_the_fence(monkeypatch):
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, printed = _chat_cli(monkeypatch, conn)
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic --once")
    assert not _model_routes(conn)[0], "the one-turn pick wrote the session"
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic")
    assert not [line for line in printed if "fail" in line.lower()], printed
    mutates = [p for m, p in conn.calls if m == "session.mutate" and p.get("operation") == "model"]
    assert mutates and mutates[-1]["payload"]["model"] == "claude-sonnet-4.6", mutates
    assert "expected_generation" in mutates[-1], mutates
    assert conn.state("s").info["model"] == "claude-sonnet-4.6"
    stub.chat("now the session route")
    stub.chat("and the next")
    assert all("model_override" not in submit for submit in _submits(conn)), _submits(conn)
    assert (stub.model, stub.agent.model) == ("claude-sonnet-4.6", "claude-sonnet-4.6")


@pytest.mark.parametrize("conn_class", [RecordingConnection, GuardedConnection], ids=["plain", "guarded"])
def test_session_model_switch_commits_the_route_on_the_owner(monkeypatch, conn_class):
    conn = conn_class({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, printed = _cli(monkeypatch, conn)
    asked = []
    stub.agent.model_confirm_callback = lambda refusal: asked.append(refusal["confirm_message"]) or True
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic")
    assert not [line for line in printed if "fail" in line.lower()], printed
    assert asked == ([_CONFIRM_MESSAGE] if conn_class is GuardedConnection else [])
    stub.agent.run_conversation("next turn")
    mutates = [p for m, p in conn.calls if m == "session.mutate" and p.get("operation") == "model"]
    assert mutates[-1]["payload"] == {"model": "claude-sonnet-4.6", "provider": "anthropic",
                                      **({"confirm": "tok"} if conn_class is GuardedConnection else {})}, mutates
    assert all("expected_generation" in m for m in mutates), mutates
    assert conn.state("s").info["model"] == "claude-sonnet-4.6"
    assert "model_override" not in _submits(conn)[-1], "the session already runs it"


def test_once_on_a_resumed_session_asks_before_it_switches(monkeypatch):
    """The owner's large-context guard reads the persisted anchor. ``--once`` does not mutate,
    so the CLI has to ask from that anchor when this process has not measured a turn yet."""
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, _printed = _cli(monkeypatch, conn)
    asked = []

    class _AnchorDB:
        def get_messages_as_conversation(self, sid):
            return [{"role": "user", "content": "hello"}]

        def get_session_model_config_value(self, sid, key, default=None):
            return default

    stub._session_db = _AnchorDB()
    stub._confirm_expensive_model_switch = cli_mod.HermesCLI._confirm_expensive_model_switch.__get__(stub)
    stub._prompt_text_input_modal = lambda **kw: asked.append(kw) or "cancel"
    stub._normalize_slash_confirm_choice = cli_mod.HermesCLI._normalize_slash_confirm_choice.__get__(stub)
    monkeypatch.setattr("agent.usage_anchor.persisted_anchor_tokens", lambda *_a, **_k: 200_000)
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic --once")
    assert asked, "the large-context confirm did not appear"
    assert any("LARGE CONTEXT" in (kw.get("detail") or "") or "context" in (kw.get("title") or "").lower()
               for kw in asked), asked
    assert not _model_routes(conn)[0]
    assert conn.state("s").info["model"] == "old/model"
    assert stub._pending_one_turn_model_restore is None


def test_one_turn_kwargs_reach_only_a_switch_that_names_them():
    """``AIAgent.switch_model`` is a lazy ``**kwargs`` forwarder to a function without ``once`` /
    ``restore``: it gets the plain in-process switch. Its own errors still surface."""
    from hermes_cli.cli_model_switch_mixin import _call_agent_switch
    from run_agent import AIAgent

    seen = []

    def _in_process(agent, new_model, new_provider, api_key="", base_url="", api_mode="", capabilities=None):
        seen.append(new_model)
        if new_model == "broken":
            raise TypeError("provider client could not be built")
        return True

    class _Forwarder:
        switch_model = AIAgent.switch_model

    class _OneTurnAware:
        def switch_model(self, new_model, new_provider="", *, once=False, restore=False):
            seen.append((new_model, once, restore))
            return True

    import agent.agent_runtime_helpers as helpers
    original = helpers.switch_model
    helpers.switch_model = _in_process
    try:
        assert _call_agent_switch(_Forwarder(), new_model="next", new_provider="openrouter", once=True) is True
        assert _call_agent_switch(_Forwarder(), new_model="prev", new_provider="openrouter",
                                  once=True, restore=True) is True
        with pytest.raises(TypeError, match="could not be built"):
            _call_agent_switch(_Forwarder(), new_model="broken", new_provider="openrouter", once=True)
    finally:
        helpers.switch_model = original
    assert _call_agent_switch(_OneTurnAware(), new_model="next", once=True, restore=True) is True
    assert seen == ["next", "prev", "broken", ("next", True, True)]


def test_switching_to_the_route_the_session_runs_commits_nothing(monkeypatch):
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, _printed = _cli(monkeypatch, conn)
    _pick(monkeypatch, "old/model", "openrouter", "https://openrouter.ai/api/v1", "chat_completions")
    cli_mod.HermesCLI._handle_model_switch(stub, "/model old/model --provider openrouter")
    stub.agent.run_conversation("next turn")
    mutates, overrides = _model_routes(conn)
    assert not mutates and not overrides, (mutates, overrides)
