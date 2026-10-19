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


def test_switching_to_the_route_the_session_runs_commits_nothing(monkeypatch):
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, _printed = _cli(monkeypatch, conn)
    _pick(monkeypatch, "old/model", "openrouter", "https://openrouter.ai/api/v1", "chat_completions")
    cli_mod.HermesCLI._handle_model_switch(stub, "/model old/model --provider openrouter")
    stub.agent.run_conversation("next turn")
    mutates, overrides = _model_routes(conn)
    assert not mutates and not overrides, (mutates, overrides)


def _reasoning_mutates(conn):
    return [p["payload"] for m, p in conn.calls if m == "session.mutate" and p.get("operation") == "reasoning"]


def test_once_reasoning_rides_the_admission_and_never_mutates_the_session(monkeypatch):
    """Review of #140969: ``/model B --once --reasoning high`` sent session-scope ``reasoning`` mutations
    (each a durable ``session_pins.reasoning``), so the session stopped following its profile's level."""
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, printed = _chat_cli(monkeypatch, conn)
    stub._confirm_and_apply_cli_model_switch = cli_mod.HermesCLI._confirm_and_apply_cli_model_switch.__get__(stub)
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic --once --reasoning high")
    assert not [line for line in printed if "fail" in line.lower()], printed
    assert stub.chat("the once turn") == "ok"
    assert stub.chat("the turn after") == "ok"
    assert _reasoning_mutates(conn) == []
    once, after = _submits(conn)
    assert once.get("model_override") == {"model": "claude-sonnet-4.6", "provider": "anthropic", "reasoning": "high"}
    assert "model_override" not in after, after
    assert stub._reasoning_user_set is False


def test_once_reasoning_equal_to_the_inherited_level_commits_nothing(monkeypatch):
    """dokterdok's probe: the session already runs ``high`` by inheritance; neither the --once pick nor its
    restore may write the session's reasoning."""
    import cli
    from hermes_cli.cli_model_switch_mixin import _commit_model_switch
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter", "reasoning_effort": "high"})
    _cli_mod, stub, _printed = _cli(monkeypatch, conn)
    stub.agent = _agent(conn, model="old/model", provider="openrouter",
                        reasoning_config={"enabled": True, "effort": "high"})
    stub.reasoning_config = {"enabled": True, "effort": "high"}
    stub._reasoning_user_set = False
    result = ModelSwitchResult(success=True, new_model="next/model", target_provider="openrouter",
                               api_key="key", base_url=stub.base_url, api_mode="chat_completions")
    _commit_model_switch(stub, result, persist_global=False, one_turn=True, reasoning_effort="high")
    cli.HermesCLI._restore_model_runtime_snapshot(stub, stub._pending_one_turn_model_restore)
    assert _reasoning_mutates(conn) == []


@pytest.mark.parametrize("session_model", ["shared-model", "other-model"])
def test_once_sends_the_alias_it_resolved_through(monkeypatch, session_model):
    """Review by @dokterdok: two ``model_aliases`` entries name one model on one provider, at different
    endpoints. ``/model second --once`` sends ``second``, not the bare model the owner would re-resolve to
    the first alias, and still sends it when the session already runs that model."""
    from gateway.session_cli_turn import one_turn_runtime
    from hermes_cli import model_switch
    resolve = model_switch.switch_model
    conn = RecordingConnection({"model": session_model, "provider": "custom"})
    cli_mod, stub, printed = _chat_cli(monkeypatch, conn)
    monkeypatch.setattr(model_switch, "switch_model", resolve)  # the real resolver over these aliases
    for name, port in (("first", 7002), ("second", 7003)):
        monkeypatch.setitem(model_switch.DIRECT_ALIASES, name, model_switch.DirectAlias(
            model="shared-model", provider="custom", base_url=f"http://127.0.0.1:{port}/v1", api_key=f"sk-{name}"))
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *_a, **_k: {
        "accepted": True, "persist": True, "recognized": True, "message": ""})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    session = {"provider": "custom", "api_key": "sk-session", "base_url": "http://127.0.0.1:7001/v1"}
    stub.model, stub.provider, stub.requested_provider = session_model, "custom", "custom"
    stub.base_url = stub._explicit_base_url = session["base_url"]
    stub.api_key = stub._explicit_api_key = session["api_key"]
    stub.agent.model, stub.agent.provider = session_model, "custom"
    cli_mod.HermesCLI._handle_model_switch(stub, "/model second --once")
    assert not [line for line in printed if "fail" in line.lower()], printed
    assert stub.agent.model == "shared-model", "the agent shows the resolved model, not the alias"
    stub.chat("run on second")
    stub.chat("the turn after")
    once, after = _submits(conn)
    override = once.get("model_override")
    assert override == {"model": "second", "provider": "custom"}, override
    assert "model_override" not in after, after
    _model, runtime = one_turn_runtime(override, session_model, {**session, "api_mode": "chat_completions"}, {})
    assert (runtime["base_url"], runtime["api_key"]) == ("http://127.0.0.1:7003/v1", "sk-second")
    assert not _model_routes(conn)[0]


def test_once_through_an_alias_named_like_the_session_model_still_sends_it(monkeypatch):
    """Review by @dokterdok: the session runs ``shared-model`` on its own endpoint and key, and a
    ``model_aliases`` entry ``shared-model`` names that model at another endpoint. ``/model shared-model
    --once`` resolves the alias, but the submit compared the staged ``shared-model`` with the session's
    route again, dropped it, and the turn ran on the session's endpoint and key."""
    from gateway.session_cli_turn import one_turn_runtime
    from hermes_cli import model_switch
    resolve = model_switch.switch_model
    conn = RecordingConnection({"model": "shared-model", "provider": "custom"})
    cli_mod, stub, printed = _chat_cli(monkeypatch, conn)
    monkeypatch.setattr(model_switch, "switch_model", resolve)  # the real resolver over this alias
    monkeypatch.setitem(model_switch.DIRECT_ALIASES, "shared-model", model_switch.DirectAlias(
        model="shared-model", provider="custom", base_url="http://127.0.0.1:7003/v1", api_key="sk-alias"))
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *_a, **_k: {
        "accepted": True, "persist": True, "recognized": True, "message": ""})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    session = {"provider": "custom", "api_key": "sk-session", "base_url": "http://127.0.0.1:7001/v1"}
    stub.model, stub.provider, stub.requested_provider = "shared-model", "custom", "custom"
    stub.base_url = stub._explicit_base_url = session["base_url"]
    stub.api_key = stub._explicit_api_key = session["api_key"]
    stub.agent.model, stub.agent.provider = "shared-model", "custom"
    cli_mod.HermesCLI._handle_model_switch(stub, "/model shared-model --once")
    assert not [line for line in printed if "fail" in line.lower()], printed
    assert stub.base_url == "http://127.0.0.1:7003/v1", "the CLI shows the alias's endpoint"
    stub.chat("run on the alias")
    stub.chat("the turn after")
    once, after = _submits(conn)
    override = once.get("model_override")
    assert override == {"model": "shared-model", "provider": "custom"}, once
    assert "model_override" not in after, after
    _model, runtime = one_turn_runtime(override, "shared-model", {**session, "api_mode": "chat_completions"}, {})
    assert (runtime["base_url"], runtime["api_key"]) == ("http://127.0.0.1:7003/v1", "sk-alias")
    assert not _model_routes(conn)[0]


class BusyOnceConnection(RecordingConnection):
    """The owner refuses the first ``prompt.submit`` with ``session_busy``: nothing was admitted."""

    def __init__(self, info):
        super().__init__(info)
        self.refused = False

    def call(self, method, *, timeout=None, **params):
        if method == "prompt.submit" and not self.refused:
            from hermes_cli.gateway_client import GatewayRPCError
            self.refused = True
            self.calls.append((method, params))
            raise GatewayRPCError("session_busy")
        return super().call(method, timeout=timeout, **params)


def test_a_once_pick_refused_as_session_busy_rides_the_resend(monkeypatch):
    """Review of #140969: the submit's ``finally`` cleared the pick even when the owner refused it with
    ``session_busy`` (nothing admitted), and the CLI restored its route, so the resend ran on the session
    model."""
    conn = BusyOnceConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, _printed = _chat_cli(monkeypatch, conn)
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic --once")
    stub.chat("refused")
    assert stub.chat("the resend") == "ok"
    assert stub.chat("the turn after") == "ok"
    refused, resend, after = _submits(conn)
    pick = {"model": "claude-sonnet-4.6", "provider": "anthropic"}
    assert refused.get("model_override") == pick
    assert resend.get("model_override") == pick, resend
    assert "model_override" not in after, after
    assert (stub.model, stub.agent.model) == ("old/model", "old/model")


def test_the_once_pick_is_taken_when_the_admission_is_built(monkeypatch):
    """dokterdok on #140969: consume the pick when the params are built, not in the call's ``finally``, so
    a second ``run_conversation`` submitting while the first ``prompt.submit`` is still blocked cannot carry
    it too."""
    seen = []

    class Watching(RecordingConnection):
        def call(self, method, *, timeout=None, **params):
            if method == "prompt.submit":
                seen.append(agent._turn_route)  # the first submit is "blocked" here
            return super().call(method, timeout=timeout, **params)

    conn = Watching({"model": "old/model", "provider": "openrouter"})
    agent = _agent(conn, model="claude-sonnet-4.6", provider="anthropic", one_turn=True)
    assert agent._turn_route
    agent.run_conversation("the once turn")
    assert seen == [None]
    assert _submits(conn)[0]["model_override"] == {"model": "claude-sonnet-4.6", "provider": "anthropic"}


def test_an_image_staging_failure_is_a_failed_turn_not_an_exception():
    """dokterdok on #140969: ``_submit_params`` stages images over the gateway (``image.attach_bytes``); it
    moved out of the ``try``, so a lost connection while staging raised out of ``run_conversation``."""
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})  # attach_bytes is unsupported
    agent = _agent(conn, model="old/model", provider="openrouter")
    parts = [{"type": "text", "text": "what is this"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
    result = agent.run_conversation(parts)
    assert (result.get("failed"), result.get("failure_reason")) == (True, "gateway_connection"), result
    assert not _submits(conn)


def test_a_session_reasoning_pick_is_committed_once_not_on_every_rebuild(monkeypatch):
    """Review of #140969: ``_reasoning_user_set`` never cleared after the commit, so every later agent
    rebuild (``/fast``, ``/personality``, a route change, a --once restore) re-sent the explicit level: a
    full ``session.resume``, a prepare+commit, a receipt and a gateway agent eviction each time."""
    from hermes_cli.cli_agent_setup_mixin import _retire_agent
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    _cli_mod, stub, _printed = _chat_cli(monkeypatch, conn)
    stub.agent = None
    stub.reasoning_config = {"enabled": True, "effort": "high"}
    stub._reasoning_user_set = True  # ``/reasoning high`` (cli_commands_mixin) then retires the agent
    assert stub.chat("first") == "ok"
    _retire_agent(stub)  # e.g. /personality
    assert stub.chat("second") == "ok"
    assert _reasoning_mutates(conn) == [{"effort": "high"}]
    assert stub._reasoning_user_set is False


@pytest.mark.parametrize("override", [
    {"model": "first", "reasoning": "high"},
    {"model": "claude-sonnet-4.6", "provider": "anthropic", "reasoning": "high"},
])
def test_once_reasoning_admission_passes_the_real_submit_contract(monkeypatch, override):
    """Live pass on #140969: ``/model first --once --reasoning high`` failed its turn ``invalid_params``.
    The CLI sends ``model_override.reasoning`` but ``CanonicalModelOverride`` forbade the key, and the
    owner's dispatcher (``canonical_param_problems``) refuses an unknown key before the handler runs.
    The recorded submit goes through that same production check, not a stubbed connection."""
    from tui_gateway.contracts.registry import CANONICAL_METHODS, canonical_param_problems
    contract = CANONICAL_METHODS["prompt.submit"]
    params = {"session_id": "s", "text": "t", "submission_id": "x", "model_override": override}
    assert canonical_param_problems(contract, params) == []
    assert contract.params.model_validate(params).model_override.reasoning == "high"

    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    cli_mod, stub, _printed = _chat_cli(monkeypatch, conn)
    stub._confirm_and_apply_cli_model_switch = cli_mod.HermesCLI._confirm_and_apply_cli_model_switch.__get__(stub)
    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic --once --reasoning high")
    assert stub.chat("the once turn") == "ok"
    (submit,) = _submits(conn)
    assert submit["model_override"].get("reasoning") == "high", submit
    assert canonical_param_problems(contract, submit) == [], submit


def test_every_model_override_key_the_cli_sends_is_in_the_contract():
    """``GatewayAgent._take_turn_route`` and the ``/moa`` one-shot build ``model_override`` from
    ``_route_off_session`` (``model``/``provider``) plus ``reasoning``: each key must be declared."""
    from tui_gateway.contracts.canonical_sessions import CanonicalModelOverride
    assert {"model", "provider", "reasoning"} <= set(CanonicalModelOverride.model_fields)

