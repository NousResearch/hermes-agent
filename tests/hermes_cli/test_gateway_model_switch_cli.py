"""The classic CLI's model switches over the shared gateway session (#134889 M1, review of #138717).

The CLI's agent is a ``GatewayAgent`` stand-in and the gateway session is the source of truth for the
model a turn runs: ``/new`` must not rewrite the session it just ended, a ``/model`` before the first
turn must commit on the session that turn runs on, and the owner's model-guard refusal is the user's
to answer in the CLI's own prompt (never silently confirmed).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.model_switch import ModelSwitchResult
from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with

REFUSAL = {"status": "confirmation_required", "confirm": "tok-1",
           "confirm_message": "This session holds ~150,000 tokens of context.",
           "warnings": [{"kind": "context_cache", "title": "Large Context Switch Warning", "message": "..."}]}


@pytest.fixture
def gateway():
    client = FakeClient([])
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    conn.state(SID)
    yield conn, client
    conn.close()


def _mutations(client):
    return [params for method, params in client.calls if method == "session.mutate"]


def _cli(**fields):
    base = dict(agent=None, _gateway_session_open=True, session_id=SID, model="old-model", provider="openrouter",
                requested_provider="openrouter", api_key="k", base_url="", api_mode="", _explicit_api_key=None,
                _explicit_base_url=None, reasoning_config=None)
    return SimpleNamespace(**{**base, **fields})


def _answering(cli, *answers):
    """Bind main's guard prompt onto ``cli`` with scripted answers; returns what each prompt showed."""
    from cli import HermesCLI
    shown, pending = [], list(answers)

    def modal(*, title, detail, choices, timeout):
        shown.append((title, detail, [choice[0] for choice in choices]))
        return pending.pop(0)
    cli._prompt_text_input_modal = modal
    cli._normalize_slash_confirm_choice = lambda raw, choices: HermesCLI._normalize_slash_confirm_choice(cli, raw, choices)
    cli._confirm_owner_model_switch = lambda refusal: HermesCLI._confirm_owner_model_switch(cli, refusal)
    return shown


def test_new_after_a_model_switch_leaves_the_ended_session_alone(gateway):
    """Review item 1: /new reset the model to the config default through the stand-in before it was
    rebound, so ``session.mutate model`` rewrote the session just ended (a later /resume of it came back
    on the default). The new session is created on the profile's default and runs it."""
    from cli import CLI_CONFIG, HermesCLI
    from hermes_cli.cli_init_mixin import CLIInitMixin
    from hermes_cli.gateway_session_store import GatewaySessionStore
    conn, client = gateway
    rpc = client.rpc

    async def create_on_the_default(method, **params):
        if method != "session.create":
            return await rpc(method, **params)
        client.calls.append((method, params))
        return {"stored_session_id": "local-new", "revision": 1, "execution_generation": 1,
                "info": {"model": "config-default-model", "provider": "openrouter"}}
    client.rpc = create_on_the_default
    agent = GatewayAgent(connection=conn, session_id=SID, model="config-default-model", provider="openrouter")
    agent.switch_model(new_model="gpt-x", new_provider="openrouter")  # /model gpt-x
    client.calls.clear()
    cli = _cli(agent=agent, model="gpt-x", _session_db=GatewaySessionStore(connection=conn), conversation_history=[],
               _pending_title=None, _resumed=False, _notify_session_boundary=MagicMock(),
               _discard_session_if_empty=MagicMock(), service_tier=None, enabled_toolsets=None, _explicit_toolsets=False)
    cli._new_session_id = lambda now: CLIInitMixin._new_session_id(cli, now)
    default = ModelSwitchResult(success=True, new_model="config-default-model", target_provider="openrouter",
                                api_key="k", base_url="", api_mode="chat_completions")
    with patch.dict(CLI_CONFIG, {"model": {"default": "config-default-model", "provider": "openrouter"}}), \
            patch("hermes_cli.model_switch.switch_model", return_value=default):
        HermesCLI.new_session(cli, silent=True)
    assert _mutations(client) == []
    assert "model" not in dict(client.calls)["session.create"]  # the profile's default
    assert cli.session_id == agent.session_id == "local-new"
    assert (cli.model, agent.model) == ("config-default-model", "config-default-model")


def test_model_before_the_first_turn_commits_on_the_session_that_turn_runs(gateway, monkeypatch):
    """Review item 3: before the first turn there is no stand-in, and the one built for that turn read
    the session's (old) model: "Model switched" was printed and the turn ran the old model. The switch
    now commits on the gateway session."""
    from cli import HermesCLI
    conn, client = gateway
    conn.state(SID).info.update(model="old-model", provider="openrouter")
    cli = _cli()

    def init_agent():  # main's _init_agent: the stand-in on this CLI's gateway session
        cli.agent = GatewayAgent(connection=conn, session_id=SID, model=cli.model, provider=cli.provider)
        return True
    cli._init_agent = init_agent
    monkeypatch.setattr("cli._cprint", lambda *_a, **_k: None)
    picked = ModelSwitchResult(success=True, new_model="new-model", target_provider="openrouter", api_key="k",
                               base_url="", api_mode="chat_completions")
    assert HermesCLI._stage_and_swap_model(cli, picked, "old-model") is True
    [mutation] = _mutations(client)
    assert (mutation["session_id"], mutation["payload"]["model"]) == (SID, "new-model")
    first_turn = GatewayAgent(connection=conn, session_id=SID, model="old-model", provider="openrouter")
    assert first_turn.model == "new-model"


def test_the_owners_refusal_is_asked_in_the_clis_model_guard_prompt():
    """Review item 2: the owner's warning is the prompt's title and body, as Ink/Desktop/ACP show it."""
    from cli import HermesCLI
    cli = SimpleNamespace()
    shown = _answering(cli, "once", "cancel")
    assert HermesCLI._confirm_owner_model_switch(cli, REFUSAL) is True
    assert HermesCLI._confirm_owner_model_switch(cli, REFUSAL) is False
    assert shown == [("!!! Large Context Switch Warning !!!", REFUSAL["confirm_message"], ["once", "cancel"])] * 2


def test_a_declined_owner_warning_switches_nothing(gateway, monkeypatch):
    from cli import HermesCLI
    conn, client = gateway
    printed = []
    monkeypatch.setattr("cli._cprint", lambda text, *_a, **_k: printed.append(str(text)))
    cli = _cli()
    shown = _answering(cli, "cancel")
    cli.agent = GatewayAgent(connection=conn, session_id=SID, model="old-model", provider="openrouter",
                             model_confirm_callback=cli._confirm_owner_model_switch)
    client.mutate_replies = [REFUSAL]
    picked = ModelSwitchResult(success=True, new_model="big-model", target_provider="openrouter", api_key="k",
                               base_url="", api_mode="chat_completions")
    assert HermesCLI._stage_and_swap_model(cli, picked, "old-model") is False
    assert [title for title, _detail, _choices in shown] == ["!!! Large Context Switch Warning !!!"]
    assert [m["payload"] for m in _mutations(client)] == [{"model": "big-model", "provider": "openrouter"}]
    assert (cli.model, cli.agent.model) == ("old-model", "old-model")
    assert any("Model switch cancelled." in line for line in printed)


def test_one_turn_restore_does_not_rewrite_the_session(gateway, monkeypatch):
    """``/model --once`` never commits, so restoring the previous route is local. An owner guard
    cannot pin the shared session to the one-turn model, and the next turn is not an override."""
    from cli import HermesCLI
    conn, client = gateway
    monkeypatch.setattr("cli._cprint", lambda *_a, **_k: None)
    conn.state(SID).info.update(model="small-model", provider="openrouter")
    cli = _cli(model="big-model", provider="openrouter")
    cli.agent = GatewayAgent(connection=conn, session_id=SID, model="small-model", provider="openrouter")
    assert cli.agent.switch_model(new_model="big-model", new_provider="openrouter", once=True) is True
    client.calls.clear()
    client.mutate_replies = [REFUSAL]
    HermesCLI._restore_model_runtime_snapshot(cli, {
        "model": "small-model", "provider": "openrouter", "api_key": "k", "base_url": "", "api_mode": ""})
    assert _mutations(client) == []
    assert (cli.model, cli.agent.model) == ("small-model", "small-model")
    assert cli.agent._turn_route is None
