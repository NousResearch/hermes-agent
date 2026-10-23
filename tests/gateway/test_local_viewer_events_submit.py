"""The classic CLI's per-turn ``prompt.submit`` fields survive canonical admission (CALLMAP W2).

Main's HermesCLI sends the model an API-only message (voice prefix, model-switch and skills-reload
notes) while persisting the clean text, names the turn's author, and runs ``/model --once`` / ``/moa``
as a one-turn override. Each is committed with the admission row as ``cli_turn_v1`` and bound to that
admission's execution only.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform

from tests.gateway.test_surface_admission import _authority


@pytest.mark.asyncio
async def test_cli_turn_fields_are_admitted_and_bound_to_their_own_turn(tmp_path, monkeypatch):
    from gateway.session_cli_turn import cli_turn
    from gateway.session_controls import AuthorityConnection
    from gateway.session_ingress import admission_author
    from hermes_state_runtime import list_session_admissions

    seen = []
    async def answer(event):
        seen.append((event.text, cli_turn(event.message_id), cli_turn('another-event'), admission_author.get()))
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    author = {'id': 'cli:alice', 'name': 'Alice', 'is_bot': False}
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        for rid, extra in (
                ('full', {'turn_author': author, 'api_text': '[voice] hi', 'clarify_batch': True,
                          'model_override': {'model': ' gpt-x ', 'provider': 'openrouter'}}),
                ('plain', {})):
            reply = await connection.dispatch({'id': 2, 'method': 'prompt.submit', 'params': {
                'session_id': 's', 'submission_id': rid, 'text': rid, **extra}})
            assert reply.get('result', {}).get('status') == 'queued', reply
            await authority.sessions['s'].task
        for rid, bad in (('no-model', {'model_override': {'provider': 'x'}}),
                         ('extra-key', {'model_override': {'model': 'm', 'base_url': 'http://x'}}),
                         ('bad-text', {'api_text': 7}),
                         ('bad-author', {'turn_author': 'not json'}),
                         ('bad-batch', {'clarify_batch': 'yes'})):
            refused = await connection.dispatch({'id': 3, 'method': 'prompt.submit', 'params': {
                'session_id': 's', 'submission_id': rid, 'text': 'x', **bad}})
            assert refused['error']['message'] == 'invalid_params', (rid, refused)
    finally:
        await connection.close()

    rows = {row['request_id']: row for row in list_session_admissions(authority.db, session_id='s', pending_only=False)}
    assert rows['full']['payload']['cli_turn_v1'] == {
        'turn_author': author, 'api_text': '[voice] hi', 'clarify_batch': True,
        'model_override': {'model': 'gpt-x', 'provider': 'openrouter'}}
    assert 'cli_turn_v1' not in rows['plain']['payload']
    assert not {'no-model', 'extra-key', 'bad-text', 'bad-author', 'bad-batch'} & set(rows)
    assert seen[0] == ('full', rows['full']['payload']['cli_turn_v1'], {}, author)
    # The next admission never inherits the previous turn's fields.
    assert seen[1][:3] == ('plain', {}, {})


class _RecordingAgent:
    built: list = []

    def __init__(self, **kwargs):
        type(self).built.append(kwargs)
        self.model = kwargs["model"]
        self.session_id = kwargs.get("session_id")
        self.tools = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200_000)
        self.session_prompt_tokens = self.session_completion_tokens = 0
        self.calls = []

    def run_conversation(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return {"final_response": "ok", "messages": []}


def _gateway_runner():
    runner = MagicMock()
    runner.config = SimpleNamespace(streaming=None)
    runner._provider_routing = {}
    runner._agent_cache_lock = None
    runner._agent_cache = {}
    runner._session_db = runner._prefill_messages = None
    runner._pending_model_notes = runner._pending_skills_reload_notes = {}
    runner.session_store._entries = {}
    runner._pre_agent_fallback_notice = None
    runner._get_system_prompt_for_channel.return_value = None
    runner._resolve_session_agent_runtime.return_value = (
        "session-model", {"provider": "custom", "api_key": "k1", "base_url": "http://session"})
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._resolve_turn_agent_config.side_effect = lambda _msg, model, rt: {
        "model": model, "runtime": {k: v for k, v in rt.items() if k != "request_overrides"}}
    runner._agent_config_signature.return_value = ("sig",)
    runner._extract_cache_busting_config.return_value = {}
    runner._refresh_fallback_model.return_value = None
    runner._consume_pending_native_image_paths.return_value = []
    runner._consume_pending_turn_sidecar_notes.return_value = []
    for lane in ("_is_telegram_topic_lane", "_is_discord_auto_thread_lane", "_is_relay_discord_channel_lane"):
        getattr(runner, lane).return_value = False
    return runner


def _run_turn(runner, admission_id="adm-1"):
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext
    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        message="clean text", history=[], session_id="sid", session_key="key", user_config={},
        inbound_message_id=admission_id, AIAgent=_RecordingAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    result = TurnRunner(runner, ctx).run_sync()
    return result, ctx.agent_holder[0]


def test_the_turn_sends_api_text_persists_the_clean_text_and_runs_the_one_turn_model():
    from gateway.session_cli_turn import cli_turn_scope
    from hermes_cli.model_switch import ModelSwitchResult

    switched = ModelSwitchResult(success=True, new_model="gpt-x", target_provider="openrouter",
                                 api_key="k2", base_url="http://or", api_mode="chat_completions")
    _RecordingAgent.built = []
    runner = _gateway_runner()
    committed = {"api_text": "[voice] clean text", "model_override": {"model": "gpt-x", "provider": "openrouter"}}
    with patch("hermes_cli.model_switch.switch_model", return_value=switched) as switch, \
            cli_turn_scope("adm-1", committed):
        result, agent = _run_turn(runner)
    assert result["final_response"] == "ok"
    [(message, kwargs)] = agent.calls
    assert message == "[voice] clean text" and kwargs["persist_user_message"] == "clean text"
    assert switch.call_args.kwargs["raw_input"] == "gpt-x"
    assert switch.call_args.kwargs["explicit_provider"] == "openrouter"
    assert switch.call_args.kwargs["current_model"] == "session-model"
    built = _RecordingAgent.built[-1]
    assert (built["model"], built["provider"], built["api_key"], built["base_url"]) == (
        "gpt-x", "openrouter", "k2", "http://or")

    # The next turn (and any turn whose event is not that admission) runs the session's own route.
    with patch("hermes_cli.model_switch.switch_model") as switch, cli_turn_scope("adm-1", committed):
        _result, agent = _run_turn(runner, admission_id="adm-2")
    switch.assert_not_called()
    [(message, kwargs)] = agent.calls
    assert message == "clean text" and "persist_user_message" not in kwargs
    assert _RecordingAgent.built[-1]["model"] == "session-model"


def test_an_unresolvable_one_turn_model_fails_the_turn_without_running_it():
    from gateway.session_cli_turn import cli_turn_scope
    from hermes_cli.model_switch import ModelSwitchResult

    _RecordingAgent.built = []
    failed = ModelSwitchResult(success=False, error_message="Unknown model 'nope'")
    with patch("hermes_cli.model_switch.switch_model", return_value=failed), \
            cli_turn_scope("adm-1", {"model_override": {"model": "nope"}}):
        result, agent = _run_turn(_gateway_runner())
    assert result["failed"] is True and "Unknown model 'nope'" in result["final_response"]
    assert agent is None and _RecordingAgent.built == []


class _SteeredAgent(_RecordingAgent):
    def run_conversation(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return {"final_response": "", "messages": [], "failed": True, "error": "HTTP 402",
                "failure_reason": "billing", "pending_steer": "also check the logs",
                "pre_transform_response": "raw draft", "response_transformed": True,
                "billing_block": {"provider": "openrouter", "url": "https://openrouter.ai/credits"},
                "last_reasoning": "thought", "turn_exit_reason": "billing"}


@pytest.mark.parametrize("classic", [True, False])
def test_a_classic_cli_receipt_carries_the_run_conversation_keys(classic):
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.session_results import execution_result
    from gateway.turn_context import TurnContext
    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        message="hi", history=[], session_id="sid", session_key="key", user_config={},
        AIAgent=_SteeredAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    captured = {}
    token = execution_result.set(captured)
    try:
        with patch("gateway.session_cli_turn.classic_cli_turn", return_value=classic):
            delivered = TurnRunner(_gateway_runner(), ctx).run_sync()
    finally:
        execution_result.reset(token)
    stored = captured["result"]
    if classic:
        # main's CLI renders its own chat_error_copy for an empty failed turn and queues the
        # leftover /steer itself, so the gateway must not also drain it as a follow-up.
        assert stored["final_response"] == ""
        assert stored["pending_steer"] == "also check the logs"
        assert stored["pre_transform_response"] == "raw draft" and stored["response_transformed"] is True
        assert stored["billing_block"]["provider"] == "openrouter"
        assert (stored["last_reasoning"], stored["turn_exit_reason"]) == ("thought", "billing")
        assert "pending_steer" not in ctx.result_holder[0]
    else:
        assert "pending_steer" not in stored and "billing_block" not in stored
        assert ctx.result_holder[0]["pending_steer"] == "also check the logs"
    # The delivered reply (message.complete text) keeps the gateway's own copy either way.
    assert delivered["final_response"] and stored["failed"] is True


def test_only_a_classic_cli_surface_session_is_a_classic_cli_turn():
    from gateway.session_cli_turn import classic_cli_turn
    turn = SimpleNamespace(_approval_owner=("authority", "s", 1), _runner=object(),
                           _ctx=SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL)))
    for platform, expected in (("cli", True), ("tui", False), ("desktop", False)):
        with patch("gateway.session_policy.policy_for_source", return_value=SimpleNamespace(platform=platform)):
            assert classic_cli_turn(turn) is expected
    turn._ctx.source.platform = Platform.TELEGRAM
    assert classic_cli_turn(turn) is False


def test_a_finite_turn_answers_clarify_headless_at_once():
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_finite import FINITE_CLARIFY_NOTICE, finite_turn_scope
    turn = object.__new__(TurnRunner)
    turn._ask_clarify_question = lambda *a, **k: pytest.fail("a finite viewer can never answer")
    questions = [{"qid": "q1", "question": "Which env?", "choices": ["dev", "prod"], "multi_select": False}]
    with finite_turn_scope(True):
        reply = turn._clarify_callback_sync(questions)
    assert reply == {"answers": {}, "outcome": "undelivered", "notice": FINITE_CLARIFY_NOTICE}
    # main's -q text, verbatim
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback
    assert reply == _single_query_clarify_callback(questions)


def test_a_finite_managed_worker_answers_clarify_headless():
    from agent.managed_worker import WorkerControls
    from gateway.session_finite import finite_turn_scope
    controls = object.__new__(WorkerControls)
    controls._ask = lambda *a: pytest.fail("a finite viewer can never answer")
    with finite_turn_scope(True):
        assert controls.clarify([{"qid": "q", "question": "?", "choices": [], "multi_select": False}])["outcome"] == "undelivered"


def test_a_one_turn_model_never_becomes_the_sessions_stored_model():
    """``/moa`` and ``/model --once`` run one admission on another route. The turn's post-run sync
    wrote the executing agent's model to the session row (``default``, the MoA preset), so a /resume
    right after restored onto the preset. Only a turn on the session's own route records its model."""
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from gateway.session_cli_turn import cli_turn_scope
    from hermes_cli.model_switch import ModelSwitchResult

    written = []
    db = SimpleNamespace(get_session=lambda sid: {"model": "session-model", "model_config": "{}"},
                         update_session_meta=lambda sid, config, model=None: written.append((sid, model)))
    runner = _gateway_runner()
    runner._session_db = SimpleNamespace(_db=db)
    runner._sync_session_model_from_agent = (
        lambda sid, agent: GatewayTurnPrepareMixin._sync_session_model_from_agent(runner, sid, agent))
    moa = ModelSwitchResult(success=True, new_model="default", target_provider="moa", api_key="",
                            base_url="", api_mode="chat_completions")
    committed = {"model_override": {"model": "default", "provider": "moa"}}
    with patch("hermes_cli.model_switch.switch_model", return_value=moa), cli_turn_scope("adm-1", committed):
        _result, agent = _run_turn(runner)
    assert agent.model == "default" and written == []
    with cli_turn_scope("adm-1", committed):
        _run_turn(runner, admission_id="adm-2")
    assert [model for _sid, model in written] == ["session-model"]
