"""Gateway delegation joins inside the turn when interim assistant messages are off.

Top-level ``delegate_task`` on a messaging gateway runs in the background, and its tool text tells
the model to "give a one-line status and END YOUR TURN". With ``display.interim_assistant_messages``
off that status is the turn's FINAL answer, so the chat still receives a "checks are running, no
findings yet" message and the real answer later, as a second message. ``delegation.gateway_mode``
(default ``auto``) now joins the children inside the turn on such sessions: the model waits for them
in the tool call and the chat gets one completed answer. Related: #95522, #141778.

Real ``GatewayRunner._run_agent`` + real ``AIAgent`` + real child ``AIAgent`` against a loopback
model. The model obeys whatever the ``delegate_task`` tool tells it: given a dispatch handle it ends
its turn with a status, given the child's result it answers. The adapter is a recording transport
driven by the base adapter's own ``_process_message_background`` (typing loop + final send).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

import hermes_yaml as yaml
from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import FakeLLMServer, Hang, Text, ToolCall, write_hermes_home
from tests.gateway.test_run_progress_topics import _make_runner
from tests.waiting import wait_until

GOAL = "Audit the billing module for stale invoices"
FINDING = "BILLING-OK-7731"
PAUSE = "Checks are running in parallel, I don't have the findings yet."
ANSWER = f"Done. The billing audit came back clean: {FINDING}"
END_TURN = "END YOUR TURN"


class _Chat(BasePlatformAdapter):
    """Recording transport: every outbound message lands in ``sent``."""

    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: list[str] = []
        self.typing = 0

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content, **_):
        self.sent.append(content)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        self.typing += 1

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _first_user_text(record: dict) -> str:
    messages = record["body"].get("messages", [])
    return str(next((m.get("content") for m in messages if m.get("role") == "user"), ""))


def _model(child_reply):
    """A model that follows the tool contract: a dispatch handle gets a status, a result gets the answer."""

    def respond(request):
        if GOAL in _first_user_text(request):  # the child's own conversation
            return child_reply
        last = request["body"]["messages"][-1]
        if last.get("role") == "tool":
            return Text(ANSWER if FINDING in str(last.get("content")) else PAUSE)
        return ToolCall("delegate_task", {"tasks": [{"goal": GOAL, "context": "Repo at /srv/billing."}]})

    def aux(request):
        return child_reply if GOAL in _first_user_text(request) else Text("Fake summary.")

    return respond, aux


def _delegate_def(request: dict) -> dict:
    return next(t["function"] for t in request["tools"] if t["function"]["name"] == "delegate_task")


@pytest.fixture
def dispatches(monkeypatch):
    """Every hand-off to the async completion registry (the background path)."""
    from tools import async_delegation

    calls = []
    real = async_delegation.dispatch_async_delegation_batch

    def recording(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", recording)
    return calls


async def _turn(monkeypatch, config: dict, *, platform=Platform.WHATSAPP, model=None, during=None):
    """One inbound message through the real gateway turn; returns (chat, the parent's model requests)."""
    import gateway.run as gateway_run
    from tools import async_delegation

    home = Path(os.environ["HERMES_HOME"])
    chat = _Chat(platform)
    source = SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1")
    event = MessageEvent(text="How is billing doing?", source=source, message_type=MessageType.TEXT)
    session_key = f"agent:main:{platform.value}:dm:c1"
    respond, aux = model or _model(Text(f"Audit finished: {FINDING}"))
    base = {
        "tools": {"tool_search": {"enabled": "off"}},
        "platform_toolsets": {platform.value: ["delegation", "todo"]},
    }
    # Tool progress lines are their own operator knob (display.tool_progress); a quiet chat turns them off.
    merged = {**base, **config, "display": {"tool_progress": "off", **config.get("display", {})}}
    with FakeLLMServer(respond, aux=aux) as srv:
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW
                          + yaml.safe_dump(merged))
        monkeypatch.setattr(gateway_run, "_hermes_home", home)
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
            "api_key": "sk-fake-e2e", "base_url": srv.base_url, "provider": "custom", "api_mode": "chat_completions"})
        runner = _make_runner(chat)
        runner.config.streaming = StreamingConfig(enabled=False)

        async def handler(ev):
            result = await runner._run_agent(message=ev.text, context_prompt="", history=[], source=source,
                                             session_id="s1", session_key=session_key)
            return None if result.get("already_sent") else result["final_response"]

        chat.set_message_handler(handler)
        turn = asyncio.ensure_future(chat._process_message_background(event, session_key))
        if during is not None:
            await during()
        await asyncio.wait_for(turn, timeout=60)
        # A background unit must settle before its provider goes away.
        await asyncio.to_thread(wait_until, lambda: async_delegation.active_count() == 0, 30, 0.02, "async units")
    parent_requests = [r["body"] for r in srv.requests
                       if r["kind"] == "main" and GOAL not in _first_user_text(r)]
    return chat, parent_requests


def _interim(value) -> dict:
    return {"display": {"interim_assistant_messages": value}}


@pytest.mark.asyncio
async def test_interim_off_joins_the_children_and_the_chat_gets_one_completed_answer(monkeypatch, dispatches):
    chat, requests = await _turn(monkeypatch, _interim(False))

    assert chat.sent == [ANSWER]  # one message, written after the child's result
    assert chat.typing  # the typing indicator is what the user sees while the children run
    delegate = _delegate_def(requests[0])
    assert END_TURN not in delegate["description"] and "BETWEEN your turns" not in delegate["description"]
    assert "results return in this tool call" in delegate["description"]
    assert not {"action", "subagent_id", "message"} & set(delegate["parameters"]["properties"])
    tool_result = requests[-1]["messages"][-1]
    assert tool_result["role"] == "tool" and FINDING in tool_result["content"]
    assert END_TURN not in tool_result["content"] and '"dispatched"' not in tool_result["content"]
    assert dispatches == []  # the async completion registry is never used


@pytest.mark.asyncio
async def test_interim_on_keeps_the_background_path_and_its_schema_byte_identical(monkeypatch, dispatches):
    import model_tools

    chat, requests = await _turn(monkeypatch, {})

    assert chat.sent == [PAUSE]  # today's behaviour: the status is the turn's answer
    assert len(dispatches) == 1
    delegate = _delegate_def(requests[0])
    assert END_TURN in delegate["description"]
    registry_delegate = next(t["function"] for t in model_tools.get_tool_definitions(
        enabled_toolsets=["delegation", "todo"], quiet_mode=True) if t["function"]["name"] == "delegate_task")
    assert delegate == registry_delegate
    assert '"dispatched"' in requests[-1]["messages"][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("config", "inline"), [
    ({**_interim(True), "delegation": {"gateway_mode": "inline"}}, True),
    ({**_interim(False), "delegation": {"gateway_mode": "background"}}, False),
    ({**_interim(False), "delegation": {"gateway_mode": "async"}}, False),
    ({**_interim(False), "delegation": {"gateway_mode": "auto"}}, True),
], ids=["inline-overrides-interim-on", "background-overrides-interim-off", "async-alias", "explicit-auto"])
async def test_an_explicit_gateway_mode_wins_over_the_interim_setting(monkeypatch, dispatches, config, inline):
    chat, requests = await _turn(monkeypatch, config)

    assert chat.sent == ([ANSWER] if inline else [PAUSE])
    assert (END_TURN in _delegate_def(requests[0])["description"]) is not inline
    assert len(dispatches) == (0 if inline else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(("platform", "inline"), [(Platform.WHATSAPP, True), (Platform.TELEGRAM, False)])
async def test_a_platform_interim_override_decides_for_that_platform_only(monkeypatch, dispatches, platform, inline):
    config = {"display": {"interim_assistant_messages": True,
                          "platforms": {"whatsapp": {"interim_assistant_messages": False}}}}
    chat, _ = await _turn(monkeypatch, config, platform=platform)

    assert chat.sent == ([ANSWER] if inline else [PAUSE])
    assert len(dispatches) == (0 if inline else 1)


@pytest.mark.asyncio
async def test_a_platform_override_can_turn_interim_back_on_under_a_global_off(monkeypatch, dispatches):
    config = {"display": {"interim_assistant_messages": False,
                          "platforms": {"whatsapp": {"interim_assistant_messages": True}}}}
    chat, _ = await _turn(monkeypatch, config)

    assert chat.sent == [PAUSE] and len(dispatches) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("hard", [True, False], ids=["stop-command", "interrupting-follow-up"])
async def test_stop_during_the_inline_wait_cancels_the_children(monkeypatch, dispatches, hard):
    """``/stop`` hard-interrupts the running parent and a follow-up message under the default
    ``busy_input_mode: interrupt`` soft-interrupts it; inline children are attached to the parent, so either
    reaches the child's in-flight request and the turn ends without waiting for the provider."""
    from tools import delegate_tool

    parents, stops = [], []
    real_build = delegate_tool._build_child_agent

    def recording_build(*args, **kwargs):
        child = real_build(*args, **kwargs)
        parents.append(kwargs.get("parent_agent"))
        for name in ("interrupt", "hard_interrupt"):  # instance attributes win the static lookup too
            real = getattr(child, name)
            setattr(child, name, lambda *a, _real=real, _name=name, **k: (stops.append(_name), _real(*a, **k))[1])
        return child

    monkeypatch.setattr(delegate_tool, "_build_child_agent", recording_build)
    child_waiting = threading.Event()
    respond, _ = _model(Text("unused"))

    def routed(request):
        if GOAL in _first_user_text(request):
            if not child_waiting.is_set():  # the first child's request never gets an answer
                child_waiting.set()
                return Hang(seconds=120)
            return Text(f"Audit finished: {FINDING}")
        return respond(request)

    async def stop_once_the_child_waits():
        assert await asyncio.to_thread(child_waiting.wait, 30), "the child never reached its provider"
        if hard:
            parents[0].interrupt(None, hard_cancel=True)  # what /stop does to the running agent
        else:
            parents[0].interrupt("actually, check payroll instead")  # a follow-up in interrupt mode

    started = time.monotonic()
    chat, _ = await _turn(monkeypatch, _interim(False), model=(routed, lambda req: Text("Fake summary.")),
                          during=stop_once_the_child_waits)

    assert stops  # the stop reached the child through the parent's interrupt fan-out
    assert time.monotonic() - started < 60  # the turn did not sit out the child's 120 s provider hang
    if hard:
        assert all(FINDING not in text for text in chat.sent)
    else:  # the follow-up then runs as the next turn, which delegates afresh and answers once
        assert chat.sent[-1] == ANSWER
    assert dispatches == []


def test_each_profile_resolves_its_own_mode_under_the_multiplexer(monkeypatch, tmp_path):
    """A -> B -> A through ``_profile_scope_for_source`` (the seam every turn enters): the mode comes from the
    routed profile's own config, never from whichever profile resolved last."""
    from agent import secret_scope
    from tools.delegate_tool_config import gateway_delegation_inline

    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(_interim(False)))
    (beta / "config.yaml").write_text(yaml.safe_dump(_interim(True)))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        runner = _make_runner(_Chat(Platform.WHATSAPP))
        runner.config.multiplex_profiles = True
        source_a = SessionSource(platform=Platform.WHATSAPP, chat_id="D1", chat_type="dm", profile="default")
        source_b = SessionSource(platform=Platform.WHATSAPP, chat_id="D2", chat_type="dm", profile="beta")
        seen = []
        for source in (source_a, source_b, source_a):
            with runner._profile_scope_for_source(source):  # the inputs a fresh agent's build resolves from
                disp = runner._run_agent_display_settings(source)
                seen.append(gateway_delegation_inline(
                    disp.user_config, disp.platform_key, interim_enabled=disp.interim_assistant_messages_enabled))
    finally:
        secret_scope.set_multiplex_active(prev)
    assert seen == [True, False, True]


async def _two_turns(monkeypatch, config: dict, *, first_heartbeat: bool, restart_between=False):
    """Two turns on one Telegram chat through a runner WITH an agent cache (the second turn reuses the first turn's
    agent) and a real SessionDB in the temp HERMES_HOME (the tools[] pin is persisted). ``restart_between`` drops
    the cache between the turns, so the second turn builds a fresh agent that restores the session's pin.
    Returns (chat, the delegate_task definition of each parent request, the persisted pin)."""
    import json

    import gateway.run as gateway_run
    from hermes_state import SessionDB
    from tools import async_delegation

    home = Path(os.environ["HERMES_HOME"])
    platform = Platform.TELEGRAM
    chat = _Chat(platform)
    source = SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1")
    session_key = f"agent:main:{platform.value}:dm:c1"
    respond, aux = _model(Text(f"Audit finished: {FINDING}"))
    merged = {"tools": {"tool_search": {"enabled": "off"}}, "platform_toolsets": {platform.value: ["delegation", "todo"]},
              **config, "display": {"tool_progress": "off", **config.get("display", {})}}
    with FakeLLMServer(respond, aux=aux) as srv:
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW
                          + yaml.safe_dump(merged))
        monkeypatch.setattr(gateway_run, "_hermes_home", home)
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
            "api_key": "sk-fake-e2e", "base_url": srv.base_url, "provider": "custom", "api_mode": "chat_completions"})
        runner = _make_runner(chat)
        runner.config.streaming = StreamingConfig(enabled=False)
        runner._session_db = SessionDB(home / "state.db")
        runner._session_db.create_session("s1", source=platform.value)  # the session store creates the row first
        runner._agent_cache, runner._agent_cache_lock = {}, threading.Lock()
        heartbeats, history = [first_heartbeat, False], []

        async def handler(ev):
            # The second turn continues the session (a fresh agent reads the session row, and its pin, only then).
            result = await runner._run_agent(message=ev.text, context_prompt="", history=list(history), source=source,
                                             session_id="s1", session_key=session_key,
                                             scheduled_heartbeat=heartbeats.pop(0))
            history.extend([{"role": "user", "content": ev.text},
                            {"role": "assistant", "content": result["final_response"]}])
            return None if result.get("already_sent") else result["final_response"]

        chat.set_message_handler(handler)
        for i in range(2):
            if i and restart_between:
                runner._agent_cache.clear()
            event = MessageEvent(text=f"How is billing doing? ({i})", source=source, message_type=MessageType.TEXT)
            await asyncio.wait_for(chat._process_message_background(event, session_key), timeout=60)
            await asyncio.to_thread(wait_until, lambda: async_delegation.active_count() == 0, 30, 0.02, "async units")
        row = runner._session_db.get_session("s1") or {}
        pin = json.loads(row["tool_names"]) if row.get("tool_names") else None
    schemas = [_delegate_def(r["body"]) for r in srv.requests if r["kind"] == "main" and GOAL not in _first_user_text(r)]
    return chat, schemas, pin


@pytest.mark.asyncio
@pytest.mark.parametrize(("config", "first_heartbeat", "inline"), [
    ({}, True, False),
    ({}, False, False),
    (_interim(False), True, True),
], ids=["interim-on-heartbeat-first", "interim-on-control", "interim-off-heartbeat-first"])
async def test_a_heartbeat_turn_does_not_decide_the_chats_mode(monkeypatch, dispatches, config, first_heartbeat, inline):
    """A scheduled heartbeat mutes interim messages for its own turn, and it runs under the user's session key: when
    it is the turn that builds the session's (cached) agent, the mode must still come from the chat's configured
    interim setting, or an interim-on chat runs inline for the rest of the session."""
    chat, schemas, _ = await _two_turns(monkeypatch, config, first_heartbeat=first_heartbeat)

    assert len(schemas) >= 2
    assert all((END_TURN in schema["description"]) is not inline for schema in schemas)
    assert len(dispatches) == (0 if inline else 2)
    assert chat.sent[-1] == (ANSWER if inline else PAUSE)


@pytest.mark.asyncio
async def test_a_heartbeat_built_agent_pins_the_configured_mode_across_a_restart(monkeypatch, dispatches):
    """Composed: agent cache + persisted tools[] pin + heartbeat. The heartbeat builds the session's agent on a cold
    cache, the gateway restarts, the user's next turn builds a fresh agent that restores the pin: it is the
    background schema an interim-on chat configured."""
    chat, schemas, pin = await _two_turns(monkeypatch, {}, first_heartbeat=True, restart_between=True)

    assert pin is not None
    delegate = next(t["function"] for t in pin["tools"] if t["function"]["name"] == "delegate_task")
    assert END_TURN in delegate["description"]
    assert all(END_TURN in schema["description"] for schema in schemas)
    assert len(dispatches) == 2 and chat.sent[-1] == PAUSE


@pytest.mark.parametrize("platform_key", ["api_server", "local", "webhook"])
def test_auto_never_joins_on_a_surface_that_is_not_a_chat(platform_key):
    """``auto`` is about a pause message reaching a person: API, local and webhook turns keep today's path."""
    from tools.delegate_tool_config import gateway_delegation_inline

    assert gateway_delegation_inline(_interim(False), platform_key, interim_enabled=False) is False
    assert gateway_delegation_inline({"delegation": {"gateway_mode": "inline"}}, platform_key, interim_enabled=True)
