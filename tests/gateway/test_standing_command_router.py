import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.config import Platform


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("what's next?", "next"),
        ("Whats next", "next"),
        ("where are we at?", "next"),
        ("status", "status"),
        ("reflect", "reflect"),
    ],
)
def test_exact_standing_command_triggers(text, name):
    from gateway.standing_command_router import resolve_standing_command

    route = resolve_standing_command(text)

    assert route is not None
    assert route.name == name
    assert route.mode == "exact"


def test_adjacent_phrase_requires_confirmation_not_autofire():
    from gateway.standing_command_router import resolve_standing_command

    route = resolve_standing_command("what should we adapt next?")

    assert route is not None
    assert route.name == "next"
    assert route.mode == "confirm"


@pytest.mark.parametrize("text", ["nice", "make it so", "/status", "status report on the repo"])
def test_unrelated_or_slash_text_does_not_route(text):
    from gateway.standing_command_router import resolve_standing_command

    assert resolve_standing_command(text) is None


def _make_runner_for_gateway_router():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized = lambda _source: True
    runner._session_key_for_source = lambda _source: "telegram:12345:u1"
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._update_prompt_pending = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = {}
    runner.pairing_store = object()
    runner.session_store = object()

    async def emit_collect(*_args, **_kwargs):
        return []

    runner.hooks = type("Hooks", (), {"emit_collect": staticmethod(emit_collect)})()
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._begin_session_run_generation = lambda _key: 1
    runner._release_running_agent_state = lambda _key: runner._running_agents.pop(_key, None)
    return runner


def _make_event(text):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="u1",
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.mark.asyncio
async def test_gateway_exact_status_routes_without_agent(monkeypatch):
    runner = _make_runner_for_gateway_router()
    event = _make_event("status")

    async def fake_status(_event):
        return "STATUS ROUTED"

    async def fail_agent(*_args, **_kwargs):
        raise AssertionError("standing command should not reach the agent")

    monkeypatch.setattr(runner, "_handle_status_command", fake_status)
    monkeypatch.setattr(runner, "_handle_message_with_agent", fail_agent)

    result = await runner._handle_message(event)

    assert result == "STATUS ROUTED"


@pytest.mark.asyncio
async def test_gateway_exact_next_rewrites_before_agent(monkeypatch):
    runner = _make_runner_for_gateway_router()
    event = _make_event("what's next?")
    seen = {}

    async def fake_agent(ev, *_args, **_kwargs):
        seen["text"] = ev.text
        return "NEXT ROUTED"

    monkeypatch.setattr(runner, "_handle_message_with_agent", fake_agent)

    result = await runner._handle_message(event)

    assert result == "NEXT ROUTED"
    assert seen["text"].startswith("Standing command: what's next / where are we at.")
    assert "STATE.md" in seen["text"]


@pytest.mark.asyncio
async def test_gateway_adjacent_next_returns_confirmation(monkeypatch):
    runner = _make_runner_for_gateway_router()
    event = _make_event("what should we adapt next?")

    async def fail_agent(*_args, **_kwargs):
        raise AssertionError("confirmation route should not reach the agent")

    monkeypatch.setattr(runner, "_handle_message_with_agent", fail_agent)

    result = await runner._handle_message(event)

    assert "Reply exactly" in result
    assert "what’s next?" in result


@pytest.mark.asyncio
async def test_gateway_busy_plain_status_does_not_interrupt_agent(monkeypatch):
    runner = _make_runner_for_gateway_router()
    event = _make_event("status")
    runner._running_agents["telegram:12345:u1"] = object()
    interrupted = []

    class RunningAgent:
        def interrupt(self, text):
            interrupted.append(text)

    runner._running_agents["telegram:12345:u1"] = RunningAgent()
    runner._running_agents_ts["telegram:12345:u1"] = 1

    async def fake_status(_event):
        return "BUSY STATUS ROUTED"

    monkeypatch.setattr(runner, "_handle_status_command", fake_status)

    result = await runner._handle_message(event)

    assert result == "BUSY STATUS ROUTED"
    assert interrupted == []
