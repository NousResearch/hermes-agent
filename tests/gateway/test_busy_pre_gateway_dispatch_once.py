"""``pre_gateway_dispatch`` fires exactly once per inbound message, also when it arrives mid-turn.

The cold path runs the hook in ``_hm_admit_event``. The shared runtime's busy text reaches that
same handler. Input the adapter hands to the busy-session handler instead (slash text that is no
command, and every busy input without a session authority) used to interrupt, steer or queue the
running turn before any plugin saw it, and a steered input never reached the hook at all. The busy
handler now runs the hook on arrival, before auth and before its plaintext-approval and skill-slash
routing (the cold path's order), and the drain of a queued arrival does not run it again.
Real GatewayRunner, adapter and busy routing. The agent turn is a stand-in, and
``hermes_cli.lifecycle.ainvoke_hook`` is replaced by a recorder that returns each test's plugin
result, so plugin registration itself is not exercised here.
"""
import asyncio

import pytest

from tests.gateway.test_shared_busy_receive_loop import Adapter, _wait_for


class Agent:
    def __init__(self):
        self.interrupts, self.steers = [], []

    def interrupt(self, message=None, *args, **kwargs):
        self.interrupts.append(message)

    def steer(self, text):
        self.steers.append(text)
        return True


async def _busy(monkeypatch, *, authority=False, hook=None, mode="interrupt", text_mode=None):
    """Start turn 'one' and hold it; return a helper that sends a follow-up while it runs.

    ``hook(text)`` gives the plugin's result list for each event; ``fired`` records the text of
    every event the hook saw, in order."""
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_results import execution_result

    fired = []

    async def invoke(name, **kwargs):
        if name != "pre_gateway_dispatch":
            return []
        fired.append(kwargs["event"].text)
        return hook(kwargs["event"].text) if hook else []

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", invoke)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    if authority:
        from gateway.session_authority import initialize_session_authority
        await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    runner._busy_input_mode = mode
    if text_mode:
        runner._busy_text_mode = text_mode
    agent, gate, ran = Agent(), asyncio.Event(), []

    async def turn(event, source, quick_key, run_generation):
        ran.append(event.text)
        if event.text == "one":
            runner._session_state(quick_key).turn.agent = agent
            await gate.wait()
        if authority:
            execution_result.get()["result"] = {"final_response": "reply to " + event.text, "messages": []}
        return "reply to " + event.text

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    source = adapter.build_source(chat_id="busy-chat", chat_type="dm", user_id="fixture-user")
    await adapter.handle_message(MessageEvent(text="one", source=source, message_id="m1"))
    await _wait_for(lambda: ran == ["one"])

    async def follow_up(text, *, user_id="fixture-user"):
        src = adapter.build_source(chat_id="busy-chat", chat_type="dm", user_id=user_id)
        await asyncio.wait_for(adapter.handle_message(MessageEvent(text=text, source=src, message_id="m2")), 5)

    async def finish(expected_ran):
        """Release turn 'one'; wait until the turns that should run have run and the adapter is idle."""
        gate.set()
        await _wait_for(lambda: ran == expected_ran and "reply to one" in adapter.sent
                        and not adapter._background_tasks and not adapter._active_sessions, 15)

    return follow_up, finish, fired, agent, adapter


@pytest.mark.asyncio
async def test_idle_message_fires_the_hook_once(monkeypatch):
    """Guard: the cold path already fires it once; the busy-path marker must not change that."""
    _follow_up, finish, fired, _agent, _adapter = await _busy(monkeypatch)
    assert fired == ["one"]
    await finish(["one"])
    assert fired == ["one"]


@pytest.mark.asyncio
async def test_busy_text_on_the_shared_runtime_fires_the_hook_once(monkeypatch):
    """Guard: the shared runtime's busy text already reaches ``_handle_message`` and its hook."""
    follow_up, finish, fired, _agent, _adapter = await _busy(monkeypatch, authority=True)
    await follow_up("two")
    assert fired == ["one", "two"]
    await finish(["one", "two"])
    assert fired == ["one", "two"]


@pytest.mark.asyncio
async def test_busy_slash_text_on_the_shared_runtime_fires_the_hook_once_on_arrival(monkeypatch):
    """Slash text that is no command skips the shared runtime's FIFO and goes to the busy handler,
    which interrupts the turn. The hook must see it then, and not again when it drains."""
    follow_up, finish, fired, agent, adapter = await _busy(monkeypatch, authority=True)
    await follow_up("/notacommand two")
    assert fired == ["one", "/notacommand two"], "the hook must run before the turn is interrupted"
    assert agent.interrupts
    await finish(["one"])
    await _wait_for(lambda: any("Unknown command" in m for m in adapter.sent), 15)  # it drained
    assert fired == ["one", "/notacommand two"]


@pytest.mark.asyncio
async def test_busy_interrupt_fires_the_hook_once_on_arrival_not_again_at_drain(monkeypatch):
    follow_up, finish, fired, agent, _adapter = await _busy(monkeypatch)
    await follow_up("two")
    assert fired == ["one", "two"] and agent.interrupts == ["two"]
    await finish(["one", "two"])
    assert fired == ["one", "two"], "the queued arrival drained through _handle_message ran the hook again"


@pytest.mark.asyncio
async def test_busy_steer_fires_the_hook_once(monkeypatch):
    """A steered arrival goes into the running turn and is never queued, so it has no later
    chance to reach the hook."""
    follow_up, finish, fired, agent, _adapter = await _busy(monkeypatch, mode="steer")
    await follow_up("two")
    assert fired == ["one", "two"] and agent.steers
    await finish(["one"])
    assert fired == ["one", "two"]


@pytest.mark.asyncio
async def test_busy_queued_rewrite_fires_the_hook_once_and_the_rewrite_runs(monkeypatch):
    """Queue-mode text goes back to the adapter's own queue as the same event, so the rewrite must
    land on that event and its drain must not run the hook a second time."""
    follow_up, finish, fired, agent, _adapter = await _busy(
        monkeypatch, text_mode="queue",
        hook=lambda text: [{"action": "rewrite", "text": "rewritten two"}] if text == "two" else [])
    await follow_up("two")
    assert fired == ["one", "two"] and not agent.interrupts
    await finish(["one", "rewritten two"])
    assert fired == ["one", "two"]


@pytest.mark.asyncio
async def test_busy_skip_drops_the_arrival_before_it_interrupts(monkeypatch):
    follow_up, finish, fired, agent, adapter = await _busy(
        monkeypatch, hook=lambda text: [{"action": "skip", "reason": "handover"}] if text == "two" else [])
    await follow_up("two")
    assert fired == ["one", "two"]
    assert agent.interrupts == [] and adapter.sent == [], "a skipped arrival must not interrupt or ack"
    await finish(["one"])
    assert fired == ["one", "two"] and adapter.sent == ["reply to one"]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [[{"action": "skip"}], []], ids=["skip", "no-action"])
async def test_busy_hook_runs_before_auth_and_auth_still_drops_the_rest(monkeypatch, result):
    """As on the cold path the hook sees an unauthorized sender (a plugin may take it over);
    without a skip the auth gate still drops it."""
    follow_up, finish, fired, agent, adapter = await _busy(
        monkeypatch, hook=lambda text: result if text == "two" else [])
    await follow_up("two", user_id="stranger")
    assert fired == ["one", "two"]
    assert agent.interrupts == [] and adapter.sent == []
    await finish(["one"])
    assert fired == ["one", "two"]


@pytest.mark.asyncio
@pytest.mark.parametrize("skip", [True, False], ids=["skip", "allow"])
async def test_busy_plaintext_approval_reply_passes_the_hook_first(monkeypatch, skip):
    """A bare "yes" answering a pending dangerous-command approval mid-turn meets the hook first,
    as on the cold path: a plugin skip swallows it (the approval stays pending), an allow resolves it."""
    from gateway.platforms.event import MessageEvent
    from tools.approval import _gateway_queues
    from tools.approval_gateway_wait import _ApprovalEntry

    _follow_up, finish, fired, agent, adapter = await _busy(
        monkeypatch, hook=lambda text: [{"action": "skip"}] if skip and text == "yes" else [])
    source = adapter.build_source(chat_id="busy-chat", chat_type="dm", user_id="fixture-user")
    key = adapter.gateway_runner._session_key_for_source(source)
    entry = _ApprovalEntry({"command": "echo fixture"})  # never executed: only its wait event is read
    _gateway_queues.setdefault(key, []).append(entry)
    try:
        await adapter.handle_message(MessageEvent(text="yes", source=source, message_id="approval-reply"))
        resolved = entry.event.is_set()
    finally:
        _gateway_queues.pop(key, None)
    assert resolved is not skip, "skip leaves the approval pending; allow resolves it"
    assert fired == ["one", "yes"] and agent.interrupts == []
    await finish(["one"])
    assert fired == ["one", "yes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("skip", [True, False], ids=["skip", "allow"])
async def test_busy_skill_slash_passes_the_hook_before_it_expands(monkeypatch, tmp_path, skip):
    """A mid-turn /<skill> on the shared runtime goes to the busy handler (it is no gateway command)
    and meets the hook before the skill expands: a skip drops it, an allow expands and runs it."""
    from agent import skill_commands
    from gateway.platforms.event import MessageEvent

    skill = tmp_path / "skills" / "fixture-guide" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: fixture-guide\ndescription: fixture\n---\nFixture skill body\n")
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    _follow_up, finish, fired, _agent, adapter = await _busy(
        monkeypatch, authority=True,
        hook=lambda text: [{"action": "skip"}] if skip and text.startswith("/fixture-guide") else [])
    event = MessageEvent(text="/fixture-guide do it", source=adapter.build_source(
        chat_id="busy-chat", chat_type="dm", user_id="fixture-user"), message_id="skill")
    await asyncio.wait_for(adapter.handle_message(event), 5)
    expanded = "Fixture skill body" in event.text
    assert expanded is not skip, "skip drops the skill slash; allow expands it"
    assert fired == ["one", "/fixture-guide do it"]
    await finish(["one"] if skip else ["one", event.text])
    assert fired == ["one", "/fixture-guide do it"]
