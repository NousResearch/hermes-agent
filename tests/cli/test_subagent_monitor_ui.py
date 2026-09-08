"""Monitor navigation and controls run without borrowing the chat composer."""
import asyncio
from types import SimpleNamespace


def test_monitor_keys_navigate_steer_and_preserve_chat_draft(monkeypatch):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor, build_monitor_application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from tools import delegate_tool_registry as registry
    monkeypatch.setattr(registry, '_active_subagents', {})
    received = []
    child = SimpleNamespace(steer=lambda text: received.append(text) or True,
                            interrupt=lambda *a, **kw: received.append('STOP') or True)
    for sid in ['first', 'second']:
        registry._register_subagent(dict(subagent_id=sid, goal=sid, owner_agent_session_id='owner',
            started_at=1, status='running', agent=child))
    composer = Buffer()
    composer.text = 'my unfinished draft'
    dock = SubagentMonitor(SimpleNamespace(agent=SimpleNamespace(session_id='owner')))
    dock.refresh()

    async def run():
        with create_pipe_input() as pipe:
            app = build_monitor_application(dock, input=pipe, output=DummyOutput())
            task = asyncio.create_task(app.run_async())
            await asyncio.sleep(0.1)
            pipe.send_text('\x1b[B\rsFocus on tests\r')
            await asyncio.sleep(0.2)
            pipe.send_text('\x1b')
            await asyncio.sleep(0.6)
            pipe.send_text('q')
            await asyncio.wait_for(task, 3)
    asyncio.run(run())
    assert dock.selected_id == 'second'
    assert received == ['Focus on tests']
    assert composer.text == 'my unfinished draft'


def test_extended_tail_is_bounded_and_literal(tmp_path):
    from hermes_cli.cli_subagent_monitor import read_tail
    path = tmp_path / 'child.log'
    path.write_text('old data\n' * 20000 + '\x1b[2Jlast activity\n', encoding='utf-8')
    tail = read_tail(str(path))
    assert 'last activity' in tail and '\x1b' not in tail
    assert len(tail) <= 32768
    assert 'not available' in read_tail(str(tmp_path / 'missing.log'))
