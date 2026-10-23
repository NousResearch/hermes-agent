"""Monitor navigation keeps controls pinned while live children finish."""
import asyncio
from types import SimpleNamespace


def test_monitor_keys_pin_controls_across_roster_changes(monkeypatch):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor, build_monitor_application

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

    dock = SubagentMonitor(SimpleNamespace(agent=SimpleNamespace(session_id='owner')))
    dock.refresh()

    async def run():
        with create_pipe_input() as pipe:
            output = DummyOutput()
            from prompt_toolkit.data_structures import Size
            output.get_size = lambda: Size(rows=14, columns=32)
            app = build_monitor_application(dock, input=pipe, output=output)
            footer = app.layout.container.children[-1].content.text()
            assert 'Ctrl+T' in footer and len(footer) <= 32
            rendered = asyncio.Event()
            app.after_render += lambda app: rendered.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(rendered.wait(), 3)

            async def send(text):
                rendered.clear()
                pipe.send_text(text)
                await asyncio.wait_for(rendered.wait(), 3)

            await send('\x1b[B\rsFocus on tests\r')
            await send('x')
            registry._active_subagents.pop('second')
            dock.refresh()
            await send('y')
            await send('\x1b')
            pipe.send_text('\x14')
            await asyncio.wait_for(task, 3)
    asyncio.run(run())
    assert dock.selected_id == 'first'
    assert received == ['Focus on tests']



def test_extended_tail_is_bounded_and_literal(tmp_path):
    from hermes_cli.cli_subagent_monitor import read_tail
    path = tmp_path / 'child.log'
    path.write_text('old data\n' * 20000 + '\x1b[2Jlast activity\n', encoding='utf-8')
    tail = read_tail(str(path))
    assert 'last activity' in tail and '\x1b' not in tail
    assert len(tail) <= 32768
    assert 'not available' in read_tail(str(tmp_path / 'missing.log'))


def test_a_stalled_steer_never_freezes_the_monitor(monkeypatch):
    """Steer runs off the UI loop: while the backend call hangs, the monitor still repaints and Esc
    still closes it; the answer lands in the notice line when it arrives."""
    import threading
    from hermes_cli.cli_subagent_monitor import SubagentMonitor, build_monitor_application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from tools import delegate_tool_registry as registry
    monkeypatch.setattr(registry, '_active_subagents', {})
    registry._register_subagent(dict(subagent_id='only', goal='only', owner_agent_session_id='owner',
                                     started_at=1, status='running', agent=SimpleNamespace()))
    dock = SubagentMonitor(SimpleNamespace(agent=SimpleNamespace(session_id='owner')))
    dock.refresh()
    release, entered = threading.Event(), threading.Event()

    def stalled(action, message=None, *, target=None):
        entered.set()
        release.wait(5)
        return {'note': 'steered late'}
    monkeypatch.setattr(dock, 'control', stalled)

    async def run():
        with create_pipe_input() as pipe:
            app = build_monitor_application(dock, input=pipe, output=DummyOutput())
            rendered = asyncio.Event()
            app.after_render += lambda app: rendered.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(rendered.wait(), 3)
            rendered.clear()
            pipe.send_text('sFocus\r')
            await asyncio.wait_for(rendered.wait(), 3)  # repainted while the call is still hanging
            await asyncio.get_running_loop().run_in_executor(None, entered.wait, 3)
            assert entered.is_set() and not release.is_set()
            rendered.clear()
            pipe.send_text('\x1b')  # Esc still closes it mid-call
            await asyncio.wait_for(rendered.wait(), 3)
            pipe.send_text('\x1b')
            await asyncio.wait_for(task, 3)
            release.set()
    asyncio.run(run())
