"""The live-work footer under the composer: Down reaches it, Enter opens a row, Esc leaves, and a
gateway-backed CLI lists the children its gateway runs (the in-process registry is empty there)."""
import asyncio
from types import SimpleNamespace

from agent.i18n import t


def _footer_app(cli, editor, pipe, output):
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.layout import HSplit, Layout
    from hermes_cli.cli_dock_footer import bind_keys

    kb = KeyBindings()
    bind_keys(cli, kb)
    return Application(layout=Layout(HSplit([editor, cli._subagent_dock_widget]), focused_element=editor),
                       key_bindings=merge_key_bindings([editor.control.key_bindings or KeyBindings(), kb]),
                       input=pipe, output=output)


def test_down_on_an_empty_composer_selects_rows_and_enter_opens_that_rows_log(monkeypatch):
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.widgets import TextArea
    from hermes_cli import cli_subagent_monitor as dock
    from hermes_cli.cli_dock_footer import selecting
    from tests.waiting import async_wait_until

    opened = []
    monkeypatch.setattr(dock, 'open_monitor', lambda cli, key=None: opened.append(key))
    cli = SimpleNamespace(agent=None)
    dock.install_dock(cli)
    monitor = cli._subagent_monitor
    monitor.entries = [dict(subagent_id=f'sa-{i}', key=f'sa-{i}', goal=f'Worker {i}', elapsed=3,
                            last_tool='terminal', status='running') for i in range(2)]
    monitor._last_poll = float('inf')
    editor = TextArea(height=1, multiline=False)
    output = DummyOutput()
    output.get_size = lambda: Size(rows=30, columns=100)

    async def run():
        with create_pipe_input() as pipe:
            app = _footer_app(cli, editor, pipe, output)
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                await async_wait_until(lambda: app.is_running, what='app running')
                pipe.send_text('\x1b[B')  # Down: into the footer, first row selected
                await async_wait_until(lambda: monitor.footer_active, what='footer selected')
                assert monitor.selected_id == 'sa-0'
                assert t('cli.subagents.select_hint') in monitor.dock_text(columns=100, rows=30, selecting=True)
                pipe.send_text('\x1b[B')
                await async_wait_until(lambda: monitor.selected_id == 'sa-1', what='second row')
                pipe.send_text('\r')  # Enter opens the selected row's live log, never submits
                await async_wait_until(lambda: opened, what='row opened')
                assert opened == ['sa-1'] and not monitor.footer_active and editor.text == ''
                pipe.send_text('\x1b[B')
                await async_wait_until(lambda: monitor.footer_active, what='footer reselected')
                pipe.send_text('\x1b[A')  # Up past the first row returns to the composer
                await async_wait_until(lambda: not monitor.footer_active, what='footer left')
                pipe.send_text('\x1b[Bh')  # typing after Down hands the arrows back
                await async_wait_until(lambda: editor.text == 'h', what='typed')
                assert not selecting(cli) and not monitor.footer_active
            finally:
                app.exit()
                await task
    asyncio.run(run())


def test_footer_stop_asks_twice_before_stopping_the_selected_row(monkeypatch):
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli.cli_subagent_monitor import SubagentMonitor

    cli = SimpleNamespace(agent=None)
    monitor = cli._subagent_monitor = SubagentMonitor(cli)
    monitor.entries = [dict(subagent_id='sa-1', key='sa-1', goal='Worker', elapsed=1, status='running')]
    calls = []
    monkeypatch.setattr(monitor, 'control', lambda action, message=None, *, target=None:
                        calls.append((action, target)) or {'status': 'interrupt_requested', 'note': 'stopping'})
    footer.enter(cli)
    footer.stop_selected(cli)
    assert calls == [] and monitor.confirm_key == 'sa-1'
    assert t('cli.subagents.confirm_stop', noun=t('cli.subagents.noun_subagent')) in footer.plain_text(
        footer.dock_lines(monitor, columns=120, rows=30, selecting=True))
    footer.stop_selected(cli)
    assert calls == [('stop', 'sa-1')] and monitor.notice == 'stopping'


def test_gateway_backed_dock_lists_the_sessions_children_and_routes_controls_to_the_gateway(monkeypatch):
    from hermes_cli import gateway_tools
    from hermes_cli.cli_subagent_monitor import SubagentMonitor

    calls = []

    class Connection:
        def call(self, method, **params):
            calls.append((method, params))
            if method == 'subagent.list':
                return {'subagents': [{'subagent_id': 'sa-9', 'goal': 'Check the build', 'status': 'running',
                                       'started_at': 100.0, 'last_tool': 'terminal', 'live_transcript': ''}]}
            if method == 'subagent.interrupt':
                return {'found': True}
            if method == 'subagent.tail':
                return {'available': True, 'text': 'child output'}
            return {}

        def state(self, session_id):
            return SimpleNamespace(usage={'active_subagents': 1})

    view = gateway_tools.GatewayProcessView(SimpleNamespace(connection=Connection(), session_id='sess-1'))
    monkeypatch.setattr(gateway_tools, 'process_view', lambda: view)
    monkeypatch.setattr('hermes_cli.cli_process_dock.process_rows', lambda now: [])

    monitor = SubagentMonitor(SimpleNamespace(agent=None))
    assert monitor.refresh(now=112)
    assert [(r['subagent_id'], r['elapsed']) for r in monitor.entries] == [('sa-9', 12)]
    assert ('subagent.list', {'session_id': 'sess-1', 'timeout': 5}) in calls
    text = monitor.dock_text(columns=100, rows=30)
    assert t('cli.subagents.pill_agents_one', count=1) in text and 'Check the build' in text

    from hermes_cli.cli_subagent_monitor import subagent_tail
    assert subagent_tail(monitor.entries[0], {}) == 'child output'
    monitor.selected_id = 'sa-9'
    assert monitor.control('stop')['status'] == 'interrupt_requested'
    assert ('subagent.interrupt', {'session_id': 'sess-1', 'subagent_id': 'sa-9'}) in calls


def test_watched_background_processes_count_as_monitors():
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli.cli_subagent_monitor import SubagentMonitor

    monitor = SubagentMonitor(SimpleNamespace(agent=None))
    monitor.processes = [
        dict(kind='process', key='p1', id='p1', command='tail -f app.log', status='running',
             elapsed=4, detail='GET /', watching=True),
        dict(kind='process', key='p2', id='p2', command='npm run build', status='running',
             elapsed=9, detail='', watching=False)]
    text = monitor.dock_text(columns=100, rows=30)
    assert t('cli.subagents.pill_shells_one', count=1) in text
    assert t('cli.subagents.pill_monitors_one', count=1) in text
    assert f'{footer.MONITOR_GLYPH} tail -f app.log' in text and f'{footer.SHELL_GLYPH} npm run build' in text

