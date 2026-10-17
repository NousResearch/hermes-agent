"""Live-work footer controls: a click on a pill badge selects that kind's first row, Tab hops between
kinds, and ``s`` steers the selected subagent from a one-line input without touching the composer."""
import asyncio
from types import SimpleNamespace

from agent.i18n import t


def _agent(i):
    return dict(subagent_id=f'sa-{i}', key=f'sa-{i}', goal=f'Worker {i}', elapsed=3,
                last_tool='terminal', status='running')


def _process(key, *, watching=False, status='running'):
    return dict(kind='process', key=key, id=key, command=f'cmd {key}', status=status, elapsed=4,
                detail='', watching=watching, exit_code=0 if status != 'running' else None)


def _monitor_with_rows():
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    cli = SimpleNamespace(agent=None)
    monitor = cli._subagent_monitor = SubagentMonitor(cli)
    monitor.entries = [_agent(0), _agent(1)]
    monitor.processes = [_process('sh-1'), _process('mon-1', watching=True), _process('sh-2')]
    return cli, monitor


def _click(handler):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    return handler(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))


def _badge_handler(monitor, actions, label):
    from hermes_cli import cli_dock_footer as footer
    pill = footer.dock_lines(monitor, columns=160, rows=30, actions=actions)[0]
    return next(fragment[2] for fragment in pill if label in fragment[1])


def test_each_pill_badge_click_selects_the_first_row_of_its_kind():
    from hermes_cli import cli_dock_footer as footer
    cli, monitor = _monitor_with_rows()
    actions = footer.click_actions(cli)
    badges = {'sa-0': t('cli.subagents.pill_agents_other', count=2),
              'sh-1': t('cli.subagents.pill_shells_other', count=2),
              'mon-1': t('cli.subagents.pill_monitors_one', count=1)}
    for expected, label in badges.items():
        footer.leave(cli)
        _click(_badge_handler(monitor, actions, label))
        assert monitor.footer_active and monitor.selected_id == expected, label


def test_clicking_the_pill_outside_a_badge_still_toggles_selection():
    from hermes_cli import cli_dock_footer as footer
    cli, monitor = _monitor_with_rows()
    actions = footer.click_actions(cli)
    pill = footer.dock_lines(monitor, columns=160, rows=30, actions=actions)[0]
    hint = next(fragment for fragment in pill if fragment[0] == 'class:live-dock.hint')
    _click(hint[2])
    assert monitor.footer_active and monitor.selected_id == monitor._key(monitor.roster[0])
    _click(hint[2])
    assert not monitor.footer_active


def test_tab_visits_each_kind_present_in_pill_order_and_wraps():
    from hermes_cli import cli_dock_footer as footer
    cli, monitor = _monitor_with_rows()
    footer.enter(cli)
    seen = [monitor.selected_id]
    for _ in range(3):
        footer.next_kind(cli)
        seen.append(monitor.selected_id)
    assert seen == ['sa-0', 'sh-1', 'mon-1', 'sa-0']
    footer.next_kind(cli, -1)
    assert monitor.selected_id == 'mon-1'


def test_select_hint_offers_steer_only_on_a_subagent_row():
    from hermes_cli import cli_dock_footer as footer
    cli, monitor = _monitor_with_rows()
    footer.enter(cli, 'sa-0')
    text = monitor.dock_text(columns=160, rows=30, selecting=True)
    assert t('cli.subagents.select_hint_steer') in text
    footer.enter(cli, 'sh-1')
    text = monitor.dock_text(columns=160, rows=30, selecting=True)
    assert t('cli.subagents.select_hint') in text and t('cli.subagents.select_hint_steer') not in text


def test_s_on_a_process_row_explains_it_cannot_be_steered():
    from hermes_cli import cli_dock_footer as footer
    cli, monitor = _monitor_with_rows()
    footer.enter(cli, 'sh-1')
    footer.start_steer(cli)
    assert monitor.steer_target is None
    assert t('cli.subagents.cannot_steer_process') in monitor.dock_text(columns=200, rows=30, selecting=True)


def test_steer_input_sends_typed_guidance_to_the_selected_subagent_and_leaves_the_composer_alone(monkeypatch):
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.layout import HSplit, Layout
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.widgets import TextArea
    from hermes_cli import cli_subagent_monitor as dock
    from hermes_cli.cli_dock_footer import bind_keys
    from tests.waiting import async_wait_until

    cli = SimpleNamespace(agent=None)
    dock.install_dock(cli)
    monitor = cli._subagent_monitor
    monitor.entries = [_agent(0), _agent(1)]
    monitor._last_poll = float('inf')
    sent = []
    monkeypatch.setattr(monitor, 'control', lambda action, message=None, *, target=None:
                        sent.append((action, target, message)) or {'status': 'queued', 'note': 'guidance queued'})
    editor = TextArea(height=1, multiline=False)
    output = DummyOutput()
    output.get_size = lambda: Size(rows=30, columns=120)

    async def run():
        with create_pipe_input() as pipe:
            kb = KeyBindings()
            bind_keys(cli, kb)
            app = Application(layout=Layout(HSplit([editor, cli._subagent_dock_widget]), focused_element=editor),
                              key_bindings=merge_key_bindings([editor.control.key_bindings or KeyBindings(), kb]),
                              input=pipe, output=output)
            task = asyncio.create_task(app.run_async())
            try:
                await async_wait_until(lambda: app.is_running, what='app running')
                pipe.send_text('\x1b[B\x1b[B')  # into the footer, then the second agent
                await async_wait_until(lambda: monitor.selected_id == 'sa-1', what='second agent')
                pipe.send_text('s')
                await async_wait_until(lambda: monitor.steer_target == 'sa-1', what='steer input open')
                pipe.send_text('use the stagx')
                await async_wait_until(lambda: monitor.steer_draft == 'use the stagx', what='typed')
                assert t('cli.subagents.steer_prompt').strip() in monitor.dock_text(
                    columns=120, rows=30, selecting=True)
                pipe.send_text('\x7fing db\r')  # Backspace, more text, Enter sends
                await async_wait_until(lambda: sent, what='steer sent')
                assert sent == [('steer', 'sa-1', 'use the staging db')]
                assert monitor.steer_target is None and monitor.notice == 'guidance queued'
                assert editor.text == '' and monitor.footer_active
                pipe.send_text('s')
                await async_wait_until(lambda: monitor.steer_target == 'sa-1', what='steer reopened')
                pipe.send_text('never sent')
                await async_wait_until(lambda: monitor.steer_draft == 'never sent', what='typed again')
                pipe.send_text('\x1b')  # Esc cancels the input but keeps the row selected
                await async_wait_until(lambda: monitor.steer_target is None, what='steer cancelled')
                assert len(sent) == 1 and monitor.footer_active and editor.text == ''
            finally:
                app.exit()
                await task
    asyncio.run(run())


def test_idle_pill_suggests_mouse_only_while_clicks_are_off_and_room_allows():
    from hermes_cli import cli_dock_footer as footer
    cli, monitor = _monitor_with_rows()
    mouse_hint = t('cli.subagents.hint_mouse')
    cli._mouse_enabled = False
    assert mouse_hint in footer.plain_text(footer.dock_lines(monitor, columns=200, rows=30))
    # Just wide enough for the regular hint: the mouse suggestion is the first thing to go.
    pill = footer.plain_text(footer.dock_lines(monitor, columns=200, rows=30)[:1])
    regular = t('cli.subagents.pill_hint', toggle=t('cli.subagents.hint_hide'))
    fits_regular_only = len(pill) - len(mouse_hint) - 3 + 1
    narrow = footer.plain_text(footer.dock_lines(monitor, columns=fits_regular_only, rows=30)[:1])
    assert mouse_hint not in narrow and regular in narrow
    cli._mouse_enabled = True
    assert mouse_hint not in footer.plain_text(footer.dock_lines(monitor, columns=200, rows=30))
