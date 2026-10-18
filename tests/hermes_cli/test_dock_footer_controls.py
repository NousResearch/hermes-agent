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
                await async_wait_until(lambda: monitor.notice == 'guidance queued', what='steer outcome')
                assert sent == [('steer', 'sa-1', 'use the staging db')]
                assert monitor.steer_target is None
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


def _racing_monitor(rows, *, on_read):
    """A monitor whose ``on_read`` attribute runs ``finish_first_row`` on that attribute's Nth read:
    the spinner thread's refresh landing in the middle of a footer key handler."""
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    attr, nth = on_read

    class Racing(SubagentMonitor):
        reads = 0
        armed = False

        def __getattribute__(self, name):
            if name == attr and object.__getattribute__(self, 'armed'):
                self.reads += 1
                if self.reads == nth:
                    self.finish_first_row()
            return object.__getattribute__(self, name)

        def finish_first_row(self):
            # What refresh() does when the selected row finishes: select the next one and drop
            # the pending confirmation / steer input.
            self.armed = False
            self.entries = [r for r in self.entries if r['key'] != 'sa-0']
            self.processes = [r for r in self.processes if r['key'] != 'sh-0']
            self.selected_id = self._key(self.roster[0])
            self.confirm_key = None
            self.steer_target, self.steer_draft = None, ''

    cli = SimpleNamespace(agent=None)
    monitor = cli._subagent_monitor = Racing(cli)
    monitor.entries = [r for r in rows if r.get('kind') != 'process']
    monitor.processes = [r for r in rows if r.get('kind') == 'process']
    return cli, monitor


def test_a_row_finishing_mid_stop_or_mid_steer_never_redirects_the_control_to_the_next_row(monkeypatch):
    """The stop / steer goes to the row it was armed on, read once; a refresh that finishes that
    row and selects the next one meanwhile never redirects it, and a control whose row is gone
    reports that instead of falling back to whatever row is selected now."""
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli import cli_process_dock as procs
    from tools import delegate_tool_registry as registry
    from tests.waiting import wait_until

    killed, steered = [], []
    monkeypatch.setattr(procs, 'kill', lambda key: killed.append(key) or {'status': 'killed'})
    monkeypatch.setattr(registry, '_handle_control_action', lambda action, sid, message, parent:
                        steered.append((action, sid)) or '{"status": "queued"}')

    # The refresh lands as the confirmed stop is dispatched: sh-0 is gone, sh-1 is selected.
    cli, monitor = _racing_monitor([_process('sh-0'), _process('sh-1')], on_read=('control', 1))
    footer.enter(cli)
    footer.stop_selected(cli)  # armed on sh-0
    monitor.armed = True
    footer.stop_selected(cli)
    wait_until(lambda: monitor.notice, what='stop outcome')
    assert killed == [] and monitor.notice == t('cli.subagents.entry_gone')

    # The refresh closes the steer input (its child finished) as Enter reads the target.
    cli, monitor = _racing_monitor([_agent(0), _agent(1)], on_read=('steer_target', 1))
    footer.enter(cli)
    footer.start_steer(cli)
    footer.steer_type(cli, 'use the staging db')
    monitor.armed = True
    footer.steer_send(cli)
    assert steered == []

    # No target left: an error notice, never the selected row.
    assert monitor.control('stop', target=None) == {'error': t('cli.subagents.entry_gone')}
    assert monitor.control('steer', 'x', target='sa-0') == {'error': t('cli.subagents.entry_gone')}
    assert steered == [] and killed == []


def test_footer_controls_and_monitor_tails_never_wait_for_the_gateway(monkeypatch):
    """A slow ``subagent.tail`` / stop RPC must not block painting or keys: the call returns what
    is cached at once, and the result lands (with a repaint) when the worker finishes."""
    import threading
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli import gateway_tools
    from hermes_cli.cli_subagent_monitor import subagent_tail
    from tests.waiting import wait_until

    release = threading.Event()
    view = SimpleNamespace(subagent_tail=lambda sid: release.wait(5) and 'child output')
    monkeypatch.setattr(gateway_tools, 'process_view', lambda: view)
    repaints, cache = [], {}
    row = dict(subagent_id='sa-9', live_transcript='')
    assert subagent_tail(row, cache, on_update=lambda: repaints.append(1)) != 'child output'
    assert subagent_tail(row, cache) != 'child output'  # still fetching: no second RPC, no wait
    release.set()
    wait_until(lambda: repaints, what='tail fetched')
    assert subagent_tail(row, cache) == 'child output'

    cli, monitor = _monitor_with_rows()
    cli._invalidate = lambda *a, **k: repaints.append('footer')
    stopping = threading.Event()
    monkeypatch.setattr(monitor, 'control', lambda action, message=None, *, target=None:
                        stopping.wait(5) and {'status': 'interrupt_requested', 'note': 'stopping'})
    footer.enter(cli)
    footer.stop_selected(cli)
    footer.stop_selected(cli)  # sent: returns while the stop is still in flight
    assert monitor.notice == '' and not stopping.is_set()
    stopping.set()
    wait_until(lambda: monitor.notice == 'stopping', what='stop outcome')
    wait_until(lambda: 'footer' in repaints, what='footer repainted with the outcome')


def test_a_long_steer_draft_renders_its_tail_in_one_pass():
    """A huge paste into the steer input shows the tail that fits, without re-measuring the
    whole draft once per dropped character (a 12k paste took seconds per paint)."""
    import time
    from prompt_toolkit.utils import get_cwidth
    from hermes_cli import cli_dock_footer as footer
    cli, monitor = _monitor_with_rows()
    footer.enter(cli)
    footer.start_steer(cli)
    footer.steer_type(cli, 'ab界' * 20000 + 'END')
    started = time.monotonic()
    lines = footer.dock_lines(monitor, columns=80, rows=30, selecting=True)
    assert time.monotonic() - started < 2
    steer = lines[-1]
    draft, room = steer[1][1], 79 - get_cwidth(steer[0][1]) - 1
    assert draft.endswith('END') and monitor.steer_draft.endswith(draft)
    assert room - 1 <= get_cwidth(draft) <= room  # the widest tail that fits (a wide char may not)
