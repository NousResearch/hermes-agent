"""Classic CLI live-work dock (subagents, background processes, active goal, standing loop, queued
prompts) and scoped controls; no agent-loop state is changed. Process rows come from
``cli_process_dock``, goal/loop/queue rows from ``cli_session_dock``."""
from __future__ import annotations

import json
import logging
import time

from prompt_toolkit.utils import get_cwidth

from agent.i18n import t
from hermes_cli import cli_process_dock as procs
from hermes_cli import cli_session_dock as session_rows


def _clip(value, width):
    text = ' '.join(str(value or '').split())
    text = ''.join(c for c in text if c.isprintable())
    if get_cwidth(text) <= width:
        return text
    result = ''
    for char in text:
        if get_cwidth(result + char) > max(0, width - 1):
            break
        result += char
    return result + ('…' if width else '')


class SubagentMonitor:
    def __init__(self, cli):
        self.cli = cli
        self.entries = []
        self.processes = []
        self.goal = ''
        self.loop = ''
        self.loop_badge = ''
        self.loop_prompt = ''
        self.queued = []
        self.selected_id = None
        self._signature = None
        self._last_poll = 0
        self.app = None
        self.opening = False
        self.collapsed = False
        # Footer selection (``cli_dock_footer``): Down on an empty composer, Esc/Up leaves.
        self.footer_active = False
        self.confirm_key = None
        self.confirm_action = None  # a plugin row action waiting for its second key press
        self.notice = ''
        # The footer's one-line steer input (``s``): target row key and the typed draft.
        self.steer_target = None
        self.steer_draft = ''
        # Plugin footer items (``ctx.register_footer_item``): polled snapshots and their rows.
        self.plugin_items = []
        self.plugin_rows = []
        self._plugin_dirty = False
        self._plugin_poller = None
        # ``cli_footer_tips.FooterTips`` while display.cli_footer_tips is on (install_dock).
        self.tips = None

    @property
    def has_rows(self):
        return bool(self.entries or self.processes or self.goal or self.loop or self.queued
                    or self.plugin_items)

    @property
    def tasks(self):
        """Agents first, then processes — what the full-height monitor lists."""
        return [*self.entries, *self.processes]

    @property
    def roster(self):
        """The footer's selectable rows: agents, processes, then plugin rows."""
        return [*self.entries, *self.processes, *self.plugin_rows]

    @staticmethod
    def _key(row):
        return row.get('key', row.get('subagent_id'))

    @property
    def selected(self):
        return next((r for r in self.roster if self._key(r) == self.selected_id), None)

    @property
    def selected_process(self):
        row = self.selected
        return row if row and row.get('kind') == 'process' else None

    def _local_entries(self, now):
        from tools.delegate_tool_registry import _list_payload, list_active_subagents
        parent = getattr(self.cli, 'agent', None)
        entries = _list_payload(parent)['subagents'] if parent is not None else []
        # The scoped control-plane snapshot supplies authority and transcript paths;
        # its matching public lifecycle record supplies the latest observed tool.
        activity = {r['subagent_id']: r for r in list_active_subagents()} if entries else {}
        for row in entries:
            live = activity.get(row['subagent_id'], {})
            row['elapsed'] = max(0, int(now - live.get('started_at', now)))
            row['last_tool'] = live.get('last_tool') or ''
            row['key'] = row['subagent_id']
            row.pop('running_seconds', None)
        return entries

    def _poll_plugin_items(self):
        """Last good plugin footer snapshots; each runs in its own worker thread, never here."""
        from hermes_cli import cli_plugin_footer as plugin_footer
        if self._plugin_poller is None:
            self._plugin_poller = plugin_footer.FooterItemPoller(on_change=self._plugin_changed)
        try:
            return self._plugin_poller.poll(plugin_footer.session_info(self.cli))
        except Exception:
            logging.getLogger(__name__).debug('plugin footer poll failed', exc_info=True)
            return self.plugin_items

    def _plugin_changed(self):
        self._plugin_dirty = True

    def refresh(self, now=None):
        from hermes_cli.gateway_tools import process_view
        from hermes_cli.cli_plugin_footer import footer_rows
        now = time.time() if now is None else now
        # A gateway-backed CLI runs its children in the gateway process, so the in-process
        # registry is empty there: read the session's ``subagent.list`` snapshot instead.
        view = process_view()
        entries = view.subagent_rows(now) if view is not None else self._local_entries(now)
        processes = procs.process_rows(now)
        goal = session_rows.goal_line(self.cli)
        loop_badge, loop, loop_prompt = session_rows.loop_row(self.cli, now)
        queued = session_rows.queued_prompts(self.cli)
        plugin_items = self._poll_plugin_items()
        signature = json.dumps([entries, processes, goal, loop, queued, plugin_items], sort_keys=True, default=str)
        changed = signature != self._signature
        self._signature = signature
        self.entries = entries
        self.processes = processes
        self.goal = goal
        self.loop = loop
        self.loop_badge = loop_badge
        self.loop_prompt = loop_prompt
        self.queued = queued
        self.plugin_items = plugin_items
        self.plugin_rows = footer_rows(plugin_items)
        if self.selected is None:
            roster = self.roster
            self.selected_id = self._key(roster[0]) if roster else None
            self.confirm_key = self.confirm_action = None
        if not self.roster:
            self.footer_active = False
        if self.steer_target is not None and not any(self._key(r) == self.steer_target for r in self.entries):
            self.steer_target, self.steer_draft = None, ''  # the child finished while composing
        return changed

    def invalidate(self):
        from hermes_cli.cli_terminal_mixin import _run_on_app_loop

        app = self.app
        if app is not None:
            # Teardown clears app.loop; don't let it interleave with a worker's
            # invalidate call, which reads the loop more than once.
            _run_on_app_loop(app, app.invalidate)

    def tick(self):
        now = time.monotonic()
        # A plugin snapshot that finished with a new value repaints without waiting a whole second.
        if now - self._last_poll < 1 and not self._plugin_dirty:
            return
        self._last_poll = now
        self._plugin_dirty = False
        changed = self.refresh()
        if self.tips is not None and not self.has_rows and self.tips.due():
            changed = self.tips.rotate() or changed  # repaint only when the tip text changed
        if changed:
            if self.app is not None:
                self.invalidate()
            else:
                self.cli._invalidate()

    def select(self, delta):
        roster = self.tasks
        if roster:
            index = next((i for i, r in enumerate(roster) if self._key(r) == self.selected_id), 0)
            self.selected_id = self._key(roster[(index + delta) % len(roster)])

    def control(self, action, message=None, *, target=None):
        target = target or self.selected_id
        if any(r['key'] == target for r in self.processes):
            if action != 'stop':
                return {'error': t('cli.subagents.cannot_steer_process')}
            return procs.kill(target)
        from hermes_cli.gateway_tools import process_view
        view = process_view()
        if view is not None:
            return view.control_subagent(action, target, message)
        from tools.delegate_tool_registry import _handle_control_action
        return json.loads(_handle_control_action(action, target, message, getattr(self.cli, 'agent', None)))

    def _counts(self, *, session=True):
        """Count fragment: ``2 live``, ``1 proc``, ``2 live · 3 procs``; the collapsed heading
        (``session=True``) adds ``goal active|paused|parked``, the loop badge and ``N queued``."""
        parts = []
        if self.entries:
            parts.append(t('cli.subagents.count_live', count=len(self.entries)))
        if self.processes:
            running = sum(r['status'] == 'running' for r in self.processes)
            if running:
                parts.append(t('cli.subagents.count_procs_one' if running == 1 else 'cli.subagents.count_procs_other',
                               count=running))
            else:
                parts.append(t('cli.subagents.count_done', count=len(self.processes)))
        if session and self.goal:
            parts.append(t('cli.subagents.goal_parked' if self.goal.startswith('⏳') else
                           'cli.subagents.goal_paused' if self.goal.startswith('⏸') else 'cli.subagents.goal_active'))
        if session and self.loop_badge:
            parts.append(self.loop_badge)
        if session and self.queued:
            parts.append(t('cli.subagents.count_queued', count=len(self.queued)))
        return ' · '.join(parts)

    def _title(self):
        if self.entries and self.processes:
            return t('cli.subagents.title_live_work')
        return t('cli.subagents.title_subagents') if self.entries else t('cli.subagents.title_processes')

    def dock_text(self, *, columns, rows, selecting=False):
        """The live-work footer as plain text (``cli_dock_footer.dock_lines`` without styles)."""
        from hermes_cli import cli_dock_footer as footer
        return footer.plain_text(footer.dock_lines(self, columns=columns, rows=rows, selecting=selecting))


def read_tail(path):
    if not path:
        return t('cli.subagents.transcript_unavailable')
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 32768))
            text = stream.read(32768).decode('utf-8', errors='replace')
        return ''.join(c for c in text if c.isprintable() or c in '\n\t')
    except OSError:
        return t('cli.subagents.transcript_unavailable')


def modal_prompt_active(cli):
    return any(getattr(cli, name, None) for name in (
        '_clarify_state', '_approval_state', '_slash_confirm_state', '_sudo_state',
        '_secret_state', '_model_picker_state', '_command_palette_state'))


def subagent_tail(row, cache):
    """A child's transcript tail: its file in-process, ``subagent.tail`` when it runs in the gateway
    (at most one RPC a second; the monitor repaints on every key)."""
    from hermes_cli.gateway_tools import process_view
    view = process_view()
    if view is None or row.get('live_transcript'):
        return read_tail(row.get('live_transcript'))
    key, now = row['subagent_id'], time.monotonic()
    if cache.get('key') != key or now - cache.get('at', 0) >= 1:
        text = view.subagent_tail(key)
        cache.update(key=key, at=now, text=''.join(c for c in text if c.isprintable() or c in '\n\t')
                     if text else t('cli.subagents.transcript_unavailable'))
    return cache['text']


def build_monitor_application(monitor, *, detail_key=None, **kwargs):
    """Full-screen roster + tail. ``detail_key`` opens straight on that row's tail (the footer's
    Enter); Esc then returns to the composer instead of the roster."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea

    state = {'detail': False, 'steering': False, 'confirm': False, 'notice': '', 'direct': False}
    tail_cache = {}

    def current():
        # The footer's plugin rows are not tasks: this view never opens or stops them.
        row = monitor.selected
        return row if row is not None and row.get('kind') != 'plugin' else None
    steer = TextArea(height=1, prompt=t('cli.subagents.steer_prompt'), multiline=False)
    tail = TextArea(read_only=True, scrollbar=True, wrap_lines=True)

    def roster_text():
        size = app.output.get_size()
        rows = []
        for row in monitor.tasks:
            selected = monitor._key(row) == monitor.selected_id
            if row.get('kind') == 'process':
                prefix = f"{procs.process_glyph(row)} {procs.process_activity(row)} · "
                activity = ''
                subject = row['command']
            else:
                prefix = f"{row['elapsed']}s · {row.get('status') or t('cli.subagents.starting')} · "
                activity = ' · ' + t('cli.subagents.last_tool', tool=row['last_tool']) if row.get('last_tool') else ''
                subject = row.get('goal') or row['subagent_id']
            goal_width = max(0, size.columns - 2 - get_cwidth(prefix + activity))
            text = f"{'❯' if selected else ' '} " + _clip(prefix + _clip(subject, goal_width) + activity, max(0, size.columns - 2))
            # Pad selection in terminal cells, not codepoints (task names may be wide).
            text += ' ' * max(0, size.columns - get_cwidth(text))
            rows.append(('class:subagent-dock.selected' if selected else '', text + '\n'))
        return rows or [('', t('cli.subagents.roster_empty'))]

    def cursor():
        index = next((i for i, row in enumerate(monitor.tasks) if monitor._key(row) == monitor.selected_id), 0)
        return Point(x=0, y=index)

    roster = Window(FormattedTextControl(roster_text, focusable=True, get_cursor_position=cursor))

    def update_tail():
        row = current()
        if row is None:
            text = t('cli.subagents.entry_gone')
        elif row.get('kind') == 'process':
            text = procs.process_tail(row['id'])
        else:
            text = subagent_tail(row, tail_cache)
        if text != tail.text:
            following = tail.buffer.cursor_position == len(tail.text)
            position = tail.buffer.cursor_position
            tail.text = text
            tail.buffer.cursor_position = len(text) if following else min(position, len(text))

    def header():
        row = current()
        title = f"{monitor._title()} · {monitor._counts(session=False)}"
        if state['detail'] and row:
            title += f" · {monitor._key(row)} · {row.get('goal') or row.get('command') or ''}"
        return [('class:subagent-dock.heading', _clip(title, app.output.get_size().columns))]

    def footer():
        narrow = app.output.get_size().columns < 60
        process = monitor.selected_process is not None
        if state['confirm']:
            noun = t('cli.subagents.noun_process' if process else 'cli.subagents.noun_subagent')
            return (t('cli.subagents.footer_confirm_stop_compact') if narrow
                    else t('cli.subagents.footer_confirm_stop', noun=noun))
        if state['steering']:
            return t('cli.subagents.footer_steering_compact' if narrow else 'cli.subagents.footer_steering')
        steer = '' if process else t('cli.subagents.footer_steer_key')
        if narrow:
            return (t('cli.subagents.footer_detail_compact', steer=steer) if state['detail']
                    else t('cli.subagents.footer_roster_compact'))
        steer = '' if process else t('cli.subagents.footer_steer_key_long')
        if state['detail']:
            # Opened from the footer row: Esc goes back to the prompt, not to the roster.
            return t('cli.subagents.footer_detail_direct' if state['direct'] else 'cli.subagents.footer_detail',
                     steer=steer)
        return t('cli.subagents.footer_roster', steer=steer)

    kb = KeyBindings()
    normal = Condition(lambda: not state['steering'] and not state['confirm'])
    listing = normal & Condition(lambda: not state['detail'])

    @kb.add('up', filter=listing)
    def up(event):
        monitor.select(-1)

    @kb.add('down', filter=listing)
    def down(event):
        monitor.select(1)

    @kb.add('enter', filter=listing)
    def detail(event):
        if current():
            state['detail'] = True
            update_tail()
            app.layout.focus(tail)

    @kb.add('s', filter=normal)
    def start_steer(event):
        if current() and monitor.selected_process is None:
            state['steering'] = True
            state['target'] = monitor.selected_id
            app.layout.focus(steer)

    @kb.add('enter', filter=Condition(lambda: state['steering']))
    def send_steer(event):
        if not steer.text.strip():
            return
        result = monitor.control('steer', steer.text, target=state['target'])
        state['notice'] = result.get('error') or result.get('note') or str(result)
        steer.text = ''
        state['steering'] = False
        app.layout.focus(tail if state['detail'] else roster)

    @kb.add('x', filter=normal)
    def stop(event):
        if current():
            state['confirm'] = True
            state['target'] = monitor.selected_id

    @kb.add('y', filter=Condition(lambda: state['confirm']))
    def confirm(event):
        result = monitor.control('stop', target=state['target'])
        state['notice'] = result.get('error') or result.get('note') or str(result)
        state['confirm'] = False

    @kb.add('escape', eager=True)
    def back(event):
        if state['steering'] or state['confirm']:
            state['steering'] = state['confirm'] = False
            app.layout.focus(tail if state['detail'] else roster)
        elif state['detail'] and not state['direct']:
            state['detail'] = False
            app.layout.focus(roster)
        else:
            app.exit()

    @kb.add('q', filter=normal)
    @kb.add('f6', filter=normal)
    @kb.add('c-t', filter=normal)
    @kb.add('c-c')
    def close(event):
        app.exit()

    layout = Layout(HSplit([
        Window(FormattedTextControl(header), height=1),
        ConditionalContainer(roster, filter=Condition(lambda: not state['detail'])),
        ConditionalContainer(tail, filter=Condition(lambda: state['detail'])),
        ConditionalContainer(steer, filter=Condition(lambda: state['steering'])),
        Window(FormattedTextControl(lambda: _clip(state['notice'], app.output.get_size().columns)), height=1),
        Window(FormattedTextControl(footer), height=1),
    ], style='class:subagent-dock'), focused_element=roster)
    def before_render(app):
        # Prompts arrive on worker threads; exit on the UI loop, including the
        # first frame if a prompt won the race with in_terminal() acquisition.
        if modal_prompt_active(monitor.cli) and not app.is_done:
            app.exit()
        elif state['detail']:
            update_tail()

    from prompt_toolkit.styles import Style
    from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides
    kwargs.setdefault('style', Style.from_dict(get_prompt_toolkit_style_overrides()))
    if current() is None and monitor.tasks:
        monitor.selected_id = monitor._key(monitor.tasks[0])
    if detail_key is not None and any(monitor._key(r) == detail_key for r in monitor.tasks):
        monitor.selected_id = detail_key
        state['detail'] = state['direct'] = True
        update_tail()
    app = Application(layout=layout, key_bindings=kb, full_screen=True, mouse_support=False,
                      before_render=before_render, **kwargs)
    if state['detail']:
        app.layout.focus(tail)
    return app


def open_monitor(cli, key=None):
    """Ctrl+T: the full-screen roster. ``key`` (the footer's Enter / click) opens that row's tail."""
    import asyncio
    from prompt_toolkit.application import in_terminal
    monitor = getattr(cli, '_subagent_monitor', None)
    if monitor is None or monitor.opening:
        return
    monitor.opening = True

    async def run():
        try:
            async with in_terminal():
                monitor.refresh()
                monitor.app = build_monitor_application(monitor, detail_key=key)
                await monitor.app.run_async()
        finally:
            monitor.app = None
            monitor.opening = False
            cli._invalidate()

    asyncio.get_running_loop().create_task(run())


def toggle_dock(cli):
    monitor = getattr(cli, '_subagent_monitor', None)
    if monitor is not None:
        monitor.collapsed = not monitor.collapsed
        cli._invalidate()


def install_dock(cli):
    """The live-work footer under the composer (``cli_dock_footer``): a summary pill and one row
    per subagent / background process, selectable with Down and clickable with ``display.cli_mouse``.
    With nothing to list it shows a rotating tip instead (``display.cli_footer_tips``)."""
    from prompt_toolkit.application import get_app
    from prompt_toolkit.layout import ConditionalContainer, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.filters import Condition
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli import cli_footer_tips
    monitor = SubagentMonitor(cli)
    cli._subagent_monitor = monitor
    if getattr(cli, '_footer_tips_enabled', False):
        monitor.tips = cli_footer_tips.FooterTips()
    monitor.refresh()
    actions = footer.click_actions(cli)

    def text():
        size = get_app().output.get_size()
        lines = footer.dock_lines(monitor, columns=size.columns, rows=size.rows,
                                  selecting=footer.selecting(cli), actions=actions)
        fragments = []
        for index, line in enumerate(lines):
            if index:
                fragments.append(('', '\n'))
            fragments.extend(line)
        return fragments

    cli._subagent_dock_widget = ConditionalContainer(
        Window(FormattedTextControl(text), wrap_lines=False, dont_extend_height=True),
        filter=Condition(lambda: (monitor.has_rows and not modal_prompt_active(cli))
                         or cli_footer_tips.showing(cli, get_app().output.get_size().rows)),
    )
