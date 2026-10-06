"""Classic CLI live-work dock (subagents, background processes, active goal, queued prompts) and
scoped controls; no agent-loop state is changed. Process rows come from ``cli_process_dock``,
goal/queue rows from ``cli_session_dock``."""
from __future__ import annotations

import json
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
        self.queued = []
        self.selected_id = None
        self._signature = None
        self._last_poll = 0
        self.app = None
        self.opening = False
        self.collapsed = False

    @property
    def has_rows(self):
        return bool(self.entries or self.processes or self.goal or self.queued)

    @property
    def roster(self):
        """Agents first, then processes — the order the dock and the full-height monitor paint."""
        return [*self.entries, *self.processes]

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

    def refresh(self, now=None):
        from tools.delegate_tool_registry import _list_payload, list_active_subagents
        now = time.time() if now is None else now
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
        processes = procs.process_rows(now)
        goal = session_rows.goal_line(self.cli)
        queued = session_rows.queued_prompts(self.cli)
        signature = json.dumps([entries, processes, goal, queued], sort_keys=True, default=str)
        changed = signature != self._signature
        self._signature = signature
        self.entries = entries
        self.processes = processes
        self.goal = goal
        self.queued = queued
        if self.selected is None:
            roster = self.roster
            self.selected_id = self._key(roster[0]) if roster else None
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
        if now - self._last_poll < 1:
            return
        self._last_poll = now
        if self.refresh():
            if self.app is not None:
                self.invalidate()
            else:
                self.cli._invalidate()

    def select(self, delta):
        roster = self.roster
        if roster:
            index = next((i for i, r in enumerate(roster) if self._key(r) == self.selected_id), 0)
            self.selected_id = self._key(roster[(index + delta) % len(roster)])

    def control(self, action, message=None, *, target=None):
        target = target or self.selected_id
        if any(r['key'] == target for r in self.processes):
            if action != 'stop':
                return {'error': t('cli.subagents.cannot_steer_process')}
            return procs.kill(target)
        from tools.delegate_tool_registry import _handle_control_action
        return json.loads(_handle_control_action(action, target, message, getattr(self.cli, 'agent', None)))

    def _counts(self, *, session=True):
        """Count fragment: ``2 live``, ``1 proc``, ``2 live · 3 procs``; the collapsed heading
        (``session=True``) adds ``goal active|paused|parked`` and ``N queued``."""
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
        if session and self.queued:
            parts.append(t('cli.subagents.count_queued', count=len(self.queued)))
        return ' · '.join(parts)

    def _title(self):
        if self.entries and self.processes:
            return t('cli.subagents.title_live_work')
        return t('cli.subagents.title_subagents') if self.entries else t('cli.subagents.title_processes')

    @staticmethod
    def _agent_activity(row):
        """``last: <tool>`` while a tool runs, else the raw status id, else ``starting``."""
        if row.get('last_tool'):
            return t('cli.subagents.last_tool', tool=row['last_tool'])
        return row.get('status') or t('cli.subagents.starting')

    def _collapsed_activity(self):
        if self.entries:
            return self._agent_activity(self.entries[0])
        if self.processes:
            return procs.process_activity(self.processes[0])
        return self.goal or t('cli.subagents.next_queued', text=self.queued[0])

    def dock_text(self, *, columns, rows):
        if not self.has_rows:
            return ''
        if self.collapsed:
            count = self._counts()
            # Keep both controls before spending scarce cells on activity. Ctrl+T opens the
            # subagent/process monitor, so a goal/queue-only dock offers just the
            # collapse/restore shortcut.
            if self.entries or self.processes:
                headings = (
                    t('cli.subagents.heading_full', title=self._title(), count=count),
                    t('cli.subagents.heading_medium', count=count),
                    t('cli.subagents.heading_compact', count=count),
                    count,
                )
            else:
                headings = (t('cli.subagents.heading_session_restore', count=count),
                            t('cli.subagents.heading_session_compact', count=count), count)
            width = max(0, columns - 1)
            heading = next((text for text in headings if get_cwidth(text) <= width), count)
            activity = self._collapsed_activity()
            # A goal/queue preview is long prose: clip it into the room left instead of dropping it.
            room = width - get_cwidth(heading + ' · ')
            if get_cwidth(activity) <= room or (room >= 12 and not (self.entries or self.processes)):
                heading += ' · ' + _clip(activity, room)
            return _clip(' ' + heading, max(0, columns))
        columns = max(0, columns - 2)
        lines = [_clip(f' {self.goal}', columns)] if self.goal else []
        budget = max(1, min(4, (rows - 10) // 3))
        # Both blocks present: split the row budget so neither hides the other entirely.
        agent_budget = budget if not self.processes else max(1, budget - max(1, budget // 2))
        agent_count = min(len(self.entries), agent_budget)
        if self.entries:
            hidden = len(self.entries) - agent_count
            lines.append(_clip(' ' + t('cli.subagents.subagents_heading', count=len(self.entries)), columns))
            for row in self.entries[:agent_count]:
                activity = f"{row['elapsed']}s · " + self._agent_activity(row)
                # Reserve activity even on narrow terminals; task names use the remainder.
                goal_width = max(3, columns - get_cwidth(activity) - 5)
                lines.append(_clip(f" ● {_clip(row.get('goal'), goal_width)} · {activity}", columns))
            if hidden:
                lines.append(_clip(' ' + t('cli.subagents.more_subagents', count=hidden), columns))
        if self.processes:
            proc_count = min(len(self.processes), max(1, budget - agent_count))
            running = sum(r['status'] == 'running' for r in self.processes)
            done = len(self.processes) - running
            summary = ' · '.join(p for p in (
                t('cli.subagents.count_running', count=running) if running else '',
                t('cli.subagents.count_done', count=done) if done else '') if p)
            controls = t('cli.subagents.controls_expand_collapse') if not self.entries else ''
            lines.append(_clip(' ' + t('cli.subagents.processes_heading', summary=summary, controls=controls), columns))
            for row in self.processes[:proc_count]:
                activity = procs.process_activity(row)
                command_width = max(3, columns - get_cwidth(activity) - 5)
                lines.append(_clip(f" {procs.process_glyph(row)} {_clip(row['command'], command_width)} · {activity}", columns))
            if len(self.processes) > proc_count:
                lines.append(_clip(
                    ' ' + t('cli.subagents.more_processes', count=len(self.processes) - proc_count), columns))
        if self.queued:
            # Last, so the next prompt to run sits right above the input it came from.
            shown = min(len(self.queued), session_rows.QUEUE_ROWS if rows >= 24 else 1)
            controls = '' if self.entries or self.processes else t('cli.subagents.controls_collapse')
            lines.append(_clip(
                ' ' + t('cli.subagents.queue_heading', count=len(self.queued), controls=controls), columns))
            for index, text in enumerate(self.queued[:shown], 1):
                lines.append(_clip(f'  {index}. {text}', columns))
            if len(self.queued) > shown:
                lines.append(_clip('  ' + t('cli.subagents.more_queued', count=len(self.queued) - shown), columns))
        return '\n'.join(' ' + line for line in lines)


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


def build_monitor_application(monitor, **kwargs):
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea

    state = {'detail': False, 'steering': False, 'confirm': False, 'notice': ''}
    steer = TextArea(height=1, prompt=t('cli.subagents.steer_prompt'), multiline=False)
    tail = TextArea(read_only=True, scrollbar=True, wrap_lines=True)

    def roster_text():
        size = app.output.get_size()
        rows = []
        for row in monitor.roster:
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
        index = next((i for i, row in enumerate(monitor.roster) if monitor._key(row) == monitor.selected_id), 0)
        return Point(x=0, y=index)

    roster = Window(FormattedTextControl(roster_text, focusable=True, get_cursor_position=cursor))

    def update_tail():
        row = monitor.selected
        if row is None:
            text = t('cli.subagents.entry_gone')
        elif row.get('kind') == 'process':
            text = procs.process_tail(row['id'])
        else:
            text = read_tail(row.get('live_transcript'))
        if text != tail.text:
            following = tail.buffer.cursor_position == len(tail.text)
            position = tail.buffer.cursor_position
            tail.text = text
            tail.buffer.cursor_position = len(text) if following else min(position, len(text))

    def header():
        row = monitor.selected
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
        return t('cli.subagents.footer_detail' if state['detail'] else 'cli.subagents.footer_roster', steer=steer)

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
        if monitor.selected:
            state['detail'] = True
            update_tail()
            app.layout.focus(tail)

    @kb.add('s', filter=normal)
    def start_steer(event):
        if monitor.selected and monitor.selected_process is None:
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
        if monitor.selected:
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
        elif state['detail']:
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
    app = Application(layout=layout, key_bindings=kb, full_screen=True, mouse_support=False,
                      before_render=before_render, **kwargs)
    return app


def open_monitor(cli):
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
                monitor.app = build_monitor_application(monitor)
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
    from prompt_toolkit.application import get_app
    from prompt_toolkit.layout import ConditionalContainer, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.filters import Condition
    monitor = SubagentMonitor(cli)
    cli._subagent_monitor = monitor
    monitor.refresh()

    def text():
        size = get_app().output.get_size()
        lines = monitor.dock_text(columns=size.columns, rows=size.rows).splitlines()
        return [('class:subagent-dock.heading' if i == 0 else '',
                 line + ('\n' if i < len(lines) - 1 else ''))
                for i, line in enumerate(lines)]

    cli._subagent_dock_widget = ConditionalContainer(
        Window(FormattedTextControl(text), wrap_lines=False, dont_extend_height=True,
               style='class:subagent-dock'),
        filter=Condition(lambda: monitor.has_rows and not modal_prompt_active(cli)),
    )
