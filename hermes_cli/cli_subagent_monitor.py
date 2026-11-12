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

logger = logging.getLogger(__name__)


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
        self.finished = []  # stored subagents of this conversation that are no longer live (list only)
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
        self.detail = False  # the monitor's detail view is open (its tick fetches the gateway tail)
        self.viewer = None  # the open subagent viewer (cli_subagent_viewer.SubagentViewer)

    @property
    def has_rows(self):
        return bool(self.entries or self.processes or self.goal or self.loop or self.queued)

    @property
    def roster(self):
        """Live agents, then finished ones, then processes — the full-height monitor's order (the
        dock paints only the live rows)."""
        return [*self.entries, *self.finished, *self.processes]

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

    def subagents(self):
        """The subagent backend: the gateway session's view when its children run there (CLI-12),
        else this process's own registry; both answer the same ``subagent_*`` calls."""
        from hermes_cli.gateway_tools import LocalSubagents, process_view
        view = process_view()
        if view is not None:
            return view
        if getattr(self, '_local', None) is None:
            self._local = LocalSubagents(self.cli)
        return self._local

    def refresh(self, now=None, *, history=None):
        """Re-read the live rows; ``history`` also reads this conversation's finished subagents from
        the session store (the full-height monitor's list; the dock never paints them). By default
        they are read only while the full-height monitor is open."""
        now = time.time() if now is None else now
        history = self.app is not None if history is None else history
        entries = self.subagents().subagent_rows(busy=bool(getattr(self.cli, '_agent_running', False)))
        for row in entries:
            row['elapsed'] = max(0, int(now - (row.get('started_at') or now)))
            row['last_tool'] = row.get('last_tool') or ''
            row['key'] = row['subagent_id']
            row.pop('running_seconds', None)
        finished = finished_rows(self.cli, entries) if history else []
        processes = procs.process_rows(now)
        session_rows.sync_automation(self.cli)
        goal = session_rows.goal_line(self.cli)
        loop_badge, loop, loop_prompt = session_rows.loop_row(self.cli, now)
        queued = session_rows.queued_prompts(self.cli)
        signature = json.dumps([entries, finished, processes, goal, loop, queued], sort_keys=True, default=str)
        changed = signature != self._signature
        self._signature = signature
        self.entries = entries
        self.finished = finished
        self.processes = processes
        self.goal = goal
        self.loop = loop
        self.loop_badge = loop_badge
        self.loop_prompt = loop_prompt
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

    def poll_soon(self):
        """Let the spinner thread's next tick poll the gateway, instead of the caller blocking the
        UI loop on it."""
        self._last_poll = 0

    def tick(self):
        now = time.monotonic()
        if now - self._last_poll < 1:
            return
        self._last_poll = now
        try:
            changed = self.refresh()
            changed = self._poll_tail() or changed
            viewer = self.viewer
            changed = (viewer is not None and viewer.poll()) or changed
        except Exception:  # health: allow BLE001 -- the spinner thread runs this tick; one raise would end that thread for the session
            logger.debug("subagent monitor tick failed", exc_info=True)
            return
        if changed:
            if self.app is not None:
                self.invalidate()
            else:
                self.cli._invalidate()

    def _poll_tail(self):
        """Off the render path (the spinner thread's tick): refresh the gateway tail the open detail
        view paints from the cache. True when the text changed."""
        row = self.selected
        if not self.detail or row is None or row.get('kind') == 'process':
            return False
        backend = self.subagents()
        before = backend.subagent_tail(row['subagent_id'])
        return backend.subagent_tail(row['subagent_id'], fetch=True) != before

    def select(self, delta):
        roster = self.roster
        if roster:
            index = next((i for i, r in enumerate(roster) if self._key(r) == self.selected_id), 0)
            self.selected_id = self._key(roster[(index + delta) % len(roster)])

    def control(self, action, message=None, *, target=None):
        target = target or self.selected_id
        if any(r['key'] == target for r in self.finished):
            return {'error': t('cli.subagents.finished_cannot_control')}
        if any(r['key'] == target for r in self.processes):
            if action != 'stop':
                return {'error': t('cli.subagents.cannot_steer_process')}
            return procs.kill(target)
        return self.subagents().subagent_control(action, target, message)

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

    def list_title(self):
        """The full-screen list's heading: live and finished subagents, then processes."""
        subagents = bool(self.entries or self.finished)
        title = (t('cli.subagents.title_live_work') if subagents and self.processes
                 else t('cli.subagents.title_subagents') if subagents else t('cli.subagents.title_processes'))
        parts = [self._counts(session=False)] if (self.entries or self.processes) else []
        if self.finished:
            parts.append(t('cli.subagents.count_finished', count=len(self.finished)))
        return ' · '.join([title, *[part for part in parts if part]])

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
        # The loop's state is already the heading's badge; its prompt is the activity.
        return self.goal or self.loop_prompt or t('cli.subagents.next_queued', text=self.queued[0])

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
        lines = [_clip(f' {row}', columns) for row in (self.goal, self.loop) if row]
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


def master_session_id(cli):
    """The stored id of the conversation the CLI shows (the agent's follows compression rotations)."""
    return getattr(getattr(cli, 'agent', None), 'session_id', None) or getattr(cli, 'session_id', None)


def finished_rows(cli, live_rows):
    """This conversation's stored subagents that are not live (oldest first), as list rows keyed
    by their stored session id. Empty without a session store; a store error reads as none."""
    db, session_id = getattr(cli, '_session_db', None), master_session_id(cli)
    if db is None or not session_id:
        return []
    try:
        stored = db.list_subagent_sessions(session_id)
    except Exception:  # health: allow BLE001 -- the list is a view; a locked or older store must not break the monitor
        logger.debug("stored subagent read failed", exc_info=True)
        return []
    live = {row.get('child_session_id') for row in live_rows if row.get('child_session_id')}
    rows = []
    for record in stored:
        if record['session_id'] in live:
            continue
        ended, started = record.get('ended_at'), record.get('started_at')
        rows.append({
            'kind': 'finished', 'key': record['session_id'], 'subagent_id': record['session_id'],
            'child_session_id': record['session_id'], 'goal': record.get('goal') or record.get('title') or '',
            # 0 = a direct child: the live registry's (and subagent.list's) convention
            'depth': record.get('depth', 1) - 1, 'model': record.get('model'), 'ended': ended is not None,
            'elapsed': max(0, int(ended - started)) if ended and started else 0,
        })
    return rows


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


def subagent_tail(monitor, row):
    """The child's transcript tail from the monitor's backend (the gateway's is fetched and filtered by
    the monitor's tick, never here: the detail view calls this every frame)."""
    text = monitor.subagents().subagent_tail(row['subagent_id'])
    return t('cli.subagents.transcript_unavailable') if text is None else text


def modal_prompt_active(cli):
    """A prompt owns the composer (its text is an answer or a secret, never a chat message):
    every composer shortcut stays off while one is open."""
    return any(getattr(cli, name, None) for name in (
        '_clarify_state', '_approval_state', '_slash_confirm_state', '_sudo_state',
        '_secret_state', '_connection_state', '_model_picker_state', '_command_palette_state'))


def build_monitor_application(monitor, **kwargs):
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea

    state = {'steering': False, 'confirm': False, 'notice': ''}
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
            elif row.get('kind') == 'finished':
                state_text = t('cli.subagents.finished_status' if row.get('ended') else 'cli.subagents.not_live_status')
                prefix = f"✓ {row['elapsed']}s · {state_text} · "
                activity = ''
                subject = row.get('goal') or row['subagent_id']
            else:
                prefix = f"● {row['elapsed']}s · {row.get('status') or t('cli.subagents.starting')} · "
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
            text = subagent_tail(monitor, row)
        if text != tail.text:
            following = tail.buffer.cursor_position == len(tail.text)
            position = tail.buffer.cursor_position
            tail.text = text
            tail.buffer.cursor_position = len(text) if following else min(position, len(text))

    def header():
        row = monitor.selected
        title = monitor.list_title()
        if monitor.detail and row:
            title += f" · {monitor._key(row)} · {row.get('goal') or row.get('command') or ''}"
        return [('class:subagent-dock.heading', _clip(title, app.output.get_size().columns))]

    def footer():
        narrow = app.output.get_size().columns < 60
        process = monitor.selected_process is not None
        if state['confirm']:
            noun = t('cli.subagents.noun_process' if process else 'cli.subagents.noun_subagent')
            return (t('cli.subagents.footer_confirm_stop_compact') if narrow
                    else t('cli.subagents.footer_confirm_stop', noun=noun))
        if monitor.viewer is not None:
            return viewer_footer()
        if state['steering']:
            return t('cli.subagents.footer_steering_compact' if narrow else 'cli.subagents.footer_steering')
        steer = '' if process else t('cli.subagents.footer_steer_key')
        if narrow:
            return (t('cli.subagents.footer_detail_compact', steer=steer) if monitor.detail
                    else t('cli.subagents.footer_roster_compact'))
        steer = '' if process else t('cli.subagents.footer_steer_key_long')
        return t('cli.subagents.footer_detail' if monitor.detail else 'cli.subagents.footer_roster', steer=steer)

    def run_call(call):
        """Steer/stop off the UI loop: a stalled gateway call (up to its timeout) must not freeze the
        monitor's repaint, Esc or Ctrl+C. The answer lands in the notice line when it arrives."""
        import asyncio
        state['notice'] = t('cli.subagents.control_pending')

        def done(future):
            try:
                result = future.result()
            except Exception as exc:  # health: allow BLE001 -- a control failure is shown, never raised into the UI loop
                result = {'error': str(exc)}
            state['notice'] = result.get('error') or result.get('note') or str(result)
            app.invalidate()

        asyncio.get_running_loop().run_in_executor(None, call).add_done_callback(done)

    def run_control(action, message, target):
        run_call(lambda: monitor.control(action, message, target=target))

    from hermes_cli.cli_subagent_viewer import SubagentViewer, viewer_ui
    viewer_box, viewer_entry, viewer_kb, viewer_header, viewer_footer = viewer_ui(
        monitor, state, lambda: app, run_call)
    viewing = Condition(lambda: monitor.viewer is not None)

    kb = KeyBindings()
    normal = Condition(lambda: not state['steering'] and not state['confirm'])
    keys = normal & ~viewing  # single-letter shortcuts; in the viewer, letters are typed into its input
    listing = keys & Condition(lambda: not monitor.detail)

    @kb.add('up', filter=listing)
    def up(event):
        monitor.select(-1)

    @kb.add('down', filter=listing)
    def down(event):
        monitor.select(1)

    @kb.add('enter', filter=listing)
    def open_entry(event):
        row = monitor.selected
        if row is None:
            return
        if row.get('kind') == 'process':
            show_tail()
            return
        monitor.viewer = SubagentViewer(monitor, row.get('child_session_id') or row['subagent_id'])
        app.layout.focus(viewer_entry)

    @kb.add('t', filter=listing)
    def raw_tail(event):
        """The raw live transcript tail of a running subagent (or a process's log)."""
        row = monitor.selected
        if row is not None and row.get('kind') != 'finished':
            show_tail()

    def show_tail():
        monitor.detail = True
        monitor.poll_soon()  # the spinner thread's next tick fetches the tail
        update_tail()
        app.layout.focus(tail)

    @kb.add('m', filter=listing)
    def master_view(event):
        monitor.viewer = SubagentViewer(monitor, 'master')
        app.layout.focus(viewer_entry)

    @kb.add('s', filter=keys)
    def start_steer(event):
        if monitor.selected and monitor.selected_process is None:
            state['steering'] = True
            state['target'] = monitor.selected_id
            app.layout.focus(steer)

    @kb.add('enter', filter=Condition(lambda: state['steering']))
    def send_steer(event):
        if not steer.text.strip():
            return
        run_control('steer', steer.text, state['target'])
        steer.text = ''
        state['steering'] = False
        app.layout.focus(tail if monitor.detail else roster)

    @kb.add('x', filter=keys)
    def stop(event):
        if monitor.selected:
            state['confirm'] = True
            state['target'] = monitor.selected_id

    @kb.add('y', filter=Condition(lambda: state['confirm']))
    def confirm(event):
        action = state.pop('confirm_action', None)
        if action is not None:
            run_call(action)
        else:
            run_control('stop', None, state['target'])
        state['confirm'] = False

    def focus_home():
        app.layout.focus(viewer_entry if monitor.viewer else tail if monitor.detail else roster)

    # Eager everywhere but the viewer, where Alt+Left/Right arrive as Esc + arrow and must win.
    @kb.add('escape', eager=True, filter=~viewing)
    @kb.add('escape', filter=viewing)
    def back(event):
        if state['steering'] or state['confirm']:
            state['steering'] = state['confirm'] = False
            state.pop('confirm_action', None)
            focus_home()
        elif monitor.viewer is not None:
            monitor.viewer = None
            app.layout.focus(roster)
        elif monitor.detail:
            monitor.detail = False
            app.layout.focus(roster)
        else:
            app.exit()

    @kb.add('q', filter=keys)
    @kb.add('f6', filter=normal)
    @kb.add('c-t', filter=normal)
    @kb.add('c-c')
    def close(event):
        app.exit()

    layout = Layout(HSplit([
        Window(FormattedTextControl(lambda: viewer_header() if monitor.viewer else header()), height=1),
        ConditionalContainer(roster, filter=Condition(lambda: not monitor.detail) & ~viewing),
        ConditionalContainer(tail, filter=Condition(lambda: monitor.detail) & ~viewing),
        ConditionalContainer(viewer_box, filter=viewing),
        ConditionalContainer(steer, filter=Condition(lambda: state['steering'])),
        Window(FormattedTextControl(lambda: _clip(state['notice'], app.output.get_size().columns)), height=1),
        Window(FormattedTextControl(footer), height=1),
    ], style='class:subagent-dock'), focused_element=roster)
    def before_render(app):
        # Prompts arrive on worker threads; exit on the UI loop, including the
        # first frame if a prompt won the race with in_terminal() acquisition.
        if modal_prompt_active(monitor.cli) and not app.is_done:
            app.exit()
        elif monitor.detail:
            update_tail()

    from prompt_toolkit.styles import Style
    from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides
    kwargs.setdefault('style', Style.from_dict(get_prompt_toolkit_style_overrides()))
    from prompt_toolkit.key_binding import merge_key_bindings
    app = Application(layout=layout, key_bindings=merge_key_bindings([kb, viewer_kb]), full_screen=True,
                      mouse_support=False,
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
        from hermes_cli import cli_render
        try:
            async with in_terminal():
                # Master output (``_cprint``) paints through the app prompt_toolkit reports as current,
                # which is the monitor while it is open: each paint would leave its alternate screen,
                # print on the main screen and redraw it, a flicker per streamed chunk. Hold it until
                # the monitor closes, then paint it in order on the main screen. Paints a pending
                # resize recovery already holds stay held for that recovery to release.
                owns_hold = cli_render._HELD_PAINTS is None
                cli_render._hold_paints()
                try:
                    # Paint the roster the dock's tick already holds; refreshing here would block the
                    # UI loop on the gateway's subagent.list. The spinner thread's next tick re-reads it.
                    monitor.poll_soon()
                    monitor.app = build_monitor_application(monitor)
                    await monitor.app.run_async()
                finally:
                    monitor.app = None  # paints released below go to the main screen, not the monitor
                    if owns_hold:
                        cli_render._release_paints()
        finally:
            monitor.app = monitor.viewer = None
            monitor.detail = monitor.opening = False
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
