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


_SELECTED = object()  # ``control``'s default target: the selected row


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
        self.notice = ''
        # The footer's one-line steer input (``s``): target row key and the typed draft.
        self.steer_target = None
        self.steer_draft = ''
        # ``cli_footer_tips.FooterTips`` while display.cli_footer_tips is on (install_dock).
        self.tips = None
        self.detail = False  # the monitor's detail view is open (its tick fetches the gateway tail)

    @property
    def has_rows(self):
        return bool(self.entries or self.processes or self.goal or self.loop or self.queued)

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

    def refresh(self, now=None):
        now = time.time() if now is None else now
        entries = self.subagents().subagent_rows(busy=bool(getattr(self.cli, '_agent_running', False)))
        for row in entries:
            row['elapsed'] = max(0, int(now - (row.get('started_at') or now)))
            row['last_tool'] = row.get('last_tool') or ''
            row['key'] = row['subagent_id']
            row.pop('running_seconds', None)
        processes = procs.process_rows(now)
        session_rows.sync_automation(self.cli)
        goal = session_rows.goal_line(self.cli)
        loop_badge, loop, loop_prompt = session_rows.loop_row(self.cli, now)
        queued = session_rows.queued_prompts(self.cli)
        signature = json.dumps([entries, processes, goal, loop, queued], sort_keys=True, default=str)
        changed = signature != self._signature
        self._signature = signature
        self.entries = entries
        self.processes = processes
        self.goal = goal
        self.loop = loop
        self.loop_badge = loop_badge
        self.loop_prompt = loop_prompt
        self.queued = queued
        if self.selected is None:
            roster = self.roster
            self.selected_id = self._key(roster[0]) if roster else None
            self.confirm_key = None
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

    def poll_soon(self):
        """Let the spinner thread's next tick poll the gateway, instead of the caller blocking the
        UI loop on it."""
        self._last_poll = 0

    def tick(self):
        now = time.monotonic()
        if now - self._last_poll < 1:
            return
        self._last_poll = now
        changed = self.refresh()
        if self.tips is not None and not self.has_rows and self.tips.due():
            changed = self.tips.rotate() or changed  # repaint only when the tip text changed
        if self._poll_tail() or changed:
            self.repaint()

    def repaint(self, *, now=False):
        """Repaint the full-screen monitor while it is open, else the CLI's footer (any thread);
        ``now`` skips the CLI's repaint throttle."""
        if self.app is not None:
            self.invalidate()
        elif callable(getattr(self.cli, '_invalidate', None)):
            self.cli._invalidate(min_interval=0) if now else self.cli._invalidate()

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

    def control(self, action, message=None, *, target=_SELECTED):
        """Steer / stop ``target`` (default: the selected row). An explicit target that is no longer
        listed is reported gone; it never falls back to whatever row is selected now."""
        target = self.selected_id if target is _SELECTED else target
        if target is None or not any(self._key(r) == target for r in self.roster):
            return {'error': t('cli.subagents.entry_gone')}
        if any(r['key'] == target for r in self.processes):
            if action != 'stop':
                return {'error': t('cli.subagents.cannot_steer_process')}
            return procs.kill(target)
        return self.subagents().subagent_control(action, target, message)

    def run_control(self, action, *, target, message=None, done):
        """``control`` on a worker thread: a gateway RPC (or a shell's shutdown grace) must never
        block the UI loop. ``done(result)`` runs on that worker, then the UI repaints."""
        from agent.memory_provider import spawn_context_thread

        def run():
            try:
                result = self.control(action, message, target=target)
            except Exception as exc:  # the worker's boundary: show it in the notice, keep the trace
                logger.warning('footer %s of %s failed', action, target, exc_info=True)
                result = {'error': str(exc)}
            done(result)
            self.repaint(now=True)  # a throttled repaint would hide the outcome until the next one
        spawn_context_thread(run, name=f'subagent-{action}').start()

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

    def queue_lines(self, rows):
        """The queued-prompt block as plain text: ``(heading, prompt rows, '+N more' or '')``.
        Like ``goal`` and ``loop``, the monitor words the session rows and the footer styles them."""
        shown = min(len(self.queued), session_rows.QUEUE_ROWS if rows >= 24 else 1)
        controls = '' if self.roster else t('cli.subagents.controls_collapse')
        heading = t('cli.subagents.queue_heading', count=len(self.queued), controls=controls)
        items = [f'{index}. {text}' for index, text in enumerate(self.queued[:shown], 1)]
        hidden = len(self.queued) - shown
        return heading, items, t('cli.subagents.more_queued', count=hidden) if hidden > 0 else ''

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

    state = {'steering': False, 'confirm': False, 'notice': '', 'direct': False}
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
            text = subagent_tail(monitor, row)
        if text != tail.text:
            following = tail.buffer.cursor_position == len(tail.text)
            position = tail.buffer.cursor_position
            tail.text = text
            tail.buffer.cursor_position = len(text) if following else min(position, len(text))

    def header():
        row = monitor.selected
        title = f"{monitor._title()} · {monitor._counts(session=False)}"
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
        if state['steering']:
            return t('cli.subagents.footer_steering_compact' if narrow else 'cli.subagents.footer_steering')
        steer = '' if process else t('cli.subagents.footer_steer_key')
        if narrow:
            return (t('cli.subagents.footer_detail_compact', steer=steer) if monitor.detail
                    else t('cli.subagents.footer_roster_compact'))
        steer = '' if process else t('cli.subagents.footer_steer_key_long')
        if monitor.detail:
            # Opened from the footer row: Esc goes back to the prompt, not to the roster.
            return t('cli.subagents.footer_detail_direct' if state['direct'] else 'cli.subagents.footer_detail',
                     steer=steer)
        return t('cli.subagents.footer_roster', steer=steer)

    def run_control(action, message, target):
        """Steer/stop off the UI loop (``monitor.run_control``): a stalled gateway call (up to its
        timeout) must not freeze the monitor's repaint, Esc or Ctrl+C. The answer lands in the
        notice line when it arrives."""
        state['notice'] = t('cli.subagents.control_pending')

        def done(result):
            state['notice'] = result.get('error') or result.get('note') or str(result)
        monitor.run_control(action, target=target, message=message, done=done)

    kb = KeyBindings()
    normal = Condition(lambda: not state['steering'] and not state['confirm'])
    listing = normal & Condition(lambda: not monitor.detail)

    @kb.add('up', filter=listing)
    def up(event):
        monitor.select(-1)

    @kb.add('down', filter=listing)
    def down(event):
        monitor.select(1)

    @kb.add('enter', filter=listing)
    def detail(event):
        if monitor.selected:
            monitor.detail = True
            monitor.poll_soon()  # the spinner thread's next tick fetches the tail
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
        run_control('steer', steer.text, state['target'])
        steer.text = ''
        state['steering'] = False
        app.layout.focus(tail if monitor.detail else roster)

    @kb.add('x', filter=normal)
    def stop(event):
        if monitor.selected:
            state['confirm'] = True
            state['target'] = monitor.selected_id

    @kb.add('y', filter=Condition(lambda: state['confirm']))
    def confirm(event):
        run_control('stop', None, state['target'])
        state['confirm'] = False

    @kb.add('escape', eager=True)
    def back(event):
        if state['steering'] or state['confirm']:
            state['steering'] = state['confirm'] = False
            app.layout.focus(tail if monitor.detail else roster)
        elif monitor.detail and not state['direct']:
            monitor.detail = False
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
        ConditionalContainer(roster, filter=Condition(lambda: not monitor.detail)),
        ConditionalContainer(tail, filter=Condition(lambda: monitor.detail)),
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
    if detail_key is not None and any(monitor._key(r) == detail_key for r in monitor.roster):
        monitor.selected_id = detail_key
        monitor.detail = state['direct'] = True
        monitor.poll_soon()  # the spinner thread's next tick fetches the tail
        update_tail()
    app = Application(layout=layout, key_bindings=kb, full_screen=True, mouse_support=False,
                      before_render=before_render, **kwargs)
    if monitor.detail:
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
                # Paint the roster the dock's tick already holds; refreshing here would block the UI
                # loop on the gateway's subagent.list. The spinner thread's next tick re-reads it.
                monitor.poll_soon()
                monitor.app = build_monitor_application(monitor, detail_key=key)
                await monitor.app.run_async()
        finally:
            monitor.app = None
            monitor.detail = monitor.opening = False
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
