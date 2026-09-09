"""Classic CLI subagent dock and scoped controls; no agent-loop state is changed."""
from __future__ import annotations

import json
import time

from prompt_toolkit.utils import get_cwidth


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
        self.selected_id = None
        self._signature = None
        self._last_poll = 0
        self.app = None
        self.opening = False
        self.collapsed = False

    @property
    def selected(self):
        return next((r for r in self.entries if r['subagent_id'] == self.selected_id), None)

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
            row.pop('running_seconds', None)
        signature = json.dumps(entries, sort_keys=True, default=str)
        changed = signature != self._signature
        self._signature = signature
        self.entries = entries
        if self.selected is None:
            self.selected_id = entries[0]['subagent_id'] if entries else None
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
        if self.entries:
            index = next((i for i, r in enumerate(self.entries) if r['subagent_id'] == self.selected_id), 0)
            self.selected_id = self.entries[(index + delta) % len(self.entries)]['subagent_id']

    def control(self, action, message=None, *, target=None):
        from tools.delegate_tool_registry import _handle_control_action
        return json.loads(_handle_control_action(action, target or self.selected_id, message, getattr(self.cli, 'agent', None)))

    def dock_text(self, *, columns, rows):
        if not self.entries:
            return ''
        if self.collapsed:
            count = f'{len(self.entries)} live'
            # Keep both controls before spending scarce cells on activity.
            headings = (
                f'Subagents · {count} · Ctrl+T expand · F7 restore',
                f'{count} · Ctrl+T expand · F7 restore',
                f'{count} · Ctrl+T · F7',
                count,
            )
            width = max(0, columns - 1)
            heading = next((text for text in headings if get_cwidth(text) <= width), count)
            row = self.entries[0]
            activity = f"last: {row['last_tool']}" if row.get('last_tool') else row.get('status') or 'starting'
            if get_cwidth(heading + ' · ' + activity) <= width:
                heading += ' · ' + activity
            return _clip(' ' + heading, max(0, columns))
        columns = max(0, columns - 2)
        count = min(len(self.entries), max(1, min(4, (rows - 10) // 3)))
        hidden = len(self.entries) - count
        heading = f' Subagents · {len(self.entries)} live · Ctrl+T expand · F7 collapse'
        lines = [_clip(heading, columns)]
        for row in self.entries[:count]:
            activity = f"{row['elapsed']}s · " + (f"last: {row['last_tool']}" if row['last_tool'] else row.get('status') or 'starting')
            # Reserve activity even on narrow terminals; task names use the remainder.
            goal_width = max(3, columns - get_cwidth(activity) - 5)
            lines.append(_clip(f" ● {_clip(row.get('goal'), goal_width)} · {activity}", columns))
        if hidden:
            lines.append(_clip(f' +{hidden} more · Ctrl+T all subagents', columns))
        return '\n'.join(' ' + line for line in lines)


def read_tail(path):
    if not path:
        return 'Live transcript not available yet.'
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 32768))
            text = stream.read(32768).decode('utf-8', errors='replace')
        return ''.join(c for c in text if c.isprintable() or c in '\n\t')
    except OSError:
        return 'Live transcript not available yet.'


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
    steer = TextArea(height=1, prompt='Steer: ', multiline=False)
    tail = TextArea(read_only=True, scrollbar=True, wrap_lines=True)

    def roster_text():
        size = app.output.get_size()
        rows = []
        for row in monitor.entries:
            selected = row['subagent_id'] == monitor.selected_id
            prefix = f"{row['elapsed']}s · {row.get('status') or 'starting'} · "
            activity = f" · last: {row['last_tool']}" if row.get('last_tool') else ''
            goal_width = max(0, size.columns - 2 - get_cwidth(prefix + activity))
            goal = _clip(row.get('goal') or row['subagent_id'], goal_width)
            text = f"{'❯' if selected else ' '} " + _clip(prefix + goal + activity, max(0, size.columns - 2))
            # Pad selection in terminal cells, not codepoints (task names may be wide).
            text += ' ' * max(0, size.columns - get_cwidth(text))
            rows.append(('class:subagent-dock.selected' if selected else '', text + '\n'))
        return rows or [('', 'No live subagents. Results arrive in the conversation.')]

    def cursor():
        index = next((i for i, row in enumerate(monitor.entries) if row['subagent_id'] == monitor.selected_id), 0)
        return Point(x=0, y=index)

    roster = Window(FormattedTextControl(roster_text, focusable=True, get_cursor_position=cursor))

    def update_tail():
        row = monitor.selected
        text = read_tail(row.get('live_transcript')) if row else 'This subagent is no longer live.'
        if text != tail.text:
            following = tail.buffer.cursor_position == len(tail.text)
            position = tail.buffer.cursor_position
            tail.text = text
            tail.buffer.cursor_position = len(text) if following else min(position, len(text))

    def header():
        row = monitor.selected
        title = f"Subagents · {len(monitor.entries)} live"
        if state['detail'] and row:
            title += f" · {row['subagent_id']} · {row.get('goal') or ''}"
        return [('class:subagent-dock.heading', _clip(title, app.output.get_size().columns))]

    def footer():
        narrow = app.output.get_size().columns < 60
        if state['confirm']:
            return 'Stop? y yes · Esc cancel' if narrow else 'Stop selected subagent? y confirm · Esc cancel'
        if state['steering']:
            return 'Enter send · Esc cancel' if narrow else 'Enter queues guidance · Esc cancels (does not interrupt)'
        if narrow:
            return 'PgUp/Dn · s steer x stop · Esc' if state['detail'] else '↑↓ · Enter tail · Ctrl+T close'
        return ('Esc roster · PgUp/PgDn tail · s steer · x stop' if state['detail'] else
                '↑/↓ select · Enter tail · s steer · x stop · q/Ctrl+T close')

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
        if monitor.selected:
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
        filter=Condition(lambda: bool(monitor.entries) and not modal_prompt_active(cli)),
    )
