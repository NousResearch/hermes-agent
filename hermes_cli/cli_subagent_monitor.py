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

    def tick(self):
        now = time.monotonic()
        if now - self._last_poll < 1:
            return
        self._last_poll = now
        if self.refresh():
            if self.app is not None:
                self.app.invalidate()
            else:
                self.cli._invalidate()

    def select(self, delta):
        if self.entries:
            index = next((i for i, r in enumerate(self.entries) if r['subagent_id'] == self.selected_id), 0)
            self.selected_id = self.entries[(index + delta) % len(self.entries)]['subagent_id']

    def control(self, action, message=None):
        from tools.delegate_tool_registry import _handle_control_action
        return json.loads(_handle_control_action(action, self.selected_id, message, getattr(self.cli, 'agent', None)))

    def dock_text(self, *, columns, rows):
        if not self.entries:
            return ''
        count = min(len(self.entries), max(1, min(4, (rows - 10) // 3)))
        hidden = len(self.entries) - count
        heading = f' Subagents · {len(self.entries)} live · F6 expand'
        lines = [_clip(heading, columns)]
        for row in self.entries[:count]:
            activity = f"{row['elapsed']}s · " + (f"last: {row['last_tool']}" if row['last_tool'] else row.get('status') or 'starting')
            # Reserve activity even on narrow terminals; task names use the remainder.
            goal_width = max(3, columns - get_cwidth(activity) - 5)
            lines.append(_clip(f" ● {_clip(row.get('goal'), goal_width)} · {activity}", columns))
        if hidden:
            lines.append(_clip(f' +{hidden} more · F6 all subagents', columns))
        return '\n'.join(lines)


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
        return [('class:subagent-border', monitor.dock_text(columns=size.columns, rows=size.rows))]

    cli._subagent_dock_widget = ConditionalContainer(
        Window(FormattedTextControl(text), wrap_lines=False),
        filter=Condition(lambda: bool(monitor.entries)),
    )
