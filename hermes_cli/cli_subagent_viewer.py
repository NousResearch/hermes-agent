"""Full-screen subagent viewer inside the classic CLI's Ctrl+T monitor.

Views are the master conversation followed by every subagent in the monitor's list (live, then
finished). Each view shows its stored conversation as readable entries (``subagent_transcript``);
the selected entry opens or jumps (a tool call shows its result, a delegation task opens that
subagent). The input line sends to the view on screen: a live subagent is steered, the master gets
the text exactly as if typed in the main composer. Each view keeps its own selection and scroll.
Transcripts come from the local session store, so finished and pre-resume subagents read the same
as live ones; the monitor's tick re-reads the view on screen while it can still change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from agent.i18n import t
from hermes_cli.subagent_transcript import Block, build_blocks, render

logger = logging.getLogger(__name__)

MASTER = "master"


@dataclass
class ViewState:
    cursor: int = 0
    top: int = 0
    follow: bool = True  # pinned to the newest entry until the user moves away from it
    expanded: set = field(default_factory=set)


class SubagentViewer:
    def __init__(self, monitor, key: str):
        self.monitor = monitor
        self.cli = monitor.cli
        self.states: dict[str, ViewState] = {}
        self.blocks: list[Block] = []
        self._loaded: Optional[tuple] = None  # (view key, change marker) of the blocks on screen
        self.notice = ""
        self.current = key if key in self.view_keys else MASTER
        self.reload()

    # ── views ────────────────────────────────────────────────────────────────────────────────
    def _subagent_rows(self) -> list[dict]:
        return [row for row in self.monitor.roster if row.get("kind") != "process"]

    @staticmethod
    def _row_key(row: dict) -> str:
        return row.get("child_session_id") or row.get("subagent_id") or ""

    @property
    def view_keys(self) -> list[str]:
        return [MASTER, *[self._row_key(row) for row in self._subagent_rows() if self._row_key(row)]]

    def row(self, key: Optional[str] = None) -> Optional[dict]:
        key = key or self.current
        return next((r for r in self._subagent_rows() if self._row_key(r) == key), None)

    @property
    def state(self) -> ViewState:
        return self.states.setdefault(self.current, ViewState())

    def switch(self, delta: int) -> None:
        keys = self.view_keys
        index = keys.index(self.current) if self.current in keys else 0
        self.jump(keys[(index + delta) % len(keys)])

    def jump(self, key: str) -> None:
        if key in self.view_keys and key != self.current:
            self.current = key
            self.notice = ""
            self.reload()

    def target_label(self) -> str:
        """Who the input line sends to, as the prompt names it."""
        if self.current == MASTER:
            return t("cli.subagent_view.master")
        row = self.row() or {}
        return _short(row.get("goal") or self.current, 40)

    def live(self) -> bool:
        row = self.row()
        return bool(row) and row.get("kind") != "finished"

    # ── data ─────────────────────────────────────────────────────────────────────────────────
    def _session_id(self) -> Optional[str]:
        if self.current == MASTER:
            from hermes_cli.cli_subagent_monitor import master_session_id
            return master_session_id(self.cli)
        row = self.row() or {}
        return row.get("child_session_id")

    def _marker(self, db, session_id) -> tuple:
        row = db.get_session(session_id) or {}
        return (self.current, row.get("message_count"), row.get("last_activity_at"), row.get("ended_at"))

    def reload(self, *, force: bool = True) -> bool:
        """Re-read the view on screen; True when its entries changed. Without ``force`` the store is
        read only when the session row says something changed."""
        db, session_id = getattr(self.cli, "_session_db", None), self._session_id()
        if db is None or not session_id:
            blocks = [Block("event", title=t("cli.subagent_view.no_transcript"))]
            changed = self._loaded != (self.current, None)
            self.blocks, self._loaded = blocks, (self.current, None)
            return changed
        try:
            marker = self._marker(db, session_id)
            if not force and marker == self._loaded:
                return False
            messages = db.get_messages_as_conversation(session_id, include_ancestors=self.current == MASTER)
        except Exception:  # health: allow BLE001 -- a locked store must not tear down the viewer; the next tick retries
            logger.debug("subagent viewer read failed", exc_info=True)
            return False
        blocks = build_blocks(messages, subagent=self.current != MASTER)
        if self.current == MASTER:
            self._link_delegations(blocks)
        if not blocks:
            blocks = [Block("event", title=t("cli.subagent_view.no_transcript"))]
        changed = [(b.kind, b.title, b.body, b.outcome) for b in blocks] != [
            (b.kind, b.title, b.body, b.outcome) for b in self.blocks]
        self.blocks, self._loaded = blocks, marker
        state = self.state
        if state.follow or state.cursor >= len(blocks):
            state.cursor = len(blocks) - 1
        return changed

    def _link_delegations(self, blocks: list[Block]) -> None:
        """Point each delegation task at the subagent that ran it (same goal, first unused match).
        A background task still marked running is shown done once that subagent has finished."""
        from hermes_cli.subagent_transcript import delegation_title
        rows = [r for r in self._subagent_rows() if not r.get("depth")] or self._subagent_rows()  # direct children
        used: set = set()
        for block in blocks:
            if block.kind != "delegation":
                continue
            goal = " ".join(block.goal.split())
            match = next((r for r in rows if self._row_key(r) not in used
                          and " ".join(str(r.get("goal") or "").split()) == goal), None)
            if match is None:
                continue
            used.add(self._row_key(match))
            block.link = self._row_key(match)
            if block.outcome == "running" and match.get("kind") == "finished" and match.get("ended"):
                block.outcome = "completed"
                block.title = delegation_title(block.goal, block.outcome)

    def poll(self) -> bool:
        """The monitor's tick: re-read the view on screen when its session changed."""
        return self.reload(force=False)

    # ── navigation ───────────────────────────────────────────────────────────────────────────
    def move(self, delta: int) -> None:
        state = self.state
        state.cursor = max(0, min(len(self.blocks) - 1, state.cursor + delta))
        state.follow = state.cursor == len(self.blocks) - 1

    def scroll(self, delta: int) -> None:
        state = self.state
        state.top = max(0, state.top + delta)
        state.follow = False
        state.cursor = -1  # scrolling detaches the selection from the visible window

    def activate(self) -> None:
        """Enter on the selected entry: open or close a tool call or task, or jump to the task's subagent."""
        state = self.state
        if not 0 <= state.cursor < len(self.blocks):
            return
        block = self.blocks[state.cursor]
        if block.kind == "delegation" and block.link:
            self.jump(block.link)
        elif block.collapsible:
            state.expanded ^= {state.cursor}

    def visible_lines(self, width: int, height: int) -> list:
        """The window of screen lines to paint, keeping the selected entry in view."""
        state = self.state
        cursor = state.cursor if 0 <= state.cursor < len(self.blocks) else None
        rendered = render(self.blocks, width=width, expanded=state.expanded, cursor=cursor,
                          subagent=self.current != MASTER)
        total, height = len(rendered.lines), max(1, height)
        if state.follow:
            state.top = max(0, total - height)
        elif cursor is not None:
            start = rendered.starts[cursor]
            end = rendered.starts[cursor + 1] - 1 if cursor + 1 < len(rendered.starts) else total
            if start < state.top:
                state.top = start
            elif end > state.top + height:
                state.top = max(start, end - height) if end - start <= height else start
        state.top = max(0, min(state.top, max(0, total - height)))
        return rendered.lines[state.top:state.top + height]

    # ── sending ──────────────────────────────────────────────────────────────────────────────
    def send(self, text: str) -> dict:
        """Send ``text`` to the view on screen: the master as composer input, a live subagent as
        steering. A finished subagent refuses."""
        text = text.strip()
        if not text:
            return {"error": t("cli.subagent_view.nothing_to_send")}
        if self.current == MASTER:
            return send_to_master(self.cli, text)
        row = self.row()
        if not row:
            return {"error": t("cli.subagents.entry_gone")}
        return self.monitor.control("steer", text, target=row["subagent_id"])  # refuses a finished row

    def stop(self) -> dict:
        row = self.row()
        if self.current == MASTER or not row:
            return {"error": t("cli.subagent_view.nothing_to_stop")}
        return self.monitor.control("stop", None, target=row["subagent_id"])


def send_to_master(cli, text: str) -> dict:
    """Route ``text`` exactly as the main composer would: the busy-input policy while a turn runs,
    otherwise the next prompt."""
    if getattr(cli, "_agent_running", False):
        cli._tui_enter_while_busy(text, [], text)
    else:
        cli._pending_input.put(text)
    return {"note": t("cli.subagent_view.sent_to_master")}


def _short(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def viewer_ui(monitor, state: dict, get_app, run_control):
    """The viewer's widgets for the monitor application: ``(container, entry, key_bindings,
    header, footer)``. ``state`` is the monitor app's shared dict (``confirm``/``notice``);
    ``run_control`` runs a callable off the UI loop and shows its answer in the notice line."""
    from prompt_toolkit.filters import Condition, has_focus
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea

    def prompt():
        viewer = monitor.viewer
        label = viewer.target_label() if viewer else ""
        return [("class:subagent-view.prompt", t("cli.subagent_view.input_prompt", target=label))]

    entry = TextArea(height=1, multiline=False, prompt=prompt)

    def body():
        viewer = monitor.viewer
        if viewer is None:
            return []
        size = get_app().output.get_size()
        fragments = []
        for line in viewer.visible_lines(size.columns, size.rows - 4):
            fragments.extend(line)
            fragments.append(("", "\n"))
        return fragments[:-1]

    container = HSplit([Window(FormattedTextControl(body), wrap_lines=False), entry])
    viewing = Condition(lambda: monitor.viewer is not None) & Condition(lambda: not state.get("confirm"))
    kb = KeyBindings()

    @kb.add("tab", filter=viewing)
    @kb.add("escape", "right", filter=viewing)
    def next_view(event):
        monitor.viewer.switch(1)

    @kb.add("s-tab", filter=viewing)
    @kb.add("escape", "left", filter=viewing)
    def previous_view(event):
        monitor.viewer.switch(-1)

    @kb.add("up", filter=viewing)
    def up(event):
        monitor.viewer.move(-1)

    @kb.add("down", filter=viewing)
    def down(event):
        monitor.viewer.move(1)

    @kb.add("pageup", filter=viewing)
    def page_up(event):
        monitor.viewer.scroll(-max(1, get_app().output.get_size().rows - 6))

    @kb.add("pagedown", filter=viewing)
    def page_down(event):
        monitor.viewer.scroll(max(1, get_app().output.get_size().rows - 6))

    @kb.add("enter", filter=viewing & has_focus(entry))
    def enter(event):
        viewer = monitor.viewer
        text = entry.text
        if not text.strip():
            viewer.activate()
            return
        entry.text = ""
        if viewer.current == MASTER:
            result = viewer.send(text)
            state["notice"] = result.get("error") or result.get("note") or ""
        else:
            run_control(lambda: viewer.send(text))

    @kb.add("c-x", filter=viewing)
    def stop(event):
        if monitor.viewer.current != MASTER and monitor.viewer.live():
            state["confirm"] = True
            state["confirm_action"] = monitor.viewer.stop

    def header():
        viewer = monitor.viewer
        return tab_strip(viewer, get_app().output.get_size().columns) if viewer else []

    def footer():
        narrow = get_app().output.get_size().columns < 80
        return t("cli.subagent_view.footer_compact" if narrow else "cli.subagent_view.footer")

    return container, entry, kb, header, footer


def tab_strip(viewer: SubagentViewer, width: int) -> list:
    """One header line naming every view, the current one highlighted, scrolled to keep it visible."""
    labels = []
    for index, key in enumerate(viewer.view_keys):
        if key == MASTER:
            label = t("cli.subagent_view.master")
        else:
            row = viewer.row(key) or {}
            glyph = "✓" if row.get("kind") == "finished" else "●"
            label = f"{glyph} {index} {_short(row.get('goal') or key, 24)}"
        labels.append((key, f" {label} "))
    current = next((i for i, (key, _) in enumerate(labels) if key == viewer.current), 0)
    start = current
    used = len(labels[current][1])
    while start > 0 and used + len(labels[start - 1][1]) + 1 <= width:
        start -= 1
        used += len(labels[start][1]) + 1
    fragments, cells = [], 0
    for key, label in labels[start:]:
        if cells + len(label) > width:
            break
        style = "class:subagent-view.tab-current" if key == viewer.current else "class:subagent-view.tab"
        fragments.append((style, label))
        fragments.append(("", "│"))
        cells += len(label) + 1
    return fragments[:-1] if fragments else fragments
