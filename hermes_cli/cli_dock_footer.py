"""Live-work footer under the classic CLI composer: a one-line summary pill plus one row per
subagent / background process, in the glanceable shape of Claude Code's task footer.

``Down`` on an empty composer moves the selection into the rows, ``Enter`` opens the selected
row's live log in the full-screen monitor (``Esc`` comes straight back), ``x`` twice stops it, ``s``
steers a subagent from a one-line input, ``Tab`` jumps to the next kind (agents, shells, monitors)
and ``Esc`` / ``Up`` past the first row returns to the composer. With mouse support on
(``display.cli_mouse`` / ``/mouse``), a click on a pill badge selects that kind's first row, the
rest of the pill toggles the selection and a click on a row opens it. Plugin footer items
(``ctx.register_footer_item``, ``cli_plugin_footer``) add a pill part and rows after the tasks;
on their rows ``Enter`` / a click runs the row's Enter action and its letter keys run the others.
The model is ``cli_subagent_monitor.SubagentMonitor``; nothing here changes agent-loop state.
"""
from __future__ import annotations

from prompt_toolkit.utils import get_cwidth

from agent.i18n import t
from hermes_cli import cli_process_dock as procs

AGENT_GLYPH = "☤"
SHELL_GLYPH = "⚙"
MONITOR_GLYPH = "◉"
PLUGIN_GLYPH = "◇"
_TONE_STYLES = {"ok": "class:live-dock.ok", "warn": "class:live-dock.warn", "bad": "class:live-dock.bad",
                "dim": "class:live-dock.dim"}
_POINTER = "❯ "


def is_monitor(row) -> bool:
    """A background process the agent watches (``watch_patterns``): Hermes's monitor."""
    return row.get("kind") == "process" and bool(row.get("watching"))


def one_line(value) -> str:
    """``value`` as one display line: whitespace runs become a space, control characters go."""
    return "".join(c for c in " ".join(str(value or "").split()) if c.isprintable())


def clip(value, width):
    text = one_line(value)
    if get_cwidth(text) <= width:
        return text
    out = ""
    for char in text:
        if get_cwidth(out + char) > max(0, width - 1):
            break
        out += char
    return out + ("…" if width > 0 else "")


def elapsed(seconds) -> str:
    seconds = max(0, int(seconds or 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def row_kind(row) -> str:
    """The pill badge a row counts under: ``agent``, ``shell``, ``monitor``, or ``done``."""
    if row.get("kind") != "process":
        return "agent"
    if row.get("status") != "running":
        return "done"
    return "monitor" if is_monitor(row) else "shell"


_KIND_ORDER = ("agent", "shell", "monitor", "done")


def first_row_of_kind(monitor, kind):
    """Key of the first roster row under ``kind``'s badge, or None."""
    return next((monitor._key(r) for r in monitor.roster if row_kind(r) == kind), None)


def _plural(key, count):
    return t(f"cli.subagents.{key}_{'one' if count == 1 else 'other'}", count=count)


# -- the pill ------------------------------------------------------------------------------------

def _count_parts(monitor):
    """``(style, text, kind)`` count fragments: agents, shells, monitors, finished, goal, loop,
    queue; ``kind`` names the rows a badge click selects (None: not a row badge)."""
    running = [r for r in monitor.processes if r["status"] == "running"]
    watchers = sum(is_monitor(r) for r in running)
    parts = []
    if monitor.entries:
        parts.append(("class:live-dock.agent", f"{AGENT_GLYPH} " + _plural("pill_agents", len(monitor.entries)), "agent"))
    if len(running) - watchers:
        parts.append(("class:live-dock.count", f"{SHELL_GLYPH} " + _plural("pill_shells", len(running) - watchers), "shell"))
    if watchers:
        parts.append(("class:live-dock.count", f"{MONITOR_GLYPH} " + _plural("pill_monitors", watchers), "monitor"))
    if len(monitor.processes) > len(running):
        parts.append(("class:live-dock.dim", t("cli.subagents.count_done", count=len(monitor.processes) - len(running))))
    if monitor.goal:
        key = ("goal_parked" if monitor.goal.startswith("⏳") else
               "goal_paused" if monitor.goal.startswith("⏸") else "goal_active")
        parts.append(("class:live-dock.count", t(f"cli.subagents.{key}")))
    if monitor.loop_badge:
        parts.append(("class:live-dock.count", monitor.loop_badge))
    if monitor.queued:
        parts.append(("class:live-dock.count", t("cli.subagents.count_queued", count=len(monitor.queued))))
    parts += [(_TONE_STYLES[item["tone"]], item["label"]) for item in monitor.plugin_items if item["label"]]
    return [part if len(part) == 3 else (*part, None) for part in parts]


def _is_plugin(row) -> bool:
    return bool(row) and row.get("kind") == "plugin"


def _plugin_hint(row) -> str:
    keys = " · ".join(f"{'↵' if a['key'] == 'enter' else a['key']} {a['label']}" for a in row["actions"])
    return t("cli.subagents.plugin_select_hint", actions=keys) if keys else t("cli.subagents.select_hint_compact")


def _hints(monitor, selecting):
    """Hint texts, longest first; the pill keeps the first that fits."""
    if selecting and monitor.steer_target is not None:
        return (t("cli.subagents.footer_steering"), t("cli.subagents.footer_steering_compact"))
    if monitor.notice:
        return (monitor.notice,)
    if selecting:
        row = monitor.selected
        if _is_plugin(row):
            action = monitor.confirm_action
            if action is not None and monitor.confirm_key == monitor.selected_id:
                return (t("cli.subagents.plugin_confirm", action=action["label"], letter=action["key"]),)
            return (_plugin_hint(row),)
        if monitor.confirm_key is not None and monitor.confirm_key == monitor.selected_id:
            noun = t("cli.subagents.noun_process" if monitor.selected_process else "cli.subagents.noun_subagent")
            return (t("cli.subagents.confirm_stop", noun=noun),)
        if monitor.selected is not None and monitor.selected_process is None:
            return (t("cli.subagents.select_hint_steer"), t("cli.subagents.select_hint_steer_compact"))
        return (t("cli.subagents.select_hint"), t("cli.subagents.select_hint_compact"))
    toggle = t("cli.subagents.hint_show" if monitor.collapsed else "cli.subagents.hint_hide")
    if not monitor.roster:
        return (toggle,)
    hint = t("cli.subagents.pill_hint", toggle=toggle)
    # Lowest priority: only where the full hint still fits, and only while clicks are off.
    mouse = () if getattr(monitor.cli, "_mouse_enabled", True) else (f"{hint} · {t('cli.subagents.hint_mouse')}",)
    return (*mouse, hint, t("cli.subagents.pill_hint_compact"))


def pill_fragments(monitor, columns, *, selecting=False, on_click=None, on_badge=None):
    """``on_click(event)`` handles the pill; ``on_badge(event, kind)`` a row-kind badge."""
    sep = ("class:live-dock.dim", " · ", None)
    body = []
    # Plugin labels, action labels and notices are measured as the single line they paint as.
    for style, text, kind in _count_parts(monitor):
        body += [sep, (style, one_line(text), kind)] if body else [(style, one_line(text), kind)]
    used = 2 + sum(get_cwidth(text) for _, text, _ in body)
    for hint in map(one_line, _hints(monitor, selecting)):
        if used + 3 + get_cwidth(hint) <= columns:
            body += [("class:live-dock.dim", "   ", None), ("class:live-dock.hint", hint, None)]
            used += 3 + get_cwidth(hint)
            break
    # Collapsed with no task rows: the goal / loop prompt / next queued prompt is the only
    # reminder of what is pending, so it takes whatever room is left.
    preview = (monitor.goal or monitor.loop_prompt
               or (t("cli.subagents.next_queued", text=monitor.queued[0]) if monitor.queued else ""))
    room = columns - used - 3
    if monitor.collapsed and not monitor.roster and preview and room >= 12:
        body += [("class:live-dock.dim", " · ", None), ("", clip(preview, room), None)]
    indent = "  "[:max(0, columns)]
    fragments, width = [("", indent, None)], len(indent)
    for style, text, kind in body:
        room = columns - width
        if room <= 0:
            break
        text = text if get_cwidth(text) <= room else clip(text, room)
        fragments.append((style, text, kind))
        width += get_cwidth(text)
    if on_click is None and on_badge is None:
        return [(style, text) for style, text, _ in fragments]
    return [(style, text, _fragment_click(kind, on_click, on_badge)) for style, text, kind in fragments]


def _fragment_click(kind, on_click, on_badge):
    if kind is None or on_badge is None:
        return on_click
    return lambda event: on_badge(event, kind)


# -- rows ----------------------------------------------------------------------------------------

def _agent_cells(row):
    activity = row.get("last_tool") or row.get("status") or t("cli.subagents.starting")
    return ("class:live-dock.agent", AGENT_GLYPH, row.get("goal") or row.get("subagent_id") or "",
            f"{activity} · {elapsed(row.get('elapsed'))}")


def _process_cells(row):
    if row["status"] != "running":
        glyph_style = "class:live-dock.ok" if row["status"] == "done" else "class:live-dock.bad"
        return glyph_style, procs.process_glyph(row), row["command"], procs.process_activity(row)
    detail = row.get("detail") or t("cli.dock.starting")
    return ("class:live-dock.count", MONITOR_GLYPH if is_monitor(row) else SHELL_GLYPH, row["command"],
            f"{detail} · {elapsed(row.get('elapsed'))}")


def _plugin_cells(row):
    return _TONE_STYLES.get(row.get("tone"), "class:live-dock.dim"), PLUGIN_GLYPH, row["label"], row["detail"]


def row_fragments(row, columns, *, selected, on_click=None):
    kind = row.get("kind")
    glyph_style, glyph, subject, activity = (
        _process_cells(row) if kind == "process" else _plugin_cells(row) if kind == "plugin" else _agent_cells(row))
    lead = _POINTER if selected else "  "
    right_width = min(get_cwidth(activity), max(0, columns // 3))
    right = clip(activity, right_width) if right_width >= 6 else ""
    subject_width = max(3, columns - get_cwidth(lead) - 2 - (get_cwidth(right) + 2 if right else 0) - 1)
    subject = clip(subject, subject_width)
    pad = columns - get_cwidth(lead) - 2 - get_cwidth(subject) - get_cwidth(right) - 1
    text_style = "class:live-dock.selected" if selected else ""
    fragments = [("class:live-dock.selected" if selected else "class:live-dock.dim", lead),
                 (glyph_style, glyph + " "), (text_style, subject)]
    if right and pad >= 1:
        fragments += [("", " " * pad), ("class:live-dock.dim", right)]
    if on_click is not None:
        fragments = [(style, text, on_click) for style, text in fragments]
    return fragments


def _steer_line(monitor, columns):
    """The one-line steer input painted under the rows while ``s`` is composing."""
    prompt = "  " + t("cli.subagents.steer_prompt")
    room = max(1, columns - get_cwidth(prompt) - 1)
    return [("class:live-dock.hint", prompt), ("", _tail_fitting(monitor.steer_draft, room)), ("reverse", " ")]


def _tail_fitting(text, width):
    """The longest suffix of ``text`` at most ``width`` cells wide (the tail, where typing happens,
    stays in view). One backward pass: a long paste must not cost a re-measure per dropped char."""
    used = 0
    for index in range(len(text) - 1, -1, -1):
        used += get_cwidth(text[index])
        if used > width:
            return text[index + 1:]
    return text


def _row_budget(rows):
    """Task rows shown at once: 4 on a 30-row terminal, 1 on a 14-row one (the rest scroll)."""
    return max(1, min(4, (rows - 12) // 4))


def dock_lines(monitor, *, columns, rows, selecting=False, actions=None):
    """Footer lines as fragment lists (the widget joins them with newlines). ``actions`` maps
    ``'pill'`` / ``'row'`` to mouse-click callbacks. With nothing to list, the footer is the
    rotating tip line (``cli_footer_tips``) or empty."""
    if not monitor.has_rows:
        from hermes_cli import cli_footer_tips
        if cli_footer_tips.showing(monitor.cli, rows):
            return [line] if (line := monitor.tips.line(columns)) else []
        return []
    columns = max(0, columns - 1)
    roster = monitor.roster
    lines = []
    # The pill heads live tasks; a goal/loop/queue-only footer paints its own rows (as before)
    # and only folds into the pill when collapsed.
    if roster or monitor.plugin_items or (monitor.collapsed and not selecting):
        lines.append(pill_fragments(monitor, columns, selecting=selecting, on_click=(actions or {}).get("pill"),
                                    on_badge=(actions or {}).get("badge")))
    if monitor.collapsed and not selecting:
        return lines
    lines += [[("class:live-dock.dim", "  " + clip(text, columns - 2))] for text in (monitor.goal, monitor.loop) if text]
    budget = _row_budget(rows)
    start = 0
    if selecting and monitor.selected_id is not None:
        index = next((i for i, r in enumerate(roster) if monitor._key(r) == monitor.selected_id), 0)
        start = max(0, index - budget + 1)
    shown_rows = roster[start:start + budget]
    for row in shown_rows:
        key = monitor._key(row)
        click = (lambda event, key=key: actions["row"](event, key)) if actions and "row" in actions else None
        lines.append(row_fragments(row, columns, selected=selecting and key == monitor.selected_id, on_click=click))
    hidden = [r for r in roster if r not in shown_rows]
    if hidden:
        kinds = {r.get("kind") for r in hidden}
        key = ("more_queued" if "plugin" in kinds else "more_processes" if kinds == {"process"}
               else "more_subagents")
        lines.append([("class:live-dock.dim", "  " + clip(t(f"cli.subagents.{key}", count=len(hidden)), columns - 2))])
    if selecting and monitor.steer_target is not None:
        lines.append(_steer_line(monitor, columns))
    if monitor.queued:
        from hermes_cli.cli_session_dock import QUEUE_ROWS
        shown = min(len(monitor.queued), QUEUE_ROWS if rows >= 24 else 1)
        controls = "" if roster else t("cli.subagents.controls_collapse")
        lines.append([("class:live-dock.dim", "  " + clip(
            t("cli.subagents.queue_heading", count=len(monitor.queued), controls=controls), columns - 2))])
        lines += [[("class:live-dock.dim", "    " + clip(f"{i}. {text}", columns - 4))]
                  for i, text in enumerate(monitor.queued[:shown], 1)]
        if len(monitor.queued) > shown:
            lines.append([("class:live-dock.dim", "    " + clip(
                t("cli.subagents.more_queued", count=len(monitor.queued) - shown), columns - 4))])
    return lines


def plain_text(lines) -> str:
    return "\n".join("".join(fragment[1] for fragment in line) for line in lines)


# -- selection and keys --------------------------------------------------------------------------

def _monitor(cli):
    return getattr(cli, "_subagent_monitor", None)


def _composer_idle(cli) -> bool:
    from prompt_toolkit.application import get_app_or_none
    app = get_app_or_none()
    buffer = app.current_buffer if app is not None else None
    if buffer is None or buffer.text or buffer.complete_state is not None:
        return False
    stash = getattr(cli, "_prompt_stash", None)
    return not getattr(stash, "panel_open", False) and not getattr(cli, "_connection_state", None)


def selecting(cli) -> bool:
    """The footer owns the arrows: entered with Down and the composer is still empty. Typing
    (or a prompt opening) hands them back for good, not just until the composer is empty again."""
    from hermes_cli.cli_subagent_monitor import modal_prompt_active
    monitor = _monitor(cli)
    if monitor is None or not monitor.footer_active:
        return False
    if monitor.roster and _composer_idle(cli) and not modal_prompt_active(cli):
        return True
    leave(cli)
    return False


def steering(cli) -> bool:
    """The footer's one-line steer input owns the keyboard."""
    return selecting(cli) and _monitor(cli).steer_target is not None


def can_enter(cli) -> bool:
    from hermes_cli.cli_subagent_monitor import modal_prompt_active
    monitor = _monitor(cli)
    return bool(monitor is not None and not monitor.footer_active and monitor.roster
                and _composer_idle(cli) and not modal_prompt_active(cli))


def enter(cli, key=None):
    monitor = _monitor(cli)
    if monitor is None or not monitor.roster:
        return
    monitor.footer_active = True
    monitor.notice = ""
    monitor.confirm_key = monitor.confirm_action = None
    _end_steer(monitor)
    # Down always lands on the first row, so one Up always leaves again.
    monitor.selected_id = key if key is not None else monitor._key(monitor.roster[0])


def leave(cli):
    monitor = _monitor(cli)
    if monitor is not None:
        monitor.footer_active = False
        monitor.confirm_key = monitor.confirm_action = None
        monitor.notice = ""
        _end_steer(monitor)


def move(cli, delta):
    monitor = _monitor(cli)
    roster = monitor.roster
    index = next((i for i, r in enumerate(roster) if monitor._key(r) == monitor.selected_id), 0)
    monitor.confirm_key = monitor.confirm_action = None
    monitor.notice = ""
    if index + delta < 0:
        leave(cli)
        return
    monitor.selected_id = monitor._key(roster[min(len(roster) - 1, index + delta)])


def next_kind(cli, delta=1):
    """Tab / Shift+Tab: the first row of the next (previous) kind present in the roster, in pill
    order (agents, shells, monitors, finished), wrapping around."""
    monitor = _monitor(cli)
    present = [kind for kind in _KIND_ORDER if first_row_of_kind(monitor, kind) is not None]
    if not present:
        return
    current = row_kind(monitor.selected) if monitor.selected is not None else present[0]
    index = present.index(current) if current in present else -delta
    monitor.selected_id = first_row_of_kind(monitor, present[(index + delta) % len(present)])
    monitor.confirm_key = None
    monitor.notice = ""


def _end_steer(monitor):
    monitor.steer_target = None
    monitor.steer_draft = ""


def start_steer(cli):
    """``s``: open the one-line steer input for the selected subagent (processes can't be steered)."""
    monitor = _monitor(cli)
    monitor.confirm_key = None
    if monitor.selected is None:
        return
    if monitor.selected_process is not None:
        monitor.notice = t("cli.subagents.cannot_steer_process")
        return
    monitor.notice = ""
    monitor.steer_target = monitor.selected_id
    monitor.steer_draft = ""


def steer_type(cli, text):
    monitor = _monitor(cli)
    monitor.steer_draft += "".join(c for c in " ".join(str(text).splitlines()) if c.isprintable())


def steer_backspace(cli):
    monitor = _monitor(cli)
    monitor.steer_draft = monitor.steer_draft[:-1]


def steer_cancel(cli):
    _end_steer(_monitor(cli))


def _notice(result):
    return str(result.get("error") or result.get("note") or result.get("status") or "")


def _send_control(monitor, action, target, message=None):
    """Steer / stop ``target`` off the UI thread (a gateway RPC or a shell's shutdown grace must
    not freeze typing); the outcome lands in the pill's notice."""
    def done(result):
        monitor.notice = _notice(result)
    monitor.run_control(action, target=target, message=message, done=done)


def steer_send(cli):
    """Enter: queue the draft as guidance for the target subagent; an empty draft does nothing.
    The target is the row ``s`` opened the input on, read once: a refresh that closed the input
    meanwhile (the child finished) leaves nothing to send, never the newly selected row."""
    monitor = _monitor(cli)
    target, message = monitor.steer_target, monitor.steer_draft.strip()
    if target is None or not message:
        return
    _end_steer(monitor)
    _send_control(monitor, "steer", target, message)


def plugin_key(cli, letter) -> bool:
    """Whether ``letter`` runs an action of the selected plugin row (a key binding filter)."""
    from hermes_cli.cli_plugin_footer import action_for
    monitor = _monitor(cli)
    row = monitor.selected if monitor is not None else None
    return _is_plugin(row) and action_for(row, letter) is not None


def run_plugin_action(cli, letter, key=None):
    """Run the plugin row's action for ``letter`` (``'enter'``: its Enter action, else the first).
    An action marked ``confirm`` asks for the same key twice, like stopping a task."""
    from hermes_cli import cli_plugin_footer as plugin_footer
    monitor = _monitor(cli)
    key = key if key is not None else monitor.selected_id
    row = next((r for r in monitor.roster if monitor._key(r) == key), None)
    action = plugin_footer.action_for(row, letter) if _is_plugin(row) else None
    if action is None:
        return None
    if action.get("confirm") and not (monitor.confirm_key == key and monitor.confirm_action == action):
        monitor.confirm_key, monitor.confirm_action, monitor.notice = key, action, ""
        return None
    monitor.confirm_key = monitor.confirm_action = None
    monitor.notice = ""
    return plugin_footer.run_action(cli, monitor, row, action)


def open_selected(cli, key=None):
    from hermes_cli.cli_subagent_monitor import open_monitor
    monitor = _monitor(cli)
    key = key if key is not None else monitor.selected_id
    row = next((r for r in monitor.roster if monitor._key(r) == key), None)
    if _is_plugin(row):
        # The plugin's Enter action; the selection stays so its notice and other keys remain.
        run_plugin_action(cli, "enter", key)
        return
    leave(cli)
    if key is not None:
        open_monitor(cli, key=key)


def stop_selected(cli):
    """``x`` arms a stop on the selected row, a second ``x`` on that same row sends it. The key is
    read once, so a refresh that moves the selection meanwhile can never redirect the stop."""
    monitor = _monitor(cli)
    key = monitor.selected_id
    if key is None or _is_plugin(monitor.selected):
        return  # a plugin row's own ``x`` action is one of its letter keys
    if monitor.confirm_key != key:
        monitor.confirm_key = key
        return
    monitor.confirm_key = None
    _send_control(monitor, "stop", key)


def click_actions(cli):
    """Mouse-click callbacks for the dock fragments (only delivered with ``display.cli_mouse``)."""
    from prompt_toolkit.mouse_events import MouseEventType

    def pill(event):
        if event.event_type != MouseEventType.MOUSE_UP:
            return NotImplemented
        monitor = _monitor(cli)
        if monitor is not None and monitor.roster and not monitor.footer_active:
            enter(cli)
        else:
            leave(cli)
        return None

    def badge(event, kind):
        if event.event_type != MouseEventType.MOUSE_UP:
            return NotImplemented
        key = first_row_of_kind(_monitor(cli), kind) if _monitor(cli) is not None else None
        return pill(event) if key is None else enter(cli, key)

    def row(event, key):
        if event.event_type != MouseEventType.MOUSE_UP:
            return NotImplemented
        monitor = _monitor(cli)
        clicked = next((r for r in monitor.roster if monitor._key(r) == key), None) if monitor else None
        if _is_plugin(clicked) and not (monitor.footer_active and monitor.selected_id == key):
            # Select the clicked row first, so a confirm-required action's question is painted on
            # it and only a second click on that same row answers it.
            enter(cli, key)
        open_selected(cli, key)
        return None

    return {"pill": pill, "badge": badge, "row": row}


def _chrome_reaches_bottom(renderer, screen) -> bool:
    """Whether the chrome sits on the terminal's bottom rows: it fills the screen, startup pinned it
    there, a refill left its top at a known row, or the output recorded since the last clear fills
    every row above it. After a clear (``/clear``, a scrollback rebuild) a short chrome sits under
    the top rows with blank ones below, where a click mapped from the bottom row lands on the wrong
    line, so clicks are dropped until the transcript fills those rows. Not covered: blank rows a
    print leaves when the chrome shrinks at the same time (no CPR tells), which ``_transcript_room``
    assumes away too."""
    from hermes_cli.cli_render import _chrome_floor, _chrome_pinned, _cli, _output_history_rows, _terminal_reflows
    size = renderer.output.get_size()
    room = size.rows - screen.height
    if room <= 0 or _chrome_pinned() or _chrome_floor() is not None:
        return True
    if not _cli()._OUTPUT_HISTORY_ENABLED:
        return False
    return _output_history_rows(room, size.columns, _terminal_reflows() is not True) is None


def _bottom_anchored_mouse(pt_handler):
    """prompt_toolkit drops clicks until it knows the rows above its layout, which only a cursor
    position report tells it, and Hermes turns CPR off (delayed replies leak into the input). Once
    the transcript has pushed the chrome down to the terminal's bottom row (``_transcript_room``
    counts the same way), map the click from there; until then clicks are dropped as before (the
    keyboard still reaches every row). The guess is restored at once and never changes a paint."""
    def handler(event):
        renderer = event.app.renderer
        screen = renderer._last_screen
        if renderer.height_is_known or screen is None:
            return pt_handler(event)
        if not _chrome_reaches_bottom(renderer, screen):
            return NotImplemented
        renderer._min_available_height = screen.height
        try:
            return pt_handler(event)
        finally:
            renderer._min_available_height = 0
    return handler


def bind_keys(cli, kb):
    """Dock shortcuts: Ctrl+T/F6 monitor, Ctrl+R/F7 collapse, and the Down-to-select footer."""
    from prompt_toolkit.filters import Condition
    from hermes_cli.cli_subagent_monitor import modal_prompt_active, open_monitor, toggle_dock

    free = Condition(lambda: not modal_prompt_active(cli))
    for key in ("c-t", "f6"):
        kb.add(key, filter=free)(lambda event: open_monitor(cli))
    # F7 is retained for terminals that forward the function-key sequence; Ctrl+R is the portable
    # fallback because macOS often reserves F7 for a hardware control unless Fn/globe is held.
    for key in ("c-r", "f7"):
        kb.add(key, filter=free)(lambda event: toggle_dock(cli))
    # Registered after the composer's history bindings, so these win while their filter holds.
    typing = Condition(lambda: steering(cli))
    active = Condition(lambda: selecting(cli)) & ~typing
    kb.add("down", filter=Condition(lambda: can_enter(cli)))(lambda event: enter(cli))
    kb.add("up", filter=active, eager=True)(lambda event: move(cli, -1))
    kb.add("down", filter=active, eager=True)(lambda event: move(cli, 1))
    kb.add("tab", filter=active, eager=True)(lambda event: next_kind(cli, 1))
    kb.add("s-tab", filter=active, eager=True)(lambda event: next_kind(cli, -1))
    kb.add("enter", filter=active, eager=True)(lambda event: open_selected(cli))
    # Stop and steer act on task rows only; on a plugin row a letter is captured only when the row
    # advertises it (below), so an unadvertised one reaches the composer like any other letter.
    task_row = Condition(lambda: not _is_plugin(_monitor(cli).selected))
    kb.add("x", filter=active & task_row, eager=True)(lambda event: stop_selected(cli))
    kb.add("s", filter=active & task_row, eager=True)(lambda event: start_steer(cli))
    kb.add("escape", filter=active, eager=True)(lambda event: leave(cli))
    _bind_steer_keys(cli, kb, typing)
    # Plugin rows: their own letter keys while selected.
    for letter in "abcdefghijklmnopqrstuvwxyz":
        kb.add(letter, filter=active & Condition(lambda letter=letter: plugin_key(cli, letter)), eager=True)(
            lambda event, letter=letter: run_plugin_action(cli, letter))
    # Only reached when display.cli_mouse asked the terminal for mouse reports.
    from prompt_toolkit.key_binding.bindings.mouse import load_mouse_bindings
    from prompt_toolkit.keys import Keys
    pt_mouse = next(b.handler for b in load_mouse_bindings().bindings if b.keys == (Keys.Vt100MouseEvent,))
    kb.add(Keys.Vt100MouseEvent)(_bottom_anchored_mouse(pt_mouse))



def _bind_steer_keys(cli, kb, typing):
    """The footer steer input reads keys itself, so the composer's draft and history stay put."""
    from prompt_toolkit.keys import Keys

    def any_key(event):
        key = event.key_sequence[0].key if event.key_sequence else None
        if key in ("c-c", Keys.ControlC, Keys.SIGINT):
            steer_cancel(cli)
        elif event.data and event.data.isprintable() and len(event.data) == 1:
            steer_type(cli, event.data)

    kb.add(Keys.Any, filter=typing, eager=True)(any_key)
    for key in ("tab", "s-tab"):  # never reach the composer's completion while typing here
        kb.add(key, filter=typing, eager=True)(lambda event: None)
    kb.add(Keys.BracketedPaste, filter=typing, eager=True)(lambda event: steer_type(cli, event.data))
    kb.add("backspace", filter=typing, eager=True)(lambda event: steer_backspace(cli))
    kb.add("enter", filter=typing, eager=True)(lambda event: steer_send(cli))
    kb.add("escape", filter=typing, eager=True)(lambda event: steer_cancel(cli))
