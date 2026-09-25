"""The interactive ``hermes plugins`` composite UI: general-plugin checkboxes (saved through the one
admission authority) and the memory-provider / context-engine category pickers.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade re-exports the names other modules use and is
imported late here, never at module level).
"""

from __future__ import annotations

import functools
import sys


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _discover_memory_providers() -> list[tuple[str, str]]:
    """``[(name, description), ...]`` for available memory providers."""
    try:
        from plugins.memory import discover_memory_providers
        return [(name, desc) for name, desc, _avail in discover_memory_providers()]
    except Exception:
        return []


def _discover_context_engines() -> list[tuple[str, str]]:
    """``[(name, description), ...]`` for repo-shipped context engines plus the plugin-registered
    one (``ctx.register_context_engine``); repo-shipped descriptions win on a name collision."""
    engines: dict[str, str] = {}
    try:
        from plugins.context_engine import discover_context_engines
        for name, desc, _avail in discover_context_engines():
            engines.setdefault(name, desc)
    except Exception:
        pass
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_context_engine
        discover_plugins()
        plugin_engine = get_plugin_context_engine()
        if plugin_engine and getattr(plugin_engine, "name", None):
            engines.setdefault(plugin_engine.name, "installed plugin")
    except Exception:
        pass
    return list(engines.items())


# (title, default label, default name, current-value reader, discovery fn, saver) per provider
# category. Readers/savers are looked up at call time so module-level patching still applies.
_PROVIDER_CATEGORY_SPECS = (
    ("Memory Provider", "built-in", "", lambda: _pc()._get_current_memory_provider(),
     lambda: _discover_memory_providers(), lambda v: _pc()._save_memory_provider(v)),
    ("Context Engine", "compressor", "compressor", lambda: _pc()._get_current_context_engine(),
     lambda: _pc()._discover_context_engines(), lambda v: _pc()._save_context_engine(v)),
)


def _configure_category_spec(spec) -> bool:
    """Radio picker for one ``_PROVIDER_CATEGORY_SPECS`` row: the built-in default first, then the
    discovered choices; a current value not among them is appended as ``(not found)``. Saves and
    returns True when the choice changed."""
    from hermes_cli.curses_ui import curses_radiolist
    title, default_label, default_name, current, discover, save = spec
    current = current()
    choices = discover()
    names = [default_name] + [name for name, _desc in choices]
    items = [f"{default_label} (default)"] + [f"{name} \u2014 {desc}" if desc else name for name, desc in choices]
    if current not in names:
        names.append(current)
        items.append(f"{current} (not found)")
    selected = max(i for i, name in enumerate(names) if name == current)
    new_value = names[curses_radiolist(title=f"{title} (select one)", items=items, selected=selected)]
    if new_value == current:
        return False
    save(new_value)
    return True


def _provider_categories() -> list:
    """``[(title, current_label, configure_fn), ...]`` rows for the composite UI."""
    return [(s[0], s[3]() or s[1], functools.partial(_configure_category_spec, s)) for s in _PROVIDER_CATEGORY_SPECS]


def cmd_toggle() -> None:
    """Interactive composite UI — general plugins + provider plugin categories."""
    console = _pc()._console()
    entries = _pc()._discover_all_plugins()
    expected_config = _pc()._plugin_selection_version()
    enabled_set = _pc()._get_enabled_set()
    disabled_set = _pc()._get_disabled_set()

    # Track by CANONICAL KEY, not manifest name: the loader and enable/disable all gate on the
    # key (``web/firecrawl``) while the name may differ (``web-firecrawl``); persisting the bare
    # name let plugins.disabled drift so "explicit disable wins" kept a plugin off forever.
    plugin_keys = [entry[5] for entry in entries]
    # Keys keep every surface aligned. See #40190.
    plugin_labels = [
        (f"{name} \u2014 {description}" if description else name) + (" [bundled]" if source == "bundled" else "")
        for name, _version, description, source, _d, _key in entries
    ]
    # Selected when enabled AND not disabled; the legacy bare name counts on either side.
    plugin_selected = {
        i for i, (name, _v, _desc, _src, _d, key) in enumerate(entries)
        if {key, name} & enabled_set and not ({key, name} & disabled_set)
    }
    categories = _pc()._provider_categories()

    if not sys.stdin.isatty():
        console.print("[dim]Interactive mode requires a terminal.[/dim]")
        return
    try:
        import curses
        _run_composite_ui(curses, plugin_keys, plugin_labels, plugin_selected, disabled_set, categories, console, expected_config=expected_config)
    except ImportError:
        _run_composite_fallback(plugin_keys, plugin_labels, plugin_selected, disabled_set, categories, console, expected_config=expected_config)


def _persist_plugin_selection(plugin_keys, chosen, disabled, *, expected_config=None) -> tuple[bool, set]:
    """Save the composite UI's checkbox state; returns ``(changed, new_enabled)``.

    Unchecked plugins go to the disabled-list (so they stay off even if something auto-enables
    them) under the canonical key ONLY, so the list can't drift from what ``cmd_enable`` clears.
    Re-checking also drops any stale legacy bare-leaf disable.
    """
    # See #40190.
    # Persist by canonical key only — never the bare manifest name — so the disabled-list stays aligned with
    # cmd_enable / PluginManager (#40190).
    if expected_config is None:
        expected_config = _pc()._plugin_selection_version()
    new_enabled: set = set()
    new_disabled: set = set(disabled)  # preserve existing disabled state for unseen plugins
    for i, key in enumerate(plugin_keys):
        if i in chosen:
            new_enabled.add(key)
            _pc()._discard_key_and_leaf(new_disabled, key)
        else:
            new_disabled.add(key)

    changed = new_enabled != _pc()._get_enabled_set() or new_disabled != disabled
    if changed:
        # C13: the composite UI's candidate goes through the ONE admission
        # authority — refusal raises AdmissionRefused BEFORE any config
        # write; the caller surfaces it and the selection stays unsaved.
        _pc()._admit_and_save_plugin_sets(new_enabled, new_disabled, action="Save plugin selection", expected_config=expected_config)
    return changed, new_enabled


def _run_composite_ui(curses, plugin_keys, plugin_labels, plugin_selected, disabled, categories, console, *, expected_config=None):
    """Custom curses screen with checkboxes + category action rows."""
    from hermes_cli.curses_ui import _addnstr, flush_stdin
    chosen = set(plugin_selected)
    n_plugins, n_categories = len(plugin_keys), len(categories)
    total_items = n_plugins + n_categories  # navigable rows (headers/separator are skipped)
    providers_changed = False
    nav = {  # key -> new cursor, given (cursor, page_size)
        key: move
        for keys, move in (
            ((curses.KEY_UP, ord("k")), lambda c, p: (c - 1) % total_items),
            ((curses.KEY_DOWN, ord("j")), lambda c, p: (c + 1) % total_items),
            ((curses.KEY_NPAGE, ord("f")), lambda c, p: min(total_items - 1, c + p)),
            ((curses.KEY_PPAGE, ord("b")), lambda c, p: max(0, c - p)),
            ((curses.KEY_HOME,), lambda c, p: 0),
            ((curses.KEY_END,), lambda c, p: total_items - 1),
        )
        for key in keys
    }

    def _init_colors():
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            gray = 8 if curses.COLORS > 8 else curses.COLOR_WHITE
            for pair, fg in ((1, curses.COLOR_GREEN), (2, curses.COLOR_YELLOW), (3, curses.COLOR_CYAN), (4, gray)):
                curses.init_pair(pair, fg, -1)

    def _attr(base, pair):
        return base | curses.color_pair(pair) if curses.has_colors() else base

    def _row(text, idx, cursor, pair):
        """One navigable body row: arrow marker + bold color when *idx* is the cursor."""
        arrow = "\u2192" if idx == cursor else " "
        return (f" {arrow} {text}", _attr(curses.A_BOLD, pair) if idx == cursor else curses.A_NORMAL)

    def _configure_category(ci):
        """Leave curses, run the category's picker, refresh its row, re-enter curses."""
        nonlocal providers_changed
        curses.endwin()
        cat_name, _cat_cur, cat_fn = categories[ci]
        if cat_fn():
            providers_changed = True
            categories[ci] = (cat_name, _pc()._provider_categories()[ci][1], cat_fn)
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        _init_colors()
        curses.curs_set(0)
        return stdscr

    def _body_lines(cursor, scroll_offset, visible_rows):
        """Body rows as (text, attr); "" is a blank separator."""
        lines = []
        if n_plugins > 0:
            lines.append(("  General Plugins", _attr(curses.A_BOLD, 2)))
            for i in range(scroll_offset, min(n_plugins, scroll_offset + max(visible_rows, 0))):
                check = "\u2713" if i in chosen else " "
                lines.append(_row(f"[{check}] {plugin_labels[i]}", i, cursor, 1))
        lines.append(("", curses.A_NORMAL))
        if n_categories > 0:
            lines.append(("  Provider Plugins", _attr(curses.A_BOLD, 2)))
            lines += [
                _row(f"  {cat_name:<24} \u25b8 {cat_current}", n_plugins + ci, cursor, 3)
                for ci, (cat_name, cat_current, _cat_fn) in enumerate(categories)
            ]
        return lines

    def _draw(stdscr):
        curses.curs_set(0)
        _init_colors()
        cursor = scroll_offset = 0
        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()
            _addnstr(stdscr, 0, 0, "Plugins", max_x - 1, _attr(curses.A_BOLD, 2))
            _addnstr(
                stdscr, 1, 0, "  ↑↓/j/k navigate  PgUp/PgDn page  SPACE toggle  ENTER configure/confirm  ESC done",
                max_x - 1, curses.A_DIM)
            visible_rows = max_y - 4
            if cursor < scroll_offset:
                scroll_offset = cursor
            elif cursor >= scroll_offset + visible_rows:
                scroll_offset = cursor - visible_rows + 1
            lines = _body_lines(cursor, scroll_offset, visible_rows)
            for y, (text, attr) in enumerate(lines[: max(0, max_y - 4)], start=3):
                if text:
                    _addnstr(stdscr, y, 0, text, max_x - 1, attr)
            stdscr.refresh()
            key = stdscr.getch()

            if key in nav:
                if total_items > 0:  # (with no rows, every motion leaves cursor at 0)
                    cursor = nav[key](cursor, max(1, max_y - 5))
            elif key == ord(" ") or key in {curses.KEY_ENTER, 10, 13}:
                if cursor >= n_plugins:
                    # Provider category — launch sub-screen (SPACE and ENTER alike)
                    if cursor - n_plugins < n_categories:
                        stdscr = _configure_category(cursor - n_plugins)
                elif key == ord(" "):
                    chosen.symmetric_difference_update({cursor})
                else:
                    return  # ENTER on a plugin checkbox — confirm and exit
            elif key in {27, ord("q")}:
                return  # plugin changes are saved on exit

    curses.wrapper(_draw)
    flush_stdin()

    from hermes_cli.plugins_admission import AdmissionRefused

    try:
        changed, new_enabled = _persist_plugin_selection(plugin_keys, chosen, disabled, expected_config=expected_config)
    except AdmissionRefused as exc:
        console.print(f"[red]✗[/red] Plugin selection refused, not saved: {exc}")
        console.print(
            "[dim]config.yaml and the active environment are unchanged. "
            "Run `hermes pm install` to resolve, then retry.[/dim]"
        )
        return
    if changed:
        console.print(
            f"\n[green]\u2713[/green] General plugins: {len(new_enabled)} enabled, "
            f"{len(plugin_keys) - len(new_enabled)} disabled.")
    elif n_plugins > 0:
        console.print("\n[dim]General plugins unchanged.[/dim]")
    if providers_changed:
        console.print(
            f"[green]\u2713[/green] Memory provider: [bold]{_pc()._get_current_memory_provider() or 'built-in'}[/bold]  "
            f"Context engine: [bold]{_pc()._get_current_context_engine()}[/bold]")
    if n_plugins > 0 or providers_changed:
        console.print("[dim]Changes take effect on next session.[/dim]")
    console.print()


def _run_composite_fallback(plugin_keys, plugin_labels, plugin_selected, disabled, categories, console, *, expected_config=None):
    """Text-based fallback for the composite plugins UI."""
    from hermes_cli.colors import Colors, color
    print(color("\n  Plugins", Colors.YELLOW))
    if plugin_keys:
        chosen = set(plugin_selected)
        print(color("\n  General Plugins", Colors.YELLOW))
        print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))
        while True:
            for i, label in enumerate(plugin_labels):
                marker = color("[\u2713]", Colors.GREEN) if i in chosen else "[ ]"
                print(f"  {marker} {i + 1:>2}. {label}")
            print()
            try:
                val = input(color("  Toggle # (or Enter to confirm): ", Colors.DIM)).strip()
                if not val:
                    break
                idx = int(val) - 1
                if 0 <= idx < len(plugin_keys):
                    chosen.symmetric_difference_update({idx})
            except (ValueError, KeyboardInterrupt, EOFError):
                return
            print()
        _save_plugin_selection_fallback(plugin_keys, chosen, disabled, expected_config=expected_config)

    if categories:
        print(color("\n  Provider Plugins", Colors.YELLOW))
        for ci, (cat_name, cat_current, _cat_fn) in enumerate(categories):
            print(f"  {ci + 1}. {cat_name} [{cat_current}]")
        print()
        try:
            val = input(color("  Configure # (or Enter to skip): ", Colors.DIM)).strip()
            if val:
                ci = int(val) - 1
                if 0 <= ci < len(categories):
                    categories[ci][2]()
        except (ValueError, KeyboardInterrupt, EOFError):
            pass
    print()


def _save_plugin_selection_fallback(plugin_keys, chosen, disabled, *, expected_config=None) -> None:
    """The text fallback's save: same admission authority, refusal printed."""
    from hermes_cli.plugins_admission import AdmissionRefused

    try:
        _persist_plugin_selection(plugin_keys, chosen, disabled, expected_config=expected_config)
    except AdmissionRefused as exc:
        print(f"  Plugin selection refused, not saved: {exc}")
        print("  config.yaml and the active environment are unchanged.")
