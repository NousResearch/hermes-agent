"""Interactive curses-based radio menu.

Scrollable station/source picker launched by the /radio slash command.
Follows the same curses pattern as the session browser in cli.py.

Decade/mood selectors use toggleable checkboxes (Space to toggle).
"""

import curses
from typing import Any, Dict, List, Optional


# -- Menu items ------------------------------------------------------------

class MenuItem:
    """A single item in the radio menu."""
    __slots__ = ("label", "sublabel", "action", "data", "is_header", "is_toggle", "toggled", "toggle_key")

    def __init__(
        self,
        label: str,
        sublabel: str = "",
        action: str = "",
        data: Optional[Dict[str, Any]] = None,
        is_header: bool = False,
        is_toggle: bool = False,
        toggled: bool = False,
        toggle_key: str = "",
    ):
        self.label = label
        self.sublabel = sublabel
        self.action = action
        self.data = data or {}
        self.is_header = is_header
        self.is_toggle = is_toggle
        self.toggled = toggled
        self.toggle_key = toggle_key  # e.g. "decade:1970", "mood:weird", "mic_breaks"


def _build_menu_items(
    soma_channels: Optional[list] = None,
    now_playing: Optional[dict] = None,
    presets: Optional[Dict[str, dict]] = None,
    active_decades: Optional[set] = None,
    active_moods: Optional[set] = None,
    mic_breaks: bool = True,
) -> List[MenuItem]:
    """Build the full menu item list."""
    if active_decades is None:
        active_decades = {1950, 1960, 1970, 1980, 1990}
    if active_moods is None:
        active_moods = {"slow", "fast", "weird"}

    items: List[MenuItem] = []

    # Now playing / controls (if active)
    if now_playing and now_playing.get("active"):
        items.append(MenuItem(label="NOW PLAYING", is_header=True))
        title = now_playing.get("title", "")
        artist = now_playing.get("artist", "")
        display = f"{artist} \u2014 {title}" if artist else title
        items.append(MenuItem(
            label=f"  \u25b6 {display}" if not now_playing.get("paused") else f"  \u2759\u2759 {display}",
            sublabel=now_playing.get("station_name", ""),
            action="toggle_pause",
        ))
        items.append(MenuItem(label="  Skip", action="skip"))
        items.append(MenuItem(label="  Stop", action="stop"))
        items.append(MenuItem(label="", is_header=True))

    # Decade selectors
    items.append(MenuItem(label="DECADES  (Space to toggle)", is_header=True))
    for decade in [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]:
        items.append(MenuItem(
            label=f"  {decade}s",
            is_toggle=True,
            toggled=decade in active_decades,
            toggle_key=f"decade:{decade}",
        ))

    # Mood selectors
    items.append(MenuItem(label="", is_header=True))
    items.append(MenuItem(label="MOODS", is_header=True))
    for mood, desc in [("weird", "the good stuff"), ("slow", "deep, contemplative"), ("fast", "upbeat, energetic")]:
        items.append(MenuItem(
            label=f"  {mood}",
            sublabel=desc,
            is_toggle=True,
            toggled=mood in active_moods,
            toggle_key=f"mood:{mood}",
        ))

    # Mic breaks toggle
    items.append(MenuItem(label="", is_header=True))
    items.append(MenuItem(label="OPTIONS", is_header=True))
    items.append(MenuItem(
        label="  Mic breaks",
        sublabel="AI DJ commentary between tracks",
        is_toggle=True,
        toggled=mic_breaks,
        toggle_key="mic_breaks",
    ))

    # Crate dig actions
    items.append(MenuItem(label="", is_header=True))
    items.append(MenuItem(label="CRATE DIGGER", is_header=True))
    items.append(MenuItem(
        label="  Dig (selected decades + moods)",
        sublabel="Radiooooo global archive",
        action="crate",
    ))
    items.append(MenuItem(
        label="  Dig Japan",
        sublabel="JPN, selected moods",
        action="crate",
        data={"country": "JPN"},
    ))
    items.append(MenuItem(
        label="  Dig France",
        sublabel="FRA, selected moods",
        action="crate",
        data={"country": "FRA"},
    ))
    items.append(MenuItem(
        label="  Dig UK",
        sublabel="GBR, selected moods",
        action="crate",
        data={"country": "GBR"},
    ))
    items.append(MenuItem(
        label="  Dig USA",
        sublabel="USA, selected moods",
        action="crate",
        data={"country": "USA"},
    ))

    # SomaFM
    items.append(MenuItem(label="", is_header=True))
    items.append(MenuItem(label="SOMAFM", is_header=True))
    if soma_channels:
        for ch in soma_channels:
            items.append(MenuItem(
                label=f"  {ch.get('title', ch.get('id', '?'))}",
                sublabel=ch.get("genre", ""),
                action="somafm",
                data={"channel_id": ch.get("id", "")},
            ))
    else:
        items.append(MenuItem(
            label="  Loading channels...",
            action="somafm_refresh",
        ))

    # Presets
    if presets:
        items.append(MenuItem(label="", is_header=True))
        items.append(MenuItem(label="PRESETS", is_header=True))
        for name, preset in presets.items():
            items.append(MenuItem(
                label=f"  {name}",
                sublabel=preset.get("source", ""),
                action="preset",
                data={"name": name, **preset},
            ))

    # Search
    items.append(MenuItem(label="", is_header=True))
    items.append(MenuItem(label="SEARCH", is_header=True))
    items.append(MenuItem(
        label="  Search Radio Browser",
        sublabel="45,000+ stations worldwide",
        action="search_rb",
    ))
    items.append(MenuItem(
        label="  Search Radio Garden",
        sublabel="Explore by city",
        action="search_rg",
    ))

    return items


# -- Curses menu -----------------------------------------------------------

def radio_menu(
    now_playing: Optional[dict] = None,
    soma_channels: Optional[list] = None,
    presets: Optional[Dict[str, dict]] = None,
) -> Optional[MenuItem]:
    """Launch the interactive radio menu.  Returns the selected MenuItem or None.

    Toggle items (decades, moods, mic breaks) are toggled with Space.
    The selected decades/moods are injected into crate dig items' data
    when Enter is pressed on a crate action.
    """
    # Mutable toggle state
    active_decades = {1950, 1960, 1970, 1980, 1990}
    active_moods = {"slow", "fast", "weird"}
    mic_breaks = True

    result_holder = [None]

    def _rebuild():
        return _build_menu_items(
            soma_channels=soma_channels,
            now_playing=now_playing,
            presets=presets,
            active_decades=active_decades,
            active_moods=active_moods,
            mic_breaks=mic_breaks,
        )

    def _run(stdscr):
        nonlocal active_decades, active_moods, mic_breaks

        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)    # selected
            curses.init_pair(2, curses.COLOR_YELLOW, -1)   # header
            curses.init_pair(3, curses.COLOR_CYAN, -1)     # toggle on
            curses.init_pair(4, 8, -1)                     # dim
            curses.init_pair(5, curses.COLOR_BLUE, -1)     # checkbox

        items = _rebuild()
        selectable = [i for i, item in enumerate(items) if not item.is_header]
        if not selectable:
            return

        cursor = 0
        scroll_offset = 0

        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 5 or max_x < 30:
                try:
                    stdscr.addstr(0, 0, "Terminal too small")
                except curses.error:
                    pass
                stdscr.refresh()
                stdscr.getch()
                return

            # Header
            header = "  HERMES RADIO  \u2500  \u2191\u2193 navigate  Space toggle  Enter select  q quit"
            header_attr = curses.A_BOLD
            if curses.has_colors():
                header_attr |= curses.color_pair(2)
            try:
                stdscr.addnstr(0, 0, header, max_x - 1, header_attr)
            except curses.error:
                pass

            try:
                stdscr.addnstr(1, 0, "\u2500" * (max_x - 1), max_x - 1, curses.A_DIM)
            except curses.error:
                pass

            visible_rows = max_y - 3
            if visible_rows < 1:
                visible_rows = 1

            cursor_abs = selectable[cursor] if cursor < len(selectable) else 0

            if cursor_abs < scroll_offset:
                scroll_offset = max(0, cursor_abs - 2)
            elif cursor_abs >= scroll_offset + visible_rows:
                scroll_offset = cursor_abs - visible_rows + 3

            y = 2
            for idx in range(scroll_offset, len(items)):
                if y >= max_y - 1:
                    break
                item = items[idx]

                if item.is_header:
                    if item.label:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(2)
                        try:
                            stdscr.addnstr(y, 0, f"  {item.label}", max_x - 1, attr)
                        except curses.error:
                            pass
                    y += 1
                    continue

                is_selected = (idx == cursor_abs)

                # Build display line
                arrow = " \u25b8 " if is_selected else "   "

                if item.is_toggle:
                    # Checkbox prefix
                    check = "\u25a0" if item.toggled else "\u25a1"  # filled/empty square
                    line = f"{arrow}{check} {item.label.strip()}"
                else:
                    line = f"{arrow}{item.label}"

                # Truncate
                max_label = max_x - 4 - len(item.sublabel) - 4 if item.sublabel else max_x - 4
                if len(line) > max_label > 10:
                    line = line[:max_label - 3] + "..."

                # Color
                if is_selected:
                    attr = curses.A_BOLD
                    if curses.has_colors():
                        attr |= curses.color_pair(1)
                elif item.is_toggle and item.toggled:
                    attr = curses.A_NORMAL
                    if curses.has_colors():
                        attr |= curses.color_pair(3)
                elif item.is_toggle and not item.toggled:
                    attr = curses.A_DIM
                else:
                    attr = curses.A_NORMAL

                try:
                    stdscr.addnstr(y, 0, line, max_x - 1, attr)
                except curses.error:
                    pass

                # Sublabel
                if item.sublabel:
                    sub_attr = curses.color_pair(4) if curses.has_colors() else curses.A_DIM
                    sub_x = max(len(line) + 2, max_x - len(item.sublabel) - 2)
                    if sub_x < max_x - 1:
                        try:
                            stdscr.addnstr(y, sub_x, item.sublabel, max_x - sub_x - 1, sub_attr)
                        except curses.error:
                            pass

                y += 1

            # Footer
            footer_y = max_y - 1
            # Show active decades summary
            decades_str = ",".join(f"{d}s" for d in sorted(active_decades))
            moods_str = ",".join(sorted(active_moods))
            mic_str = "on" if mic_breaks else "off"
            footer = f"  {cursor + 1}/{len(selectable)}  |  decades: {decades_str}  moods: {moods_str}  mic: {mic_str}"
            try:
                stdscr.addnstr(footer_y, 0, footer, max_x - 1, curses.A_DIM)
            except curses.error:
                pass

            stdscr.refresh()

            # Input
            key = stdscr.getch()
            if key == ord("q") or key == 27:  # q or Esc
                return
            elif key == curses.KEY_UP or key == ord("k"):
                if cursor > 0:
                    cursor -= 1
            elif key == curses.KEY_DOWN or key == ord("j"):
                if cursor < len(selectable) - 1:
                    cursor += 1
            elif key == curses.KEY_PPAGE:
                cursor = max(0, cursor - visible_rows)
            elif key == curses.KEY_NPAGE:
                cursor = min(len(selectable) - 1, cursor + visible_rows)
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = len(selectable) - 1
            elif key == ord(" "):  # Space = toggle
                if cursor < len(selectable):
                    item = items[selectable[cursor]]
                    if item.is_toggle:
                        tk = item.toggle_key
                        if tk.startswith("decade:"):
                            decade = int(tk.split(":")[1])
                            if decade in active_decades:
                                active_decades.discard(decade)
                            else:
                                active_decades.add(decade)
                        elif tk.startswith("mood:"):
                            mood = tk.split(":")[1]
                            if mood in active_moods:
                                if len(active_moods) > 1:  # keep at least one
                                    active_moods.discard(mood)
                            else:
                                active_moods.add(mood)
                        elif tk == "mic_breaks":
                            mic_breaks = not mic_breaks
                        # Rebuild items with new toggle state
                        items = _rebuild()
                        selectable = [i for i, it in enumerate(items) if not it.is_header]
            elif key in (curses.KEY_ENTER, 10, 13):  # Enter
                if cursor < len(selectable):
                    item = items[selectable[cursor]]
                    if item.is_toggle:
                        # Treat Enter same as Space for toggles
                        tk = item.toggle_key
                        if tk.startswith("decade:"):
                            decade = int(tk.split(":")[1])
                            if decade in active_decades:
                                active_decades.discard(decade)
                            else:
                                active_decades.add(decade)
                        elif tk.startswith("mood:"):
                            mood = tk.split(":")[1]
                            if mood in active_moods:
                                if len(active_moods) > 1:
                                    active_moods.discard(mood)
                            else:
                                active_moods.add(mood)
                        elif tk == "mic_breaks":
                            mic_breaks = not mic_breaks
                        items = _rebuild()
                        selectable = [i for i, it in enumerate(items) if not it.is_header]
                    else:
                        # Action item -- inject current toggle state
                        if item.action == "crate":
                            item.data["decades"] = sorted(active_decades) if active_decades else None
                            item.data["moods"] = sorted(active_moods) if active_moods else None
                            item.data["mic_breaks"] = mic_breaks
                        else:
                            item.data["mic_breaks"] = mic_breaks
                        result_holder[0] = item
                        return

    curses.wrapper(_run)
    return result_holder[0]


# -- Search submenu --------------------------------------------------------

def search_menu(results: List[Dict[str, Any]], title: str = "Search Results") -> Optional[Dict[str, Any]]:
    """Show search results in a scrollable picker.  Returns selected result or None."""
    if not results:
        return None

    result_holder = [None]

    def _run(stdscr):
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, 8, -1)

        cursor = 0
        scroll_offset = 0

        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()
            visible = max_y - 3

            header = f"  {title}  \u2500  \u2191\u2193 navigate  Enter select  q back"
            try:
                attr = curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0)
                stdscr.addnstr(0, 0, header, max_x - 1, attr)
                stdscr.addnstr(1, 0, "\u2500" * (max_x - 1), max_x - 1, curses.A_DIM)
            except curses.error:
                pass

            if cursor < scroll_offset:
                scroll_offset = cursor
            elif cursor >= scroll_offset + visible:
                scroll_offset = cursor - visible + 1

            for draw_i, idx in enumerate(range(scroll_offset, min(len(results), scroll_offset + visible))):
                y = draw_i + 2
                if y >= max_y - 1:
                    break
                r = results[idx]
                is_sel = idx == cursor
                arrow = " \u25b8 " if is_sel else "   "
                name = r.get("name") or r.get("title") or r.get("id", "?")
                extra = r.get("country") or r.get("genre") or r.get("tags", "")
                if isinstance(extra, str) and len(extra) > 30:
                    extra = extra[:27] + "..."
                line = f"{arrow}{name}"

                attr = curses.A_BOLD | (curses.color_pair(1) if curses.has_colors() else 0) if is_sel else curses.A_NORMAL
                try:
                    stdscr.addnstr(y, 0, line, max_x - 1, attr)
                except curses.error:
                    pass
                if extra:
                    ex = max(len(line) + 2, max_x - len(extra) - 2)
                    try:
                        stdscr.addnstr(y, ex, extra, max_x - ex - 1, curses.color_pair(4) if curses.has_colors() else curses.A_DIM)
                    except curses.error:
                        pass

            try:
                stdscr.addnstr(max_y - 1, 0, f"  {cursor + 1}/{len(results)}", max_x - 1, curses.A_DIM)
            except curses.error:
                pass

            stdscr.refresh()
            key = stdscr.getch()
            if key == ord("q") or key == 27:
                return
            elif key == curses.KEY_UP or key == ord("k"):
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_DOWN or key == ord("j"):
                cursor = min(len(results) - 1, cursor + 1)
            elif key == curses.KEY_PPAGE:
                cursor = max(0, cursor - visible)
            elif key == curses.KEY_NPAGE:
                cursor = min(len(results) - 1, cursor + visible)
            elif key in (curses.KEY_ENTER, 10, 13):
                result_holder[0] = results[cursor]
                return

    curses.wrapper(_run)
    return result_holder[0]
