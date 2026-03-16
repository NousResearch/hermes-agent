"""Radio menu for the Hermes CLI.

Two modes:
1. TUI mode -- ConditionalContainer inside prompt_toolkit with arrow key
   navigation, space to toggle, enter to select, q/esc to close.
   State is managed via cli._radio_menu_state dict.
2. Fallback mode -- numbered print+input for environments where the TUI
   widget can't render.

The CLI integration (widget, key bindings) lives in cli.py. This module
provides the data model and rendering logic.
"""

import threading
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from radio.visualizers import list_presets
except Exception:
    def list_presets() -> List[str]:
        return []


# -- Box-drawing characters ---------------------------------------------------

BOX_TL = "\u256d"  # ╭
BOX_TR = "\u256e"  # ╮
BOX_BL = "\u2570"  # ╰
BOX_BR = "\u256f"  # ╯
BOX_H  = "\u2500"  # ─
BOX_V  = "\u2502"  # │

TOGGLE_ON  = "\u25cf"  # ●
TOGGLE_OFF = "\u25cb"  # ○


# -- Data model ----------------------------------------------------------------

class MenuItem:
    __slots__ = ("label", "sublabel", "action", "data",
                 "is_header", "is_toggle", "toggled", "toggle_key")

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
        self.toggle_key = toggle_key


# -- Menu state (shared between menu.py and cli.py) ------------------------

class RadioMenuState:
    """Mutable state for the radio menu widget."""

    def __init__(self, items: List[MenuItem]):
        self.items = items
        self.selectable = [i for i, it in enumerate(items) if not it.is_header]
        self.cursor = 0  # index into self.selectable
        self.result: Optional[MenuItem] = None
        self.done = threading.Event()
        self.viewport_start = 0

        # Load persisted toggle state
        try:
            from radio.config import get_decades, get_moods, get_mic_breaks
            self.active_decades: Set[int] = get_decades()
            self.active_moods: Set[str] = get_moods()
            self.mic_breaks: bool = get_mic_breaks()
        except Exception:
            self.active_decades = {1950, 1960, 1970, 1980, 1990}
            self.active_moods = {"slow", "fast", "weird"}
            self.mic_breaks = True

        # Sync toggle visual state with loaded config
        for item in items:
            if item.is_toggle:
                tk = item.toggle_key
                if tk.startswith("decade:"):
                    item.toggled = int(tk.split(":")[1]) in self.active_decades
                elif tk.startswith("mood:"):
                    item.toggled = tk.split(":")[1] in self.active_moods
                elif tk == "mic_breaks":
                    item.toggled = self.mic_breaks

        # Section boundaries (for left/right tab navigation)
        self._sections: List[int] = []
        for i, idx in enumerate(self.selectable):
            if idx > 0 and self.items[idx - 1].is_header:
                self._sections.append(i)

    @property
    def cursor_abs(self) -> int:
        """Absolute index into items list."""
        if not self.selectable:
            return 0
        return self.selectable[min(self.cursor, len(self.selectable) - 1)]

    def move_up(self):
        self.cursor = max(0, self.cursor - 1)

    def move_down(self):
        self.cursor = min(len(self.selectable) - 1, self.cursor + 1)

    def page_up(self, page_size: int = 10):
        self.cursor = max(0, self.cursor - page_size)

    def page_down(self, page_size: int = 10):
        self.cursor = min(len(self.selectable) - 1, self.cursor + page_size)

    def jump_to_section(self, direction: int = 1):
        """Jump to the next (1) or previous (-1) section header."""
        start = self.cursor
        pos = start + direction
        while 0 <= pos < len(self.selectable):
            abs_idx = self.selectable[pos]
            # Check if the item above is a header (section boundary)
            if abs_idx > 0 and self.items[abs_idx - 1].is_header:
                self.cursor = pos
                return
            pos += direction

    def toggle_current(self):
        """Toggle the current item if it's a toggle. Returns True if toggled."""
        item = self.items[self.cursor_abs]
        if not item.is_toggle:
            return False

        tk = item.toggle_key
        if tk.startswith("decade:"):
            decade = int(tk.split(":")[1])
            if decade in self.active_decades:
                self.active_decades.discard(decade)
            else:
                self.active_decades.add(decade)
            item.toggled = decade in self.active_decades
        elif tk.startswith("mood:"):
            mood = tk.split(":")[1]
            if mood in self.active_moods:
                if len(self.active_moods) > 1:
                    self.active_moods.discard(mood)
            else:
                self.active_moods.add(mood)
            item.toggled = mood in self.active_moods
        elif tk == "mic_breaks":
            self.mic_breaks = not self.mic_breaks
            item.toggled = self.mic_breaks

        # Persist to disk
        try:
            from radio import config as rc
            rc.set_decades(self.active_decades)
            rc.set_moods(self.active_moods)
            rc.set_mic_breaks(self.mic_breaks)
        except Exception:
            pass
        return True

    def select_current(self):
        """Select the current item. For toggles, toggle. For actions, set result and signal done."""
        item = self.items[self.cursor_abs]
        if item.is_toggle:
            self.toggle_current()
            return

        # Inject toggle state into the action data
        if item.action == "crate":
            item.data["decades"] = sorted(self.active_decades) if self.active_decades else None
            item.data["moods"] = sorted(self.active_moods) if self.active_moods else None
        item.data["mic_breaks"] = self.mic_breaks
        self.result = item
        self.done.set()

    def cancel(self):
        self.result = None
        self.done.set()


# -- Build items -----------------------------------------------------------

def build_menu_items(
    soma_channels=None, now_playing=None, presets=None,
    active_decades=None, active_moods=None, mic_breaks=True,
    active_visualizer=None,
) -> List[MenuItem]:
    if active_decades is None:
        active_decades = {1950, 1960, 1970, 1980, 1990}
    if active_moods is None:
        active_moods = {"slow", "fast", "weird"}
    if active_visualizer is None:
        try:
            from radio.config import get_visualizer
            active_visualizer = get_visualizer()
        except Exception:
            active_visualizer = "braille"

    items: List[MenuItem] = []

    # Now playing
    if now_playing and now_playing.get("active"):
        items.append(MenuItem(label="NOW PLAYING", is_header=True))
        title = now_playing.get("title", "")
        artist = now_playing.get("artist", "")
        display = f"{artist} — {title}" if artist else title
        prefix = "▶" if not now_playing.get("paused") else "⏸"
        items.append(MenuItem(label=f"{prefix} {display}", sublabel=now_playing.get("station_name", ""), action="toggle_pause"))
        items.append(MenuItem(label="Skip track", action="skip"))
        items.append(MenuItem(label="Stop radio", action="stop"))

    # Visualizer presets near the top so they are easy to discover
    visualizer_names = list_presets()
    if visualizer_names:
        items.append(MenuItem(label="VISUALIZER", is_header=True))
        for name in visualizer_names:
            sublabel = "active" if name == active_visualizer else ""
            items.append(MenuItem(
                label=name,
                sublabel=sublabel,
                action="visualizer",
                data={"name": name},
            ))

    # Options near the top too
    items.append(MenuItem(label="OPTIONS", is_header=True))
    items.append(MenuItem(label="Mic breaks", sublabel="AI DJ commentary", is_toggle=True, toggled=mic_breaks, toggle_key="mic_breaks"))

    # Crate digger (with decades + moods nested inside)
    items.append(MenuItem(label="CRATE DIGGER", is_header=True))
    items.append(MenuItem(label="Dig (selected decades + moods)", sublabel="Radiooooo", action="crate"))
    items.append(MenuItem(label="Dig Japan", action="crate", data={"country": "JPN"}))
    items.append(MenuItem(label="Dig France", action="crate", data={"country": "FRA"}))
    items.append(MenuItem(label="Dig UK", action="crate", data={"country": "GBR"}))
    items.append(MenuItem(label="Dig USA", action="crate", data={"country": "USA"}))

    # Decades (sub-section of crate digger)
    items.append(MenuItem(label="DECADES", is_header=True))
    for decade in [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]:
        items.append(MenuItem(label=f"{decade}s", is_toggle=True, toggled=decade in active_decades, toggle_key=f"decade:{decade}"))

    # Moods (sub-section of crate digger)
    items.append(MenuItem(label="MOODS", is_header=True))
    for mood, desc in [("weird", "the good stuff"), ("slow", "deep, contemplative"), ("fast", "upbeat, energetic")]:
        items.append(MenuItem(label=mood, sublabel=desc, is_toggle=True, toggled=mood in active_moods, toggle_key=f"mood:{mood}"))
    # Curated world stations (from stations.yaml)
    try:
        from radio.stations import load_stations
        curated = load_stations()
        if curated:
            # Group by region
            regions_seen = []
            for station in curated:
                region = station.get("region", "").upper()
                if region and region not in regions_seen:
                    regions_seen.append(region)
                    items.append(MenuItem(label=region, is_header=True))
                items.append(MenuItem(
                    label=station["name"],
                    sublabel=station.get("genre", ""),
                    action="stream",
                    data={"url": station["url"], "name": station["name"]},
                ))
    except Exception:
        pass

    # SomaFM (if no curated stations loaded, or as additional)
    if soma_channels and not curated:
        items.append(MenuItem(label="SOMAFM", is_header=True))
        for ch in soma_channels:
            items.append(MenuItem(label=ch.get("title", ch.get("id", "?")), sublabel=ch.get("genre", ""), action="somafm", data={"channel_id": ch.get("id", "")}))

    # Presets
    if presets:
        items.append(MenuItem(label="PRESETS", is_header=True))
        for name, preset in presets.items():
            items.append(MenuItem(label=name, sublabel=preset.get("source", ""), action="preset", data={"name": name, **preset}))

    # Recently listened
    try:
        from radio.config import get_recent_stations
        recent = get_recent_stations()
        if recent:
            items.append(MenuItem(label="RECENT", is_header=True))
            for station in recent:
                items.append(MenuItem(
                    label=station.get("name", "?"),
                    sublabel=station.get("source", "stream"),
                    action="stream",
                    data={"url": station.get("url", ""), "name": station.get("name", "")},
                ))
    except Exception:
        pass

    # Search
    items.append(MenuItem(label="SEARCH", is_header=True))
    items.append(MenuItem(label="Search Radio Browser", sublabel="45k+ stations", action="search_rb"))
    items.append(MenuItem(label="Search Radio Garden", sublabel="by city", action="search_rg"))

    return items


# -- Render helpers ------------------------------------------------------------

MENU_WIDTH = 58
INNER_WIDTH = MENU_WIDTH - 4  # inside borders: "│  " ... " │"


def _section_header(label: str) -> str:
    """Build an embedded section separator: ── LABEL ──────"""
    pad = INNER_WIDTH - len(label) - 4  # 4 = "── " + " "
    if pad < 2:
        pad = 2
    return f"{BOX_H}{BOX_H} {label} {BOX_H * pad}"


def _pad_line(text: str, width: int = INNER_WIDTH) -> str:
    """Pad text to fixed width, truncating if needed."""
    if len(text) > width:
        return text[:width - 1] + "\u2026"  # …
    return text + " " * (width - len(text))


# -- Render for FormattedTextControl ---------------------------------------

def _get_visible_rows() -> int:
    import shutil
    term_h = shutil.get_terminal_size((80, 24)).lines
    return max(6, term_h - 12)

VISIBLE_ROWS = 20  # fallback; render_menu uses _get_visible_rows()


def render_menu(state: RadioMenuState) -> List[Tuple[str, str]]:
    """Return styled text fragments for the radio menu widget."""
    items = state.items
    cursor_abs = state.cursor_abs
    fragments: List[Tuple[str, str]] = []

    # ╭─ top border ─╮
    fragments.append(("class:radio-menu-border", f"  {BOX_TL}{BOX_H * (MENU_WIDTH - 2)}{BOX_TR}\n"))

    # │  HERMES RADI(Ctrl+O)  ...keybinds...  │
    title_text = "HERMES RADI"  # before the O
    title_len = len("HERMES RADIO")  # for spacing calc
    keys_hint = "\u2191\u2193 nav  Spc toggle  \u21b5 select  q close"
    gap = INNER_WIDTH - title_len - len(keys_hint)
    if gap < 2:
        gap = 2
        keys_hint = keys_hint[:INNER_WIDTH - title_len - gap]
    fragments.append(("class:radio-menu-border", f"  {BOX_V} "))
    fragments.append(("class:radio-menu-title", title_text))
    fragments.append(("class:radio-menu-accent", "O"))
    fragments.append(("", " " * gap))
    fragments.append(("class:radio-menu-dim", keys_hint))
    fragments.append(("class:radio-menu-border", f" {BOX_V}\n"))

    # ├─ separator ─┤
    fragments.append(("class:radio-menu-border", f"  \u251c{BOX_H * (MENU_WIDTH - 2)}\u2524\n"))

    # Viewport calculation -- keep cursor centered in view
    visible = _get_visible_rows()
    vp = state.viewport_start

    margin = 2
    if cursor_abs < vp + margin:
        vp = max(0, cursor_abs - margin)
    elif cursor_abs >= vp + visible - margin:
        vp = cursor_abs - visible + margin + 1

    vp = max(0, min(len(items) - visible, vp))
    state.viewport_start = vp

    # Scroll indicator top
    if vp > 0:
        scroll_up = _pad_line("  \u25b2 more above")
        fragments.append(("class:radio-menu-border", f"  {BOX_V} "))
        fragments.append(("class:radio-menu-dim", scroll_up))
        fragments.append(("class:radio-menu-border", f" {BOX_V}\n"))

    rendered = 0
    for idx in range(vp, len(items)):
        if rendered >= visible:
            break

        item = items[idx]

        if item.is_header:
            if item.label:
                # Section separator with embedded label
                sep = _section_header(item.label)
                # Empty line before section (visual breathing room)
                empty = " " * INNER_WIDTH
                fragments.append(("class:radio-menu-border", f"  {BOX_V} "))
                fragments.append(("", empty))
                fragments.append(("class:radio-menu-border", f" {BOX_V}\n"))
                # The header line
                fragments.append(("class:radio-menu-border", f"  {BOX_V} "))
                fragments.append(("class:radio-menu-header", _pad_line(sep)))
                fragments.append(("class:radio-menu-border", f" {BOX_V}\n"))
                rendered += 2
            else:
                empty = " " * INNER_WIDTH
                fragments.append(("class:radio-menu-border", f"  {BOX_V} "))
                fragments.append(("", empty))
                fragments.append(("class:radio-menu-border", f" {BOX_V}\n"))
                rendered += 1
            continue

        is_selected = (idx == cursor_abs)
        pointer = " \u25b8 " if is_selected else "   "

        if item.is_toggle:
            check = TOGGLE_ON if item.toggled else TOGGLE_OFF
            label_text = f"{pointer}{check} {item.label}"
        else:
            label_text = f"{pointer}  {item.label}"

        # Style based on state
        if is_selected:
            style = "class:radio-menu-selected"
        elif item.is_toggle and item.toggled:
            style = "class:radio-menu-on"
        elif item.is_toggle:
            style = "class:radio-menu-off"
        else:
            style = "class:radio-menu-item"

        # Render label and sublabel as separate styled fragments
        fragments.append(("class:radio-menu-border", f"  {BOX_V} "))
        if item.sublabel:
            sub_text = f"  {item.sublabel}"
            total_len = len(label_text) + len(sub_text)
            pad = max(0, INNER_WIDTH - total_len)
            fragments.append((style, label_text))
            fragments.append(("class:radio-menu-sub", sub_text))
            fragments.append(("", " " * pad))
        else:
            fragments.append((style, _pad_line(label_text)))
        fragments.append(("class:radio-menu-border", f" {BOX_V}\n"))
        rendered += 1

    # Scroll indicator bottom
    if vp + visible < len(items):
        scroll_dn = _pad_line("  \u25bc more below")
        fragments.append(("class:radio-menu-border", f"  {BOX_V} "))
        fragments.append(("class:radio-menu-dim", scroll_dn))
        fragments.append(("class:radio-menu-border", f" {BOX_V}\n"))

    # ├─ footer separator ─┤
    fragments.append(("class:radio-menu-border", f"  \u251c{BOX_H * (MENU_WIDTH - 2)}\u2524\n"))

    # Footer: current crate dig config
    decades_str = ", ".join(f"{d}s" for d in sorted(state.active_decades)) or "none"
    moods_str = ", ".join(sorted(state.active_moods)) or "none"
    mic_str = "on" if state.mic_breaks else "off"
    try:
        from radio.config import get_visualizer
        viz_str = get_visualizer()
    except Exception:
        viz_str = "braille"
    footer = f"decades: {decades_str}  moods: {moods_str}  mic: {mic_str}  viz: {viz_str}"
    fragments.append(("class:radio-menu-border", f"  {BOX_V} "))
    fragments.append(("class:radio-menu-dim", _pad_line(footer)))
    fragments.append(("class:radio-menu-border", f" {BOX_V}\n"))

    # ╰─ bottom border ─╯
    fragments.append(("class:radio-menu-border", f"  {BOX_BL}{BOX_H * (MENU_WIDTH - 2)}{BOX_BR}\n"))

    return fragments


# -- Fallback: print+input menu -------------------------------------------

def radio_menu_fallback(
    now_playing=None, soma_channels=None, presets=None,
) -> Optional[MenuItem]:
    """Numbered print+input fallback for non-TUI environments."""
    active_decades = {1950, 1960, 1970, 1980, 1990}
    active_moods = {"slow", "fast", "weird"}
    mic_breaks = True

    while True:
        items = build_menu_items(
            soma_channels=soma_channels, now_playing=now_playing,
            presets=presets, active_decades=active_decades,
            active_moods=active_moods, mic_breaks=mic_breaks,
        )

        w = MENU_WIDTH - 4
        print(f"\n  {BOX_TL}{BOX_H * (MENU_WIDTH - 2)}{BOX_TR}")
        print(f"  {BOX_V} {'HERMES RADIO':{w}} {BOX_V}")
        print(f"  \u251c{BOX_H * (MENU_WIDTH - 2)}\u2524")

        num = 0
        idx_map = {}
        for i, item in enumerate(items):
            if item.is_header:
                if item.label:
                    sep = _section_header(item.label)
                    print(f"  {BOX_V}  {' ' * (w - 1)}{BOX_V}")
                    print(f"  {BOX_V} {sep:{w}} {BOX_V}")
                continue
            num += 1
            idx_map[num] = i
            if item.is_toggle:
                check = TOGGLE_ON if item.toggled else TOGGLE_OFF
                label = f"{check} {item.label}"
            else:
                label = f"  {item.label}"
            sub = f"  ({item.sublabel})" if item.sublabel else ""
            line = f"{num:2d}. {label}{sub}"
            print(f"  {BOX_V} {line:{w}} {BOX_V}")

        print(f"  \u251c{BOX_H * (MENU_WIDTH - 2)}\u2524")
        decades_str = ", ".join(f"{d}s" for d in sorted(active_decades))
        moods_str = ", ".join(sorted(active_moods))
        mic_str = "on" if mic_breaks else "off"
        footer = f"decades: {decades_str}  moods: {moods_str}  mic: {mic_str}"
        print(f"  {BOX_V} {footer:{w}} {BOX_V}")
        print(f"  {BOX_BL}{BOX_H * (MENU_WIDTH - 2)}{BOX_BR}")

        try:
            raw = input("  Enter number (q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", ""):
            return None
        try:
            choice = int(raw)
        except ValueError:
            continue
        if choice not in idx_map:
            continue

        item = items[idx_map[choice]]
        if item.is_toggle:
            tk = item.toggle_key
            if tk.startswith("decade:"):
                d = int(tk.split(":")[1])
                active_decades.symmetric_difference_update({d})
            elif tk.startswith("mood:"):
                m = tk.split(":")[1]
                if m in active_moods and len(active_moods) > 1:
                    active_moods.discard(m)
                else:
                    active_moods.add(m)
            elif tk == "mic_breaks":
                mic_breaks = not mic_breaks
            continue

        if item.action == "crate":
            item.data["decades"] = sorted(active_decades) if active_decades else None
            item.data["moods"] = sorted(active_moods) if active_moods else None
        item.data["mic_breaks"] = mic_breaks
        return item


def search_menu(results: List[Dict[str, Any]], title: str = "Search Results") -> Optional[Dict[str, Any]]:
    """Print search results and return the selected one."""
    if not results:
        print("  No results found.")
        return None

    w = MENU_WIDTH - 4
    print(f"\n  {BOX_TL}{BOX_H * (MENU_WIDTH - 2)}{BOX_TR}")
    print(f"  {BOX_V} {title:{w}} {BOX_V}")
    print(f"  \u251c{BOX_H * (MENU_WIDTH - 2)}\u2524")

    for i, r in enumerate(results, 1):
        name = r.get("name") or r.get("title") or "?"
        extra = r.get("country") or r.get("genre") or r.get("tags", "")
        if isinstance(extra, str) and len(extra) > 30:
            extra = extra[:27] + "..."
        sub = f"  ({extra})" if extra else ""
        line = f"{i:2d}. {name}{sub}"
        print(f"  {BOX_V} {line:{w}} {BOX_V}")

    print(f"  {BOX_BL}{BOX_H * (MENU_WIDTH - 2)}{BOX_BR}")

    try:
        raw = input("  Enter number (q to back): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if raw.lower() in ("q", "quit", ""):
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(results):
            return results[idx]
    except ValueError:
        pass
    return None
