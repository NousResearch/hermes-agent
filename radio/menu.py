"""Interactive radio menu for the Hermes CLI.

Uses simple numbered print+input interface that works reliably inside
prompt_toolkit's patched stdout context (curses conflicts with it).
"""

from typing import Any, Dict, List, Optional


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
        self.toggle_key = toggle_key


def radio_menu(
    now_playing: Optional[dict] = None,
    soma_channels: Optional[list] = None,
    presets: Optional[Dict[str, dict]] = None,
) -> Optional[MenuItem]:
    """Print the radio menu and return the selected MenuItem or None."""

    # Mutable toggle state
    active_decades = {1950, 1960, 1970, 1980, 1990}
    active_moods = {"slow", "fast", "weird"}
    mic_breaks = True

    while True:
        items = _build_items(
            soma_channels=soma_channels,
            now_playing=now_playing,
            presets=presets,
            active_decades=active_decades,
            active_moods=active_moods,
            mic_breaks=mic_breaks,
        )

        # Print menu
        print()
        print("  HERMES RADIO")
        print("  " + "\u2500" * 50)

        # Number only selectable items
        num = 0
        idx_map = {}  # number -> items index
        for i, item in enumerate(items):
            if item.is_header:
                if item.label:
                    print(f"\n  {item.label}")
                continue

            num += 1
            idx_map[num] = i

            if item.is_toggle:
                check = "\u25a0" if item.toggled else "\u25a1"
                label = f"  {check} {item.label.strip()}"
            else:
                label = f"    {item.label.strip()}"

            sub = f"  ({item.sublabel})" if item.sublabel else ""
            print(f"  {num:2d}. {label}{sub}")

        # Footer
        decades_str = ", ".join(f"{d}s" for d in sorted(active_decades))
        moods_str = ", ".join(sorted(active_moods))
        mic_str = "on" if mic_breaks else "off"
        print(f"\n  decades: {decades_str}")
        print(f"  moods: {moods_str}  |  mic breaks: {mic_str}")
        print()

        # Get input
        try:
            raw = input("  Enter number (or q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw.lower() in ("q", "quit", ""):
            return None

        try:
            choice = int(raw)
        except ValueError:
            print(f"  Invalid input: {raw}")
            continue

        if choice not in idx_map:
            print(f"  Invalid choice: {choice}")
            continue

        item = items[idx_map[choice]]

        # Handle toggles (loop back to redisplay)
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
                    if len(active_moods) > 1:
                        active_moods.discard(mood)
                else:
                    active_moods.add(mood)
            elif tk == "mic_breaks":
                mic_breaks = not mic_breaks
            continue  # redisplay menu

        # Action item -- inject toggle state
        if item.action == "crate":
            item.data["decades"] = sorted(active_decades) if active_decades else None
            item.data["moods"] = sorted(active_moods) if active_moods else None
            item.data["mic_breaks"] = mic_breaks
        else:
            item.data["mic_breaks"] = mic_breaks

        return item


def _build_items(
    soma_channels=None, now_playing=None, presets=None,
    active_decades=None, active_moods=None, mic_breaks=True,
) -> List[MenuItem]:
    if active_decades is None:
        active_decades = {1950, 1960, 1970, 1980, 1990}
    if active_moods is None:
        active_moods = {"slow", "fast", "weird"}

    items: List[MenuItem] = []

    # Now playing
    if now_playing and now_playing.get("active"):
        items.append(MenuItem(label="NOW PLAYING", is_header=True))
        title = now_playing.get("title", "")
        artist = now_playing.get("artist", "")
        display = f"{artist} \u2014 {title}" if artist else title
        prefix = "\u25b6" if not now_playing.get("paused") else "\u2759\u2759"
        items.append(MenuItem(label=f"{prefix} {display}", sublabel=now_playing.get("station_name", ""), action="toggle_pause"))
        items.append(MenuItem(label="Skip", action="skip"))
        items.append(MenuItem(label="Stop", action="stop"))

    # Decades
    items.append(MenuItem(label="DECADES (toggle)", is_header=True))
    for decade in [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]:
        items.append(MenuItem(label=f"{decade}s", is_toggle=True, toggled=decade in active_decades, toggle_key=f"decade:{decade}"))

    # Moods
    items.append(MenuItem(label="MOODS (toggle)", is_header=True))
    for mood, desc in [("weird", "the good stuff"), ("slow", "deep, contemplative"), ("fast", "upbeat, energetic")]:
        items.append(MenuItem(label=mood, sublabel=desc, is_toggle=True, toggled=mood in active_moods, toggle_key=f"mood:{mood}"))

    # Options
    items.append(MenuItem(label="OPTIONS", is_header=True))
    items.append(MenuItem(label="Mic breaks", sublabel="AI DJ commentary", is_toggle=True, toggled=mic_breaks, toggle_key="mic_breaks"))

    # Crate dig
    items.append(MenuItem(label="CRATE DIGGER", is_header=True))
    items.append(MenuItem(label="Dig (selected decades + moods)", sublabel="Radiooooo", action="crate"))
    items.append(MenuItem(label="Dig Japan", sublabel="JPN", action="crate", data={"country": "JPN"}))
    items.append(MenuItem(label="Dig France", sublabel="FRA", action="crate", data={"country": "FRA"}))
    items.append(MenuItem(label="Dig UK", sublabel="GBR", action="crate", data={"country": "GBR"}))
    items.append(MenuItem(label="Dig USA", sublabel="USA", action="crate", data={"country": "USA"}))

    # SomaFM
    items.append(MenuItem(label="SOMAFM", is_header=True))
    if soma_channels:
        for ch in soma_channels:
            items.append(MenuItem(label=ch.get("title", ch.get("id", "?")), sublabel=ch.get("genre", ""), action="somafm", data={"channel_id": ch.get("id", "")}))
    else:
        items.append(MenuItem(label="(loading...)", action="somafm_refresh"))

    # Presets
    if presets:
        items.append(MenuItem(label="PRESETS", is_header=True))
        for name, preset in presets.items():
            items.append(MenuItem(label=name, sublabel=preset.get("source", ""), action="preset", data={"name": name, **preset}))

    # Search
    items.append(MenuItem(label="SEARCH", is_header=True))
    items.append(MenuItem(label="Search Radio Browser", sublabel="45k+ stations", action="search_rb"))
    items.append(MenuItem(label="Search Radio Garden", sublabel="by city", action="search_rg"))

    return items


def search_menu(results: List[Dict[str, Any]], title: str = "Search Results") -> Optional[Dict[str, Any]]:
    """Print search results and return the selected one."""
    if not results:
        print("  No results found.")
        return None

    print(f"\n  {title}")
    print("  " + "\u2500" * 50)
    for i, r in enumerate(results, 1):
        name = r.get("name") or r.get("title") or "?"
        extra = r.get("country") or r.get("genre") or r.get("tags", "")
        if isinstance(extra, str) and len(extra) > 30:
            extra = extra[:27] + "..."
        sub = f"  ({extra})" if extra else ""
        print(f"  {i:2d}. {name}{sub}")

    print()
    try:
        raw = input("  Enter number (or q to quit): ").strip()
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

    print(f"  Invalid choice: {raw}")
    return None
