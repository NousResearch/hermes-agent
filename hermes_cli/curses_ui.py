"""Curses checklist / radiolist / single-select menus with keyboard navigation and fuzzy ``/``
search, plus a numbered text fallback for terminals without curses."""
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Protocol, Sequence, Set, Tuple, Union

from hermes_cli.colors import Colors, color

# Rich radiolist rows: (text, style). style is None | "yellow" | "dim". Plain ``str`` works too.
RadioItem = Union[str, Sequence[Tuple[str, Optional[str]]]]
_NO_REPLAY = object()


@dataclass(frozen=True)
class MenuNavigationStart:
    """Navigation instructions returned when a scoped menu begins."""

    allow_back: bool = False
    replay_value: object = _NO_REPLAY

    @property
    def should_replay(self) -> bool:
        return self.replay_value is not _NO_REPLAY


class MenuNavigationEvent(str, Enum):
    BEGIN = "begin"
    RESOLVE = "resolve"
    CANCEL = "cancel"
    BACK = "back"


# Scoped flow controller: ``handler(event, value=None) -> MenuNavigationStart | None``.
MenuNavigationHandler = Callable[..., "MenuNavigationStart | None"]
_MENU_NAVIGATION_HANDLER: ContextVar[MenuNavigationHandler | None] = ContextVar(
    "hermes_menu_navigation_handler", default=None)
_NUMBERED_BACK_ENABLED: ContextVar[bool] = ContextVar("hermes_numbered_back_enabled", default=False)


def set_menu_navigation_handler(handler: MenuNavigationHandler) -> Token:
    """Scope setup-style cancel/back behavior to the current CLI invocation."""
    return _MENU_NAVIGATION_HANDLER.set(handler)


def reset_menu_navigation_handler(token: Token) -> None:
    """Restore the menu navigation handler active before ``token``."""
    _MENU_NAVIGATION_HANDLER.reset(token)


def _notify_scoped_navigation(event: MenuNavigationEvent) -> None:
    handler = _MENU_NAVIGATION_HANDLER.get()
    if handler is not None:
        handler(event)


class _NumberedNavigation(Enum):
    CANCEL = "cancel"
    BACK = "back"


_NAV_ABORT = object()


def _read_numbered_choice(prompt_text: str) -> int | None | object:
    """Numbered-fallback choice as a 0-based index; ``None`` for empty input, ``_NAV_ABORT`` when
    cancelled / backed out / interrupted / non-integer (scoped navigation is notified)."""
    try:
        val = _read_numbered_input(prompt_text)
    except (KeyboardInterrupt, EOFError):
        _notify_scoped_navigation(MenuNavigationEvent.CANCEL)
        return _NAV_ABORT
    if isinstance(val, _NumberedNavigation):
        _notify_scoped_navigation(MenuNavigationEvent(val.value))
        return _NAV_ABORT
    if not val.strip():
        return None
    idx = _parse_int(val.strip(), default=None)
    return _NAV_ABORT if idx is None else idx - 1


def _read_numbered_input(prompt_text: str) -> str | _NumberedNavigation:
    """Plain ``input()`` outside setup flows; inside a scoped flow prompt_toolkit supplies portable
    Escape / Ctrl+C / Left bindings on POSIX and native Windows."""
    if _MENU_NAVIGATION_HANDLER.get() is None:
        return input(prompt_text)
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    # Setup may run without the classic CLI, which normally installs CSI-u aliases at startup.
    from hermes_cli.pt_input_extras import install_modify_other_keys_aliases
    install_modify_other_keys_aliases()
    bindings = KeyBindings()

    @bindings.add(Keys.Escape)
    @bindings.add(Keys.ControlC)
    def _cancel(event) -> None:
        event.app.exit(result=_NumberedNavigation.CANCEL)

    if _NUMBERED_BACK_ENABLED.get():
        @bindings.add(Keys.Left)
        def _back(event) -> None:
            event.app.exit(result=_NumberedNavigation.BACK)
    return PromptSession().prompt(ANSI(prompt_text), key_bindings=bindings)


_NO_REPLAY = object()


@dataclass(frozen=True)
class MenuNavigationStart:
    """Navigation instructions returned when a scoped menu begins."""

    allow_back: bool = False
    replay_value: object = _NO_REPLAY

    @property
    def should_replay(self) -> bool:
        return self.replay_value is not _NO_REPLAY


class MenuNavigationEvent(str, Enum):
    BEGIN = "begin"
    RESOLVE = "resolve"
    CANCEL = "cancel"
    BACK = "back"


class MenuNavigationHandler(Protocol):
    """Typed contract between shared menus and a scoped flow controller."""

    def __call__(
        self,
        event: MenuNavigationEvent,
        value: object = None,
    ) -> MenuNavigationStart | None: ...


_MENU_NAVIGATION_HANDLER: ContextVar[MenuNavigationHandler | None] = ContextVar(
    "hermes_menu_navigation_handler", default=None
)
_NUMBERED_BACK_ENABLED: ContextVar[bool] = ContextVar(
    "hermes_numbered_back_enabled", default=False
)


def set_menu_navigation_handler(
    handler: MenuNavigationHandler,
) -> Token[MenuNavigationHandler | None]:
    """Scope setup-style cancel/back behavior to the current CLI invocation."""
    return _MENU_NAVIGATION_HANDLER.set(handler)


def reset_menu_navigation_handler(token: Token[MenuNavigationHandler | None]) -> None:
    """Restore the menu navigation handler active before ``token``."""
    _MENU_NAVIGATION_HANDLER.reset(token)


def _cancel_scoped_navigation() -> None:
    """Notify an active menu flow that a text fallback was interrupted."""
    handler = _MENU_NAVIGATION_HANDLER.get()
    if handler is not None:
        handler(MenuNavigationEvent.CANCEL)


def _back_scoped_navigation() -> None:
    """Notify an active menu flow that its text fallback requested back."""
    handler = _MENU_NAVIGATION_HANDLER.get()
    if handler is not None:
        handler(MenuNavigationEvent.BACK)


class _NumberedNavigation(Enum):
    CANCEL = "cancel"
    BACK = "back"


def _read_numbered_input(prompt_text: str) -> str | _NumberedNavigation:
    """Read a numbered fallback choice with setup navigation key bindings.

    Ordinary numbered menus retain their historical ``input()`` behavior.
    During setup/model flows, prompt_toolkit supplies portable Escape, Ctrl+C,
    and Left bindings on POSIX and native Windows when curses is unavailable.
    """
    if _MENU_NAVIGATION_HANDLER.get() is None:
        return input(prompt_text)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    # Setup can be invoked without importing the classic CLI, which normally
    # installs Ghostty/Kitty CSI-u aliases at process startup.
    from hermes_cli.pt_input_extras import install_modify_other_keys_aliases

    install_modify_other_keys_aliases()
    bindings = KeyBindings()

    @bindings.add(Keys.Escape)
    @bindings.add(Keys.ControlC)
    def _cancel(event) -> None:
        event.app.exit(result=_NumberedNavigation.CANCEL)

    if _NUMBERED_BACK_ENABLED.get():

        @bindings.add(Keys.Left)
        def _back(event) -> None:
            event.app.exit(result=_NumberedNavigation.BACK)

    return PromptSession().prompt(ANSI(prompt_text), key_bindings=bindings)


def radio_item_plain(item: RadioItem) -> str:
    """Flatten a radiolist item to searchable/plain display text."""
    return item if isinstance(item, str) else "".join(text for text, _style in item)


def _curses_style_attr(curses, style: Optional[str], *, is_cursor: bool):
    """Map a segment style to a curses attribute."""
    has_colors = curses.has_colors()
    if is_cursor:
        return curses.A_BOLD | (curses.color_pair(1) if has_colors else 0)
    if style == "yellow" and has_colors:
        return curses.color_pair(2)
    if style != "dim":
        return curses.A_NORMAL
    attr = curses.A_DIM
    if has_colors:
        try:
            attr |= curses.color_pair(3)  # dim-gray status pair (extra_color_pairs)
        except curses.error:
            pass
    return attr


def _addnstr(stdscr, y: int, x: int, text: str, n: int, attr) -> None:
    """``stdscr.addnstr`` that swallows ``curses.error`` (drawing past the screen edge)."""
    import curses
    try:
        stdscr.addnstr(y, x, text, n, attr)
    except curses.error:
        pass


def _draw_title_and_hint(stdscr, title: str, hint: str, max_x: int, *, hint_row: int = 1) -> None:
    """Bold/yellow menu title on row 0, dim key hint on ``hint_row``."""
    import curses
    hattr = curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0)
    _addnstr(stdscr, 0, 0, title, max_x - 1, hattr)
    _addnstr(stdscr, hint_row, 0, hint, max_x - 1, curses.A_DIM)


def _draw_segments(stdscr, y: int, x: int, segments, max_x: int) -> None:
    """Draw ``(text, attr)`` segments left to right from column ``x``, clipped at the edge."""
    col = x
    for text, attr in segments:
        remaining = max_x - 1 - col
        if remaining <= 0:
            break
        chunk = text[:remaining]
        _addnstr(stdscr, y, col, chunk, remaining, attr)
        col += len(chunk)


def _draw_description_line(stdscr, y: int, text: str, max_x: int) -> None:
    """Draw a description line, highlighting ★ in yellow when colors exist."""
    import curses
    star_attr = curses.color_pair(2) if curses.has_colors() else curses.A_NORMAL
    segments = []
    for i, part in enumerate(text.split("★")):
        if i:
            segments.append(("★", star_attr))
        if part:
            segments.append((part, curses.A_NORMAL))
    _draw_segments(stdscr, y, 0, segments, max_x)


def _draw_radio_item(
    stdscr, y: int, x: int, item: RadioItem, max_x: int, *, is_cursor: bool) -> None:
    """Draw a plain or segmented radiolist item starting at column ``x``."""
    import curses
    if isinstance(item, str):
        attr = _curses_style_attr(curses, None, is_cursor=is_cursor)
        _addnstr(stdscr, y, x, item, max(0, max_x - 1 - x), attr)
        return
    _draw_segments(
        stdscr, y, x,
        ((text, _curses_style_attr(curses, style, is_cursor=is_cursor)) for text, style in item),
        max_x)


def _draw_plain_row(stdscr, y: int, line: str, max_x: int, *, is_cursor: bool) -> None:
    """Draw a plain menu row, bold green when it is the cursor row."""
    import curses
    _addnstr(stdscr, y, 0, line, max_x - 1, _curses_style_attr(curses, None, is_cursor=is_cursor))


_WORD_BOUNDARY = frozenset("-_/. ")


def _is_boundary(target: str, index: int) -> bool:
    """Mirrors ``isBoundary`` in the TS scorer: start, after a separator, or lower->Upper."""
    if index == 0:
        return True
    prev, cur = target[index - 1], target[index]
    return prev in _WORD_BOUNDARY or (
        prev == prev.lower() and cur != cur.lower() and cur == cur.upper())


def _token_score(orig: str, lower: str, token: str) -> float | None:
    """Score one token against a target; None if not a subsequence. Faithful port of ``fuzzyScore``
    in ui-tui / web ``fuzzy.ts`` so all surfaces rank identically; matches run against ``lower``,
    boundary detection uses ``orig`` for camelCase."""
    score, prev = 0.0, -1
    positions: list[int] = []
    for ch in token:
        idx = lower.find(ch, prev + 1)
        if idx < 0:
            return None
        positions.append(idx)
        score += 1
        if prev >= 0 and idx == prev + 1:
            score += 5
        elif prev >= 0:
            score -= min(idx - prev - 1, 3)
        if _is_boundary(orig, idx):
            score += 3
        if idx == 0:
            score += 5
        prev = idx
    if positions and positions[0] == 0 and positions[-1] == len(positions) - 1:
        score += 8  # contiguous prefix
    if lower == token:
        score += 20  # exact match dominates
    return score - len(lower) * 0.01  # slightly prefer shorter targets


def _fuzzy_score(label: str, query: str) -> float | None:
    """Multi-token AND score (``fuzzyScoreMulti``): sum of per-token scores, None if any fails."""
    lower = label.lower()
    scores = [_token_score(label, lower, token) for token in query.lower().split()]
    return None if None in scores else sum(scores)


def _filter_indices(items: List[str], query: str) -> List[int]:
    """Item indices matching *query*, best-first; ties keep catalog order. Empty query = all."""
    q = query.strip()
    if not q:
        return list(range(len(items)))
    scored = [(i, s) for i, label in enumerate(items) if (s := _fuzzy_score(label, q)) is not None]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [i for i, _ in scored]


@dataclass
class _SearchState:
    """Mutable search state shared by curses picker loops."""
    active: bool = False
    query: str = ""


def _reconcile_cursor(filtered: List[int], cursor: int) -> tuple[int, int]:
    """Return ``(cursor, cursor_pos)`` inside the filtered index list."""
    if not filtered:
        return cursor, 0
    return (cursor, filtered.index(cursor)) if cursor in filtered else (filtered[0], 0)


def _move_filtered_cursor(filtered: List[int], cursor: int, cursor_pos: int, delta: int) -> int:
    """Move through the filtered index list, wrapping like the legacy menus."""
    return filtered[(cursor_pos + delta) % len(filtered)] if filtered else cursor


def _scroll_for_cursor(
    scroll_offset: int, cursor_pos: int, visible_rows: int, total_rows: int) -> int:
    """Clamp scroll offset so the cursor remains visible."""
    visible_rows = max(1, visible_rows)
    if cursor_pos < scroll_offset:
        scroll_offset = cursor_pos
    elif cursor_pos >= scroll_offset + visible_rows:
        scroll_offset = cursor_pos - visible_rows + 1
    return max(0, min(scroll_offset, max(0, total_rows - visible_rows)))


def _handle_active_search_key(
    curses_mod, key: int, search: _SearchState) -> tuple[bool, bool, bool]:
    """Handle a key while the search prompt is active -> ``(handled, confirm, changed)``."""
    if not search.active:
        return False, False, False
    if key == 27:
        # Esc stops search AND clears the query so a no-match filter can't strand the user on
        # an empty list; `changed` when there was a query so the driver resets scroll/cursor.
        had_query = bool(search.query)
        search.active = False
        search.query = ""
        return True, False, had_query
    if key in (curses_mod.KEY_ENTER, 10, 13):
        return True, True, False
    if key in (curses_mod.KEY_BACKSPACE, 127, 8):
        search.query = search.query[:-1]
    elif key == 21:  # Ctrl+U
        search.query = ""
    elif 32 <= key < 127:  # printable ASCII; avoids Latin-1 mojibake from 128-255
        search.query += chr(key)
    else:
        return False, False, False
    return True, False, True


def flush_stdin() -> None:
    """Drain stray stdin bytes after ``curses.wrapper()`` and before the next ``input()``:
    ``curses.endwin()`` restores the terminal but does NOT drain the OS input buffer."""
    try:
        if sys.stdin.isatty():
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


# Normalized menu actions returned by ``read_menu_key``.  Using sentinels keeps
# every menu's key-handling branch identical and free of raw escape-byte logic.
NAV_UP = "up"
NAV_DOWN = "down"
NAV_BACK = "back"
NAV_SELECT = "select"
NAV_TOGGLE = "toggle"
NAV_CANCEL = "cancel"
NAV_INTERRUPT = "interrupt"
NAV_NONE = "none"


def read_menu_key(stdscr) -> str:
    """Read one keypress and normalize it to a ``NAV_*`` action: lone ESC (no continuation byte
    within a short window) and ``q`` cancel; unknown sequences map to ``NAV_NONE``."""
    return _decode_menu_key(stdscr, stdscr.getch())


@dataclass(frozen=True)
class _EnhancedKey:
    codepoint: int
    modifier: int = 1
    event_type: int = 1


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _parse_csi_u_key(raw: str) -> _EnhancedKey | None:
    """Parse a Kitty/CSI-u key, preserving its press/repeat/release type."""
    parts = raw.split(";")
    codepoint = _parse_int(parts[0].split(":", 1)[0]) if parts else 0
    if not codepoint:
        return None

    modifier = 1
    event_type = 1
    if len(parts) > 1:
        modifier_parts = parts[1].split(":")
        modifier = _parse_int(modifier_parts[0], 1)
        if len(modifier_parts) > 1:
            event_type = _parse_int(modifier_parts[1], 1)
    return _EnhancedKey(codepoint, modifier, event_type)


def _parse_csi_numbers(raw: str) -> list[int]:
    """Parse semicolon-delimited CSI numbers for modifyOtherKeys."""
    return [_parse_int(part.split(":", 1)[0]) for part in raw.split(";")]


def _enhanced_key_action(codepoint: int, modifier: int = 1) -> str:
    """Map CSI-u/modifyOtherKeys codepoints to setup menu actions."""
    if codepoint in (10, 13):
        return NAV_SELECT
    if codepoint == 27:
        return NAV_CANCEL
    if codepoint == 32:
        return NAV_TOGGLE

    # CSI-u encodes Ctrl+C as codepoint `c` plus the Ctrl modifier. Lock-state
    # bits may be added to the modifier, so inspect the Ctrl bit rather than
    # matching only the canonical value 5.
    has_ctrl = bool((max(1, modifier) - 1) & 4)
    if codepoint == 3 or (codepoint in (ord("c"), ord("C")) and has_ctrl):
        return NAV_INTERRUPT
    return NAV_NONE


def _read_csi_tail(stdscr) -> tuple[str, int | None]:
    """Read CSI/SS3 parameter bytes through the final byte."""
    raw: list[str] = []
    for _ in range(32):
        value = stdscr.getch()
        if value == -1:
            return "".join(raw), None
        if 0x40 <= value <= 0x7E:
            return "".join(raw), value
        if 0x20 <= value <= 0x3F:
            raw.append(chr(value))
            continue
        return "".join(raw), None
    return "".join(raw), None


def _decode_menu_key(stdscr, key: int) -> str:
    """Normalize an already-read keypress to a menu action.


    if key in (curses.KEY_UP, ord("k")):
        return NAV_UP
    if key in (curses.KEY_DOWN, ord("j")):
        return NAV_DOWN
    if key == curses.KEY_LEFT:
        return NAV_BACK
    if key == 3:  # Ctrl+C in curses raw/cbreak mode.
        return NAV_INTERRUPT
    if key in (curses.KEY_ENTER, 10, 13):
        return NAV_SELECT
    if key == ord(" "):
        return NAV_TOGGLE
    if key == ord("q"):
        return NAV_CANCEL

    if key == 27:  # ESC — could be a lone ESC (cancel) or an escape sequence.
        # Wait briefly for a continuation byte.  On slow PTYs (SSH/tmux) the
        # bytes of an arrow key can arrive across separate reads, so a tiny
        # timeout avoids misreading a split sequence as a bare ESC.
        try:
            stdscr.timeout(60)
            nxt = stdscr.getch()
            if nxt == -1:
                return NAV_CANCEL  # genuine lone ESC

            if nxt in (ord("["), ord("O")):  # CSI / SS3 introducer
                raw_params, final = _read_csi_tail(stdscr)
                if final in (ord("A"), ord("k")):
                    return NAV_UP
                if final in (ord("B"), ord("j")):
                    return NAV_DOWN
                if final == ord("D"):
                    return NAV_BACK
                if final == ord("u"):
                    enhanced = _parse_csi_u_key(raw_params)
                    if enhanced is not None:
                        if enhanced.event_type == 3:  # key release
                            return NAV_NONE
                        return _enhanced_key_action(
                            enhanced.codepoint, enhanced.modifier
                        )
                if final == ord("~"):
                    params = _parse_csi_numbers(raw_params)
                    if len(params) >= 3 and params[0] == 27:
                        return _enhanced_key_action(params[2], params[1])
                return NAV_NONE
            # ESC followed by some other byte we don't handle — swallow it.
            return NAV_NONE
        finally:
            stdscr.timeout(-1)  # restore blocking mode

    return NAV_NONE


def _decode_menu_key(stdscr, key: int) -> str:
    """Normalize an already-read keypress to a menu action (lets loops peek the raw key first)."""
    import curses
    plain = {
        curses.KEY_UP: NAV_UP, ord("k"): NAV_UP, curses.KEY_DOWN: NAV_DOWN, ord("j"): NAV_DOWN,
        curses.KEY_LEFT: NAV_BACK, 3: NAV_INTERRUPT,  # 3 = Ctrl+C in raw/cbreak mode
        curses.KEY_ENTER: NAV_SELECT, 10: NAV_SELECT, 13: NAV_SELECT,
        ord(" "): NAV_TOGGLE, ord("q"): NAV_CANCEL}
    if key in plain:
        return plain[key]
    if key != 27:
        return NAV_NONE
    # ESC: wait briefly for a continuation byte. On slow PTYs (SSH/tmux) an arrow key's bytes
    # can arrive across separate reads, so a tiny timeout avoids misreading it as a bare ESC.
    try:
        stdscr.timeout(60)
        return _decode_escape_sequence(stdscr)
    finally:
        stdscr.timeout(-1)  # restore blocking mode


_KEEP = object()  # on_action reducer result: keep looping (state changed, menu not resolved)
_CONSUMED = object()  # ``_route_key`` result: key eaten by the search prompt; redraw, no action


_RESOLVING = frozenset({NAV_SELECT, NAV_TOGGLE, NAV_CANCEL, NAV_INTERRUPT, NAV_BACK})


def _init_colors(curses, extra_color_pairs: bool) -> None:
    curses.curs_set(0)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        if extra_color_pairs:
            curses.init_pair(3, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)


def _route_key(curses, stdscr, key: int, search: _SearchState, use_search: bool):
    """Turn a raw key into ``(action, changed)`` honoring an active ``/`` search prompt; ``action``
    may be ``_CONSUMED`` (redraw only), ``changed`` means the query changed."""
    if not use_search:
        return _decode_menu_key(stdscr, key), False
    if search.active and key == 27:
        # Enhanced keys (Ghostty/Kitty) also start with ESC: decode the full sequence before
        # treating a genuine Escape as "stop search".
        action = _decode_menu_key(stdscr, key)
        if action == NAV_CANCEL:
            search.active = False
            search.query = ""
            return _CONSUMED, True
        return (_CONSUMED if action == NAV_NONE else action), False
    if search.active:
        # Active search consumes query-editing keys; nav keys fall through.
        handled, confirm, changed = _handle_active_search_key(curses, key, search)
        if confirm:
            return NAV_SELECT, changed
        return (_CONSUMED if handled else _decode_menu_key(stdscr, key)), changed
    if key == ord("/"):
        search.active = True
        return _CONSUMED, False
    return _decode_menu_key(stdscr, key), False


def _run_curses_menu(
    *, initial_cursor, item_count, draw_header, draw_row, on_action, reserve_bottom=1,
    draw_footer=None, extra_color_pairs=False, fallback, cancel_value, searchable=False,
    search_labels=None):
    """Shared curses single-/multi-select event loop: non-TTY guard, ``curses.wrapper`` setup,
    clear/refresh cycle, scroll math, key dispatch with cursor wrap, KeyboardInterrupt /
    curses-unavailable fallback.

    Owns every piece the three public menus used to duplicate verbatim:
    the non-TTY guard, ``curses.wrapper`` setup (cursor hide + color pairs),
    the per-frame ``clear``/``getmaxyx``/``refresh`` cycle, scroll-offset math,
    row iteration, the ``read_menu_key`` dispatch with ``NAV_UP``/``NAV_DOWN``
    cursor wrap, ``flush_stdin``, and the ``KeyboardInterrupt`` / curses-
    unavailable fallback. Per-menu behavior is supplied as callbacks so the
    rendered output stays byte-identical to the old hand-rolled loops.

    Callbacks / params:
        draw_header(stdscr, max_y, max_x, *, search=None, back_enabled=False) -> int
            Draw the title/hint/description rows. Returns the first screen row
            index where the scrollable item list should start. When search is
            active it receives the live ``_SearchState`` via the optional
            ``search`` keyword (drawn by the menu so the hint line can show it).
            ``back_enabled`` controls whether the ``← previous`` hint is shown.
        draw_row(stdscr, y, idx, is_cursor, max_x) -> None
            Draw one item row. ``idx`` is always the ORIGINAL item index, so
            per-menu rendering is unchanged whether or not a filter is active.
        on_action(action, cursor) -> value
            Reducer for SELECT/TOGGLE/CANCEL/BACK. Return ``_KEEP`` to continue the
            loop; return anything else to resolve the menu with that value.
            (UP/DOWN cursor movement is handled by the driver itself.)
        reserve_bottom: number of bottom screen rows kept clear of items
            (1 = leave the final row blank, matching the old loops).
        draw_footer(stdscr, max_y, max_x) -> None
            Optional bottom-row painter (e.g. a status bar). Drawn after the
            item rows; its row budget must be included in ``reserve_bottom``.
        extra_color_pairs: also init pair 3 (dim gray) for status bars.
        fallback() -> value
            Called when curses errors out on a real TTY (curses unavailable).
        cancel_value: returned on non-TTY stdin, ESC/cancel, or KeyboardInterrupt.
        searchable: when true, ``/`` opens a type-to-filter prompt over
            ``search_labels``. Returned values are always ORIGINAL item indices.
        search_labels: per-item text used for filtering (required when
            ``searchable`` is true; length must equal ``item_count``).
    """
    navigation_handler = _MENU_NAVIGATION_HANDLER.get()
    navigation_start = (
        navigation_handler(MenuNavigationEvent.BEGIN)
        if navigation_handler is not None
        else None
    )
    if navigation_start is not None and not isinstance(
        navigation_start, MenuNavigationStart
    ):
        raise TypeError("menu navigation 'begin' must return MenuNavigationStart")
    allow_back = bool(navigation_start and navigation_start.allow_back)
    if navigation_start is not None and navigation_start.should_replay:
        if navigation_handler is not None:
            navigation_handler(
                MenuNavigationEvent.RESOLVE, navigation_start.replay_value
            )
        return navigation_start.replay_value

    # Non-TTY (piped/redirected stdin): curses and input() both hang or spin,
    # so return the cancel value directly — matching the pre-refactor guard in
    # each menu (the numbered fallback is only for curses errors on a real TTY).
    if not sys.stdin.isatty():
        return cancel_value
    use_search = searchable and search_labels is not None and len(search_labels) == item_count

    def _run_fallback():
        back_token = _NUMBERED_BACK_ENABLED.set(allow_back)
        try:
            result = fallback()
        finally:
            _NUMBERED_BACK_ENABLED.reset(back_token)
        if navigation_handler is not None:
            navigation_handler(MenuNavigationEvent.RESOLVE, result)
        return result

    try:
        import curses
    except ImportError:
        return _run_fallback()

    try:
        result_holder = [_KEEP]

        def _draw(stdscr):
            _init_colors(curses, extra_color_pairs)
            cursor, scroll_offset, search = initial_cursor, 0, _SearchState()
            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                filtered = (
                    _filter_indices(search_labels, search.query) if use_search
                    else list(range(item_count)))
                cursor, cursor_pos = _reconcile_cursor(filtered, cursor)

                items_start = draw_header(
                    stdscr,
                    max_y,
                    max_x,
                    search=search,
                    back_enabled=allow_back,
                )

                visible_rows = max(1, max_y - items_start - reserve_bottom)
                scroll_offset = _scroll_for_cursor(
                    scroll_offset, cursor_pos, visible_rows, len(filtered))
                if use_search and search.query and not filtered:
                    _addnstr(stdscr, items_start, 0, "  No matches", max_x - 1, curses.A_DIM)
                for draw_i, i in enumerate(filtered[scroll_offset : scroll_offset + visible_rows]):
                    y = draw_i + items_start
                    if y >= max_y - reserve_bottom:
                        break
                    draw_row(stdscr, y, i, i == cursor, max_x)
                if draw_footer is not None:
                    draw_footer(stdscr, max_y, max_x)
                stdscr.refresh()

                if use_search:
                    key = stdscr.getch()

                    if search.active and key == 27:
                        # Ghostty/Kitty enhanced keys also begin with ESC.
                        # Decode the full sequence before treating a genuine
                        # Escape as "stop search"; otherwise Enter/Left/Ctrl+C
                        # lose their tail while the search prompt is active.
                        action = _decode_menu_key(stdscr, key)
                        if action == NAV_CANCEL:
                            search.active = False
                            search.query = ""
                            scroll_offset = 0
                            continue
                        if action == NAV_NONE:
                            continue
                    elif search.active:
                        # Active search consumes query-editing keys; nav keys
                        # fall through to be decoded below.
                        handled, confirm, changed = _handle_active_search_key(
                            curses, key, search
                        )
                        if changed:
                            scroll_offset = 0
                            cursor, cursor_pos = _reconcile_cursor(
                                _filter_indices(search_labels, search.query), cursor
                            )
                        if confirm:
                            if filtered:
                                outcome = on_action(NAV_SELECT, cursor)
                                if outcome is not _KEEP:
                                    if navigation_handler is not None:
                                        navigation_handler(
                                            MenuNavigationEvent.RESOLVE, outcome
                                        )
                                    result_holder[0] = outcome
                                    return
                            continue
                        if handled:
                            continue
                        action = _decode_menu_key(stdscr, key)
                    elif key == ord("/"):
                        search.active = True
                        continue
                    else:
                        action = _decode_menu_key(stdscr, key)
                else:
                    action = read_menu_key(stdscr)

                if action == NAV_UP:
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, -1)
                elif action == NAV_DOWN:
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, 1)
                elif action in (
                    NAV_SELECT,
                    NAV_TOGGLE,
                    NAV_CANCEL,
                    NAV_INTERRUPT,
                ) or (
                    action == NAV_BACK and allow_back
                ):
                    if action == NAV_SELECT and use_search and not filtered:
                        continue
                    if navigation_handler is not None:
                        if action in (NAV_CANCEL, NAV_INTERRUPT):
                            navigation_handler(MenuNavigationEvent.CANCEL)
                        elif action == NAV_BACK and allow_back:
                            navigation_handler(MenuNavigationEvent.BACK)
                    outcome = on_action(action, cursor)
                    if outcome is not _KEEP:
                        if navigation_handler is not None:
                            navigation_handler(MenuNavigationEvent.RESOLVE, outcome)
                        result_holder[0] = outcome
                        return

        curses.wrapper(_draw)
        flush_stdin()
        return result_holder[0] if result_holder[0] is not _KEEP else cancel_value
    except KeyboardInterrupt:
        if navigation_handler is not None:
            navigation_handler(MenuNavigationEvent.CANCEL)
        return cancel_value
    except curses.error:
        return _run_fallback()


def curses_checklist(
    title: str, items: List[str], selected: Set[int], *, cancel_returns: Set[int] | None = None,
    status_fn: Optional[Callable[[Set[int]], str]] = None) -> Set[int]:
    """Curses multi-select checklist -> set of selected indices. ``cancel_returns`` (default: the
    original *selected*) is returned on ESC/q; ``status_fn(chosen)`` renders on the bottom row
    for live aggregate info such as token estimates."""
    if cancel_returns is None:
        cancel_returns = set(selected)
    chosen = set(selected)

    def _draw_header(stdscr, max_y, max_x, search=None, back_enabled=False):
        import curses
        try:
            hattr = curses.A_BOLD
            if curses.has_colors():
                hattr |= curses.color_pair(2)
            stdscr.addnstr(0, 0, title, max_x - 1, hattr)
            hint = "  ↑↓ navigate  SPACE toggle  ENTER confirm  ESC cancel"
            if back_enabled:
                hint += "  ← previous"
            stdscr.addnstr(1, 0, hint, max_x - 1, curses.A_DIM)
        except curses.error:
            pass
        return 3

    def _draw_row(stdscr, y, i, is_cursor, max_x):
        line = f" {'→' if is_cursor else ' '} [{'✓' if i in chosen else ' '}] {items[i]}"
        _draw_plain_row(stdscr, y, line, max_x, is_cursor=is_cursor)

    def _draw_footer(stdscr, max_y, max_x):
        import curses
        status_text = status_fn(chosen)
        if status_text:  # right-aligned on the bottom row
            sx = max(0, max_x - len(status_text) - 1)
            sattr = curses.A_DIM | (curses.color_pair(3) if curses.has_colors() else 0)
            _addnstr(stdscr, max_y - 1, sx, status_text, max_x - sx - 1, sattr)

    def _on_action(action, cursor):
        if action == NAV_TOGGLE:
            chosen.symmetric_difference_update({cursor})
            return _KEEP
        if action == NAV_SELECT:
            return set(chosen)
        return cancel_returns  # NAV_CANCEL

    return _run_curses_menu(
        initial_cursor=0, item_count=len(items),
        draw_header=_simple_header(title, "SPACE toggle  ENTER confirm", "ESC cancel", False),
        draw_row=_draw_row, on_action=_on_action, reserve_bottom=(2 if status_fn else 1),
        draw_footer=_draw_footer if status_fn else None, extra_color_pairs=bool(status_fn),
        fallback=lambda: _numbered_fallback(title, items, selected, cancel_returns, status_fn),
        cancel_value=cancel_returns)


def _search_hint(search, searchable: bool, confirm: str, cancel: str, back_enabled: bool) -> str:
    """Key-hint row for menus, swapping to the search prompt while ``/`` is active."""
    if searchable and search is not None and search.active:
        hint = f"  Search: {search.query}\u258e  BACKSPACE edit  Ctrl+U clear  ESC stop"
    else:
        hint = f"  \u2191\u2193 navigate  {confirm}  {'/ search  ' if searchable else ''}{cancel}"
    if back_enabled:
        hint += "  \u2190 previous"
    return hint


def _simple_header(title: str, confirm: str, cancel: str, searchable: bool):
    """``draw_header`` callback: title on row 0, key hint on row 1, items start on row 3."""
    def _draw_header(stdscr, max_y, max_x, search=None, back_enabled=False):
        hint = _search_hint(search, searchable, confirm, cancel, back_enabled)
        _draw_title_and_hint(stdscr, title, hint, max_x)
        return 3

    return _draw_header


def curses_radiolist(
    title: str, items: List[RadioItem], selected: int = 0, *, cancel_returns: int | None = None,
    description: str | None = None, searchable: bool = False,
    search_labels: List[str] | None = None) -> int:
    """Curses single-select radio list -> selected index.

    Items are plain strings or ``(text, style)`` segment sequences (``None``/``"yellow"``/
    ``"dim"``); the cursor row is forced green. ``description`` is shown between title and list
    so context survives the curses screen clear. With ``searchable``, ``/`` filters over
    ``search_labels`` (default: display labels); the return value is always the ORIGINAL index.
    """
    if cancel_returns is None:
        cancel_returns = selected
    desc_lines = description.splitlines() if description else []
    if searchable:
        search_labels = (
            list(search_labels) if search_labels is not None
            else [radio_item_plain(item) for item in items])

    desc_lines: list[str] = []
    if description:
        desc_lines = description.splitlines()

    plain_labels = [radio_item_plain(item) for item in items]

    def _draw_header(stdscr, max_y, max_x, search=None, back_enabled=False):
        import curses
        row = 0
        try:
            hattr = curses.A_BOLD
            if curses.has_colors():
                hattr |= curses.color_pair(2)
            stdscr.addnstr(row, 0, title, max_x - 1, hattr)
            row += 1

            # Description lines — paint ★ yellow so the sale legend matches rows.
            for dline in desc_lines:
                if row >= max_y - 1:
                    break
                _draw_description_line(stdscr, row, dline, max_x)
                row += 1

            if searchable and search is not None and search.active:
                hint = f"  Search: {search.query}\u258e  BACKSPACE edit  Ctrl+U clear  ESC stop"
            elif searchable:
                hint = "  \u2191\u2193 navigate  ENTER/SPACE select  / search  ESC cancel"
            else:
                hint = "  \u2191\u2193 navigate  ENTER/SPACE select  ESC cancel"
            if back_enabled:
                hint += "  \u2190 previous"
            stdscr.addnstr(row, 0, hint, max_x - 1, curses.A_DIM)
            row += 1
        except curses.error:
            pass
        # One blank row between the hint and the item list.
        return row + 1

    def _draw_row(stdscr, y, i, is_cursor, max_x):
        radio, arrow = "\u25cf" if i == selected else "\u25cb", "\u2192" if is_cursor else " "
        prefix = f" {arrow} ({radio}) "
        _draw_plain_row(stdscr, y, prefix, max_x, is_cursor=is_cursor)
        _draw_radio_item(stdscr, y, len(prefix), items[i], max_x, is_cursor=is_cursor)

    def _on_action(action, cursor):
        return cursor if action in (NAV_SELECT, NAV_TOGGLE) else cancel_returns  # NAV_CANCEL

    return _run_curses_menu(
        initial_cursor=selected, item_count=len(items), draw_header=_draw_header,
        draw_row=_draw_row, on_action=_on_action,
        extra_color_pairs=True,  # dim gray (pair 3) for unselected "was …" sale chrome
        fallback=lambda: _radio_numbered_fallback(title, items, selected, cancel_returns),
        cancel_value=cancel_returns, searchable=searchable,
        search_labels=search_labels if searchable else None)


_ANSI_STYLE = {"yellow": Colors.YELLOW, "dim": Colors.DIM}


def format_radio_item_ansi(item: RadioItem) -> str:
    """Apply ANSI colors to a rich radiolist item (numbered fallback / prints)."""
    if isinstance(item, str):
        return item
    return "".join(
        color(text, _ANSI_STYLE[style]) if style in _ANSI_STYLE else text for text, style in item)


def _radio_numbered_fallback(
    title: str, items: List[RadioItem], selected: int, cancel_returns: int) -> int:
    """Text-based numbered fallback for radio selection."""
    print(color(f"\n  {title}", Colors.YELLOW))
    print(color("  Select by number, Enter to confirm.\n", Colors.DIM))
    for i, label in enumerate(items):
        marker = color("(\u25cf)", Colors.GREEN) if i == selected else "(\u25cb)"
        print(f"  {marker} {i + 1:>2}. {format_radio_item_ansi(label)}")
    print()
    try:
        val = _read_numbered_input(
            color(f"  Choice [default {selected + 1}]: ", Colors.DIM)
        )
        if val is _NumberedNavigation.BACK:
            _back_scoped_navigation()
            return cancel_returns
        if val is _NumberedNavigation.CANCEL:
            _cancel_scoped_navigation()
            return cancel_returns
        val = val.strip()
        if not val:
            return selected
        idx = int(val) - 1
        if 0 <= idx < len(items):
            return idx
        return selected
    except ValueError:
        return cancel_returns
    except (KeyboardInterrupt, EOFError):
        _cancel_scoped_navigation()
        return cancel_returns
    return idx if idx is not None and 0 <= idx < len(items) else selected


def curses_single_select(
    title: str, items: List[str], default_index: int = 0, *, cancel_label: str = "Cancel",
    searchable: bool = False) -> int | None:
    """Curses single-select menu -> selected index or None on cancel. With ``searchable``, ``/``
    opens a type-to-filter prompt; the return value is always the original item index."""
    all_items = list(items) + [cancel_label]
    cancel_idx = len(items)

    def _draw_header(stdscr, max_y, max_x, search=None, back_enabled=False):
        import curses
        try:
            hattr = curses.A_BOLD
            if curses.has_colors():
                hattr |= curses.color_pair(2)
            stdscr.addnstr(0, 0, title, max_x - 1, hattr)
            if searchable and search is not None and search.active:
                hint = f"  Search: {search.query}\u258e  BACKSPACE edit  Ctrl+U clear  ESC stop"
            elif searchable:
                hint = "  ↑↓ navigate  ENTER confirm  / search  ESC/q cancel"
            else:
                hint = "  ↑↓ navigate  ENTER confirm  ESC/q cancel"
            if back_enabled:
                hint += "  ← previous"
            stdscr.addnstr(1, 0, hint, max_x - 1, curses.A_DIM)
        except curses.error:
            pass
        return 3

    def _draw_row(stdscr, y, i, is_cursor, max_x):
        line = f" {'→' if is_cursor else ' '} {all_items[i]}"
        _draw_plain_row(stdscr, y, line, max_x, is_cursor=is_cursor)

    def _on_action(action, cursor):
        if action == NAV_SELECT:  # the synthetic cancel row resolves to None
            return None if cursor >= cancel_idx else cursor
        if action in (NAV_CANCEL, NAV_INTERRUPT):
            return None
        return _KEEP  # NAV_TOGGLE — no-op for this menu

    return _run_curses_menu(
        initial_cursor=min(default_index, len(all_items) - 1), item_count=len(all_items),
        draw_header=_simple_header(title, "ENTER confirm", "ESC/q cancel", searchable),
        draw_row=_draw_row, on_action=_on_action,
        fallback=lambda: _numbered_single_fallback(title, all_items, cancel_idx),
        cancel_value=None, searchable=searchable,
        search_labels=list(all_items) if searchable else None)


def _numbered_single_fallback(title: str, items: List[str], cancel_idx: int) -> int | None:
    """Text-based numbered fallback for single-select."""
    print(f"\n  {title}\n")
    for i, label in enumerate(items, 1):
        print(f"  {i}. {label}")
    print()
    try:
        val = _read_numbered_input(f"  Choice [1-{len(items)}]: ")
        if val is _NumberedNavigation.BACK:
            _back_scoped_navigation()
            return None
        if val is _NumberedNavigation.CANCEL:
            _cancel_scoped_navigation()
            return None
        val = val.strip()
        if not val:
            return None
        idx = int(val) - 1
        if 0 <= idx < len(items) and idx < cancel_idx:
            return idx
        if idx == cancel_idx:
            return None
    except ValueError:
        pass
    except (KeyboardInterrupt, EOFError):
        _cancel_scoped_navigation()
    return None


def _numbered_fallback(
    title: str, items: List[str], selected: Set[int], cancel_returns: Set[int],
    status_fn: Optional[Callable[[Set[int]], str]] = None) -> Set[int]:
    """Text-based toggle fallback for terminals without curses."""
    chosen = set(selected)
    print(color(f"\n  {title}", Colors.YELLOW))
    print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))
    while True:
        for i, label in enumerate(items):
            marker = color("[✓]", Colors.GREEN) if i in chosen else "[ ]"
            print(f"  {marker} {i + 1:>2}. {label}")
        status_text = status_fn(chosen) if status_fn else ""
        if status_text:
            print(color(f"\n  {status_text}", Colors.DIM))
        print()
        try:
            val = _read_numbered_input(
                color("  Toggle # (or Enter to confirm): ", Colors.DIM)
            )
            if val is _NumberedNavigation.BACK:
                _back_scoped_navigation()
                return cancel_returns
            if val is _NumberedNavigation.CANCEL:
                _cancel_scoped_navigation()
                return cancel_returns
            val = val.strip()
            if not val:
                break
            idx = int(val) - 1
            if 0 <= idx < len(items):
                chosen.symmetric_difference_update({idx})
        except ValueError:
            return cancel_returns
        except (KeyboardInterrupt, EOFError):
            _cancel_scoped_navigation()
            return cancel_returns
        if idx is None:
            return chosen
        if 0 <= idx < len(items):
            chosen.symmetric_difference_update({idx})
        print()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Protocol  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
