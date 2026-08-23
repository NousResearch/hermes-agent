"""Shared curses-based UI components for Hermes CLI.

Used by `hermes tools` and `hermes skills` for interactive checklists.
Provides a curses multi-select with keyboard navigation, plus a
text-based numbered fallback for terminals without curses support.
"""
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Protocol, Sequence, Set, Tuple, Union

from hermes_cli.colors import Colors, color

# Rich radiolist rows: (text, style). style is None | "yellow" | "dim".
# Plain ``str`` items remain fully supported.
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
    if isinstance(item, str):
        return item
    return "".join(text for text, _style in item)


def _curses_style_attr(curses, style: Optional[str], *, is_cursor: bool):
    """Map a segment style to a curses attribute.

    Cursor rows force the whole line green (pair 1) so selection stays
    one solid highlight; unselected sale chrome uses yellow / dim.
    """
    if is_cursor:
        attr = curses.A_BOLD
        if curses.has_colors():
            attr |= curses.color_pair(1)
        return attr
    if style == "yellow" and curses.has_colors():
        return curses.color_pair(2)
    if style == "dim":
        attr = curses.A_DIM
        if curses.has_colors():
            # Pair 3 is the dim-gray status color (extra_color_pairs).
            try:
                attr |= curses.color_pair(3)
            except curses.error:
                pass
        return attr
    return curses.A_NORMAL


def _draw_description_line(stdscr, y: int, text: str, max_x: int) -> None:
    """Draw a description line, highlighting ★ in yellow when colors exist."""
    import curses

    col = 0
    i = 0
    star_attr = curses.A_NORMAL
    if curses.has_colors():
        star_attr = curses.color_pair(2)
    while i < len(text) and col < max_x - 1:
        remaining = max_x - 1 - col
        if remaining <= 0:
            break
        if text[i] == "★":
            try:
                stdscr.addnstr(y, col, "★", remaining, star_attr)
            except curses.error:
                pass
            col += 1
            i += 1
            continue
        next_star = text.find("★", i)
        chunk = text[i:] if next_star < 0 else text[i:next_star]
        chunk = chunk[:remaining]
        try:
            stdscr.addnstr(y, col, chunk, remaining, curses.A_NORMAL)
        except curses.error:
            pass
        col += len(chunk)
        i += len(chunk)


def _draw_radio_item(stdscr, y: int, x: int, item: RadioItem, max_x: int, *, is_cursor: bool) -> None:
    """Draw a plain or segmented radiolist item starting at column ``x``."""
    import curses

    if isinstance(item, str):
        attr = _curses_style_attr(curses, None, is_cursor=is_cursor)
        try:
            stdscr.addnstr(y, x, item, max(0, max_x - 1 - x), attr)
        except curses.error:
            pass
        return

    col = x
    for text, style in item:
        if col >= max_x - 1:
            break
        remaining = max_x - 1 - col
        if remaining <= 0:
            break
        chunk = text[:remaining]
        attr = _curses_style_attr(curses, style, is_cursor=is_cursor)
        try:
            stdscr.addnstr(y, col, chunk, remaining, attr)
        except curses.error:
            pass
        col += len(chunk)


def _query_matches(label: str, query: str) -> bool:
    """Return True when every query token is a case-insensitive subsequence."""
    normalized = label.lower()
    tokens = query.lower().split()

    if not tokens:
        return True

    for token in tokens:
        pos = 0

        for ch in token:
            pos = normalized.find(ch, pos)

            if pos < 0:
                return False

            pos += 1

    return True


_WORD_BOUNDARY = frozenset("-_/. ")


def _is_boundary(target: str, index: int) -> bool:
    """True if position ``index`` in ``target`` starts a word.

    Mirrors ``isBoundary`` in the TS scorer: start-of-string, after a
    separator char, or a lower->upper camelCase transition.
    """
    if index == 0:
        return True

    prev = target[index - 1]

    if prev in _WORD_BOUNDARY:
        return True

    # camelCase / lower->upper transition (e.g. the `O` in `gptO`).
    cur = target[index]

    return prev == prev.lower() and cur != cur.lower() and cur == cur.upper()


def _token_score(orig: str, lower: str, token: str) -> float | None:
    """Score one token against a target. None if the token isn't a subsequence.

    A faithful port of ``fuzzyScore`` in ui-tui/src/lib/fuzzy.ts and
    web/src/lib/fuzzy.ts so all three surfaces rank model ids identically:
    contiguous runs, word-boundary / first-char starts, prefix matches, and
    exact matches all score higher than scattered subsequence hits.

    ``lower`` is ``orig`` lowercased; matching is done against ``lower`` while
    boundary detection uses ``orig`` (so the camelCase rule works), exactly as
    in the TS scorer.
    """
    score = 0.0
    prev = -1
    search_from = 0
    positions: list[int] = []

    for ch in token:
        idx = lower.find(ch, search_from)

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
        search_from = idx + 1

    # Prefix bonus: the token matched a contiguous prefix of the target.
    if positions and positions[0] == 0 and positions[-1] == len(positions) - 1:
        score += 8

    # Exact full match dominates everything else.
    if lower == token:
        score += 20

    # Slightly prefer shorter targets when scores are otherwise close.
    score -= len(lower) * 0.01

    return score


def _fuzzy_score(label: str, query: str) -> float | None:
    """Aggregate score for a multi-token query (AND). None if any token fails.

    Mirrors ``fuzzyScoreMulti`` in the TS scorer: every whitespace-separated
    token must match; per-token scores are summed.
    """
    lower = label.lower()
    tokens = query.lower().split()

    if not tokens:
        return 0.0

    total = 0.0

    for token in tokens:
        token_score = _token_score(label, lower, token)

        if token_score is None:
            return None

        total += token_score

    return total


def _filter_indices(items: List[str], query: str) -> List[int]:
    """Return item indices matching *query*, ranked best-first.

    An empty query keeps every item in original order. Otherwise items are
    filtered to fuzzy matches and sorted by score descending, ties broken by
    original index so equal-scoring rows keep their catalog order.
    """
    q = query.strip()

    if not q:
        return list(range(len(items)))

    scored = []

    for i, label in enumerate(items):
        score = _fuzzy_score(label, q)

        if score is not None:
            scored.append((i, score))

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

    if cursor not in filtered:
        cursor = filtered[0]

    return cursor, filtered.index(cursor)


def _move_filtered_cursor(
    filtered: List[int], cursor: int, cursor_pos: int, delta: int
) -> int:
    """Move through the filtered index list, wrapping like the legacy menus."""
    if not filtered:
        return cursor

    return filtered[(cursor_pos + delta) % len(filtered)]


def _scroll_for_cursor(
    scroll_offset: int, cursor_pos: int, visible_rows: int, total_rows: int
) -> int:
    """Clamp scroll offset so the cursor remains visible."""
    visible_rows = max(1, visible_rows)

    if cursor_pos < scroll_offset:
        scroll_offset = cursor_pos
    elif cursor_pos >= scroll_offset + visible_rows:
        scroll_offset = cursor_pos - visible_rows + 1

    return max(0, min(scroll_offset, max(0, total_rows - visible_rows)))


def _handle_active_search_key(
    curses_mod, key: int, search: _SearchState
) -> tuple[bool, bool, bool]:
    """Handle a key while the search prompt is active.

    Returns ``(handled, confirm, changed)``. Active search consumes query
    editing keys, but leaves navigation keys for the menu loop to handle.
    """
    if not search.active:
        return False, False, False

    if key == 27:
        # Esc stops search AND clears the query, restoring the full list (so a
        # no-match filter can't strand the user on an empty list). Signals
        # `changed` when there was a query so the driver resets scroll/cursor.
        had_query = bool(search.query)
        search.active = False
        search.query = ""
        return True, False, had_query

    if key in (curses_mod.KEY_BACKSPACE, 127, 8):
        search.query = search.query[:-1]
        return True, False, True

    if key == 21:  # Ctrl+U
        search.query = ""
        return True, False, True

    if key in (curses_mod.KEY_ENTER, 10, 13):
        return True, True, False

    if 32 <= key < 127:  # printable ASCII; avoids Latin-1 mojibake from 128-255
        search.query += chr(key)
        return True, False, True

    return False, False, False


def flush_stdin() -> None:
    """Flush any stray bytes from the stdin input buffer.

    Must be called after ``curses.wrapper()`` returns, and before the next
    ``input()`` / ``getpass.getpass()`` call.
    ``curses.endwin()`` restores the terminal but does NOT drain the OS input buffer.
    Leftover escape-sequence bytes (from arrow keys, terminal mode-switch responses, or rapid keypresses)
    remain buffered and silently get consumed by the next ``input()`` call,
    corrupting user data (e.g. writing ``^[^[`` into .env files).

    On non-TTY stdin (piped, redirected) or Windows, this is a no-op.
    """
    try:
        if not sys.stdin.isatty():
            return
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
    """Read one keypress and normalize it to a menu action.

    Decodes raw arrow-key escape sequences in addition to the translated
    ``curses.KEY_*`` values.  Even with ``keypad(True)`` (which
    ``curses.wrapper`` sets), some terminals/terminfo entries deliver cursor
    keys as raw CSI/SS3 byte sequences — ``getch()`` then returns ``27`` (ESC)
    followed by e.g. ``[`` ``A``.  Treating that leading ``27`` as a cancel is
    what made the setup wizard's provider/model pickers bail to the numbered
    fallback the moment a user pressed up/down.

    Returns one of the ``NAV_*`` constants.  A lone ESC (no continuation byte
    within a short window) is the only thing that maps to ``NAV_CANCEL`` via
    the escape path; ``q`` also cancels.  Unknown sequences map to
    ``NAV_NONE`` so the caller simply ignores them rather than misfiring.
    """
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

    Split out from ``read_menu_key`` so search-aware loops can peek the raw
    key (e.g. to catch ``/``) before falling back to nav decoding.
    """
    import curses

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


# Sentinel: an on_action reducer returns this to mean "keep looping" (the
# keypress changed cursor/selection state but didn't resolve the menu).
_KEEP = object()


def _run_curses_menu(
    *,
    initial_cursor,
    item_count,
    draw_header,
    draw_row,
    on_action,
    reserve_bottom=1,
    draw_footer=None,
    extra_color_pairs=False,
    fallback,
    cancel_value,
    searchable=False,
    search_labels=None,
):
    """Shared curses single-/multi-select event loop.

    Owns every piece the three public menus used to duplicate verbatim:
    the non-TTY guard, ``curses.wrapper`` setup (cursor hide + color pairs),
    the per-frame ``clear``/``getmaxyx``/``refresh`` cycle, scroll-offset math,
    row iteration, the ``read_menu_key`` dispatch with ``NAV_UP``/``NAV_DOWN``
    cursor wrap, ``flush_stdin``, and the ``KeyboardInterrupt`` / curses-
    unavailable fallback. Per-menu behavior is supplied as callbacks so the
    rendered output stays byte-identical to the old hand-rolled loops.

    Callbacks / params:
        draw_header(stdscr, max_y, max_x) -> int
            Draw the title/hint/description rows. Returns the first screen row
            index where the scrollable item list should start. When search is
            active it receives the live ``_SearchState`` via the optional
            ``search`` keyword (drawn by the menu so the hint line can show it).
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
    context_back = bool(navigation_start and navigation_start.allow_back)
    if navigation_start is not None and navigation_start.should_replay:
        if navigation_handler is not None:
            navigation_handler(
                MenuNavigationEvent.RESOLVE, navigation_start.replay_value
            )
        return navigation_start.replay_value
    effective_allow_back = context_back

    # Non-TTY (piped/redirected stdin): curses and input() both hang or spin,
    # so return the cancel value directly — matching the pre-refactor guard in
    # each menu (the numbered fallback is only for curses errors on a real TTY).
    if not sys.stdin.isatty():
        return cancel_value

    use_search = searchable and search_labels is not None and len(search_labels) == item_count

    def _run_fallback():
        back_token = _NUMBERED_BACK_ENABLED.set(effective_allow_back)
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
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                if extra_color_pairs:
                    curses.init_pair(
                        3, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1
                    )
            cursor = initial_cursor
            scroll_offset = 0
            search = _SearchState()
            # Non-None labels for filtering; empty when search is disabled so
            # _filter_indices stays a cheap identity range.
            labels: List[str] = (
                search_labels if (use_search and search_labels is not None) else []
            )

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()

                filtered = (
                    _filter_indices(labels, search.query)
                    if use_search
                    else list(range(item_count))
                )
                cursor, cursor_pos = _reconcile_cursor(filtered, cursor)

                items_start = draw_header(
                    stdscr,
                    max_y,
                    max_x,
                    search=search,
                    back_enabled=effective_allow_back,
                )

                visible_rows = max(1, max_y - items_start - reserve_bottom)
                scroll_offset = _scroll_for_cursor(
                    scroll_offset, cursor_pos, visible_rows, len(filtered)
                )

                if use_search and search.query and not filtered:
                    try:
                        stdscr.addnstr(items_start, 0, "  No matches", max_x - 1, curses.A_DIM)
                    except curses.error:
                        pass

                for draw_i, filtered_pos in enumerate(
                    range(scroll_offset, min(len(filtered), scroll_offset + visible_rows))
                ):
                    i = filtered[filtered_pos]
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
                    action == NAV_BACK and effective_allow_back
                ):
                    if action == NAV_SELECT and use_search and not filtered:
                        continue
                    if navigation_handler is not None:
                        if action in (NAV_CANCEL, NAV_INTERRUPT):
                            navigation_handler(MenuNavigationEvent.CANCEL)
                        elif action == NAV_BACK and context_back:
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
    title: str,
    items: List[str],
    selected: Set[int],
    *,
    cancel_returns: Set[int] | None = None,
    status_fn: Optional[Callable[[Set[int]], str]] = None,
) -> Set[int]:
    """Curses multi-select checklist. Returns set of selected indices.

    Args:
        title: Header line displayed above the checklist.
        items: Display labels for each row.
        selected: Indices that start checked (pre-selected).
        cancel_returns: Returned on ESC/q. Defaults to the original *selected*.
        status_fn: Optional callback ``f(chosen_indices) -> str`` whose return
            value is rendered on the bottom row of the terminal.  Use this for
            live aggregate info (e.g. estimated token counts).
    """
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
        import curses
        check = "✓" if i in chosen else " "
        arrow = "→" if is_cursor else " "
        line = f" {arrow} [{check}] {items[i]}"
        attr = curses.A_NORMAL
        if is_cursor:
            attr = curses.A_BOLD
            if curses.has_colors():
                attr |= curses.color_pair(1)
        try:
            stdscr.addnstr(y, 0, line, max_x - 1, attr)
        except curses.error:
            pass

    def _draw_footer(stdscr, max_y, max_x):
        import curses
        try:
            status_text = status_fn(chosen)
            if status_text:
                # Right-align on the bottom row
                sx = max(0, max_x - len(status_text) - 1)
                sattr = curses.A_DIM
                if curses.has_colors():
                    sattr |= curses.color_pair(3)
                stdscr.addnstr(max_y - 1, sx, status_text, max_x - sx - 1, sattr)
        except curses.error:
            pass

    def _on_action(action, cursor):
        if action == NAV_TOGGLE:
            chosen.symmetric_difference_update({cursor})
            return _KEEP
        if action == NAV_SELECT:
            return set(chosen)
        return cancel_returns  # NAV_CANCEL

    return _run_curses_menu(
        initial_cursor=0,
        item_count=len(items),
        draw_header=_draw_header,
        draw_row=_draw_row,
        on_action=_on_action,
        reserve_bottom=(2 if status_fn else 1),
        draw_footer=_draw_footer if status_fn else None,
        extra_color_pairs=bool(status_fn),
        fallback=lambda: _numbered_fallback(title, items, selected, cancel_returns, status_fn),
        cancel_value=cancel_returns,
    )


def curses_radiolist(
    title: str,
    items: List[RadioItem],
    selected: int = 0,
    *,
    cancel_returns: int | None = None,
    description: str | None = None,
    searchable: bool = False,
    search_labels: List[str] | None = None,
) -> int:
    """Curses single-select radio list. Returns the selected index.

    Args:
        title: Header line displayed above the list.
        items: Display labels for each row. Each entry is either a plain
            string or a sequence of ``(text, style)`` segments where
            ``style`` is ``None``, ``"yellow"``, or ``"dim"``. Cursor rows
            force the whole line green; unselected rows honor segment styles
            (used for sale chrome in the model picker).
        selected: Index that starts selected (pre-selected).
        cancel_returns: Returned on ESC/q. Defaults to the original *selected*.
        description: Optional multi-line text shown between the title and
            the item list.  Useful for context that should survive the
            curses screen clear.
        searchable: When true, ``/`` opens a type-to-filter prompt. The
            returned value is always the original item index, not a filtered
            row position.
        search_labels: Optional haystacks for type-to-filter (length must
            match ``items``). Defaults to the display labels when omitted.
    """
    if cancel_returns is None:
        cancel_returns = selected

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
        import curses
        radio = "\u25cf" if i == selected else "\u25cb"
        arrow = "\u2192" if is_cursor else " "
        prefix = f" {arrow} ({radio}) "
        prefix_attr = _curses_style_attr(curses, None, is_cursor=is_cursor)
        try:
            stdscr.addnstr(y, 0, prefix, max_x - 1, prefix_attr)
        except curses.error:
            pass
        _draw_radio_item(
            stdscr, y, len(prefix), items[i], max_x, is_cursor=is_cursor
        )

    def _on_action(action, cursor):
        if action in (NAV_SELECT, NAV_TOGGLE):
            return cursor
        return cancel_returns  # NAV_CANCEL

    return _run_curses_menu(
        initial_cursor=selected,
        item_count=len(items),
        draw_header=_draw_header,
        draw_row=_draw_row,
        on_action=_on_action,
        reserve_bottom=1,
        # Dim gray (pair 3) for unselected "was …" sale chrome.
        extra_color_pairs=True,
        fallback=lambda: _radio_numbered_fallback(title, items, selected, cancel_returns),
        cancel_value=cancel_returns,
        searchable=searchable,
        search_labels=(
            list(search_labels)
            if searchable and search_labels is not None
            else (plain_labels if searchable else None)
        ),
    )


def format_radio_item_ansi(item: RadioItem) -> str:
    """Apply ANSI colors to a rich radiolist item (numbered fallback / prints)."""
    if isinstance(item, str):
        return item
    parts: list[str] = []
    for text, style in item:
        if style == "yellow":
            parts.append(color(text, Colors.YELLOW))
        elif style == "dim":
            parts.append(color(text, Colors.DIM))
        else:
            parts.append(text)
    return "".join(parts)


def _radio_numbered_fallback(
    title: str,
    items: List[RadioItem],
    selected: int,
    cancel_returns: int,
) -> int:
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


def curses_single_select(
    title: str,
    items: List[str],
    default_index: int = 0,
    *,
    cancel_label: str = "Cancel",
    searchable: bool = False,
) -> int | None:
    """Curses single-select menu. Returns selected index or None on cancel.

    Works inside prompt_toolkit. curses.wrapper() restores the terminal safely.

    When ``searchable`` is true, ``/`` opens a type-to-filter prompt; the
    returned value is always the original item index (or None for cancel).
    """
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
        import curses
        arrow = "→" if is_cursor else " "
        line = f" {arrow} {all_items[i]}"
        attr = curses.A_NORMAL
        if is_cursor:
            attr = curses.A_BOLD
            if curses.has_colors():
                attr |= curses.color_pair(1)
        try:
            stdscr.addnstr(y, 0, line, max_x - 1, attr)
        except curses.error:
            pass

    def _on_action(action, cursor):
        if action == NAV_SELECT:
            # Selecting the synthetic cancel row resolves to None, mirroring
            # the old post-loop ``>= cancel_idx`` guard.
            return None if cursor >= cancel_idx else cursor
        if action in (NAV_CANCEL, NAV_INTERRUPT):
            return None
        return _KEEP  # NAV_TOGGLE — no-op for this menu

    return _run_curses_menu(
        initial_cursor=min(default_index, len(all_items) - 1),
        item_count=len(all_items),
        draw_header=_draw_header,
        draw_row=_draw_row,
        on_action=_on_action,
        reserve_bottom=1,
        fallback=lambda: _numbered_single_fallback(title, all_items, cancel_idx),
        cancel_value=None,
        searchable=searchable,
        search_labels=list(all_items) if searchable else None,
    )


def _numbered_single_fallback(
    title: str,
    items: List[str],
    cancel_idx: int,
) -> int | None:
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
    title: str,
    items: List[str],
    selected: Set[int],
    cancel_returns: Set[int],
    status_fn: Optional[Callable[[Set[int]], str]] = None,
) -> Set[int]:
    """Text-based toggle fallback for terminals without curses."""
    chosen = set(selected)
    print(color(f"\n  {title}", Colors.YELLOW))
    print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))

    while True:
        for i, label in enumerate(items):
            marker = color("[✓]", Colors.GREEN) if i in chosen else "[ ]"
            print(f"  {marker} {i + 1:>2}. {label}")
        if status_fn:
            status_text = status_fn(chosen)
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
        print()

    return chosen
