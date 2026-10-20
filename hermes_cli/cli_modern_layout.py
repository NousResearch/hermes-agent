"""Glyph-grammar transcript and bottom-anchored chrome for the classic CLI (``display.cli_layout``).

Every row this module prints into scrollback is width-independent (a column-0 glyph, no boxes, no
full-width rules), so resizing the terminal cannot re-wrap and garble what is already printed.
Width-dependent chrome (the composer's top rule and the footer) lives only in prompt_toolkit's live
region, which is redrawn on every resize. ``display.cli_layout: classic`` restores the boxed look.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime

from agent.i18n import t

logger = logging.getLogger(__name__)

_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_RST, _BOLD, _DIM = "\x1b[0m", "\x1b[1m", "\x1b[2m"
_RED = "\x1b[38;2;239;83;80m"
_TITLE_MAX = 72


def is_modern(cli) -> bool:
    return bool(getattr(cli, "_modern_layout", False))


def layout_from_config(display: dict) -> bool:
    """True for the glyph layout (default); ``classic`` keeps the boxed transcript."""
    return str((display or {}).get("cli_layout", "modern") or "modern").strip().lower() != "classic"


def busy_input_mode_from_config(display: dict) -> str:
    """``display.busy_input_mode``; unset interrupts in both layouts (Tab still queues in this one)."""
    return str((display or {}).get("busy_input_mode") or "interrupt").strip().lower()


def _skin_hex(key: str, fallback: str) -> str:
    from hermes_cli.skin_engine import get_active_skin
    return get_active_skin().get_color(key, fallback) or fallback


def _fg(hex_color: str, bold: bool = False) -> str:
    try:
        r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        return f"\x1b[{'1;' if bold else ''}38;2;{r};{g};{b}m"
    except (ValueError, IndexError):
        return _BOLD if bold else ""


def _brand_glyph() -> tuple[str, str]:
    """(glyph, name) from the skin's response label, e.g. ``" ☤ Hermes "`` → ("☤", "Hermes")."""
    from hermes_cli.skin_engine import get_active_skin
    label = get_active_skin().get_branding("response_label", " ☤ Hermes ")
    parts = label.split(None, 1)
    return (parts[0], parts[1].strip()) if len(parts) == 2 else ("☤", label.strip() or "Hermes")


def prompt_glyph() -> str:
    from hermes_cli.skin_engine import get_active_skin
    return get_active_skin().get_branding("prompt_symbol", "❯").strip() or "❯"


# ── transcript rows (printed into scrollback) ───────────────────────────────────────────────

# Fixed width, not the terminal's: a 40-cell rule never re-wraps on resize (classic's user rule).
_SEPARATOR_WIDTH = 40


_GAP_PENDING = False  # a user message was just echoed and nothing has opened a block under it yet


def separator(accent: bool) -> str:
    """The bar above each user message and answer (accent) or reasoning block (dim). Every caller
    prints a blank line ahead of it, so it also settles the gap a fresh user message owes."""
    global _GAP_PENDING
    _GAP_PENDING = False
    color = _fg(_skin_hex("ui_accent", "#FFBF00")) if accent else _DIM
    return f"{color}{'─' * _SEPARATOR_WIDTH}{_RST}"


def owe_gap_after_user_message() -> None:
    """Called once the user's message is echoed: the next tool row opens with a blank line, as
    reasoning and answers already do, so the message never runs straight into the tool calls."""
    global _GAP_PENDING
    _GAP_PENDING = True


def _take_gap() -> str:
    global _GAP_PENDING
    gap, _GAP_PENDING = ("\n" if _GAP_PENDING else ""), False
    return gap


def user_echo_markup(text: str, accent_hex: str, ts_suffix: str = "") -> str:
    """Rich markup for a submitted single-line user message: ``❯ text``."""
    from rich.markup import escape
    return f"[bold {accent_hex}]{escape(prompt_glyph())}[/] [bold]{escape(text)}[/]{ts_suffix}"


def answer_label(timestamp: str = "") -> str:
    """The rows that open an answer: the accent bar, then the skin's brand glyph + name."""
    glyph, name = _brand_glyph()
    stamp = f" {_DIM}{timestamp}{_RST}" if timestamp else ""
    return f"\n{separator(True)}\n{_fg(_skin_hex('ui_accent', '#FFBF00'), bold=True)}{glyph} {name}{_RST}{stamp}"


def thought_title(text: str) -> str:
    """Short title for a reasoning block: its first ``**bold**`` heading, else its first sentence."""
    text = (text or "").strip()
    match = re.search(r"\*\*(.+?)\*\*", text)
    title = match.group(1) if match else re.split(r"(?<=[.!?])\s|\n", text, maxsplit=1)[0]
    title = " ".join(title.split()).strip("*# ")
    return title if len(title) <= _TITLE_MAX else title[:_TITLE_MAX - 1].rstrip() + "…"


def thought_line(seconds: float | None, text: str) -> str:
    """``∴ Thought for 3s · Planning the change`` (dim italic), closing a reasoning block."""
    title = thought_title(text)
    took = t("cli.modern.thought_for", seconds=f"{max(seconds or 0.0, 0.0):.0f}") if seconds else t("cli.modern.thought")
    body = f"{took} · {title}" if title else took
    return f"\x1b[2;3m∴ {body}{_RST}"


_REASONING_PREVIEW_LINES = 10  # classic's post-hoc reasoning box shows this many unless /reasoning full


def reasoning_text_line(line: str) -> str:
    """One streamed reasoning line: dim, no frame (the dim bar above opens the block)."""
    return f"{_DIM}{line}{_RST}"


def thought_rows(seconds: float | None, text: str, full: bool) -> list[str]:
    """A reasoning block printed after the fact (no live stream): dim bar, its text (first 10
    lines unless ``full``, as classic), then the ``∴ Thought for 3s`` summary."""
    lines = (text or "").strip().splitlines()
    shown = lines if full or len(lines) <= _REASONING_PREVIEW_LINES else lines[:_REASONING_PREVIEW_LINES]
    rows = [separator(False), *(reasoning_text_line(line) for line in shown)]
    if len(shown) < len(lines):
        rows.append(f"{_DIM}  {t('cli.chat.reasoning_more_lines', count=len(lines) - len(shown))}{_RST}")
    return [*rows, thought_line(seconds, text)]


def completed_tool_line(cli, function_name: str, args: dict | None, duration: float, result=None,
                        is_error: bool = False) -> str:
    """The scrollback line for a finished tool call in the active layout."""
    if is_modern(cli):
        return _take_gap() + tool_row(function_name, args, duration, result, is_error)
    from agent.display import get_cute_tool_message
    return f"  {get_cute_tool_message(function_name, args, duration, result=result)}"


def tool_row(function_name: str, args: dict | None, duration: float, result=None, is_error: bool = False) -> str:
    """``📖 Read calc.py  0.2s`` / ``✗ $ pytest -q  0.4s [exit 1]`` — one row per completed call, led by
    the tool's emoji (the classic ``┊`` row's), or a red ``✗`` when that entry failed."""
    from agent.display import get_tool_row_parts
    rows = get_tool_row_parts(function_name, args or {}, result, is_error=bool(is_error))
    out = []
    for index, (emoji, verb, detail, suffix) in enumerate(rows):
        verb = (verb or function_name).strip()
        mark = f"{_RED}✗{_RST}" if suffix is not None else (emoji or "●")
        line = f"{mark} {_BOLD}{verb[:1].upper()}{verb[1:]}{_RST}"
        if detail:
            line += f" {detail}"
        if index == len(rows) - 1 and duration >= 0.1:
            line += f"  {_DIM}{duration:.1f}s{_RST}"
        if suffix:
            line += f"{_RED}{suffix}{_RST}"
        out.append(line)
    return "\n".join(out)


_ACTIVE = False  # one CLI per process; read by printers that get no CLI handle (gateway inline diffs)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_DIFF_FILE_RE = re.compile(r"^a/(.+?) → b/(.+)$")


def set_active(enabled: bool) -> None:
    global _ACTIVE
    _ACTIVE = bool(enabled)


def _display_path(path: str) -> str:
    path = path[1:] if path.startswith("//") else path
    cwd = os.getenv("TERMINAL_CWD") or os.getcwd()
    try:
        rel = os.path.relpath(path, cwd) if os.path.isabs(path) else path
    except ValueError:
        return path
    return path if rel.startswith("..") else rel


def gutter_print(print_fn):
    """Wrap a line printer so a diff hangs under its tool row behind a dim ``│``: the ``review
    diff`` header is dropped and ``a/<abs> → b/<abs>`` collapses to one cwd-relative path."""
    header = _ANSI_RE.sub("", t("display.diff.review_header")).strip()

    def _print(text):
        for line in str(text).split("\n"):
            plain = _ANSI_RE.sub("", line).strip()
            if plain == header:
                continue
            match = _DIFF_FILE_RE.match(plain)
            if match:
                old, new = _display_path(match.group(1)), _display_path(match.group(2))
                line = f"{_BOLD}{new if old == new else f'{old} → {new}'}{_RST}"
            print_fn(f"{_DIM}│{_RST} {line}" if plain else f"{_DIM}│{_RST}")
    return _print


def diff_printer(print_fn):
    """``print_fn`` with the gutter when the glyph layout is active in this process."""
    return gutter_print(print_fn) if _ACTIVE else print_fn


def turn_footer(summary: str) -> str:
    """``─ 12.4s · read 3 files · edited 1 file +4 -0 · 7:02 AM`` closing each turn."""
    body = summary.lstrip("⋯ ").strip()
    clock = datetime.now().strftime("%-I:%M %p") if os.name != "nt" else datetime.now().strftime("%I:%M %p")
    return f"{_DIM}─ {body} · {clock}{_RST}" if body else f"{_DIM}─ {clock}{_RST}"


# ── live region (prompt_toolkit fragments, redrawn every frame) ─────────────────────────────

def _cwd_label() -> str:
    cwd = os.getenv("TERMINAL_CWD") or os.getcwd()
    home = os.path.expanduser("~")
    short = "~" + cwd[len(home):] if cwd == home or cwd.startswith(home + os.sep) else cwd
    from hermes_cli.status_bar_git import current_git_branch
    branch = current_git_branch(cwd)
    return f"{short} ({branch})" if branch else short


def activity_text(cli) -> str:
    """What the agent is doing: the classic spinner's text (kaomoji face or tool emoji included)."""
    if getattr(cli, "_command_running", False):
        return str(getattr(cli, "_command_status", "") or "")
    text = str(getattr(cli, "_spinner_text", "") or "").strip()
    return text or (t("cli.modern.working") if getattr(cli, "_agent_running", False) else "")


def _turn_elapsed(cli) -> str:
    started = getattr(cli, "_prompt_start_time", None)
    if started is None:
        return ""
    from cli import format_duration_compact
    return format_duration_compact(max(0.0, time.time() - float(started)))


def rule_style() -> str:
    """Composer rules are quiet chrome: the skin's session border, not the accent."""
    return f"fg:{_skin_hex('session_border', '#8B8682')}"


def _no_bg(frags: list) -> list:
    """The footer sits on the terminal background, not the classic status bar's fill."""
    return [(f"{style} bg:default".strip(), text) for style, text in frags]


def _fit(cli, frags: list, width: int) -> list:
    used = sum(cli._status_bar_display_width(text) for _, text in frags)
    if used <= width:
        return frags
    plain = "".join(text for _, text in frags)
    return [(rule_style(), cli._trim_status_bar_text(plain, width))]


def top_rule_fragments(cli) -> list:
    """``── ⠹ Running tests · 14s ─────────── calc helpers ─``: live status left, session title right."""
    width = max(10, cli._get_tui_terminal_width())
    rule, accent = rule_style(), f"fg:{_skin_hex('ui_accent', '#FFBF00')} bold"
    busy = getattr(cli, "_agent_running", False) or getattr(cli, "_command_running", False)
    left: list = []
    if busy:
        frame = _SPIN[int(time.monotonic() * 10) % len(_SPIN)]
        left = [(rule, "── "), (accent, frame + " "), ("", activity_text(cli))]
        # Classic's spinner suffix: turn time and the streamed-token count (``↓ 547 tok``).
        for extra in (_turn_elapsed(cli), cli._spinner_token_flow()):
            if extra:
                left.append(("class:status-bar-dim bg:default", f" · {extra}"))
        left.append((rule, " "))
    title = cli._get_status_bar_session_title()
    right = [(rule, " "), (f"fg:{_skin_hex('ui_label', '#DAA520')}", title), (rule, " ─")] if title else []
    used = sum(cli._status_bar_display_width(text) for _, text in left + right)
    if used >= width:
        right, used = [], sum(cli._status_bar_display_width(text) for _, text in left)
    return _fit(cli, left + [(rule, "─" * max(0, width - used))] + right, width)


def hint_pairs(cli) -> list[tuple[str, str]]:
    """Keys that do something right now, for the footer's right half."""
    if any(getattr(cli, attr, None) for attr in (
            "_approval_state", "_clarify_state", "_sudo_state", "_secret_state", "_connection_state",
            "_slash_confirm_state", "_model_picker_state", "_command_palette_state")):
        return []
    if getattr(cli, "_agent_running", False):
        enter = {"steer": "steer", "queue": "queue"}.get(getattr(cli, "busy_input_mode", ""), "interrupt")
        pairs = [("enter", t(f"cli.modern.hint_{enter}"))]
        if enter != "queue":
            pairs.append(("tab", t("cli.modern.hint_queue")))
        return pairs + [("ctrl+c", t("cli.modern.hint_stop"))]
    return [("/", t("cli.modern.hint_commands")), ("@", t("cli.modern.hint_files")), ("!", t("cli.modern.hint_shell"))]


def status_line_fragments(cli) -> list:
    """Footer under the composer: model · context gauge · cwd (branch) left, live key hints right.
    An explicit ``display.status_bar.fields`` list keeps the user's chosen segments on the left."""
    return _render_safely(_status_line_fragments, cli)


def _render_safely(render, cli) -> list:
    """Boundary for prompt_toolkit render callbacks: a bad frame degrades to a bare line."""
    try:
        return render(cli)
    except Exception:  # health: allow BLE001 -- a render callback must never take down the app frame
        logger.debug("%s failed", render.__name__, exc_info=True)
        return [("", f" {getattr(cli, 'model', '') or 'Hermes'}")]


def _status_line_fragments(cli) -> list:
    """The classic bar's segments (same fields, same width tiers, same ``display.status_bar.fields``
    filter), then cwd (branch) and the live key hints when there is room."""
    width = max(10, cli._get_tui_terminal_width())
    dim, strong = "class:status-bar-dim", "class:status-bar-strong"
    snapshot = cli._get_status_bar_snapshot()
    field_set = cli._get_status_bar_field_set()

    def _on(name):
        return field_set is None or name in field_set

    left: list = []
    for seg in cli._status_bar_segments(snapshot, width, field_set, cli._is_session_yolo_active(), styled=True):
        if left:
            left.append((dim, " · "))
        left.extend(seg)
    stash = cli._prompt_stash.indicator() if _on("stash") and hasattr(cli, "_prompt_stash") else ""
    if stash:  # a parked draft is never silently forgotten
        left += [(dim, " · "), (strong, stash)]
    battery = snapshot.get("battery_label") or ""  # only set while /battery is on
    if battery and _on("battery"):
        left[0:0] = [("", " "), (cli._battery_status_style(snapshot.get("battery_category", "dim")), battery),
                     (dim, " ·")]
    vim = cli._vim_mode_label()
    if vim:
        left += [(dim, " · "), (strong, vim)]
    right: list = []
    for key, text in hint_pairs(cli) if width >= 76 else []:
        if right:
            right.append((dim, " · "))
        right += [(strong, key), (dim, f" {text}")]
    used_left = sum(cli._status_bar_display_width(text) for _, text in left)
    used_right = sum(cli._status_bar_display_width(text) for _, text in right)
    room = width - used_left - used_right - 6
    if field_set is None and room >= 12:  # cwd takes whatever the hints leave, trimmed from the left
        cwd = _cwd_label()
        cwd = cwd if len(cwd) <= room else "…" + cwd[-(room - 1):]
        left += [(dim, " · "), (dim, cwd)]
        used_left += 3 + cli._status_bar_display_width(cwd)
    if right and used_left + used_right + 2 > width:
        right, used_right = [], 0
    pad = [("", " " * max(1, width - used_left - used_right - 1))] if right else []
    return _no_bg(_fit(cli, left + pad + right + ([("", " ")] if right else []), width))


def modern_input_rules(cli, kb):
    """The composer's rules for this layout (the top one carries live status), plus Tab-to-queue."""
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.layout import FormattedTextControl, Window

    top = Window(FormattedTextControl(lambda: _render_safely(top_rule_fragments, cli)),
                 height=lambda: cli._tui_input_rule_height("top"), wrap_lines=False)
    bottom = Window(char="─", height=lambda: cli._tui_input_rule_height("bottom"), style=rule_style)

    def _can_queue() -> bool:
        from prompt_toolkit.application import get_app
        from cli import _looks_like_slash_command
        from hermes_cli.cli_subagent_monitor import modal_prompt_active
        buf = get_app().current_buffer
        text = buf.text.strip()
        return bool(getattr(cli, "_agent_running", False) and text and buf.complete_state is None
                    and not _looks_like_slash_command(text) and not text.startswith("!")
                    and not modal_prompt_active(cli))

    @kb.add("tab", filter=Condition(_can_queue), eager=True)
    def _queue_while_busy(event):
        # Not _tui_handle_enter: its paste/continuation rules would turn this key into a newline.
        buf = event.app.current_buffer
        text, images = buf.text.strip(), list(cli._attached_images)
        cli._attached_images.clear()
        cli._tui_enter_while_busy(text, images, (text, images) if images else text, mode="queue")
        cli._inline_pastes(buf)
        buf.reset(append_to_history=True)
        event.app.invalidate()

    return top, bottom
