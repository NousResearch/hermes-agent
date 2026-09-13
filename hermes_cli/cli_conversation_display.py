"""Pure readable display formatting plus the single CLI display-event sink."""
from collections.abc import Callable
from io import StringIO

from rich.console import Console
from rich.text import Text


def _console(width, *, full_width=False):
    # This renderer feeds prompt_toolkit's TUI scrollback. Keep its skin colors even when the
    # parent shell exports NO_COLOR/TERM=dumb; the TUI itself is intentionally colorized.
    return Console(file=StringIO(), width=max(1, width if full_width else min(width, 88)), height=25,
                   force_terminal=True, color_system='truecolor', no_color=False)


def render_user_preview(text, width, *, first=2, last=2, timestamp=''):
    from hermes_cli.skin_engine import get_active_skin
    skin = get_active_skin()
    band_bg = skin.get_color('status_bar_bg', '#1a1a2e')
    band_fg = skin.get_color('status_bar_text', '#C0C0C0')
    console = _console(width, full_width=True)
    # Wrap before applying the existing preview budget: one long pasted paragraph
    # must not bypass it. Only the displayed preview is shortened, never model input.
    rows = list(Text(text).wrap(console, max(1, console.width - 4)))
    first, last = max(1, first), max(0, last)
    if len(rows) > first + last:
        hidden = len(rows) - first - last
        rows = rows[:first] + [Text(f'… (+{hidden} more lines)',
                                  style=skin.get_color('banner_dim', '#8B949E'))] + (rows[-last:] if last else [])
    if timestamp:
        rows.append(Text(timestamp, style=skin.get_color('banner_dim', '#8B949E')))
    first_row = True
    for row in rows:
        # Generated omission/timestamp rows also need wrapping before background fill.
        for visual_row in row.wrap(console, max(1, console.width - 4)):
            prefix = ' ● ' if first_row else '   '
            first_row = False
            band = Text(prefix[:max(0, console.width - 1)], style=f'{band_fg} on {band_bg}')
            band.append_text(visual_row)
            band.pad_right(max(0, console.width - band.cell_len))
            console.print(band, overflow='crop')
    return console.file.getvalue().rstrip('\n') + '\n'


def render_notice(label, text, width):
    from hermes_cli.skin_engine import get_active_skin
    color = get_active_skin().get_color('banner_dim', '#8B949E')
    console = _console(width)
    console.print(Text('  ' + label, style='bold ' + color))
    # Preserve all action details, with natural wrapping instead of one dense banner.
    for detail in text.split('\n'):
        for row in Text(detail, style=color).wrap(console, max(1, console.width - 4)):
            console.print(Text('    ', style=color) + row)
    console.print()
    return console.file.getvalue().rstrip('\n') + '\n'


def render_review_notice(text, width):
    return render_notice('Self-improvement review', text.replace(' · ', '\n'), width)


def emit_display_event(render: Callable[[int], str]) -> None:
    """Render one display event, record its source for resize replay, and emit it atomically."""
    from cli import _cprint, _record_output_history_entry, _suspend_output_history

    def lines():
        from cli import _terminal_columns
        return render(_terminal_columns()).split('\n')

    _record_output_history_entry(lines)
    with _suspend_output_history():
        _cprint('\n'.join(lines()))


def print_notification(cli, label, details):
    """Render routine acknowledgments; callers retain their legacy fallback."""
    if getattr(cli, 'final_response_markdown', 'strip') != 'render':
        return False
    def emit():
        emit_display_event(lambda width: render_notice(label, details, width))
    stream = getattr(cli, '_markdown_stream', None)
    if stream is not None:
        stream.feed('', final=True, after=emit)
    else:
        emit()
    return True


def print_reflowing(render):
    """Compatibility name for callers outside the display-event path."""
    emit_display_event(render)


def print_review_notice(text):
    prefix = '💾 Self-improvement review: '
    if not text.strip().startswith(prefix):
        return False
    details = text.strip()[len(prefix):]
    emit_display_event(lambda width: render_review_notice(details, width))
    return True
