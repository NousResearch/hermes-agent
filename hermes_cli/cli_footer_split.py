"""Root container of the classic CLI footer: clips a late row instead of giving up.

prompt_toolkit renders a non-full-screen layout into exactly the height it measured a moment
earlier (``Renderer.render`` → ``HSplit.preferred_height``), then ``HSplit._divide_heights``
measures every child again while painting. Each footer row here has a callable height that the
agent thread flips mid-turn (``_agent_running`` → spacer, ``_spinner_text`` → spinner,
``_command_running`` → hint), so a row appearing between the two passes makes the summed minimum
exceed the budget by one and stock ``HSplit`` replaces the WHOLE footer with its
" Window too small... " placeholder (#57393) — on any terminal size, including a maximised 200×50
pane. Hermes's throttled ``_invalidate`` can then drop the healing repaint for the length of a
tool call, so the placeholder sits there looking like a geometry error.
"""
from __future__ import annotations

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit


class FooterSplit(HSplit):
    """``HSplit`` whose overflow path keeps the composer on screen and repaints right away.

    When the base class would paint the placeholder, allocate the children's minimum heights
    bottom-up (composer, rules and status bar first) so only the row that appeared late — always
    above them — loses its row for this one frame, and schedule an unthrottled redraw so the
    next frame measures and paints against the same state.
    """

    def _divide_heights(self, write_position) -> list[int] | None:
        sizes = super()._divide_heights(write_position)
        if sizes is not None:
            return sizes
        budget = write_position.height
        clipped: list[int] = []
        for child in reversed(self._all_children):
            rows = min(child.preferred_height(write_position.width, budget).min, budget)
            budget -= rows
            clipped.append(rows)
        get_app().invalidate()
        return clipped[::-1]
