"""Root container of the classic CLI footer: clips a late row instead of giving up.

prompt_toolkit renders a non-full-screen layout into at least the height it measured a moment
earlier (``Renderer.render`` → ``HSplit.preferred_height``), then ``HSplit._divide_heights``
measures every child again while painting. Each footer row here has a callable height that the
agent thread flips mid-turn (``_agent_running`` → spacer, ``_spinner_text`` → spinner,
``_command_running`` → hint), so a row appearing between the two passes makes the summed minimum
exceed the budget by one and stock ``HSplit`` replaces the WHOLE footer with its
" Window too small... " placeholder (#57393) — on any terminal size, including a maximised 200×50
pane. Hermes's throttled ``_invalidate`` can then drop the healing repaint for the length of a
tool call, so the placeholder sits there looking like a geometry error.

The CLI builds it bottom-aligned (``footer_root``). prompt_toolkit does not shrink a non-full-screen
app from one paint to the next: ``Renderer.render`` paints into ``max(rows below the cursor, the last
frame's height, the preferred height)``, and only a reset of the renderer drops the last frame (every
print resets it, and so do a resize, Ctrl+L, ``/redraw`` and ``/clear``). So after a panel closes the
root is handed more rows than its children ask for. ``HSplit``'s default ``JUSTIFY`` laid the children
out from the top and left those rows blank at the bottom: a closed clarify or approval panel pulled the
composer up by its height, and a closed completion menu (below the composer) left it as high as the
open menu had pushed it. Bottom-aligned, the spare rows pad the space above the composer, which stays
on, or returns to, the terminal's last rows.
"""
from __future__ import annotations

from collections.abc import Sequence

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import AnyContainer, HSplit, VerticalAlign


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


def footer_root(children: Sequence[AnyContainer]) -> FooterSplit:
    """The CLI layout's root: a ``FooterSplit`` whose spare rows go above the composer."""
    return FooterSplit(children, align=VerticalAlign.BOTTOM)
