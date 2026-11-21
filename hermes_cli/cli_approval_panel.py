"""Row layout of the classic CLI approval panel (moved from ``cli_tui_mixin``).

One computation serves both the drawing (``CLITuiMixin._get_approval_display_fragments``) and the
callback's decision to print a schedule card's job to scrollback first
(``CLIModalMixin._approval_callback``): the panel says "full job printed above" exactly when it
cuts the body, so both must read the same row budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.i18n import t
from hermes_cli.cli_approval_kinds import approval_card
from hermes_cli.cli_tui_wrap import _num_prefix, _wrap_rows


@dataclass(frozen=True)
class ApprovalPanelLayout:
    title: str
    box_width: int
    body_rows: list[str]
    choice_rows: list[tuple[int, str]]
    description_rows: list[str]
    compact: bool
    body_cut: bool  # the body did not fit and ends in the card's truncated notice


def approval_panel_layout(state: dict, available: int) -> ApprovalPanelLayout:
    """The panel for ``state`` in ``available`` terminal rows. Title + body + choices always render;
    the description sits last and takes only the rows left over."""
    from cli import _panel_box_width, _wrap_panel_text_keep_ws
    wrap = _wrap_panel_text_keep_ws
    card = approval_card(state.get("kind"))
    command, description = card.subject(state["command"], state["description"])
    choices = state["choices"]
    selected = state.get("selected", 0)
    title = t(card.title_key)

    preview_lines = wrap(description, 60)
    preview_lines.extend(wrap(command, 60))
    for i, choice in enumerate(choices):
        prefix = '❯ ' if i == selected else '  '
        preview_lines.extend(wrap(f"{prefix}{card.choice_label(choice)}", 60, subsequent_indent="  "))
    box_width = _panel_box_width(title, preview_lines)
    inner_text_width = max(8, box_width - 2)

    cmd_wrapped = wrap(command, inner_text_width)
    if not state.get("show_full", False) and "view" in choices and len(cmd_wrapped) > 4:
        cmd_wrapped = cmd_wrapped[:3] + wrap(
            t("cli.tui.approval_choose_full", label=t("cli.tui.approval_view")), inner_text_width)
    choice_labels = [
        f"{'❯' if i == selected else ' '} {_num_prefix(i)}. {card.choice_label(choice)}"
        for i, choice in enumerate(choices)]
    choice_wrapped = _wrap_rows(wrap, choice_labels, inner_text_width, "    ")

    # Full chrome = top border + title + blank + blank-between-cmd/choices + bottom border (5); when
    # that doesn't fit, drop the separator blanks (3) so every choice stays on-screen.
    compact = 5 + len(cmd_wrapped) + len(choice_wrapped) > available
    chrome_rows = 3 if compact else 5

    # A body too long to leave room for the choices is cut so approve/deny still render (>= 1 body row).
    max_cmd_rows = max(1, available - chrome_rows - len(choice_wrapped))
    body_cut = len(cmd_wrapped) > max_cmd_rows
    if body_cut:
        keep = max(1, max_cmd_rows - 1) if max_cmd_rows > 1 else 1
        cmd_wrapped = cmd_wrapped[:keep] + wrap(t(card.truncated_key), inner_text_width)

    # Remaining rows go to the description (minus the blank separator in full mode), capped at 10.
    mandatory_no_desc = chrome_rows + len(cmd_wrapped) + len(choice_wrapped)
    available_for_desc = max(0, min(available - mandatory_no_desc - (0 if compact else 1), 10))
    desc_wrapped = wrap(description, inner_text_width) if description else []
    if available_for_desc < 1 or not desc_wrapped:
        desc_wrapped = []
    elif len(desc_wrapped) > available_for_desc:
        desc_wrapped = desc_wrapped[:max(1, available_for_desc - 1)] + [t("cli.tui.approval_description_truncated")]
    return ApprovalPanelLayout(title, box_width, cmd_wrapped, choice_wrapped, desc_wrapped, compact, body_cut)
