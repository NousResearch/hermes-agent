"""Batch clarify panel stays inside the viewport so choices remain selectable.

The panel is an unsized Window. HSplit clips the tail, and the active
question's choices are that tail. A short terminal must still contain every
choice; a tall one must keep the full question text.
"""

import shutil
from unittest.mock import patch

from cli import HermesCLI


def _make_cli(questions, answers, choices, active):
    cli = HermesCLI.__new__(HermesCLI)
    cli._clarify_freetext = False
    cli._clarify_state = {
        "questions": [
            {"qid": f"q{i}", "question": question} for i, question in enumerate(questions)
        ],
        "answers": answers,
        "answer_meta": {},
        "active": active,
        "choices": list(choices),
        "selected": 0,
        "multi_select": False,
    }
    return cli


def _rendered(cli, columns, lines):
    size = shutil.os.terminal_size((columns, lines))
    with patch("hermes_cli.cli_tui_mixin.shutil.get_terminal_size", return_value=size):
        fragments = cli._get_clarify_display_fragments()
    return "".join(text for _style, text in fragments)


_QUESTION = ("Storage boundary for the signed graph. " * 12).strip()


class TestClarifyBatchPanelHeight:
    def test_short_terminal_keeps_active_choices_inside_the_viewport(self):
        cli = _make_cli(
            [_QUESTION, _QUESTION, _QUESTION + " ACTIVE_QUESTION_TAIL"],
            {"q0": "locked answer " * 20, "q1": "locked answer " * 20},
            ["CHOICE_ALPHA_TOKEN keep the graph", "CHOICE_BETA_TOKEN replace it"],
            active=2,
        )
        lines = 24
        rendered = _rendered(cli, 100, lines)
        available = lines - 6  # _PANEL_RESERVED_BELOW
        assert rendered.count("\n") <= available
        assert "CHOICE_ALPHA_TOKEN" in rendered
        assert "CHOICE_BETA_TOKEN" in rendered

    def test_tall_terminal_keeps_the_full_active_question(self):
        cli = _make_cli(
            [_QUESTION, _QUESTION, _QUESTION + " ACTIVE_QUESTION_TAIL"],
            {"q0": "short", "q1": "short"},
            ["CHOICE_ALPHA_TOKEN keep the graph"],
            active=2,
        )
        rendered = _rendered(cli, 100, 80)
        assert "ACTIVE_QUESTION_TAIL" in rendered
        assert "CHOICE_ALPHA_TOKEN" in rendered
