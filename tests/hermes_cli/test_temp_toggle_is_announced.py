"""Entering and leaving a temporary chat must both be announced.

A silent toggle is the dangerous failure here: the user believes they are in a
temporary chat when they are not (and keeps typing secrets into a saved
session), or believes the temporary chat ended when it did not. Both replies
are load-bearing, so assert the CLI prints them.
"""

import io
import re
from pathlib import Path

import pytest


SRC = Path("hermes_cli/cli_commands_mixin.py")


@pytest.fixture(scope="module")
def handler_src():
    text = io.open(SRC, encoding="utf-8").read()
    start = text.index("def _handle_temp_command")
    end = text.index("\n    def ", start + 10)
    return text[start:end]


def test_entering_a_temporary_chat_is_announced(handler_src):
    enter = handler_src[handler_src.index("# Enter ephemeral mode"):]
    assert re.search(r"_cprint\(.*[Tt]emporary chat", enter), (
        "entering a temporary chat prints no confirmation -- the user cannot "
        "tell whether the mode is on"
    )


def test_leaving_a_temporary_chat_is_announced(handler_src):
    leave = handler_src[: handler_src.index("# Enter ephemeral mode")]
    assert "Temporary chat ended." in leave
    # The stronger promise: say the transcript is gone, not merely that the
    # mode is off. Users ask "did that get saved?", not "what mode am I in?".
    assert "cannot be recovered" in leave


def test_both_no_op_toggles_explain_themselves(handler_src):
    assert "Not in a temporary chat" in handler_src
    assert "Already in a temporary chat" in handler_src


def test_temp_is_discoverable_from_the_tips_rotation():
    """/reset and the CLI banner surface a random tip; /temp must be in it.

    A privacy feature nobody discovers protects nobody, and /temp is not
    something users guess at.
    """
    from hermes_cli.tips import TIPS

    temp_tips = [x for x in TIPS if x.lstrip().startswith("/temp")]
    assert temp_tips, "no tip mentions /temp"
    tip = temp_tips[0]
    assert "/temp off" in tip, "the tip must show how to leave, not only enter"
