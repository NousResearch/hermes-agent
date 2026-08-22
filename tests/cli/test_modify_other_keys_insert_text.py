"""Regression tests for the character-valued half of
``install_modify_other_keys_aliases()`` — Shift+letter under
modifyOtherKeys level 2 must *type a capital*, not leak its escape sequence.

Mapping ``ESC[27;2;72~`` → ``"H"`` in ``ANSI_SEQUENCES`` fixes what the key
*is*, but prompt_toolkit's parser reports ``KeyPress(key="H",
data="\\x1b[27;2;72~")`` and the default ``Keys.Any`` binding inserts
``event.data``. The raw sequence therefore still reached the prompt buffer.

Key-level assertions cannot catch that — ``_parse()`` compares ``KeyPress.key``,
which was already correct. These tests drive a real ``PromptSession`` over a
pipe input and assert on the *submitted text*, which is what users see.
"""

from __future__ import annotations

import pytest

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import PromptSession

from hermes_cli.pt_input_extras import (
    install_ctrl_enter_alias,
    install_modify_other_keys_aliases,
    install_shift_enter_alias,
)


def _parser_class():
    """Resolve ``Vt100Parser`` per call, the way the installer does.

    `tests/cli/test_bracketed_paste_timeout.py` reloads
    ``prompt_toolkit.input.vt100_parser``, which rebinds the module's class
    to a new object. A module-level import here would hold the pre-reload
    class and patch something the parser no longer uses.
    """
    from prompt_toolkit.input.vt100_parser import Vt100Parser
    return Vt100Parser


@pytest.fixture(autouse=True)
def _ensure_alias_installed():
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES as _seq
    saved = dict(_seq)
    # The parser patch is process-global; restore it too so it cannot leak
    # into sibling test files the way the table mappings must not.
    parser_cls = _parser_class()
    saved_call_handler = parser_cls._call_handler
    # Same order as cli.py — also proves the Enter aliases are not clobbered.
    install_shift_enter_alias()
    install_ctrl_enter_alias()
    install_modify_other_keys_aliases()
    yield
    _seq.clear()
    _seq.update(saved)
    parser_cls._call_handler = saved_call_handler
    from prompt_toolkit.input.vt100_parser import _IS_PREFIX_OF_LONGER_MATCH_CACHE
    _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()


def _type(byte_seq: str) -> str:
    """Feed raw terminal bytes to a real prompt and return what was submitted."""
    with create_pipe_input() as inp:
        inp.send_text(byte_seq + "\r")
        return PromptSession(input=inp, output=DummyOutput()).prompt()


def _mok(codepoint: int, modifier: int = 2) -> str:
    return f"\x1b[27;{modifier};{codepoint}~"


def _csiu(codepoint: int, modifier: int = 2) -> str:
    return f"\x1b[{codepoint};{modifier}u"


def test_shift_letter_types_a_capital():
    assert _type(_mok(ord("h")) + "ello") == "Hello"


def test_shift_letter_shifted_codepoint_form_types_a_capital():
    """Terminals that report the already-shifted codepoint (72 = 'H')."""
    assert _type(_mok(ord("H")) + "ello") == "Hello"


def test_shift_letter_csi_u_form_types_a_capital():
    assert _type(_csiu(ord("h")) + "ello") == "Hello"


def test_mixed_sentence_round_trips():
    typed = _mok(ord("h")) + "ermes " + _mok(ord("c")) + "LI"
    assert _type(typed) == "Hermes CLI"


def test_shift_space_types_a_space():
    assert _type("a" + _mok(32) + "b") == "a b"


@pytest.mark.parametrize("letter", ["a", "m", "z"])
def test_every_letter_inserts_its_own_character(letter):
    assert _type(_mok(ord(letter))) == letter.upper()


def test_key_press_data_matches_the_character():
    """The mechanism, asserted directly: data drives insertion, so it must
    be the character rather than the bytes that produced it."""
    presses = []
    parser = _parser_class()(presses.append)
    for ch in _mok(ord("q")):
        parser.feed(ch)
    parser.flush()
    assert len(presses) == 1
    assert presses[0].key == "Q"
    assert presses[0].data == "Q"


def test_ctrl_combo_bindings_still_fire():
    """Keys-valued entries are untouched: Ctrl+A moves to line start."""
    assert _type("bc" + _mok(ord("a"), modifier=5) + "X") == "Xbc"


def test_named_keys_keep_their_raw_data():
    presses = []
    parser = _parser_class()(presses.append)
    for ch in "\x1b[A":
        parser.feed(ch)
    parser.flush()
    assert [(kp.key, kp.data) for kp in presses] == [(Keys.Up, "\x1b[A")]


def test_shift_enter_alias_is_not_clobbered():
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    assert ANSI_SEQUENCES["\x1b[27;2;13~"] == (Keys.Escape, Keys.ControlM)


def test_plain_typing_is_unaffected():
    assert _type("Hi there!") == "Hi there!"
    assert _type("café ☕") == "café ☕"


def test_bracketed_paste_is_unaffected():
    assert _type("\x1b[200~Pasted TEXT\x1b[201~") == "Pasted TEXT"


def test_missing_private_method_degrades_to_a_no_op(monkeypatch):
    """`_call_handler` is prompt_toolkit-private. If a future release renames
    it, the install must return False rather than raise — an exception here
    would propagate into cli.py's blanket handler and silently skip the
    installers that run after it."""
    from hermes_cli import pt_input_extras

    monkeypatch.delattr(_parser_class(), "_call_handler", raising=False)
    assert pt_input_extras._install_literal_key_data_patch() is False
    # The caller keeps working and still reports its table registrations.
    assert install_modify_other_keys_aliases() >= 0


def test_patch_is_idempotent_and_does_not_stack():
    from hermes_cli import pt_input_extras

    parser_cls = _parser_class()
    first = parser_cls._call_handler
    assert pt_input_extras._install_literal_key_data_patch() is False
    assert parser_cls._call_handler is first


def test_marker_cannot_outlive_the_wrapper(monkeypatch):
    """If something else replaces `_call_handler`, the marker goes with it,
    so the next install wraps the replacement instead of skipping."""
    from hermes_cli import pt_input_extras

    calls = []

    def _foreign_call_handler(self, key, insert_text):
        calls.append((key, insert_text))

    parser_cls = _parser_class()
    monkeypatch.setattr(parser_cls, "_call_handler", _foreign_call_handler)
    assert pt_input_extras._install_literal_key_data_patch() is True
    assert parser_cls._call_handler is not _foreign_call_handler

    # The replacement is wrapped, not bypassed, and still sees fixed data.
    parser = parser_cls(lambda kp: None)
    for ch in _mok(ord("q")):
        parser.feed(ch)
    parser.flush()
    assert ("Q", "Q") in calls
