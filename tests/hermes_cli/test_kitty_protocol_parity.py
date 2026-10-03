"""Regression tests for Shift+punctuation parity under the kitty protocol.

The kitty keyboard protocol differs from xterm modifyOtherKeys in two ways
that repeatedly produce kitty-only defects, because kitty removed
modifyOtherKeys support entirely and therefore has no second protocol to fall
back on:

* it reports the **unshifted** codepoint plus a Shift modifier, where
  modifyOtherKeys emitters report the already-shifted one; and
* it encodes keys that have no legacy form — the keypad, F13+, lock keys — as
  **Private Use Area** codepoints, which modifyOtherKeys never sends.

Each test below pins behaviour that was broken because of one of those two
properties.
"""

from __future__ import annotations

import pytest

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

from hermes_cli import pt_input_extras as extras


@pytest.fixture(autouse=True)
def _isolated_sequence_table():
    """Install the aliases per-test and restore the table afterwards, so the
    hundreds of registrations do not leak into sibling test files."""
    saved = dict(ANSI_SEQUENCES)
    extras.install_shift_enter_alias()
    extras.install_ctrl_enter_alias()
    extras.install_cmd_backspace_alias()
    extras.install_modify_other_keys_aliases()
    extras.install_ignored_terminal_sequences()
    yield
    ANSI_SEQUENCES.clear()
    ANSI_SEQUENCES.update(saved)
    from prompt_toolkit.input.vt100_parser import _IS_PREFIX_OF_LONGER_MATCH_CACHE

    _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()


# ---------------------------------------------------------------------------
# Shift+punctuation: the identity half is layout-independent, the base half
# is not, and only kitty reaches the base half.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char", "@^_{}|~")
def test_shifted_codepoint_form_always_resolves(char):
    """The identity half must install on every layout and every terminal.

    These seven leaked as literal ``[27;2;64~`` text before the fix; they are
    the codepoints outside prompt_toolkit's own 32..63 coverage.
    """
    assert ANSI_SEQUENCES.get(f"\x1b[27;2;{ord(char)}~") == char
    assert ANSI_SEQUENCES.get(f"\x1b[{ord(char)};2u") == char


@pytest.mark.parametrize(
    "layout,safe_by_name",
    [
        ("us", True),
        ("us.utf-8", True),
        ("us,gr", True),  # comma list — us is primary
        ("us-acentos", False),  # ' and " become dead accent keys
        ("us-intl", False),
        ("gr,us", False),  # Greek primary means Greek punctuation
    ],
)
def test_us_layout_name_check(monkeypatch, layout, safe_by_name):
    """Only bare ``us`` (plus a console encoding suffix) is safe by NAME.

    ``us-acentos`` matched an earlier ``us-`` prefix check and was wrongly
    accepted; it turns ' and " into dead accent keys, so Shift+' is not '"'.
    Names that are not provably US must fall through to the xkb table.
    """
    monkeypatch.setenv("XKB_DEFAULT_LAYOUT", layout)
    primary = extras._configured_layout()
    accepted_by_name = primary == "us" or primary.startswith("us.")
    assert accepted_by_name is safe_by_name


def test_layout_variant_is_split_out(monkeypatch):
    """``us(dvorak)`` must resolve to layout ``us`` + variant ``dvorak`` so the
    derived map reads the block the user actually types on."""
    monkeypatch.setenv("XKB_DEFAULT_LAYOUT", "us(dvorak)")
    assert extras._configured_layout_and_variant() == ("us", "dvorak")


def test_layout_name_cannot_escape_the_symbols_directory(monkeypatch):
    """A hostile layout name must never be used to open an arbitrary file."""
    assert extras._derive_shift_punctuation("../../etc/passwd") is None
    assert extras._layout_keeps_us_shift_punctuation("../../etc/passwd") is False


def test_derived_map_follows_the_layout_not_us():
    """AZERTY must derive its own answer, not the US table.

    On ``fr`` the AE01 key is ``[ampersand, 1]``, so Shift on the ``&`` key
    yields ``1`` — mapping it to ``!`` the way a US table would is the "wrong
    character typed" failure the whole gate exists to prevent.
    """
    fr = extras._derive_shift_punctuation("fr")
    if fr is None:  # no xkb data installed on this host
        pytest.skip("xkb symbol tables unavailable")
    assert fr.get(ord("&")) == "1"
    assert fr.get(ord("1")) != "!"


def test_derived_map_omits_dead_keys_rather_than_guessing():
    """Cyrillic derives nothing: leaking is correct, inventing is not."""
    ru = extras._derive_shift_punctuation("ru")
    assert not ru  # None or empty — never a US fallback


def test_derived_map_contains_no_letter_pairs():
    """Dvorak puts letters on punctuation positions; the letter table already
    covers those, so they must not appear in the punctuation map."""
    for layout, variant in (("us", "basic"), ("us", "dvorak")):
        derived = extras._derive_shift_punctuation(layout, variant)
        if derived is None:
            pytest.skip("xkb symbol tables unavailable")
        assert not [c for c in derived if chr(c).isalpha()]


