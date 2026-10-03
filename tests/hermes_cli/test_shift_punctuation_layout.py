"""Shift+punctuation under modifyOtherKeys and CSI-u (#93633)."""

from __future__ import annotations

import pytest
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

from hermes_cli import pt_input_extras as extras


@pytest.fixture(autouse=True)
def _aliases_installed():
    extras.install_modify_other_keys_aliases()


@pytest.mark.parametrize("char", "@^_{}|~")
def test_shifted_codepoint_resolves_in_the_tilde_form_only(char):
    """The produced codepoint is layout-independent only where the terminal reports it.

    In the modifyOtherKeys tilde form the codepoint IS the produced character. In CSI-u it is
    the unshifted key, so that spelling answers to the layout's base half and nothing else.
    """
    assert ANSI_SEQUENCES.get(f"\x1b[27;2;{ord(char)}~") == char
    base = extras._shift_punctuation_base_map() or {}
    assert ANSI_SEQUENCES.get(f"\x1b[{ord(char)};2u") == base.get(ord(char))


def test_base_half_follows_the_layout_and_never_falls_back_to_us():
    """German Shift+2 is '"', so a US table would type the wrong character, not leak."""
    assert (extras._derive_shift_punctuation("us") or {}).get(ord("2")) == "@"
    assert (extras._derive_shift_punctuation("de") or {}).get(ord("2")) == '"'
    # us-acentos matches a `us-` prefix and is not US-compatible: ' and " are dead keys.
    for unreadable in ("ru", "us-acentos", "us(dvorak)", "definitely-not-a-layout"):
        assert not extras._derive_shift_punctuation(unreadable), unreadable
