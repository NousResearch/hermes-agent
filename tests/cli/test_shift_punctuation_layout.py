"""Shift+punctuation under modifyOtherKeys and CSI-u (#93633)."""

from __future__ import annotations

import pytest
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

from hermes_cli import pt_input_extras as extras


@pytest.fixture(autouse=True)
def _aliases_installed():
    extras.install_modify_other_keys_aliases()


@pytest.mark.parametrize("char", "@^_{}|~")
def test_shifted_codepoint_form_resolves_in_both_spellings(char):
    """The identity half is layout-independent, so it must always install (#93633)."""
    assert ANSI_SEQUENCES.get(f"\x1b[27;2;{ord(char)}~") == char
    assert ANSI_SEQUENCES.get(f"\x1b[{ord(char)};2u") == char


def test_base_half_follows_the_layout_and_never_falls_back_to_us():
    """German Shift+2 is '"', so a US table would type the wrong character, not leak."""
    assert (extras._derive_shift_punctuation("us") or {}).get(ord("2")) == "@"
    assert (extras._derive_shift_punctuation("de") or {}).get(ord("2")) == '"'
    # us-acentos matches a `us-` prefix and is not US-compatible: ' and " are dead keys.
    for unreadable in ("ru", "us-acentos", "us(dvorak)", "definitely-not-a-layout"):
        assert not extras._derive_shift_punctuation(unreadable), unreadable
