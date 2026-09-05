"""Modified kitty keypad keys must mirror the keys they stand in for (#90640)."""

from __future__ import annotations

import pytest
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

from hermes_cli import pt_input_extras


@pytest.fixture(autouse=True)
def _aliases_installed():
    pt_input_extras.install_shift_enter_alias()
    pt_input_extras.install_ctrl_enter_alias()
    pt_input_extras.install_modify_other_keys_aliases()


# kitty PUA codepoint -> the sequence its non-keypad twin already resolves through.
TWINS = {
    57417: "\x1b[1;{mod}D",   # KP_Left   -> Left
    57418: "\x1b[1;{mod}C",   # KP_Right  -> Right
    57419: "\x1b[1;{mod}A",   # KP_Up     -> Up
    57420: "\x1b[1;{mod}B",   # KP_Down   -> Down
    57421: "\x1b[5;{mod}~",   # KP_PageUp -> PageUp
    57426: "\x1b[3;{mod}~",   # KP_Delete -> Delete
    57414: "\x1b[13;{mod}u",  # KP_Enter  -> Enter
}


@pytest.mark.parametrize("codepoint,template", sorted(TWINS.items()))
@pytest.mark.parametrize("modifier", [2, 3, 5, 6])
def test_modified_keypad_mirrors_its_non_keypad_twin(codepoint, template, modifier):
    """A modified keypad key must resolve to exactly what its twin resolves to (#90640)."""
    twin = ANSI_SEQUENCES.get(template.format(mod=modifier))
    if twin is None:
        pytest.skip("the non-keypad equivalent is itself unmapped")
    assert ANSI_SEQUENCES.get(f"\x1b[{codepoint};{modifier}u") == twin


@pytest.mark.parametrize("modifier", [2, 66, 130, 194])
def test_lock_twins_are_registered_for_csi_u_only(modifier):
    """kitty ORs CapsLock/NumLock into the modifier; modifyOtherKeys never does."""
    assert ANSI_SEQUENCES.get(f"\x1b[57417;{modifier}u") is not None
    assert ANSI_SEQUENCES.get(f"\x1b[27;{modifier};57417~") is None
