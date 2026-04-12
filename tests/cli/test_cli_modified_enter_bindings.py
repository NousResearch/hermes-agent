from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.keys import Keys

from cli import _MODIFIED_ENTER_NEWLINE_SEQUENCES, _configure_modified_enter_bindings


def test_modified_enter_sequences_are_configured_for_newlines():
    original = {sequence: ANSI_SEQUENCES.get(sequence) for sequence in _MODIFIED_ENTER_NEWLINE_SEQUENCES}
    try:
        for sequence in _MODIFIED_ENTER_NEWLINE_SEQUENCES:
            ANSI_SEQUENCES.pop(sequence, None)

        _configure_modified_enter_bindings()

        assert ANSI_SEQUENCES["\x1b[27;2;13~"] == (Keys.Escape, Keys.ControlM)
        assert ANSI_SEQUENCES["\x1b[27;5;13~"] == Keys.ControlJ
        assert ANSI_SEQUENCES["\x1b[27;6;13~"] == Keys.ControlJ
    finally:
        for sequence, binding in original.items():
            if binding is None:
                ANSI_SEQUENCES.pop(sequence, None)
            else:
                ANSI_SEQUENCES[sequence] = binding
