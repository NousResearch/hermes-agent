from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.keys import Keys

from cli import _is_shift_enter_sequence, _maybe_insert_newline_for_modified_enter


class _FakeBuffer:
    def __init__(self):
        self.inserted = []

    def insert_text(self, text):
        self.inserted.append(text)


class _FakeKeyPress:
    def __init__(self, data):
        self.data = data


class _FakeEvent:
    def __init__(self, data):
        self.current_buffer = _FakeBuffer()
        self.key_sequence = [_FakeKeyPress(data)] if data is not None else []


def test_shift_enter_sequence_detected_for_ghostty_xterm_mode():
    assert _is_shift_enter_sequence("\x1b[27;2;13~") is True


def test_shift_enter_sequence_detected_for_csi_u_mode():
    assert _is_shift_enter_sequence("\x1b[13;2u") is True


def test_plain_enter_sequence_is_not_treated_as_shift_enter():
    assert _is_shift_enter_sequence("\r") is False
    assert _is_shift_enter_sequence("\n") is False
    assert _is_shift_enter_sequence(None) is False


def test_modified_enter_event_inserts_newline():
    event = _FakeEvent("\x1b[27;2;13~")

    assert _maybe_insert_newline_for_modified_enter(event) is True
    assert event.current_buffer.inserted == ["\n"]


def test_plain_enter_event_still_falls_through_to_normal_submit_path():
    event = _FakeEvent("\r")

    assert _maybe_insert_newline_for_modified_enter(event) is False
    assert event.current_buffer.inserted == []


def test_csi_u_shift_enter_is_mapped_to_enter_key():
    assert ANSI_SEQUENCES["\x1b[13;2u"] == Keys.ControlM
