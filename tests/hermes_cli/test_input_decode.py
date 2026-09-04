"""Tests for hermes_cli.input_decode (#97975) — pure string transform, no TTY."""

import pytest

from hermes_cli.input_decode import decode_modified_key_sequences, safe_input

ESC = "\x1b"


class TestDecodeModifiedKeySequences:
    def test_plain_text_untouched(self):
        for text in ("", "y", "hello world", "s3cret-key_123", "n\n"):
            assert decode_modified_key_sequences(text) == text

    def test_shift_only_modify_other_keys(self):
        # The issue's exact repro: Shift+T arrives as ESC[27;2;84~ (84 == 'T').
        assert decode_modified_key_sequences(f"{ESC}[27;2;84~") == "T"

    def test_mixed_string_round_trip(self):
        # Typing "Type THIS" with the shifted letters reported as sequences.
        raw = (
            "Type "
            f"{ESC}[27;2;84~"   # Shift+T -> T
            f"{ESC}[27;2;72~"   # Shift+H -> H
            f"{ESC}[27;2;73~"   # Shift+I -> I
            f"{ESC}[27;2;83~"   # Shift+S -> S
        )
        assert decode_modified_key_sequences(raw) == "Type THIS"

    def test_kitty_csi_u_shift_only(self):
        assert decode_modified_key_sequences(f"{ESC}[84;2u") == "T"
        # Kitty press/repeat low bits: shift|press(1)+1=3, shift|release+1=4.
        assert decode_modified_key_sequences(f"{ESC}[84;3u") == "T"
        assert decode_modified_key_sequences(f"{ESC}[84;4u") == "T"

    def test_control_and_alt_combos_pass_through(self):
        # Ctrl+T (5), Ctrl+Shift+T (6), Super+T (12) — decoding would invent
        # control characters or silently drop keys.  Alt+T (3) passes through
        # under modifyOtherKeys too: unlike Kitty, that protocol never ORs
        # event bits into the modifier, so 3 is genuinely Alt there.
        for mod in ("5", "6", "12"):
            seq = f"{ESC}[27;{mod};84~"
            assert decode_modified_key_sequences(seq) == seq
        assert decode_modified_key_sequences(f"{ESC}[27;3;84~") == f"{ESC}[27;3;84~"
        assert decode_modified_key_sequences(f"{ESC}[84;6u") == f"{ESC}[84;6u"

    def test_lock_bits_pass_through(self):
        # CapsLock is applied by modifyOtherKeys *before* encoding: 65
        # (shift|caps) on T sends codepoint 116 = lowercase 't'.  Decoding
        # it would silently rewrite an un-shifted keypress, so it stays raw.
        for seq in (f"{ESC}[27;65;116~", f"{ESC}[27;129;84~"):
            assert decode_modified_key_sequences(seq) == seq

    def test_functional_keys_pass_through(self):
        # Shift+Enter under Kitty: codepoint 13 is a key, not a printable char.
        seq = f"{ESC}[13;2u"
        assert decode_modified_key_sequences(seq) == seq

    def test_bare_csi_u_without_modifier_passes_through(self):
        # CSI-u with omitted mod parameter (mod==1) is not a modified key.
        seq = f"{ESC}[97u"
        assert decode_modified_key_sequences(seq) == seq

    def test_unparsable_codepoints_pass_through(self):
        # Out-of-range codepoints: chr() raises, sequence stays raw.
        assert decode_modified_key_sequences(f"{ESC}[27;2;1114112~") == f"{ESC}[27;2;1114112~"
        assert decode_modified_key_sequences(f"{ESC}[99999999999999;2u") == f"{ESC}[99999999999999;2u"

    def test_surrogate_codepoints_pass_through(self):
        seq = f"{ESC}[27;2;55296~"  # 0xD800 — chr() raises
        assert decode_modified_key_sequences(seq) == seq

    def test_unrelated_escapes_untouched(self):
        # Bracketed paste, colors, cursor moves are not modified-key sequences.
        for seq in (f"{ESC}[?2004h", f"{ESC}[31m", f"{ESC}[1;2H", f"{ESC}[27;5m"):
            assert decode_modified_key_sequences(seq) == seq


class TestSafeInput:
    def test_decodes_reader_output(self):
        reader = lambda prompt: (print(prompt, end=""), f"{ESC}[27;2;89~")[1]  # Shift+Y
        assert safe_input("Allow? [y/N]: ", _reader=reader) == "Y"

    def test_passthrough_for_plain_reads(self):
        assert safe_input("", _reader=lambda _p: "y") == "y"

    def test_eof_propagates(self):
        def boom(_p):
            raise EOFError

        with pytest.raises(EOFError):
            safe_input("", _reader=boom)
