import os
import sys
import time

import pytest

from hermes_cli.secret_prompt import (
    _collect_masked_input,
    _read_posix_raw_char,
    masked_secret_prompt,
)


def _run_collect(chars: str):
    output: list[str] = []
    iterator = iter(chars)

    def read_char() -> str:
        return next(iterator, "")

    def write(text: str) -> None:
        output.append(text)

    value = _collect_masked_input(
        read_char,
        write,
        "API key: ",
    )
    return value, "".join(output)


def test_collect_masked_input_shows_feedback_without_echoing_secret():
    value, output = _run_collect("secret\n")

    assert value == "secret"
    assert output == "API key: ******\r\n"
    assert "secret" not in output


def test_collect_masked_input_handles_backspace():
    value, output = _run_collect("sec\x7fret\r")

    assert value == "seret"
    assert output == "API key: ***\b \b***\r\n"
    assert "secret" not in output


def test_collect_masked_input_raises_keyboard_interrupt():
    output: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        _collect_masked_input(
            lambda: "\x03",
            output.append,
            "API key: ",
        )

    assert "".join(output) == "API key: \r\n"


def test_masked_secret_prompt_falls_back_to_getpass_for_non_tty(monkeypatch):
    class NonTty:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", NonTty())
    monkeypatch.setattr("sys.stdout", NonTty())
    monkeypatch.setattr("getpass.getpass", lambda prompt: f"value from {prompt}")

    assert masked_secret_prompt("API key: ") == "value from API key: "


# ---- POSIX raw-char reader ---------------------------------------------------
#
# These tests only exercise the POSIX byte-reader in isolation, so they run
# anywhere ``os.pipe`` works (i.e. non-Windows). They intentionally do NOT
# spin up a real pty — the drain uses ``select`` on the fd, which works
# identically on pipes and pseudo-terminals.

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="_read_posix_raw_char is POSIX-only; Windows has its own path.",
)


def _pipe_pair():
    r, w = os.pipe()
    return r, w


@_POSIX_ONLY
def test_read_posix_raw_char_reads_single_ascii_byte():
    r, w = _pipe_pair()
    try:
        os.write(w, b"x")
        assert _read_posix_raw_char(r) == "x"
    finally:
        os.close(r)
        os.close(w)


@_POSIX_ONLY
def test_read_posix_raw_char_reassembles_multibyte_utf8():
    r, w = _pipe_pair()
    try:
        # "é" is 0xc3 0xa9 in UTF-8; must come back as one Python character
        # even though the reader pulls one byte at a time.
        os.write(w, "é".encode("utf-8"))
        assert _read_posix_raw_char(r) == "é"
    finally:
        os.close(r)
        os.close(w)


@_POSIX_ONLY
def test_read_posix_raw_char_drains_arrow_key_escape_sequence():
    # This is the fix for the silent-secret-corruption bug: pressing arrow-up
    # mid-password used to leak "[A" (or the tail of any CSI/SS3 sequence)
    # into the captured value, because ``_collect_masked_input`` only skipped
    # the leading ESC byte. The drain here must consume the "[A" so the
    # NEXT read_char() call sees the byte AFTER the escape sequence.
    #
    # The sleep between writes mirrors real terminal timing: escape sequences
    # arrive as a fast burst (sub-millisecond), then there is a human-scale
    # gap before the next typed character. The reader's drain uses a 5 ms
    # ``select`` window, so 20 ms comfortably separates the two events.
    r, w = _pipe_pair()
    try:
        os.write(w, b"\x1b[A")   # arrow-up: ESC [ A
        first = _read_posix_raw_char(r)
        time.sleep(0.02)
        os.write(w, b"w")
        second = _read_posix_raw_char(r)
        assert first == "\x1b"
        assert second == "w", (
            f"escape-sequence tail leaked: expected 'w' after drain, got {second!r}"
        )
    finally:
        os.close(r)
        os.close(w)


@_POSIX_ONLY
def test_read_posix_raw_char_drains_ss3_escape_sequence():
    # SS3 sequences (ESC O <final>) are emitted by some terminals for F1-F4
    # and, in application cursor mode, for the arrow keys. Only the two-byte
    # tail must be consumed — one introducer byte plus one final byte.
    r, w = _pipe_pair()
    try:
        os.write(w, b"\x1bOP")   # F1 in xterm's SS3 form
        first = _read_posix_raw_char(r)
        time.sleep(0.02)
        os.write(w, b"x")
        second = _read_posix_raw_char(r)
        assert first == "\x1b"
        assert second == "x", (
            f"SS3 tail leaked: expected 'x' after drain, got {second!r}"
        )
    finally:
        os.close(r)
        os.close(w)


@_POSIX_ONLY
def test_read_posix_raw_char_lone_escape_returns_escape():
    # A lone Escape keypress (no follow-up bytes) must still surface as ESC
    # so the collector can skip it. The short peek window is what
    # distinguishes this from a sequence.
    r, w = _pipe_pair()
    try:
        os.write(w, b"\x1b")
        assert _read_posix_raw_char(r) == "\x1b"
    finally:
        os.close(r)
        os.close(w)


@_POSIX_ONLY
def test_read_posix_raw_char_preserves_meta_alt_follow_up_char():
    # Some terminals encode Meta/Alt-<char> as ESC + <char>. The follow-up
    # byte is NOT part of a CSI/SS3 sequence and must not be silently
    # swallowed alongside the ESC — otherwise we'd trade the arrow-key
    # corruption bug for a Meta-key corruption bug.
    r, w = _pipe_pair()
    try:
        os.write(w, b"\x1ba")   # Alt-a on many terminals
        first = _read_posix_raw_char(r)
        assert first == "a", (
            f"Meta/Alt follow-up byte was dropped: expected 'a', got {first!r}"
        )
    finally:
        os.close(r)
        os.close(w)


@_POSIX_ONLY
def test_read_posix_raw_char_end_to_end_typed_secret_with_alt_key():
    # End-to-end companion to the arrow-key test: if the user hits Alt-<char>
    # mid-secret, the <char> must land in the captured value. Pre-fix
    # (original drain-everything approach) this would produce "pasword"
    # because the ESC drain swallowed the 's'.
    r, w = _pipe_pair()
    output: list[str] = []

    def read_char():
        return _read_posix_raw_char(r)

    def write(text):
        output.append(text)

    import threading

    def feed():
        os.write(w, b"pa")
        time.sleep(0.02)
        os.write(w, b"\x1bs")   # Alt-s burst mid-secret
        time.sleep(0.02)
        os.write(w, b"sword\r")

    writer = threading.Thread(target=feed)
    writer.start()
    try:
        value = _collect_masked_input(read_char, write, "pw: ", mask="*")
    finally:
        writer.join(timeout=1.0)
        os.close(r)
        os.close(w)

    assert value == "password", f"Alt-key follow-up byte lost: {value!r}"
    assert "".join(c for c in output if c == "*") == "*" * 8


@_POSIX_ONLY
def test_read_posix_raw_char_end_to_end_typed_secret_with_arrow_key():
    # End-to-end: drive ``_collect_masked_input`` with the reader used by
    # ``_masked_secret_prompt_posix`` and prove that "pass<Up>word<Enter>"
    # captures "password" rather than the pre-fix "pass[Aword".
    r, w = _pipe_pair()
    output: list[str] = []

    def read_char():
        return _read_posix_raw_char(r)

    def write(text):
        output.append(text)

    # A background writer paces the bytes the way a real user would, so the
    # 5 ms drain window can observe the boundary between the arrow-up burst
    # and the trailing typed characters.
    import threading

    def feed():
        os.write(w, b"pass")
        time.sleep(0.02)
        os.write(w, b"\x1b[A")
        time.sleep(0.02)
        os.write(w, b"word\r")

    writer = threading.Thread(target=feed)
    writer.start()
    try:
        value = _collect_masked_input(read_char, write, "pw: ", mask="*")
    finally:
        writer.join(timeout=1.0)
        os.close(r)
        os.close(w)

    assert value == "password", f"arrow-key bytes leaked into secret: {value!r}"
    # 8 typed characters -> 8 mask chars, one per keystroke.
    assert "".join(c for c in output if c == "*") == "*" * 8
