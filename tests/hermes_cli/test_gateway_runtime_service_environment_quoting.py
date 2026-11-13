"""Every ``systemctl show-environment`` output systemd can produce, read back: the reader is
tested against a model of systemd's own writer, not hand-written samples, and refuses what
systemd never writes."""
import itertools
import json
import os
import random
import subprocess

import pytest

from hermes_cli.gateway_runtime_service_identity import manager_environment_pairs

# Asking the host's systemd manager writes to its real environment: opt in.
REAL_SYSTEMD = os.environ.get("HERMES_TEST_REAL_SYSTEMD") == "1"

# ── systemd's writer, translated from src/basic/escape.c (v262) ──

# shell_maybe_quote() quotes a value when it holds whitespace, a control byte, an invalid UTF-8
# sequence, or a byte from SHELL_NEED_QUOTES. The first such byte starts the quoted part; the
# bytes before it are copied as they are.
WHITESPACE = b" \t\n\r"
GLOB_CHARS = b"*?["
SHELL_NEED_ESCAPE = b'"\\`$'
SHELL_NEED_QUOTES = SHELL_NEED_ESCAPE + GLOB_CHARS + b"'()<>|&;!"
# strcpy_backslash_escaped() escapes only these two inside the quoted part.
SHELL_NEED_ESCAPE_POSIX = b"\\'"


def _is_control(byte):
    return byte < 0x20 or byte == 0x7F


def _utf8_length(raw, index):
    """The length of the UTF-8 character at *index*, or -1 when the bytes are not valid UTF-8."""
    first = raw[index]
    if first < 0x80:
        return 1
    length = 2 if first < 0xE0 else 3 if first < 0xF0 else 4 if first < 0xF5 else -1
    if length < 0 or index + length > len(raw):
        return -1
    try:
        raw[index:index + length].decode("utf-8")
    except UnicodeDecodeError:
        return -1
    if length == 2 and first < 0xC2:
        return -1
    if length == 3 and first == 0xE0 and raw[index + 1] < 0xA0:
        return -1
    if length == 4 and first == 0xF0 and raw[index + 1] < 0x90:
        return -1
    if length == 4 and first == 0xF4 and raw[index + 1] > 0x8F:
        return -1
    if 0xED <= first <= 0xEF and 0xA0 <= raw[index + 1] <= 0xBF:
        return -1
    return length


def _cescape_char(byte):
    """cescape_char(): the C escape systemd writes for one byte."""
    for value, escape in (
        (0x07, b"\\a"), (0x08, b"\\b"), (0x0C, b"\\f"), (0x0A, b"\\n"),
        (0x0D, b"\\r"), (0x09, b"\\t"), (0x0B, b"\\v"), (0x5C, b"\\\\"),
        (0x22, b'\\"'), (0x27, b"\\'"),
    ):
        if byte == value:
            return escape
    if byte < 0x20 or byte >= 0x7F:
        return b"\\%03o" % byte
    return bytes([byte])


def systemd_quotes(value: bytes) -> bytes:
    """shell_maybe_quote(SHELL_ESCAPE_POSIX): what systemd prints for one value."""
    index = 0
    while index < len(value):
        byte = value[index]
        if (_is_control(byte) or _utf8_length(value, index) < 0
                or bytes([byte]) in WHITESPACE or bytes([byte]) in SHELL_NEED_QUOTES):
            break
        index += _utf8_length(value, index)
    if index == len(value):
        return value
    out = bytearray(b"$'")
    out += value[:index]
    while index < len(value):
        byte = value[index]
        length = _utf8_length(value, index)
        if _is_control(byte) or length < 0:
            out += _cescape_char(byte)
            index += 1
        elif length == 1:
            if byte == 0x5C or bytes([byte]) in SHELL_NEED_ESCAPE_POSIX:
                out += b"\\"
            out.append(byte)
            index += 1
        else:
            out += value[index:index + length]
            index += length
    out += b"'"
    return bytes(out)


# The bytes that change how systemd writes a value: the control bytes it
# escapes, the shell characters it quotes for, and the backslash itself.
INTERESTING = [0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1B, 0x7F, 0x80, 0xFF,
                0x20, 0x22, 0x27, 0x28, 0x5C, 0x60, 0x24, 0x2A, 0x3F, 0x5B, 0x7C, 0x3B, 0x21]


def systemd_dump(value: bytes, name: str = "VAR") -> str:
    """The line systemd prints for one variable."""
    return f"{name}={systemd_quotes(value).decode('utf-8', 'surrogateescape')}\n"


# ── the reader has to give back exactly what the writer was given ──

def _round_trip(value: bytes):
    read = manager_environment_pairs(systemd_dump(value))
    assert read == {"VAR": value.decode("utf-8", "surrogateescape")}


def test_every_single_byte_round_trips():
    """Every byte a value can hold survives the writer and the reader."""
    for byte in range(1, 256):
        _round_trip(bytes([byte]))


def test_every_pair_of_bytes_round_trips():
    """Two bytes are enough to put a quote next to a backslash."""
    for pair in itertools.product(range(1, 256), repeat=2):
        _round_trip(bytes(pair))


def test_every_pair_of_the_interesting_bytes_round_trips():
    """The bytes that change how systemd writes a value, in every order."""
    for pair in itertools.product(INTERESTING, repeat=2):
        _round_trip(bytes(pair))


def test_every_triple_of_the_interesting_bytes_round_trips():
    """Three bytes are where an escape meets the byte that follows it."""
    for triple in itertools.product(INTERESTING, repeat=3):
        _round_trip(bytes(triple))


def test_a_backslash_next_to_every_byte_round_trips():
    """The backslash decides whether a value comes out quoted."""
    for byte in range(1, 256):
        _round_trip(bytes([0x5C, byte]))
        _round_trip(bytes([byte, 0x5C]))


def test_an_escaped_backslash_next_to_every_byte_round_trips():
    """``\\\\`` is how systemd writes a literal backslash."""
    for byte in range(1, 256):
        _round_trip(bytes([0x5C, 0x5C, byte]))


def test_a_long_random_value_round_trips():
    """A value with the bytes a session mixes."""
    generator = random.Random(20261110)
    for _ in range(20000):
        length = generator.randrange(1, 12)
        _round_trip(bytes(generator.randrange(1, 256) for _ in range(length)))


@pytest.mark.parametrize("value", [
    b"sh -c 'col -bx | bat -l man -p'",   # the MANPAGER a shell exports
    b"https://debuginfod.cachyos.org https://debuginfod.archlinux.org ",
    b"--prompt=\\u276f \\' --pointer=\\u258c\\'",
    b"plain",
    b"with space",
    b"with#hash",
    b"caf\xc3\xa9",
    b"\xf0\x9f\x98\x80",
])
def test_the_values_a_real_session_exports_round_trip(value):
    _round_trip(value)


# ── the model has to match the real writer ──

def _real_systemctl_dump(value: bytes) -> bytes | None:
    """What the host's systemctl prints for one variable, or None when it cannot be asked."""
    try:
        before = subprocess.run(["systemctl", "--user", "show-environment", "--output=json"],
                                capture_output=True, timeout=10, check=False).stdout.decode(
            "utf-8", "surrogateescape")
    except (OSError, subprocess.SubprocessError):
        return None
    if not before.startswith("{"):
        # This systemd prints no JSON form: the value it holds cannot be read back.
        return None
    try:
        data = json.loads(before)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    # The JSON form carries the value the manager holds, not the form it prints:
    # the printed form of a quoted value would be restored as its own quoting.
    previous = data.get("HERMES_TEST_VAR")
    try:
        subprocess.run(["systemctl", "--user", "unset-environment", "HERMES_TEST_VAR"],
                       capture_output=True, timeout=10, check=False)
        subprocess.run(["systemctl", "--user", "set-environment", f"HERMES_TEST_VAR={value.decode('utf-8', 'surrogateescape')}"],
                       capture_output=True, timeout=10, check=False)
        dumped = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True,
                                timeout=10, check=False).stdout.decode("utf-8", "surrogateescape")
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        # The variable belongs to the host: give it back what it held.
        if previous is None:
            subprocess.run(["systemctl", "--user", "unset-environment", "HERMES_TEST_VAR"],
                           capture_output=True, timeout=10, check=False)
        else:
            subprocess.run(["systemctl", "--user", "set-environment", f"HERMES_TEST_VAR={previous}"],
                           capture_output=True, timeout=10, check=False)
    for line in dumped.split("\n"):
        # A plain split, not a shell grep: the value can hold a glob character.
        if line.startswith("HERMES_TEST_VAR="):
            return line[len("HERMES_TEST_VAR="):].encode("utf-8", "surrogateescape")
    return None


@pytest.mark.skipif(not REAL_SYSTEMD, reason="set HERMES_TEST_REAL_SYSTEMD=1 to ask the host's systemctl")
@pytest.mark.parametrize("value", [
    b"a", b"a b", b"a\\b", b"a'b", b"a\x07b", b"a?b", b"a*b", b"a\"b",
    b"sh -c 'col -bx | bat -l man -p'", b"caf\xc3\xa9", b"\\q", b"\\12",
])
def test_the_model_matches_the_real_systemctl(value):
    """The model above has to be what the host's systemd writes, or it proves nothing."""
    real = _real_systemctl_dump(value)
    if real is None:
        pytest.skip("this host has no user systemd manager to ask")
    assert real == systemd_quotes(value)


# ── a byte systemd never writes must not be accepted ──

@pytest.mark.parametrize("value", [
    "Z=$'a\\qc'",     # an escape systemd does not write: it doubles the backslash instead
    "Z=$'a\\12'",     # an incomplete octal escape: systemd always writes three digits
    "Z=$'abc",       # a quote that is never closed: systemd always closes it
])
def test_an_escape_systemd_never_writes_is_refused(value):
    """The reader refuses what systemd cannot produce, so a grammar that changed is noticed
    instead of quietly misreading a variable."""
    from hermes_cli.gateway_runtime_service_identity import Unverified
    with pytest.raises(Unverified):
        manager_environment_pairs(value + "\n")
