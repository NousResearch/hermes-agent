"""The Windows ``.cmd`` launcher fallback must stay ASCII and unbuffered.

cmd.exe parses a batch file with the console's OEM code page, so a literal
UTF-8 interpreter path is corrupted before the line ever runs ("The system
cannot find the path specified."). With a profile like
``C:\\Users\\张三\\AppData\\Local\\hermes`` the staged ``.cmd`` therefore
failed, the desktop's ``--version`` probe (``execProbe`` in
``apps/desktop``'s ``source-backend.ts``) read the install as missing, and its
first-run bootstrap re-ran forever; the repo root rides the base64 payload,
which is ASCII by construction.

Length is the second constraint (PR #122594 review): cmd.exe copies a line
that carries a variable reference into a bounded (~8191 char) post-expansion
buffer, and the payload makes the executable line ~9 KB -- so ``%~dp0`` inline
exits 255 and the interpreter resolves through a delayed reference instead.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

from hermes_cli import _launchers

_CMD_PAYLOAD = re.compile(rb"exec\(base64\.b64decode\('([A-Za-z0-9+/=]+)'\)\)")
_VARIABLE_REFERENCE = re.compile(r"%~|%[A-Za-z_][A-Za-z0-9_]*%")


@pytest.fixture()
def cmd_fallback(monkeypatch):
    """Force the no-distlib branch — the ``.cmd`` is the *fallback* launcher."""
    monkeypatch.setattr(_launchers, "_load_script_maker", lambda: None)


def _cjk_install(tmp_path: Path):
    """A CJK profile whose interpreter still has an ASCII relative path."""
    home = tmp_path / "\u5f20\u4e09home"
    root = home / "hermes-agent"
    out = home / "bin"
    out.mkdir(parents=True)
    py = home / "tools" / "python-3.14.7-win32-x64" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"MZ")  # the mint never executes or reads the interpreter
    return root, out, py


def test_cmd_body_is_ascii_when_the_install_path_is_not(cmd_fallback, tmp_path):
    """A non-ASCII profile (CJK user name) still yields an ASCII parser window."""
    root, out, py = _cjk_install(tmp_path)

    written = _launchers.mint_launcher("hermes", root, out, py, None)

    assert written is not None and written.suffix == ".cmd"
    body = written.read_bytes()
    assert body.isascii(), body.decode("utf-8", "replace")
    # The interpreter resolves at run time from the launcher's own directory,
    # through a delayed reference: a plain %~dp0 line would be copied into
    # cmd's ~8191-char post-expansion buffer, and the payload line is ~9 KB.
    assert b"setlocal EnableDelayedExpansion" in body
    assert b'set "HPY=%~dp0..\\tools\\python-3.14.7-win32-x64\\python.exe"' in body
    assert b'"!HPY!" -I -c "' in body
    # The CJK repo root survives where it is DATA (decoded by Python at boot),
    # not parsed shell text: the payload is intact, ASCII, valid Python.
    payload = _CMD_PAYLOAD.search(body)
    assert payload, body
    script = base64.b64decode(payload.group(1)).decode("utf-8")
    assert "\u5f20\u4e09home" in script
    compile(script, "<launcher payload>", "exec")


def test_cmd_body_keeps_the_absolute_interpreter_when_relative_is_not_ascii(
    cmd_fallback, tmp_path
):
    """No ASCII relative form (CJK sibling dir) — the absolute path stays."""
    root = tmp_path / "repo"
    out = tmp_path / "bin"
    out.mkdir(parents=True)
    py = tmp_path / "\u5f20\u4e09tools" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"MZ")

    written = _launchers.mint_launcher("hermes", root, out, py, None)

    assert written is not None and written.suffix == ".cmd"
    body = written.read_text(encoding="utf-8")
    assert str(py) in body
    assert "%~dp0" not in body


def test_cmd_body_keeps_the_literal_when_it_is_ascii(cmd_fallback, tmp_path):
    """An ASCII interpreter needs no indirection (and no delayed expansion)."""
    root = tmp_path / "repo"
    out = tmp_path / "bin"
    out.mkdir(parents=True)
    # A virtual path: ASCII by construction (tmppaths themselves can sit under
    # a non-ASCII user name) and the mint never reads or executes it.
    py = Path("C:/tools/python-3.14.7-win32-x64/python.exe")

    written = _launchers.mint_launcher("hermes", root, out, py, None)

    assert written is not None and written.suffix == ".cmd"
    body = written.read_text(encoding="utf-8")
    assert f'"{py}" -I -c "' in body
    assert "%~dp0" not in body
    assert "!HPY!" not in body


def test_no_variable_reference_rides_a_long_line(cmd_fallback, tmp_path):
    """cmd.exe buffers an expanded line (~8191 chars post-expansion).

    The payload line is ~9 KB in a real install, so a %-reference on it exits
    255 ("The input line is too long."). References belong on their own short
    lines; only a delayed reference may sit next to the payload.
    """
    root, out, py = _cjk_install(tmp_path)

    written = _launchers.mint_launcher("hermes", root, out, py, None)
    body = written.read_text(encoding="utf-8")

    payload_lines = [line for line in body.splitlines() if "b64decode" in line]
    assert payload_lines
    for line in payload_lines:
        assert not _VARIABLE_REFERENCE.search(line), line
    for line in body.splitlines():
        if _VARIABLE_REFERENCE.search(line):
            assert len(line) < 8191, "an expanded line must stay under cmd's buffer"
