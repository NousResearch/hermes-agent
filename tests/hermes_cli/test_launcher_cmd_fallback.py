"""The Windows ``.cmd`` launcher fallback must stay ASCII.

cmd.exe parses a batch file with the console's OEM code page, so a literal
UTF-8 interpreter path is corrupted before the line ever runs ("The system
cannot find the path specified."). With a profile like
``C:\\Users\\张三\\AppData\\Local\\hermes`` the staged ``.cmd`` therefore
failed, the desktop's ``--version`` probe (``execProbe`` in
``apps/desktop``'s ``source-backend.ts``) read the install as missing, and its
first-run bootstrap re-ran forever. The fallback now derives the interpreter
from the batch file's own directory (``%~dp0``) whenever that keeps the body
ASCII — the repo root rides the base64 payload, which is ASCII by
construction.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

from hermes_cli import _launchers

_CMD_PAYLOAD = re.compile(rb"exec\(base64\.b64decode\('([A-Za-z0-9+/=]+)'\)\)")


@pytest.fixture()
def cmd_fallback(monkeypatch):
    """Force the no-distlib branch — the ``.cmd`` is the *fallback* launcher."""
    monkeypatch.setattr(_launchers, "_load_script_maker", lambda: None)


def test_cmd_body_is_ascii_when_the_install_path_is_not(cmd_fallback, tmp_path):
    """A non-ASCII profile (CJK user name) still yields an ASCII parser window."""
    home = tmp_path / "\u5f20\u4e09home"
    root = home / "hermes-agent"
    out = home / "bin"
    out.mkdir(parents=True)
    py = home / "tools" / "python-3.14.7-win32-x64" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"MZ")  # the mint never executes or reads the interpreter

    written = _launchers.mint_launcher("hermes", root, out, py, None)

    assert written is not None and written.suffix == ".cmd"
    body = written.read_bytes()
    assert body.isascii(), body.decode("utf-8", "replace")
    # The interpreter is resolved at run time from the launcher's own directory.
    assert b'"%~dp0..\\tools\\python-3.14.7-win32-x64\\python.exe"' in body
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
