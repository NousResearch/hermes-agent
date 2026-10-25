"""Regression for #94391: generated Windows ``.vbs`` launchers must be UTF-16 with a BOM.

``wscript.exe`` decodes a ``.vbs`` with the system ANSI code page unless the file carries a
UTF-16 BOM. Both launchers bake ``HERMES_HOME`` and the interpreter path into string literals, so
on an install whose profile path is not representable in that code page (``C:\\Users\\José``, CJK
usernames, ...) the literals mojibake at parse time: the Startup launcher's own ``FileExists``
guard then quits silently at every logon, and ``//B`` swallows the task launcher's runtime error.
``hermes update`` re-encodes both files on every run, so a hand-repaired launcher does not survive.

The ``.cmd`` compatibility wrapper is deliberately out of scope: ``cmd.exe`` has its own code-page
behaviour and service persistence routes through the ``.vbs`` pair only.

These assertions are on what the *consumer* sees, not on the writer's parameters.
"""

from pathlib import Path

import pytest

from hermes_cli import gateway
from hermes_cli import gateway_windows


def _non_ascii_home(tmp_path: Path) -> Path:
    """A HERMES_HOME that a BOM-less .vbs cannot round-trip (the ANSI code page mangles it)."""
    home = tmp_path / "h\u00f4m\u00e8-\u674e\u4e91\u9f99"
    home.mkdir()
    return home


def _stub_launcher_resolution(home: Path, tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    scripts = project / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    (tmp_path / "uv").mkdir()
    venv_python = scripts / "python.exe"
    venv_python.write_text("", encoding="utf-8")
    (project / "venv" / "pyvenv.cfg").write_text(f"home = {tmp_path / 'uv'}\n", encoding="utf-8")

    monkeypatch.setattr(gateway, "PROJECT_ROOT", project)
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(venv_python))
    monkeypatch.setattr(gateway, "_profile_arg", lambda hermes_home: "")
    # _hermes_home() imports this inside the function, so the config module is the patch seam.
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(home))
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))


@pytest.mark.platforms("windows")
def test_task_launcher_is_utf16_so_wscript_resolves_the_profile_path(tmp_path, monkeypatch):
    home = _non_ascii_home(tmp_path)
    _stub_launcher_resolution(home, tmp_path, monkeypatch)

    raw = gateway_windows._write_task_script().with_suffix(".vbs").read_bytes()

    assert raw[:2] == b"\xff\xfe", "a BOM-less .vbs is decoded with the ANSI code page"
    assert str(home) in raw[2:].decode("utf-16-le")


@pytest.mark.platforms("windows")
def test_startup_launcher_is_utf16_so_its_file_exists_guard_can_pass(tmp_path, monkeypatch):
    home = _non_ascii_home(tmp_path)
    _stub_launcher_resolution(home, tmp_path, monkeypatch)

    entry = gateway_windows._install_startup_entry(gateway_windows.get_task_script_path())

    raw = entry.read_bytes()
    assert raw[:2] == b"\xff\xfe", "a BOM-less .vbs is decoded with the ANSI code page"
    assert str(home) in raw[2:].decode("utf-16-le")
