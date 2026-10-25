"""`hermes config show` and `hermes dump` must survive a hand-edited, non-mapping config section.

``terminal: docker``, ``compression: x`` or ``auxiliary: {vision: x}`` used to raise ``AttributeError``,
and a quoted ``compression.threshold: "0.5"`` a ``TypeError``, so the two read-only diagnostics crashed
on the configs users most need them for. Each test writes a real config.yaml into the sandboxed home.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
HEADERS = ("◆ Display", "◆ Terminal", "◆ Context Compression")


@pytest.fixture
def home(monkeypatch, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    path = get_hermes_home()
    path.mkdir(parents=True, exist_ok=True)
    (path / ".env").write_text("", encoding="utf-8")
    return path


def _show(home: Path, config_yaml: str, capsys) -> str:
    from hermes_cli.config_show import show_config

    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    show_config()
    return capsys.readouterr().out


def _line(out: str, label: str) -> str:
    """The last line starting with ``label`` (``Model:`` also appears in the Model section above)."""
    return [line.strip() for line in out.splitlines() if line.strip().startswith(label)][-1]


@pytest.mark.parametrize("config_yaml,label,expected", [
    ("terminal: docker\n", "Backend:", "Backend:      local"),
    ("display: x\n", "Backend:", "Backend:      local"),
    ("compression: x\n", "Threshold:", "Threshold:    50%"),
    ('compression:\n  threshold: "0.5"\n', "Threshold:", "Threshold:    50%"),
    ("compression:\n  threshold: null\n", "Threshold:", "Threshold:    None (not a number)"),
    ("compression:\n  threshold: abc\n", "Threshold:", "Threshold:    'abc' (not a number)"),
    ("compression:\n  target_ratio: x\n", "Target ratio:", "Target ratio: 'x' (not a number) of threshold preserved"),
    ("auxiliary:\n  compression: x\n", "Model:", "Model:        (auto)"),
    ("auxiliary:\n  vision: x\n", "Threshold:", "Threshold:    50%"),
])
def test_config_show_survives_malformed_section(home, capsys, config_yaml, label, expected):
    out = _show(home, config_yaml, capsys)

    for header in HEADERS:
        assert header in out
    assert _line(out, label) == expected


def test_config_show_valid_config_is_unchanged(home, capsys):
    out = _show(home, "terminal:\n  backend: docker\ncompression:\n  threshold: 0.5\n  target_ratio: 0.3\n", capsys)

    assert _line(out, "Backend:") == "Backend:      docker"
    assert _line(out, "Threshold:") == "Threshold:    50%"
    assert _line(out, "Target ratio:") == "Target ratio: 30% of threshold preserved"


def _dump(home: Path, config_yaml: str, capsys) -> dict[str, str]:
    from hermes_cli import dump

    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    dump.run_dump(SimpleNamespace(show_keys=False))
    rows = {}
    for line in capsys.readouterr().out.splitlines():
        key, sep, value = line.strip().partition(":")
        if sep and key not in rows:
            rows[key] = value.strip()
    return rows


def test_dump_survives_malformed_sections(home, capsys):
    rows = _dump(home, "terminal: docker\nmcp: x\nmemory: x\n", capsys)

    assert rows["terminal"] == "local"
    assert rows["mcp_servers"] == "0"
    assert rows["memory_provider"] == "built-in"


def test_dump_valid_config_is_unchanged(home, capsys):
    rows = _dump(home, "terminal:\n  backend: docker\nmcp:\n  servers:\n    a: {command: x}\n"
                       "memory:\n  provider: honcho\n", capsys)

    assert rows["terminal"] == "docker"
    assert rows["mcp_servers"] == "1"
    assert rows["memory_provider"] == "honcho"


@pytest.mark.parametrize("argv", [["config", "show"], ["dump"]])
def test_cli_exits_cleanly_on_malformed_sections(tmp_path, argv):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text('terminal: docker\ncompression:\n  threshold: "0.5"\n', encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "TERMINAL_"))}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(home), COLUMNS="200",
               PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))

    proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", *argv], env=env, cwd=tmp_path,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180, check=False)

    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "Traceback" not in output
    if argv == ["dump"]:
        assert "terminal:         local" in output
    else:
        assert "Backend:      local" in output
        assert "Threshold:    50%" in output
