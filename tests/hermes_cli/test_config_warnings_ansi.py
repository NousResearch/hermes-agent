"""Config warnings must not leak raw ANSI into non-TTY / NO_COLOR output (#94024, #87919).

``print_config_warnings()`` and ``warn_deprecated_cwd_env_vars()`` wrote ``\\033[...]``
literals unconditionally: piped stderr rendered ``^[[33m…``, and interactive UIs that
strip ESC bytes (prompt_toolkit's ``patch_stdout``) showed the leftover ``[33m…``
garbage. Both now route through ``hermes_cli.colors.color()``, like every other
CLI warning.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.config import print_config_warnings, warn_deprecated_cwd_env_vars


@pytest.fixture()
def one_issue(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.validate_config_structure",
        lambda config: [SimpleNamespace(severity="error", message="boom")],
    )


def test_no_color_strips_escapes_from_config_warnings(monkeypatch, capsys, one_issue):
    monkeypatch.setenv("NO_COLOR", "1")
    print_config_warnings({})
    err = capsys.readouterr().err
    assert "boom" in err
    assert "\033[" not in err


def test_non_tty_output_has_no_escapes(monkeypatch, capsys, one_issue):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    print_config_warnings({})
    assert "\033[" not in capsys.readouterr().err


def test_tty_output_keeps_color(monkeypatch, capsys, one_issue):
    monkeypatch.setattr("hermes_cli.colors.should_use_color", lambda: True)
    print_config_warnings({})
    err = capsys.readouterr().err
    assert "\033[31m" in err  # the error marker keeps its color
    assert "\033[0m" in err  # ...and is reset


def test_deprecated_cwd_warning_respects_no_color(monkeypatch, tmp_path, capsys):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text("TERMINAL_CWD=/x\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("MESSAGING_CWD", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    warn_deprecated_cwd_env_vars()

    err = capsys.readouterr().err
    assert "TERMINAL_CWD" in err
    assert "\033[" not in err
