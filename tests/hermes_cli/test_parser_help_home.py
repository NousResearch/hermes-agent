"""The top-level parser's help text must not trip the wrong-profile fallback warning.

``main._apply_profile_override`` builds the parser (``top_level_value_flag_sets``) before it
re-homes the process to the sticky ``active_profile``; a help string that resolved the home through
``get_hermes_home()`` printed ``[HERMES_HOME fallback] ... wrong profile`` on every ``hermes``
command for users of ``hermes profile use`` (#112319, also reported in #112839).
"""

from pathlib import Path

import pytest


@pytest.fixture
def sticky_profile_home(monkeypatch, tmp_path):
    import hermes_constants

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    root = tmp_path / ".hermes"
    (root / "profiles" / "coder").mkdir(parents=True)
    (root / "active_profile").write_text("coder\n", encoding="utf-8")
    monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", False)
    return root


def test_building_the_parser_before_the_profile_override_stays_silent(sticky_profile_home, capsys):
    from hermes_cli._parser import build_top_level_parser

    build_top_level_parser()

    assert "HERMES_HOME fallback" not in capsys.readouterr().err


def test_help_text_names_the_profile_config_once_the_process_is_re_homed(sticky_profile_home, monkeypatch):
    from hermes_cli._parser import _cfg_path

    monkeypatch.setenv("HERMES_HOME", str(sticky_profile_home / "profiles" / "coder"))

    assert _cfg_path() == "~/.hermes/profiles/coder/config.yaml"
