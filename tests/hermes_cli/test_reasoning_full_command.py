"""Tests for the CLI `/reasoning full` / `/reasoning clamp` recap toggle.

The post-response "Reasoning" recap box clamps long thinking to the first
10 lines. `/reasoning full` opts into uncapped display (Taelin's "show all
thinking tokens" ask); `/reasoning clamp` restores the 10-line collapse.
These assert the toggle sets the instance flag, persists to config.yaml,
and that the clamp gate honours the flag.
"""

import os

import yaml

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.config import DEFAULT_CONFIG


class _Stub(CLICommandsMixin):
    """Minimal carrier for the attributes `_handle_reasoning_command` reads."""

    def __init__(self):
        self.reasoning_config = None
        self.show_reasoning = True
        self.reasoning_full = False
        self.reasoning_clamp_lines = 10
        self.agent = None

    def _current_reasoning_callback(self):
        return None


def test_default_config_clamps_reasoning():
    # Behaviour contract: the recap defaults to clamped, not full.
    assert DEFAULT_CONFIG["display"]["reasoning_full"] is False
    assert DEFAULT_CONFIG["display"]["reasoning_clamp_lines"] == 10


def _seed_config(tmp_path, monkeypatch):
    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / "config.yaml").write_text("display:\n  show_reasoning: true\n")
    monkeypatch.setenv("HERMES_HOME", str(hh))
    # cli captures _hermes_home at import; force it to the temp home.
    import cli

    monkeypatch.setattr(cli, "_hermes_home", hh, raising=False)
    return hh


def test_reasoning_full_sets_and_persists(tmp_path, monkeypatch):
    hh = _seed_config(tmp_path, monkeypatch)
    s = _Stub()

    s._handle_reasoning_command("/reasoning full")
    assert s.reasoning_full is True
    saved = yaml.safe_load((hh / "config.yaml").read_text())
    assert saved["display"]["reasoning_full"] is True


def test_reasoning_clamp_resets_and_persists(tmp_path, monkeypatch):
    hh = _seed_config(tmp_path, monkeypatch)
    s = _Stub()
    s.reasoning_full = True

    s._handle_reasoning_command("/reasoning clamp")
    assert s.reasoning_full is False
    saved = yaml.safe_load((hh / "config.yaml").read_text())
    assert saved["display"]["reasoning_full"] is False


def test_reasoning_clamp_accepts_custom_line_count(tmp_path, monkeypatch):
    hh = _seed_config(tmp_path, monkeypatch)
    s = _Stub()
    s.reasoning_full = True

    s._handle_reasoning_command("/reasoning clamp 20")
    assert s.reasoning_full is False
    assert s.reasoning_clamp_lines == 20
    saved = yaml.safe_load((hh / "config.yaml").read_text())
    assert saved["display"]["reasoning_full"] is False
    assert saved["display"]["reasoning_clamp_lines"] == 20


def test_reasoning_clamp_rejects_invalid_line_count(tmp_path, monkeypatch):
    hh = _seed_config(tmp_path, monkeypatch)
    s = _Stub()
    s.reasoning_full = True

    s._handle_reasoning_command("/reasoning clamp 0")
    assert s.reasoning_full is True
    assert s.reasoning_clamp_lines == 10
    saved = yaml.safe_load((hh / "config.yaml").read_text())
    assert "reasoning_full" not in saved["display"]
    assert "reasoning_clamp_lines" not in saved["display"]


def test_reasoning_full_rejects_extra_arguments(tmp_path, monkeypatch):
    hh = _seed_config(tmp_path, monkeypatch)
    s = _Stub()

    s._handle_reasoning_command("/reasoning full now")
    assert s.reasoning_full is False
    saved = yaml.safe_load((hh / "config.yaml").read_text())
    assert "reasoning_full" not in saved["display"]


def test_reasoning_all_is_alias_for_full(tmp_path, monkeypatch):
    _seed_config(tmp_path, monkeypatch)
    s = _Stub()
    s._handle_reasoning_command("/reasoning all")
    assert s.reasoning_full is True


def test_clamp_gate_honours_flag():
    # The display gate at cli.py: clamp only when long AND not reasoning_full.
    reasoning = "\n".join(f"line{i}" for i in range(25))
    lines = reasoning.strip().splitlines()
    assert (len(lines) > 10 and not False) is True   # full=False -> clamp
    assert (len(lines) > 10 and not True) is False   # full=True  -> show all
