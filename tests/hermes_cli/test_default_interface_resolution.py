"""Tests for the configurable default interface (cli vs tui).

`hermes` launches the classic prompt_toolkit REPL by default, but users can
flip ``display.interface: tui`` in config.yaml to make the modern Ink TUI the
default for bare ``hermes`` / ``hermes chat``. Explicit flags always win:

    --cli                forces the classic REPL (highest precedence)
    --tui / HERMES_TUI=1 forces the TUI
    display.interface    the configured default
    (unset)              classic REPL

These tests pin that precedence at every layer that makes the decision:

  * ``_resolve_use_tui(args)``  — the canonical args-aware resolver used by
    ``cmd_chat`` and the Termux fast-TUI path.
  * ``_wants_tui_early(argv)``  — the dependency-free early resolver used by
    mouse-residue suppression and the Termux fast paths, before argparse and
    ``hermes_cli.config`` are importable.
  * the argument parser   — both ``--cli`` and ``--tui`` parse at the top
    level and under the ``chat`` subcommand and are relaunch-inherited.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import main as m


@pytest.fixture(autouse=True)
def _reset_early_cache(monkeypatch):
    # The early resolver memoizes the config read; clear it so each test sees
    # a fresh value, and make sure no stray HERMES_TUI leaks in.
    monkeypatch.setattr(m, "_EARLY_INTERFACE_CACHE", None)
    monkeypatch.delenv("HERMES_TUI", raising=False)
    yield
    monkeypatch.setattr(m, "_EARLY_INTERFACE_CACHE", None)


def _args(**kw):
    kw.setdefault("cli", False)
    kw.setdefault("tui", False)
    return SimpleNamespace(**kw)


def _patch_config(monkeypatch, interface):
    import hermes_cli.config as cfg

    monkeypatch.setattr(
        cfg, "load_config", lambda: {"display": {"interface": interface}}
    )


# ---------------------------------------------------------------------------
# _resolve_use_tui — args-aware resolver
# ---------------------------------------------------------------------------
class TestResolveUseTui:
    def test_cli_flag_beats_config_tui(self, monkeypatch):
        _patch_config(monkeypatch, "tui")
        assert m._resolve_use_tui(_args(cli=True)) is False

    def test_cli_flag_beats_tui_flag_and_env(self, monkeypatch):
        _patch_config(monkeypatch, "tui")
        monkeypatch.setenv("HERMES_TUI", "1")
        assert m._resolve_use_tui(_args(cli=True, tui=True)) is False

    def test_tui_flag_beats_config_cli(self, monkeypatch):
        _patch_config(monkeypatch, "cli")
        assert m._resolve_use_tui(_args(tui=True)) is True

    def test_env_beats_config_cli(self, monkeypatch):
        _patch_config(monkeypatch, "cli")
        monkeypatch.setenv("HERMES_TUI", "1")
        assert m._resolve_use_tui(_args()) is True

    def test_config_tui_with_no_flags(self, monkeypatch):
        _patch_config(monkeypatch, "tui")
        assert m._resolve_use_tui(_args()) is True

    def test_config_cli_is_default(self, monkeypatch):
        _patch_config(monkeypatch, "cli")
        assert m._resolve_use_tui(_args()) is False

    def test_interface_value_is_case_insensitive(self, monkeypatch):
        _patch_config(monkeypatch, "TUI")
        assert m._resolve_use_tui(_args()) is True

    def test_load_config_failure_falls_back_to_cli(self, monkeypatch):
        import hermes_cli.config as cfg

        def boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(cfg, "load_config", boom)
        assert m._resolve_use_tui(_args()) is False


# ---------------------------------------------------------------------------
# _wants_tui_early — dependency-free early resolver
# ---------------------------------------------------------------------------
class TestWantsTuiEarly:
    @pytest.fixture
    def home_with_interface(self, tmp_path, monkeypatch):
        def _make(interface):
            (tmp_path / "config.yaml").write_text(
                f"display:\n  interface: {interface}\n"
            )
            monkeypatch.setenv("HERMES_HOME", str(tmp_path))
            monkeypatch.setattr(m, "_EARLY_INTERFACE_CACHE", None)

        return _make

    def test_config_tui_bare_argv(self, home_with_interface):
        home_with_interface("tui")
        assert m._wants_tui_early([]) is True

    def test_cli_flag_overrides_config_tui(self, home_with_interface):
        home_with_interface("tui")
        assert m._wants_tui_early(["--cli"]) is False

    def test_tui_flag_with_config_cli(self, home_with_interface):
        home_with_interface("cli")
        assert m._wants_tui_early(["--tui"]) is True

    def test_env_with_config_cli(self, home_with_interface, monkeypatch):
        home_with_interface("cli")
        monkeypatch.setenv("HERMES_TUI", "1")
        assert m._wants_tui_early([]) is True

    def test_config_cli_bare_argv(self, home_with_interface):
        home_with_interface("cli")
        assert m._wants_tui_early([]) is False

    def test_missing_config_defaults_to_cli(self, tmp_path, monkeypatch):
        # HERMES_HOME points at an empty dir — no config.yaml.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(m, "_EARLY_INTERFACE_CACHE", None)
        assert m._wants_tui_early([]) is False

    def test_unreadable_config_defaults_to_cli(self, tmp_path, monkeypatch):
        # Garbage YAML must not crash the hot path; falls back to cli.
        (tmp_path / "config.yaml").write_text("this: : : not valid yaml\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(m, "_EARLY_INTERFACE_CACHE", None)
        assert m._wants_tui_early([]) is False


# ---------------------------------------------------------------------------
# argument parser — flags exist at both levels and are relaunch-inherited
# ---------------------------------------------------------------------------
class TestParserFlags:
    def _parser(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat = build_top_level_parser()
        return parser

    def test_top_level_cli_flag(self):
        args = self._parser().parse_args(["--cli"])
        assert args.cli is True and args.tui is False

    def test_top_level_tui_flag(self):
        args = self._parser().parse_args(["--tui"])
        assert args.tui is True and args.cli is False

    def test_chat_subcommand_cli_flag(self):
        args = self._parser().parse_args(["chat", "--cli"])
        assert args.cli is True

    def test_chat_subcommand_tui_flag(self):
        args = self._parser().parse_args(["chat", "--tui"])
        assert args.tui is True

    def test_cli_and_tui_are_relaunch_inherited(self):
        from hermes_cli.relaunch import _INHERITED_FLAGS_TABLE

        inherited = {flag for flag, _takes_value in _INHERITED_FLAGS_TABLE}
        assert "--cli" in inherited
        assert "--tui" in inherited


# ---------------------------------------------------------------------------
# config default — shipped default preserves classic behavior
# ---------------------------------------------------------------------------
def test_default_config_interface_is_cli():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["display"]["interface"] == "cli"
