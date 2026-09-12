"""Tests for the --hindsight-bank CLI flag (root/chat dual position) and the
env bridge that carries it into the memory provider pre-dispatch.

Uses generic bank ids (acme/lead) — no real projects or user data.
"""

import os
import types

import pytest

from hermes_cli._parser import build_top_level_parser


def _parse(argv):
    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=lambda args: None)
    return parser.parse_args(argv)


class TestHindsightBankFlagParsing:
    def test_root_position_before_command(self):
        args = _parse(["--hindsight-bank", "acme", "chat"])
        assert args.hindsight_bank == "acme"

    def test_subcommand_position_after_chat(self):
        args = _parse(["chat", "--hindsight-bank", "team"])
        assert args.hindsight_bank == "team"

    def test_oneshot_pairing_at_root(self):
        args = _parse(["-z", "prompt", "--hindsight-bank", "acme"])
        assert args.hindsight_bank == "acme"

    def test_absent_flag_defaults_to_none(self):
        args = _parse(["chat"])
        assert args.hindsight_bank is None


class TestHindsightBankOverrideBridge:
    def _make_args(self, **kwargs):
        return types.SimpleNamespace(**kwargs)

    def test_flag_sets_internal_env_var(self, monkeypatch):
        from hermes_cli.main import _apply_hindsight_bank_override
        monkeypatch.delenv("HERMES_HINDSIGHT_BANK_OVERRIDE", raising=False)
        _apply_hindsight_bank_override(self._make_args(hindsight_bank="acme"))
        assert os.environ["HERMES_HINDSIGHT_BANK_OVERRIDE"] == "acme"

    def test_no_flag_leaves_env_unset(self, monkeypatch):
        from hermes_cli.main import _apply_hindsight_bank_override
        monkeypatch.delenv("HERMES_HINDSIGHT_BANK_OVERRIDE", raising=False)
        _apply_hindsight_bank_override(self._make_args(hindsight_bank=None))
        assert "HERMES_HINDSIGHT_BANK_OVERRIDE" not in os.environ

    def test_whitespace_flag_stripped(self, monkeypatch):
        from hermes_cli.main import _apply_hindsight_bank_override
        monkeypatch.delenv("HERMES_HINDSIGHT_BANK_OVERRIDE", raising=False)
        _apply_hindsight_bank_override(self._make_args(hindsight_bank="  acme  "))
        assert os.environ["HERMES_HINDSIGHT_BANK_OVERRIDE"] == "acme"

    def test_prepare_agent_startup_calls_bridge(self, monkeypatch):
        """The shared pre-dispatch hook must apply the override so both
        chat and one-shot paths carry the flag."""
        from hermes_cli import main as main_mod
        calls = []
        monkeypatch.setattr(
            main_mod,
            "_apply_hindsight_bank_override",
            lambda args: calls.append(getattr(args, "hindsight_bank", None)),
        )
        args = self._make_args(
            yolo=False, command="chat",
            hindsight_bank="acme",
        )
        monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda args: None)
        monkeypatch.setattr(main_mod, "_AGENT_SUBCOMMANDS", {})
        monkeypatch.setattr(main_mod, "_AGENT_COMMANDS", set())
        main_mod._prepare_agent_startup(args)
        assert calls == ["acme"]