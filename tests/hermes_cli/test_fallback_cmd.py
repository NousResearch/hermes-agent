"""Tests for `hermes fallback` — chain reading, add/remove/clear, legacy migration."""
from __future__ import annotations

import io
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Shared fixture — isolate HERMES_HOME so save_config writes to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


def _write_config(home: Path, data: dict) -> None:
    config_path = home / ".hermes" / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _read_config(home: Path) -> dict:
    config_path = home / ".hermes" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# _read_chain / _write_chain
# ---------------------------------------------------------------------------

class TestReadChain:
    def test_returns_empty_list_when_unset(self):
        from hermes_cli.fallback_cmd import _read_chain
        assert _read_chain({}) == []

    def test_reads_new_list_format(self):
        from hermes_cli.fallback_cmd import _read_chain
        cfg = {
            "fallback_providers": [
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
                {"provider": "nous", "model": "Hermes-4-Llama-3.1-405B"},
            ]
        }
        assert _read_chain(cfg) == [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            {"provider": "nous", "model": "Hermes-4-Llama-3.1-405B"},
        ]

    def test_migrates_legacy_single_dict(self):
        from hermes_cli.fallback_cmd import _read_chain
        cfg = {"fallback_model": {"provider": "openrouter", "model": "gpt-5.4"}}
        assert _read_chain(cfg) == [{"provider": "openrouter", "model": "gpt-5.4"}]

    def test_skips_incomplete_entries(self):
        from hermes_cli.fallback_cmd import _read_chain
        cfg = {
            "fallback_providers": [
                {"provider": "openrouter"},            # missing model
                {"model": "gpt-5.4"},                  # missing provider
                {"provider": "nous", "model": "foo"},  # valid
                "not-a-dict",                          # noise
            ]
        }
        assert _read_chain(cfg) == [{"provider": "nous", "model": "foo"}]

    def test_returns_copies_not_aliases(self):
        from hermes_cli.fallback_cmd import _read_chain
        cfg = {"fallback_providers": [{"provider": "nous", "model": "foo"}]}
        result = _read_chain(cfg)
        result[0]["provider"] = "mutated"
        assert cfg["fallback_providers"][0]["provider"] == "nous"


# ---------------------------------------------------------------------------
# _extract_fallback_from_model_cfg
# ---------------------------------------------------------------------------

class TestExtractFallback:
    def test_extracts_from_default_field(self):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        model_cfg = {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"}
        assert _extract_fallback_from_model_cfg(model_cfg) == {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4.6",
        }

    def test_extracts_optional_base_url_and_api_mode(self):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        model_cfg = {
            "provider": "custom",
            "default": "local-model",
            "base_url": "http://localhost:11434/v1",
            "api_mode": "chat_completions",
        }
        assert _extract_fallback_from_model_cfg(model_cfg) == {
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://localhost:11434/v1",
            "api_mode": "chat_completions",
        }

    def test_returns_none_without_provider(self):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        assert _extract_fallback_from_model_cfg({"default": "foo"}) is None

    def test_returns_none_without_model(self):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        assert _extract_fallback_from_model_cfg({"provider": "openrouter"}) is None

    def test_returns_none_for_non_dict(self):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        assert _extract_fallback_from_model_cfg("plain-string") is None
        assert _extract_fallback_from_model_cfg(None) is None


# ---------------------------------------------------------------------------
# cmd_fallback_list
# ---------------------------------------------------------------------------

class TestListCommand:
    def test_list_empty(self, isolated_home, capsys):
        _write_config(isolated_home, {})
        from hermes_cli.fallback_cmd import cmd_fallback_list
        cmd_fallback_list(types.SimpleNamespace())
        out = capsys.readouterr().out
        assert "No fallback providers configured" in out
        assert "hermes fallback add" in out

    def test_list_with_entries(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"},
            "fallback_providers": [
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
                {"provider": "nous", "model": "Hermes-4"},
            ],
        })
        from hermes_cli.fallback_cmd import cmd_fallback_list
        cmd_fallback_list(types.SimpleNamespace())
        out = capsys.readouterr().out
        assert "Fallback chain (2 entries)" in out
        assert "anthropic/claude-sonnet-4.6" in out
        assert "Hermes-4" in out
        # Primary should be shown too
        assert "claude-sonnet-4-6" in out

    def test_list_migrates_legacy_for_display(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "fallback_model": {"provider": "openrouter", "model": "gpt-5.4"},
        })
        from hermes_cli.fallback_cmd import cmd_fallback_list
        cmd_fallback_list(types.SimpleNamespace())
        out = capsys.readouterr().out
        assert "1 entry" in out
        assert "gpt-5.4" in out


# ---------------------------------------------------------------------------
# cmd_fallback_add — mock select_provider_and_model
# ---------------------------------------------------------------------------

class TestAddCommand:
    def test_add_appends_new_entry(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"},
        })

        def fake_picker(args=None):
            # Simulate what the real picker does: writes the selection to config["model"]
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = {
                "provider": "openrouter",
                "default": "anthropic/claude-sonnet-4.6",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
            save_config(cfg)

        with patch("hermes_cli.main.select_provider_and_model", side_effect=fake_picker), \
                patch("hermes_cli.main._require_tty"):
            from hermes_cli.fallback_cmd import cmd_fallback_add
            cmd_fallback_add(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        # Primary is preserved
        assert cfg["model"]["provider"] == "anthropic"
        assert cfg["model"]["default"] == "claude-sonnet-4-6"
        # Fallback was appended
        assert cfg["fallback_providers"] == [
            {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4.6",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
        ]
        out = capsys.readouterr().out
        assert "Added fallback" in out

    def test_add_rejects_duplicate(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"},
            "fallback_providers": [
                {"provider": "openrouter", "model": "gpt-5.4"},
            ],
        })

        def fake_picker(args=None):
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = {"provider": "openrouter", "default": "gpt-5.4"}
            save_config(cfg)

        with patch("hermes_cli.main.select_provider_and_model", side_effect=fake_picker), \
                patch("hermes_cli.main._require_tty"):
            from hermes_cli.fallback_cmd import cmd_fallback_add
            cmd_fallback_add(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        # Should still have exactly one entry
        assert len(cfg["fallback_providers"]) == 1
        out = capsys.readouterr().out
        assert "already in the fallback chain" in out

    def test_add_rejects_same_as_primary(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "model": {"provider": "openrouter", "default": "gpt-5.4"},
        })

        def fake_picker(args=None):
            # User picks the same thing that's already the primary
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = {"provider": "openrouter", "default": "gpt-5.4"}
            save_config(cfg)

        with patch("hermes_cli.main.select_provider_and_model", side_effect=fake_picker), \
                patch("hermes_cli.main._require_tty"):
            from hermes_cli.fallback_cmd import cmd_fallback_add
            cmd_fallback_add(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        assert "fallback_providers" not in cfg or cfg["fallback_providers"] == []
        out = capsys.readouterr().out
        assert "matches the current primary" in out

    def test_add_preserves_primary_when_picker_changes_it(self, isolated_home):
        """The picker mutates config["model"]; fallback_add must restore the primary."""
        _write_config(isolated_home, {
            "model": {
                "provider": "anthropic",
                "default": "claude-sonnet-4-6",
                "base_url": "https://api.anthropic.com",
                "api_mode": "anthropic_messages",
            },
        })

        def fake_picker(args=None):
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = {
                "provider": "openrouter",
                "default": "anthropic/claude-sonnet-4.6",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
            save_config(cfg)

        with patch("hermes_cli.main.select_provider_and_model", side_effect=fake_picker), \
                patch("hermes_cli.main._require_tty"):
            from hermes_cli.fallback_cmd import cmd_fallback_add
            cmd_fallback_add(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        # Primary exactly as it was
        assert cfg["model"]["provider"] == "anthropic"
        assert cfg["model"]["default"] == "claude-sonnet-4-6"
        assert cfg["model"]["base_url"] == "https://api.anthropic.com"
        assert cfg["model"]["api_mode"] == "anthropic_messages"
        # Fallback added
        assert len(cfg["fallback_providers"]) == 1
        assert cfg["fallback_providers"][0]["provider"] == "openrouter"

    def test_add_noop_when_picker_cancelled(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"},
        })

        def fake_picker(args=None):
            # User cancelled — no change to config
            pass

        with patch("hermes_cli.main.select_provider_and_model", side_effect=fake_picker), \
                patch("hermes_cli.main._require_tty"):
            from hermes_cli.fallback_cmd import cmd_fallback_add
            cmd_fallback_add(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        assert "fallback_providers" not in cfg or cfg["fallback_providers"] == []
        out = capsys.readouterr().out
        # Either "No fallback added" (picker fully cancelled) or "matches the current primary"
        # (picker left config untouched) — both indicate a non-add outcome.
        assert ("No fallback added" in out) or ("matches the current primary" in out)

    def test_add_noop_when_picker_clears_model(self, isolated_home, capsys):
        """Simulate picker explicitly clearing model.default (unusual but possible)."""
        _write_config(isolated_home, {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"},
        })

        def fake_picker(args=None):
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = {"provider": "", "default": ""}
            save_config(cfg)

        with patch("hermes_cli.main.select_provider_and_model", side_effect=fake_picker), \
                patch("hermes_cli.main._require_tty"):
            from hermes_cli.fallback_cmd import cmd_fallback_add
            cmd_fallback_add(types.SimpleNamespace())

        out = capsys.readouterr().out
        assert "No fallback added" in out


# ---------------------------------------------------------------------------
# cmd_fallback_remove
# ---------------------------------------------------------------------------

class TestRemoveCommand:
    def test_remove_empty_chain(self, isolated_home, capsys):
        _write_config(isolated_home, {})
        from hermes_cli.fallback_cmd import cmd_fallback_remove
        cmd_fallback_remove(types.SimpleNamespace())
        out = capsys.readouterr().out
        assert "nothing to remove" in out

    def test_remove_selected_entry(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "fallback_providers": [
                {"provider": "openrouter", "model": "gpt-5.4"},
                {"provider": "nous", "model": "Hermes-4"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
        })

        # Picker returns index 1 (the middle entry, "nous / Hermes-4")
        with patch("hermes_cli.setup_wizard._curses_prompt_choice", return_value=1):
            from hermes_cli.fallback_cmd import cmd_fallback_remove
            cmd_fallback_remove(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        assert cfg["fallback_providers"] == [
            {"provider": "openrouter", "model": "gpt-5.4"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        ]
        out = capsys.readouterr().out
        assert "Removed fallback" in out
        assert "Hermes-4" in out

    def test_remove_cancel_keeps_chain(self, isolated_home):
        _write_config(isolated_home, {
            "fallback_providers": [
                {"provider": "openrouter", "model": "gpt-5.4"},
            ],
        })

        # Cancel = last item (index == len(chain) == 1 in our menu)
        with patch("hermes_cli.setup_wizard._curses_prompt_choice", return_value=1):
            from hermes_cli.fallback_cmd import cmd_fallback_remove
            cmd_fallback_remove(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        assert len(cfg["fallback_providers"]) == 1


# ---------------------------------------------------------------------------
# cmd_fallback_clear
# ---------------------------------------------------------------------------

class TestClearCommand:
    def test_clear_empty_chain(self, isolated_home, capsys):
        _write_config(isolated_home, {})
        from hermes_cli.fallback_cmd import cmd_fallback_clear
        cmd_fallback_clear(types.SimpleNamespace())
        out = capsys.readouterr().out
        assert "nothing to clear" in out

    def test_clear_with_confirmation(self, isolated_home, capsys, monkeypatch):
        _write_config(isolated_home, {
            "fallback_providers": [
                {"provider": "openrouter", "model": "gpt-5.4"},
                {"provider": "nous", "model": "Hermes-4"},
            ],
        })
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "y")
        from hermes_cli.fallback_cmd import cmd_fallback_clear
        cmd_fallback_clear(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        assert cfg.get("fallback_providers") == []
        out = capsys.readouterr().out
        assert "Fallback chain cleared" in out

    def test_clear_cancelled(self, isolated_home, monkeypatch):
        _write_config(isolated_home, {
            "fallback_providers": [{"provider": "openrouter", "model": "gpt-5.4"}],
        })
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "n")
        from hermes_cli.fallback_cmd import cmd_fallback_clear
        cmd_fallback_clear(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        assert len(cfg["fallback_providers"]) == 1


# ---------------------------------------------------------------------------
# cmd_fallback dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_no_subcommand_lists(self, isolated_home, capsys):
        _write_config(isolated_home, {})
        from hermes_cli.fallback_cmd import cmd_fallback
        cmd_fallback(types.SimpleNamespace(fallback_command=None))
        out = capsys.readouterr().out
        assert "No fallback providers configured" in out

    def test_list_alias(self, isolated_home, capsys):
        _write_config(isolated_home, {})
        from hermes_cli.fallback_cmd import cmd_fallback
        cmd_fallback(types.SimpleNamespace(fallback_command="ls"))
        out = capsys.readouterr().out
        assert "No fallback providers configured" in out

    def test_remove_alias(self, isolated_home, capsys):
        _write_config(isolated_home, {})
        from hermes_cli.fallback_cmd import cmd_fallback
        cmd_fallback(types.SimpleNamespace(fallback_command="rm"))
        out = capsys.readouterr().out
        assert "nothing to remove" in out

    def test_unknown_subcommand_exits(self, isolated_home):
        _write_config(isolated_home, {})
        from hermes_cli.fallback_cmd import cmd_fallback
        with pytest.raises(SystemExit):
            cmd_fallback(types.SimpleNamespace(fallback_command="nope"))


# ---------------------------------------------------------------------------
# argparse wiring — verify the subparser is registered
# ---------------------------------------------------------------------------

class TestArgparseWiring:
    """Verify `hermes fallback` is wired into main.py's argparse tree.

    main() builds the parser inline, so we invoke main([...]) via subprocess
    with --help to introspect registered subcommands without side effects.
    """

    def test_fallback_help_lists_subcommands(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "fallback", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # --help exits 0
        assert result.returncode == 0, f"stderr: {result.stderr}"
        out = result.stdout + result.stderr
        # All four subcommands should appear in help
        assert "list" in out
        assert "add" in out
        assert "remove" in out
        assert "clear" in out
