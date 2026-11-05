"""Tests for `hermes fallback` — chain reading, add/remove/clear, legacy migration."""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

# ---------------------------------------------------------------------------
# Shared fixture — isolate HERMES_HOME so save_config writes to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
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

    def test_returns_none_without_model(self):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        assert _extract_fallback_from_model_cfg({"provider": "openrouter"}) is None


class TestExtractFallbackNamedIdentity:
    """A legacy list-form ``custom_providers:`` entry loses its name in the picker (no
    ``provider_key`` → bare ``custom``), which then also loses the key: bare custom looks for
    ``OPENAI_API_KEY`` or a base_url matching the primary's and finds neither. The entry must keep
    the configured identity so the key resolves through that provider's own entry."""

    def _home(self, isolated_home, providers):
        _write_config(isolated_home, {
            "model": {"provider": "custom", "default": "primary-model",
                      "base_url": "https://primary.example/v1", "api_key": "sk-primary"},
            "custom_providers": providers,
        })

    def test_bare_custom_pick_recovers_the_configured_name(self, isolated_home):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        self._home(isolated_home, [{
            "name": "hetune", "base_url": "https://cf.example/v1",
            "api_key": "sk-hetune", "model": "deepseek-v4.1-flash",
        }])
        assert _extract_fallback_from_model_cfg({
            "provider": "custom", "default": "deepseek-v4.1-flash",
            "base_url": "https://cf.example/v1", "api_key": "sk-hetune",
        }) == {
            "provider": "custom:hetune", "model": "deepseek-v4.1-flash",
            "base_url": "https://cf.example/v1",
        }

    def test_route_match_ignores_trailing_slash_and_case(self, isolated_home):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        self._home(isolated_home, [{"name": "Hetune", "base_url": "https://cf.example/v1"}])
        assert _extract_fallback_from_model_cfg({
            "provider": "custom", "default": "m", "base_url": "https://CF.example/v1/",
        })["provider"] == "custom:hetune"

    def test_dict_form_providers_entry_uses_its_config_key(self, isolated_home):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        _write_config(isolated_home, {
            "model": {"provider": "custom", "default": "primary-model",
                      "base_url": "https://primary.example/v1"},
            "providers": {"litellm": {"name": "LiteLLM Proxy", "base_url": "https://litellm.example/v1"}},
        })
        assert _extract_fallback_from_model_cfg({
            "provider": "custom", "default": "m", "base_url": "https://litellm.example/v1",
        })["provider"] == "custom:litellm"

    def test_ambiguous_route_keeps_bare_custom(self, isolated_home):
        """Two endpoints on one base_url: guessing one would send its key to the other's config."""
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        self._home(isolated_home, [
            {"name": "alpha", "base_url": "https://shared.example/v1", "api_key": "sk-a"},
            {"name": "beta", "base_url": "https://shared.example/v1", "api_key": "sk-b"},
        ])
        assert _extract_fallback_from_model_cfg({
            "provider": "custom", "default": "m", "base_url": "https://shared.example/v1",
        })["provider"] == "custom"

    def test_unmatched_or_missing_base_url_keeps_bare_custom(self, isolated_home):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        self._home(isolated_home, [{"name": "hetune", "base_url": "https://cf.example/v1"}])
        assert _extract_fallback_from_model_cfg(
            {"provider": "custom", "default": "m", "base_url": "https://other.example/v1"}
        )["provider"] == "custom"
        assert _extract_fallback_from_model_cfg(
            {"provider": "custom", "default": "m"}
        )["provider"] == "custom"

    def test_named_and_builtin_providers_are_not_rewritten(self, isolated_home):
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        self._home(isolated_home, [{"name": "hetune", "base_url": "https://cf.example/v1"}])
        assert _extract_fallback_from_model_cfg(
            {"provider": "custom:wb2api", "default": "m", "base_url": "https://cf.example/v1"}
        )["provider"] == "custom:wb2api"
        assert _extract_fallback_from_model_cfg(
            {"provider": "openrouter", "default": "m"}
        )["provider"] == "openrouter"

    def test_generated_entry_authenticates_with_the_configured_key(self, isolated_home):
        """End to end: bare custom sends the placeholder, the named identity resolves the key."""
        from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg
        from agent.auxiliary_client import resolve_provider_client
        self._home(isolated_home, [{
            "name": "hetune", "base_url": "https://cf.example/v1",
            "api_key": "sk-configured", "model": "m",
        }])
        entry = _extract_fallback_from_model_cfg({
            "provider": "custom", "default": "m", "base_url": "https://cf.example/v1",
        })
        assert entry is not None
        # Exactly how agent/route_binding.py binds a chain entry: the entry's own credential hint
        # (absent here) plus the entry's provider identity and base_url.
        client, _ = resolve_provider_client(
            entry["provider"], model=entry["model"], raw_codex=True,
            explicit_base_url=entry["base_url"], explicit_api_key=None)
        assert str(getattr(client, "api_key", "")) == "sk-configured"
        # The bare class the picker would otherwise store falls back to the placeholder.
        bare, _ = resolve_provider_client(
            "custom", model="m", raw_codex=True,
            explicit_base_url="https://cf.example/v1", explicit_api_key=None)
        assert str(getattr(bare, "api_key", "")) == "no-key-required"


# ---------------------------------------------------------------------------
# cmd_fallback_list
# ---------------------------------------------------------------------------

class TestListCommand:

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
        assert "anthropic/claude-sonnet-4.6" in out
        assert "Hermes-4" in out
        # Primary should be shown too
        assert "claude-sonnet-4-6" in out

# ---------------------------------------------------------------------------
# cmd_fallback_add — mock select_provider_and_model
# ---------------------------------------------------------------------------

class TestAddCommand:
    def test_add_appends_new_entry(self, isolated_home):
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

    def test_add_rejects_same_as_primary(self, isolated_home):
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
            cfg["custom_providers"] = [
                {
                    "name": "Picker-created endpoint",
                    "base_url": "https://picker.example/v1",
                    "model": "picker-model",
                }
            ]
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
        assert cfg["custom_providers"] == [
            {
                "name": "Picker-created endpoint",
                "base_url": "https://picker.example/v1",
                "model": "picker-model",
            }
        ]

    def test_restore_preserves_absent_active_provider(self):
        from contextlib import nullcontext

        from hermes_cli import auth, fallback_cmd

        store = {"version": 1, "providers": {}}

        def save_auth(value):
            store.clear()
            store.update(value)

        with patch.object(auth, "_load_auth_store", lambda: dict(store)), patch.object(
            auth, "_save_auth_store", save_auth
        ), patch.object(auth, "_auth_store_lock", nullcontext):
            before = fallback_cmd._snapshot_auth_active_provider()
            fallback_cmd._restore_auth_active_provider(before)

        assert "active_provider" not in store

    @pytest.mark.parametrize("picker_error", [LookupError("picker failed"), KeyboardInterrupt()],
                             ids=["exception", "ctrl-c"])
    def test_picker_failure_restores_persisted_primary_without_masking_error(
        self, isolated_home, picker_error
    ):
        """An ordinary picker exception or a Ctrl+C mid-picker must leave config.yaml's
        ``model`` exactly as it was before ``fallback add`` started (base only handled SystemExit)."""
        from hermes_cli import fallback_cmd

        primary_model = {
            "provider": "anthropic",
            "default": "claude-sonnet-4-6",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        }
        _write_config(isolated_home, {"model": primary_model, "theme": "midnight"})

        def failing_picker(args=None):
            from hermes_cli.config import load_config, save_config

            cfg = load_config()
            cfg["model"] = {
                "provider": "openrouter",
                "default": "anthropic/claude-sonnet-4.6",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }
            save_config(cfg)
            raise picker_error

        with patch(
            "hermes_cli.main.select_provider_and_model",
            side_effect=failing_picker,
        ), patch("hermes_cli.main._require_tty"):
            with pytest.raises(type(picker_error)) as exc_info:
                fallback_cmd.cmd_fallback_add(types.SimpleNamespace())

        assert exc_info.value is picker_error
        persisted = _read_config(isolated_home)
        assert persisted["model"] == primary_model
        assert persisted["theme"] == "midnight"

# ---------------------------------------------------------------------------
# cmd_fallback_remove
# ---------------------------------------------------------------------------

class TestRemoveCommand:

    def test_remove_selected_entry(self, isolated_home):
        _write_config(isolated_home, {
            "fallback_providers": [
                {"provider": "openrouter", "model": "gpt-5.4"},
                {"provider": "nous", "model": "Hermes-4"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
        })

        # Picker returns index 1 (the middle entry, "nous / Hermes-4")
        with patch("hermes_cli.setup._curses_prompt_choice", return_value=1):
            from hermes_cli.fallback_cmd import cmd_fallback_remove
            cmd_fallback_remove(types.SimpleNamespace())

        cfg = _read_config(isolated_home)
        assert cfg["fallback_providers"] == [
            {"provider": "openrouter", "model": "gpt-5.4"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        ]

# ---------------------------------------------------------------------------
# cmd_fallback_clear
# ---------------------------------------------------------------------------

class TestClearCommand:

    def test_clear_with_confirmation(self, isolated_home, monkeypatch):
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

# ---------------------------------------------------------------------------
# cmd_fallback dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_unknown_subcommand_exits(self, isolated_home):
        _write_config(isolated_home, {})
        from hermes_cli.fallback_cmd import cmd_fallback
        with pytest.raises(SystemExit):
            cmd_fallback(types.SimpleNamespace(fallback_command="nope"))

# ---------------------------------------------------------------------------
# argparse wiring — verify the subparser is registered
# ---------------------------------------------------------------------------
