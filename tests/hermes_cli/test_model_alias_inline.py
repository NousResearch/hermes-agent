"""Tests for persistent /model aliases and one-turn $alias routing."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


class _FakeModule(ModuleType):
    """A module that returns a MagicMock for any missing attribute."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock


# prompt_toolkit may not be installed in test environments.  Install
# fake module stubs that auto-create attributes so `from prompt_toolkit.X
# import Y` resolves.
_pt = _FakeModule("prompt_toolkit")
_pt.__path__ = []  # package marker
for _sub in (
    "history",
    "styles",
    "formatted_text",
    "layout",
    "layout.processors",
    "layout.dimension",
    "layout.menus",
    "key_binding",
    "key_binding.key_processor",
    "application",
    "completion",
    "shortcuts",
    "filters",
    "keys",
    "lexers",
    "document",
    "patch_stdout",
    "widgets",
    "utils",
    "enums",
    "output",
    "input",
    "renderer",
    "buffer",
    "selection",
    "clipboard",
    "auto_suggest",
    "validation",
    "search",
    "mouse",
    "cache",
    "data_structures",
    "eventloop",
    "cursor_shapes",
):
    _sub_mod = _FakeModule(f"prompt_toolkit.{_sub}")
    sys.modules.setdefault(f"prompt_toolkit.{_sub}", _sub_mod)
sys.modules.setdefault("prompt_toolkit", _pt)


def _make_route_cli():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.model = "gpt-5.5"
    cli.provider = "openai-codex"
    cli.api_key = "session-key"
    cli.base_url = "https://chatgpt.com/backend-api/codex"
    cli.api_mode = "chat_completions"
    cli.acp_command = None
    cli.acp_args = []
    cli._credential_pool = None
    cli.service_tier = None
    return cli


def test_parse_inline_model_alias_invocation_strips_prefix():
    from hermes_cli.model_switch import parse_inline_model_alias_invocation

    invocation = parse_inline_model_alias_invocation("$dsr analyze this")

    assert invocation is not None
    assert invocation.alias == "dsr"
    assert invocation.prompt == "analyze this"


def test_parse_inline_model_alias_invocation_allows_literal_escaped_dollar():
    from hermes_cli.model_switch import parse_inline_model_alias_invocation

    assert parse_inline_model_alias_invocation(r"\$dsr keep the dollar") is None


def test_save_model_alias_persists_model_mapping_without_secrets(monkeypatch):
    import hermes_cli.model_switch as ms

    captured = {}
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"model": {"default": "old"}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: captured.update(cfg))
    ms.DIRECT_ALIASES.clear()

    saved = ms.save_model_alias(
        "DsR",
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
    )

    assert saved == ms.DirectAlias(
        model="deepseek-v4-pro",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
    )
    assert captured["model_aliases"]["dsr"] == {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com/v1",
    }
    assert "api_key" not in captured["model_aliases"]["dsr"]
    assert "token" not in captured["model_aliases"]["dsr"]
    assert ms.DIRECT_ALIASES["dsr"] == saved


def test_remove_model_alias_deletes_mapping_and_cache(monkeypatch):
    import hermes_cli.model_switch as ms

    captured = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "model_aliases": {
                "dsr": {"provider": "deepseek", "model": "deepseek-v4-pro"},
                "gpt55": {"provider": "openai-codex", "model": "gpt-5.5"},
            }
        },
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: captured.update(cfg))
    ms.DIRECT_ALIASES.clear()
    ms.DIRECT_ALIASES["dsr"] = ms.DirectAlias("deepseek-v4-pro", "deepseek", "")

    assert ms.remove_model_alias("dsr") is True

    assert "dsr" not in captured["model_aliases"]
    assert "gpt55" in captured["model_aliases"]
    assert "dsr" not in ms.DIRECT_ALIASES


def test_resolve_turn_agent_config_uses_alias_for_one_turn_without_mutating_session(monkeypatch):
    from hermes_cli.model_switch import ModelSwitchResult

    cli = _make_route_cli()

    def fake_switch_model(**kwargs):
        assert kwargs["raw_input"] == "dsr"
        assert kwargs["current_provider"] == "openai-codex"
        assert kwargs["current_model"] == "gpt-5.5"
        return ModelSwitchResult(
            success=True,
            new_model="deepseek-v4-pro",
            target_provider="deepseek",
            api_key="resolved-key",
            base_url="https://api.deepseek.com/v1",
            api_mode="chat_completions",
            provider_label="DeepSeek",
            resolved_via_alias="dsr",
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)

    route = cli._resolve_turn_agent_config("$dsr analyze this")

    assert route["model"] == "deepseek-v4-pro"
    assert route["runtime"]["provider"] == "deepseek"
    assert route["runtime"]["base_url"] == "https://api.deepseek.com/v1"
    assert route["clean_text"] == "analyze this"
    assert route["model_alias"] == "dsr"
    assert cli.model == "gpt-5.5"
    assert cli.provider == "openai-codex"


def test_resolve_turn_agent_config_unknown_alias_fails_closed(monkeypatch):
    from hermes_cli.model_switch import ModelSwitchResult, ModelAliasError

    cli = _make_route_cli()
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: ModelSwitchResult(
            success=False,
            error_message="Unknown model alias 'missing'",
        ),
    )

    with pytest.raises(ModelAliasError, match="Unknown model alias"):
        cli._resolve_turn_agent_config("$missing hello")


def test_model_alias_add_without_model_saves_current_route(monkeypatch):
    from cli import HermesCLI
    from hermes_cli.model_switch import DirectAlias

    cli = HermesCLI.__new__(HermesCLI)
    cli.model = "deepseek-v4-pro"
    cli.provider = "deepseek"
    cli.base_url = "https://api.deepseek.com/v1"
    cli.api_key = "secret-key-that-must-not-be-saved"
    cli.api_mode = "chat_completions"

    saved = []
    monkeypatch.setattr(
        "hermes_cli.model_switch.save_model_alias",
        lambda alias, provider, model, base_url="": saved.append((alias, provider, model, base_url))
        or DirectAlias(model=model, provider=provider, base_url=base_url),
    )
    monkeypatch.setattr("cli._cprint", lambda *args, **kwargs: None)

    cli._handle_model_switch("/model alias add dsr")

    assert saved == [("dsr", "deepseek", "deepseek-v4-pro", "https://api.deepseek.com/v1")]
