"""A saved ``model.api_mode`` belongs to the saved ``model.default`` only.

Providers whose profile picks the wire per model (``api_mode_for_model``) serve several wires behind one
endpoint. A ``model.api_mode`` saved for one pick (``/model --global``, the Desktop / dashboard main-model
assignment) used to be honored for every model of that provider. Repro on main with the published GitHub
Copilot plugin (GPT-5 on ``codex_responses``, everything else on ``chat_completions``): ``/model gpt-4.1
--global`` saved ``api_mode: chat_completions``, and every later GPT-5 target on copilot (delegation, cron,
auxiliary tasks, ``-m``) resolved to ``chat_completions``, which Responses-only models reject. Now runtime
resolution honors the saved mode only for the saved default, and a selection on such a provider no longer
pins ``api_mode`` at all (#96066). The fake profile below has the published plugin's shape (plus a Claude
wire, to cover the reverse direction); nothing here needs any other unreleased core change.
"""

import pytest

import hermes_yaml as yaml
from hermes_cli.model_switch import ModelSwitchResult


@pytest.fixture
def per_model_provider(monkeypatch):
    import hermes_cli.auth
    import providers
    from providers.base import ProviderProfile

    class PerModelWire(ProviderProfile):
        def api_mode_for_model(self, model, *, api_key=None):
            if model.startswith("claude"):
                return "anthropic_messages"
            if model.startswith("gpt-5"):
                return "codex_responses"
            return "chat_completions"

    providers.list_providers()  # discovery first, so the fake survives it
    providers.register_provider(PerModelWire(
        name="perwire", base_url="https://api.perwire.test/v1", api_mode="chat_completions",
        env_vars=("PERWIRE_API_KEY",)))
    monkeypatch.setenv("PERWIRE_API_KEY", "sk-fake")
    yield
    providers._REGISTRY.pop("perwire", None)
    providers._SOURCES.pop("perwire", None)
    providers._PROVIDER_LIST_CACHE = None
    hermes_cli.auth.PROVIDER_REGISTRY.pop("perwire", None)


def _write_model_block(tmp_path, monkeypatch, **model):
    from hermes_cli import config as cfg

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": model}), encoding="utf-8")
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return tmp_path / "config.yaml"


def _mode(target_model=None):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(requested="perwire", target_model=target_model)["api_mode"]


def _result(model, provider="perwire", api_mode="anthropic_messages", base_url="https://api.perwire.test/v1"):
    return ModelSwitchResult(success=True, new_model=model, target_provider=provider, api_key="sk-fake",
                             base_url=base_url, api_mode=api_mode, is_global=True)


def test_saved_chat_completions_does_not_pin_a_responses_only_target(per_model_provider, tmp_path, monkeypatch):
    """The repro: a ``chat_completions`` saved for a chat model used to send GPT-5 targets to Chat Completions."""
    _write_model_block(tmp_path, monkeypatch, provider="perwire", default="gpt-4.1", api_mode="chat_completions")
    assert _mode() == "chat_completions"
    assert _mode("gpt-5.9") == "codex_responses"


def test_an_explicit_api_mode_is_still_honored_for_the_saved_default(per_model_provider, tmp_path, monkeypatch):
    _write_model_block(tmp_path, monkeypatch, provider="perwire", default="claude-y", api_mode="chat_completions")
    assert _mode() == "chat_completions"
    assert _mode("claude-y") == "chat_completions"
    assert _mode("claude-z") == "anthropic_messages"  # another model: the profile decides


def test_global_pick_of_claude_then_a_gpt_target_resolves_gpt_wire(per_model_provider, tmp_path, monkeypatch):
    """End to end on disk: ``/model claude-y --global`` (persist_model_selection) leaves no api_mode, and a
    GPT target on the same provider afterwards resolves its own wire."""
    from hermes_cli import config as cfg
    from hermes_cli.model_switch import persist_model_selection

    path = _write_model_block(tmp_path, monkeypatch, provider="perwire", default="gpt-4.1",
                              api_mode="chat_completions")
    persist_model_selection(_result("claude-y"), path)
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
    assert saved["default"] == "claude-y" and "api_mode" not in saved
    assert _mode() == "anthropic_messages"
    assert _mode("gpt-5.9") == "codex_responses"


def test_selection_on_a_fixed_wire_provider_still_saves_api_mode():
    from hermes_cli.model_switch import model_selection_config_updates

    result = _result("MiniMax-M3", provider="custom:minimax", api_mode="anthropic_messages",
                     base_url="https://api.minimax.io/anthropic")
    assert model_selection_config_updates(result, {})["api_mode"] == "anthropic_messages"


def test_a_selection_never_calls_api_mode_for_model(per_model_provider, tmp_path, monkeypatch):
    """Deciding that a selection saves no api_mode is pure: the profile's hook (which may read config or
    secrets) never runs on a selection path, even inside ``_CONFIG_LOCK``."""
    import providers
    from hermes_cli.model_switch import model_selection_config_updates, persist_model_selection

    profile = providers.get_provider_profile("perwire")
    assert profile is not None
    calls = []
    monkeypatch.setattr(type(profile), "api_mode_for_model",
                        lambda self, model, *, api_key=None: calls.append(model))
    assert model_selection_config_updates(_result("other-z"), {"provider": "perwire"})["api_mode"] is None
    path = _write_model_block(tmp_path, monkeypatch, provider="perwire", default="gpt-4.1", api_mode="chat_completions")
    persist_model_selection(_result("claude-y"), path)
    assert "api_mode" not in yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
    assert calls == []


def test_opencode_go_selection_saves_no_api_mode_and_still_routes_per_model(tmp_path, monkeypatch):
    """OpenCode Zen/Go (and Nous) also derive the wire per model, so their selections stop saving api_mode
    too; their runtime already ignored it and re-derives the wire for each model (#16878)."""
    from hermes_cli.model_switch import model_selection_config_updates
    from hermes_cli.runtime_provider import resolve_runtime_provider

    result = _result("minimax-m2.7", provider="opencode-go", base_url="https://opencode.ai/zen/go")
    assert model_selection_config_updates(result, {"provider": "opencode-go"})["api_mode"] is None
    _write_model_block(tmp_path, monkeypatch, provider="opencode-go", default="minimax-m2.7")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)
    assert resolve_runtime_provider(requested="opencode-go")["api_mode"] == "anthropic_messages"
    assert resolve_runtime_provider(requested="opencode-go", target_model="gpt-5.9")["api_mode"] == "codex_responses"
    assert resolve_runtime_provider(requested="opencode-go", target_model="glm-5.3-flash")["api_mode"] == "chat_completions"


def test_session_info_derives_api_mode_for_the_sessions_model(per_model_provider):
    """A session with no live agent on a per-model provider shows the wire its next turn derives for the
    session's own model, not the profile default's."""
    from types import SimpleNamespace
    from gateway.session_local import _classic_cli_fields

    policy = SimpleNamespace(yolo=False, effective=lambda authority, secrets=True: (
        {"model": {"provider": "perwire", "default": "gpt-4.1"}}, "claude-y"))
    authority = SimpleNamespace(db=SimpleNamespace(get_session=lambda sid: {}))
    live = SimpleNamespace(route=None)
    info = _classic_cli_fields(authority, SimpleNamespace(session_id="s"), live, None, policy)
    assert info["api_mode"] == "anthropic_messages"
