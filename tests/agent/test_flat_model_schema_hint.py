"""Regression for #75651 part 2: a config.yaml in the removed flat model schema
(``model_provider: <provider>`` + ``model: <id>`` strings) is silently ignored by the
loader — which reads only the structured ``model:`` dict — and the user got the generic
"No provider configured" message with no hint that their YAML was the problem.

Both no-provider raise sites (agent init's ``No LLM provider configured`` and
``resolve_provider``'s ``AuthError(no_provider_configured)``) now append a pointed
suffix naming the flat keys and the structured replacement.
"""

from pathlib import Path

import pytest

from agent.auxiliary_unavailable import flat_model_schema_hint


@pytest.fixture()
def flat_config_home(tmp_path, monkeypatch):
    """A config.yaml in the removed flat layout, with no credentials anywhere."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model_provider: custom\nmodel: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Keep the resolution chain hermetic: no host credentials may satisfy any rung.
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GLM_API_KEY", "KIMI_API_KEY",
                "TOGETHER_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return home


def _read_config_with(path: Path):
    """Point the config loader's home at ``path`` (it caches on the file signature)."""
    import hermes_cli.config as config_mod

    original = config_mod.get_hermes_home

    def _fake_home():
        return path

    config_mod.get_hermes_home = _fake_home
    try:
        config_mod._CONFIG_CACHE.clear() if hasattr(config_mod, "_CONFIG_CACHE") else None
    except Exception:
        pass
    return original


def test_flat_schema_hint_names_the_flat_keys(flat_config_home, monkeypatch):
    """The helper detects both flat keys and says what to move where."""
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "model_provider": "custom", "model": "Qwen3.6-35B"})
    hint = flat_model_schema_hint()
    assert "model_provider" in hint
    assert "flat model schema" in hint
    assert "model:" in hint
    assert "default:" in hint and "provider:" in hint


def test_flat_schema_hint_silent_for_structured_config(monkeypatch):
    """A structured (or missing) model config produces no suffix — the generic setup
    message stays clean."""
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "model": {"default": "qwen", "provider": "custom", "base_url": "http://x/v1"}})
    assert flat_model_schema_hint() == ""
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    assert flat_model_schema_hint() == ""


def test_agent_init_no_provider_error_carries_the_schema_hint(flat_config_home, monkeypatch):
    """The Desktop-visible raise (agent init's "No LLM provider configured") names the
    flat schema when the config uses it, so a red error card is actionable."""
    import types

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **kw: (None, None))
    from agent.agent_init import _routed_client_kwargs

    agent = types.SimpleNamespace(provider="auto", model="m", base_url=None, api_key=None,
                                  _fallback_activated=False, _explicit_provider="")
    with pytest.raises(RuntimeError) as excinfo:
        _routed_client_kwargs(agent, None, 60)

    assert "No LLM provider configured" in str(excinfo.value)
    assert "flat model schema" in str(excinfo.value)
    assert "model_provider" in str(excinfo.value)


def test_resolve_provider_auth_error_carries_the_schema_hint(flat_config_home, monkeypatch):
    """The CLI-visible raise (resolve_provider's AuthError) names the flat schema too."""
    monkeypatch.setattr("agent.auxiliary_unavailable.flat_model_schema_hint", flat_model_schema_hint)
    from hermes_cli import auth as auth_mod

    # No credentials anywhere: every rung of the "auto" chain refuses.
    monkeypatch.setattr(auth_mod, "_openrouter_auto_detected", lambda env: False)
    monkeypatch.setattr(auth_mod, "get_auth_status", lambda *a, **kw: {})

    with pytest.raises(auth_mod.AuthError) as excinfo:
        auth_mod.resolve_provider("auto")

    assert excinfo.value.code == "no_provider_configured"
    assert "flat model schema" in str(excinfo.value)
