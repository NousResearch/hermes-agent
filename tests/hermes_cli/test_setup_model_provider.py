"""Regression tests for interactive setup provider/model persistence.

Since setup_model_provider delegates to select_provider_and_model()
from hermes_cli.main, these tests mock the delegation point and verify
that the setup wizard correctly syncs config from disk after the call.
"""

from __future__ import annotations

from hermes_cli.config import load_config, save_config, save_env_value
from hermes_cli.nous_subscription import NousFeatureState, NousSubscriptionFeatures
from hermes_cli.setup import _print_setup_summary, setup_model_provider


def _maybe_keep_current_tts(question, choices):
    if question != "Select TTS provider:":
        return None
    assert choices[-1].startswith("Keep current (")
    return len(choices) - 1


def _clear_provider_env(monkeypatch):
    for key in (
        "HERMES_INFERENCE_PROVIDER",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "MINIMAX_API_KEY",
        "MINIMAX_CN_API_KEY",
        "ANTHROPIC_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _stub_tts(monkeypatch):
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda q, c, d=0: (
        _maybe_keep_current_tts(q, c) if _maybe_keep_current_tts(q, c) is not None
        else d
    ))
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *a, **kw: False)


def _write_model_config(provider, base_url="", model_name="test-model"):
    """Simulate what a _model_flow_* function writes to disk."""
    cfg = load_config()
    m = cfg.get("model")
    if not isinstance(m, dict):
        m = {"default": m} if m else {}
        cfg["model"] = m
    m["provider"] = provider
    if base_url:
        m["base_url"] = base_url
    else:
        m.pop("base_url", None)
    if model_name:
        m["default"] = model_name
    m.pop("api_mode", None)
    save_config(cfg)


def _write_aux_config(task="compression", provider="gemini", model_name="gemini-2.5-flash"):
    """Simulate the aux picker writing a task override to disk."""
    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    entry = aux.setdefault(task, {})
    entry["provider"] = provider
    entry["model"] = model_name
    save_config(cfg)


def test_setup_model_provider_preserves_auxiliary_choices_written_by_picker(tmp_path, monkeypatch):
    """Aux choices made inside hermes setup must survive the wizard's final save."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()
    assert config["auxiliary"]["compression"]["provider"] == "auto"

    def fake_select():
        _write_aux_config("compression", "gemini", "gemini-2.5-flash")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config, quick=True)
    save_config(config)  # mirrors run_setup_wizard(section="model") final save

    reloaded = load_config()
    compression = reloaded["auxiliary"]["compression"]
    assert compression["provider"] == "gemini"
    assert compression["model"] == "gemini-2.5-flash"


def test_setup_copilot_acp_skips_same_provider_pool_step(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select your inference provider:":
            return 15  # GitHub Copilot ACP
        if question == "Select default model:":
            return 0
        if question == "Configure vision:":
            return len(choices) - 1
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    def fake_prompt_yes_no(question, default=True):
        if question == "Add another credential for same-provider fallback?":
            raise AssertionError("same-provider pool prompt should not appear for copilot-acp")
        return False

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", fake_prompt_yes_no)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)

    assert config.get("credential_pool_strategies", {}) == {}


def test_setup_summary_local_browser_unavailable_without_chromium(
    tmp_path, monkeypatch, capsys
):
    """End-to-end: agent-browser present but no Chromium in local mode must
    render as unavailable with an install hint — not a false 'available'.

    Unlike the mocked-feature tests above, this drives the real
    ``get_nous_subscription_features`` so the surface stays aligned with the
    runtime gate in ``tools.browser_tool.check_browser_requirements``.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    cfg = load_config()
    browser_cfg = cfg.get("browser")
    if not isinstance(browser_cfg, dict):
        browser_cfg = {}
        cfg["browser"] = browser_cfg
    browser_cfg["cloud_provider"] = "local"
    save_config(cfg)

    # Only stub the readiness probes; the feature resolver itself is real.
    monkeypatch.setattr("hermes_cli.nous_subscription._has_agent_browser", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.nous_subscription.get_nous_portal_account_info",
        lambda *a, **k: None,
    )
    monkeypatch.setattr("tools.browser_tool._chromium_installed", lambda: False)
    monkeypatch.setattr("tools.browser_tool._using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_available_vision_backends", lambda: []
    )

    _print_setup_summary(load_config(), tmp_path)
    output = capsys.readouterr().out

    assert "Browser Automation (Local browser)" not in output
    assert "agent-browser install --with-deps" in output
