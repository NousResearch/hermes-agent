"""Regression tests for interactive setup provider/model persistence.

Since setup_model_provider delegates to select_provider_and_model()
from hermes_cli.main, these tests mock the delegation point and verify
that the setup wizard correctly syncs config from disk after the call.
"""

from __future__ import annotations

from hermes_cli.config import load_config, save_config, save_env_value
from hermes_cli.nous_subscription import NousFeatureState, NousSubscriptionFeatures
from hermes_cli.setup_wizard import _print_setup_summary, setup_model_provider


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
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt_choice", lambda q, c, d=0: (
        _maybe_keep_current_tts(q, c) if _maybe_keep_current_tts(q, c) is not None
        else d
    ))
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt_yes_no", lambda *a, **kw: False)


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


def test_setup_keep_current_custom_from_config_does_not_fall_through(tmp_path, monkeypatch):
    """Keep-current custom should not fall through to the generic model menu."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    # Pre-set custom provider
    _write_model_config("custom", "http://localhost:8080/v1", "local-model")

    config = load_config()
    assert config["model"]["provider"] == "custom"

    def fake_select():
        pass  # user chose "cancel" or "keep current"

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://localhost:8080/v1"


def test_setup_keep_current_config_provider_uses_provider_specific_model_menu(
    tmp_path, monkeypatch
):
    """Keeping current provider preserves the config on disk."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    _write_model_config("zai", "https://open.bigmodel.cn/api/paas/v4", "glm-5")

    config = load_config()

    def fake_select():
        pass  # keep current

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "zai"


def test_setup_same_provider_rotation_strategy_saved_for_multi_credential_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    save_env_value("OPENROUTER_API_KEY", "or-key")

    # Pre-write config so the pool step sees provider="openrouter"
    _write_model_config("openrouter", "", "anthropic/claude-opus-4.6")

    config = load_config()

    class _Entry:
        def __init__(self, label):
            self.label = label

    class _Pool:
        def entries(self):
            return [_Entry("primary"), _Entry("secondary")]

    def fake_select():
        pass  # no-op — config already has provider set

    def fake_prompt_choice(question, choices, default=0):
        if "rotation strategy" in question:
            return 1  # round robin
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        return default

    def fake_prompt_yes_no(question, default=True):
        return False

    # Patch directly on the module objects to ensure local imports pick them up.
    import hermes_cli.main as _main_mod
    import hermes_cli.setup_wizard as _setup_mod
    import agent.credential_pool as _pool_mod
    import agent.auxiliary_client as _aux_mod

    monkeypatch.setattr(_main_mod, "select_provider_and_model", fake_select)
    # NOTE: _stub_tts overwrites prompt_choice, so set our mock AFTER it.
    _stub_tts(monkeypatch)
    monkeypatch.setattr(_setup_mod, "prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(_setup_mod, "prompt_yes_no", fake_prompt_yes_no)
    monkeypatch.setattr(_setup_mod, "prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr(_pool_mod, "load_pool", lambda provider: _Pool())
    monkeypatch.setattr(_aux_mod, "get_available_vision_backends", lambda: [])

    setup_model_provider(config)

    # The pool has 2 entries, so the strategy prompt should fire
    strategy = config.get("credential_pool_strategies", {}).get("openrouter")
    assert strategy == "round_robin", f"Expected round_robin but got {strategy}"


def test_setup_same_provider_fallback_can_add_another_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    save_env_value("OPENROUTER_API_KEY", "or-key")

    # Pre-write config so the pool step sees provider="openrouter"
    _write_model_config("openrouter", "", "anthropic/claude-opus-4.6")

    config = load_config()
    pool_sizes = iter([1, 2])
    add_calls = []

    class _Entry:
        def __init__(self, label):
            self.label = label

    class _Pool:
        def __init__(self, size):
            self._size = size

        def entries(self):
            return [_Entry(f"cred-{idx}") for idx in range(self._size)]

    def fake_load_pool(provider):
        return _Pool(next(pool_sizes))

    def fake_auth_add_command(args):
        add_calls.append(args.provider)

    def fake_select():
        pass  # no-op — config already has provider set

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select same-provider rotation strategy:":
            return 0
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        return default

    yes_no_answers = iter([True, False])

    def fake_prompt_yes_no(question, default=True):
        if question == "Add another credential for same-provider fallback?":
            return next(yes_no_answers)
        return False

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)
    _stub_tts(monkeypatch)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt_yes_no", fake_prompt_yes_no)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("agent.credential_pool.load_pool", fake_load_pool)
    monkeypatch.setattr("hermes_cli.auth_commands.auth_add_command", fake_auth_add_command)
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)

    assert add_calls == ["openrouter"]
    assert config.get("credential_pool_strategies", {}).get("openrouter") == "fill_first"


def test_setup_same_provider_single_credential_keeps_existing_rotation_strategy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    save_env_value("OPENROUTER_API_KEY", "or-key")

    _write_model_config("openrouter", "", "anthropic/claude-opus-4.6")

    config = load_config()
    config["credential_pool_strategies"] = {"openrouter": "round_robin"}
    save_config(config)

    class _Entry:
        def __init__(self, label):
            self.label = label

    class _Pool:
        def entries(self):
            return [_Entry("primary")]

    def fake_select():
        pass

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)
    _stub_tts(monkeypatch)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: _Pool())
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)

    assert config.get("credential_pool_strategies", {}).get("openrouter") == "round_robin"


def test_setup_pool_step_shows_manual_vs_auto_detected_counts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    save_env_value("OPENROUTER_API_KEY", "or-key")

    # Pre-write config so the pool step sees provider="openrouter"
    _write_model_config("openrouter", "", "anthropic/claude-opus-4.6")

    config = load_config()

    class _Entry:
        def __init__(self, label, source):
            self.label = label
            self.source = source

    class _Pool:
        def entries(self):
            return [
                _Entry("primary", "manual"),
                _Entry("secondary", "manual"),
                _Entry("OPENROUTER_API_KEY", "env:OPENROUTER_API_KEY"),
            ]

    def fake_select():
        pass  # no-op — config already has provider set

    def fake_prompt_choice(question, choices, default=0):
        if "rotation strategy" in question:
            return 0
        tts_idx = _maybe_keep_current_tts(question, choices)
        if tts_idx is not None:
            return tts_idx
        return default

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)
    _stub_tts(monkeypatch)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: _Pool())
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)

    out = capsys.readouterr().out
    assert "Current pooled credentials for openrouter: 3 (2 manual, 1 auto-detected from env/shared auth)" in out


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

    monkeypatch.setattr("hermes_cli.setup_wizard.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt_yes_no", fake_prompt_yes_no)
    monkeypatch.setattr("hermes_cli.setup_wizard.prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    setup_model_provider(config)

    assert config.get("credential_pool_strategies", {}) == {}


def test_setup_copilot_uses_gh_auth_and_saves_provider(tmp_path, monkeypatch):
    """Copilot provider saves correctly through delegation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        _write_model_config("copilot", "https://models.github.ai/inference/v1", "gpt-4o")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "copilot"


def test_setup_copilot_acp_uses_model_picker_and_saves_provider(tmp_path, monkeypatch):
    """Copilot ACP provider saves correctly through delegation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        _write_model_config("copilot-acp", "", "claude-sonnet-4")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "copilot-acp"


def test_setup_switch_custom_to_codex_clears_custom_endpoint_and_updates_config(
    tmp_path, monkeypatch
):
    """Switching from custom to codex updates config correctly."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    # Start with custom
    _write_model_config("custom", "http://localhost:11434/v1", "qwen3.5:32b")

    config = load_config()
    assert config["model"]["provider"] == "custom"

    def fake_select():
        _write_model_config("openai-codex", "https://api.openai.com/v1", "gpt-4o")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "openai-codex"
    assert reloaded["model"]["default"] == "gpt-4o"


def test_setup_switch_preserves_non_model_config(tmp_path, monkeypatch):
    """Provider switch preserves other config sections (terminal, display, etc.)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()
    config["terminal"]["timeout"] = 999
    save_config(config)

    config = load_config()

    def fake_select():
        _write_model_config("openrouter", model_name="gpt-4o")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert reloaded["terminal"]["timeout"] == 999
    assert reloaded["model"]["provider"] == "openrouter"


def test_setup_summary_marks_anthropic_auth_as_vision_available(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-key")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: ["anthropic"])

    _print_setup_summary(load_config(), tmp_path)
    output = capsys.readouterr().out

    assert "Vision (image analysis)" in output
    assert "missing run 'hermes setup' to configure" not in output


def test_setup_summary_shows_camofox_when_browser_feature_is_camofox(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.setup_wizard.get_nous_subscription_features",
        lambda config: NousSubscriptionFeatures(
            subscribed=False,
            nous_auth_present=False,
            provider_is_nous=False,
            features={
                "web": NousFeatureState("web", "Web tools", True, False, False, False, False, True, ""),
                "image_gen": NousFeatureState("image_gen", "Image generation", True, False, False, False, False, True, ""),
                "tts": NousFeatureState("tts", "OpenAI TTS", True, False, False, False, False, True, ""),
                "browser": NousFeatureState("browser", "Browser automation", True, True, True, False, True, True, "Camofox"),
                "modal": NousFeatureState("modal", "Modal execution", False, False, False, False, False, True, "local"),
            },
        ),
    )
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    _print_setup_summary(load_config(), tmp_path)
    output = capsys.readouterr().out

    assert "Browser Automation (Camofox)" in output


def test_setup_summary_does_not_mark_incomplete_browserbase_as_available(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("BROWSERBASE_API_KEY", "bb-key")
    monkeypatch.setattr(
        "hermes_cli.setup_wizard.get_nous_subscription_features",
        lambda config: NousSubscriptionFeatures(
            subscribed=False,
            nous_auth_present=False,
            provider_is_nous=False,
            features={
                "web": NousFeatureState("web", "Web tools", True, False, False, False, False, True, ""),
                "image_gen": NousFeatureState("image_gen", "Image generation", True, False, False, False, False, True, ""),
                "tts": NousFeatureState("tts", "OpenAI TTS", True, False, False, False, False, True, ""),
                "browser": NousFeatureState("browser", "Browser automation", True, False, False, False, False, True, "Browserbase"),
                "modal": NousFeatureState("modal", "Modal execution", False, False, False, False, False, True, "local"),
            },
        ),
    )
    monkeypatch.setattr("agent.auxiliary_client.get_available_vision_backends", lambda: [])

    _print_setup_summary(load_config(), tmp_path)
    output = capsys.readouterr().out

    assert "Browser Automation (Browserbase)" not in output
    assert "Browser Automation" in output
    assert "BROWSERBASE_API_KEY/BROWSERBASE_PROJECT_ID" in output
