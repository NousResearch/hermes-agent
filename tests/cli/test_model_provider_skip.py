import hermes_cli.main as cli_main


def test_skip_provider_option_is_default_when_no_active_provider(monkeypatch):
    captured = {}

    def fake_prompt_provider_choice(choices, default=0):
        captured["default"] = default
        captured["choices"] = list(choices)
        return 0  # pick "Skip provider setup for now"

    monkeypatch.setattr(cli_main, "_prompt_provider_choice", fake_prompt_provider_choice)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda config: [])
    monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda provider: None)

    cli_main.select_provider_and_model()

    assert captured["default"] == 0
    assert captured["choices"][0] == "Skip provider setup for now"


def test_active_provider_default_advances_after_skip_entry(monkeypatch):
    captured = {}

    def fake_prompt_provider_choice(choices, default=0):
        captured["default"] = default
        captured["choices"] = list(choices)
        return 0  # pick "Skip provider setup for now"

    monkeypatch.setattr(cli_main, "_prompt_provider_choice", fake_prompt_provider_choice)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda config: [])
    monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda provider: "openrouter")

    cli_main.select_provider_and_model()

    # CANONICAL_PROVIDERS starts with "nous", so openrouter is index 1 pre-insert,
    # which becomes index 2 after adding the skip item at the front.
    assert captured["default"] == 2
    assert captured["choices"][0] == "Skip provider setup for now"
