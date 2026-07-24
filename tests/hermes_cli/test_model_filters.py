"""Provider-scoped model exclusion behavior."""

from hermes_cli.model_filters import (
    filter_model_ids,
    filter_provider_rows,
)


def _exclusion_config():
    return {
        "model_catalog": {
            "excluded_models": {
                "openrouter": ["anthropic/*", "openai/*"],
            },
        },
    }


def test_filter_model_ids_matches_exact_and_glob_case_insensitively():
    rules = {
        "OpenRouter": [
            "anthropic/*",
            "openai/gpt-5.6-sol",
        ],
    }

    assert filter_model_ids(
        "openrouter",
        [
            "anthropic/claude-opus-4.8",
            "OpenAI/GPT-5.6-SOL",
            "openai/gpt-5.6-mini",
            "moonshotai/kimi-k2.6",
        ],
        rules,
    ) == [
        "openai/gpt-5.6-mini",
        "moonshotai/kimi-k2.6",
    ]


def test_filter_model_ids_is_scoped_to_provider():
    rules = {"openrouter": ["anthropic/*", "openai/*"]}

    assert filter_model_ids(
        "meridian",
        ["anthropic/claude-opus-4.8", "claude-opus-4-8"],
        rules,
    ) == [
        "anthropic/claude-opus-4.8",
        "claude-opus-4-8",
    ]
    assert filter_model_ids(
        "openai-codex",
        ["openai/gpt-5.6-sol", "gpt-5.6-sol"],
        rules,
    ) == [
        "openai/gpt-5.6-sol",
        "gpt-5.6-sol",
    ]


def test_filter_provider_rows_updates_counts_without_mutating_input():
    rows = [
        {
            "slug": "openrouter",
            "models": ["anthropic/claude-opus-4.8", "deepseek/deepseek-v4"],
            "total_models": 2,
        },
        {
            "slug": "meridian",
            "models": ["claude-opus-4-8"],
            "total_models": 1,
        },
    ]

    filtered = filter_provider_rows(rows, {"openrouter": ["anthropic/*"]})

    assert filtered[0]["models"] == ["deepseek/deepseek-v4"]
    assert filtered[0]["total_models"] == 1
    assert filtered[1] is rows[1]
    assert rows[0]["models"] == [
        "anthropic/claude-opus-4.8",
        "deepseek/deepseek-v4",
    ]


def test_filter_model_ids_ignores_malformed_rules():
    models = ["anthropic/claude-opus-4.8"]

    assert filter_model_ids("openrouter", models, None) == models
    assert filter_model_ids("openrouter", models, []) == models
    assert filter_model_ids(
        "openrouter",
        models,
        {"openrouter": "anthropic/*"},
    ) == models
    assert filter_model_ids(
        "openrouter",
        models,
        {"openrouter": [None, "", 42]},
    ) == models


def test_terminal_model_picker_hides_excluded_models(monkeypatch):
    from hermes_cli.auth import _prompt_model_selection

    captured = {}

    def _choose_first(_title, choices, **_kwargs):
        captured["choices"] = choices
        return 0

    monkeypatch.setattr("hermes_cli.config.load_config", _exclusion_config)
    monkeypatch.setattr(
        "hermes_cli.curses_ui.curses_radiolist",
        _choose_first,
    )
    monkeypatch.setattr(
        "hermes_cli.auth._confirm_expensive_model_selection",
        lambda *args, **kwargs: True,
    )

    selected = _prompt_model_selection(
        [
            "anthropic/claude-opus-4.8",
            "openai/gpt-5.6-sol",
            "deepseek/deepseek-v4",
        ],
        confirm_provider="openrouter",
    )

    assert selected == "deepseek/deepseek-v4"
    assert not any("anthropic/" in choice for choice in captured["choices"])
    assert not any("openai/" in choice for choice in captured["choices"])


def test_terminal_model_picker_still_accepts_explicit_model_id(monkeypatch):
    from hermes_cli.auth import _prompt_model_selection

    def _choose_custom(_title, choices, **_kwargs):
        return choices.index("Enter custom model name")

    monkeypatch.setattr("hermes_cli.config.load_config", _exclusion_config)
    monkeypatch.setattr(
        "hermes_cli.curses_ui.curses_radiolist",
        _choose_custom,
    )
    monkeypatch.setattr(
        "hermes_cli.auth._confirm_expensive_model_selection",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: "anthropic/claude-opus-4.8",
    )

    selected = _prompt_model_selection(
        ["anthropic/claude-opus-4.8", "deepseek/deepseek-v4"],
        confirm_provider="openrouter",
    )

    assert selected == "anthropic/claude-opus-4.8"
