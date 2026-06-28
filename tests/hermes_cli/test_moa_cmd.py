from hermes_cli.moa_cmd import _print_config


def test_print_config_includes_reasoning_efforts(capsys):
    _print_config(
        {
            "moa": {
                "default_preset": "review",
                "presets": {
                    "review": {
                        "reference_models": [
                            {"provider": "deepseek", "model": "deepseek-v4-pro"}
                        ],
                        "aggregator": {"provider": "openai-codex", "model": "gpt-5.5"},
                        "reference_reasoning_effort": "low",
                        "aggregator_reasoning_effort": "xhigh",
                    }
                },
            }
        }
    )

    out = capsys.readouterr().out
    assert "Reference reasoning: low" in out
    assert "Aggregator reasoning: xhigh" in out
