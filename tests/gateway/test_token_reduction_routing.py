"""Gateway integration coverage for token-reduction routing."""

from gateway.run import GatewayRunner


def test_resolve_turn_agent_config_routes_only_when_enabled():
    runner = GatewayRunner.__new__(GatewayRunner)
    config = {
        "agent": {
            "simple_model_routing": {
                "enabled": True,
                "model": "cheap-model",
                "max_chars": 240,
            }
        }
    }

    simple = runner._resolve_turn_agent_config(
        "Define confidence interval",
        "primary-model",
        {},
        history=None,
        user_config=config,
    )
    action = runner._resolve_turn_agent_config(
        "Deploy the server",
        "primary-model",
        {},
        history=None,
        user_config=config,
    )
    continued = runner._resolve_turn_agent_config(
        "thanks",
        "primary-model",
        {},
        history=[{"role": "user", "content": "Build the service"}],
        user_config=config,
    )

    assert simple["model"] == "cheap-model"
    assert action["model"] == "primary-model"
    assert continued["model"] == "primary-model"


def test_cache_busting_config_includes_prompt_and_budget_settings():
    extracted = GatewayRunner._extract_cache_busting_config(
        {
            "skills": {"catalog_mode": "compact"},
            "tool_output": {
                "result_budget_chars": 4000,
                "turn_budget_chars": 12000,
                "preview_chars": 1000,
                "tool_overrides": {"read_file": 12000},
            },
        }
    )

    assert extracted["skills.catalog_mode"] == "compact"
    assert extracted["tool_output.result_budget_chars"] == 4000
    assert extracted["tool_output.turn_budget_chars"] == 12000
    assert extracted["tool_output.preview_chars"] == 1000
    assert extracted["tool_output.tool_overrides"] == {"read_file": 12000}
