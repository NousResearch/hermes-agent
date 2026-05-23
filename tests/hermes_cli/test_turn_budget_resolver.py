from hermes_cli.config import get_default_agent_turn_budgets, resolve_agent_max_turns


def test_default_task_shaped_budgets():
    defaults = get_default_agent_turn_budgets()
    assert defaults["gateway"] == 32
    assert defaults["api_server"] == 32
    assert defaults["cli"] == 16
    assert defaults["lightweight_followup"] == 6
    assert defaults["coding"] == 90
    assert defaults["ship_mode"] == 128
    assert defaults["cron"] == 90


def test_per_surface_budget_wins_over_legacy_global_and_env():
    cfg = {
        "agent": {
            "max_turns": 256,
            "turn_budgets": {"gateway": 32, "api_server": 31, "cli": 16},
        }
    }
    env = {"HERMES_MAX_ITERATIONS": "512"}
    assert resolve_agent_max_turns(cfg, mode="discord", env=env) == 32
    assert resolve_agent_max_turns(cfg, mode="gateway", env=env) == 32
    assert resolve_agent_max_turns(cfg, mode="api_server", env=env) == 31
    assert resolve_agent_max_turns(cfg, mode="local", env=env) == 16


def test_non_default_legacy_global_is_compat_fallback_only_when_no_surface_budget():
    cfg = {"agent": {"max_turns": 77}}
    assert resolve_agent_max_turns(cfg, mode="gateway", env={}) == 77


def test_legacy_default_90_does_not_override_task_defaults_or_lightweight():
    cfg = {"agent": {"max_turns": 90}}
    env = {"HERMES_MAX_ITERATIONS": "90"}
    assert resolve_agent_max_turns(cfg, mode="gateway", env=env) == 32
    assert resolve_agent_max_turns(cfg, mode="cli", env=env) == 16
    assert (
        resolve_agent_max_turns(
            cfg,
            mode="lightweight_followup",
            env={"HERMES_MAX_ITERATIONS": "256"},
            include_legacy_global_override=False,
        )
        == 6
    )


def test_cron_uses_durable_worker_budget():
    cfg = {"agent": {"max_turns": 90, "turn_budgets": {"gateway": 32, "cron": 120}}}
    assert resolve_agent_max_turns(cfg, mode="cron", env={}) == 120
