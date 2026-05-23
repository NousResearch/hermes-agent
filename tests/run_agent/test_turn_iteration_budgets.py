from run_agent import AIAgent


BUDGET_CONFIG = {
    "agent": {
        "max_turns": 90,
        "turn_budgets": {
            "gateway": 32,
            "lightweight_followup": 6,
            "coding": 90,
            "ship_mode": 128,
            "cron": 90,
        },
    }
}


def _agent(platform="discord", configured=32, external=False):
    agent = AIAgent.__new__(AIAgent)
    setattr(agent, "platform", platform)
    setattr(agent, "max_iterations", configured)
    setattr(agent, "_configured_max_iterations", configured)
    setattr(agent, "_external_turn_ceiling_active", external)
    setattr(agent, "_last_resolved_turn_max_iterations", configured)
    return agent


def test_passive_gateway_question_does_not_escalate(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent()
    history = [{"role": "user", "content": "Fix the gateway runtime and run tests"}]
    assert agent._resolve_turn_max_iterations("what changed in gateway?", history) == 32
    assert agent._last_turn_budget_classification == "surface"


def test_lightweight_followup_stays_small_even_after_coding_history(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent()
    history = [{"role": "user", "content": "Implement the runtime gateway budget fix"}]
    assert agent._resolve_turn_max_iterations("ok", history) == 6


def test_simple_check_stays_tiny(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent()
    assert agent._resolve_turn_max_iterations("check status") == 8
    assert agent._last_turn_budget_classification == "simple_check"


def test_coding_turn_escalates_to_coding_budget(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent()
    assert agent._resolve_turn_max_iterations("fix the gateway runtime bug and run tests") == 90
    assert agent._last_turn_budget_classification == "coding"


def test_ship_mode_turn_escalates_to_ship_budget(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent()
    history = [{"role": "user", "content": "We are fixing Hermes runtime budgets and gateway performance"}]
    assert agent._resolve_turn_max_iterations("use 128 iterations if needed and ship this", history) == 128
    assert agent._last_turn_budget_classification == "ship_mode"


def test_fix_it_uses_recent_runtime_context_for_ship_mode(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent()
    history = [{"role": "user", "content": "Hermes gateway runtime performance is broken; implement the full app-dev fix"}]
    assert agent._resolve_turn_max_iterations("fix it", history) == 128


def test_external_ceiling_is_not_reexpanded(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent(configured=4, external=True)
    assert agent._resolve_turn_max_iterations("fix the gateway runtime bug") == 4
    assert agent._last_turn_budget_classification == "external_ceiling"


def test_gateway_refresh_clears_external_ceiling_shape(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    agent = _agent(configured=4, external=True)
    assert agent._resolve_turn_max_iterations("fix the gateway runtime bug") == 4

    # Gateway cached-agent refresh should set a new configured base and clear the external ceiling.
    setattr(agent, "_configured_max_iterations", 32)
    setattr(agent, "max_iterations", 32)
    setattr(agent, "_last_resolved_turn_max_iterations", None)
    setattr(agent, "_external_turn_ceiling_active", False)

    assert agent._resolve_turn_max_iterations("fix the gateway runtime bug and run tests") == 90


def test_cron_and_kanban_keep_configured_worker_budget(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: BUDGET_CONFIG)
    cron_agent = _agent(platform="cron", configured=90)
    assert cron_agent._resolve_turn_max_iterations("ok") == 90
    assert cron_agent._last_turn_budget_classification == "durable_worker"

    kanban_agent = _agent(platform="discord", configured=90)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    assert kanban_agent._resolve_turn_max_iterations("ok") == 90
