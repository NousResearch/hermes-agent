from gateway.config import EvaluatorShadowConfig, GatewayConfig
from gateway.run import GatewayRunner


def _runner(config):
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.pre_delivery_gate = None
    runner.pre_delivery_gate_mode = "shadow"
    return runner


def test_evaluator_shadow_is_disabled_by_default():
    config = GatewayConfig.from_dict({})
    assert config.evaluator_shadow.enabled is False
    runner = _runner(config)
    runner._init_evaluator_shadow()
    assert runner.pre_delivery_gate is None


def test_invalid_enabled_shadow_does_not_inject(tmp_path):
    config = GatewayConfig.from_dict({
        "gateway": {
            "evaluator_shadow": {
                "enabled": True,
                "evaluator_root": str(tmp_path),
                "agent_configuration_id": "hermes-gateway-default-v1",
                "evaluator_configuration_id": "evaluator-main-v1",
            }
        }
    })
    runner = _runner(config)
    runner._init_evaluator_shadow()
    assert runner.pre_delivery_gate is None


def test_complete_shadow_config_injects_adapter(tmp_path):
    script = tmp_path / "scripts" / "live_shadow_policy.py"
    script.parent.mkdir()
    script.write_text("# fake evaluator entrypoint\n")
    config = GatewayConfig.from_dict({
        "gateway": {
            "evaluator_shadow": {
                "enabled": True,
                "evaluator_root": str(tmp_path),
                "evidence_output": "~/evidence/live-shadow.jsonl",
                "agent_configuration_id": "hermes-gateway-default-v1",
                "evaluator_configuration_id": "evaluator-main-v1",
            }
        }
    })
    runner = _runner(config)
    runner._init_evaluator_shadow()
    assert runner.pre_delivery_gate is not None
    assert runner.pre_delivery_gate.agent_configuration_id == "hermes-gateway-default-v1"
    assert runner.pre_delivery_gate.evaluator_configuration_id == "evaluator-main-v1"


def test_top_level_evaluator_shadow_config_wins_over_nested():
    config = GatewayConfig.from_dict({
        "evaluator_shadow": {"agent_configuration_id": "top"},
        "gateway": {"evaluator_shadow": {"agent_configuration_id": "nested"}},
    })
    assert config.evaluator_shadow.agent_configuration_id == "top"


def test_loader_bridges_evaluator_shadow_and_preserves_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "evaluator_shadow:\n"
        "  enabled: true\n"
        "  evaluator_root: /top\n"
        "  agent_configuration_id: top-agent\n"
        "  evaluator_configuration_id: top-evaluator\n"
        "gateway:\n"
        "  evaluator_shadow:\n"
        "    enabled: false\n"
        "    evaluator_root: /nested\n"
        "    agent_configuration_id: nested-agent\n"
        "    evaluator_configuration_id: nested-evaluator\n",
        encoding="utf-8",
    )
    from gateway.config import load_gateway_config
    config = load_gateway_config()
    assert config.evaluator_shadow.enabled is True
    assert config.evaluator_shadow.evaluator_root == "/top"
    assert config.evaluator_shadow.agent_configuration_id == "top-agent"
    assert config.evaluator_shadow.evaluator_configuration_id == "top-evaluator"


def test_runner_rejects_whitespace_shadow_ids(monkeypatch, tmp_path):
    from gateway.config import EvaluatorShadowConfig, GatewayConfig
    from gateway.run import GatewayRunner
    root = tmp_path / "evaluator"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "live_shadow_policy.py").write_text("", encoding="utf-8")
    config = GatewayConfig(evaluator_shadow=EvaluatorShadowConfig(
        enabled=True, evaluator_root=str(root), evidence_output=str(tmp_path / "evidence.jsonl"),
        agent_configuration_id="   ", evaluator_configuration_id="\\t",
    ))
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.pre_delivery_gate = None
    runner._init_evaluator_shadow()
    assert runner.pre_delivery_gate is None


def test_mode_defaults_to_shadow():
    assert EvaluatorShadowConfig().mode == "shadow"
    assert EvaluatorShadowConfig.from_dict({}).mode == "shadow"


def test_mode_accepts_strict_and_rejects_unknown_values():
    assert EvaluatorShadowConfig.from_dict({"mode": "strict"}).mode == "strict"
    assert EvaluatorShadowConfig.from_dict({"mode": "STRICT"}).mode == "strict"
    assert EvaluatorShadowConfig.from_dict({"mode": "bogus"}).mode == "shadow"


def _complete_shadow_dict(tmp_path, **overrides):
    script = tmp_path / "scripts" / "live_shadow_policy.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# fake evaluator entrypoint\n")
    data = {
        "enabled": True,
        "evaluator_root": str(tmp_path),
        "agent_configuration_id": "hermes-gateway-default-v1",
        "evaluator_configuration_id": "evaluator-main-v1",
        **overrides,
    }
    return GatewayConfig.from_dict({"gateway": {"evaluator_shadow": data}})


def test_complete_strict_config_injects_adapter_and_sets_mode(tmp_path):
    config = _complete_shadow_dict(tmp_path, mode="strict")
    runner = _runner(config)
    runner._init_evaluator_shadow()
    assert runner.pre_delivery_gate is not None
    assert runner.pre_delivery_gate_mode == "strict"


def test_complete_shadow_config_keeps_shadow_mode(tmp_path):
    config = _complete_shadow_dict(tmp_path)
    runner = _runner(config)
    runner._init_evaluator_shadow()
    assert runner.pre_delivery_gate is not None
    assert runner.pre_delivery_gate_mode == "shadow"


def test_incomplete_strict_config_stays_disabled(tmp_path, caplog):
    """Requesting strict mode without a complete config must not silently degrade to a
    working-but-unnoticed shadow gate, nor crash startup — it stays disabled with a loud log."""
    import logging
    config = GatewayConfig.from_dict({
        "gateway": {"evaluator_shadow": {"enabled": True, "mode": "strict"}},
    })
    runner = _runner(config)
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        runner._init_evaluator_shadow()
    assert runner.pre_delivery_gate is None
    assert runner.pre_delivery_gate_mode == "shadow"
