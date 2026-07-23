"""Runtime coverage for gateway filesystem-checkpoint configuration."""


def test_gateway_checkpoint_config_reaches_real_agent(tmp_path, monkeypatch):
    """Raw gateway YAML must configure the real agent checkpoint manager."""
    from gateway import run as gateway_run
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        """checkpoints:
  enabled: true
  max_snapshots: 11
  max_total_size_mb: 345
  max_file_size_mb: 6
""",
        encoding="utf-8",
    )

    config = gateway_run._load_gateway_config()
    agent = AIAgent(
        model="anthropic/claude-sonnet-4",
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        **gateway_run._checkpoint_agent_kwargs(config),
    )
    try:
        manager = agent._checkpoint_mgr
        assert manager.enabled is True
        assert manager.max_snapshots == 11
        assert manager.max_total_size_mb == 345
        assert manager.max_file_size_mb == 6
    finally:
        agent.close()


def test_checkpoint_agent_kwargs_supports_legacy_boolean_config():
    from gateway.run import _checkpoint_agent_kwargs
    from hermes_cli.config import DEFAULT_CONFIG

    kwargs = _checkpoint_agent_kwargs({"checkpoints": True})
    defaults = DEFAULT_CONFIG["checkpoints"]

    assert kwargs["checkpoints_enabled"] is True
    assert kwargs["checkpoint_max_snapshots"] == defaults["max_snapshots"]
    assert kwargs["checkpoint_max_total_size_mb"] == defaults["max_total_size_mb"]
    assert kwargs["checkpoint_max_file_size_mb"] == defaults["max_file_size_mb"]
