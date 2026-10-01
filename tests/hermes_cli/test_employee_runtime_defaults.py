"""Employee defaults reach native readers without overriding profile preferences."""

from types import SimpleNamespace


def test_native_runtime_defaults_and_profile_overrides(tmp_path):
    from agent import secret_scope
    from agent.agent_init import _parse_compression_config
    from gateway.config import load_gateway_config
    from gateway.display_config import resolve_display_setting
    from hermes_cli.config import load_config
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.approval_context import _get_approval_mode

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.yaml").write_text("{}\n")
    (second / "config.yaml").write_text("""\
compression:
  threshold: 0.9
  min_tail_user_messages: 5
approvals:
  mode: manual
  destructive_slash_confirm: true
  mcp_reload_confirm: true
stt:
  echo_transcripts: true
display:
  platforms:
    telegram:
      streaming: true
      tool_progress: all
      show_reasoning: true
      interim_assistant_messages: false
""")
    was_multiplexed = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        for home, overridden in ((first, False), (second, True), (first, False)):
            home_token = set_hermes_home_override(home)
            scope_token = secret_scope.set_secret_scope({}, profile_home=str(home))
            try:
                config = load_config()
                raw = load_user_config_effective()
                # The presence-sensitive loader must not acquire a second defaults layer.
                if not overridden:
                    assert raw == {}
                agent = SimpleNamespace(
                    model="test-model", provider="openrouter", api_mode="chat_completions",
                    quiet_mode=True,
                )
                compression = _parse_compression_config(agent, config)
                assert compression.threshold == (0.9 if overridden else 0.85)
                assert compression.min_tail_users == (5 if overridden else 3)
                assert _get_approval_mode() == ("manual" if overridden else "off")
                for key in ("destructive_slash_confirm", "mcp_reload_confirm"):
                    assert config["approvals"][key] is overridden
                assert load_gateway_config().stt_echo_transcripts is overridden
                for setting in ("streaming", "show_reasoning"):
                    assert resolve_display_setting(raw, "telegram", setting) is overridden
                assert resolve_display_setting(raw, "telegram", "interim_assistant_messages") is not overridden
                assert resolve_display_setting(raw, "telegram", "tool_progress") == (
                    "all" if overridden else "off"
                )
            finally:
                secret_scope.reset_secret_scope(scope_token)
                reset_hermes_home_override(home_token)
    finally:
        secret_scope.set_multiplex_active(was_multiplexed)


def test_cli_compression_defaults_match_runtime_and_accept_overrides(tmp_path, monkeypatch):
    import cli
    from hermes_cli.config import DEFAULT_CONFIG

    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    config = cli.load_cli_config()
    for key in ("threshold", "min_tail_user_messages"):
        assert config["compression"][key] == DEFAULT_CONFIG["compression"][key]
    assert config["agent"]["reasoning_effort"] == DEFAULT_CONFIG["agent"]["reasoning_effort"]

    path.write_text("compression:\n  threshold: 0.9\n  min_tail_user_messages: 5\n")
    config = cli.load_cli_config()
    assert config["compression"]["threshold"] == 0.9
    assert config["compression"]["min_tail_user_messages"] == 5
