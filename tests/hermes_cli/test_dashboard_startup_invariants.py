"""Dashboard startup invariant for plugin resolution."""

from __future__ import annotations

from types import SimpleNamespace


def test_dashboard_runtime_refreshes_plugins_before_auth_gate(tmp_path, monkeypatch):
    """Startup must rescan when an earlier discovery used stale configuration."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.plugins as plugins_mod
    from hermes_cli.dashboard_auth import clear_providers, list_providers
    from hermes_cli import main as cli_main

    plugins_mod._reset_plugin_managers_for_tests()
    clear_providers()
    try:
        plugins_mod.discover_plugins()
        assert not any(provider.name == "basic" for provider in list_providers())

        config_path.write_text(
            "dashboard:\n"
            "  basic_auth:\n"
            "    username: test-user\n"
            "    password_hash: unused-test-hash\n"
            "    secret: test-only-signing-secret-not-a-real-key\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(cli_main, "_sync_bundled_skills_quietly", lambda: None)
        monkeypatch.setattr(cli_main, "_resolve_dashboard_web_dist", lambda *_: None)
        from hermes_cli import config as config_mod, mcp_startup
        monkeypatch.setattr(config_mod, "apply_terminal_config_to_env", lambda: None)
        monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **_: None)

        cli_main._dashboard_prepare_runtime(SimpleNamespace(), headless_backend=False)

        assert any(provider.name == "basic" for provider in list_providers())
    finally:
        plugins_mod._reset_plugin_managers_for_tests()
        clear_providers()
