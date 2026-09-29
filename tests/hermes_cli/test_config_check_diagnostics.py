"""Config check reports stale saved capabilities without changing profile state."""

from pathlib import Path

from hermes_cli.config import DEFAULT_CONFIG, _cmd_config_check


def _check(home: Path, monkeypatch, capsys) -> str:
    monkeypatch.setenv("HERMES_HOME", str(home))
    _cmd_config_check(None)
    return capsys.readouterr().out


def test_config_check_reports_stale_toolsets_without_mislabeling_mcp_or_plugin_bundles(
    tmp_path, monkeypatch, capsys,
):
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    version = DEFAULT_CONFIG["_config_version"]
    (home_a / "config.yaml").write_text(
        f"_config_version: {version}\n"
        "toolsets: [hermes-cli, messaging]\n"
        "platform_toolsets:\n"
        "  cli: [hermes-cli, messaging, linear]\n"
        "  teams: [hermes-teams]\n"
        "mcp_servers:\n"
        "  linear:\n"
        "    command: linear-mcp\n",
        encoding="utf-8",
    )
    (home_b / "config.yaml").write_text(
        f"_config_version: {version}\ntoolsets: [hermes-cli]\n",
        encoding="utf-8",
    )

    for home, stale in ((home_a, True), (home_b, False), (home_a, True)):
        output = _check(home, monkeypatch, capsys)
        assert ("toolsets contains unknown name 'messaging'" in output) is stale
        assert ("platform 'cli' references unknown toolset 'messaging'" in output) is stale
        assert "unknown toolset 'linear'" not in output
        assert "unknown toolset 'hermes-teams'" not in output


def test_config_check_reports_only_configured_disabled_platforms(tmp_path, monkeypatch, capsys):
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_c = tmp_path / "c"
    for home in (home_a, home_b, home_c):
        home.mkdir()
        disabled_name = "telegram-platform" if home == home_c else "platforms/telegram"
        (home / "config.yaml").write_text(
            f"_config_version: {DEFAULT_CONFIG['_config_version']}\n"
            f"plugins:\n  disabled: [{disabled_name}]\n",
            encoding="utf-8",
        )
    (home_a / ".env").write_text("TELEGRAM_BOT_TOKEN=synthetic-test-token\n", encoding="utf-8")
    (home_c / ".env").write_text("TELEGRAM_BOT_TOKEN=synthetic-test-token\n", encoding="utf-8")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    for home, configured in ((home_a, True), (home_b, False), (home_c, True), (home_a, True)):
        output = _check(home, monkeypatch, capsys)
        assert ("platform plugin 'platforms/telegram' is disabled" in output) is configured
        if configured:
            assert "hermes plugins enable platforms/telegram" in output
        assert "synthetic-test-token" not in output
