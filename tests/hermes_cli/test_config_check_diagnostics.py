"""Config check reports stale saved capabilities without changing profile state."""

from pathlib import Path

import hermes_cli.config as config_mod
import hermes_cli.plugins as plugins_mod
from hermes_cli.config import DEFAULT_CONFIG, _cmd_config_check, _warn_invalid_platform_toolsets


def _write_home(home: Path, body: str, env: str = "") -> Path:
    home.mkdir()
    (home / "config.yaml").write_text(f"_config_version: {DEFAULT_CONFIG['_config_version']}\n{body}", encoding="utf-8")
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    return home


def _check(home: Path, monkeypatch, capsys) -> str:
    monkeypatch.setenv("HERMES_HOME", str(home))
    _cmd_config_check(None)
    return capsys.readouterr().out


def test_config_check_and_migrate_agree_on_stale_platform_toolsets(tmp_path, monkeypatch, capsys):
    # No plugin provides 'ghost'; keep host-installed plugins out of the verdict.
    monkeypatch.setattr(plugins_mod, "get_plugin_toolset_keys_nowait", set)
    monkeypatch.setattr(plugins_mod, "get_portable_mcp_server_names_nowait", set)
    stale = _write_home(
        tmp_path / "stale",
        "platform_toolsets:\n"
        "  cli: [hermes-cli, messaging, linear, ghost]\n"
        "  teams: [hermes-teams]\n"
        "mcp_servers:\n"
        "  linear:\n"
        "    command: linear-mcp\n"
        "    enabled: false\n"
        # Bookkeeping for a plugin that is no longer installed is not proof its toolset exists.
        "known_plugin_toolsets:\n"
        "  cli: [ghost]\n",
    )
    # A malformed mcp_servers section must not crash the check.
    clean = _write_home(tmp_path / "clean", "platform_toolsets:\n  cli: [hermes-cli]\nmcp_servers: [a]\n")

    for home, expected in ((stale, {"messaging", "ghost"}), (clean, set()), (stale, {"messaging", "ghost"})):
        output = _check(home, monkeypatch, capsys)
        for name in ("messaging", "ghost", "linear", "hermes-teams"):
            assert (f"unknown toolset '{name}'" in output) is (name in expected), (home.name, name)

        results = {"warnings": []}
        _warn_invalid_platform_toolsets(results, quiet=True)
        assert {w for w in results["warnings"] if "unknown toolset" in w} == {
            line.strip().removeprefix("⚠ ") for line in output.splitlines() if "unknown toolset" in line
        }


def test_config_check_reports_disabled_platform_only_when_runtime_disables_it(tmp_path, monkeypatch, capsys):
    manifest = {
        "name": "fakechat-platform",
        "requires_env": [{"name": "FAKECHAT_TOKEN"}],
        "optional_env": [{"name": "FAKECHAT_HOME_CHANNEL"}],
    }
    monkeypatch.setattr(config_mod, "_platform_plugin_manifests", lambda *a, **k: iter([("fakechat", manifest)]))
    monkeypatch.delenv("FAKECHAT_TOKEN", raising=False)
    token = "FAKECHAT_TOKEN=synthetic-test-token\n"
    cases = (
        (_write_home(tmp_path / "key", "plugins:\n  disabled: [platforms/fakechat]\n", token), True),
        (_write_home(tmp_path / "name", "plugins:\n  disabled: [fakechat-platform]\n", token), True),
        (_write_home(tmp_path / "no_token", "plugins:\n  disabled: [platforms/fakechat]\n"), False),
        # The runtime ignores a non-list plugins.disabled, so the platform is not disabled.
        (_write_home(tmp_path / "string", "plugins:\n  disabled: platforms/fakechat\n", token), False),
    )

    for home, reported in (*cases, cases[0]):
        output = _check(home, monkeypatch, capsys)
        assert ("platform plugin 'platforms/fakechat' is disabled" in output) is reported, home.name
        if reported:
            assert "hermes plugins enable platforms/fakechat" in output
        assert "synthetic-test-token" not in output


def test_config_check_reports_config_yaml_structure_issues(tmp_path, monkeypatch, capsys):
    home = _write_home(
        tmp_path / "malformed",
        "custom_providers:\n"
        "  fallback_model: misplaced\n",
    )

    output = _check(home, monkeypatch, capsys)

    assert "config.yaml:" in output
    assert "fallback_model appears inside custom_providers" in output
    assert "custom_providers is a dict" in output
