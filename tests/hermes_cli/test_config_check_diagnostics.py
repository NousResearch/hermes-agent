"""Config check reports stale saved capabilities without changing profile state."""

from pathlib import Path

import hermes_cli.config as config_mod
import hermes_cli.plugins as plugins_mod
import pytest
from hermes_cli.config import DEFAULT_CONFIG, _cmd_config_check, _warn_invalid_platform_toolsets
from hermes_cli.config_check_diagnostics import _delivery_policy_diagnostics


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
    # Only a plugin MCP server named 'portable' exists; keep host-installed plugins out of the verdict.
    monkeypatch.setattr(plugins_mod, "get_plugin_toolset_keys_nowait", set)
    monkeypatch.setattr(plugins_mod, "get_portable_mcp_server_names_nowait", lambda: {"portable"})
    stale = _write_home(
        tmp_path / "stale",
        "platform_toolsets:\n"
        "  cli: [hermes-cli, messaging, linear, mcp-linear, mcp-portable, mcp-ghost, ghost]\n"
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

    for home, expected in ((stale, {"messaging", "ghost", "mcp-ghost"}), (clean, set()), (stale, {"messaging", "ghost", "mcp-ghost"})):
        output = _check(home, monkeypatch, capsys)
        for name in ("messaging", "ghost", "mcp-ghost", "linear", "mcp-linear", "mcp-portable", "hermes-teams"):
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


def test_delivery_policy_diagnostics_distinguish_warnings_from_blocking_errors() -> None:
    warnings = _delivery_policy_diagnostics({
        "delegation": {"require_delivery_role": False, "subagent_auto_approve": True},
        "command_allowlist": ["gh *"],
    })
    assert warnings and all(item.startswith("WARNING:") for item in warnings)

    diagnostics = _delivery_policy_diagnostics({
        "delegation": {
            "require_delivery_role": True,
            "subagent_auto_approve": True,
            "role_defaults": {"reviewer": {"toolsets": ["all"]}},
        },
        "approvals": {"unattended_mode": "approve"},
        "command_allowlist": ["git *"],
    })
    assert len(diagnostics) == 4
    assert len([item for item in diagnostics if item.startswith("ERROR:")]) == 1
    assert len([item for item in diagnostics if item.startswith("WARNING:")]) == 3
    assert any("immutable" in item for item in diagnostics)

    assert _delivery_policy_diagnostics({"command_allowlist": ["gh pr view *"]}) == []
    unsafe = _delivery_policy_diagnostics({"command_allowlist": ["gh pr merge * --squash"]})
    assert len(unsafe) == 1 and unsafe[0].startswith("WARNING:")


def test_invalid_delivery_policy_config_is_blocking() -> None:
    diagnostics = _delivery_policy_diagnostics({"delegation": {"require_delivery_role": "true"}})
    assert diagnostics == [
        "ERROR: delegation.require_delivery_role must be true or false; invalid values fail closed at runtime."
    ]
    assert _delivery_policy_diagnostics({"delegation": "unsafe"})[0].startswith("ERROR:")


def test_prefill_diagnostic_targets_delivery_policy_not_legitimate_few_shot_use(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    policy_prefill = tmp_path / "policy.json"
    policy_prefill.write_text(
        '[{"role":"user","content":"You are the independent reviewer and must not merge"}]',
        encoding="utf-8",
    )
    generic_prefill = tmp_path / "generic.json"
    generic_prefill.write_text(
        '[{"role":"user","content":"Translate this sentence"},{"role":"assistant","content":"Bonjour"}]',
        encoding="utf-8",
    )

    diagnostics = _delivery_policy_diagnostics({"prefill_messages_file": "policy.json"})
    assert len(diagnostics) == 1
    assert "fabricated dialogue" in diagnostics[0]
    assert _delivery_policy_diagnostics({"prefill_messages_file": "generic.json"}) == []


def test_config_check_exits_nonzero_for_blocking_delivery_hazard(tmp_path, monkeypatch, capsys) -> None:
    home = _write_home(
        tmp_path / "blocking",
        "delegation:\n"
        "  require_delivery_role: true\n"
        "  role_defaults:\n"
        "    reviewer:\n"
        "      toolsets: [all]\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    with pytest.raises(SystemExit) as excinfo:
        _cmd_config_check(None)
    assert excinfo.value.code == 1
    output = capsys.readouterr().out
    assert "ERROR:" in output
    assert "role_defaults" in output
