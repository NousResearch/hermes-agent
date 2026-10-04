"""Config check reports stale saved capabilities without changing profile state."""

from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_config_check_reports_yaml_structure_issues(tmp_path, monkeypatch, capsys):
    home = _write_home(tmp_path / "invalid", "timezone: Not/A/Timezone\n")
    output = _check(home, monkeypatch, capsys)
    assert "not a valid IANA zone name" in output


def test_config_check_profiles_option_reports_named_profile(tmp_path, monkeypatch, capsys):
    home = _write_home(tmp_path / "default", "")
    profiles = tmp_path / "profiles" / "research"
    profiles.mkdir(parents=True)
    (profiles / "config.yaml").write_text("timezone: Not/A/Timezone\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants as config_home
    monkeypatch.setattr(config_home, "get_default_hermes_root", lambda: tmp_path)
    class Args:
        profiles = True
    _cmd_config_check(Args())
    output = capsys.readouterr().out
    assert "Profile 'research'" in output
    assert "not a valid IANA zone name" in output


def _check_profiles_readonly(home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    root = hermes_constants.get_default_hermes_root()
    before = {p: p.read_bytes() for p in root.rglob("config.yaml")}
    _cmd_config_check(SimpleNamespace(profiles=True))
    output = capsys.readouterr().out
    assert {p: p.read_bytes() for p in root.rglob("config.yaml")} == before
    return output


@pytest.mark.parametrize("body,kind", [
    ("- a\n- b\n", "list"), ("[]\n", "list"),
    ("hello\n", "str"), ("''\n", "str"),
    ("42\n", "int"), ("0\n", "int"), ("false\n", "bool"),
])
def test_profiles_report_non_mapping_roots(tmp_path, monkeypatch, capsys, body, kind):
    home = _write_home(tmp_path / "root", "")
    profile = home / "profiles" / "invalid"
    profile.mkdir(parents=True)
    path = profile / "config.yaml"
    path.write_text(body, encoding="utf-8")
    output = _check_profiles_readonly(home, monkeypatch, capsys)
    assert "Profile 'invalid'" in output
    assert f"top-level value must be a mapping, got {kind}" in output
    assert "cannot parse config.yaml" not in output
    assert "has no attribute" not in output


@pytest.mark.parametrize("body", ["", "null\n", "{}\n"])
def test_profiles_accept_empty_mapping_state(tmp_path, monkeypatch, capsys, body):
    home = _write_home(tmp_path / "root", "")
    profile = home / "profiles" / "empty"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(body, encoding="utf-8")
    output = _check_profiles_readonly(home, monkeypatch, capsys)
    assert "top-level value must be a mapping" not in output
    assert "cannot parse config.yaml" not in output


def test_profiles_preserve_timezone_and_parse_hints(tmp_path, monkeypatch, capsys):
    home = _write_home(tmp_path / "root", "timezone: Not/A/Zone\n")
    for name, body in (("research", "timezone: Not/A/Zone\n"), ("broken", "timezone: [\n")):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text(body, encoding="utf-8")
    output = _check_profiles_readonly(home, monkeypatch, capsys)
    hint = "Hint: Use an IANA zone name such as America/New_York or Asia/Tokyo"
    assert output.count(hint) == 2
    assert output.count("schedules silently fall back to server-local time") == 2
    assert "Profile 'broken'" in output
    assert "cannot parse config.yaml" in output
    assert "Hint: Fix the YAML syntax" in output


def test_profiles_skip_current_home_and_check_siblings(tmp_path, monkeypatch, capsys):
    root = _write_home(tmp_path / "root", "")
    profiles = root / "profiles"
    profiles.mkdir()
    current = _write_home(profiles / "research", "timezone: Not/A/Zone\n")
    _write_home(profiles / "sibling", "timezone: Not/A/Zone\n")
    # Match --profile's actual HERMES_HOME selection; do not stub root discovery.
    output = _check_profiles_readonly(current, monkeypatch, capsys)
    assert "Saved configuration:" in output
    assert "Profile 'research'" not in output
    assert "Profile 'sibling'" in output
    assert output.count("not a valid IANA zone name") == 2


@pytest.mark.parametrize("stamp,expected", [
    ("_config_version: 1\n", 1), ("", 0),
    ("_config_version: invalid\n", 0), ("_config_version: false\n", 0),
])
def test_profiles_report_stale_versions(tmp_path, monkeypatch, capsys, stamp, expected):
    home = _write_home(tmp_path / "root", "")
    profile = home / "profiles" / "legacy"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(stamp, encoding="utf-8")
    output = _check_profiles_readonly(home, monkeypatch, capsys)
    assert "Profile 'legacy'" in output
    assert f"Config version: {expected} → {DEFAULT_CONFIG['_config_version']} (update available)" in output


def test_profiles_current_and_future_versions_are_silent(tmp_path, monkeypatch, capsys):
    home = _write_home(tmp_path / "root", "")
    for name, version in (("current", DEFAULT_CONFIG['_config_version']), ("future", 999)):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text(f"_config_version: {version}\n", encoding="utf-8")
    output = _check_profiles_readonly(home, monkeypatch, capsys)
    assert "Profile 'current'" not in output
    assert "Profile 'future'" not in output


def test_profiles_scan_is_opt_in(tmp_path, monkeypatch, capsys):
    home = _write_home(tmp_path / "root", "")
    profile = home / "profiles" / "invalid"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("- a\n", encoding="utf-8")
    output = _check(home, monkeypatch, capsys)
    assert "Profile 'invalid'" not in output
    assert "top-level value must be a mapping" not in output


def test_profiles_do_not_weaken_current_home_root_validation(tmp_path, monkeypatch):
    home = tmp_path / "root"
    home.mkdir()
    (home / "config.yaml").write_text("- a\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    with pytest.raises(config_mod.InvalidUserConfigError, match="top-level value must be a mapping, got list"):
        _cmd_config_check(SimpleNamespace(profiles=True))
