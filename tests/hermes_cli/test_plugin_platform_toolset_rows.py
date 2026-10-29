"""``platform_toolsets`` rows for plugin platforms (teams, google_chat, homeassistant) that the installers
seeded from cli-config.yaml.example: inert until the plugin loads, so they must not log the #38798
"no valid toolsets" error, while a genuinely invalid row still does."""

import json
import logging

import pytest

# The rows cli-config.yaml.example seeded into every install (install.sh, install.ps1, the Docker hook,
# `hermes doctor --fix`), plus Home Assistant's from before it left core. None of these platforms is set up.
_SEEDED_ROWS = {
    "cli": ["hermes-cli"],
    "telegram": ["hermes-telegram"],
    "homeassistant": ["hermes-homeassistant"],
    "teams": ["hermes-teams"],
    "google_chat": ["hermes-google_chat"],
}


def _write_config(home, platform_toolsets):
    # JSON is valid YAML, and the CI test env has no pyyaml.
    (home / "config.yaml").write_text(json.dumps({"platform_toolsets": platform_toolsets}), encoding="utf-8")


def _toolset_errors(caplog):
    return [r.getMessage() for r in caplog.records if "#38798" in r.getMessage()]


@pytest.fixture(autouse=True)
def _fresh_warning_latch():
    import hermes_cli.tools_config as tc

    tc._warned_invalid_platform_toolsets.clear()
    yield
    tc._warned_invalid_platform_toolsets.clear()


def test_seeded_plugin_platform_rows_log_nothing_on_schema_build(caplog):
    """The kanban gate runs on every schema build and skills scan and resolves every saved row."""
    from hermes_constants import get_hermes_home
    from tools.kanban_tools import _profile_has_kanban_toolset

    _write_config(get_hermes_home(), _SEEDED_ROWS)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        _profile_has_kanban_toolset()

    assert _toolset_errors(caplog) == []


def test_invalid_plugin_platform_row_still_warns(caplog):
    from hermes_constants import get_hermes_home
    from tools.kanban_tools import _profile_has_kanban_toolset

    _write_config(get_hermes_home(), {"cli": ["hermes-cli"], "teams": ["hermes-tems"]})
    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        _profile_has_kanban_toolset()

    errors = _toolset_errors(caplog)
    assert len(errors) == 1 and "'teams'" in errors[0] and "hermes-tems" in errors[0], errors


def test_seeded_config_rows_need_no_plugin():
    """A config.yaml seeded the way the installers seed it names only toolsets that resolve without any
    plugin loaded: a plugin platform falls back to ``hermes-<platform>`` on its own."""
    from hermes_cli.config import read_raw_config, seed_config_file
    from hermes_constants import get_hermes_home
    from toolsets import validate_toolset

    assert seed_config_file(get_hermes_home() / "config.yaml") is True
    rows = read_raw_config().get("platform_toolsets") or {}
    unresolved = {p: names for p, names in rows.items() if not all(validate_toolset(n) for n in names)}
    assert unresolved == {}


def test_config_check_accepts_seeded_rows_of_uninstalled_plugin_platforms():
    """`hermes config check` and the update-time check judge the same rows the same way: Home Assistant's
    catalog plugin is not installed here, and its seeded row is inert, not an unknown toolset."""
    from hermes_cli.toolset_validation import saved_toolset_resolver, validate_platform_toolsets

    config = {"platform_toolsets": _SEEDED_ROWS}
    assert validate_platform_toolsets(_SEEDED_ROWS, saved_toolset_resolver(config)) == []

    typo = {"platform_toolsets": {"cli": ["hermes-cli"], "homeassistant": ["hermes-homeasistant"]}}
    warnings = validate_platform_toolsets(typo["platform_toolsets"], saved_toolset_resolver(typo))
    assert any("unknown toolset 'hermes-homeasistant'" in w for w in warnings), warnings


def test_misspelt_platform_key_naming_its_own_bundle_still_warns(caplog):
    """Only a KNOWN platform's own bundle is exempt: ``telegarm: [hermes-telegarm]`` is a typo of the key,
    and both the session-time check and `hermes config check` still report it."""
    from hermes_cli.tools_config import _get_platform_tools
    from hermes_cli.toolset_validation import saved_toolset_resolver, validate_platform_toolsets

    rows = {"cli": ["hermes-cli"], "telegarm": ["hermes-telegarm"]}
    config = {"platform_toolsets": rows}
    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        _get_platform_tools(config, "telegarm")

    errors = _toolset_errors(caplog)
    assert len(errors) == 1 and "'telegarm'" in errors[0], errors
    warnings = validate_platform_toolsets(rows, saved_toolset_resolver(config))
    assert any("unknown toolset 'hermes-telegarm'" in w for w in warnings), warnings
