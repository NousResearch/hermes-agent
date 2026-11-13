"""Config migrations survive malformed nested values (#116593).

Hand-edited or legacy config.yaml files hold scalars where a migration step expects a
mapping. Each step must guard the shapes it indexes, and ``run_migrations`` must isolate
a step that still raises so one bad value cannot wedge ``hermes config migrate`` /
``hermes update`` and leave the config unversioned.
"""

import logging
import os
from unittest.mock import patch

import pytest
import hermes_agent.hermes_yaml as yaml


def _write_config(tmp_path, config):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _read_config(tmp_path):
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("current_ver", "config", "path", "expected"),
    [
        # _migrate_to_12: a non-string custom_providers name falls back to the hostname key.
        (11, {"custom_providers": [{"name": 5, "base_url": "https://api.example.com/v1"}]},
         ("providers", "api-example-com", "api"), "https://api.example.com/v1"),
        # _migrate_to_14: a mapping stt.provider is treated as the "local" default.
        (13, {"stt": {"model": "tiny", "provider": {"nested": True}}},
         ("stt", "local", "model"), "tiny"),
        # _migrate_to_14: a scalar stt.<section> is replaced, not indexed.
        (13, {"stt": {"model": "base", "provider": "openai", "openai": 5}},
         ("stt", "openai", "model"), "base"),
        # _migrate_to_16: a scalar display.platforms.<plat> slot is replaced, not indexed.
        (15, {"display": {"tool_progress_overrides": {"telegram": "all"}, "platforms": {"telegram": 5}}},
         ("display", "platforms", "telegram", "tool_progress"), "all"),
        # _migrate_to_17: a scalar auxiliary / auxiliary.compression is rebuilt as a mapping.
        (16, {"compression": {"summary_model": "fast-model"}, "auxiliary": 5},
         ("auxiliary", "compression", "model"), "fast-model"),
        (16, {"compression": {"summary_model": "fast-model"}, "auxiliary": {"compression": "x"}},
         ("auxiliary", "compression", "model"), "fast-model"),
    ],
    ids=["v12-name-int", "v14-provider-map", "v14-section-scalar", "v16-platform-scalar",
         "v17-auxiliary-scalar", "v17-compression-scalar"],
)
def test_malformed_nested_value_is_migrated_not_crashed(tmp_path, current_ver, config, path, expected):
    """Each cited step replaces the malformed slot and still lands the migrated value."""
    from hermes_agent.hermes_cli.config_migrations import run_migrations

    _write_config(tmp_path, {"_config_version": current_ver, **config})
    results = {"env_added": [], "config_added": [], "warnings": []}
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        run_migrations(current_ver, results, quiet=True)

    node = _read_config(tmp_path)
    for key in path:
        node = node[key]
    assert node == expected
    assert not results["warnings"], "a guarded shape must migrate cleanly, not be skipped"


def test_failing_step_stops_the_ladder_and_is_retried(tmp_path, caplog):
    """A raising step stops the pass: the stamp stays at the last step that succeeded, so the
    next pass retries it, and the steps after it wait. v21 raises on a nested list in
    ``plugins.disabled``; v53 (a one-time 3600 -> 90 rewrite) must not run past it, or the pass
    that retries v21 would run v53 again over a 3600 the user set back on purpose."""
    from hermes_agent.hermes_cli.config import DEFAULT_CONFIG, migrate_config

    _write_config(tmp_path, {"_config_version": 16, "compression": {"summary_model": "m"},
                             "plugins": {"disabled": [["x"]]}, "agent": {"clarify_timeout": 3600}})
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
            caplog.at_level(logging.WARNING, logger="hermes_agent.hermes_cli.config_migrations"):
        results = migrate_config(interactive=False, quiet=True)

    raw = _read_config(tmp_path)
    assert results["failed_step"] == 21
    assert results["failed_warning"].startswith("config migration to v21 failed"), \
        "boot and doctor show this one warning: they must not search the list for it"
    assert raw["_config_version"] == 20, "stamped at the last step that succeeded, not past v21"
    assert raw["auxiliary"]["compression"]["model"] == "m", "v17, before the failure, is kept"
    assert raw["agent"]["clarify_timeout"] == 3600, "v53 waits for v21"
    assert [r.getMessage() for r in caplog.records if "v21 failed" in r.getMessage()], \
        "quiet callers drop results: the failure must reach the log"

    raw["plugins"]["disabled"] = []
    _write_config(tmp_path, raw)
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        results = migrate_config(interactive=False, quiet=True)
    raw = _read_config(tmp_path)
    assert "failed_step" not in results
    assert "enabled" in raw["plugins"], "v21 ran on the retry"
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]


def test_failing_step_on_an_unversioned_config_leaves_it_unversioned(tmp_path):
    """An unversioned file gets only the legacy-key steps. A stamp after a failed one would hand
    it the whole ladder, or (latest) hide the failed step forever: it stays unstamped."""
    from hermes_agent.hermes_cli import config_migrations
    from hermes_agent.hermes_cli.config import migrate_config

    def _boom(results, quiet):
        raise RuntimeError("boom")

    _write_config(tmp_path, {"model": {"default": "x/y"}})
    ladder = tuple((v, _boom if v == 14 else fn) for v, fn in config_migrations.MIGRATIONS)
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
            patch.object(config_migrations, "MIGRATIONS", ladder):
        results = migrate_config(interactive=False, quiet=True)

    assert results["failed_step"] == 14
    assert "_config_version" not in _read_config(tmp_path)


def test_successful_migration_warning_still_stamps_latest_version(tmp_path):
    """The real v38 Relay cleanup warns while succeeding; it must still stamp latest."""
    from hermes_agent.hermes_cli import config_migrations
    from hermes_agent.hermes_cli.config import migrate_config

    _write_config(tmp_path, {
        "_config_version": 37,
        "model": {"default": "x/y"},
        "plugins": {"enabled": ["observability/nemo_relay"]},
    })
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        results = migrate_config(interactive=False, quiet=True)

    assert any("Removed legacy Relay plugin" in warning for warning in results["warnings"])
    migrated = _read_config(tmp_path)
    assert "observability/nemo_relay" not in migrated.get("plugins", {}).get("enabled", [])
    assert migrated["_config_version"] == config_migrations.MIGRATIONS[-1][0]


def test_config_migrate_command_reports_a_stopped_pass(tmp_path, capsys):
    """`hermes config migrate` must not say "Configuration updated!" for a pass that stopped,
    and exits 1 so scripts see it."""
    import pytest
    from hermes_agent.hermes_cli.config import _cmd_config_migrate

    # A new default key is listed in config_added, as on a real upgrade; then v21 raises.
    _write_config(tmp_path, {"_config_version": 20, "plugins": {"disabled": [["x"]]}})
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
            patch("hermes_agent.hermes_cli.config.get_missing_env_vars", return_value=[]), \
            patch("hermes_agent.hermes_cli.config.get_missing_config_fields",
                  return_value=[{"key": "display.zz_new", "default": 1}]), \
            pytest.raises(SystemExit) as exited:
        _cmd_config_migrate(None)
    out = capsys.readouterr().out
    assert exited.value.code == 1
    assert "Configuration updated" not in out
    assert "v21 failed" in out


def test_console_config_migrate_error_says_why(tmp_path):
    """The web console drops a failing command's printed output, so the error itself carries the cause."""
    import pytest
    from hermes_agent.hermes_cli.console_engine import ConsoleCommandError, _config_migrate

    _write_config(tmp_path, {"_config_version": 20, "plugins": {"disabled": [["x"]]}})
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), pytest.raises(ConsoleCommandError) as err:
        _config_migrate(None, [])
    assert "v21 failed" in str(err.value) and "unhashable" in str(err.value)
