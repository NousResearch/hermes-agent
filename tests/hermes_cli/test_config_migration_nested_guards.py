"""Config migrations survive malformed nested values (#116593).

Hand-edited or legacy config.yaml files hold scalars where a migration step expects a
mapping. Each step must guard the shapes it indexes, and ``run_migrations`` must isolate
a step that still raises so one bad value cannot wedge ``hermes config migrate`` /
``hermes update`` and leave the config unversioned.
"""

import os
from unittest.mock import patch

import pytest
import yaml


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
    from hermes_cli.config_migrations import run_migrations

    _write_config(tmp_path, {"_config_version": current_ver, **config})
    results = {"env_added": [], "config_added": [], "warnings": []}
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        run_migrations(current_ver, results, quiet=True)

    node = _read_config(tmp_path)
    for key in path:
        node = node[key]
    assert node == expected
    assert not results["warnings"], "a guarded shape must migrate cleanly, not be skipped"


def test_failing_step_is_skipped_with_warning_and_config_still_migrates(tmp_path):
    """``migrate_config`` (the ``hermes config migrate`` / ``hermes update`` path) keeps going
    past a raising step, records the skip in ``warnings`` and stamps the latest version."""
    from hermes_cli import config_migrations
    from hermes_cli.config import migrate_config

    def _boom(results, quiet):
        raise RuntimeError("boom")

    _write_config(tmp_path, {"_config_version": 12, "model": {"default": "x/y"}})
    ladder = tuple((v, _boom if v == 13 else fn) for v, fn in config_migrations.MIGRATIONS)
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
            patch.object(config_migrations, "MIGRATIONS", ladder):
        results = migrate_config(interactive=False, quiet=True)

    assert any(w.startswith("config migration to v13 failed and was skipped") for w in results["warnings"])
    assert _read_config(tmp_path)["_config_version"] == config_migrations.MIGRATIONS[-1][0]
