"""Migration 52→53: prompt_caching.cache_ttl defaults to "auto".

Contract: a saved ``cache_ttl: "5m"`` (the old default, copied in by the template seeder) is dropped
so the file follows the new default; ``1h``, ``auto`` and an off value are the user's and survive.
Driven through ``run_migrations`` against a temp home.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml
import pytest


def _run(home, ttl):
    from hermes_cli.config_migrations import run_migrations

    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"_config_version": 52, "prompt_caching": {"cache_ttl": ttl}}), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}, clear=False):
        run_migrations(52, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")).get("prompt_caching") or {}


def test_seeded_old_default_is_dropped_and_follows_auto(tmp_path):
    from hermes_cli.config import load_config

    home = tmp_path / "seeded"
    assert "cache_ttl" not in _run(home, "5m"), "the seeded default is the template copied, not a choice"
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        assert load_config()["prompt_caching"]["cache_ttl"] == "auto"


@pytest.mark.parametrize("ttl", ["1h", "auto", "off", False])
def test_other_values_survive(tmp_path, ttl):
    assert _run(tmp_path / "pinned", ttl)["cache_ttl"] == ttl
