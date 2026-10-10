"""Migration 46→47: compression.threshold_tokens defaults back to null (ratio-only).

Contract: the briefly shipped 256000 default — copied into config.yaml by the
template seeder and ``doctor --fix``, where it reads as a user choice and keeps
capping 1M-window models at 256K — is dropped so compaction follows
compression.threshold (50% of the window) again. Any other explicit cap, and an
explicit null, are preserved. Driven through ``run_migrations`` against a temp
home.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml


def _run(tmp_path, config):
    from hermes_cli.config_migrations import run_migrations

    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    results = {"env_added": [], "config_added": [], "warnings": []}
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        run_migrations(46, results, quiet=True)
    raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    return raw.get("compression", {}), results


def test_stale_256000_default_is_dropped(tmp_path):
    from hermes_cli.config import load_config

    compression, results = _run(tmp_path, {"_config_version": 46, "compression": {
        "threshold_tokens": 256000,
    }})
    assert "threshold_tokens" not in compression, "the old default is a template copy, not a pin: drop it"
    assert any("threshold_tokens" in added for added in results["config_added"])
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        merged = load_config()["compression"]
    assert merged["threshold_tokens"] is None, "the dropped key follows the new default (ratio-only)"
    assert merged["threshold"] == 0.50


def test_explicit_cap_and_explicit_null_survive(tmp_path):
    compression, _ = _run(tmp_path, {"_config_version": 46, "compression": {
        "threshold_tokens": 100000,
    }})
    assert compression["threshold_tokens"] == 100000, "a user's own cap must never be rewritten"

    compression, _ = _run(tmp_path, {"_config_version": 46, "compression": {
        "threshold_tokens": None,
    }})
    assert compression["threshold_tokens"] is None, "an explicit null is already ratio-only: leave it"
