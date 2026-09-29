"""Migration 49→50: a saved compression timeout of 120 (the old default) is dropped, others stay.

Regression for #126769: the floor now yields to a saved timeout, so the old default must not.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml


def _run(tmp_path, compression):
    from hermes_cli.config_migrations import run_migrations

    tmp_path.mkdir()
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"_config_version": 49, "auxiliary": {"compression": compression}}), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        run_migrations(49, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
        from agent.auxiliary_client import _effective_aux_timeout
        effective = _effective_aux_timeout("compression", None)
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))["auxiliary"]["compression"]
    return saved, effective


def test_saved_default_is_dropped_and_a_chosen_timeout_survives(tmp_path):
    saved, effective = _run(tmp_path / "default", {"provider": "openrouter", "timeout": 120})
    assert "timeout" not in saved and saved["provider"] == "openrouter"
    assert effective > 120, "the old default is the template copied, not a choice: the floor applies again"

    saved, effective = _run(tmp_path / "chosen", {"timeout": 90})
    assert saved["timeout"] == 90 and effective == 90
