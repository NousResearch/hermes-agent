"""Migration 46→47: the container sandbox default becomes nousresearch/hermes-sandbox:desktop.

Contract: a saved image still equal to the OLD default follows the new default (so Bot Screen
lands inside the sandbox for everyone who never chose an image), while an image the user
pinned themselves is never touched. Driven through ``run_migrations`` against a temp home.
"""

import os
from unittest.mock import patch

import yaml


def _run(tmp_path, config):
    from hermes_cli.config_migrations import run_migrations

    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        run_migrations(46, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))["terminal"]


def test_stale_default_moves_and_a_user_pin_survives(tmp_path):
    from hermes_cli.config_defaults import DEFAULT_SANDBOX_IMAGE, LEGACY_SANDBOX_IMAGE

    terminal = _run(tmp_path, {"_config_version": 46, "terminal": {
        "backend": "docker",
        "docker_image": LEGACY_SANDBOX_IMAGE,
        "singularity_image": f"docker://{LEGACY_SANDBOX_IMAGE}",
        "modal_image": "ghcr.io/me/custom:1",
    }})
    assert terminal["docker_image"] == DEFAULT_SANDBOX_IMAGE
    assert terminal["singularity_image"] == f"docker://{DEFAULT_SANDBOX_IMAGE}"
    assert terminal["modal_image"] == "ghcr.io/me/custom:1", "a user's own image must never be rewritten"
    assert "daytona_image" not in terminal, "an unset key inherits the default at read time, no write"
