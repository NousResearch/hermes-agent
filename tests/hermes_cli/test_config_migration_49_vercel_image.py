"""Migration 48→49: Vercel Sandbox moves from the deprecated `node24` runtime to a managed image.

Contract: a saved runtime still equal to the old seeded default is dropped from config.yaml AND from
the .env mirror the setup wizard wrote, so fresh sandboxes follow terminal.vercel_image; a runtime the
user chose themselves survives untouched. Driven through ``run_migrations`` against a temp home.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml


def _run(tmp_path, runtime):
    from hermes_cli.config_migrations import run_migrations

    tmp_path.mkdir()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {"_config_version": 48, "terminal": {"backend": "vercel_sandbox", "vercel_runtime": runtime}}),
        encoding="utf-8")
    (tmp_path / ".env").write_text(f"VERCEL_TOKEN=tok\nTERMINAL_VERCEL_RUNTIME={runtime}\n", encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        os.environ.pop("TERMINAL_VERCEL_RUNTIME", None)
        run_migrations(48, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
    terminal = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))["terminal"]
    return terminal, (tmp_path / ".env").read_text(encoding="utf-8")


def test_seeded_default_runtime_is_dropped_everywhere_and_a_user_pin_survives(tmp_path):
    from hermes_cli.config import load_config
    from hermes_cli.config_defaults import DEFAULT_VERCEL_IMAGE, LEGACY_VERCEL_RUNTIME

    terminal, dotenv = _run(tmp_path / "seeded", LEGACY_VERCEL_RUNTIME)
    assert "vercel_runtime" not in terminal, "the seeded default is the template copied, not a choice"
    assert "TERMINAL_VERCEL_RUNTIME" not in dotenv and "VERCEL_TOKEN=tok" in dotenv
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / "seeded")}):
        merged = load_config()["terminal"]
    assert merged["vercel_image"] == DEFAULT_VERCEL_IMAGE and not merged["vercel_runtime"]

    terminal, dotenv = _run(tmp_path / "pinned", "python3.13")
    assert terminal["vercel_runtime"] == "python3.13", "a runtime the user chose is never rewritten"
    assert "TERMINAL_VERCEL_RUNTIME=python3.13" in dotenv
