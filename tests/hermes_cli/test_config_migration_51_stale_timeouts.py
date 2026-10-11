"""Migration 50→51: installs seeded by the early config writer stay pinned to old timeout defaults.

Contract: a saved approvals.timeout still equal to 60, or an agent.clarify_timeout still equal to
600, is the June-2026 template copied into config.yaml — never a choice — so the key is dropped and
the runtime default (300 / 3600) applies. Any other explicit value survives untouched, and
approvals.mode is never rewritten (a security posture the user must change on purpose). Driven
through ``run_migrations`` against a temp home.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml


def _run(tmp_path, approvals, agent):
    from hermes_cli.config_migrations import run_migrations

    tmp_path.mkdir()
    config = {"_config_version": 50}
    if approvals is not None:
        config["approvals"] = approvals
    if agent is not None:
        config["agent"] = agent
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        run_migrations(
            50, {"env_added": [], "config_added": [], "warnings": []}, quiet=True
        )
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


def test_seeded_old_timeouts_are_dropped_and_the_runtime_default_applies(tmp_path):
    from hermes_cli.config import load_config

    migrated = _run(
        tmp_path / "seeded",
        approvals={"mode": "manual", "timeout": 60},
        agent={"clarify_timeout": 600, "status_interval": 180},
    )
    assert "timeout" not in migrated["approvals"], (
        "the seeded 60 is the template copied, not a choice"
    )
    assert migrated["approvals"]["mode"] == "manual", (
        "a security posture is never rewritten"
    )
    assert "clarify_timeout" not in migrated["agent"]
    assert migrated["agent"]["status_interval"] == 180, (
        "unrelated keys in the same section stay"
    )
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / "seeded")}):
        merged = load_config()
    assert merged["approvals"]["timeout"] == 300
    assert merged["agent"]["clarify_timeout"] == 3600


def test_explicit_user_values_survive(tmp_path):
    migrated = _run(
        tmp_path / "pinned",
        approvals={"mode": "smart", "timeout": 90},
        agent={"clarify_timeout": 1200},
    )
    assert migrated["approvals"]["timeout"] == 90, (
        "a timeout the user chose is never rewritten"
    )
    assert migrated["agent"]["clarify_timeout"] == 1200


def test_missing_sections_or_keys_are_left_alone(tmp_path):
    migrated = _run(tmp_path / "fresh", approvals={"mode": "manual"}, agent=None)
    assert migrated["approvals"] == {"mode": "manual"}
    assert "agent" not in migrated
