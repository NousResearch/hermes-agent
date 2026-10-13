"""Migration 50→51: installs seeded by the early config writer stay pinned to old defaults.

Contract: a saved value still equal to the June-2026 default (approvals.timeout 60,
agent.clarify_timeout 600, agent.restart_drain_timeout 180,
compression.hygiene_hard_message_limit 400, delegation.child_timeout_seconds 600,
curator.prune_builtins true, updates.pre_update_backup false) is the template copied into
config.yaml — never a choice — so the key is dropped and the runtime default applies. Any other
explicit value survives untouched. approvals.mode is never rewritten (a security posture the user
must change on purpose); it gets a one-time notice naming the unset command instead. Driven
through ``run_migrations`` against a temp home.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml


def _run_config(tmp_path, config, quiet=False):
    from hermes_cli.config_migrations import run_migrations

    tmp_path.mkdir()
    config = dict({"_config_version": 50}, **config)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        run_migrations(
            50, {"env_added": [], "config_added": [], "warnings": []}, quiet
        )
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


def _run(tmp_path, approvals, agent):
    config = {}
    if approvals is not None:
        config["approvals"] = approvals
    if agent is not None:
        config["agent"] = agent
    return _run_config(tmp_path, config)


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


def test_other_seeded_stale_defaults_are_dropped(tmp_path):
    from hermes_cli.config import load_config

    migrated = _run_config(
        tmp_path / "seeded",
        {
            "agent": {"restart_drain_timeout": 180},
            "compression": {"hygiene_hard_message_limit": 400},
            "delegation": {"child_timeout_seconds": 600},
            "curator": {"prune_builtins": True},
            "updates": {"pre_update_backup": False},
        },
    )
    assert migrated["agent"] == {}
    assert migrated["compression"] == {}
    assert migrated["delegation"] == {}
    assert migrated["curator"] == {}
    assert migrated["updates"] == {}
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / "seeded")}):
        merged = load_config()
    assert merged["agent"]["restart_drain_timeout"] == 0
    assert merged["compression"]["hygiene_hard_message_limit"] == 5000
    assert merged["delegation"]["child_timeout_seconds"] == 0
    assert merged["curator"]["prune_builtins"] is False
    assert merged["updates"]["pre_update_backup"] == "quick"


def test_other_explicit_values_survive(tmp_path):
    migrated = _run_config(
        tmp_path / "pinned",
        {
            "agent": {"restart_drain_timeout": 60},
            "compression": {"hygiene_hard_message_limit": 4000},
            "delegation": {"child_timeout_seconds": 300},
        },
    )
    assert migrated["agent"]["restart_drain_timeout"] == 60
    assert migrated["compression"]["hygiene_hard_message_limit"] == 4000
    assert migrated["delegation"]["child_timeout_seconds"] == 300


def test_manual_mode_gets_a_notice_not_a_rewrite(tmp_path, capsys):
    migrated = _run_config(
        tmp_path / "notice",
        {"approvals": {"mode": "manual", "timeout": 60}},
        quiet=False,
    )
    assert migrated["approvals"] == {"mode": "manual"}, (
        "the security posture is never flipped, only announced"
    )
    assert "hermes config unset approvals.mode" in capsys.readouterr().out

    migrated = _run_config(
        tmp_path / "smart", {"approvals": {"mode": "smart"}}, quiet=False
    )
    assert migrated["approvals"] == {"mode": "smart"}
    assert "hermes config unset approvals.mode" not in capsys.readouterr().out
