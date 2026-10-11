"""Migration 50→51: installs seeded before defaults stopped being persisted re-baseline.

Contract: a config.yaml still carrying a value equal to an OLD default (the whole default set
was written literally by pre-stripping installs, #136754) is rewritten to the current default;
a value the user actually chose survives untouched, and a missing key is never materialised.
``approvals.mode: manual`` is never rewritten — flipping a security posture silently is not the
ladder's call — but a one-time notice names the unset command. Driven through ``run_migrations``
against a temp home.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml

_SEEDS = {
    "approvals": {"mode": "manual", "timeout": 60},
    "agent": {"clarify_timeout": 600, "restart_drain_timeout": 180},
    "compression": {"hygiene_hard_message_limit": 400},
    "delegation": {"child_timeout_seconds": 600},
    "curator": {"prune_builtins": True},
    "updates": {"pre_update_backup": False},
}


def _run(tmp_path, config, quiet=False):
    from hermes_cli.config_migrations import run_migrations

    tmp_path.mkdir()
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(dict({"_config_version": 50}, **config)), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        run_migrations(50, {"env_added": [], "config_added": [], "warnings": []}, quiet)
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


def test_stale_seeds_rebaselined_and_manual_mode_notice_kept(tmp_path, capsys):
    migrated = _run(tmp_path / "seeded", _SEEDS, quiet=False)

    assert migrated["approvals"]["timeout"] == 300
    assert migrated["agent"]["clarify_timeout"] == 3600
    assert migrated["agent"]["restart_drain_timeout"] == 0
    assert migrated["compression"]["hygiene_hard_message_limit"] == 5000
    assert migrated["delegation"]["child_timeout_seconds"] == 0
    assert migrated["curator"]["prune_builtins"] is False
    assert migrated["updates"]["pre_update_backup"] == "quick"
    # A security posture is never silently flipped; the notice names the escape hatch.
    assert migrated["approvals"]["mode"] == "manual"
    out = capsys.readouterr().out
    assert "hermes config unset approvals.mode" in out


def test_user_chosen_values_survive(tmp_path, capsys):
    chosen = {
        "approvals": {"mode": "smart", "timeout": 90},
        "agent": {"clarify_timeout": 1200, "restart_drain_timeout": 60},
        "compression": {"hygiene_hard_message_limit": 4000},
        "delegation": {"child_timeout_seconds": 300},
        "curator": {"prune_builtins": True},
        "updates": {"pre_update_backup": False},
    }
    migrated = _run(tmp_path / "chosen", chosen, quiet=False)

    assert migrated["approvals"]["timeout"] == 90
    assert migrated["agent"]["clarify_timeout"] == 1200
    assert migrated["agent"]["restart_drain_timeout"] == 60
    assert migrated["compression"]["hygiene_hard_message_limit"] == 4000
    assert migrated["delegation"]["child_timeout_seconds"] == 300
    # True/False literals equal to a default are indistinguishable from a seed, so they move
    # with the default; only values that differ from the old seed are provably chosen.
    assert migrated["curator"]["prune_builtins"] is False
    assert migrated["updates"]["pre_update_backup"] == "quick"
    assert "hermes config unset approvals.mode" not in capsys.readouterr().out


def test_missing_sections_are_not_materialised(tmp_path):
    migrated = _run(
        tmp_path / "sparse",
        {"approvals": {"mode": "manual", "timeout": 60}},
    )

    assert migrated["approvals"]["timeout"] == 300
    for section in ("agent", "compression", "delegation", "curator", "updates"):
        assert section not in migrated, f"{section} must stay absent and inherit at read time"


def test_current_default_values_are_untouched(tmp_path):
    current = {
        "approvals": {"mode": "smart", "timeout": 300},
        "agent": {"clarify_timeout": 3600},
        "updates": {"pre_update_backup": "quick"},
    }
    migrated = _run(tmp_path / "current", current)

    assert migrated["approvals"] == {"mode": "smart", "timeout": 300}
    assert migrated["agent"]["clarify_timeout"] == 3600
    assert migrated["updates"]["pre_update_backup"] == "quick"
