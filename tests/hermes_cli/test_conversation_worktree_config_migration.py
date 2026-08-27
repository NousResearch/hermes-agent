"""Config-v39 migration coverage for conversation worktree policy ownership."""

from __future__ import annotations

import os
from copy import deepcopy
from unittest.mock import patch

import yaml

from hermes_cli.config import migrate_config


LEGACY_POLICY = {
    "enabled": True,
    "source_worktree": "/srv/hermes/stable",
    "worktree_root": "/srv/hermes/conversations",
    "branch_prefix": "hermes/session",
    "bootstrap": True,
    "bootstrap_command": ["python3", "scripts/bootstrap.py"],
    "bootstrap_timeout": 420,
    "create_timeout": 90,
    "retain_until_explicit_cleanup": True,
}


def _migrate(tmp_path, config):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        migrate_config(interactive=False, quiet=True)

    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_v39_moves_legacy_desktop_policy_without_changing_unrelated_fields(tmp_path):
    original = {
        "_config_version": 38,
        "model": "local/test-model",
        "compression": {"threshold": 0.61},
        "auxiliary": {
            "compression": {
                "provider": "main",
                "model": "local/fast-summary",
                "reasoning_effort": "none",
                "max_output_tokens": 2048,
            }
        },
        "desktop": {
            "conversation_worktree": deepcopy(LEGACY_POLICY),
            "repo_scan_enabled": False,
            "electron_flags": ["--disable-gpu"],
        },
    }

    migrated = _migrate(tmp_path, original)

    assert migrated["_config_version"] == 40
    assert migrated["conversation_worktree"] == LEGACY_POLICY
    assert migrated["desktop"] == {
        "repo_scan_enabled": False,
        "electron_flags": ["--disable-gpu"],
    }
    assert migrated["model"] == original["model"]
    assert migrated["compression"] == original["compression"]
    assert migrated["auxiliary"] == original["auxiliary"]


def test_v39_explicit_top_level_policy_wins_and_legacy_duplicate_is_removed(tmp_path):
    canonical = {
        **LEGACY_POLICY,
        "source_worktree": "/srv/hermes/canonical-stable",
        "branch_prefix": "hermes/canonical",
    }
    original = {
        "_config_version": 38,
        "conversation_worktree": deepcopy(canonical),
        "desktop": {
            "conversation_worktree": deepcopy(LEGACY_POLICY),
            "repo_scan_roots": ["/srv/projects"],
        },
        "session": {"terminal_continue": False},
    }

    migrated = _migrate(tmp_path, original)

    assert migrated["_config_version"] == 40
    assert migrated["conversation_worktree"] == canonical
    assert migrated["desktop"] == {"repo_scan_roots": ["/srv/projects"]}
    assert migrated["session"] == original["session"]
