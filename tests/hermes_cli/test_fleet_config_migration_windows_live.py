"""LIVE Windows E2E for the fleet-wide config migration (wine2e lane).

Real profile homes on the Windows filesystem, real migration pipeline,
fresh-process semantics via HERMES_HOME env — mirrors the Linux live E2E.
"""
import sys
from pathlib import Path

import pytest
import yaml

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="live Windows E2E")


def test_fleet_config_migration_live_windows(tmp_path, monkeypatch):
    active = tmp_path / "hermes-home"
    profiles = active / "profiles"
    for name, ver in [("research", 12), ("work", 25)]:
        home = profiles / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump({"_config_version": ver, "model": {"provider": "nous"}}),
            encoding="utf-8",
        )
    active.mkdir(exist_ok=True)
    (active / "config.yaml").write_text(
        yaml.safe_dump({"_config_version": 12}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(active))

    import hermes_cli.update_cmd as update_cmd
    from hermes_cli.config import DEFAULT_CONFIG

    latest = int(DEFAULT_CONFIG["_config_version"])
    migrated = update_cmd._migrate_sibling_profile_configs()

    by_name = {m[0]: m for m in migrated}
    assert set(by_name) == {"research", "work"}, migrated
    assert by_name["research"][1] == 12 and by_name["research"][2] == latest

    for name in ("research", "work"):
        on_disk = yaml.safe_load((profiles / name / "config.yaml").read_text())
        assert on_disk["_config_version"] == latest
        assert on_disk["model"]["provider"] == "nous"

    # active untouched; idempotent second run
    assert yaml.safe_load((active / "config.yaml").read_text())["_config_version"] == 12
    assert update_cmd._migrate_sibling_profile_configs() == []
