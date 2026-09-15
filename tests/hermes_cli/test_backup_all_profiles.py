"""Tests for the #66140 fix: pre-update snapshots cover every profile."""

import json
import re
from pathlib import Path

import pytest

import hermes_cli.backup as backup


def _mk_profile(home: Path, jobs: int = 0) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    if jobs:
        cron = home / "cron"
        cron.mkdir(exist_ok=True)
        payload = {"jobs": [{"id": f"j{i}"} for i in range(jobs)]}
        (cron / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")
    return home


@pytest.fixture()
def profiles(monkeypatch, tmp_path):
    """default (invoking) + work + sparks profile homes."""
    default_home = _mk_profile(tmp_path / "home", jobs=3)
    work = _mk_profile(tmp_path / "home" / "profiles" / "work", jobs=5)
    sparks = _mk_profile(tmp_path / "home" / "profiles" / "sparks", jobs=0)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: default_home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "home" / "profiles"
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._PROFILE_ID_RE",
        re.compile(r"^[a-z0-9][a-z0-9_-]*$"),
        raising=False,
    )
    return {"default": default_home, "work": work, "sparks": sparks}


class TestSiblingEnumeration:
    def test_excludes_invoking_profile(self, profiles):
        names = [n for n, _ in backup._sibling_profile_homes(profiles["default"])]
        assert names == ["sparks", "work"]

    def test_invoked_from_named_profile_includes_default(self, profiles):
        names = [n for n, _ in backup._sibling_profile_homes(profiles["work"])]
        assert names == ["default", "sparks"]

    def test_never_raises(self, monkeypatch, tmp_path):
        def _boom():
            raise RuntimeError("no profiles module")

        monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", _boom)
        assert backup._sibling_profile_homes(tmp_path) == []


class TestAllProfileSnapshots:
    def test_each_sibling_snapshotted_into_own_home(self, profiles):
        result = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"], keep=1
        )
        assert set(result) == {"work", "sparks"}
        for name, snap_id in result.items():
            snap_dir = profiles[name] / "state-snapshots" / snap_id
            assert snap_dir.is_dir()
            assert (snap_dir / "config.yaml").is_file()
            assert "pre-update" in snap_id
        # invoking profile untouched by THIS call
        assert not (profiles["default"] / "state-snapshots").exists()

    def test_size_cap_forwarded(self, profiles):
        big = profiles["work"] / "state.db"
        big.write_bytes(b"\x00" * 4096)
        result = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"], keep=1, max_file_size=1024
        )
        snap_dir = profiles["work"] / "state-snapshots" / result["work"]
        assert not (snap_dir / "state.db").exists()  # capped out
        assert (snap_dir / "config.yaml").is_file()  # small files captured

    def test_one_failing_sibling_does_not_block_others(self, profiles, monkeypatch):
        real = backup.create_quick_snapshot

        def _flaky(label=None, hermes_home=None, keep=None, max_file_size=None):
            if hermes_home == profiles["work"]:
                raise OSError("disk full")
            return real(
                label=label, hermes_home=hermes_home, keep=keep,
                max_file_size=max_file_size,
            )

        monkeypatch.setattr(backup, "create_quick_snapshot", _flaky)
        result = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"]
        )
        assert "sparks" in result and "work" not in result


class TestPerProfileCronRestore:
    def test_lost_jobs_restored_from_own_snapshot(self, profiles):
        snaps = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"], keep=1
        )
        # simulate the migration emptying work's jobs.json
        jobs_path = profiles["work"] / "cron" / "jobs.json"
        jobs_path.write_text(json.dumps({"jobs": []}), encoding="utf-8")

        restored = backup.restore_cron_jobs_all_profiles(
            snaps, invoking_home=profiles["default"]
        )
        assert len(restored) == 1
        assert restored[0]["profile"] == "work"
        assert restored[0]["job_count"] == 5
        live = json.loads(jobs_path.read_text(encoding="utf-8"))
        assert len(live["jobs"]) == 5

    def test_healthy_profiles_untouched(self, profiles):
        snaps = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"], keep=1
        )
        assert backup.restore_cron_jobs_all_profiles(
            snaps, invoking_home=profiles["default"]
        ) == []

    def test_empty_map_is_noop(self, profiles):
        assert backup.restore_cron_jobs_all_profiles({}) == []
