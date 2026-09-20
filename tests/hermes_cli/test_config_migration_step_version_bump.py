"""Per-step ``_config_version`` bumps in the migration ladder.

``run_migrations()`` selected its steps from the on-disk version captured once
before the ladder started, and the version key was written only by the caller
after every step returned. An abort mid-ladder (e.g. the call-time import failure
this PR's reload leg fixes) therefore persisted earlier steps' body writes while
leaving ``_config_version`` at the pre-run value — "did the version advance?"
could not tell untouched from half-applied, and a retry re-ran the whole ladder.

History:
- #111286 review (kriscolab): a live fleet install reproduced the predicted
  failure and diffed the updater's own backup against the live file — the
  43→44 rewrites had landed, ``model_catalog.ttl_hours`` had been dropped, yet
  ``_config_version`` still read 37. Asked for the version key to advance per
  applied step (or a transactional body write).

The fix bumps the on-disk key after each *version group* fully succeeds
(entries sharing a target version, e.g. the two 44 steps), so a version is only
ever stamped over a complete set of rewrites and a retry resumes from the
breakpoint.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml


def _write_config(tmp_path, config):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _read_config(tmp_path):
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


def _make_step(tag):
    """A migration-shaped step: mutate body through the real persist path + run counter."""
    from hermes_cli import config_migrations

    def step(results, quiet):
        config = config_migrations.read_raw_config()
        key = f"_runs_{tag}"
        config[key] = config.get(key, 0) + 1
        config[f"_body_{tag}"] = "written"
        config_migrations._persist_migration(config)

    return step


def _aborting_step(results, quiet):
    raise RuntimeError("simulated mid-ladder abort")


def _run_ladder(tmp_path, current_ver, monkeypatch, table):
    from hermes_cli import config_migrations

    monkeypatch.setattr(config_migrations, "MIGRATIONS", tuple(table))
    results = {"env_added": [], "config_added": [], "warnings": []}
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        with pytest.raises(RuntimeError, match="mid-ladder abort"):
            config_migrations.run_migrations(current_ver, results, quiet=True)
    return results


class TestStepVersionBump:
    def test_abort_after_completed_group_parks_version_there(self, tmp_path, monkeypatch):
        """Steps 13a+13b applied and stamped; the aborting 14 step must not unwind 13."""
        table = [(13, _make_step("13a")), (13, _make_step("13b")), (14, _aborting_step)]
        _write_config(tmp_path, {"_config_version": 12})
        _run_ladder(tmp_path, 12, monkeypatch, table)

        raw = _read_config(tmp_path)
        assert raw["_config_version"] == 13
        assert raw["_runs_13a"] == 1 and raw["_runs_13b"] == 1

    def test_abort_inside_version_group_keeps_previous_version(self, tmp_path, monkeypatch):
        """A version is stamped only over a COMPLETE group: 14a landing + 14b aborting
        must leave the key at 13 so the retry re-runs both 14 steps (idempotence)."""
        table = [(13, _make_step("13")), (14, _make_step("14a")), (14, _aborting_step)]
        _write_config(tmp_path, {"_config_version": 13})
        _run_ladder(tmp_path, 13, monkeypatch, table)

        raw = _read_config(tmp_path)
        assert raw["_config_version"] == 13
        assert raw["_runs_14a"] == 1  # body write landed, but the version stays behind

    def test_retry_after_abort_resumes_from_breakpoint(self, tmp_path, monkeypatch):
        """The parked version gates the retry: completed groups never re-run."""
        from hermes_cli import config_migrations

        table = [(13, _make_step("13")), (14, _aborting_step)]
        _write_config(tmp_path, {"_config_version": 12})
        _run_ladder(tmp_path, 12, monkeypatch, table)
        assert _read_config(tmp_path)["_config_version"] == 13

        monkeypatch.setattr(
            config_migrations, "MIGRATIONS", ((13, _make_step("13")), (14, _make_step("14"))))
        results = {"env_added": [], "config_added": [], "warnings": []}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            # Re-read the parked version the way migrate_config() would.
            current = _read_config(tmp_path)["_config_version"]
            config_migrations.run_migrations(current, results, quiet=True)

        raw = _read_config(tmp_path)
        assert raw["_runs_13"] == 1  # not re-run by the retry
        assert raw["_runs_14"] == 1
        assert raw["_config_version"] == 14

    def test_successful_ladder_stamps_every_group(self, tmp_path, monkeypatch):
        from hermes_cli import config_migrations

        table = [
            (13, _make_step("13")),
            (14, _make_step("14a")),
            (14, _make_step("14b")),
            (15, _make_step("15")),
        ]
        monkeypatch.setattr(config_migrations, "MIGRATIONS", tuple(table))
        _write_config(tmp_path, {"_config_version": 12})
        results = {"env_added": [], "config_added": [], "warnings": []}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config_migrations.run_migrations(12, results, quiet=True)

        raw = _read_config(tmp_path)
        assert raw["_config_version"] == 15
        assert raw["_runs_13"] == 1 and raw["_runs_14a"] == 1 and raw["_runs_15"] == 1

    def test_real_ladder_abort_parks_version_at_44(self, tmp_path, monkeypatch):
        """kriscolab's fleet scenario on the real registry: the 43→44 curator rewrites
        land, the 44→45 step aborts — the key must read 44, not the pre-run 43/37."""
        from hermes_cli import config_migrations

        table = tuple(
            (45, _aborting_step) if ver == 45 else (ver, fn)
            for ver, fn in config_migrations.MIGRATIONS
        )
        monkeypatch.setattr(config_migrations, "MIGRATIONS", table)
        _write_config(
            tmp_path,
            {
                "_config_version": 43,
                "curator": {"stale_after_days": 30, "archive_after_days": 90},
            },
        )
        results = {"env_added": [], "config_added": [], "warnings": []}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with pytest.raises(RuntimeError, match="mid-ladder abort"):
                config_migrations.run_migrations(43, results, quiet=True)

        raw = _read_config(tmp_path)
        assert raw["_config_version"] == 44
        assert raw["curator"]["stale_after_days"] == 14
        assert raw["curator"]["archive_after_days"] == 30
