"""Tests for config migration on the \"Already up to date\" repair path.

Covers ``_maybe_migrate_config_on_current``, added for #91360: a previous
update attempt can pull new code onto disk and then fail before reaching
the config-migration block (e.g. PyPI timeout during dependency sync); the
retry then enters the ``commit_count == 0`` branch and returns early,
skipping config migration entirely. The fresh code (which may require a
newer ``_config_version``) keeps running against the old config and the
next Hermes launch refuses to start.
"""

from __future__ import annotations

from unittest.mock import patch

import hermes_cli.update_cmd as update_cmd


def _run(current: int, latest: int):
    """Run _maybe_migrate_config_on_current with mocked config checks.

    Returns (stdout, migrate_calls).
    """
    migrate_calls = []

    def _fake_migrate(interactive=False, quiet=False):
        migrate_calls.append((interactive, quiet))
        return {"env_added": [], "config_added": [], "warnings": []}

    with patch.object(
        update_cmd, "_run_config_check_fresh", return_value=(current, latest)
    ), patch.object(
        update_cmd, "_run_migrate_config_fresh", side_effect=_fake_migrate
    ) as mig:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            update_cmd._maybe_migrate_config_on_current(lambda msg: None)
        return buf.getvalue(), migrate_calls


def test_migrates_when_config_behind():
    """Version bump on the repair path must be applied silently."""
    out, calls = _run(current=37, latest=38)
    assert "v37 → v38" in out
    assert "Config format updated" in out
    assert calls == [(False, True)]  # non-interactive, quiet


def test_noop_when_config_current():
    """No migration when the config version is already current."""
    out, calls = _run(current=38, latest=38)
    assert out == ""
    assert calls == []


def test_noop_when_config_ahead():
    """No migration when local config is newer than the code's default."""
    out, calls = _run(current=39, latest=38)
    assert out == ""
    assert calls == []


def test_surfaces_migration_warnings():
    """Warnings from a quiet migration must be re-surfaced (#86656)."""

    def _fake_migrate(interactive=False, quiet=False):
        return {
            "env_added": [],
            "config_added": [],
            "warnings": ["personality reset: kawaii → default"],
        }

    with patch.object(
        update_cmd, "_run_config_check_fresh", return_value=(37, 38)
    ), patch.object(
        update_cmd, "_run_migrate_config_fresh", side_effect=_fake_migrate
    ):
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            update_cmd._maybe_migrate_config_on_current(lambda msg: None)
        out = buf.getvalue()

    assert "personality reset" in out


def test_check_failure_is_silent():
    """A config-check failure must not break the repair path."""
    with patch.object(
        update_cmd, "_run_config_check_fresh", side_effect=RuntimeError("boom")
    ), patch.object(
        update_cmd, "_run_migrate_config_fresh", return_value={}
    ) as mig:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # Should not raise and should not attempt migration.
            update_cmd._maybe_migrate_config_on_current(lambda msg: None)
        assert buf.getvalue() == ""
        mig.assert_not_called()