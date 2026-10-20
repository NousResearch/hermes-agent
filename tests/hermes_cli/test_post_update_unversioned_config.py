"""Boot maintenance must migrate an unversioned config, not refuse it as ancient.

#123555: ``step_migrate_config`` compared ``check_config_version()`` against the v12 support
floor. That helper reports a config with no ``_config_version`` key as **v0** — but a missing
stamp means "never stamped", not "from before v12". Template-seeded homes were unversioned until
mid-August (including Desktop's ``--non-interactive`` bootstrap), so boot maintenance logged
"This config predates version 12 (~2 years old)" and skipped a migration that ``migrate_config()``
performs without complaint.

``scripts/docker_config_migrate.py`` already draws this distinction after #121251: it reads the
raw stamp and only refuses when the stamp is present *and* below the floor. This is the sibling
that was missed.

These tests drive the real config path against a temp ``HERMES_HOME`` — no mocked version
readers, so a regression in the predicate cannot be hidden behind a stub.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp ``HERMES_HOME`` with a real (unstamped) config.yaml on disk."""
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return h


def _write(home: Path, body: str) -> Path:
    path = home / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_unversioned_config_is_migrated_not_refused(home):
    """A config with no ``_config_version`` is a current-schema file that was never stamped.
    It must migrate, exactly as ``migrate_config()`` would."""
    from hermes_cli.post_update import step_migrate_config

    path = _write(home, "model:\n  default: anthropic/claude-sonnet-5\ndisplay:\n  streaming: true\n")

    result = step_migrate_config()

    assert result.get("skipped") != "below-support-floor", (
        "an unversioned config must not be refused as predating the support floor"
    )
    assert "_config_version" in path.read_text(encoding="utf-8"), (
        "boot maintenance must stamp the config it migrated"
    )


def test_genuinely_old_config_is_still_refused(home):
    """The guard the issue asks to keep: a config carrying a REAL version below the floor is
    still refused, with the actionable message. Only a *missing* stamp is exempt."""
    from hermes_cli.config_migrations import SUPPORT_FLOOR_VERSION
    from hermes_cli.post_update import step_migrate_config

    assert SUPPORT_FLOOR_VERSION > 1, "floor must leave room for a genuinely old stamp"
    _write(home, f"_config_version: {SUPPORT_FLOOR_VERSION - 1}\nmodel:\n  default: x\n")

    result = step_migrate_config()

    assert result == {"ok": True, "skipped": "below-support-floor"}


def test_current_config_is_still_a_no_op(home):
    """The 99% case must keep short-circuiting before any migration work."""
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.post_update import step_migrate_config

    _write(home, f"_config_version: {DEFAULT_CONFIG['_config_version']}\nmodel:\n  default: x\n")

    result = step_migrate_config()

    assert result == {"ok": True, "skipped": "up-to-date"}