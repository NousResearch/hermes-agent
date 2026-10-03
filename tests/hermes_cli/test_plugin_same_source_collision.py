"""Regression tests: same-source plugin manifest name collisions.

Regression for #121078. Two plugin directories from the SAME source (e.g. two
user plugins ``foo/`` and a hand-made backup ``foo.bak-<ts>/`` that both declare
``name: foo`` in their manifest) collide on the registry key. Discovery sorts
directories, so the backup sorts last and silently wins, and the gateway runs
the old plugin with zero log lines. Cross-source override (user > bundled,
project > user) is the documented shape and must keep working exactly as it
did: later source wins, logged at INFO.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pytest

from hermes_cli.plugins_discovery import resolve_manifest_winners
from hermes_cli.plugins_manifest import PluginManifest


def _manifest(name: str, source: str, dirname: Optional[str] = None) -> PluginManifest:
    """A flat (no category prefix) manifest: registry key == name, dir under
    ``~/.hermes/plugins/<dirname>`` (dirname defaults to name)."""
    return PluginManifest(
        name=name,
        source=source,
        path=str(Path("~/.hermes/plugins") / (dirname or name)),
    )


class TestSameSourceCollision:
    def test_backup_dir_loses_to_its_namesake(self, caplog):
        """``foo.bak-2/`` (name: foo) vs ``foo/`` (name: foo), both user: the
        well-named directory wins and a WARNING names both paths."""
        live = _manifest("foo", "user", "foo")
        backup = _manifest("foo", "user", "foo.bak-2")  # sorts after "foo"

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            winners = resolve_manifest_winners([live, backup])

        assert winners["foo"] is live
        joined = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert "foo.bak-2" in joined and "foo" in joined

    def test_both_misnamed_later_still_wins_but_warns(self, caplog):
        """No well-named side: keep the existing last-in-order semantics but
        stop being silent — a WARNING must name both paths."""
        a = _manifest("foo", "user", "aaa-renamed")
        b = _manifest("foo", "user", "zzz-renamed")

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            winners = resolve_manifest_winners([a, b])

        assert winners["foo"] is b
        joined = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert "aaa-renamed" in joined and "zzz-renamed" in joined


class TestCrossSourceOverrideUnchanged:
    def test_user_override_of_bundled_still_wins(self, caplog):
        """Documented shape: a user plugin in the SAME directory name as the
        bundled one overrides it, at INFO level — not a warning."""
        bundled = _manifest("kanban", "bundled", "kanban")
        user = _manifest("kanban", "user", "kanban")

        with caplog.at_level(logging.INFO, logger="hermes_cli.plugins"):
            winners = resolve_manifest_winners([bundled, user])

        assert winners["kanban"] is user
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings

    def test_project_beats_user_across_sources(self):
        """Cross-source precedence (project > user) is untouched."""
        user = _manifest("notify", "user", "notify")
        project = _manifest("notify", "project", "notify")

        winners = resolve_manifest_winners([user, project])

        assert winners["notify"] is project
