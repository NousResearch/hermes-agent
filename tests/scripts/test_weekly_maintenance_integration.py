"""End-to-end integration tests for ``scripts/weekly_maintenance.py``.

These tests run the full driver under a synthetic ``HERMES_HOME`` and
assert that the script

  * resolves every path against the override (the headline #24035 fix),
  * leaves the *real* user's ``~/.hermes`` completely untouched even
    when there's a state.db sitting there, and
  * actually shrinks a bloated state.db / removes old snapshots /
    rotates large logs.

The unit-level helper contracts live in
``tests/scripts/test_weekly_maintenance.py``; this file is the
regression anchor that catches anyone reintroducing
``Path.home() / ".hermes"`` to the script — it would silently start
operating on the developer's real Hermes home and the assertions below
would explode.
"""

import importlib.util
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "weekly_maintenance.py"


@pytest.fixture(scope="module")
def weekly_module() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "weekly_maintenance_integration", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def synthetic_profile(tmp_path):
    """Create a realistic on-disk profile structure.

    Layout (everything under tmp_path so the real ~/.hermes stays
    untouched):

      tmp_path/
        state.db           — bloated SQLite file
        state-snapshots/
          2024-01-01-old/  — should be pruned (100 days stale)
          2026-05-08-fresh/— should be kept
        logs/
          gateway.log      — 11 MB, should be rotated
          agent.log        — 200 B, should be left alone
          old.log.123.gz   — 100 days old, should be pruned
    """
    home = tmp_path / "synthetic-profile"
    home.mkdir()

    db = home / "state.db"
    with sqlite3.connect(str(db)) as c:
        c.execute("CREATE TABLE junk (k INTEGER PRIMARY KEY, v TEXT)")
        c.executemany(
            "INSERT INTO junk(v) VALUES (?)",
            [("x" * 256,) for _ in range(2000)],
        )
        c.commit()
        c.execute("DELETE FROM junk")
        c.commit()
    bloated_size = db.stat().st_size

    snapshots = home / "state-snapshots"
    snapshots.mkdir()
    old_snap = snapshots / "2024-01-01-old"
    old_snap.mkdir()
    (old_snap / "manifest.json").write_text("{}")
    fresh_snap = snapshots / "2026-05-08-fresh"
    fresh_snap.mkdir()
    (fresh_snap / "manifest.json").write_text("{}")

    logs = home / "logs"
    logs.mkdir()
    big_log = logs / "gateway.log"
    big_log.write_bytes(b"0" * (11 * 1024 * 1024))
    small_log = logs / "agent.log"
    small_log.write_text("a small log")
    old_archive = logs / "old.log.123.gz"
    old_archive.write_bytes(b"\x1f\x8b\x08\x00")

    now = time.time()
    os.utime(old_snap, (now - 100 * 86400, now - 100 * 86400))
    os.utime(old_archive, (now - 100 * 86400, now - 100 * 86400))

    return {
        "home": home,
        "bloated_size": bloated_size,
        "old_snap": old_snap,
        "fresh_snap": fresh_snap,
        "big_log": big_log,
        "small_log": small_log,
        "old_archive": old_archive,
        "now": now,
    }


# ---------------------------------------------------------------------------
# End-to-end run() — the full driver under a synthetic profile
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_run_shrinks_db_prunes_snapshots_and_rotates_logs(
        self, weekly_module, synthetic_profile
    ):
        with patch.object(
            weekly_module, "get_hermes_home", return_value=synthetic_profile["home"]
        ):
            results = weekly_module.run()

        # No phase errored.
        assert all(r.error is None for r in results), [
            (r.name, r.error) for r in results if r.error
        ]

        # state.db shrank.
        new_size = (synthetic_profile["home"] / "state.db").stat().st_size
        assert new_size < synthetic_profile["bloated_size"], (
            "VACUUM did not shrink the bloated state.db end-to-end"
        )

        # Snapshots: old gone, fresh kept.
        assert not synthetic_profile["old_snap"].exists()
        assert synthetic_profile["fresh_snap"].exists()

        # Logs: big rotated, small kept, old archive pruned.
        logs = synthetic_profile["home"] / "logs"
        assert not synthetic_profile["big_log"].exists()
        assert list(logs.glob("gateway.log.*.gz"))
        assert synthetic_profile["small_log"].exists()
        assert not synthetic_profile["old_archive"].exists()

    def test_dry_run_changes_nothing(self, weekly_module, synthetic_profile):
        before = _snapshot_dir_state(synthetic_profile["home"])

        with patch.object(
            weekly_module, "get_hermes_home", return_value=synthetic_profile["home"]
        ):
            results = weekly_module.run(dry_run=True)

        assert all(r.error is None for r in results)
        after = _snapshot_dir_state(synthetic_profile["home"])
        assert before == after, (
            "dry-run modified the profile — that's a safety regression"
        )

    def test_only_phase_isolation(self, weekly_module, synthetic_profile):
        """``--only logs`` must not touch state.db or snapshots."""
        db = synthetic_profile["home"] / "state.db"
        size_before = db.stat().st_size

        with patch.object(
            weekly_module, "get_hermes_home", return_value=synthetic_profile["home"]
        ):
            results = weekly_module.run(only="logs")

        assert all(r.error is None for r in results)
        assert [r.name for r in results] == ["rotate_logs"]
        # state.db untouched.
        assert db.stat().st_size == size_before
        # Snapshots untouched.
        assert synthetic_profile["old_snap"].exists()


# ---------------------------------------------------------------------------
# Subprocess regression — the script must honour HERMES_HOME end-to-end
# ---------------------------------------------------------------------------


class TestSubprocessProfileResolution:
    """Spawn the script as the user would and inspect stdout.

    These tests don't import the module — they spawn it as a fresh
    Python process so the real production code path runs (sys.path
    bootstrap, hermes_constants import, argparse, the lot). They only
    check the dry-run path so no state on disk is mutated.
    """

    def test_hermes_home_env_var_flows_into_resolved_paths(self, tmp_path):
        target = tmp_path / "work-profile"
        target.mkdir()

        env = dict(os.environ)
        env["HERMES_HOME"] = str(target)
        env.pop("HERMES_LOG_LEVEL", None)

        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--dry-run"],
            capture_output=True, text=True, env=env, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr

        # Every printed path must root at the override.
        assert str(target) in proc.stdout, proc.stdout
        # Critically, NO path may root at the developer's real ~/.hermes
        # while HERMES_HOME points at the synthetic profile.
        real_home = str(Path.home() / ".hermes")
        assert f"{real_home}/state.db" not in proc.stdout, (
            "#24035 regression: weekly_maintenance leaked the real "
            "~/.hermes/state.db into output while HERMES_HOME pointed "
            "elsewhere — the profile is being ignored."
        )

    def test_subprocess_resolves_home_from_get_hermes_home(self, tmp_path):
        """Run the script in a subprocess that imports hermes_constants
        the same way the script does, and verify the path it prints
        matches what hermes_constants.get_hermes_home() returns inside
        that same subprocess.

        This avoids tying the test to the in-process conftest, which
        patches Path.home / HERMES_HOME for hermeticity in unit tests
        but cannot influence the spawned process.
        """
        env = dict(os.environ)
        env.pop("HERMES_HOME", None)
        env.pop("HERMES_LOG_LEVEL", None)

        # Step 1: ask a fresh subprocess what get_hermes_home() returns.
        ask = subprocess.run(
            [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
                "from hermes_constants import get_hermes_home; "
                "print(str(get_hermes_home()))",
            ],
            capture_output=True, text=True, env=env, timeout=30,
        )
        assert ask.returncode == 0, ask.stderr
        expected = ask.stdout.strip()
        assert expected, "get_hermes_home() returned empty path"

        # Step 2: run the script in the same env and assert the resolved
        # path matches.
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--dry-run"],
            capture_output=True, text=True, env=env, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        assert expected in proc.stdout, (
            f"Resolved home {expected!r} not in script output — "
            "the script is using a different resolver than "
            "hermes_constants.get_hermes_home()."
        )


# ---------------------------------------------------------------------------
# Bug-shape regression — explicit #24035 anchor
# ---------------------------------------------------------------------------


class TestBug24035Anchor:
    """If any of these fail, the #24035 regression is back.

    Don't 'fix' the test by deleting the assertion — fix the script
    that started using ``os.path.expanduser('~')`` /
    ``Path.home() / '.hermes'`` again.
    """

    def test_script_source_does_not_derive_data_paths_from_path_home(self):
        """The script must never derive a *data* path from
        ``Path.home() / ".hermes"`` or ``os.path.expanduser("~/.hermes")``.

        The exception (which this test deliberately tolerates) is the
        sys.path bootstrap in :data:`_CANDIDATE_REPO_ROOTS` that probes
        ``~/.hermes/hermes-agent`` when looking for the source checkout
        — that's a *code* lookup, not a data path. We pin that exact
        usage so any *new* occurrence of the same pattern fails the test.
        """
        text = SCRIPT_PATH.read_text()

        # The bug pattern: assigning STATE_DB / LCM_DB / SNAPSHOTS_DIR /
        # LOGS_DIR from os.path.expanduser('~') or Path.home(). If any
        # appears within 200 chars of one of those names, fail.
        bug_markers = ('STATE_DB', 'LCM_DB', 'SNAPSHOTS_DIR', 'LOGS_DIR')
        bad_root_patterns = (
            'expanduser("~")', "expanduser('~')",
            'expanduser("~/.hermes', "expanduser('~/.hermes",
            'Path.home() / ".hermes" / "state',
            'Path.home() / \'.hermes\' / \'state',
        )
        for marker in bug_markers:
            idx = 0
            while True:
                pos = text.find(marker, idx)
                if pos < 0:
                    break
                window = text[max(0, pos - 200): pos + 200]
                for pat in bad_root_patterns:
                    assert pat not in window, (
                        f"#24035 regression: the script derives {marker} "
                        f"from {pat!r} — that bypasses get_hermes_home() "
                        "and silently no-ops users on non-default profiles."
                    )
                idx = pos + len(marker)

        # The legitimate sys.path bootstrap (~/.hermes/hermes-agent for
        # users who copied the script into ~/.hermes/scripts/) is the
        # only tolerated *code* use of `Path.home() / ".hermes"`. We
        # tolerate one extra match in the explanatory docstring above
        # _CANDIDATE_REPO_ROOTS that documents why the bootstrap exists.
        # Anything beyond that is suspicious.
        assert text.count('Path.home() / ".hermes"') <= 2, (
            "More than two occurrences of `Path.home() / \".hermes\"` — "
            "the only tolerated uses are the sys.path bootstrap in "
            "_CANDIDATE_REPO_ROOTS and the docstring that explains it."
        )

    def test_script_source_imports_get_hermes_home(self):
        text = SCRIPT_PATH.read_text()
        assert "from hermes_constants import get_hermes_home" in text, (
            "#24035 regression: the script no longer imports "
            "hermes_constants.get_hermes_home()."
        )

    def test_script_source_documents_24035(self):
        """Anchor the issue number in the source so future maintainers
        understand why this care exists.
        """
        text = SCRIPT_PATH.read_text()
        assert "#24035" in text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _snapshot_dir_state(root: Path) -> dict:
    """Capture (relative path → size) for every file under root."""
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.stat().st_size
    return out
