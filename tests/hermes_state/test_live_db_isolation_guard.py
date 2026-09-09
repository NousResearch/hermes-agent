"""Behavioral tests for the live-DB test-isolation guard.

Forensic background (Aug 2026): pytest fixture rows (chat-1 / wx-chat
sessions, gateway_routing scopes under /tmp/pytest-of-*) were found in the
developer's REAL ~/.hermes/state.db, and a pytest-spawned process flipped
the journal mode under the WAL-mode gateway writer, destroying committed
transcripts. The guard under test makes any pytest-context ``SessionDB``
construction that resolves to a production state.db fail hard instead of
falling through.

These tests are behavioral: they construct real ``SessionDB`` objects (or
drive the real guard function) and assert outcomes — no source reading.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_state
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from hermes_state import SessionDB

# Must match the root the guard itself computes.  Hardcoding ``~/.hermes``
# silently disarmed every assertion below on Windows, where the real root is
# ``%LOCALAPPDATA%\hermes``: the paths under test were then *correctly*
# classified as non-production, so the guard never raised and the whole
# TestProductionPathRefused class failed for the wrong reason (#82770).
REAL_ROOT = hermes_state._real_platform_state_root()
if REAL_ROOT is None:  # pragma: no cover - no resolvable home on this platform
    pytest.skip(
        "no real platform state root to assert against", allow_module_level=True
    )


class TestProductionPathRefused:
    def test_explicit_production_db_path_raises(self):
        """SessionDB pointed at the real ~/.hermes/state.db must fail hard."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            SessionDB(db_path=REAL_ROOT / "state.db")

    def test_production_profile_db_path_raises(self):
        """Profile homes under the real root are production too."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            SessionDB(db_path=REAL_ROOT / "profiles" / "work" / "state.db")

    def test_read_only_open_of_production_db_raises(self):
        """Read-only opens are refused too — tests must not READ live data."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            SessionDB(db_path=REAL_ROOT / "state.db", read_only=True)

    def test_unnormalized_production_path_raises(self):
        """Symlink-free but unnormalized spellings still resolve and refuse."""
        sneaky = REAL_ROOT.parent / "subdir" / ".." / REAL_ROOT.name / "state.db"
        with pytest.raises(RuntimeError, match="live-system guard"):
            SessionDB(db_path=sneaky)

    def test_default_resolution_to_production_raises(self, monkeypatch):
        """The argless-construction path is guarded, not just explicit paths.

        Simulates the escape vector: HERMES_HOME leaked/reset to the real
        home (subprocess child, stale worktree, gateway-launched shell) so
        ``_default_db_path()`` resolves the production DB.
        """
        monkeypatch.setenv("HERMES_HOME", str(REAL_ROOT))
        # Neutralize the conftest's DEFAULT_DB_PATH re-pin so the default
        # resolver follows the (production-pointing) env, as it would in a
        # process that never imported the hermetic conftest.
        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
        )
        with pytest.raises(RuntimeError, match="live-system guard"):
            SessionDB()


class TestHermeticPathsAllowed:
    def test_tmp_db_path_works(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("iso-guard-session", "cli")
            assert db.get_session("iso-guard-session") is not None
        finally:
            db.close()

    def test_tmp_hermes_home_default_resolution_works(self, tmp_path, monkeypatch):
        """Argless SessionDB() under a hermetic HERMES_HOME must succeed."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermetic-home"))
        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
        )
        db = SessionDB()
        try:
            assert str(tmp_path) in str(db.db_path)
        finally:
            db.close()


class TestBypassMarker:
    @pytest.mark.live_system_guard_bypass
    def test_bypass_marker_disables_state_db_guard(self):
        """The established escape-hatch marker must let production paths pass.

        Drives the guard function directly (never actually opens the live
        DB) — with the bypass marker active it must not raise.
        """
        hermes_state._ensure_test_isolation(REAL_ROOT / "state.db")


class TestSessionStoreLoudFailure:
    def test_guard_error_is_not_swallowed_into_jsonl_fallback(
        self, tmp_path, monkeypatch
    ):
        """SessionStore must re-raise the guard error, not degrade to JSONL.

        The historical failure mode: SessionDB() blew up (or silently
        opened the live DB) inside SessionStore.__init__'s blanket
        ``except Exception`` and the gateway carried on. A guard trip must
        be loud.
        """

        def _boom(*args, **kwargs):
            raise RuntimeError(
                "live-system guard: test attempted to open production state.db"
            )

        monkeypatch.setattr(hermes_state, "SessionDB", _boom)
        with pytest.raises(RuntimeError, match="live-system guard"):
            SessionStore(sessions_dir=tmp_path, config=GatewayConfig())

    def test_ordinary_db_failure_still_degrades_to_jsonl(
        self, tmp_path, monkeypatch
    ):
        """Non-guard SQLite failures keep the existing graceful fallback."""

        def _boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(hermes_state, "SessionDB", _boom)
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None


class TestSubprocessChildCovered:
    def test_child_without_hermes_home_is_refused(self, tmp_path):
        """A subprocess child of a test (no HERMES_HOME) must be blocked.

        This is the real leak vector: tests spawning ``python -m ...``
        children that never import the hermetic conftest. The guard is
        env-activated (PYTEST_CURRENT_TEST / PYTEST_VERSION are inherited),
        so the child's argless SessionDB() must fail hard instead of
        opening the developer's real state.db.
        """
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("HERMES_HOME", "PYTEST_PLUGINS", "PYTHONPATH")
        }
        env["PYTEST_CURRENT_TEST"] = "tests/fake.py::test_child (call)"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        code = (
            "from hermes_state import SessionDB\n"
            "SessionDB()\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode != 0
        assert "live-system guard" in proc.stderr

    def test_child_with_tmp_hermes_home_succeeds(self, tmp_path):
        """Same child, hermetic HERMES_HOME: must work — no false positive."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTEST_PLUGINS", "PYTHONPATH")
        }
        env["PYTEST_CURRENT_TEST"] = "tests/fake.py::test_child (call)"
        env["HERMES_HOME"] = str(tmp_path / "child-home")
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        code = (
            "from hermes_state import SessionDB\n"
            "db = SessionDB()\n"
            "db.close()\n"
            "print('OK', db.db_path)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
