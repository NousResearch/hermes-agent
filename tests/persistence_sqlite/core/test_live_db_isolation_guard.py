"""Behavioral tests for the live-DB test-isolation guard.

Forensic background (Aug 2026): pytest fixture rows (chat-1 / wx-chat
sessions, gateway_routing scopes under /tmp/pytest-of-*) were found in the
developer's REAL ~/.hermes/state.db, and a pytest-spawned process flipped
the journal mode under the WAL-mode gateway writer, destroying committed
transcripts. The guard under test makes any pytest-context ``StoreHolder``
construction that resolves to a production state.db fail hard instead of
falling through.

These tests are behavioral: they construct real ``StoreHolder`` objects (or
drive the real guard function) and assert outcomes — no source reading.
"""

import pytest

from persistence.sqlite._state import guard as state_guard
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from persistence.sqlite import _holder
from persistence.sqlite._holder import StoreHolder
import persistence_assembly

# These probe the live-DB guard against the real root on purpose.
pytestmark = pytest.mark.allow_real_home_io

# Must match the root the guard itself computes.  Hardcoding ``~/.hermes``
# silently disarmed every assertion below on Windows, where the real root is
# ``%LOCALAPPDATA%\hermes``: the paths under test were then *correctly*
# classified as non-production, so the guard never raised and the whole
# TestProductionPathRefused class failed for the wrong reason (#82770).
REAL_ROOT = state_guard._real_platform_state_root()
if REAL_ROOT is None:  # pragma: no cover - no resolvable home on this platform
    pytest.skip(
        "no real platform state root to assert against", allow_module_level=True
    )


class TestProductionPathRefused:
    def test_explicit_production_db_path_raises(self):
        """StoreHolder pointed at the real ~/.hermes/state.db must fail hard."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            StoreHolder(db_path=REAL_ROOT / "state.db")

    def test_production_profile_db_path_raises(self):
        """Profile homes under the real root are production too."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            StoreHolder(db_path=REAL_ROOT / "profiles" / "work" / "state.db")

    def test_read_only_open_of_production_db_raises(self):
        """Read-only opens are refused too — tests must not READ live data."""
        with pytest.raises(RuntimeError, match="live-system guard"):
            StoreHolder(db_path=REAL_ROOT / "state.db", read_only=True)

    def test_unnormalized_production_path_raises(self):
        """Symlink-free but unnormalized spellings still resolve and refuse."""
        sneaky = REAL_ROOT.parent / "subdir" / ".." / REAL_ROOT.name / "state.db"
        with pytest.raises(RuntimeError, match="live-system guard"):
            StoreHolder(db_path=sneaky)


class TestBypassMarker:
    @pytest.mark.live_system_guard_bypass
    def test_bypass_marker_disables_state_db_guard(self):
        """The established escape-hatch marker must let production paths pass.

        Drives the guard function directly (never actually opens the live
        DB) — with the bypass marker active it must not raise.
        """
        _holder._ensure_test_isolation(REAL_ROOT / "state.db")


class TestSessionStoreLoudFailure:
    def test_guard_error_is_not_swallowed_into_jsonl_fallback(
        self, tmp_path, monkeypatch
    ):
        """SessionStore must re-raise the guard error, not degrade to JSONL.

        The historical failure mode: the store open blew up (or silently
        opened the live DB) inside SessionStore.__init__'s blanket
        ``except Exception`` and the gateway carried on. A guard trip must
        be loud.
        """

        def _boom(*args, **kwargs):
            raise RuntimeError(
                "live-system guard: test attempted to open production state.db"
            )

        monkeypatch.setattr(persistence_assembly, "open_persistence", _boom)
        with pytest.raises(RuntimeError, match="live-system guard"):
            SessionStore(sessions_dir=tmp_path, config=GatewayConfig())

    def test_ordinary_db_failure_still_degrades_to_jsonl(
        self, tmp_path, monkeypatch
    ):
        """Non-guard SQLite failures keep the existing graceful fallback."""

        def _boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(persistence_assembly, "open_persistence", _boom)
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store.persistence is None
