"""Tests for per-board dispatch/auto-decompose/review-dispatch overrides.

Covers the narrowing-only override contract added alongside the existing
global ``kanban.dispatch_in_gateway`` / ``kanban.auto_decompose`` /
``kanban.review_dispatch`` config switches:

* ``board.json`` schema: new fields default to ``True`` (absent key on an
  old board.json means "inherit global", not "off").
* ``write_board_metadata()`` / ``read_board_metadata()`` round trip for all
  three new fields, independently of the pre-existing fields.
* The gateway dispatcher's per-board guards (``tick_once_for_board``,
  ``auto_decompose_tick``) skip a board whose own flag is off, and do not
  skip a board whose flag is on (or absent).
* ``review_dispatch_enabled(board=...)`` applies the per-board override only
  when the global switch is already on; a disabled global switch always
  wins regardless of the board's own flag.
* CLI surface: ``hermes kanban boards set-dispatch/set-auto-decompose/
  set-review-dispatch <slug> on|off`` round trip through ``boards show``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


# ---------------------------------------------------------------------------
# Schema defaults and round trip
# ---------------------------------------------------------------------------

class TestBoardDispatchSchema:
    def test_absent_fields_default_to_enabled(self, fresh_home):
        """An old board.json (or one that predates this feature) must read
        as "inherit global" — never silently "off"."""
        kb.create_board("legacy-board")
        meta = kb.read_board_metadata("legacy-board")
        assert meta["dispatch_enabled"] is True
        assert meta["auto_decompose_enabled"] is True
        assert meta["review_dispatch_enabled"] is True

    def test_write_then_read_round_trips_dispatch_enabled(self, fresh_home):
        kb.create_board("proj")
        kb.write_board_metadata("proj", dispatch_enabled=False)
        assert kb.read_board_metadata("proj")["dispatch_enabled"] is False
        kb.write_board_metadata("proj", dispatch_enabled=True)
        assert kb.read_board_metadata("proj")["dispatch_enabled"] is True

    def test_write_then_read_round_trips_auto_decompose_enabled(self, fresh_home):
        kb.create_board("proj")
        kb.write_board_metadata("proj", auto_decompose_enabled=False)
        assert kb.read_board_metadata("proj")["auto_decompose_enabled"] is False

    def test_write_then_read_round_trips_review_dispatch_enabled(self, fresh_home):
        kb.create_board("proj")
        kb.write_board_metadata("proj", review_dispatch_enabled=False)
        assert kb.read_board_metadata("proj")["review_dispatch_enabled"] is False

    def test_none_leaves_field_unchanged(self, fresh_home):
        """write_board_metadata(..., dispatch_enabled=None) must not reset
        an already-disabled flag back to the default."""
        kb.create_board("proj")
        kb.write_board_metadata("proj", dispatch_enabled=False)
        kb.write_board_metadata("proj", name="Renamed")
        assert kb.read_board_metadata("proj")["dispatch_enabled"] is False

    def test_unrelated_board_is_unaffected(self, fresh_home):
        """Disabling one board's flag must not leak onto a sibling board."""
        kb.create_board("proj-a")
        kb.create_board("proj-b")
        kb.write_board_metadata("proj-a", dispatch_enabled=False)
        assert kb.read_board_metadata("proj-a")["dispatch_enabled"] is False
        assert kb.read_board_metadata("proj-b")["dispatch_enabled"] is True


# ---------------------------------------------------------------------------
# review_dispatch_enabled(board=...) precedence
# ---------------------------------------------------------------------------

class TestReviewDispatchPrecedence:
    def test_board_none_falls_back_to_global_only(self, fresh_home, monkeypatch):
        """Calling with no board argument must behave exactly as before
        this change (pure global read, no board.json touched)."""
        assert kbd.review_dispatch_enabled() is True
        assert kbd.review_dispatch_enabled(board=None) is True

    def test_global_on_board_off_is_disabled(self, fresh_home):
        kb.create_board("proj")
        kb.write_board_metadata("proj", review_dispatch_enabled=False)
        assert kbd.review_dispatch_enabled(board="proj") is False

    def test_global_on_board_on_is_enabled(self, fresh_home):
        kb.create_board("proj")
        assert kbd.review_dispatch_enabled(board="proj") is True

    def test_global_off_wins_even_if_board_flag_on(self, fresh_home, monkeypatch):
        """A disabled global switch always wins — a board cannot force
        review dispatch on when the global switch is off."""
        kb.create_board("proj")
        kb.write_board_metadata("proj", review_dispatch_enabled=True)

        def _fake_load_config():
            return {"kanban": {"review_dispatch": False}}

        monkeypatch.setattr("hermes_cli.config.load_config", _fake_load_config)
        assert kbd.review_dispatch_enabled(board="proj") is False

    def test_missing_board_metadata_fails_open_to_global(self, fresh_home):
        """A board whose metadata can't be read (never created, deleted
        mid-flight) must not crash the dispatcher tick — fail open to the
        already-resolved global value."""
        assert kbd.review_dispatch_enabled(board="never-created-board") is True


# ---------------------------------------------------------------------------
# Gateway dispatcher guards
# ---------------------------------------------------------------------------

class TestDispatcherBoardGuards:
    def test_tick_once_for_board_skips_disabled_board(self, fresh_home, monkeypatch):
        from hermes_cli import kanban_db as kbmod
        from gateway import kanban_watchers_dispatcher as wd

        kb.create_board("proj")
        kb.write_board_metadata("proj", dispatch_enabled=False)

        settings = wd._DispatcherSettings(
            interval=1.0, max_spawn=None, max_in_progress=None, failure_limit=2,
            stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
            max_in_progress_per_profile=None,
        )
        dispatcher = wd._KanbanDispatcher(kb=kbmod, settings=settings)

        called = {"n": 0}

        def _boom(*a, **k):
            called["n"] += 1
            raise AssertionError("dispatch_once must not run for a disabled board")

        monkeypatch.setattr(wd, "_kbd", lambda: type("M", (), {"dispatch_once": staticmethod(_boom)}))
        result = dispatcher.tick_once_for_board("proj")
        assert result is None
        assert called["n"] == 0

    def test_tick_once_for_board_runs_when_enabled(self, fresh_home, monkeypatch):
        from hermes_cli import kanban_db as kbmod
        from gateway import kanban_watchers_dispatcher as wd

        kb.create_board("proj")  # dispatch_enabled defaults to True

        settings = wd._DispatcherSettings(
            interval=1.0, max_spawn=None, max_in_progress=None, failure_limit=2,
            stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
            max_in_progress_per_profile=None,
        )
        dispatcher = wd._KanbanDispatcher(kb=kbmod, settings=settings)

        called = {"n": 0}

        def _fake_dispatch_once(conn, board=None, **kwargs):
            called["n"] += 1
            return "ok"

        class _Conn:
            def close(self):
                pass

        monkeypatch.setattr(
            wd, "_kbc",
            lambda: type("M", (), {"connect": staticmethod(lambda board=None: _Conn())}),
        )
        monkeypatch.setattr(
            wd, "_kbd",
            lambda: type("M", (), {"dispatch_once": staticmethod(_fake_dispatch_once)}),
        )
        result = dispatcher.tick_once_for_board("proj")
        assert result == "ok"
        assert called["n"] == 1

    def test_auto_decompose_tick_skips_disabled_board(self, fresh_home, monkeypatch):
        from hermes_cli import kanban_db as kbmod
        from gateway import kanban_watchers_dispatcher as wd

        kb.create_board("proj")
        kb.write_board_metadata("proj", auto_decompose_enabled=False)

        settings = wd._DispatcherSettings(
            interval=1.0, max_spawn=None, max_in_progress=None, failure_limit=2,
            stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
            max_in_progress_per_profile=None,
        )
        dispatcher = wd._KanbanDispatcher(kb=kbmod, settings=settings)
        monkeypatch.setattr(dispatcher, "_board_slugs", lambda: ["proj"])

        called = {"n": 0}

        class _FakeDecomp:
            @staticmethod
            def list_triage_ids():
                called["n"] += 1
                return ["t_should_never_be_reached"]

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "hermes_cli.kanban_decompose", _FakeDecomp)

        result = dispatcher.auto_decompose_tick(auto_decompose_per_tick=5)
        assert result == 0
        assert called["n"] == 0, "list_triage_ids must not run for a disabled board"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def _cli(args, env_extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_WORKTREE)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban"] + args,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
        timeout=30,
    )


class TestDispatchToggleCLI:
    def test_set_dispatch_off_then_on_round_trips_via_show(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "proj"], env_extra=env).returncode == 0
        assert _cli(["boards", "switch", "proj"], env_extra=env).returncode == 0

        r = _cli(["boards", "set-dispatch", "proj", "off"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "disabled" in r.stdout

        r = _cli(["boards", "show"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "Dispatch:     off (board override)" in r.stdout

        r = _cli(["boards", "set-dispatch", "proj", "on"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "enabled" in r.stdout

        r = _cli(["boards", "show"], env_extra=env)
        assert "Dispatch:     on" in r.stdout

    def test_set_auto_decompose_off_reflected_in_show(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "proj"], env_extra=env).returncode == 0
        assert _cli(["boards", "switch", "proj"], env_extra=env).returncode == 0
        assert _cli(["boards", "set-auto-decompose", "proj", "off"], env_extra=env).returncode == 0
        r = _cli(["boards", "show"], env_extra=env)
        assert "Auto-decompose: off (board override)" in r.stdout

    def test_set_review_dispatch_off_reflected_in_show(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "proj"], env_extra=env).returncode == 0
        assert _cli(["boards", "switch", "proj"], env_extra=env).returncode == 0
        assert _cli(["boards", "set-review-dispatch", "proj", "off"], env_extra=env).returncode == 0
        r = _cli(["boards", "show"], env_extra=env)
        assert "Review-dispatch: off (board override)" in r.stdout

    def test_default_new_board_shows_all_on(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "fresh"], env_extra=env).returncode == 0
        assert _cli(["boards", "switch", "fresh"], env_extra=env).returncode == 0
        r = _cli(["boards", "show"], env_extra=env)
        assert "Dispatch:     on" in r.stdout
        assert "Auto-decompose: on" in r.stdout
        assert "Review-dispatch: on" in r.stdout

    def test_toggling_one_board_does_not_affect_another(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "proj-a"], env_extra=env).returncode == 0
        assert _cli(["boards", "create", "proj-b"], env_extra=env).returncode == 0
        assert _cli(["boards", "set-dispatch", "proj-a", "off"], env_extra=env).returncode == 0

        assert _cli(["boards", "switch", "proj-a"], env_extra=env).returncode == 0
        r_a = _cli(["boards", "show"], env_extra=env)
        assert _cli(["boards", "switch", "proj-b"], env_extra=env).returncode == 0
        r_b = _cli(["boards", "show"], env_extra=env)
        assert "Dispatch:     off" in r_a.stdout
        assert "Dispatch:     on" in r_b.stdout

    def test_set_dispatch_rejects_unknown_board(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        r = _cli(["boards", "set-dispatch", "does-not-exist", "off"], env_extra=env)
        assert r.returncode != 0

    def test_set_dispatch_rejects_invalid_state(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "proj"], env_extra=env).returncode == 0
        r = _cli(["boards", "set-dispatch", "proj", "maybe"], env_extra=env)
        assert r.returncode != 0
