"""Real integration tests for the PTY session-resume backend round-trip.

These tests drive _resolve_chat_argv (the REAL function, not mocked) against a
real temp SQLite SessionDB, and assert that HERMES_TUI_RESUME is set to the
correct session id in the returned env dict.

The only things stubbed out are:
  - hermes_cli.main._make_tui_argv — to avoid requiring a built Node TUI
  - hermes_cli.web_server._open_session_db_for_profile — injected per-test
    so we control which DB file is opened without touching HERMES_HOME

Everything else — _resolve_chat_argv, _session_latest_descendant,
_read_active_session_file — is the production code under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="PTY bridge is POSIX-only"
)

# ---------------------------------------------------------------------------
# Tiny fake argv that requires no Node build
# ---------------------------------------------------------------------------
_FAKE_ARGV = ["node", "dist/entry.js"]
_FAKE_CWD = "/tmp"


def _make_fake_tui_argv(_project_root, tui_dev=False):
    """Why: stub out the TUI build so tests don't need a real Node install.
    What: returns a minimal argv tuple pointing at a fake entry point.
    Test: confirmed by checking argv in env assertions.
    """
    return _FAKE_ARGV, Path(_FAKE_CWD)


# ---------------------------------------------------------------------------
# Shared fixture: temp SessionDB with sessions seeded per-test
# ---------------------------------------------------------------------------

@pytest.fixture()
def session_db(tmp_path):
    """Why: provides an isolated real SQLite DB so tests don't share state.
    What: creates a SessionDB at a tmp path, yields it, closes on teardown.
    Test: verify db_path exists and create_session works.
    """
    from hermes_state import SessionDB
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Helper: patch _resolve_chat_argv's two dependencies
# ---------------------------------------------------------------------------

def _patch_resolve_env(monkeypatch, session_db):
    """Why: isolates _resolve_chat_argv from the TUI build and from the
    default HERMES_HOME SessionDB so tests can provide their own sessions.
    What: monkeypatches _make_tui_argv and _open_session_db_for_profile.
    Test: each calling test verifies side-effects in the returned env.
    """
    import hermes_cli.main as main_mod
    import hermes_cli.web_server as ws

    monkeypatch.setattr(main_mod, "_make_tui_argv", _make_fake_tui_argv)

    # _open_session_db_for_profile(None) → our test db
    monkeypatch.setattr(ws, "_open_session_db_for_profile", lambda _profile: session_db)


# ============================================================
# Priority 1a — direct resume: HERMES_TUI_RESUME is set
# ============================================================

def test_resolve_chat_argv_sets_resume_env_for_known_session(
    monkeypatch, session_db, _isolate_hermes_home
):
    """Why: verifies that _resolve_chat_argv propagates a resume id via env.
    What: seeds a real session row; calls the real function; asserts env var.
    Test: HERMES_TUI_RESUME must equal the seeded session id.
    """
    import hermes_cli.web_server as ws

    _patch_resolve_env(monkeypatch, session_db)

    sid = "20260408_120000_aabbcc"
    session_db.create_session(sid, source="tui")

    _argv, _cwd, env = ws._resolve_chat_argv(resume=sid)

    assert env is not None, "env must not be None"
    assert "HERMES_TUI_RESUME" in env, (
        f"HERMES_TUI_RESUME missing from env; got keys: {sorted(env.keys())}"
    )
    assert env["HERMES_TUI_RESUME"] == sid


# ============================================================
# Priority 1b — compression chain resolves to latest descendant
# ============================================================

def test_resolve_chat_argv_resolves_compression_chain_to_latest(
    monkeypatch, session_db, _isolate_hermes_home
):
    """Why: /model creates child sessions; dashboard must resume the newest leaf.
    What: seeds parent + child; calls _resolve_chat_argv(resume=parent_id);
          asserts env points at the child (latest descendant).
    Test: HERMES_TUI_RESUME must equal child_id, not parent_id.
    """
    import hermes_cli.web_server as ws

    _patch_resolve_env(monkeypatch, session_db)

    parent_id = "20260408_100000_parent"
    child_id  = "20260408_110000_child"
    session_db.create_session(parent_id, source="tui")
    session_db.create_session(child_id, source="tui", parent_session_id=parent_id)

    _argv, _cwd, env = ws._resolve_chat_argv(resume=parent_id)

    assert env is not None
    assert "HERMES_TUI_RESUME" in env, (
        "HERMES_TUI_RESUME missing; _session_latest_descendant may not have run"
    )
    assert env["HERMES_TUI_RESUME"] == child_id, (
        f"Expected latest descendant {child_id!r}, got {env['HERMES_TUI_RESUME']!r}"
    )


def test_resolve_chat_argv_resolves_multi_level_chain_to_deepest(
    monkeypatch, session_db, _isolate_hermes_home
):
    """Why: chains can be more than one level deep; must reach the leaf.
    What: seeds parent → child → grandchild; asserts grandchild is resolved.
    Test: HERMES_TUI_RESUME == grandchild_id.
    """
    import hermes_cli.web_server as ws

    _patch_resolve_env(monkeypatch, session_db)

    import time
    parent_id = "20260408_100000_p"
    child_id  = "20260408_110000_c"
    grand_id  = "20260408_120000_g"
    session_db.create_session(parent_id, source="tui")
    time.sleep(0.01)
    session_db.create_session(child_id, source="tui", parent_session_id=parent_id)
    time.sleep(0.01)
    session_db.create_session(grand_id, source="tui", parent_session_id=child_id)

    _argv, _cwd, env = ws._resolve_chat_argv(resume=parent_id)

    assert env["HERMES_TUI_RESUME"] == grand_id, (
        f"Expected deepest {grand_id!r}, got {env['HERMES_TUI_RESUME']!r}"
    )


# ============================================================
# Priority 1c — unknown/missing resume id degrades gracefully
# ============================================================

def test_resolve_chat_argv_missing_session_id_does_not_crash(
    monkeypatch, session_db, _isolate_hermes_home
):
    """Why: an unknown resume id must not crash — the PTY child will handle
    the stale id gracefully on its own.
    What: passes an id not in the DB; asserts no exception and env is returned.
    Test: no exception raised; env dict is returned (HERMES_TUI_RESUME may be
          set to the raw id since _session_latest_descendant degrades by
          returning None and the code falls back to the original id).
    """
    import hermes_cli.web_server as ws

    _patch_resolve_env(monkeypatch, session_db)

    stale_id = "totally-nonexistent-id-xyz"
    # Must not raise, even with an unknown session id
    _argv, _cwd, env = ws._resolve_chat_argv(resume=stale_id)

    assert env is not None, "env must be returned even for unknown resume id"
    # The code sets HERMES_TUI_RESUME to the original id when the DB lookup
    # finds nothing — the TUI child handles the stale id gracefully.
    assert "HERMES_TUI_RESUME" in env, (
        "HERMES_TUI_RESUME must be set so the TUI can attempt to resume or skip gracefully"
    )
    assert env["HERMES_TUI_RESUME"] == stale_id


# ============================================================
# Priority 1d — no resume arg → HERMES_TUI_RESUME absent
# ============================================================

def test_resolve_chat_argv_no_resume_omits_resume_env(
    monkeypatch, session_db, _isolate_hermes_home
):
    """Why: fresh chat must not inherit any stale HERMES_TUI_RESUME value.
    What: calls _resolve_chat_argv with no resume arg.
    Test: HERMES_TUI_RESUME must not be in the returned env.
    """
    import hermes_cli.web_server as ws

    _patch_resolve_env(monkeypatch, session_db)

    _argv, _cwd, env = ws._resolve_chat_argv()

    assert env is not None
    assert "HERMES_TUI_RESUME" not in env, (
        f"HERMES_TUI_RESUME must be absent for fresh chat; got: {env.get('HERMES_TUI_RESUME')}"
    )


# ============================================================
# Priority 3 — active-session breadcrumb: _read_active_session_file
# ============================================================

def test_read_active_session_file_returns_session_id(tmp_path):
    """Why: _read_active_session_file is the breadcrumb the reconnect path
    uses; it must parse the JSON and return the session_id string.
    What: writes a JSON file with session_id; calls the real function.
    Test: returned value equals the written session_id.
    """
    from hermes_cli.web_server import _read_active_session_file

    active_file = tmp_path / "active.json"
    sid = "20260408_130000_breadcrumb"
    active_file.write_text(json.dumps({"session_id": sid}), encoding="utf-8")

    result = _read_active_session_file(active_file)

    assert result == sid, f"Expected {sid!r}, got {result!r}"


def test_read_active_session_file_missing_file_returns_none(tmp_path):
    """Why: a non-existent breadcrumb file must not crash reconnect logic.
    What: calls _read_active_session_file on a path that does not exist.
    Test: returns None without raising.
    """
    from hermes_cli.web_server import _read_active_session_file

    result = _read_active_session_file(tmp_path / "does-not-exist.json")

    assert result is None


def test_read_active_session_file_malformed_json_returns_none(tmp_path):
    """Why: corrupted breadcrumb files must degrade gracefully.
    What: writes invalid JSON; calls _read_active_session_file.
    Test: returns None without raising.
    """
    from hermes_cli.web_server import _read_active_session_file

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")

    result = _read_active_session_file(bad_file)

    assert result is None


def test_read_active_session_file_empty_session_id_returns_none(tmp_path):
    """Why: a breadcrumb file with empty/null session_id must be treated as absent.
    What: writes {"session_id": ""} and {"session_id": null}.
    Test: both return None.
    """
    from hermes_cli.web_server import _read_active_session_file

    for payload in ['{"session_id": ""}', '{"session_id": null}']:
        f = tmp_path / "empty.json"
        f.write_text(payload, encoding="utf-8")
        assert _read_active_session_file(f) is None, f"Expected None for payload: {payload}"
