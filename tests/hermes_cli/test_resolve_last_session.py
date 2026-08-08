"""Verify `hermes -c` picks the session the user most recently used."""

from __future__ import annotations

import pytest

from hermes_cli.main import _resolve_last_session


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def search_sessions(self, source=None, limit=20, **_kw):
        rows = [r for r in self._rows if r.get("source") == source] if source else list(self._rows)
        rows.sort(
            key=lambda r: float(r.get("last_active") or r.get("started_at") or 0),
            reverse=True,
        )
        return rows[:limit]

    def close(self):
        self.closed = True


def test_search_sessions_exposes_last_active_column(tmp_path, monkeypatch):
    # End-to-end: SessionDB must surface last_active and order by MRU.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import hermes_state

    from pathlib import Path

    db = hermes_state.SessionDB(db_path=Path(tmp_path / "state.db"))
    try:
        db.create_session("s_started_later", source="cli")
        db.create_session("s_active_later", source="cli")
        # Force started_at ordering so the test is deterministic regardless
        # of how quickly the two inserts land.
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (2000.0, "s_started_later"))
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (1000.0, "s_active_later"))
            db._conn.commit()

        db.append_message("s_active_later", role="user", content="hi")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (3000.0, "s_active_later"),
            )
            db._conn.commit()

        rows = db.search_sessions(source="cli", limit=5)
        ids = {r["id"]: r.get("last_active") for r in rows}

        assert ids["s_started_later"] == 2000.0
        assert ids["s_active_later"] == 3000.0
        assert rows[0]["id"] == "s_active_later"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# cwd-scoped resume: -c prefers the last session in the current workspace.
# ---------------------------------------------------------------------------


class _WorkspaceAwareDB:
    """Fake SessionDB whose ``search_sessions`` honors ``workspace_key`` the
    same way the real ``_workspace_key_clause`` does: a row matches the key
    when its ``git_repo_root`` equals it, or (no repo root recorded) when its
    ``cwd`` is at or under it."""

    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def search_sessions(self, source=None, limit=20, workspace_key=None, **_kw):
        rows = [r for r in self._rows if r.get("source") == source] if source else list(self._rows)
        if workspace_key:
            key = workspace_key.rstrip("/")
            def _in_ws(r):
                grr = (r.get("git_repo_root") or "").rstrip("/")
                if grr:
                    return grr == key
                cwd = (r.get("cwd") or "").rstrip("/")
                return cwd == key or cwd.startswith(key + "/")
            rows = [r for r in rows if _in_ws(r)]
        rows.sort(
            key=lambda r: float(r.get("last_active") or r.get("started_at") or 0),
            reverse=True,
        )
        return rows[:limit]

    def close(self):
        self.closed = True


def test_resolve_last_session_real_db_prefers_workspace(monkeypatch, tmp_path):
    # End-to-end through the real SessionDB + _resolve_last_session: -c from
    # repo A picks repo A's session even though repo B is globally newer.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import hermes_state
    from pathlib import Path

    repo_a = tmp_path / "repo-a"
    repo_a.mkdir()
    state_db = Path(tmp_path / "state.db")
    real_db = hermes_state.SessionDB
    db = real_db(db_path=state_db)
    try:
        db.create_session("repo_a", source="cli", cwd=str(repo_a), git_repo_root=str(repo_a))
        db.create_session("repo_b", source="cli", cwd="/other/repo-b", git_repo_root="/other/repo-b")
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (100.0, "repo_a"))
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (9000.0, "repo_b"))
            db._conn.commit()
    finally:
        db.close()

    monkeypatch.chdir(repo_a)
    monkeypatch.setattr(
        "hermes_cli.main.subprocess.run",
        lambda cmd, **kw: __import__("subprocess").CompletedProcess(
            cmd, 0, stdout=str(repo_a), stderr=""
        ),
    )
    monkeypatch.setattr("hermes_state.SessionDB", lambda: real_db(db_path=state_db))
    assert _resolve_last_session("cli") == "repo_a"


# ---------------------------------------------------------------------------
# Continue across interfaces (#47214): a bare `hermes -c` prefers the
# launching interface, then falls back across the local interactive family
# (cli/webui/tui). Gateway and automation sessions are never picked
# automatically. Webui chats were tagged "tui" in older builds and "webui" in
# newer ones, so both tags are tried. These cover the family preference and
# fallback, the gateway/automation exclusion, workspace scoping, and the
# string form backwards compat contract.
# ---------------------------------------------------------------------------

# Source tuples cmd_chat builds per interface mode (mirrors main.py).
_CLI_FAMILY = ("cli", "webui", "tui")
_TUI_FAMILY = ("tui", "webui", "cli")


def _use_fake_db(monkeypatch, rows, *, workspace_key=None):
    """Point _resolve_last_session at an in memory fake over the given rows.

    ``workspace_key`` selects the workspace-aware fake (which honors the key
    the way ``_workspace_key_clause`` does). None selects the plain fake and
    skips workspace resolution so only the global MRU path runs.
    """
    db = (_WorkspaceAwareDB(rows) if workspace_key else _FakeDB(rows))
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)
    monkeypatch.setattr(
        "hermes_cli.main._resolve_workspace_key", lambda: workspace_key
    )
    return db


def test_cli_family_prefers_cli_even_when_webui_is_newer(monkeypatch):
    # A more recent webui session must not beat a cli session in CLI mode.
    rows = [
        {"id": "web_newer", "source": "webui", "last_active": 9000},
        {"id": "cli_older", "source": "cli", "last_active": 1000},
    ]
    _use_fake_db(monkeypatch, rows)
    assert _resolve_last_session(source=_CLI_FAMILY) == "cli_older"


def test_cli_family_falls_back_to_webui_then_tui(monkeypatch):
    rows = [
        {"id": "tui1", "source": "tui", "last_active": 9000},
        {"id": "web1", "source": "webui", "last_active": 1000},
    ]
    _use_fake_db(monkeypatch, rows)
    # webui precedes tui in the CLI family, so it wins despite being older.
    assert _resolve_last_session(source=_CLI_FAMILY) == "web1"


def test_cli_family_falls_back_to_tui_alone(monkeypatch):
    rows = [{"id": "tui1", "source": "tui", "last_active": 1000}]
    _use_fake_db(monkeypatch, rows)
    assert _resolve_last_session(source=_CLI_FAMILY) == "tui1"


def test_tui_family_prefers_tui_then_webui_then_cli(monkeypatch):
    rows = [{"id": "cli1", "source": "cli", "last_active": 9000}]
    _use_fake_db(monkeypatch, rows)
    # No tui/webui -> cli is the last resort fallback in TUI mode.
    assert _resolve_last_session(source=_TUI_FAMILY) == "cli1"

    rows = [
        {"id": "web1", "source": "webui", "last_active": 9000},
        {"id": "cli1", "source": "cli", "last_active": 8000},
    ]
    _use_fake_db(monkeypatch, rows)
    # webui precedes cli in the TUI family.
    assert _resolve_last_session(source=_TUI_FAMILY) == "web1"


def test_family_never_returns_gateway_or_automation(monkeypatch):
    # Only gateway/automation sessions exist. None are in the local family.
    rows = [
        {"id": "tg1", "source": "telegram", "last_active": 9000},
        {"id": "cr1", "source": "cron", "last_active": 8000},
        {"id": "tool1", "source": "tool", "last_active": 7000},
    ]
    _use_fake_db(monkeypatch, rows)
    assert _resolve_last_session(source=_CLI_FAMILY) is None
    assert _resolve_last_session(source=_TUI_FAMILY) is None


def test_family_returns_none_when_no_session(monkeypatch):
    _use_fake_db(monkeypatch, [])
    assert _resolve_last_session(source=_CLI_FAMILY) is None


def test_string_source_keeps_single_source_behaviour(monkeypatch):
    # Backwards compat: a plain string source resolves only that source, with
    # no family fallback (the _print_tui_exit_summary caller relies on this).
    rows = [
        {"id": "cli1", "source": "cli", "last_active": 9000},
        {"id": "tui1", "source": "tui", "last_active": 1000},
    ]
    _use_fake_db(monkeypatch, rows)
    assert _resolve_last_session(source="tui") == "tui1"
    # No tui only db -> None, NOT a fallback into cli.
    _use_fake_db(monkeypatch, [{"id": "cli1", "source": "cli", "last_active": 1000}])
    assert _resolve_last_session(source="tui") is None


def test_workspace_scope_preferred_within_each_family_source(monkeypatch):
    # The workspace scoping (workspace first, then global MRU) is preserved on the tuple path:
    # an older cli session in THIS workspace beats a newer cli session elsewhere.
    rows = [
        {"id": "cli_global_newer", "source": "cli", "last_active": 9000,
         "git_repo_root": "/other/repo-b"},
        {"id": "cli_ws_older", "source": "cli", "last_active": 1000,
         "git_repo_root": "/ws/repo-a"},
    ]
    _use_fake_db(monkeypatch, rows, workspace_key="/ws/repo-a")
    assert _resolve_last_session(source=_CLI_FAMILY) == "cli_ws_older"


def test_workspace_then_global_then_family_fallback(monkeypatch):
    # No cli in the workspace, no cli anywhere, but a webui session globally ->
    # the family fallback still resolves it after workspace + global cli miss.
    rows = [
        {"id": "web_global", "source": "webui", "last_active": 5000,
         "git_repo_root": "/other/repo-b"},
    ]
    _use_fake_db(monkeypatch, rows, workspace_key="/ws/repo-a")
    assert _resolve_last_session(source=_CLI_FAMILY) == "web_global"


def test_cmd_chat_continue_passes_local_family_per_mode(monkeypatch):
    # cmd_chat builds the family tuple that prefers the launching interface and hands it to
    # _resolve_last_session. Halt right after the call so we don't run startup.
    import hermes_cli.main as main_mod
    from types import SimpleNamespace

    class _Stop(Exception):
        pass

    captured: dict = {}

    def fake_resolve(*, source=None):
        captured["source"] = source
        raise _Stop

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve)

    def run(use_tui, args):
        captured.clear()
        monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: use_tui)
        monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda _args: None)
        with pytest.raises(_Stop):
            main_mod.cmd_chat(
                SimpleNamespace(continue_last=True, resume=None)
            )
        return captured["source"]

    assert run(use_tui=False, args=None) == ("cli", "webui", "tui")
    assert run(use_tui=True, args=None) == ("tui", "webui", "cli")
