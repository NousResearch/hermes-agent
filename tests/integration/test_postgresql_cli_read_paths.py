"""Selected-PostgreSQL CLI read paths: status sessions, breadcrumbs, update FTS probe.

Under ``state_store.backend=postgresql`` these three surfaces must read the selected
store (or explicitly skip) instead of swallowing the activation error and rendering
empty data — and none of them may open or create ``state.db``.
"""
from __future__ import annotations

import re

import pytest
from types import SimpleNamespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from state_store import open_state_store
from state_store_runtime_readiness import trap_state_db_opens

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_CONFIG = {"state_store": {"backend": "postgresql", "postgresql": {
    "dsn_env": "HERMES_STATE_STORE_TEST_DSN", "connect_timeout_seconds": 5, "pool_max_size": 2,
}}}

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_read_home(tmp_path, monkeypatch, postgresql_test_target):
    home = tmp_path / ".hermes-pg-cli-read-paths"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n"
        "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n    connect_timeout_seconds: 5\n    pool_max_size: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    import state_store
    monkeypatch.setattr(state_store, "postgresql_tenant_schema", lambda *_a, **_k: postgresql_test_target.schema)
    token = set_hermes_home_override(str(home))
    stores: list = []
    try:
        yield home, stores, postgresql_test_target.schema
    finally:
        for store in stores:
            try:
                store.close()
            except Exception:
                pass
        reset_hermes_home_override(token)


@pytest.fixture
def no_terminal_env(monkeypatch):
    for var in ("ZELLIJ_PANE_ID", "TMUX_PANE", "KITTY_WINDOW_ID", "WEZTERM_PANE", "TERM_SESSION_ID", "WT_SESSION"):
        monkeypatch.delenv(var, raising=False)


def _seed_route(stores, namespace, *, session_key, session_id, source="gateway"):
    store = open_state_store(_CONFIG)
    stores.append(store)
    store.get_or_create_gateway_session_route(namespace, session_key, session_id, {
        "source": source, "session_key": session_key,
    })
    return store


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ------------------------------------------------------------------ hermes status

def test_status_sessions_renders_pg_route_counts(pg_read_home, capsys):
    home, stores, namespace = pg_read_home
    _seed_route(stores, namespace, session_key="telegram:chat-a", session_id="20260921_010101_pgstatus_a")
    _seed_route(stores, namespace, session_key="discord:chan-b", session_id="20260921_010102_pgstatus_b")

    from hermes_cli.status import _render_sessions

    with trap_state_db_opens(home) as events:
        _render_sessions(SimpleNamespace(config=_CONFIG, deep=False))

    out = _strip_ansi(capsys.readouterr().out)
    assert "Active:" in out and "2 session(s)" in out
    assert not re.search(r"Active:\s+0\b", out)
    assert events == []
    assert not (home / "state.db").exists()


def test_status_sessions_active_only_excludes_ended_sessions(pg_read_home, capsys):
    home, stores, namespace = pg_read_home
    store = _seed_route(stores, namespace, session_key="telegram:chat-a", session_id="20260921_010103_pgstatus_live")
    _seed_route(stores, namespace, session_key="slack:chan-c", session_id="20260921_010104_pgstatus_done")
    store.end_session("20260921_010104_pgstatus_done", "session_reset")

    from hermes_cli.status import _render_sessions

    with trap_state_db_opens(home) as events:
        _render_sessions(SimpleNamespace(config=_CONFIG, deep=False))

    out = _strip_ansi(capsys.readouterr().out)
    assert "Active:" in out and "1 session(s)" in out
    assert events == []
    assert not (home / "state.db").exists()


def test_status_unavailable_pg_store_surfaces_error_not_zero(pg_read_home, monkeypatch, capsys):
    home, _stores, _namespace = pg_read_home
    # Simulate an unreachable selected store: selection still resolves PostgreSQL, opening fails.
    import cli_session_store
    real_open = cli_session_store.open_state_store
    monkeypatch.setattr(
        cli_session_store, "open_state_store",
        lambda config, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated selected-store outage")))
    from hermes_cli.status import _render_sessions
    try:
        with trap_state_db_opens(home) as events:
            _render_sessions(SimpleNamespace(config=_CONFIG, deep=False))
    finally:
        monkeypatch.setattr(cli_session_store, "open_state_store", real_open)

    out = _strip_ansi(capsys.readouterr().out)
    assert "error reading selected state store" in out
    assert "simulated selected-store outage" in out
    assert not re.search(r"Active:\s+0\b", out)
    assert events == []
    assert not (home / "state.db").exists()


def test_status_invalid_pg_config_surfaces_error_not_zero(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes-pg-invalid"
    home.mkdir()
    (home / "config.yaml").write_text("state_store:\n  backend: postgresql\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.status import _render_sessions

    invalid_config = {"state_store": {"backend": "postgresql"}}  # mirrors the broken config.yaml
    with trap_state_db_opens(home) as events:
        _render_sessions(SimpleNamespace(config=invalid_config, deep=False))

    out = _strip_ansi(capsys.readouterr().out)
    assert "error reading selected state store" in out
    assert not re.search(r"Active:\s+0\b", out)
    assert events == []
    assert not (home / "state.db").exists()


# ------------------------------------------------------------------ breadcrumbs

def test_breadcrumbs_resolve_pg_session_lineage(pg_read_home, monkeypatch, no_terminal_env, capfd):
    home, stores, namespace = pg_read_home
    store = _seed_route(stores, namespace, session_key="telegram:chat-a", session_id="20260921_010105_pgcrumb_parent")
    # Compression chain: ended parent -> live child carries the conversation tip.
    store.ensure_session("20260921_010106_pgcrumb_child", "gateway", metadata={
        "parent_session_id": "20260921_010105_pgcrumb_parent"})
    store.end_session("20260921_010105_pgcrumb_parent", "compression")

    from hermes_cli import terminal_breadcrumbs as tb
    monkeypatch.setattr(tb.os, "ttyname", lambda fd: "/dev/pts/5")
    tb.write_breadcrumb("20260921_010105_pgcrumb_parent")

    with trap_state_db_opens(home) as events:
        assert tb.resolve_breadcrumb_session() == "20260921_010106_pgcrumb_child"
    assert capfd.readouterr().err == ""
    assert events == []
    assert not (home / "state.db").exists()


def test_breadcrumbs_deleted_pg_session_falls_back(pg_read_home, monkeypatch, no_terminal_env, capfd):
    home, _stores, _namespace = pg_read_home
    from hermes_cli import terminal_breadcrumbs as tb
    monkeypatch.setattr(tb.os, "ttyname", lambda fd: "/dev/pts/5")
    tb.write_breadcrumb("20260921_010107_pgcrumb_gone")

    with trap_state_db_opens(home) as events:
        assert tb.resolve_breadcrumb_session() is None
    assert capfd.readouterr().err == ""
    assert events == []
    assert not (home / "state.db").exists()


def test_breadcrumbs_unavailable_pg_store_skips_with_message(pg_read_home, monkeypatch, no_terminal_env, capfd):
    home, _stores, _namespace = pg_read_home
    import cli_session_store
    monkeypatch.setattr(
        cli_session_store, "open_state_store",
        lambda config, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated selected-store outage")))
    from hermes_cli import terminal_breadcrumbs as tb
    monkeypatch.setattr(tb.os, "ttyname", lambda fd: "/dev/pts/5")
    tb.write_breadcrumb("20260921_010108_pgcrumb_any")

    with trap_state_db_opens(home) as events:
        assert tb.resolve_breadcrumb_session() is None
    err = capfd.readouterr().err
    assert "terminal breadcrumb" in err
    assert "skipping breadcrumb resume" in err
    assert events == []
    assert not (home / "state.db").exists()


# ------------------------------------------------------------------ hermes update FTS probe

def test_update_fts_probe_no_op_under_selected_pg(pg_read_home, capsys):
    home, _stores, _namespace = pg_read_home
    # A sparse >0.5 GB state.db clears the size gate, so only the selected-backend
    # branch (not the early returns) can prevent the SQLite FTS probe.
    db_file = home / "state.db"
    db_file.touch()
    import os as _os
    _os.truncate(db_file, 600 * 1024 * 1024)

    from hermes_cli.update_cmd_maint import _print_fts_optimize_available_notice

    with trap_state_db_opens(home) as events:
        _print_fts_optimize_available_notice()

    assert capsys.readouterr().out == ""
    assert events == []
