"""The tenancy file scanner checks each routing row the multiplex launch-home index holds for another
profile as that profile's: its own data is no leak, a third profile's canary in it still is, and a
foreign canary anywhere else in the launch home (its own session rows and routing entries) is a leak.
A foreign secret is a leak anywhere in the raw bytes, deleted rows still held in the SQLite WAL included."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import _helpers as H


def _tenants(tmp_path: Path) -> dict[str, H.Tenant]:
    home = tmp_path / "home" / ".hermes"
    return {name: H.Tenant(name=name, home=home if name == "default" else home / "profiles" / name,
                           workdir=tmp_path / f"work-{name}")
            for name in ("default", "alpha", "beta")}


def _write_launch_home(t: dict[str, H.Tenant], *, own_row_cwd: str, own_route_cwd: str,
                       alpha_route_cwd: str | None = None, alpha_route_key: str | None = None,
                       own_route_key: str | None = None) -> None:
    home = t["default"].home
    (home / "sessions").mkdir(parents=True)
    alpha_cwd = alpha_route_cwd or str(t["alpha"].workdir)
    alpha_meta = {"terminal_cwd": alpha_cwd, **({"api_key": alpha_route_key} if alpha_route_key else {})}
    routes = {"agent:alpha:api_server:dm:a": {"metadata": alpha_meta},
              "agent:main:api_server:dm:m": {"metadata": {"terminal_cwd": own_route_cwd,
                                                          **({"api_key": own_route_key} if own_route_key else {})}}}
    (home / "sessions" / "sessions.json").write_text(json.dumps(routes), encoding="utf-8")
    conn = sqlite3.connect(home / "state.db")
    conn.executescript("CREATE TABLE sessions (id TEXT, cwd TEXT);"
                       "CREATE TABLE gateway_routing (scope TEXT, session_key TEXT, entry_json TEXT);")
    conn.execute("INSERT INTO sessions VALUES ('m', ?)", (own_row_cwd,))
    conn.executemany("INSERT INTO gateway_routing VALUES ('', ?, ?)",
                     [(k, json.dumps(v)) for k, v in routes.items()])
    conn.commit()
    conn.close()


def test_another_profiles_routing_entry_in_the_launch_index_is_not_a_leak(tmp_path: Path) -> None:
    t = _tenants(tmp_path)
    _write_launch_home(t, own_row_cwd=str(t["default"].workdir), own_route_cwd=str(t["default"].workdir))
    assert H.file_leaks(t) == []


def test_a_foreign_workdir_on_the_launch_profiles_own_records_is_still_a_leak(tmp_path: Path) -> None:
    t = _tenants(tmp_path)
    alpha = str(t["alpha"].workdir)
    _write_launch_home(t, own_row_cwd=alpha, own_route_cwd=alpha)
    assert set(H.file_leaks(t)) == {"default: state.db contains alpha's workdir",
                                    "default: sessions/sessions.json contains alpha's workdir"}


def test_a_third_profiles_workdir_in_another_profiles_routing_row_is_a_leak(tmp_path: Path) -> None:
    t = _tenants(tmp_path)
    own = str(t["default"].workdir)
    _write_launch_home(t, own_row_cwd=own, own_route_cwd=own, alpha_route_cwd=str(t["beta"].workdir))
    assert set(H.file_leaks(t)) == {"default: alpha's routing row in state.db contains beta's workdir",
                                    "default: alpha's routing row in sessions/sessions.json contains beta's workdir"}


def test_the_launch_profiles_secret_in_another_profiles_routing_row_is_a_leak(tmp_path: Path) -> None:
    t = _tenants(tmp_path)
    own = str(t["default"].workdir)
    _write_launch_home(t, own_row_cwd=own, own_route_cwd=own, alpha_route_key=t["default"].provider_key)
    assert set(H.file_leaks(t)) == {
        "default: alpha's routing row in state.db contains default's provider_key",
        "default: alpha's routing row in sessions/sessions.json contains default's provider_key"}


def test_the_launch_profiles_secret_in_its_own_routing_row_is_not_a_leak(tmp_path: Path) -> None:
    t = _tenants(tmp_path)
    own = str(t["default"].workdir)
    _write_launch_home(t, own_row_cwd=own, own_route_cwd=own, own_route_key=t["default"].provider_key)
    assert H.file_leaks(t) == []


def test_a_deleted_foreign_credential_still_in_the_wal_is_a_leak(tmp_path: Path) -> None:
    t = _tenants(tmp_path)
    home = t["default"].home
    home.mkdir(parents=True)
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE sessions (id TEXT, content TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('m', ?)", (t["alpha"].provider_key,))
        conn.commit()
        conn.execute("DELETE FROM sessions")
        conn.commit()
        assert t["alpha"].provider_key.encode() in (home / "state.db-wal").read_bytes()
        assert "default: state.db-wal contains alpha's provider_key" in H.file_leaks(t)
    finally:
        conn.close()
