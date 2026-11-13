"""The tenancy file scanner exempts only what the multiplex routing index holds for other profiles:
a foreign canary anywhere else in the launch home (its own session rows, its own routing entries,
any other file) is still a leak."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import _helpers as H


def _tenants(tmp_path: Path) -> dict[str, H.Tenant]:
    home = tmp_path / "home" / ".hermes"
    return {name: H.Tenant(name=name, home=home if name == "default" else home / "profiles" / name,
                           workdir=tmp_path / f"work-{name}")
            for name in ("default", "alpha")}


def _write_launch_home(t: dict[str, H.Tenant], *, own_row_cwd: str, own_route_cwd: str) -> None:
    home = t["default"].home
    (home / "sessions").mkdir(parents=True)
    routes = {"agent:alpha:api_server:dm:a": {"metadata": {"terminal_cwd": str(t["alpha"].workdir)}},
              "agent:main:api_server:dm:m": {"metadata": {"terminal_cwd": own_route_cwd}}}
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
