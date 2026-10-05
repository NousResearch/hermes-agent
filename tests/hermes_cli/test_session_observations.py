"""Exact-ID observation reads are authenticated, profile-isolated and non-mutating."""
import argparse
import json
import os
import sqlite3
from starlette.testclient import TestClient
from hermes_state import SessionDB


def test_dashboard_exact_identity_read_only_profiles_and_auth(tmp_path, monkeypatch):
    home = tmp_path / "home"
    b = home / "profiles" / "b"
    b.mkdir(parents=True)
    for path in [home, b]:
        (path / "config.yaml").write_text("model: test\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    for path, profile in [(home, "default"), (b, "b")]:
        with SessionDB(path / "state.db") as db:
            db.create_session("same", "desktop", profile_name=profile)
            db.create_session("hidden", "desktop", profile_name=profile)
            db._write_sql("UPDATE sessions SET hidden = 1 WHERE id = 'hidden'", ())
            db.create_session("private", "tool", profile_name=profile)
            holder = f"pid={os.getpid()}:turn={profile}"
            db.try_acquire_session_turn_lease("same", holder)
            turn = db.begin_session_observation("same", holder)
            db.open_session_attention("same", turn, "validation")
            db.finish_session_observation("same", holder, turn, "complete")
            db.release_session_turn_lease("same", holder)
    from hermes_cli import web_server as web
    client = TestClient(web.app)
    url = "/api/session-observations"
    assert client.get(url, params={"session_id": "same"}).status_code == 401
    client.headers[web._SESSION_HEADER_NAME] = web._SESSION_TOKEN
    seen = []
    for profile, path in [("default", home), ("b", b), ("default", home)]:
        before = (path / "state.db").read_bytes()
        response = client.get(url, params=[("profile", profile), ("session_id", "same"), ("session_id", "missing"), ("session_id", "hidden"), ("session_id", "private")])
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        wire = response.json()
        assert wire["schema_version"] == 1 and wire["profile"] == profile
        row = wire["observations"][0]
        assert row["session_id"] == row["lineage_tip_id"] == "same" and row["profile"] == profile
        assert row["execution"] == "idle" and row["attention"]["kind"] == "validation"
        seen.append(row["turn_id"])
        assert all(r["execution"] == "unknown" and r["lineage_tip_id"] is None for r in wire["observations"][1:])
        assert (path / "state.db").read_bytes() == before
    assert seen[0] == seen[2] != seen[1]
    for ids in [[" same"], ["samé"], ["same", "same"], ["x" * 101], [f"x{i}" for i in range(41)]]:
        assert client.get(url, params=[("session_id", i) for i in ids]).status_code == 400
    assert client.get(url, params={"session_id": "same", "profile": "missing"}).status_code == 404
    empty = home / "profiles" / "empty"
    empty.mkdir()
    (empty / "config.yaml").write_text("model: test\n")
    assert client.get(url, params={"session_id": "same", "profile": "empty"}).json()["observations"][0]["execution"] == "unknown"
    assert not (empty / "state.db").exists()
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("DROP TABLE session_observations")
    before = (home / "state.db").read_bytes()
    assert client.get(url, params={"session_id": "same", "profile": "default"}).json()["observations"][0]["execution"] == "unknown"
    assert (home / "state.db").read_bytes() == before


def test_native_cli_declare_and_resolve_exact_generation(tmp_path, monkeypatch, capsys):
    from hermes_cli.subcommands.sessions import build_sessions_parser
    from hermes_cli.sessions_cmd import cmd_sessions
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    with SessionDB(home / "state.db") as db:
        db.create_session("s", "cli", profile_name="default")
        holder = f"pid={os.getpid()}:turn=cli"
        db.try_acquire_session_turn_lease("s", holder)
        turn = db.begin_session_observation("s", holder)
        db.finish_session_observation("s", holder, turn, "complete")
        db.release_session_turn_lease("s", holder)
    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=cmd_sessions)
    assert "attention" in parser._subparsers._group_actions[0].choices["sessions"].format_help(), "explicit native CLI missing"
    args = parser.parse_args(["sessions", "attention", "open", "s", "--turn-id", turn])
    assert args.func(args) == 0
    request = json.loads(capsys.readouterr().out)["request_id"]
    args = parser.parse_args(["sessions", "attention", "resolve", "s", "--turn-id", "obsolete", "--request-id", request])
    assert args.func(args) == 1
    capsys.readouterr()
    args = parser.parse_args(["sessions", "attention", "resolve", "s", "--turn-id", turn, "--request-id", request])
    assert args.func(args) == 0
    with SessionDB(home / "state.db", read_only=True) as db:
        row = db.read_session_observations(["s"], profile="default")[0]
        assert row["attention"]["kind"] == "none" and row["last_result"]["status"] == "complete"


def test_dashboard_mixed_unverifiable_configs_remain_read_only_unknown(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model: test\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    path = home / "state.db"
    configs = ["[]", "null", "true", "42", '"synthetic"', "{bad"]
    with SessionDB(path) as db:
        db.create_session("good", "desktop", profile_name="default")
        for i, cfg in enumerate(configs):
            sid = f"bad{i}"
            db.create_session(sid, "desktop", profile_name="default")
            db._write_sql("UPDATE sessions SET model_config=? WHERE id=?", (cfg, sid))
    from hermes_cli import web_server as web
    client = TestClient(web.app, raise_server_exceptions=False)
    params = [("profile", "default"), ("session_id", "good"),
              *(("session_id", f"bad{i}") for i in range(len(configs)))]
    assert client.get("/api/session-observations", params=params).status_code == 401
    client.headers[web._SESSION_HEADER_NAME] = web._SESSION_TOKEN
    before = path.read_bytes()
    response = client.get("/api/session-observations", params=params)
    assert response.status_code == 200, "an invalid identity must not fail the authenticated batch"
    assert response.headers["cache-control"] == "no-store"
    rows = response.json()["observations"]
    assert rows[0]["lineage_tip_id"] == "good", "invalid sibling discarded healthy identity"
    assert all(r["execution"] == "unknown" and r["lineage_tip_id"] is None for r in rows[1:])
    assert path.read_bytes() == before
