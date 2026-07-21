"""Behavior and integration coverage for the stdlib lightweight dashboard."""

from __future__ import annotations

import argparse
import builtins
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import cmd_dashboard


def _arguments(**overrides):
    defaults = {
        "headless_backend": False,
        "host": "127.0.0.1",
        "insecure": False,
        "isolated": False,
        "light_dashboard": False,
        "no_open": True,
        "open_profile": "",
        "port": 0,
        "skip_build": False,
        "status": False,
        "stop": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_lightweight_branch_precedes_full_web_imports():
    imported = builtins.__import__
    starts = []

    def guarded_import(name, *args, **kwargs):
        if name in {"fastapi", "uvicorn", "hermes_cli.web_server"}:
            raise AssertionError(f"full dashboard import reached: {name}")
        return imported(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=guarded_import),
        patch(
            "hermes_cli.lightweight_dashboard.run_lightweight_dashboard",
            side_effect=lambda **kwargs: starts.append(kwargs),
        ),
    ):
        cmd_dashboard(_arguments(light_dashboard=True))

    assert starts == [
        {
            "allow_remote": False,
            "host": "127.0.0.1",
            "initial_profile": "",
            "open_browser": False,
            "port": 0,
        }
    ]


def test_config_mode_selects_lightweight_only_for_dashboard():
    starts = []
    with (
        patch(
            "hermes_cli.config.read_raw_config",
            return_value={"dashboard": {"mode": "lightweight"}},
        ),
        patch(
            "hermes_cli.lightweight_dashboard.run_lightweight_dashboard",
            side_effect=lambda **kwargs: starts.append(kwargs),
        ),
    ):
        cmd_dashboard(_arguments())
    assert len(starts) == 1


def test_profile_view_resolves_named_profile(monkeypatch, tmp_path):
    from hermes_cli.lightweight_dashboard import ProfileView

    home = tmp_path / "worker"
    home.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "worker"
    )
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: home)

    view = ProfileView("Worker")

    assert view.name == "worker"
    assert view.home == home


def test_sessions_use_compact_read_only_database(monkeypatch, tmp_path):
    from hermes_cli import lightweight_dashboard as light

    home = tmp_path / "profile"
    home.mkdir()
    (home / "state.db").touch()
    observed = {}

    class FakeDatabase:
        def __init__(self, *, db_path, read_only):
            observed["open"] = (db_path, read_only)

        def list_sessions_rich(self, **kwargs):
            observed["list"] = kwargs
            return [
                {
                    "archived": 0,
                    "ended_at": None,
                    "id": "s1",
                    "last_active": 1,
                    "model_config": "large",
                    "system_prompt": "large",
                }
            ]

        def session_count(self, **kwargs):
            observed["count"] = kwargs
            return 1

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(light.ProfileView, "identity", ("default", home))
    monkeypatch.setattr("hermes_state.SessionDB", FakeDatabase)

    payload = light.ProfileView().sessions(limit=5, offset=2, order="recent")

    assert observed["open"] == (home / "state.db", True)
    assert observed["list"] == {
        "compact_rows": True,
        "limit": 5,
        "offset": 2,
        "order_by_last_active": True,
    }
    assert observed["count"] == {"exclude_children": True}
    assert observed["closed"] is True
    assert "system_prompt" not in payload["sessions"][0]
    assert "model_config" not in payload["sessions"][0]


def test_file_view_is_confined_and_hides_credentials(monkeypatch, tmp_path):
    from hermes_cli.lightweight_dashboard import DashboardProblem, ProfileView

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_text("hello", encoding="utf-8")
    (root / ".env").write_text("SECRET=value", encoding="utf-8")
    (root / "config.yaml").write_text("api_key: value", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside)
    view = ProfileView()
    monkeypatch.setattr(ProfileView, "files_root", lambda self: root)

    listing = view.directory(None)
    preview = view.file_preview("README.md")

    assert [entry["name"] for entry in listing["entries"]] == ["README.md"]
    assert preview["content"] == "hello"
    with pytest.raises(DashboardProblem) as exc:
        view.file_preview(str(outside))
    assert exc.value.status.value == 403


def test_config_view_is_allowlisted(monkeypatch):
    from hermes_cli.lightweight_dashboard import ProfileView

    view = ProfileView()
    monkeypatch.setattr(ProfileView, "identity", ("default", Path("/tmp/profile")))
    monkeypatch.setattr(
        ProfileView,
        "raw_config",
        lambda self: {
            "api_key": "must-not-leak",
            "gateway": {"telegram": {"token": "must-not-leak"}},
            "model": "example/model",
            "provider": "example",
            "terminal": {"backend": "docker", "cwd": "/work", "secret": "no"},
        },
    )

    payload = view.safe_config()

    assert payload["config"] == {
        "dashboard": {"mode": "lightweight"},
        "model": "example/model",
        "provider": "example",
        "terminal": {"backend": "docker", "cwd": "/work"},
    }
    assert "must-not-leak" not in json.dumps(payload)


def test_remote_bind_requires_explicit_override():
    from hermes_cli.lightweight_dashboard import run_lightweight_dashboard

    with pytest.raises(SystemExit, match="non-loopback"):
        run_lightweight_dashboard(
            host="100.64.0.10",
            port=0,
            open_browser=False,
            initial_profile="",
            allow_remote=False,
        )


def test_http_server_reads_real_profile_state(monkeypatch, tmp_path):
    from hermes_cli import lightweight_dashboard as light
    from hermes_state import SessionDB

    home = tmp_path / "profile"
    workspace = tmp_path / "workspace"
    logs = home / "logs"
    home.mkdir()
    workspace.mkdir()
    logs.mkdir()
    (workspace / "notes.txt").write_text("workspace preview", encoding="utf-8")
    (logs / "agent.log").write_text("INFO lightweight ready\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "model: example/model\n"
        "api_key: must-not-leak\n"
        "terminal:\n"
        f"  cwd: {workspace}\n",
        encoding="utf-8",
    )
    database = SessionDB(db_path=home / "state.db")
    database.create_session("session-http", "cli", model="example/model")
    database.append_message("session-http", "user", "hello from sqlite")
    database.close()

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: home)

    server = ThreadingHTTPServer(("127.0.0.1", 0), light.LightweightHandler)
    server.accepted_hosts = {"127.0.0.1"}
    server.accept_ip_hosts = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as response:
            return response.status, json.loads(response.read())

    try:
        sessions = get("/api/sessions")[1]
        detail = get("/api/sessions/session-http")[1]
        transcript = get("/api/sessions/session-http/messages")[1]
        files = get("/api/files")[1]
        preview = get("/api/files/read?path=notes.txt")[1]
        log_tail = get("/api/logs?file=agent")[1]
        config = get("/api/config")[1]
        request = urllib.request.Request(base, headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as bad_host:
            urllib.request.urlopen(request, timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert sessions["sessions"][0]["id"] == "session-http"
    assert detail["profile"] == "default"
    assert transcript["messages"][0]["content"] == "hello from sqlite"
    assert files["entries"][0]["name"] == "notes.txt"
    assert preview["content"] == "workspace preview"
    assert log_tail["lines"] == ["INFO lightweight ready"]
    assert config["config"]["model"] == "example/model"
    assert "must-not-leak" not in json.dumps(config)
    assert bad_host.value.code == 400


def test_handler_forwards_profile_query(monkeypatch):
    from hermes_cli import lightweight_dashboard as light

    received = {}
    handler = object.__new__(light.LightweightHandler)
    handler.path = "/api/config?profile=Worker"
    handler.headers = {"Host": "127.0.0.1:9119"}
    handler.server = SimpleNamespace(
        accepted_hosts={"127.0.0.1"}, accept_ip_hosts=False
    )
    handler.json_reply = lambda status, payload: received.update(
        status=status, payload=payload
    )
    monkeypatch.setattr(
        light.ProfileView,
        "safe_config",
        lambda self: {"requested": self.requested_name},
    )

    handler.do_GET()

    assert received["payload"] == {"requested": "Worker"}
