"""Tests for the lightweight dashboard mode."""

from __future__ import annotations

import argparse
import builtins
import json
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import cmd_dashboard


def _ns(**kw):
    defaults = dict(
        port=9119,
        host="127.0.0.1",
        no_open=True,
        insecure=False,
        stop=False,
        status=False,
        isolated=False,
        open_profile="",
        skip_build=False,
        headless_backend=False,
        light_dashboard=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_light_dashboard_does_not_import_full_web_stack():
    calls = []
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"fastapi", "uvicorn", "hermes_cli.web_server"}:
            raise AssertionError(f"unexpected heavy import: {name}")
        return orig_import(name, *args, **kwargs)

    def fake_start(**kwargs):
        calls.append(kwargs)

    with patch("builtins.__import__", side_effect=fake_import), patch(
        "hermes_cli.light_dashboard_server.start_light_dashboard_server",
        side_effect=fake_start,
    ):
        cmd_dashboard(_ns(light_dashboard=True, port=0))

    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 0,
            "open_browser": False,
            "initial_profile": "",
        }
    ]


def test_light_dashboard_can_be_enabled_from_config():
    calls = []

    with patch(
        "hermes_cli.config.load_config",
        return_value={"dashboard": {"mode": "lightweight"}},
    ), patch(
        "hermes_cli.light_dashboard_server.start_light_dashboard_server",
        side_effect=lambda **kwargs: calls.append(kwargs),
    ):
        cmd_dashboard(_ns(port=0))

    assert calls


def test_light_dashboard_refuses_non_loopback_bind():
    from hermes_cli.light_dashboard_server import start_light_dashboard_server

    with pytest.raises(SystemExit) as exc:
        start_light_dashboard_server(host="0.0.0.0", port=0, open_browser=False)

    assert "non-loopback" in str(exc.value)


def test_light_dashboard_ready_file(tmp_path):
    from hermes_cli.light_dashboard_server import _write_dashboard_ready_file

    target = tmp_path / "ready.json"
    with patch.dict("os.environ", {"HERMES_DESKTOP_READY_FILE": str(target)}):
        _write_dashboard_ready_file(9123)

    assert json.loads(target.read_text(encoding="utf-8")) == {"port": 9123}


def test_light_dashboard_host_header_helpers_reject_rebinding_hosts():
    from hermes_cli.light_dashboard_server import (
        _allowed_host_headers,
        _normalise_host_header,
    )

    allowed = _allowed_host_headers("127.0.0.1")

    assert _normalise_host_header("127.0.0.1:9119") in allowed
    assert _normalise_host_header("evil.example:9119") not in allowed
    assert _normalise_host_header("[::1]:9119") in _allowed_host_headers("::1")


def test_light_dashboard_sessions_use_compact_read_only_db(monkeypatch):
    from hermes_cli import light_dashboard_server as light

    captured = {}

    class FakeDB:
        def list_sessions_rich(self, **kwargs):
            captured["list"] = kwargs
            return [
                {
                    "id": "s1",
                    "title": "Session",
                    "started_at": 1,
                    "last_active": 2,
                    "ended_at": None,
                    "message_count": 3,
                    "archived": 0,
                    "system_prompt": "large",
                    "model_config": "large",
                }
            ]

        def session_count(self, **kwargs):
            captured["count"] = kwargs
            return 1

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(
        light, "_open_session_db_read_only", lambda profile=None: FakeDB()
    )

    payload = light.build_sessions_payload(limit=5, offset=7, order="recent")

    assert captured["list"] == {
        "limit": 5,
        "offset": 7,
        "order_by_last_active": True,
        "compact_rows": True,
    }
    assert captured["count"] == {"exclude_children": True}
    assert captured["closed"] is True
    assert payload["sessions"][0]["archived"] is False
    assert "system_prompt" not in payload["sessions"][0]
    assert "model_config" not in payload["sessions"][0]


def test_light_dashboard_resolves_named_profile_data(monkeypatch, tmp_path):
    from hermes_cli import light_dashboard_server as light

    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (profile_home / "state.db").touch()
    captured = {}

    class FakeDB:
        def __init__(self, *, db_path, read_only):
            captured["db_path"] = db_path
            captured["read_only"] = read_only

        def list_sessions_rich(self, **kwargs):
            return []

        def session_count(self, **kwargs):
            return 0

        def close(self):
            pass

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_home)
    monkeypatch.setattr("hermes_state.SessionDB", FakeDB)

    light.build_sessions_payload(profile="Worker")

    assert captured == {"db_path": profile_home / "state.db", "read_only": True}


def test_light_dashboard_status_uses_named_profile_runtime(monkeypatch, tmp_path):
    from hermes_cli import light_dashboard_server as light

    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    captured = {}
    runtime = {"gateway_state": "running", "active_agents": 2}

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_home)
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda path: captured.setdefault("runtime_path", path) and runtime,
    )
    monkeypatch.setattr(
        "gateway.status.get_running_pid_cached",
        lambda path, **kwargs: captured.update(pid_path=path, pid_kwargs=kwargs),
    )

    def runtime_pid(payload, **kwargs):
        captured["runtime_pid"] = (payload, kwargs)
        return 42

    monkeypatch.setattr("gateway.status.get_runtime_status_running_pid", runtime_pid)
    monkeypatch.setattr(light, "_active_session_count", lambda profile=None: 0)

    payload = light.build_status_payload("Worker")

    assert captured == {
        "runtime_path": profile_home / "gateway_state.json",
        "pid_path": profile_home / "gateway.pid",
        "pid_kwargs": {"cleanup_stale": False},
        "runtime_pid": (runtime, {"expected_home": profile_home}),
    }
    assert payload["gateway_pid"] == 42
    assert payload["gateway_running"] is True


def test_light_dashboard_forwards_page_profile_to_status_and_sessions():
    from hermes_cli import light_dashboard_server as light

    html = light._dashboard_html().decode("utf-8")

    assert 'new URLSearchParams(window.location.search).get("profile")' in html
    assert 'statusParams.set("profile", profile)' in html
    assert 'sessionParams.set("profile", profile)' in html

    captured = {}
    handler = object.__new__(light._LightDashboardHandler)
    handler.path = "/api/status?profile=Worker"
    handler.headers = {"Host": "127.0.0.1:9119"}
    handler.server = SimpleNamespace(allowed_host_headers={"127.0.0.1"})
    handler._send_json = lambda status, payload: captured.update(
        status=status, payload=payload
    )

    with patch.object(
        light,
        "build_status_payload",
        side_effect=lambda profile: {"profile": profile},
    ):
        handler.do_GET()

    assert captured == {
        "status": HTTPStatus.OK,
        "payload": {"profile": "Worker"},
    }


def test_light_dashboard_rejects_invalid_or_missing_profiles(monkeypatch):
    from hermes_cli import light_dashboard_server as light

    with pytest.raises(ValueError, match="Invalid profile name"):
        light._resolve_profile_home("../worker")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)
    with pytest.raises(ValueError, match="does not exist"):
        light._resolve_profile_home("worker")
