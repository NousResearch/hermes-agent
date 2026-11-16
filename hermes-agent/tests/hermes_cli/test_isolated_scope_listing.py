"""An operator's ``-p <name> serve --isolated`` backend lists only <name> (#76932, #91330).

Listing UX, not a boundary (SECURITY.md §2.2): the profile list, ``/api/status`` names, the
unified sidebar fan-out and the gateway ``profiles.list`` roster stop advertising sibling
profiles, so Desktop's profile rail, sessions sidebar and SOUL editor show the launch profile.
Real launch path (subprocess) for the authority; in-process TestClient for the fan-outs.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import psutil
import pytest

from hermes_cli import isolated_scope


@pytest.fixture(autouse=True)
def _reset_scope(monkeypatch):
    from hermes_cli.web_routers import profiles as profiles_routes

    monkeypatch.setattr(profiles_routes, "_SIDEBAR_CACHE_TTL_SECONDS", 0.0)
    yield
    isolated_scope._LAUNCH_PROFILE = None


@pytest.fixture
def homes(monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home

    root = get_hermes_home()
    worker = root / "profiles" / "worker"
    for home in (root, worker):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    return {"default": root, "worker": worker}


def _seed(home, session_id):
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id, source="cli")
        db.append_message(session_id=session_id, role="user", content="hi")
    finally:
        db.close()


def _seed_cron(home, job_id):
    (home / "cron").mkdir(exist_ok=True)
    job = {"id": job_id, "name": job_id, "prompt": "hi", "enabled": True,
           "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"}}
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}), encoding="utf-8")


def _cron_ids(client):
    return {job["id"] for job in client.get("/api/cron/jobs").json()}


def _launch_in(monkeypatch, home, *, isolated=True, desktop_owned=False):
    monkeypatch.setenv("HERMES_HOME", str(home))
    return isolated_scope.record_launch_scope(isolated=isolated, desktop_owned=desktop_owned)


@pytest.fixture
def client(homes):
    from starlette.testclient import TestClient

    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _sidebar_ids(client):
    payload = client.get("/api/profiles/sessions/sidebar", params={"recents_profile": "all"}).json()
    return {row["id"] for row in payload["recents"]["sessions"]}


def test_isolated_named_profile_lists_only_itself(client, homes, monkeypatch):
    _seed(homes["default"], "default-chat")
    _seed(homes["worker"], "worker-chat")
    _seed_cron(homes["default"], "default-job")
    _seed_cron(homes["worker"], "worker-job")
    assert _launch_in(monkeypatch, homes["worker"]) == "worker"

    assert [p["name"] for p in client.get("/api/profiles").json()["profiles"]] == ["worker"]
    status = client.get("/api/status").json()
    assert status["profiles"] == ["worker"]
    assert status["isolated_profile"] == "worker"
    assert _sidebar_ids(client) == {"worker-chat"}
    assert _cron_ids(client) == {"worker-job"}

    import tui_gateway.server as srv
    from tui_gateway.methods_complete import _profile_mention_items

    rows = srv._methods["profiles.list"](1, {"include_sessions": False})["result"]["profiles"]
    assert [row["name"] for row in rows] == ["worker"]
    assert [item["text"] for item in _profile_mention_items("")] == ["@worker"]


@pytest.mark.parametrize("launch", [
    {"isolated": False},                        # the machine dashboard
    {"isolated": True, "desktop_owned": True},  # Desktop's SSH serve: one backend, every profile
])
def test_unscoped_backends_still_list_every_profile(client, homes, monkeypatch, launch):
    _seed(homes["default"], "default-chat")
    _seed(homes["worker"], "worker-chat")
    _seed_cron(homes["default"], "default-job")
    _seed_cron(homes["worker"], "worker-job")
    assert _launch_in(monkeypatch, homes["worker"], **launch) is None

    assert {p["name"] for p in client.get("/api/profiles").json()["profiles"]} == {"default", "worker"}
    assert client.get("/api/status").json()["isolated_profile"] is None
    assert _sidebar_ids(client) == {"default-chat", "worker-chat"}
    assert _cron_ids(client) == {"default-job", "worker-job"}
    from tui_gateway.methods_complete import _profile_mention_items
    assert {item["text"] for item in _profile_mention_items("")} == {"@default", "@worker", "@hermes"}


def test_isolated_root_home_stays_unscoped(homes, monkeypatch):
    # `serve --isolated` from the root home is the default profile's dedicated server; it owns
    # the profiles/ tree, so there is no sibling to hide from it.
    assert _launch_in(monkeypatch, homes["default"]) is None


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_profiles(port, token, proc, log, timeout=60):
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/profiles",
                                     headers={"X-Hermes-Session-Token": token})
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.loads(response.read())
        except OSError:
            time.sleep(0.2)
    pytest.fail(f"serve never answered (exit={proc.poll()}): {log.read_text(errors='replace')}")


def test_real_profile_flag_isolated_serve_lists_only_its_profile(tmp_path):
    root = tmp_path / "hermes"
    for home in (root, root / "profiles" / "worker"):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(root), "HERMES_GATEWAY_LOCK_DIR": str(tmp_path / "locks"),
           "HERMES_DASHBOARD_SESSION_TOKEN": "isolated-scope-test-token"}
    for key in ("HERMES_DESKTOP", "HERMES_PARENT_PID", "HERMES_PARENT_START_MARKER", "HERMES_SPAWN"):
        env.pop(key, None)
    port = _free_port()
    log = tmp_path / "serve.log"
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "-p", "worker", "serve", "--isolated",
             "--host", "127.0.0.1", "--port", str(port)],
            env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        body = _get_profiles(port, "isolated-scope-test-token", proc, log)
    finally:
        if proc.poll() is None:
            family = [*psutil.Process(proc.pid).children(recursive=True), psutil.Process(proc.pid)]
            for p in family:
                p.terminate()
            _, alive = psutil.wait_procs(family, timeout=5)
            for p in alive:
                p.kill()
        proc.wait(timeout=10)
    assert [p["name"] for p in body["profiles"]] == ["worker"]
