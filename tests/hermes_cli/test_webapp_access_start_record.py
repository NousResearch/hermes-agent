"""A Start's record covers the whole Start, not just its request.

``start_progress`` on ``/api/webapp-access/status`` names the step a Start is on. It is reported
from the moment the Start is asked for, ``preparing`` while another Web Access change holds the
backend's one change lock, until the restart it asked for has put a gateway serving the new config
in place, that restart failed, or the pane's own wait for it has run out. A pane opened at any point
in between shows the Start running and keeps its controls disabled instead of offering Start or
Restart gateway. Each Start keeps its own record, so one Start ending cannot hide another.
Review of #148408 (teknium1, yoniebans).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tests.hermes_cli.native_gateway_client import native_desktop_client
from tests.waiting import wait_until

RESTART_PID = 4242


@pytest.fixture(autouse=True)
def starts(monkeypatch):
    """This backend's Start records, fresh for each test: a followed restart outlives its request."""
    from hermes_cli.web_routers import webapp_access as routes

    records: dict = {}
    monkeypatch.setattr(routes, "_STARTS", records)
    return records


@pytest.fixture()
def client(monkeypatch):
    return native_desktop_client(monkeypatch)


@pytest.fixture()
def nous(monkeypatch):
    from hermes_cli import webapp_access

    monkeypatch.setattr(webapp_access, "nous_signed_in", lambda: True)
    monkeypatch.setattr(webapp_access, "_register_nous_sign_in", lambda _callback: None)
    monkeypatch.setattr(webapp_access, "lan_address", lambda: "192.168.1.50")


@pytest.fixture()
def gateway(monkeypatch):
    """The gateway as its control socket answers: the old one, serving no Webapp, until a test puts
    its replacement in place. A restart is recorded and answers with its child's pid."""
    import hermes_cli.web_server_gateway as gw
    from hermes_cli import webapp_access

    state = {"now": {"state": "off", "url": None, "reason": None}, "restarts": [], "failure": None,
             "restart": {"restart_started": True, "restart_action": "gateway-restart", "restart_pid": RESTART_PID}}

    def restart(_profile, *, what, label):
        state["restarts"].append(what)
        return dict(state["restart"])

    monkeypatch.setattr(webapp_access, "_gateway_state", lambda _home: dict(state["now"]))
    monkeypatch.setattr(gw, "_restart_gateway_after", restart)
    monkeypatch.setattr(gw, "gateway_restart_failure", lambda _profile: state["failure"])
    return state


@pytest.fixture()
def instant_build(monkeypatch):
    from hermes_cli import webapp

    monkeypatch.setattr(webapp, "prepare_webapp_renderer", lambda project_root, **_options: webapp.webapp_dist_dir(project_root))


class _Builds:
    """``prepare_webapp_renderer`` that holds each build until the test allows one more to finish.
    Teardown lets every held build go, so a failed test leaves no Start holding the change lock."""

    def __init__(self, monkeypatch):
        from hermes_cli import webapp

        self._started = threading.Semaphore(0)
        self._allowed = threading.Semaphore(0)
        dist = webapp.webapp_dist_dir

        def build(project_root, **_options):
            self._started.release()
            assert self._allowed.acquire(timeout=20), "the build was never allowed to finish"
            return dist(project_root)

        monkeypatch.setattr(webapp, "prepare_webapp_renderer", build)

    def wait_started(self) -> None:
        assert self._started.acquire(timeout=20), "the build never started"

    def finish_one(self) -> None:
        self._allowed.release()


@pytest.fixture()
def builds(monkeypatch):
    held = _Builds(monkeypatch)
    yield held
    for _ in range(4):
        held.finish_one()


def _status(client, profile: str | None = None) -> dict:
    other = TestClient(client.app, raise_server_exceptions=False)
    response = other.get("/api/webapp-access/status", params={"profile": profile} if profile else None)
    assert response.status_code == 200, response.text
    return response.json()


def _progress(client, profile: str | None = None) -> Any:
    return _status(client, profile)["start_progress"]


def _start_in_background(client, profile: str | None = None) -> threading.Thread:
    def run() -> None:
        other = TestClient(client.app, raise_server_exceptions=False)
        response = other.post("/api/webapp-access/start", json={"mode": "lan"},
                              params={"profile": profile} if profile else None)
        assert response.status_code == 200, response.text

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _joined(*threads: threading.Thread) -> None:
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive(), "a Start never answered"


def _reports(client, step: str, profile: str | None = None) -> Any:
    return wait_until(lambda: (p := _progress(client, profile)) and p["step"] == step and p,
                      what=f"a Start reported {step!r}")


def test_a_start_waiting_for_another_gateways_start_is_reported_from_when_it_was_asked(nous, gateway, builds, monkeypatch):
    from hermes_constants import get_hermes_home

    solo = get_hermes_home() / "profiles" / "solo"
    solo.mkdir(parents=True)
    (solo / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
    client = native_desktop_client(monkeypatch, {"solo": solo})

    first = _start_in_background(client)
    builds.wait_started()
    asked = time.time()
    second = _start_in_background(client, "solo")

    waiting = _reports(client, "preparing", "solo")
    assert asked <= waiting["started_at"] <= time.time()
    assert _progress(client)["step"] == "building"

    # The default gateway's Start lets go of the change lock; the same record moves on, clock and all.
    builds.finish_one()
    builds.wait_started()
    assert _progress(client, "solo") == {"step": "building", "started_at": waiting["started_at"]}

    builds.finish_one()
    _joined(first, second)


def test_one_start_ending_leaves_the_start_queued_behind_it_on_the_same_gateway(client, nous, gateway, builds, starts):
    # Refused restarts: each Start's record ends with its own request.
    gateway["restart"] = {"restart_started": False, "restart_error": "a chat is running", "restart_busy": True}

    first = _start_in_background(client)
    builds.wait_started()
    first_started_at = _progress(client)["started_at"]
    second = _start_in_background(client)
    # Black-box status shows the running Start, not the one waiting behind it; count the records.
    wait_until(lambda: sum(map(len, starts.values())) == 2, what="the second Start to queue")
    released = time.time()

    builds.finish_one()
    builds.wait_started()

    queued = _progress(client)
    assert queued["step"] == "building"
    assert first_started_at < queued["started_at"] < released

    builds.finish_one()
    _joined(first, second)
    assert _progress(client) is None


def test_a_start_is_reported_restarting_until_the_restarted_gateway_serves_it(client, nous, gateway, instant_build):
    response = client.post("/api/webapp-access/start", json={"mode": "lan"})
    assert response.status_code == 200, response.text
    answer = response.json()
    port = answer["port"]
    # The pane keeps Start's answer as the status it shows: where the backend answering is the
    # gateway being restarted, no status read succeeds until the new one is up, so the answer
    # itself says the Start is restarting.
    assert answer["start_progress"]["step"] == "restarting"

    # The request answered; the old gateway still serves no Webapp.
    assert _progress(client) == answer["start_progress"]

    gateway["now"] = {"state": "serving", "url": f"http://192.168.1.50:{port}", "reason": None}
    assert _progress(client) is None


@pytest.mark.parametrize("ending", ["restart failed", "followed as long as the pane waits", "stopped"])
def test_a_followed_restart_is_no_longer_reported_once_it_cannot_serve_the_start(
        client, nous, gateway, instant_build, monkeypatch, ending):
    from hermes_cli.web_routers import webapp_access as routes

    if ending == "followed as long as the pane waits":
        monkeypatch.setattr(routes, "_RESTART_FOLLOW_SECONDS", 0.0)
    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 200

    if ending == "restart failed":
        assert _progress(client)["step"] == "restarting"
        gateway["failure"] = {"pid": RESTART_PID, "exit_code": 1, "reason": "port 9187 is in use"}
    elif ending == "stopped":
        assert _progress(client)["step"] == "restarting"
        gateway["restart"] = {**gateway["restart"], "restart_pid": RESTART_PID + 1}
        assert client.post("/api/webapp-access/stop").status_code == 200

    assert _progress(client) is None


def test_a_refused_restart_ends_the_start_with_its_request(client, nous, gateway, instant_build):
    gateway["restart"] = {"restart_started": False, "restart_error": "a chat is running", "restart_busy": True}

    response = client.post("/api/webapp-access/start", json={"mode": "lan"})

    assert response.status_code == 200 and response.json()["restart_started"] is False
    assert response.json()["start_progress"] is None
    assert _progress(client) is None
