"""While Web Access starts, ``/api/webapp-access/status`` says which step the Start is on.

Start builds the Webapp bundle (minutes on a first build), registers the sign-in callback with the
Portal, then writes the config and restarts the gateway, all in one request. The pane polls the
status meanwhile: ``start_progress`` names the step and when the Start began, so the pane can say
what is happening, and a pane opened during the build shows that Start running instead of
offering Start again. It belongs to the gateway being started, so a profile with a gateway of its
own never shows another gateway's Start.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    from tests.hermes_cli.native_gateway_client import native_desktop_client

    return native_desktop_client(monkeypatch)


@pytest.fixture()
def restarts(monkeypatch):
    import hermes_cli.web_server_gateway as gw

    calls: list[str] = []

    def _restart(profile, *, what, label):
        calls.append(what)
        return {"restart_started": True}

    monkeypatch.setattr(gw, "_restart_gateway_after", _restart)
    return calls


@pytest.fixture()
def nous(monkeypatch):
    from hermes_cli import webapp_access

    monkeypatch.setattr(webapp_access, "nous_signed_in", lambda: True)
    monkeypatch.setattr(webapp_access, "_register_nous_sign_in", lambda _callback: None)
    monkeypatch.setattr(webapp_access, "lan_address", lambda: "192.168.1.50")


def _status_from_another_window(client, profile: str | None = None) -> dict:
    """``/status`` as a second pane asks it while the Start runs, on a client of its own."""
    answers: list[dict] = []

    def ask() -> None:
        other = TestClient(client.app, raise_server_exceptions=False)
        response = other.get("/api/webapp-access/status", params={"profile": profile} if profile else None)
        assert response.status_code == 200, response.text
        answers.append(response.json())

    request = threading.Thread(target=ask, daemon=True)
    request.start()
    request.join(timeout=20)
    assert len(answers) == 1, "status did not answer while Start ran"
    return answers[0]


def test_status_names_each_step_of_a_start_in_flight_and_none_after(client, restarts, nous, monkeypatch):
    import hermes_cli.web_server_gateway as gw
    from hermes_cli import webapp, webapp_access

    seen: dict[str, dict] = {}

    def build(project_root, **_options):
        seen["build"] = _status_from_another_window(client)["start_progress"]
        return webapp.webapp_dist_dir(project_root)

    def register(_callback):
        seen["register"] = _status_from_another_window(client)["start_progress"]

    def restart(_profile, *, what, label):
        seen["restart"] = _status_from_another_window(client)["start_progress"]
        return {"restart_started": True}

    monkeypatch.setattr(webapp, "prepare_webapp_renderer", build)
    monkeypatch.setattr(webapp_access, "_register_nous_sign_in", register)
    monkeypatch.setattr(gw, "_restart_gateway_after", restart)
    before = time.time()

    response = client.post("/api/webapp-access/start", json={"mode": "lan"})

    assert response.status_code == 200, response.text
    assert [seen[name]["step"] for name in ("build", "register", "restart")] == [
        "building", "registering", "restarting"]
    # One Start, one origin: the pane's elapsed time does not restart at each step.
    assert len({seen[name]["started_at"] for name in seen}) == 1
    assert before <= seen["build"]["started_at"] <= time.time()
    assert client.get("/api/webapp-access/status").json()["start_progress"] is None


def test_a_failed_build_leaves_no_start_in_flight(client, restarts, nous, monkeypatch):
    from hermes_cli import webapp

    def fail(_project_root, **_options):
        raise webapp.WebappBuildError("Browser-hosted Desktop build failed: npm exited 1")

    monkeypatch.setattr(webapp, "prepare_webapp_renderer", fail)

    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 400
    assert client.get("/api/webapp-access/status").json()["start_progress"] is None
    assert restarts == []


def test_a_profile_with_its_own_gateway_never_shows_another_gateways_start(restarts, nous, monkeypatch):
    """``solo`` runs a gateway of its own and ``served`` is served by the default one: while the
    default gateway's Start builds, ``served`` shows it and ``solo`` shows none."""
    from hermes_cli import webapp
    from hermes_constants import get_hermes_home
    from tests.hermes_cli.native_gateway_client import native_desktop_client

    profiles = {"solo": get_hermes_home() / "profiles" / "solo", "served": get_hermes_home() / "profiles" / "served"}
    for name, config in (("solo", "gateway:\n  standalone: true\n"), ("served", "")):
        profiles[name].mkdir(parents=True)
        (profiles[name] / "config.yaml").write_text(config, encoding="utf-8")
    client = native_desktop_client(monkeypatch, profiles)
    seen: dict[str, dict | None] = {}

    def build(project_root, **_options):
        seen["default"] = _status_from_another_window(client)["start_progress"]
        seen["served"] = _status_from_another_window(client, "served")["start_progress"]
        seen["solo"] = _status_from_another_window(client, "solo")["start_progress"]
        return webapp.webapp_dist_dir(project_root)

    monkeypatch.setattr(webapp, "prepare_webapp_renderer", build)

    assert client.post("/api/webapp-access/start", json={"mode": "lan"}).status_code == 200
    assert (seen["default"] or {}).get("step") == "building"
    assert seen["served"] == seen["default"]
    assert seen["solo"] is None


def test_a_status_read_during_the_last_step_still_reports_that_start(client, nous, monkeypatch):
    """A second window's status read that starts while the Start restarts the gateway, and whose
    control-socket round trip outlasts the Start, reports the Start it began during: the pane shows
    it running and polls again instead of showing Stopped with Start enabled until its next refetch."""
    import hermes_cli.web_server_gateway as gw
    from hermes_cli import webapp, webapp_access

    in_restart, release, start_done = threading.Event(), threading.Event(), threading.Event()
    armed = threading.Event()
    real_status = webapp_access.access_status

    def restart(_profile, *, what, label):
        in_restart.set()
        assert release.wait(20)
        return {"restart_started": True}

    def slow_status():
        # Only the second window's read: the Start finishes while it waits on the gateway.
        if armed.is_set():
            armed.clear()
            release.set()
            assert start_done.wait(20)
        return real_status()

    monkeypatch.setattr(webapp, "prepare_webapp_renderer", lambda project_root, **_options: webapp.webapp_dist_dir(project_root))
    monkeypatch.setattr(gw, "_restart_gateway_after", restart)
    monkeypatch.setattr(webapp_access, "access_status", slow_status)

    def run_start() -> None:
        response = client.post("/api/webapp-access/start", json={"mode": "lan"})
        assert response.status_code == 200, response.text
        start_done.set()

    start = threading.Thread(target=run_start, daemon=True)
    start.start()
    assert in_restart.wait(20)
    armed.set()

    progress = _status_from_another_window(client)["start_progress"]

    start.join(timeout=20)
    assert start_done.is_set()
    assert (progress or {}).get("step") == "restarting"
