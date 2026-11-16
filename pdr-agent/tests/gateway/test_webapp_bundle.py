"""A Webapp whose bundle is not built is reported unavailable, with how to build it.

The gateway serves the Webapp from its built bundle (``apps/desktop/dist-webapp``). Until that bundle
has its ``index.html`` the gateway reports the Webapp unavailable, says to run ``hermes webapp`` (which
builds it), mints no launch link and answers ``/`` with that instruction rather than the dashboard's
build command. The listener picks the bundle up as soon as it is built, without a restart. A bundle
``hermes update`` could not rebuild is reported the same way until ``hermes webapp`` rebuilds it. The
live cases run a real gateway listener in a child process (``fixtures/webapp_fallback_peer.py``).
"""
import pytest

from gateway import webapp_listener
from hermes_cli import webapp_launcher as launcher
from tests.gateway.fixtures.webapp_fallback_peer import free_port, run


@pytest.fixture(scope="module")
def unbuilt(tmp_path_factory):
    return run(tmp_path_factory.mktemp("unbuilt"), {"webapp": {"enabled": True, "port": free_port()},
                                                    "built": False})


def test_an_unbuilt_webapp_is_unavailable_and_says_how_to_build_it(unbuilt):
    assert unbuilt["start_error"] is None, unbuilt.get("start_traceback")
    status = unbuilt["status"]
    assert (status["state"], status["url"]) == ("unavailable", None)
    assert "hermes webapp" in status["reason"] and "index.html" in status["reason"]


def test_an_unbuilt_webapp_mints_no_launch_link(unbuilt):
    assert "hermes webapp" in unbuilt["launch"]["refused"]


def test_an_unbuilt_webapp_page_names_the_webapps_build(unbuilt):
    assert unbuilt["index"]["status"] == 404
    assert "hermes webapp" in unbuilt["index"]["body"]
    assert "npm run build" not in unbuilt["index"]["body"]


def test_the_webapp_is_served_once_built_without_a_restart(unbuilt):
    assert unbuilt["status_once_built"]["state"] == "serving"
    assert unbuilt["index_once_built"] == {"status": 200, "is_webapp": True}


@pytest.fixture(scope="module")
def update_failed(tmp_path_factory):
    return run(tmp_path_factory.mktemp("update-failed"), {"webapp": {"enabled": True, "port": free_port()},
                                                          "update_fails": True})


def test_a_bundle_hermes_update_could_not_rebuild_is_unavailable_and_says_how_to_rebuild_it(update_failed):
    """``hermes update`` keeps the previous bundle when it cannot rebuild it, built for the code
    before the update. The gateway then reports the Webapp unavailable with what to run (``hermes
    webapp``, which rebuilds it) rather than serving that bundle as current: no launch link, and
    ``/`` answers with the same instruction instead of the old page."""
    assert update_failed["start_error"] is None, update_failed.get("start_traceback")
    status = update_failed["status"]
    assert (status["state"], status["url"]) == ("unavailable", None)
    assert status["reason"].startswith(webapp_listener.NOT_BUILT) and "hermes webapp" in status["reason"]
    assert "hermes webapp" in update_failed["launch"]["refused"]
    assert update_failed["index"]["status"] == 503
    assert "hermes webapp" in update_failed["index"]["body"]
    assert "WEBAPP_BUNDLE" not in update_failed["index"]["body"]


def test_hermes_webapp_rebuilds_a_bundle_the_update_left_and_it_is_served_without_a_restart(update_failed):
    assert update_failed["status_once_built"]["state"] == "serving"
    assert update_failed["index_once_built"] == {"status": 200, "is_webapp": True}


def test_the_launcher_reports_an_unbuilt_webapp_without_asking_for_a_restart(monkeypatch, tmp_path, capsys):
    # Building it is enough: the running gateway serves the bundle once it exists.
    reason = webapp_listener.bundle_missing_reason(tmp_path / "dist-webapp")
    answer = {"ok": True, "result": {"state": "unavailable", "url": None, "reason": reason}}
    monkeypatch.setattr(launcher, "_control", lambda home, verb, params=None: answer)
    assert launcher._status(tmp_path) == 0
    out = capsys.readouterr().out
    assert reason in out
    assert launcher.RESTART_COMMAND not in out


def test_a_built_webapp_has_no_missing_bundle_reason(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    assert webapp_listener.bundle_missing_reason(tmp_path) is None
