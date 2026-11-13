"""A SPA bundle that cannot be served is explained to a browser as a page, and to an API client as
JSON. The Webapp is internet-facing: a raw JSON body is all a signed-in viewer would see (#151152)."""
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BROWSER = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


def _spa(monkeypatch, dist: Path, *, surface) -> TestClient:
    from hermes_agent.hermes_cli import web_server as server
    from hermes_agent.hermes_cli.web_server_dashboard import mount_spa

    app = FastAPI()
    if surface is not None:
        app.state.ui_surface = surface
    app.state.web_dist = dist
    monkeypatch.setattr(server, "app", app)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(app)
    return TestClient(app)


def _stale_bundle(dist: Path) -> Path:
    from hermes_agent.hermes_cli.webapp import STALE_MARKER

    (dist / "index.html").write_text("<html><head></head><body>old build</body></html>")
    (dist / STALE_MARKER).write_text("")
    return dist


@pytest.mark.parametrize(("surface", "command"), [("webapp", "hermes webapp"),
                                                  (None, "pnpm --dir web run build")],
                         ids=["webapp", "dashboard"])
@pytest.mark.parametrize("path", ["/", "/sessions/abc"])
def test_a_browser_is_shown_a_page_naming_the_build_command(monkeypatch, tmp_path, surface, command, path):
    response = _spa(monkeypatch, tmp_path, surface=surface).get(path, headers=BROWSER)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"].startswith("no-store")
    assert command in response.text


def test_a_browser_is_shown_a_page_for_a_bundle_update_could_not_rebuild(monkeypatch, tmp_path):
    response = _spa(monkeypatch, _stale_bundle(tmp_path), surface="webapp").get("/", headers=BROWSER)

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("text/html")
    assert "old build" not in response.text
    assert "hermes webapp" in response.text
    # The Webapp surface keeps its page policy on this page too.
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_an_api_client_still_reads_the_reason_as_json(monkeypatch, tmp_path):
    missing = _spa(monkeypatch, tmp_path / "missing", surface="webapp").get(
        "/", headers={"accept": "application/json"})
    stale = _spa(monkeypatch, _stale_bundle(tmp_path), surface="webapp").get("/")

    assert missing.status_code == 404
    assert missing.json() == {"error": "The Webapp is not built. Run: hermes webapp"}
    assert stale.status_code == 503
    assert "hermes webapp" in stale.json()["error"]
