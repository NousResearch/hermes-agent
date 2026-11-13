from fastapi import FastAPI
from starlette.testclient import TestClient

from hermes_cli import web_server


def test_spa_bootstrap_includes_dashboard_initial_profile(tmp_path, monkeypatch):
    dist = tmp_path / "web_dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<html><head></head><body>Dashboard</body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "WEB_DIST", dist)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)

    app = FastAPI()
    app.state.initial_profile = "worker_x"
    web_server.mount_spa(app)

    response = TestClient(app).get("/chat?resume=session-1")

    assert response.status_code == 200
    assert 'window.__HERMES_INITIAL_PROFILE__="worker_x";' in response.text


def test_spa_bootstrap_escapes_initial_profile_for_script_context(
    tmp_path, monkeypatch
):
    dist = tmp_path / "web_dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<html><head></head><body>Dashboard</body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "WEB_DIST", dist)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)

    app = FastAPI()
    app.state.initial_profile = "bad</script><script>alert(1)</script>"
    web_server.mount_spa(app)

    response = TestClient(app).get("/chat")

    assert response.status_code == 200
    assert "bad<\\/script><script>alert(1)<\\/script>" in response.text
    assert "bad</script><script>alert(1)</script>" not in response.text
