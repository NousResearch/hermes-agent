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


def test_dashboard_profile_helpers_keep_captured_launch_root(tmp_path, monkeypatch):
    from hermes_cli import web_server_profiles
    from tui_gateway import launch_profile_policy

    launch_root = tmp_path / "launch-root"
    launch_home = launch_root / "profiles" / "alpha"
    launch_target = launch_root / "profiles" / "beta"
    poison_root = tmp_path / "poison-root"
    poison_home = poison_root / "profiles" / "poison"
    poison_target = poison_root / "profiles" / "beta"
    for home in (launch_home, launch_target, poison_home, poison_target):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    launch_profile_policy.capture_launch_authority()
    monkeypatch.setenv("HERMES_HOME", str(poison_home))

    assert web_server_profiles.serving_profile_name() == "alpha"
    with web_server_profiles._config_profile_scope("beta") as scoped:
        assert scoped == launch_target
