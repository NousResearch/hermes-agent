"""Dashboard agent-plugin mutations honor the selected management profile."""

import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    return TestClient(
        app,
        raise_server_exceptions=False,
        headers={_SESSION_HEADER_NAME: _SESSION_TOKEN},
    )


def test_install_route_writes_plugin_to_selected_profile_home(client, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_cli import plugins_cmd
    from hermes_cli.profiles import get_profile_dir
    from agent.secret_scope import set_multiplex_active

    work_home = get_profile_dir("work")
    work_home.mkdir(parents=True, exist_ok=True)
    (work_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    default_home = get_hermes_home()
    observed = {}

    def install(*_args, **_kwargs):
        home = get_hermes_home()
        observed["home"] = home
        plugin_dir = home / "plugins" / "profile-scoped-plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "installed.txt").write_text("installed")
        return {"ok": True, "after_install_path": str(plugin_dir)}

    from hermes_cli.web_routers import dashboard_ui

    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", install)
    monkeypatch.setattr(dashboard_ui, "_get_dashboard_plugins", lambda **_kwargs: [])
    set_multiplex_active(True)
    try:
        response = client.post(
            "/api/dashboard/agent-plugins/install?profile=work",
            json={"identifier": "owner/repo"},
        )
    finally:
        set_multiplex_active(False)

    assert response.status_code == 200, response.text
    assert observed["home"] == work_home
    assert (work_home / "plugins" / "profile-scoped-plugin" / "installed.txt").is_file()
    assert not (default_home / "plugins" / "profile-scoped-plugin").exists()
