"""``PUT /api/profiles/{name}/model`` must validate under the target profile's secret scope.

``_write_profile_model`` used to enter only the HERMES_HOME override. Once the dashboard
has served a secondary profile (fail-closed multiplexing on), ``switch_model``'s
``key_env`` probe reads through ``get_secret``, which fails closed without an installed
scope — the pick was rejected with "<provider> is not connected" even though the named
profile's ``.env`` held the key (#114676). ``POST /api/model/set`` already binds
``_config_profile_scope``; the profiles router now composes the same scope (home +
secrets) around both the validate and save spans.
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import secret_scope  # noqa: E402

ACME_YAML = (
    "providers:\n"
    "  acme:\n"
    "    base_url: https://api.acme.test/v1\n"
    "    key_env: ACME_RELAY_KEY\n"
)


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    """A throwaway HERMES_HOME whose named profile carries its own ``.env`` credential.

    The process env holds a DIFFERENT value for the same variable: a scoped read must
    resolve the profile's key, never the dashboard home's (fail-closed isolation).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ACME_RELAY_KEY", "dashboard-home-key")
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.config import invalidate_env_cache

    demo = profiles_mod.get_profile_dir("demo")
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "config.yaml").write_text(ACME_YAML, encoding="utf-8")
    (demo / ".env").write_text("ACME_RELAY_KEY=profile-key\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(ACME_YAML, encoding="utf-8")
    invalidate_env_cache()
    return tmp_path, demo


@pytest.fixture()
def probe(monkeypatch):
    """Capture the credential ``switch_model`` resolved, without network round-trips."""
    captured = {}

    def _fake_runtime(
        requested,
        explicit_api_key=None,
        explicit_base_url=None,
        target_model=None,
        **kw,
    ):
        captured.setdefault("api_key", explicit_api_key)
        return {
            "api_key": explicit_api_key or "",
            "base_url": explicit_base_url,
            "api_mode": "",
        }

    def _fake_validate(
        model, provider, api_key=None, base_url=None, api_mode=None, headers=None, **kw
    ):
        return {"accepted": True, "persist": True, "recognized": True, "message": ""}

    import hermes_cli.model_switch as ms
    import hermes_cli.models_validate as mv

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _fake_runtime
    )
    monkeypatch.setattr(ms, "resolve_alias", lambda *a, **k: None)
    monkeypatch.setattr(mv, "validate_requested_model", _fake_validate)
    return captured


@pytest.fixture()
def client(homes):
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        yield c


def test_model_pick_resolves_key_env_from_profile_scope(client, homes, probe):
    """Multiplexed dashboard: the named profile's ``.env`` authenticates the pick."""
    secret_scope.set_multiplex_active(True)
    try:
        resp = client.put(
            "/api/profiles/demo/model", json={"provider": "acme", "model": "acme/mini"}
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert resp.status_code == 200, resp.text
    # The profile's key, not the dashboard home's value from the process env.
    assert probe["api_key"] == "profile-key"


def test_model_pick_persists_into_profile_config(client, homes, probe):
    """The save span still lands in the named profile's config.yaml."""
    _, demo = homes
    secret_scope.set_multiplex_active(True)
    try:
        resp = client.put(
            "/api/profiles/demo/model", json={"provider": "acme", "model": "acme/mini"}
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert resp.status_code == 200, resp.text
    from hermes_cli.config import load_config
    from hermes_cli.web_server_profiles import _hermes_home_scope

    with _hermes_home_scope(demo):
        model_cfg = load_config().get("model") or {}
    assert model_cfg.get("default") == "acme/mini"
    assert model_cfg.get("provider") == "acme"


def test_model_pick_for_process_home_uses_launch_scope(client, homes, probe):
    """The dashboard's own profile maps to None (current-profile semantics) and still
    validates through the launch scope once multiplexing is active."""
    secret_scope.set_multiplex_active(True)
    try:
        resp = client.put(
            "/api/profiles/default/model",
            json={"provider": "acme", "model": "acme/mini"},
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert resp.status_code == 200, resp.text
    # Launch scope: live process env while single-profile... frozen at activation once
    # multiplexed — the dashboard home's value, resolved through get_secret, not a raise.
    assert probe["api_key"] == "dashboard-home-key"
