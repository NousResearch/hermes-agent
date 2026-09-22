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
    "custom_providers:\n"
    "  - name: acme\n"
    "    base_url: https://api.acme.test/v1\n"
    "    key_env: ACME_RELAY_KEY\n"
    "    models: [acme/mini]\n"
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
    """Capture the credential the real ``custom_providers`` resolution handed to the network
    validation step (the only stubbed piece); ``key_env`` is read by the production path."""
    captured = {}

    def _fake_validate(model, provider, api_key=None, base_url=None, **kw):
        captured["api_key"] = api_key
        return {"accepted": True, "persist": True, "recognized": True, "message": ""}

    import hermes_cli.models_validate as mv
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
    from hermes_cli.config import load_config
    from hermes_cli.web_server_profiles import _hermes_home_scope
    with _hermes_home_scope(homes[1]):
        assert (load_config().get("model") or {}).get("default") == "acme/mini"


def test_model_pick_for_default_from_named_profile_launch(homes, probe, monkeypatch):
    """Dashboard launched from ``profiles/demo``: targeting ``default`` must scope the ROOT
    (name ``default``), not the root directory's basename, which is not a profile name."""
    root, demo = homes
    monkeypatch.setenv("HERMES_HOME", str(demo))
    (root / ".env").write_text("ACME_RELAY_KEY=root-key\n", encoding="utf-8")
    from hermes_cli.config import invalidate_env_cache, load_config
    from hermes_cli.web_server_profiles import _hermes_home_scope
    invalidate_env_cache()
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        resp = client.put(
            "/api/profiles/default/model", json={"provider": "acme", "model": "acme/mini"}
        )

    assert resp.status_code == 200, resp.text
    assert probe["api_key"] == "root-key"  # the root's .env, not the launch profile's
    with _hermes_home_scope(root):
        assert (load_config().get("model") or {}).get("default") == "acme/mini"
    with _hermes_home_scope(demo):
        assert (load_config().get("model") or {}).get("default") is None


def test_model_set_without_a_profile_pins_to_the_launch_home(client, homes):
    """#118432: an omitted ``profile`` does NOT mean "the default profile" — it means
    whatever home this process launched with. That is why the desktop client must
    always send the concrete profile its "Applies to" chip names. This pins the
    server-side semantics so redefining "omitted → launch home" (e.g. to a
    fail-closed refusal like the destructive routes) stays a deliberate, reviewed
    decision instead of an accident."""
    root, demo = homes
    resp = client.post(
        "/api/model/set",
        json={
            "scope": "main",
            "provider": "acme",
            "model": "acme/mini",
            "confirm_expensive_model": True,
        },
    )

    assert resp.status_code == 200, resp.text
    from hermes_cli.config import load_config
    from hermes_cli.web_server_profiles import _hermes_home_scope

    with _hermes_home_scope(root):
        assert (load_config().get("model") or {}).get("default") == "acme/mini"
    with _hermes_home_scope(demo):
        assert (load_config().get("model") or {}).get("default") is None


def test_model_set_names_target_from_a_backend_launched_as_another_profile(homes, probe, monkeypatch):
    """A→B→A under one host backend (#118431/#118432): a backend launched under profile A
    serving ``POST /api/model/set?profile=B`` writes B's config.yaml and leaves A's (and the
    root's) untouched; the follow-up for A lands on A only. This is the server half the
    Desktop relies on once every Settings request names the profile its page displays."""
    root, demo = homes
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.config import invalidate_env_cache, load_config
    from hermes_cli.web_server_profiles import _hermes_home_scope

    other = profiles_mod.get_profile_dir("other")
    other.mkdir(parents=True, exist_ok=True)
    (other / "config.yaml").write_text(ACME_YAML, encoding="utf-8")
    (other / ".env").write_text("ACME_RELAY_KEY=other-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(demo))  # launched as A = demo
    invalidate_env_cache()
    from hermes_cli import web_server

    body = {"scope": "main", "provider": "acme", "model": "acme/mini", "confirm_expensive_model": True}

    def model_of(home):
        with _hermes_home_scope(home):
            return (load_config().get("model") or {}).get("default")

    with TestClient(web_server.app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        resp = client.post("/api/model/set", params={"profile": "other"}, json=body)
        assert resp.status_code == 200, resp.text
        assert probe["api_key"] == "other-key"  # validated under B's secret scope, not A's
        assert model_of(other) == "acme/mini"
        assert model_of(demo) is None and model_of(root) is None

        resp = client.post("/api/model/set", params={"profile": "demo"}, json={**body, "model": "acme/mini"})
        assert resp.status_code == 200, resp.text
        assert probe["api_key"] == "profile-key"
        assert model_of(demo) == "acme/mini" and model_of(root) is None
