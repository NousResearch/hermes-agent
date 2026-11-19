"""``POST /api/profiles/{name}/describe-auto`` must call the aux LLM under the target
profile's secret scope.

The handler ran ``describe_profile`` on a worker thread with no scope installed. Once the
dashboard has served a secondary profile (fail-closed multiplexing on), the auxiliary
client's credential read goes through ``get_secret``, which raises ``UnscopedSecretError``
with no scope — so the Profiles page's "generate description" action failed as
``LLM error: UnscopedSecretError`` for every profile, whatever the configuration. Sibling
handlers in this router already compose ``_config_profile_scope``; describe-auto now does
too (home + secrets, without the process-wide config lock ``_profile_scope`` takes — this
call has a 60 s network ceiling).
"""

import pytest

pytest.importorskip("fastapi")
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent import secret_scope

KEY_ENV = "ACME_RELAY_KEY"


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    """A throwaway HERMES_HOME whose named profile carries its own ``.env`` credential.

    The process env holds a DIFFERENT value for the same variable, so a scoped read is
    distinguishable from the dashboard home's unscoped one.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(KEY_ENV, "dashboard-home-key")
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.config import invalidate_env_cache

    demo = profiles_mod.get_profile_dir("demo")
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "config.yaml").write_text("{}\n", encoding="utf-8")
    (demo / ".env").write_text(f"{KEY_ENV}=profile-key\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    invalidate_env_cache()
    return tmp_path, demo


@pytest.fixture()
def aux(monkeypatch):
    """Stub only the network hop. The credential is read the way the real auxiliary
    client reads it, so an unscoped handler still fails closed here."""
    captured = {}

    def _fake_call_llm(task=None, messages=None, **kw):
        from agent.secret_scope import current_secret_scope_home, get_secret

        captured["task"] = task
        captured["scope_home"] = current_secret_scope_home()
        captured["api_key"] = get_secret(KEY_ENV)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"description": "a demo profile"}'))]
        )

    import agent.auxiliary_client as aux_mod
    monkeypatch.setattr(aux_mod, "call_llm", _fake_call_llm)
    return captured


@pytest.fixture()
def client(homes):
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        yield c


def test_describe_auto_runs_under_the_target_profile_scope(client, homes, aux):
    """Multiplexed dashboard: the generation succeeds and resolves the NAMED profile's key."""
    _root, demo = homes
    secret_scope.set_multiplex_active(True)
    try:
        resp = client.post("/api/profiles/demo/describe-auto", json={"overwrite": True})
    finally:
        secret_scope.set_multiplex_active(False)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The regression reported this as ok=False / "LLM error: UnscopedSecretError".
    assert body["ok"] is True, body
    assert body["description"] == "a demo profile"
    assert body["description_auto"] is True
    assert aux["task"] == "profile_describer"
    # Scoped to the named profile, and reading ITS .env — not the dashboard home's env value.
    assert aux["scope_home"] == str(demo)
    assert aux["api_key"] == "profile-key"


def test_describe_auto_persists_under_the_profile_home(client, homes, aux):
    """The generated text lands in the named profile's own metadata."""
    root, demo = homes
    secret_scope.set_multiplex_active(True)
    try:
        assert client.post(
            "/api/profiles/demo/describe-auto", json={"overwrite": True}
        ).status_code == 200
    finally:
        secret_scope.set_multiplex_active(False)

    from hermes_cli import profiles as profiles_mod

    assert profiles_mod.read_profile_meta(demo).get("description") == "a demo profile"
    assert not (root / "profile.yaml").exists()


def test_describe_auto_for_an_unknown_profile_is_still_a_404(client):
    """Binding the scope must not turn resolution failures into a generation failure."""
    secret_scope.set_multiplex_active(True)
    try:
        resp = client.post("/api/profiles/nope/describe-auto", json={"overwrite": True})
    finally:
        secret_scope.set_multiplex_active(False)

    assert resp.status_code == 404, resp.text
