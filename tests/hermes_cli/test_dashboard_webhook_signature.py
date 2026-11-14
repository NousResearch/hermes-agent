"""POST /api/webhooks signature pin: new dashboard routes are pinned, an explicit pick is stored,
an unknown scheme is refused, and an edit that omits the field keeps the stored pin (#84265)."""

import pytest


@pytest.fixture
def client(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.config import load_config, save_config
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    cfg = load_config()
    cfg.setdefault("platforms", {})["webhook"] = {"enabled": True, "extra": {"host": "127.0.0.1", "port": 8644}}
    save_config(cfg)
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _stored(name):
    from hermes_cli.webhook import _load_subscriptions

    return _load_subscriptions()[name]


def test_new_route_defaults_to_generic(client):
    r = client.post("/api/webhooks", json={"name": "deploy", "deliver": "log"})
    assert r.status_code == 200
    assert _stored("deploy")["signature"] == "generic"
    assert r.json()["signature"] == "generic"


def test_explicit_signature_is_stored(client):
    assert client.post("/api/webhooks", json={"name": "gh", "signature": "github"}).status_code == 200
    assert _stored("gh")["signature"] == "github"
    assert client.get("/api/webhooks").json()["subscriptions"][0]["signature"] == "github"


def test_unknown_signature_is_rejected(client):
    r = client.post("/api/webhooks", json={"name": "bad", "signature": "v1"})
    assert r.status_code == 422
    from hermes_cli.webhook import _load_subscriptions

    assert "bad" not in _load_subscriptions()


def test_edit_without_signature_keeps_pin_and_explicit_edit_changes_it(client):
    assert client.post("/api/webhooks", json={"name": "lin", "signature": "linear"}).status_code == 200
    assert client.post("/api/webhooks", json={"name": "lin", "prompt": "new"}).status_code == 200
    assert _stored("lin")["signature"] == "linear"
    assert client.post("/api/webhooks", json={"name": "lin", "signature": "gitlab"}).status_code == 200
    assert _stored("lin")["signature"] == "gitlab"


def test_edit_of_unpinned_route_without_signature_stays_unpinned(client):
    from hermes_cli.webhook import _mutate_subscriptions

    _mutate_subscriptions(lambda subs: subs.update({"legacy": {"secret": "s", "deliver": "log"}}))
    assert client.post("/api/webhooks", json={"name": "legacy", "deliver": "log"}).status_code == 200
    assert "signature" not in _stored("legacy")
