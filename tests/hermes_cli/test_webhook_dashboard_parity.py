"""Dashboard Webhooks page parity with the webhook listener (#149804).

The page and the gateway must answer the same two questions about the platform: is it ON, and which
routes exist. The gateway enables the platform from config.yaml's ``platforms.webhook.enabled`` OR a
truthy ``WEBHOOK_ENABLED`` (``gateway.config_env._webhook``), and serves the static
``platforms.webhook.extra.routes`` over same-named dynamic subscriptions (``{**dynamic, **static}``).
The dashboard read only the config key and only the subscription store, so a listener that was
running and serving config-defined routes reported "Webhook receiver disabled" / "Subscriptions (0)".
"""

import json

import pytest


def _client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    # Keep the state DB under the isolated HERMES_HOME for any handler that touches it.
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client


STATIC_ROUTE = {
    "description": "static route from config.yaml",
    "events": ["push"],
    "secret": "static-secret",
    "prompt": "summarize the push",
    "deliver": "log",
}


def _write_webhook_config(*, enabled=None, routes=None):
    """Write config.yaml as JSON (a YAML subset) so a nested route dict cannot be mis-indented."""
    webhook = {"extra": {"port": 8644}}
    if enabled is not None:
        webhook["enabled"] = enabled
    if routes is not None:
        webhook["extra"]["routes"] = routes
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(
        json.dumps({"platforms": {"webhook": webhook}}), encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _no_webhook_env(monkeypatch):
    for name in ("WEBHOOK_ENABLED", "WEBHOOK_PORT", "WEBHOOK_SECRET"):
        monkeypatch.delenv(name, raising=False)


def test_enabled_follows_the_env_flag_the_gateway_honours(monkeypatch):
    """``WEBHOOK_ENABLED=true`` alone starts the listener, so the page must not say disabled."""
    _write_webhook_config()
    monkeypatch.setenv("WEBHOOK_ENABLED", "true")

    from hermes_cli.webhook import _is_webhook_enabled

    assert _is_webhook_enabled() is True
    assert _client().get("/api/webhooks").json()["enabled"] is True


def test_config_enabled_still_enables():
    _write_webhook_config(enabled=True)

    from hermes_cli.webhook import _is_webhook_enabled

    assert _is_webhook_enabled() is True
    assert _client().get("/api/webhooks").json()["enabled"] is True


def test_disabled_without_either_switch(monkeypatch):
    _write_webhook_config(enabled=False)
    monkeypatch.setenv("WEBHOOK_ENABLED", "0")

    from hermes_cli.webhook import _is_webhook_enabled

    assert _is_webhook_enabled() is False
    assert _client().get("/api/webhooks").json()["enabled"] is False


def test_static_config_routes_are_listed_read_only():
    """A route defined in config.yaml is served by the listener and is not editable from the page."""
    _write_webhook_config(enabled=True, routes={"github": STATIC_ROUTE})

    subs = _client().get("/api/webhooks").json()["subscriptions"]

    assert [row["name"] for row in subs] == ["github"]
    row = subs[0]
    assert row["read_only"] is True
    assert row["source"] == "config"
    assert row["description"] == "static route from config.yaml"
    assert row["events"] == ["push"]
    assert row["deliver"] == "log"
    assert row["secret_set"] is True
    assert row["url"].endswith("/webhooks/github")


def test_a_config_route_shadows_the_subscription_of_the_same_name():
    """The adapter merges the dynamic store UNDER the static routes, so the page lists one route."""
    _write_webhook_config(enabled=True, routes={"github": STATIC_ROUTE})
    import hermes_cli.webhook as wh

    wh._mutate_subscriptions(
        lambda subs: subs.update({"github": {"secret": "dynamic", "deliver": "telegram"}})
    )

    subs = _client().get("/api/webhooks").json()["subscriptions"]

    assert [row["name"] for row in subs] == ["github"]
    assert subs[0]["read_only"] is True
    # The config.yaml definition is what the listener serves, not the shadowed subscription.
    assert subs[0]["deliver"] == "log"


def test_store_subscriptions_stay_editable_beside_config_routes():
    _write_webhook_config(enabled=True, routes={"github": STATIC_ROUTE})
    import hermes_cli.webhook as wh

    wh._mutate_subscriptions(
        lambda subs: subs.update(
            {"todoist": {"secret": "dynamic", "deliver": "log", "events": ["item:added"]}}
        )
    )

    subs = _client().get("/api/webhooks").json()["subscriptions"]

    assert [row["name"] for row in subs] == ["github", "todoist"]
    assert subs[1]["read_only"] is False
    assert subs[1]["source"] == "subscriptions"
