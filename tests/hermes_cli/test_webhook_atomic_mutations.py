"""Concurrency regressions for Webhook Revolution Task 6."""

import pytest

import hermes_cli.webhook as webhook_cli
from gateway.platforms.webhook_store import WebhookRouteStore


def _route(secret_ref: str, prompt: str = "") -> dict:
    return {"secret_ref": secret_ref, "prompt": prompt, "deliver": "log"}


def _bind_store(monkeypatch, tmp_path):
    store = WebhookRouteStore(tmp_path, profile="default")
    monkeypatch.setattr(webhook_cli, "_route_store", lambda: store)
    return store


def test_independent_concurrent_additions_are_merged(monkeypatch, tmp_path):
    store = _bind_store(monkeypatch, tmp_path)
    first = webhook_cli._load_subscriptions()
    second = webhook_cli._load_subscriptions()
    first["alpha"] = _route("WEBHOOK_ROUTE_ALPHA")
    second["beta"] = _route("WEBHOOK_ROUTE_BETA")

    webhook_cli._save_subscriptions(first)
    webhook_cli._save_subscriptions(second)

    assert set(store.load()) == {"alpha", "beta"}


def test_divergent_same_route_update_fails_closed(monkeypatch, tmp_path):
    store = _bind_store(monkeypatch, tmp_path)
    store.save({"shared": _route("WEBHOOK_ROUTE_SHARED", "before")})
    first = webhook_cli._load_subscriptions()
    second = webhook_cli._load_subscriptions()
    first["shared"]["prompt"] = "writer-one"
    second["shared"]["prompt"] = "writer-two"

    webhook_cli._save_subscriptions(first)
    with pytest.raises(webhook_cli.ConcurrentWebhookUpdateError, match="shared"):
        webhook_cli._save_subscriptions(second)

    assert webhook_cli._load_subscriptions()["shared"]["prompt"] == "writer-one"


def test_delete_preserves_unrelated_concurrent_addition(monkeypatch, tmp_path):
    store = _bind_store(monkeypatch, tmp_path)
    store.save({"old": _route("WEBHOOK_ROUTE_OLD")})
    deleter = webhook_cli._load_subscriptions()
    adder = webhook_cli._load_subscriptions()
    del deleter["old"]
    adder["new"] = _route("WEBHOOK_ROUTE_NEW")

    webhook_cli._save_subscriptions(adder)
    webhook_cli._save_subscriptions(deleter)

    assert set(store.load()) == {"new"}
