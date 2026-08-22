"""Profile-scoped persistence tests for webhook routes."""

import json
import threading
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.platforms.webhook_models import WebhookRouteConfig
from gateway.platforms.webhook_store import WebhookRouteStore


def route(name, secret=None):
    return WebhookRouteConfig(name=name, secret_ref=secret or name)


def test_profile_path_isolation(tmp_path):
    first = WebhookRouteStore(tmp_path, profile="alpha")
    second = WebhookRouteStore(tmp_path, profile="beta")
    first.save({"alpha-route": route("alpha-route")})
    second.save({"beta-route": route("beta-route")})
    assert first.path == tmp_path / "profiles" / "alpha" / "webhook_subscriptions.json"
    assert second.path == tmp_path / "profiles" / "beta" / "webhook_subscriptions.json"
    assert list(first.load()) == ["alpha-route"]
    assert list(second.load()) == ["beta-route"]


def test_save_is_atomic_and_owner_only(tmp_path):
    store = WebhookRouteStore(tmp_path, profile="default")
    store.save({"route": route("route")})
    assert store.path.exists()
    assert not list(store.path.parent.glob("*.tmp"))
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600
    assert json.loads(store.path.read_text())["route"]["secret_ref"] == "route"


def test_corrupt_file_is_quarantined_without_data_loss(tmp_path):
    store = WebhookRouteStore(tmp_path, profile="default")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not-json", encoding="utf-8")
    assert store.load() == {}
    quarantined = list(store.path.parent.glob("webhook_subscriptions.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not-json"
    assert store.path.exists() is False


def test_routes_are_sorted_by_name_on_load_and_save(tmp_path):
    store = WebhookRouteStore(tmp_path, profile="default")
    store.save({name: route(name) for name in ["zulu", "alpha", "middle"]})
    assert list(store.load()) == ["alpha", "middle", "zulu"]
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert list(raw) == ["alpha", "middle", "zulu"]


def test_update_serializes_concurrent_read_modify_writes(tmp_path):
    store = WebhookRouteStore(tmp_path, profile="default")

    def add(index):
        store.update(lambda routes: routes | {f"route-{index:02d}": route(f"route-{index:02d}")})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(24)))
    routes = store.load()
    assert len(routes) == 24
    assert list(routes) == sorted(routes)
    assert all(item.name == name for name, item in routes.items())


def test_legacy_files_are_read_and_rewritten_canonically(tmp_path):
    store = WebhookRouteStore(tmp_path, profile="default")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps({"legacy": {"secret": "s", "deliver": "log", "deliver_extra": {}}}),
        encoding="utf-8",
    )
    routes = store.load()
    assert routes["legacy"].secret_ref == "s"
    store.save(routes)
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert "secret_ref" in data["legacy"]
    assert "secret" not in data["legacy"]
