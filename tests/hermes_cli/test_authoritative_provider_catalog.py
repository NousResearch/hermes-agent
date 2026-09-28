"""Provider-owned catalogs keep their order and never resurrect offline-only ids."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest


@pytest.fixture
def catalog(monkeypatch):
    import providers
    from hermes_cli import models
    from providers.base import ProviderProfile

    state = {"rows": ["vendor-b/alpha", "vendor-a/zeta"], "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.headers.get("Authorization"))
            body = json.dumps({"data": [{"id": row} for row in state["rows"]]}).encode()
            self.send_response(state["status"])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    profile = ProviderProfile(
        name="test-authoritative-catalog", base_url=f"http://127.0.0.1:{server.server_port}/v1",
        env_vars=("TEST_CATALOG_KEY",), fallback_models=("offline/retired",),
    )
    providers.list_providers()
    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("test-key", profile.base_url))
    try:
        yield profile, state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("authoritative", [False, True])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_live_catalog_policy_reaches_picker_setup_and_disk_cache(catalog, monkeypatch, authoritative, public, api_key):
    from hermes_cli import model_setup_flows as flows, models

    profile, state = catalog
    profile.authoritative_model_catalog = authoritative
    profile.public_model_catalog = public
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: (api_key, profile.base_url))
    monkeypatch.setattr(flows, "_models_dev_merged", lambda *_: [])
    expected = state["rows"] if authoritative else list(profile.fallback_models) + state["rows"]
    if not api_key and not public:
        expected = list(profile.fallback_models)

    assert models.provider_model_ids(profile.name) == expected
    assert flows._api_key_provider_model_list(
        profile.name, SimpleNamespace(name="Test"), api_key, "", profile.base_url,
    ) == expected
    assert models.cached_provider_model_ids(profile.name, force_refresh=True) == expected
    requests = len(state["requests"])
    assert models.cached_provider_model_ids(profile.name, non_blocking=True) == expected
    assert len(state["requests"]) == requests
    assert state["requests"] == ([f"Bearer {api_key}" if api_key else None] * 3 if api_key or public else [])


@pytest.mark.parametrize("outcome", ["live", "empty", "offline"])
@pytest.mark.parametrize("route", ["canonical", "relay", "external_process"])
@pytest.mark.parametrize("static_source", ["models.dev", "curated"])
def test_authoritative_catalog_survives_static_sources_and_refresh(
    catalog, monkeypatch, outcome, route, static_source,
):
    from hermes_cli import model_setup_flows as flows, models

    profile, state = catalog
    profile.authoritative_model_catalog = True
    profile.public_model_catalog = True
    curated = [f"static/model-{i}" for i in range(8)]
    monkeypatch.setitem(models._PROVIDER_MODELS, profile.name, curated)
    monkeypatch.setattr(flows, "_models_dev_merged", lambda *_: curated if static_source == "models.dev" else [])
    if route == "external_process":
        profile.auth_type = route
    if route == "relay":
        monkeypatch.setattr(models, "_get_model_config_dict", lambda: {
            "provider": profile.name, "base_url": profile.base_url + "/relay",
        })

    warm = list(state["rows"])
    assert models.cached_provider_model_ids(profile.name, force_refresh=True) == warm
    if outcome == "empty":
        state["rows"] = []
        # The GUI's background warmer must also replace a stale non-empty catalog with [].
        from threading import Event
        stored = Event()
        store_entry = models._store_cache_entry

        def record_store(*args, **kwargs):
            store_entry(*args, **kwargs)
            stored.set()

        monkeypatch.setattr(models, "_store_cache_entry", record_store)
        models._spawn_swr_refresh(profile.name)
        assert stored.wait(5), "background catalog refresh did not persist"
        assert models.cached_provider_model_ids(profile.name, non_blocking=True) == []
    elif outcome == "offline":
        state["status"] = 503
    expected = list(profile.fallback_models) if outcome == "offline" else state["rows"]

    assert models.provider_model_ids(profile.name) == expected
    assert flows._api_key_provider_model_list(
        profile.name, SimpleNamespace(name="Test"), "test-key", "", profile.base_url,
    ) == expected
    # Outage keeps the last good disk catalog; a successful empty catalog removes it.
    cached = warm if outcome == "offline" else expected
    assert models.cached_provider_model_ids(profile.name, force_refresh=True) == cached
    assert models.cached_provider_model_ids(profile.name, non_blocking=True) == cached
    from hermes_cli.model_switch_providers import _live_or_curated_ids
    assert _live_or_curated_ids(profile.name, {profile.name: curated}, non_blocking=True) == cached
    if outcome == "empty":
        state["status"] = 503
        now = models.time.time()
        monkeypatch.setattr(models.time, "time", lambda: now + models._PROVIDER_MODELS_CACHE_TTL + 1)
        refreshes = []
        monkeypatch.setattr(models, "_spawn_swr_refresh", refreshes.append)
        assert _live_or_curated_ids(profile.name, {profile.name: curated}, non_blocking=True) == []
        assert refreshes == [profile.name]
        assert _live_or_curated_ids(profile.name, {profile.name: curated}) == []
