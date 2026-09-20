"""First-time setup must consume the registered provider catalog (#116408)."""
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("live", [True, False])
def test_setup_offers_registered_provider_catalog(monkeypatch, live):
    import providers
    from providers.base import ProviderProfile
    from hermes_cli import auth, config, model_setup_flows as flows, models

    class SetupProfile(ProviderProfile):
        def fetch_models(self, *, api_key=None, base_url=None):
            assert api_key == "synthetic-test-key"
            assert base_url == self.base_url
            return list(self.fallback_models) if live else None

    profile = SetupProfile(
        name="scout-setup-catalog", display_name="Setup catalog",
        auth_type="api_key", env_vars=("SCOUT_SETUP_TEST_KEY",),
        base_url="https://setup.example.invalid/v1",
        fallback_models=("declared-plugin-model",),
    )
    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", dict(auth.PROVIDER_REGISTRY))
    # Mirror into the auth registry the way plugin discovery does; built from public types so the
    # helper is independent of the auth module's private mirroring function.
    pconfig = auth.ProviderConfig(
        profile.name, profile.display_name or profile.name, profile.auth_type, inference_base_url=profile.base_url)
    if profile.auth_type == "api_key" and profile.env_vars:
        pconfig = auth._api_key_provider(profile.name, profile.display_name or profile.name, profile.base_url, tuple(profile.env_vars), "")
    auth.PROVIDER_REGISTRY[profile.name] = pconfig
    monkeypatch.setattr(flows, "_ensure_flow_api_key", lambda *_: (None, "synthetic-test-key", False))
    monkeypatch.setattr(flows, "_env_base_url", lambda *_: "")
    monkeypatch.setattr(flows, "_prompt_base_url_override", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(flows, "_models_dev_merged", lambda *_: [])
    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(models, "fetch_api_models", lambda *_args, **_kwargs: [])
    from hermes_cli import models_pricing
    monkeypatch.setattr(models_pricing, "get_pricing_for_provider", lambda *_: {})
    picker = Mock(return_value=None)
    monkeypatch.setattr(flows, "_pick_model_or_prompt", picker)
    monkeypatch.setattr(flows, "_finish_model", Mock())

    flows._model_flow_api_key_provider({}, profile.name)

    assert picker.call_args.args[0] == list(profile.fallback_models)


def test_setup_uses_profile_endpoint_and_headers(tmp_path, monkeypatch):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from types import SimpleNamespace

    import providers
    from providers.base import ProviderProfile
    from hermes_cli import model_setup_flows as flows

    requests = []

    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("X-Catalog-Contract")))
            payload = json.dumps({"data": [{"id": "live-catalog-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    profile = ProviderProfile(
        name="scout-live-catalog", base_url=base + "/inference",
        models_url=base + "/catalog", default_headers={"X-Catalog-Contract": "present"},
    )
    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.setattr(flows, "_models_dev_merged", lambda *_: [])
    try:
        result = flows._api_key_provider_model_list(
            profile.name, SimpleNamespace(name="Test catalog"),
            "synthetic-test-key", "", profile.base_url,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

    assert result == ["live-catalog-model"]
    assert requests == [("/catalog", "present")]


def test_setup_matches_picker_when_catalog_fetch_fails(monkeypatch):
    """A profile whose catalog is down offers its ``fallback_models`` at setup, exactly what
    ``/model`` shows via ``models.provider_model_ids`` for the same profile."""
    from types import SimpleNamespace

    import providers
    from providers.base import ProviderProfile
    from hermes_cli import model_setup_flows as flows, models

    class DownProfile(ProviderProfile):
        def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
            raise ConnectionError("catalog down")

    profile = DownProfile(
        name="scout-down-catalog", auth_type="api_key", env_vars=("SCOUT_DOWN_TEST_KEY",),
        base_url="https://down.example.invalid/v1", fallback_models=("declared-a", "declared-b"),
    )
    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.setattr(flows, "_models_dev_merged", lambda *_: [])
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("synthetic-test-key", ""))

    setup_rows = flows._api_key_provider_model_list(
        profile.name, SimpleNamespace(name="Down"), "synthetic-test-key", "", profile.base_url)

    assert setup_rows == ["declared-a", "declared-b"] == models.provider_model_ids(profile.name)


def test_switch_validation_trusts_profile_owned_catalog(monkeypatch):
    """A plugin whose ``fetch_models`` is the catalog (#101705): the model the picker offers is
    accepted even when the generic ``/v1/models`` 200s with a different product catalog; a model in
    neither is still rejected."""
    import providers
    from providers.base import ProviderProfile
    from hermes_cli import models, models_validate

    class PlanProfile(ProviderProfile):
        def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
            return ["plan/model-1"]

    profile = PlanProfile(name="scout-plan", auth_type="api_key", env_vars=("SCOUT_PLAN_TEST_KEY",),
                          base_url="https://plan.example.invalid/v1")
    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("synthetic-test-key", ""))
    monkeypatch.setattr(models, "fetch_api_models", lambda *_a, **_k: ["other-vendor/model-a"])

    kw = dict(provider=profile.name, api_key="synthetic-test-key", base_url=profile.base_url)
    assert models_validate.validate_requested_model("plan/model-1", **kw)["accepted"] is True
    assert models_validate.validate_requested_model("plan/model-9", **kw)["accepted"] is False
