"""The bare ``model.provider: custom`` picker row must survive switching away.

The row is built from the session's current-provider slice when the session is
on it; otherwise it is recovered from the static config.yaml ``model:`` block
(#59702). Before that fallback existed the row vanished whenever a different
provider was active, and the unconfigured-canonical fallback substituted a
misleading 0-model placeholder in its place.

Its ``/models`` probe authenticates with the ``model:`` block's own credential,
and only toward ``model.base_url``.
"""

import http.server
import json
import threading

import pytest

import hermes_cli.providers as providers_mod
from hermes_cli.config import save_config
from hermes_cli.model_switch import list_authenticated_providers


def test_bare_custom_row_survives_switching_away_from_it(monkeypatch):
    """Regression for #59702: the bare ``model.provider: custom`` row must not
    vanish from the picker when the session's current provider is a different
    one."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {"default": "gpt-4o", "provider": "custom", "base_url": "https://www.ccsub.net/v1"}
        },
    )
    # The config-sourced branch serves the warm cache / falls back to the
    # config default; keep the test hermetic — no live probe.
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.cached_fetch_api_models", lambda *_a, **_kw: None)

    providers = list_authenticated_providers(
        current_provider="xai",
        current_base_url="",
        current_model="grok-4",
        user_providers={},
        custom_providers=[],
        probe_custom_providers=False,
    )

    bare_custom = next((p for p in providers if p["slug"] == "custom"), None)
    assert bare_custom is not None, "bare custom endpoint row vanished after switching provider"
    assert bare_custom["name"] == "Custom endpoint"
    assert bare_custom["is_current"] is False
    assert bare_custom["models"] == ["gpt-4o"], "config default model must seed the row when no probe runs"
    assert bare_custom["api_url"] == "https://www.ccsub.net/v1"
    assert bare_custom["source"] == "model-config"


def test_bare_custom_row_config_fallback_defers_to_matching_named_entry(monkeypatch):
    """The config-sourced bare row must not duplicate a named custom_providers
    entry that already covers the same base_url (#59702)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {"default": "gpt-4o", "provider": "custom", "base_url": "https://www.ccsub.net/v1"}
        },
    )

    providers = list_authenticated_providers(
        current_provider="xai",
        current_base_url="",
        current_model="grok-4",
        user_providers={},
        custom_providers=[{"name": "my-endpoint", "base_url": "https://www.ccsub.net/v1", "model": "gpt-4o"}],
        probe_custom_providers=False,
    )

    assert not any(p["slug"] == "custom" for p in providers)
    assert any(p["slug"] == "custom:my-endpoint" for p in providers)


def test_bare_custom_row_config_fallback_tolerates_legacy_scalar_model(monkeypatch):
    """A legacy bare-string ``model:`` config value must not crash the picker
    (the ConfigContext loader already tolerates it)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": "gpt-4o"})

    providers = list_authenticated_providers(
        current_provider="xai",
        current_base_url="",
        current_model="grok-4",
        user_providers={},
        custom_providers=[],
        probe_custom_providers=False,
    )

    assert not any(p["slug"] == "custom" for p in providers)


_ENDPOINT_KEY = "endpoint-key"


@pytest.fixture
def keyed_endpoint():
    """Loopback OpenAI-compatible server whose ``/v1/models`` answers only the right key
    (a LiteLLM router with ``master_key``); records every Authorization header it saw."""
    seen_auth: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen_auth.append(self.headers.get("Authorization", ""))
            if self.path != "/v1/models" or seen_auth[-1] != f"Bearer {_ENDPOINT_KEY}":
                self.send_response(401)
                self.end_headers()
                return
            body = json.dumps({"object": "list", "data": [
                {"id": "deepseek-v4.1-flash", "object": "model"},
                {"id": "glm-5.3-flash", "object": "model"},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen_auth
    finally:
        server.shutdown()
        server.server_close()


def _write_model_block(model_block: dict) -> None:
    save_config({"model": model_block})


def _bare_custom_row(base_url: str) -> dict:
    rows = list_authenticated_providers(
        current_provider="custom", current_base_url=base_url, current_model="glm-5.3-flash",
        user_providers={}, custom_providers=[], probe_current_custom_provider=True, for_picker=True,
    )
    return next(p for p in rows if p["slug"] == "custom")


_KEY_ENV = "HERMES_TEST_BARE_CUSTOM_KEY"
_BOTH_MODELS = ["deepseek-v4.1-flash", "glm-5.3-flash"]


@pytest.fixture
def picker_stubs(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setenv(_KEY_ENV, _ENDPOINT_KEY)


@pytest.mark.parametrize("credential", [
    pytest.param({"api_key": _ENDPOINT_KEY}, id="api_key"),
    pytest.param({"api": _ENDPOINT_KEY}, id="legacy-api"),
    pytest.param({"key_env": _KEY_ENV}, id="key_env"),
    pytest.param({"api_key": "changeme", "key_env": _KEY_ENV}, id="placeholder-api_key-defers-to-key_env"),
])
def test_bare_custom_row_lists_every_model_of_a_keyed_endpoint(picker_stubs, keyed_endpoint, credential):
    """The ``/models`` probe carries the credential the runtime sends for the ``model:`` block: a
    keyed endpoint refuses an anonymous listing, and the row then showed only ``model.default``
    although the endpoint serves more (#147982, #83837)."""
    base_url, _seen_auth = keyed_endpoint
    _write_model_block({"default": "glm-5.3-flash", "provider": "custom", "base_url": base_url, **credential})

    assert _bare_custom_row(base_url)["models"] == _BOTH_MODELS


@pytest.mark.parametrize("declared_url", [
    pytest.param(lambda url: "http://127.0.0.1:9/v1", id="other-endpoint"),
    pytest.param(lambda url: url[: -len("/v1")] + "/V1", id="path-case-variant"),
])
def test_bare_custom_row_never_sends_a_key_declared_for_another_endpoint(picker_stubs, keyed_endpoint, declared_url):
    """``model.api_key`` belongs to ``model.base_url`` exactly, as the runtime compares it: a
    session on any other custom URL, even one differing only in path case, must not receive it
    while listing models. Path-case case adapted from #113149."""
    base_url, seen_auth = keyed_endpoint
    _write_model_block({
        "default": "glm-5.3-flash", "provider": "custom", "base_url": declared_url(base_url),
        "api_key": _ENDPOINT_KEY})

    assert _bare_custom_row(base_url)["models"] == ["glm-5.3-flash"]
    assert seen_auth and not any(_ENDPOINT_KEY in header for header in seen_auth)


def test_bare_custom_row_from_config_lists_and_caches_the_keyed_catalog(picker_stubs, keyed_endpoint):
    """Session on another provider: the row comes from config.yaml. An explicit refresh probes
    with the block's key, and the cache-only read (the desktop picker) then serves that catalog
    without another request."""
    base_url, seen_auth = keyed_endpoint
    _write_model_block({
        "default": "glm-5.3-flash", "provider": "custom", "base_url": base_url, "api_key": _ENDPOINT_KEY})

    def config_row(*, refresh: bool) -> dict:
        rows = list_authenticated_providers(
            current_provider="xai", current_base_url="", current_model="grok-4",
            user_providers={}, custom_providers=[], probe_custom_providers=False, refresh=refresh)
        return next(p for p in rows if p["slug"] == "custom")

    assert config_row(refresh=True)["models"] == _BOTH_MODELS
    probes = len(seen_auth)
    assert config_row(refresh=False)["models"] == _BOTH_MODELS
    assert len(seen_auth) == probes
