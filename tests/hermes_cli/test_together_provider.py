"""Together setup and catalog contracts through the shared provider paths."""

import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread


def test_together_setup_and_runtime(tmp_path, monkeypatch):
    from hermes_cli.auth import resolve_provider
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.models import CANONICAL_PROVIDERS, normalize_provider
    from hermes_cli.providers import get_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from providers import get_provider_profile

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("TOGETHER_API_KEY=together-test-key\n")
    profile = get_provider_profile("togetherai")
    assert profile is not None
    assert any(p.slug == profile.name and p.label == profile.display_name for p in CANONICAL_PROVIDERS)
    assert OPTIONAL_ENV_VARS[profile.env_vars[0]]["password"] is True
    assert profile.default_aux_model in profile.fallback_models

    for alias in (profile.name, *profile.aliases):
        assert normalize_provider(alias) == resolve_provider(alias) == profile.name
        provider = get_provider(alias, allow_network=False)
        assert provider.id == profile.name
        assert provider.base_url == profile.base_url
        runtime = resolve_runtime_provider(requested=alias)
        assert runtime["provider"] == profile.name
        assert runtime["api_key"] == "together-test-key"
        assert runtime["base_url"] == profile.base_url
        assert runtime["api_mode"] == profile.api_mode
        for model in profile.fallback_models:
            assert normalize_model_for_provider(model, alias) == model


def test_together_catalog_and_endpoint_override(tmp_path, monkeypatch):
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import provider_model_ids
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from providers import get_provider_profile

    profile = get_provider_profile("togetherai")
    assert profile is not None
    requests = []
    user_agents = []
    chat_models = ["test/new-chat-model", "test/private-model"]

    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            user_agents.append(self.headers.get("User-Agent", ""))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps([
                {"id": "test/new-chat-model", "type": "chat"},
                {"id": "test/private-model"},
                {"id": "test/image-model", "type": "image"},
                {"id": "test/audio-model", "type": "audio"},
                {"id": "test/embedding-model", "type": "embedding"},
            ]).encode())

        def log_message(self, *args):
            pass

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("TOGETHER_API_KEY=together-test-key\n")
    with ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler) as server:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        (tmp_path / "config.yaml").write_text(
            f"model:\n  provider: togetherai\n  base_url: {endpoint}\n"
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            runtime = resolve_runtime_provider(requested=profile.name)
            assert runtime["base_url"] == endpoint
            assert runtime["api_key"] == "together-test-key"
            requests.clear()  # Exclude the runtime's local-endpoint discovery probes.
            user_agents.clear()
            assert profile.fetch_models(api_key=runtime["api_key"], base_url=endpoint) == chat_models
            # The shared picker uses the registry endpoint, independently of model.base_url.
            monkeypatch.setitem(PROVIDER_REGISTRY, profile.name,
                                replace(PROVIDER_REGISTRY[profile.name], inference_base_url=endpoint))
            models = provider_model_ids(profile.name)
            assert models == [*profile.fallback_models, *chat_models]
            assert requests == [("/v1/models", "Bearer together-test-key")] * 2
            assert all(agent.startswith("hermes-cli") for agent in user_agents)
        finally:
            server.shutdown()
            thread.join()

    assert provider_model_ids(profile.name) == list(profile.fallback_models)
