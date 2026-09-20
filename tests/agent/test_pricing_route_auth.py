"""Real configured pricing route -> local HTTP authentication regressions."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.usage_pricing import get_pricing_entry


@pytest.fixture
def endpoint():
    seen = []
    state = {"empty": False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            data = [] if state["empty"] else [{"id": "fixture-model", "pricing": {
                "prompt": "0.000001", "completion": "0.000002",
            }}]
            self.wfile.write(json.dumps({"data": data}).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen, state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def configure(monkeypatch, home, base, key):
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(json.dumps({"custom_providers": [{
        "name": "fixture-proxy", "base_url": base, "key_env": "PRICING_FIXTURE_KEY",
    }]}), encoding="utf-8")
    (home / ".env").write_text(f"PRICING_FIXTURE_KEY={key}\n", encoding="utf-8")


def lookup(base, **kwargs):
    return get_pricing_entry("fixture-model", provider="custom:fixture-proxy", base_url=base, **kwargs)


def test_configured_key_reaches_real_http_probe(monkeypatch, tmp_path, endpoint):
    base, seen, _ = endpoint
    configure(monkeypatch, tmp_path / "profile", base + "/", "configured-key")
    assert lookup(base) is not None
    assert ("/v1/models", "Bearer configured-key") in seen


def test_same_origin_different_path_never_materializes_key(monkeypatch, tmp_path, endpoint):
    base, seen, _ = endpoint
    configure(monkeypatch, tmp_path / "profile", base + "/other-tenant", "private-key")
    def forbidden(_name):
        pytest.fail("Mismatched route must not even materialize the configured credential")
    monkeypatch.setattr("hermes_cli.config.get_env_value_prefer_dotenv", forbidden)
    assert lookup(base) is not None
    assert seen and all(header is None for _, header in seen)


def test_explicit_key_bypasses_config_resolution(monkeypatch, endpoint):
    base, seen, _ = endpoint
    def forbidden(*_args):
        pytest.fail("Explicit key must bypass configured credential resolution")
    monkeypatch.setattr("agent.usage_pricing._configured_endpoint_api_key", forbidden)
    assert lookup(base, api_key="caller-key") is not None
    assert ("/v1/models", "Bearer caller-key") in seen


def test_resolution_failure_preserves_anonymous_probe(monkeypatch, endpoint):
    base, seen, _ = endpoint
    def broken():
        raise RuntimeError("fixture config failure")
    monkeypatch.setattr("hermes_cli.config.load_config", broken)
    assert lookup(base) is not None
    assert seen and all(header is None for _, header in seen)


def test_empty_catalog_stays_best_effort(monkeypatch, tmp_path, endpoint):
    base, seen, state = endpoint
    state["empty"] = True
    configure(monkeypatch, tmp_path / "profile", base, "configured-key")
    assert lookup(base) is None
    assert ("/v1/models", "Bearer configured-key") in seen


def test_profile_keys_follow_home_a_b_a(monkeypatch, tmp_path, endpoint):
    from agent import model_metadata
    base, seen, _ = endpoint
    for name in ("a", "b", "a"):
        configure(monkeypatch, tmp_path / name, base, f"key-{name}")
        # Force the HTTP boundary on every iteration, not a cache-only assertion.
        model_metadata._endpoint_model_metadata_cache.clear()
        model_metadata._endpoint_model_metadata_cache_time.clear()
        seen.clear()
        assert lookup(base) is not None
        assert ("/v1/models", f"Bearer key-{name}") in seen
        assert all(header == f"Bearer key-{name}" for _, header in seen)
