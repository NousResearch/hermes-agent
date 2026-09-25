"""Picker reads do not perform models.dev I/O; capability decisions still can.

Regression for #118023: the ordinary ``model.options`` read must keep the declared
model without waiting on a cold models.dev registry, while image-capability
resolution and an explicit refresh keep their documented network semantics.
"""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from agent import models_dev
from agent.image_routing import decide_image_input_mode
from hermes_cli.inventory import build_model_options_payload, load_picker_context
from hermes_constants import get_hermes_home


@pytest.fixture
def registry(monkeypatch):
    home = get_hermes_home()
    monkeypatch.setattr(Path, "home", lambda: home.parent)
    release = threading.Event()
    requested = threading.Event()
    requests = []
    data = {
        "deepseek": {
            "name": "DeepSeek", "env": ["DEEPSEEK_API_KEY"], "api": "https://api.deepseek.com",
            "models": {"catalog-vision-model": {
                "name": "Catalog vision model", "attachment": True, "tool_call": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 64000, "output": 4000},
            }},
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            requested.set()
            release.wait(15)
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (home / "config.yaml").write_text(
        f"models_dev:\n  url: http://127.0.0.1:{server.server_port}/registry\n"
        "model_catalog:\n  enabled: false\n"
        "model:\n  provider: deepseek\n  default: declared-model\n"
        "providers:\n  deepseek:\n    models:\n      declared-model: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", 0)
    monkeypatch.setattr(models_dev, "_models_dev_retry_after", 0)
    # Other provider transports are outside this regression. Keep metadata resolution real.
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda *a, **k: ["declared-model"])
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lambda *a, **k: {})
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                yield home, data, requests, requested, release, pool
            finally:
                release.set()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        for worker in threading.enumerate():
            if worker.name in ("models-dev-refresh", "hermes-picker-pricing-prewarm") or worker.name.startswith("model-cache-swr-"):
                worker.join(5)
                assert not worker.is_alive()


@pytest.mark.parametrize("explicit_only,include_unconfigured", [(False, False), (True, True)])
def test_cold_picker_keeps_declared_models_without_fetching_registry(
    registry, explicit_only, include_unconfigured,
):
    home, data, requests, requested, release, pool = registry
    future = pool.submit(
        build_model_options_payload, load_picker_context(),
        explicit_only=explicit_only, include_unconfigured=include_unconfigured,
    )
    payload = future.result(timeout=5)
    row = next(row for row in payload["providers"] if row["slug"] == "deepseek")
    assert "declared-model" in row["models"]
    assert requests == []
    assert not (home / "models_dev_cache.json").exists()


def test_capability_lookup_and_explicit_refresh_keep_cold_network_semantics(registry):
    home, data, requests, requested, release, pool = registry
    future = pool.submit(decide_image_input_mode, "deepseek", "catalog-vision-model", {})
    assert requested.wait(5)
    assert not future.done()
    release.set()
    assert future.result(timeout=5) == "native"
    assert json.loads((home / "models_dev_cache.json").read_text(encoding="utf-8")) == data
    models_dev._models_dev_cache = {}
    (home / "models_dev_cache.json").unlink()
    payload = build_model_options_payload(load_picker_context(), refresh=True)
    assert any(row["slug"] == "deepseek" for row in payload["providers"])
    assert len(requests) == 2
