"""Tests for llm_benchmark_route_sources: stable route recognition, accurate
contexts, legacy compatibility, and failure modes. No live network."""

import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "llm_benchmark_route_sources.py"

LEGACY_CONFIG = {
    "providers": [
        {
            "name": "openrouter-free",
            "model": "qwen/qwen3-coder:free",
            "endpoint": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
        }
    ]
}

CATALOG = {
    "object": "list",
    "data": [
        {"id": "auto", "object": "model", "owned_by": "turbofit", "context_length": 204800},
        {"id": "active:main", "object": "model", "owned_by": "turbofit", "context_length": 204800},
        {"id": "active:aux", "object": "model", "owned_by": "turbofit", "context_length": 262144},
        {"id": "qwen3.8-27b", "object": "model", "owned_by": "turbofit", "context_length": 4096},
    ],
}


def load_module():
    spec = importlib.util.spec_from_file_location("llm_benchmark_route_sources", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CatalogServer:
    def __init__(self, payload, status=200):
        state = {"payload": payload, "status": status}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                body = json.dumps(state["payload"]).encode() if not isinstance(state["payload"], bytes) else state["payload"]
                self.send_response(state["status"])
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_stable_routes_recognised_with_accurate_contexts():
    module = load_module()

    entries = module.turbofit_sources(CATALOG, "http://127.0.0.1:8091")

    ids = [entry["model"] for entry in entries]
    assert ids == ["auto", "active:main", "active:aux"]
    contexts = {entry["model"]: entry["context_length"] for entry in entries}
    assert contexts == {"auto": 204800, "active:main": 204800, "active:aux": 262144}
    for entry in entries:
        assert entry["endpoint"] == "http://127.0.0.1:8091/v1"
        assert entry["backend"] == "turbofit-gateway"
        assert entry["key_env"] == "TURBOFIT_BENCH_KEY"


def test_concrete_model_tags_are_not_benchmark_targets():
    module = load_module()

    entries = module.turbofit_sources(CATALOG, "http://127.0.0.1:8091")

    assert "qwen3.8-27b" not in {entry["model"] for entry in entries}


def test_legacy_providers_pass_through_until_flip():
    module = load_module()

    result = module.build_sources(LEGACY_CONFIG, enable_turbofit=False)

    assert result["turbofit_enabled"] is False
    assert result["turbofit_status"] == "disabled"
    assert [p["name"] for p in result["providers"]] == ["openrouter-free"]
    assert result["providers"][0]["endpoint"] == "https://openrouter.ai/api/v1"


def test_turbofit_entries_append_after_legacy_when_enabled():
    module = load_module()

    result = module.build_sources(
        LEGACY_CONFIG, enable_turbofit=True, fetch_catalog=lambda url: CATALOG
    )

    names = [p["name"] for p in result["providers"]]
    assert names[0] == "openrouter-free"
    assert "turbofit-active-main" in names
    assert result["turbofit_status"] == "ok"
    assert result["key_envs"] == ["OPENROUTER_API_KEY", "TURBOFIT_BENCH_KEY"]


def test_unavailable_turbofit_does_not_drop_legacy_entries():
    module = load_module()

    def failing_fetch(url):
        raise OSError("connection refused")

    result = module.build_sources(
        LEGACY_CONFIG, enable_turbofit=True, fetch_catalog=failing_fetch
    )

    assert result["turbofit_status"].startswith("unavailable")
    assert [p["name"] for p in result["providers"]] == ["openrouter-free"]


def test_no_secret_material_in_output(tmp_path):
    module = load_module()
    legacy = {
        "providers": [
            {"name": "x", "model": "m", "endpoint": "https://e/v1",
             "key_env": "X_KEY", "api_key": "sk-super-secret"}
        ]
    }

    result = module.build_sources(legacy, enable_turbofit=False)
    rendered = json.dumps(result)

    assert "sk-super-secret" not in rendered
    assert "api_key" not in rendered  # allow-list projection drops credential fields


def test_fetch_model_catalog_rejects_bad_payload():
    module = load_module()
    server = _CatalogServer({"nope": True})
    try:
        module.fetch_model_catalog(server.url())
        raise AssertionError("expected ValueError for unrecognised payload")
    except ValueError:
        pass
    finally:
        server.close()


def test_cli_disabled_by_default_and_legacy_roundtrip(tmp_path):
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(LEGACY_CONFIG))
    output = tmp_path / "sources.json"

    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--legacy-config", str(legacy_path),
         "--output", str(output)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(output.read_text())
    assert result["turbofit_enabled"] is False
    assert result["turbofit_status"] == "disabled"
    assert [p["name"] for p in result["providers"]] == ["openrouter-free"]
    assert output.stat().st_mode & 0o777 == 0o600


def test_cli_enable_turbofit_against_local_catalog_server(tmp_path):
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(LEGACY_CONFIG))
    output = tmp_path / "sources.json"
    server = _CatalogServer(CATALOG)
    try:
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--legacy-config", str(legacy_path),
             "--turbofit-url", server.url(), "--enable-turbofit",
             "--output", str(output)],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
    finally:
        server.close()

    result = json.loads(output.read_text())
    assert result["turbofit_status"] == "ok"
    models = [p["model"] for p in result["providers"]]
    assert models[:1] == ["qwen/qwen3-coder:free"]  # legacy first
    assert {"auto", "active:main", "active:aux"} <= set(models)


def test_sources_feed_measurements_config_shape():
    """The produced providers list must satisfy llm_benchmark_measurements' contract."""
    module = load_module()

    result = module.build_sources(
        LEGACY_CONFIG, enable_turbofit=True, fetch_catalog=lambda url: CATALOG
    )

    for provider in result["providers"]:
        assert {"name", "model", "endpoint", "key_env"} <= set(provider)
        assert provider["endpoint"].rstrip("/").endswith(("v1", "8091"))