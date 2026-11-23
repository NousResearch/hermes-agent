"""Same-provider retries must bench the credential actually sent (PR #106833)."""
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import hermes_yaml as yaml

from agent import auxiliary_client as aux
from agent.credential_pool import PooledCredential
from hermes_cli.auth import read_credential_pool, write_credential_pool


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("stale_auto", [False, True])
def test_retry_exhausts_only_keys_that_failed_over_http(tmp_path, monkeypatch, caplog, async_mode, stale_auto):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            key = self.headers.get("Authorization", "").removeprefix("Bearer ")
            seen.append(key)
            if body["model"] == "fallback-model":
                status, data = 200, {
                    "id": "test", "object": "chat.completion", "created": 1,
                    "model": body["model"], "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "Recovered."},
                    "finish_reason": "stop"}],
                }
            else:
                status, data = 429, {"error": {"code": "1310", "message": "Weekly/Monthly Limit Exhausted"}}
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("GLM_BASE_URL", base)
    monkeypatch.setenv("AUX_TEST_FALLBACK_KEY", "fixture-fallback")
    rows = [PooledCredential(id=f"key-{i}", provider="zai", label=f"test-{i}",
            source="manual", auth_type="api_key", priority=i,
            access_token=f"fixture-{i}", base_url=base) for i in range(3)]
    write_credential_pool("zai", [e.to_dict() for e in rows])
    config = {"model": {"provider": "zai", "default": "glm-4.7", "base_url": base},
              "providers": {"test-fallback": {"base_url": base, "key_env": "AUX_TEST_FALLBACK_KEY"}},
              "auxiliary": {"compression": {"provider": "auto" if stale_auto else "zai",
                  "model": "glm-4.7", "fallback_chain": [{"provider": "custom:test-fallback",
                  "model": "fallback-model", "base_url": base}]}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    aux.shutdown_cached_clients()
    aux._reset_aux_unhealthy_cache()
    runtime = {"provider": "zai", "api_key": "fixture-0", "base_url": base,
               "model": "glm-4.7", "api_mode": "chat_completions"} if stale_auto else None
    try:
        kwargs = dict(task="compression", main_runtime=runtime,
                      messages=[{"role": "user", "content": "Summarize."}], timeout=5)
        response = asyncio.run(aux.async_call_llm(**kwargs)) if async_mode else aux.call_llm(**kwargs)
        assert response.choices[0].message.content == "Recovered."
        attempted = set(seen)
        assert "fixture-0" in attempted and "fixture-fallback" in attempted
        # Auto routes rebuild from the main runtime: the retry must carry the rotated-to key too.
        assert "fixture-1" in attempted
        assert "fixture-2" not in attempted
        # Inspect durable state, not the warm pool object or a mocked rotation call.
        stored = read_credential_pool("zai")
        states = {e["id"]: e.get("last_status") for e in stored}
        assert states == {"key-0": "exhausted", "key-1": "exhausted", "key-2": None}
        error_metadata = json.dumps([{k: v for k, v in e.items() if k.startswith("last_")} for e in stored])
        assert all(key not in caplog.text + error_metadata for key in attempted)
    finally:
        aux.shutdown_cached_clients()
        aux._reset_aux_unhealthy_cache()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _zai_runtime_pool(tmp_path, monkeypatch, urls, cooldowns=None):
    """Pool rows key-0..n on ``urls``; key-0 is benched by a real 429 rotation. Returns the runtime on urls[0]."""
    from agent.auxiliary_pool_retry import runtime_with_rotated_pool_key
    from agent.credential_pool import load_pool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rows = [PooledCredential(id=f"key-{i}", provider="zai", label=f"test-{i}", source="manual",
            auth_type="api_key", priority=i, access_token=f"fixture-{i}", base_url=url,
            model_cooldowns=(cooldowns or {}).get(i)) for i, url in enumerate(urls)]
    write_credential_pool("zai", [e.to_dict() for e in rows])
    load_pool("zai").mark_exhausted_and_rotate(status_code=429, credential_id="key-0")
    runtime = {"provider": "zai", "api_key": "fixture-0", "base_url": urls[0], "model": "glm-4.7"}
    return runtime_with_rotated_pool_key, runtime


def test_rotated_key_for_another_endpoint_is_never_swapped_in(tmp_path, monkeypatch):
    swap, runtime = _zai_runtime_pool(tmp_path, monkeypatch, ["https://a.example/v1", "https://b.example/v1"])
    assert swap(runtime) is runtime  # key-1 belongs to endpoint B: it must not be sent to A


def test_rotated_key_for_the_same_endpoint_is_swapped_in(tmp_path, monkeypatch):
    swap, runtime = _zai_runtime_pool(tmp_path, monkeypatch, ["https://a.example/v1"] * 2)
    assert swap(runtime)["api_key"] == "fixture-1"


def test_runtime_without_base_url_fails_closed(tmp_path, monkeypatch):
    swap, runtime = _zai_runtime_pool(tmp_path, monkeypatch, ["https://a.example/v1"] * 2)
    runtime.pop("base_url")
    assert swap(runtime) is runtime


def test_swap_only_skips_keys_cooled_down_for_the_runtime_model(tmp_path, monkeypatch):
    future = time.time() + 3600
    urls = ["https://a.example/v1"] * 3
    swap, runtime = _zai_runtime_pool(tmp_path, monkeypatch, urls, {1: {"other-model": future}})
    assert swap(runtime)["api_key"] == "fixture-1"  # key-1 is only benched for another model
    swap, runtime = _zai_runtime_pool(tmp_path, monkeypatch, urls, {1: {"glm-4.7": future}})
    assert swap(runtime)["api_key"] == "fixture-2"
