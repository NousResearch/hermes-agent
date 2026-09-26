"""Same-provider retries must bench the credential actually sent (PR #106833)."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

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
        assert ("fixture-1" in attempted) is not stale_auto
        assert "fixture-2" not in attempted
        # Inspect durable state, not the warm pool object or a mocked rotation call.
        stored = read_credential_pool("zai")
        states = {e["id"]: e.get("last_status") for e in stored}
        assert states == {"key-0": "exhausted", "key-1": None if stale_auto else "exhausted", "key-2": None}
        error_metadata = json.dumps([{k: v for k, v in e.items() if k.startswith("last_")} for e in stored])
        assert all(key not in caplog.text + error_metadata for key in attempted)
    finally:
        aux.shutdown_cached_clients()
        aux._reset_aux_unhealthy_cache()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
