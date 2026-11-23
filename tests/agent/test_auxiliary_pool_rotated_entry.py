"""An auxiliary retry after pool rotation uses the entry the rotation chose (no pool reload or re-peek) and a
later auxiliary call does not resend the benched key (follow-up to #106833)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import hermes_yaml as yaml

from agent import auxiliary_client as aux
from agent.credential_pool import PooledCredential
from hermes_cli.auth import write_credential_pool

OK = {"id": "t", "object": "chat.completion", "created": 1, "model": "glm-4.7", "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}


@pytest.fixture
def zai(tmp_path, monkeypatch):
    """Fake z.ai: keys in ``dead`` answer 429, others 200. Yields ``(base, seen_keys, dead, setup)``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    seen, dead = [], {"fixture-0"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            key = self.headers.get("Authorization", "").removeprefix("Bearer ")
            seen.append(key)
            status, data = (429, {"error": {"code": "1310", "message": "Weekly/Monthly Limit Exhausted"}}) \
                if key in dead else (200, OK)
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("GLM_BASE_URL", base)

    def setup(strategy=None):
        rows = [PooledCredential(id=f"key-{i}", provider="zai", label=f"t-{i}", source="manual", auth_type="api_key",
                priority=i, access_token=f"fixture-{i}", base_url=base) for i in range(3)]
        write_credential_pool("zai", [e.to_dict() for e in rows])
        config = {"model": {"provider": "zai", "default": "glm-4.7", "base_url": base}}
        if strategy:
            config["credential_pool_strategies"] = {"zai": strategy}
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        aux.shutdown_cached_clients()
        aux._reset_aux_unhealthy_cache()

    yield base, seen, dead, setup
    aux.shutdown_cached_clients()
    aux._reset_aux_unhealthy_cache()
    server.shutdown()
    server.server_close()


def _call(runtime=None):
    return aux.call_llm(task="compression", main_runtime=runtime, timeout=5,
                        messages=[{"role": "user", "content": "Summarize."}])


def test_retry_uses_the_rotated_entry_without_reloading_the_pool(zai, monkeypatch):
    base, seen, _dead, setup = zai
    setup("round_robin")  # a re-peek of a reloaded pool may pick another row than the rotation did
    chosen, loads = [], []
    real_recover, real_load = aux._recover_provider_pool, aux.load_pool
    monkeypatch.setattr(aux, "load_pool", lambda *a, **k: (loads.append(a), real_load(*a, **k))[1])

    def spy(*args, **kwargs):
        chosen.append(real_recover(*args, **kwargs))
        loads.clear()  # count only what the retry itself loads
        return chosen[-1]

    monkeypatch.setattr(aux, "_recover_provider_pool", spy)
    runtime = {"provider": "zai", "api_key": "fixture-0", "base_url": base, "model": "glm-4.7"}
    assert _call(runtime).choices[0].message.content == "ok"
    assert chosen and chosen[0] is not None and not isinstance(chosen[0], bool)
    assert seen[0] == "fixture-0" and seen[-1] == chosen[0].runtime_api_key  # between: the ladder's same-key backoff retry
    assert loads.count(("zai",)) == 1, loads  # only the rebuilt client's cache hint loads zai: no swap-time reload
