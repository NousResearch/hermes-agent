"""Offline production-import reproduction: no repo edits or real credentials."""

import os
import sys
import tempfile
from pathlib import Path

REPO = sys.argv[1]
sys.dont_write_bytecode = True
sandbox = tempfile.TemporaryDirectory(prefix="hermes-104260-")
os.environ.clear()
os.environ.update(
    HOME=sandbox.name,
    HERMES_HOME=sandbox.name + "/hermes",
    PATH="/usr/bin:/bin",
    PYTHONDONTWRITEBYTECODE="1",
    HERMES_DISABLE_MODEL_METADATA_FETCH="1",
)
Path(os.environ["HERMES_HOME"]).mkdir()
os.chdir(sandbox.name)
sys.path.insert(0, REPO)
import socket

_original_connect = socket.socket.connect
blocked = []


def loopback_only(self, address):
    if isinstance(address, tuple) and address[0] not in (
        "127.0.0.1",
        "::1",
        "localhost",
    ):
        blocked.append(str(address))
        raise RuntimeError("Probe blocks non-loopback network")
    return _original_connect(self, address)


socket.socket.connect = loopback_only
import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.WARNING)
requests = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        requests.append({
            "endpoint": self.server.kind,
            "path": self.path,
            "model": body.get("model"),
        })
        if self.server.kind == "cloud-simulator":
            status = self.server.failure_status
            data = {
                "error": {
                    "message": "weekly usage limit exceeded",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            }
        else:
            status = 200
            data = {
                "id": "local-probe",
                "object": "chat.completion",
                "created": 1,
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "LOCAL_OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        wire = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(wire)))
        self.end_headers()
        self.wfile.write(wire)


servers = []
for kind in ("cloud-simulator", "healthy-local"):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.kind = kind
    srv.failure_status = 402
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers.append(srv)
cloud, local = [f"http://127.0.0.1:{s.server_port}/v1" for s in servers]
config = {
    "model": {
        "provider": "custom",
        "default": "local-probe-model",
        "base_url": local,
        "api_key": "no-key-required",
        "api_mode": "chat_completions",
    },
    "auxiliary": {
        "compression": {
            "fallback_chain": [
                {
                    "provider": "custom",
                    "model": "local-probe-model",
                    "base_url": local,
                    "api_key": "no-key-required",
                    "api_mode": "chat_completions",
                }
            ]
        }
    },
}
# JSON is valid YAML; only the temporary Hermes home is written.
Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
    json.dumps(config), encoding="utf-8"
)
from agent import auxiliary_client as aux
from agent.backend_identity import BackendIdentity, FailureScope, should_skip_candidate

assert aux.__file__.startswith(REPO)
print("PRODUCTION_IMPORT", aux.__file__, flush=True)
args = dict(
    provider="custom",
    model="cloud-probe-model",
    base_url=cloud,
    api_key="no-key-required",
    api_mode="chat_completions",
    messages=[{"role": "user", "content": "probe"}],
    timeout=2,
    max_tokens=8,
)
local_args = dict(args, model="local-probe-model", base_url=local)
results = []
for mode, status in [("sync", 402), ("async", 402), ("sync", 429)]:
    aux._reset_aux_unhealthy_cache()
    servers[0].failure_status = status
    before = aux.call_llm(**local_args).choices[0].message.content
    start = len(requests)
    try:
        if mode == "sync":
            aux.call_llm(task="compression", **args)
        else:
            asyncio.run(
                aux.async_call_llm(
                    task="compression",
                    **{k: v for k, v in args.items() if k != "api_mode"},
                )
            )
        error = None
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "status": getattr(exc, "status_code", None),
            "payment": aux._is_payment_error(exc),
        }
    failure_requests = requests[start:]
    cache = {
        str(k): round(v - time.time(), 2) for k, v in aux._aux_unhealthy_until.items()
    }
    route_after = aux._try_main_provider_route(
        "custom", "local-probe-model", local, "no-key-required", "chat_completions"
    )
    direct_after = aux.call_llm(**local_args).choices[0].message.content
    aux._reset_aux_unhealthy_cache()
    route_reset = aux._try_main_provider_route(
        "custom", "local-probe-model", local, "no-key-required", "chat_completions"
    )
    entry = {
        "mode": mode,
        "status": status,
        "local_before": before,
        "cloud_error": error,
        "failure_requests": failure_requests,
        "health_cache_remaining_seconds": cache,
        "local_auto_route_after_payment": route_after is not None,
        "direct_local_after_payment": direct_after,
        "local_auto_route_after_cache_reset": route_reset is not None,
    }
    print("OBSERVED", json.dumps(entry), flush=True)
    assert before == direct_after == "LOCAL_OK"
    assert error is None
    assert any(r["endpoint"] == "healthy-local" for r in failure_requests)
    assert len(cache) == 1 and all(590 < v <= 600 for v in cache.values())
    assert route_after is not None and route_reset is not None
    results.append(entry)
    print("CASE", json.dumps(entry), flush=True)
ident_cloud = BackendIdentity.build("custom", "cloud-probe-model", cloud)
ident_local = BackendIdentity.build("custom", "local-probe-model", local)
identity = {
    "credential_skip_distinct_custom_urls": should_skip_candidate(
        ident_local, ident_cloud, FailureScope.CREDENTIAL
    ),
    "endpoint_skip_distinct_custom_urls": should_skip_candidate(
        ident_local, ident_cloud, FailureScope.ENDPOINT
    ),
    "model_skip_distinct_custom_urls": should_skip_candidate(
        ident_local, ident_cloud, FailureScope.MODEL
    ),
    "named_custom_distinct_labels_credential_skip": should_skip_candidate(
        BackendIdentity.build("custom:local", "local-probe-model", local),
        BackendIdentity.build("custom:cloud", "cloud-probe-model", cloud),
        FailureScope.CREDENTIAL,
    ),
}
assert identity == {
    "credential_skip_distinct_custom_urls": True,
    "endpoint_skip_distinct_custom_urls": False,
    "model_skip_distinct_custom_urls": False,
    "named_custom_distinct_labels_credential_skip": False,
}
print("IDENTITY", json.dumps(identity), flush=True)
print("BLOCKED_NON_LOOPBACK", json.dumps(blocked), flush=True)
print(
    "PASS: sync/async local fallback and subsequent routing survive cloud payment failures.",
    flush=True,
)
for srv in servers:
    srv.shutdown()
sandbox.cleanup()
