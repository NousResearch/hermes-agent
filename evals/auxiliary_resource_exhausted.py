"""Local HTTP contract probe; no vendor request or personal Hermes state.

Run from the repository with its Python interpreter. JSON output identifies the
loaded module, observed SDK error, request order, and preserved message payload.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spelling", default="ResourceExhausted")
    parser.add_argument("--mode", choices=("sync", "async", "sse"), default="sync")
    parser.add_argument("--status", type=int, default=403)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    requests = []
    message = f"{args.spelling}: Worker local total request limit reached (32/32)"
    messages = [{"role": "user", "content": "Summarize: keep the deployment decision."}]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "messages" not in body:  # capability probes (/api/show) are not chat requests
                self.send_response(404)
                self.end_headers()
                return
            requests.append(body)
            primary = body["model"] == "primary-probe"
            if primary and args.mode == "sse":
                data = 'data: ' + json.dumps({"error": {"message": message}}) + '\n\ndata: [DONE]\n\n'
                status, kind = 200, "text/event-stream"
            elif primary:
                data = json.dumps({"error": {"message": message, "type": "provider_error"}})
                status, kind = args.status, "application/json"
            else:
                data = json.dumps({"id": "local-summary", "object": "chat.completion", "created": 0,
                                   "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                                   "message": {"role": "assistant", "content": "Deployment decision preserved."}}]})
                status, kind = 200, "application/json"
            encoded = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    with tempfile.TemporaryDirectory(prefix="aux-resource-") as home:
        os.environ["HOME"] = home
        os.environ["HERMES_HOME"] = home
        # Two listeners: a payment/quota error is credential-wide, so the fallback must be a
        # different backend identity (distinct base_url) exactly as NIM -> OpenRouter is in the field.
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        fallback_server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        fallback_thread = threading.Thread(target=fallback_server.serve_forever, daemon=True)
        thread.start()
        fallback_thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        fallback_url = f"http://127.0.0.1:{fallback_server.server_port}/v1"
        # The primary is the real ``nvidia`` profile pointed at the local listener; the fallback is a
        # named custom provider (its own credential label, like a second vendor in the field).
        config = {"model": {"provider": "local-fallback", "model": "fallback-probe"},
                  "providers": {"local-fallback": {"base_url": fallback_url, "api_key": "local-probe"}},
                  "auxiliary": {"compression": {"provider": "nvidia", "model": "primary-probe",
                    "base_url": base_url, "api_key": "local-probe", "api_mode": "chat_completions",
                    "fallback_chain": [{"provider": "local-fallback", "model": "fallback-probe"}]}}}
        # JSON is valid YAML and keeps this standalone probe dependency-free.
        Path(home, "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        import logging
        if os.environ.get("AUX_PROBE_DEBUG"):
            logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
        import agent.auxiliary_client as aux

        result = {"module": aux.__file__, "mode": args.mode, "status": args.status,
                  "spelling": args.spelling, "surface": "local HTTP contract, not live provider"}
        try:
            if args.mode == "async":
                response = asyncio.run(aux.async_call_llm(task="compression", messages=messages, max_tokens=64))
            else:
                with aux.aux_progress_hook((lambda: None) if args.mode == "sse" else None):
                    response = aux.call_llm(task="compression", messages=messages, max_tokens=64)
            result["content"] = response.choices[0].message.content
        except Exception as exc:
            result["error"] = type(exc).__name__
            result["error_status"] = getattr(exc, "status_code", None)
            result["error_message"] = str(exc)
        finally:
            for srv, thr in ((server, thread), (fallback_server, fallback_thread)):
                srv.shutdown()
                srv.server_close()
                thr.join()
        result["models"] = [request["model"] for request in requests]
        result["messages_preserved"] = all(request["messages"] == messages for request in requests)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
