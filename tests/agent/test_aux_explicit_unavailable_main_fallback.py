"""A dead explicit aux pin (retired or frozen provider) must not leave the lane with no route (#147153)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import hermes_yaml as yaml


class _ChatHandler(BaseHTTPRequestHandler):
    seen_models: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        type(self).seen_models.append(body.get("model"))
        payload = json.dumps({
            "id": "c1", "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "summary"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        payload = json.dumps({"object": "list", "data": [{"id": "main-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_explicit_aux_provider_without_credentials_uses_working_main_model(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _ChatHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "main-model", "base_url": base_url, "api_key": "k"},
            "auxiliary": {"compression": {"provider": "alibaba", "model": "retired-model"}},
        }), encoding="utf-8")
        from agent.auxiliary_client import call_llm

        response = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}], max_tokens=5)
        assert response.choices[0].message.content == "summary"
        assert "main-model" in _ChatHandler.seen_models
    finally:
        server.shutdown()
