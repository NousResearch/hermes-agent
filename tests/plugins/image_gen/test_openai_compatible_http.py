"""Exercise the provider's HTTP and materialization boundaries without a remote service."""

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from agent import image_gen_registry
from hermes_cli.config import atomic_config_write
from hermes_cli.plugins import _ensure_plugins_discovered


def test_generation_and_cache_follow_profile_scope(monkeypatch, tmp_path):
    import requests
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "must-not-leak")
    image_gen_registry._reset_for_tests()
    _ensure_plugins_discovered(force=True)
    provider = image_gen_registry.get_provider("openai-compatible")
    seen = []

    def post(url, **kwargs):
        seen.append((url, kwargs["headers"]["Authorization"]))
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"data": [{"b64_json": base64.b64encode(b"image").decode()}]}).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    secret_scope.set_multiplex_active(True)
    try:
        for profile, credential in [("a", "token-a"), ("b", "token-b"), ("empty", "")]:
            home = tmp_path / profile
            home.mkdir()
            atomic_config_write(home / "config.yaml", {
                "image_gen": {"model": f"{profile}/model", "openai_compatible": {
                    "base_url": f"https://{profile}.example.test", "api_key": "ignored-literal",
                }},
            })
            (home / ".env").write_text(f"OPENAI_COMPATIBLE_IMAGE_API_KEY={credential}\n")
            home_token = set_hermes_home_override(home)
            secret_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
            try:
                before = len(seen)
                result = provider.generate("cat")
                if credential:
                    assert result["success"] is True
                    assert result["model"] == f"{profile}/model"
                    assert Path(result["image"]).is_relative_to(home)
                    assert seen[-1] == (f"https://{profile}.example.test/v1/images/generations", f"Bearer {credential}")
                else:
                    assert result["error_type"] == "auth_required"
                    assert len(seen) == before
            finally:
                secret_scope.reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
    finally:
        secret_scope.set_multiplex_active(False)
        image_gen_registry._reset_for_tests()


@pytest.mark.parametrize("output", ["b64_json", "url", "redirect"])
def test_real_http_preserves_credentials_and_materializes_output(monkeypatch, tmp_path, output):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "profile-token")
    seen = []
    image_bytes = b"generated-image"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), payload))
            if output == "redirect":
                self.send_response(307)
                self.send_header("Location", "/credential-sink")
                self.end_headers()
                return
            value = (base64.b64encode(image_bytes).decode() if output == "b64_json"
                     else f"http://127.0.0.1:{self.server.server_port}/output.png")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{output: value}]}).encode())

        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization"), None))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(image_bytes)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    image_gen_registry._reset_for_tests()
    try:
        atomic_config_write(tmp_path / "config.yaml", {
            "security": {"allow_private_urls": True},
            "image_gen": {
                "provider": "openai-compatible",
                "model": "../../vendor/model",
                "openai_compatible": {"base_url": f"http://127.0.0.1:{server.server_port}/v1"},
            },
        })
        _ensure_plugins_discovered(force=True)
        provider = image_gen_registry.get_provider("openai-compatible")
        result = provider.generate("cat", "square")
        assert seen[0][0] == "/v1/images/generations"
        assert seen[0][1] == "Bearer profile-token"
        assert seen[0][2]["model"] == "../../vendor/model"
        if output == "redirect":
            assert result["error_type"] == "redirect_not_allowed"
            assert len(seen) == 1
        else:
            assert result["success"] is True
            image = Path(result["image"])
            assert image.read_bytes() == image_bytes
            assert image.parent == tmp_path / "cache" / "generated" / "images"
            if output == "url":
                assert seen[1] == ("/output.png", None, None)
    finally:
        image_gen_registry._reset_for_tests()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
