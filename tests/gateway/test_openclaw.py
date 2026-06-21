import asyncio
import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gateway.config import PlatformConfig
from gateway.platforms.openclaw import _post_payload, _resolve_url, send_openclaw_direct


class _ReceiverHandler(BaseHTTPRequestHandler):
    secret = "test-secret"
    seen = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        expected = "sha256=" + hmac.new(
            self.secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        if self.headers.get("x-cron-signature") != expected:
            self.send_response(401)
            self.end_headers()
            return
        self.seen.append(json.loads(body.decode("utf-8")))
        self.send_response(202)
        self.end_headers()
        self.wfile.write(b'{"accepted":true}')

    def log_message(self, *_args):
        return


class _StaticResponseHandler(BaseHTTPRequestHandler):
    status = 202
    body = b'{"accepted":true}'

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", "0")))
        self.send_response(self.status)
        self.end_headers()
        if self.body:
            self.wfile.write(self.body)

    def log_message(self, *_args):
        return


def _serve_static_response(status: int, body: bytes):
    class Handler(_StaticResponseHandler):
        pass

    Handler.status = status
    Handler.body = body
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/cron"


def _post_to_static_response(status: int, body: bytes):
    server, url = _serve_static_response(status, body)
    try:
        return _post_payload(
            url,
            "test-secret",
            {
                "profile": "ratatosk",
                "job": "portfolio-health",
                "run_id": "run-123",
                "output": "cron output",
                "severity": "info",
            },
            2.0,
        )
    finally:
        server.shutdown()
        server.server_close()


def test_send_openclaw_direct_posts_signed_receiver_payload():
    _ReceiverHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReceiverHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/cron"

    try:
        result = asyncio.run(
            send_openclaw_direct(
                PlatformConfig(extra={"secret": "test-secret"}),
                url,
                "cron output",
                metadata={
                    "profile": "ratatosk",
                    "job": "portfolio-health",
                    "run_id": "run-123",
                    "severity": "info",
                },
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result["success"] is True
    assert result["message_id"] == "run-123"
    assert _ReceiverHandler.seen == [
        {
            "profile": "ratatosk",
            "job": "portfolio-health",
            "run_id": "run-123",
            "output": "cron output",
            "severity": "info",
        }
    ]


def test_resolve_url_rejects_file_scheme():
    import pytest

    with pytest.raises(ValueError):
        _resolve_url(PlatformConfig(), "file:///etc/passwd")


def test_resolve_url_rejects_ftp_scheme():
    import pytest

    with pytest.raises(ValueError):
        _resolve_url(PlatformConfig(), "ftp://example.com/")


def test_resolve_url_rejects_link_local():
    import pytest

    with pytest.raises(ValueError):
        _resolve_url(PlatformConfig(), "http://169.254.169.254/")


def test_resolve_url_rejects_userinfo():
    import pytest

    with pytest.raises(ValueError):
        _resolve_url(PlatformConfig(), "http://user:pass@localhost/cron")


def test_resolve_url_rejects_fragment():
    import pytest

    with pytest.raises(ValueError):
        _resolve_url(PlatformConfig(), "http://localhost/cron#frag")


def test_resolve_url_rejects_empty_host():
    import pytest

    with pytest.raises(ValueError):
        _resolve_url(PlatformConfig(), "http://")


def test_resolve_url_accepts_ipv4_loopback():
    assert _resolve_url(PlatformConfig(), "http://127.0.0.1:8789/cron") == "http://127.0.0.1:8789/cron"


def test_resolve_url_accepts_ipv6_loopback():
    assert _resolve_url(PlatformConfig(), "http://[::1]:8789/cron") == "http://[::1]:8789/cron"


def test_resolve_url_accepts_https_localhost():
    assert _resolve_url(PlatformConfig(), "https://localhost:8789/cron") == "https://localhost:8789/cron"


def test_resolve_url_env_var_emits_deprecation_warning(monkeypatch):
    import pytest

    monkeypatch.setenv("OPENCLAW_CRON_RECEIVER_URL", "http://127.0.0.1:8789/cron")
    with pytest.warns(DeprecationWarning, match="OPENCLAW_CRON_RECEIVER_URL is deprecated"):
        assert _resolve_url(PlatformConfig(), "") == "http://127.0.0.1:8789/cron"


def test_resolve_url_config_yaml_takes_precedence_over_env(monkeypatch, tmp_path):
    from gateway.config import Platform, load_gateway_config

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
gateway:
  platforms:
    openclaw:
      enabled: true
      extra:
        url: http://127.0.0.1:8789/from-config
        secret: config-secret
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENCLAW_CRON_RECEIVER_URL", "http://127.0.0.1:8789/from-env")

    config = load_gateway_config()

    assert _resolve_url(config.platforms[Platform.OPENCLAW], "") == "http://127.0.0.1:8789/from-config"


def test_resolve_url_empty_allowlist_allows_any_host():
    assert (
        _resolve_url(
            PlatformConfig(extra={"url": "https://example.com/cron", "allowlist": []}),
            "",
        )
        == "https://example.com/cron"
    )


def test_post_payload_rejects_200_without_accepted_body():
    result = _post_to_static_response(200, b'{"ok":true}')

    assert result["success"] is False
    assert "expected HTTP 202" in result["error"]


def test_post_payload_rejects_202_with_accepted_false():
    result = _post_to_static_response(202, b'{"accepted":false}')

    assert result["success"] is False
    assert "acknowledge acceptance" in result["error"]


def test_post_payload_rejects_202_with_non_json_body():
    result = _post_to_static_response(202, b"<html>not json</html>")

    assert result["success"] is False
    assert "non-JSON" in result["error"]


def test_post_payload_rejects_202_with_empty_body():
    result = _post_to_static_response(202, b"")

    assert result["success"] is False
    assert "non-JSON" in result["error"]


def test_post_payload_accepts_202_with_accepted_true():
    result = _post_to_static_response(202, b'{"accepted":true}')

    assert result["success"] is True


def test_post_payload_accepts_202_with_extra_fields():
    result = _post_to_static_response(202, b'{"accepted":true,"queue_depth":5}')

    assert result["success"] is True


def test_openclaw_env_config(monkeypatch):
    from gateway.config import GatewayConfig, Platform, _apply_env_overrides

    monkeypatch.setenv("OPENCLAW_CRON_RECEIVER_URL", "http://127.0.0.1:8789/cron")
    monkeypatch.setenv("OPENCLAW_CRON_SHARED_SECRET", "test-secret")

    config = GatewayConfig()
    _apply_env_overrides(config)

    openclaw = config.platforms[Platform.OPENCLAW]
    assert openclaw.enabled is True
    assert openclaw.extra["url"] == "http://127.0.0.1:8789/cron"
    assert openclaw.extra["secret"] == "test-secret"
    assert openclaw.home_channel.chat_id == "http://127.0.0.1:8789/cron"


def test_send_to_platform_routes_openclaw_payload():
    from gateway.config import Platform
    from tools.send_message_tool import _send_to_platform

    _ReceiverHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReceiverHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/cron"

    try:
        result = asyncio.run(
            _send_to_platform(
                Platform.OPENCLAW,
                PlatformConfig(extra={"secret": "test-secret"}),
                url,
                "cron output",
                metadata={
                    "profile": "magnus",
                    "job": "brand-intel",
                    "run_id": "run-456",
                    "severity": "warning",
                },
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result["success"] is True
    assert _ReceiverHandler.seen[0]["profile"] == "magnus"
    assert _ReceiverHandler.seen[0]["severity"] == "warning"
