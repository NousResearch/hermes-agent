"""Tests for _probe_gateway_health() — cross-container gateway detection,
including the bounded JSON response body reads (moved from test_web_server.py
so that file keeps its main-pinned line cap; moved code keeps its cap)."""

import json
from unittest.mock import MagicMock

import hermes_cli.web_server as ws
import hermes_cli.web_server_gateway as _web_server_gateway


class TestProbeGatewayHealth:
    """Tests for _probe_gateway_health() — cross-container gateway detection."""

    def test_probe_uses_configured_short_timeout(self, monkeypatch):
        """The HTTP probe must not fall through to the OS TCP timeout."""
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 0.75)
        timeouts = []

        def mock_urlopen(req, **kwargs):
            timeouts.append(kwargs.get("timeout"))
            raise TimeoutError("mock timeout")

        monkeypatch.setattr(_web_server_gateway.urllib.request, "urlopen", mock_urlopen)

        alive, body = _web_server_gateway._probe_gateway_health()

        assert alive is False
        assert body is None
        assert timeouts == [0.75, 0.75]

    def test_detailed_fails_falls_back_to_simple_health(self, monkeypatch):
        """If /health/detailed fails, falls back to /health."""
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)

        call_count = [0]

        def mock_urlopen(req, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("detailed failed")
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        monkeypatch.setattr(_web_server_gateway.urllib.request, "urlopen", mock_urlopen)
        alive, body = _web_server_gateway._probe_gateway_health()
        assert alive is True
        assert body is not None
        assert body["status"] == "ok"
        assert call_count[0] == 2

    def test_successful_probe_bounds_response_read(self, monkeypatch):
        """Gateway health JSON must be read with a defensive size cap."""
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)
        captured = {}

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, size=-1):
                captured["size"] = size
                data = json.dumps({"status": "ok", "pid": 42}).encode()
                return data if size < 0 else data[:size]

        monkeypatch.setattr(
            _web_server_gateway.urllib.request, "urlopen", lambda req, **kw: _Resp()
        )
        alive, body = _web_server_gateway._probe_gateway_health()
        assert alive is True
        assert body is not None
        assert body["pid"] == 42
        assert captured["size"] == (
            _web_server_gateway._DASHBOARD_JSON_RESPONSE_BODY_MAX_BYTES + 1
        )

    def test_oversized_detailed_probe_falls_back_to_simple_health(self, monkeypatch):
        """An oversized detailed health body must not block the fallback path."""
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)
        monkeypatch.setattr(
            _web_server_gateway, "_DASHBOARD_JSON_RESPONSE_BODY_MAX_BYTES", 16
        )
        calls = []

        class _Resp:
            status = 200

            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, size=-1):
                return self.payload if size < 0 else self.payload[:size]

        def mock_urlopen(req, **kwargs):
            calls.append(req.full_url)
            if len(calls) == 1:
                return _Resp(b"x" * 17)
            return _Resp(json.dumps({"status": "ok"}).encode())

        monkeypatch.setattr(_web_server_gateway.urllib.request, "urlopen", mock_urlopen)
        alive, body = _web_server_gateway._probe_gateway_health()
        assert alive is True
        assert body is not None
        assert body["status"] == "ok"
        assert calls == [
            "http://gw:8642/health/detailed",
            "http://gw:8642/health",
        ]


class TestProbeGatewayHealthAuth:
    """API_SERVER_KEY must authenticate the /health/detailed probe — and must
    never leak to the unauthenticated /health fallback endpoint."""

    @staticmethod
    def _recording_urlopen(monkeypatch, detailed_behavior):
        """urlopen mock that records each request's (url, Authorization header)."""
        requests = []

        def mock_urlopen(req, **kwargs):
            requests.append((req.full_url, req.get_header("Authorization")))
            if len(requests) == 1:
                detailed_behavior()
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        monkeypatch.setattr(_web_server_gateway.urllib.request, "urlopen", mock_urlopen)
        return requests

    def test_auth_key_sent_to_detailed_endpoint(self, monkeypatch):
        """With API_SERVER_KEY set, the /health/detailed probe carries the bearer
        credential so an auth-protected gateway does not log 401 warnings."""
        monkeypatch.setenv("API_SERVER_KEY", "sekrit-token")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)
        requests = self._recording_urlopen(
            monkeypatch,
            lambda: json.dumps({"status": "ok"}),
        )

        alive, body = _web_server_gateway._probe_gateway_health()

        assert alive is True
        assert requests[0] == ("http://gw:8642/health/detailed", "Bearer sekrit-token")

    def test_auth_key_never_sent_to_public_health_fallback(self, monkeypatch):
        """The credential authenticates /health/detailed only; the unauthenticated
        /health fallback must never receive it (it is the public endpoint)."""
        monkeypatch.setenv("API_SERVER_KEY", "sekrit-token")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)
        requests = self._recording_urlopen(
            monkeypatch, lambda: (_ for _ in ()).throw(ConnectionError("401"))
        )

        alive, body = _web_server_gateway._probe_gateway_health()

        assert alive is True
        assert requests[0] == ("http://gw:8642/health/detailed", "Bearer sekrit-token")
        assert requests[1] == ("http://gw:8642/health", None)

    def test_no_key_means_no_header(self, monkeypatch):
        """Without API_SERVER_KEY the probe sends no Authorization header at all."""
        monkeypatch.delenv("API_SERVER_KEY", raising=False)
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gw:8642")
        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_TIMEOUT", 1)
        requests = self._recording_urlopen(
            monkeypatch,
            lambda: json.dumps({"status": "ok"}),
        )

        alive, body = _web_server_gateway._probe_gateway_health()

        assert alive is True
        assert requests[0][1] is None
