"""Manual `hermes cron run` forwarding for relay-fronted delivery targets.

A standalone CLI process has no live relay adapter and no standalone sender,
so a manual run that targets a relay-fronted platform must forward to the
running gateway (whose live relay adapter owns that delivery) rather than
execute in-process and hit the native standalone fallback.
"""

import json
from unittest.mock import patch

from tools import cronjob_tools


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.text = ""


class TestRelayFrontedDeliveryPlatforms:
    def test_empty_when_nothing_fronted(self):
        with patch("gateway.relay.relay_fronted_platforms", return_value=set()):
            assert cronjob_tools._relay_fronted_delivery_platforms({"id": "j1"}) == set()

    def test_detects_fronted_delivery_platform(self):
        job = {"id": "j1", "deliver": "discord"}
        with patch("gateway.relay.relay_fronted_platforms", return_value={"discord"}), patch(
            "cron.scheduler._resolve_delivery_targets",
            return_value=[{"platform": "discord", "chat_id": "123"}],
        ):
            assert cronjob_tools._relay_fronted_delivery_platforms(job) == {"discord"}

    def test_ignores_non_fronted_platform(self):
        job = {"id": "j1", "deliver": "discord"}
        with patch("gateway.relay.relay_fronted_platforms", return_value={"telegram"}), patch(
            "cron.scheduler._resolve_delivery_targets",
            return_value=[{"platform": "discord", "chat_id": "123"}],
        ):
            assert cronjob_tools._relay_fronted_delivery_platforms(job) == set()


class TestForwardRelayFrontedRun:
    def test_none_on_native_topology(self):
        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value=set()
        ):
            assert cronjob_tools._forward_relay_fronted_run({"id": "j1"}) is None

    def test_forwards_on_success(self):
        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", return_value=_Resp(200)):
            out = json.loads(cronjob_tools._forward_relay_fronted_run({"id": "j1"}))
        assert out["success"] is True
        assert out["forwarded_to_gateway"] is True

    def test_errors_when_gateway_unreachable(self):
        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", side_effect=Exception("down")):
            out = json.loads(cronjob_tools._forward_relay_fronted_run({"id": "j1"}))
        assert out["success"] is False
        assert "relay-fronted" in out["error"]

    def test_errors_on_gateway_4xx(self):
        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", return_value=_Resp(401)):
            out = json.loads(cronjob_tools._forward_relay_fronted_run({"id": "j1"}))
        assert out["success"] is False
        assert "relay-fronted" in out["error"]

    def test_posts_to_run_route_with_bearer(self):
        sent = {}

        def fake_post(url, headers=None, timeout=None):
            sent["url"] = url
            sent["headers"] = headers
            return _Resp(200)

        with patch.object(
            cronjob_tools, "_relay_fronted_delivery_platforms", return_value={"discord"}
        ), patch("httpx.post", side_effect=fake_post), patch(
            "agent.secret_scope.get_secret", return_value="secret-key-16chars"
        ):
            cronjob_tools._forward_relay_fronted_run({"id": "abc123"})
        assert sent["url"].endswith("/api/jobs/abc123/run")
        assert sent["headers"]["Authorization"] == "Bearer secret-key-16chars"
