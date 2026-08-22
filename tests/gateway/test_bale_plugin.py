"""Contract tests for the bundled Bale platform plugin."""

from __future__ import annotations

import httpx
import pytest

from plugins.platforms.bale import adapter as bale_adapter
from plugins.platforms.bale import bale_ids
from plugins.platforms.bale import bale_network


class _PluginContext:
    """Capture the platform contract without loading the full plugin manager."""

    def __init__(self) -> None:
        self.platform: dict | None = None

    def register_platform(self, **kwargs) -> None:
        self.platform = kwargs


class TestBalePluginRegistration:
    def test_registers_bale_with_gateway_hooks(self):
        context = _PluginContext()

        bale_adapter.register(context)

        assert context.platform is not None
        assert context.platform["name"] == "bale"
        assert context.platform["label"] == "Bale"
        assert context.platform["required_env"] == ["BALE_BOT_TOKEN"]
        assert context.platform["allowed_users_env"] == "BALE_ALLOWED_USERS"
        assert context.platform["allow_all_env"] == "BALE_ALLOW_ALL_USERS"
        assert callable(context.platform["adapter_factory"])
        assert callable(context.platform["standalone_sender_fn"])
        assert callable(context.platform["apply_yaml_config_fn"])


class TestBaleChatIds:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(1234, 1234), ("-1001234", -1001234), (" @public_channel ", "@public_channel")],
    )
    def test_normalizes_numeric_and_username_chat_ids(self, raw, expected):
        assert bale_ids.normalize_bale_chat_id(raw) == expected

    def test_username_target_is_recognized_without_coercion(self):
        assert bale_ids.parse_bale_username_target(" @public_channel ") == "@public_channel"
        assert bale_ids.parse_bale_username_target("-1001234") is None


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, calls: list[str], behavior: dict[str, str]) -> None:
        self.calls = calls
        self.behavior = behavior

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.calls.append(host)
        if self.behavior.get(host) == "fail":
            raise httpx.ConnectError("unreachable")
        return httpx.Response(200, request=request)

    async def aclose(self) -> None:
        pass


class TestBaleFallbackNetwork:
    def test_rewrite_keeps_bale_host_and_tls_sni(self):
        request = httpx.Request("GET", "https://tapi.bale.ai/botTOKEN/getMe")

        rewritten = bale_network._rewrite_request_for_ip(request, "2.189.68.126")

        assert rewritten.url.host == "2.189.68.126"
        assert rewritten.headers["host"] == "tapi.bale.ai"
        assert rewritten.extensions["sni_hostname"] == "tapi.bale.ai"

    @pytest.mark.asyncio
    async def test_falls_back_to_hostname_after_ipv4_connect_failure(self, monkeypatch):
        calls: list[str] = []

        def factory(**_kwargs):
            return _FakeTransport(calls, {"2.189.68.126": "fail"})

        monkeypatch.setattr(bale_network, "_resolve_proxy_url", lambda **_kwargs: None)
        monkeypatch.setattr(bale_network.httpx, "AsyncHTTPTransport", factory)
        transport = bale_network.BaleFallbackTransport(["2.189.68.126"])

        response = await transport.handle_async_request(
            httpx.Request("GET", "https://tapi.bale.ai/botTOKEN/getMe")
        )

        assert response.status_code == 200
        assert calls == ["2.189.68.126", "tapi.bale.ai"]
