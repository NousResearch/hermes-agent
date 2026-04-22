"""``_build_routing_context`` must expose ``user_id`` so ``model.routes`` can
route per-sender on trusted-identity platforms — not just per ``source_kind``.

This lets an agent owner whose messages land outside the strict DM home
channel (e.g. @mentions in a shared Discord channel) still route to their
preferred model by matching on user_id directly.

Untrusted-identity transports (webhook, api_server) must NOT expose user_id;
their inbound events can carry spoofed fields, and allowing a route match on
them would let an attacker impersonate the owner for model selection.
"""

import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _mock_dotenv(monkeypatch):
    fake = types.ModuleType("dotenv")
    fake.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", fake)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(get_home_channel=lambda _p: None)
    return runner


def _src(platform_value: str, *, user_id: str = "99", chat_id: str = "111",
         chat_type: str = "group", thread_id=None):
    plat = SimpleNamespace(value=platform_value)
    return SimpleNamespace(
        platform=plat, user_id=user_id, chat_id=chat_id,
        chat_type=chat_type, thread_id=thread_id,
    )


class TestRoutingContextUserId:
    def test_discord_group_exposes_user_id(self):
        from gateway.run import GatewayRunner
        runner = _make_runner()
        ctx = GatewayRunner._build_routing_context(
            runner, _src("discord", user_id="1417636184355766305")
        )
        assert ctx["user_id"] == "1417636184355766305"
        assert ctx["platform"] == "discord"

    def test_telegram_dm_exposes_user_id(self):
        from gateway.run import GatewayRunner
        runner = _make_runner()
        ctx = GatewayRunner._build_routing_context(
            runner, _src("telegram", user_id="42", chat_type="dm")
        )
        assert ctx["user_id"] == "42"

    def test_webhook_suppresses_user_id(self):
        from gateway.run import GatewayRunner
        runner = _make_runner()
        ctx = GatewayRunner._build_routing_context(
            runner, _src("webhook", user_id="spoofed")
        )
        assert "user_id" not in ctx
        assert ctx["platform"] == "webhook"

    def test_api_server_suppresses_user_id(self):
        from gateway.run import GatewayRunner
        runner = _make_runner()
        ctx = GatewayRunner._build_routing_context(
            runner, _src("api_server", user_id="spoofed")
        )
        assert "user_id" not in ctx

    def test_missing_user_id_is_not_in_context(self):
        from gateway.run import GatewayRunner
        runner = _make_runner()
        ctx = GatewayRunner._build_routing_context(
            runner, _src("discord", user_id="")
        )
        assert "user_id" not in ctx

    def test_none_source_returns_empty(self):
        from gateway.run import GatewayRunner
        runner = _make_runner()
        assert GatewayRunner._build_routing_context(runner, None) == {}

    def test_user_id_route_fires_end_to_end(self):
        """Integration: a Discord group message from the owner's user_id
        matches a ``user_id`` route even though ``source_kind`` is stranger."""
        from agent.smart_model_routing import apply_route
        from gateway.run import GatewayRunner

        runner = _make_runner()
        src = _src("discord", user_id="1417636184355766305", chat_type="group")
        ctx = GatewayRunner._build_routing_context(runner, src)

        cfg = {
            "default": "slate-1",
            "routes": [
                {"match": {"source_kind": "owner"}, "model": "slate-3"},
                {"match": {"platform": "discord", "user_id": "1417636184355766305"},
                 "model": "slate-3", "base_url": "https://litellm-3.int.exe.xyz/v1"},
            ],
        }
        model, runtime = apply_route("slate-1", {}, cfg, ctx)
        assert model == "slate-3"
        assert runtime.get("base_url") == "https://litellm-3.int.exe.xyz/v1"
