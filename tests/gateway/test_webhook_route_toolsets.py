"""Per-route webhook toolset overrides (adapter.toolsets_for_source).

A webhook route config may carry a ``toolsets`` list that replaces the
platform-level ``platform_toolsets.webhook`` resolution for runs triggered by
that route only. The gateway validates the override through the same
``_get_platform_tools`` path as platform config, so restricted/unknown names
behave identically to a manually configured platform toolset list.
"""

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.webhook import WebhookAdapter
from gateway.run import GatewayRunner
from hermes_cli.tools_config import _get_platform_tools


class _Src:
    def __init__(self, chat_id):
        self.chat_id = chat_id


def _make_adapter(routes):
    wa = object.__new__(WebhookAdapter)
    wa._routes = routes
    return wa


def _make_runner(adapter):
    gr = object.__new__(GatewayRunner)
    gr._adapter_for_source = lambda source: adapter
    return gr


BASE_CONFIG = {"platform_toolsets": {"webhook": ["web", "vision", "clarify"]}}


class TestWebhookAdapterToolsetsForSource:
    def test_route_with_toolsets_returns_list(self):
        wa = _make_adapter({"mon": {"secret": "x", "toolsets": ["terminal", "file"]}})
        assert wa.toolsets_for_source(_Src("webhook:mon:d1")) == ["terminal", "file"]

    def test_route_without_toolsets_returns_none(self):
        wa = _make_adapter({"plain": {"secret": "x"}})
        assert wa.toolsets_for_source(_Src("webhook:plain:d1")) is None

    def test_unknown_route_returns_none(self):
        wa = _make_adapter({})
        assert wa.toolsets_for_source(_Src("webhook:ghost:d1")) is None

    def test_non_webhook_chat_id_returns_none(self):
        wa = _make_adapter({"mon": {"secret": "x", "toolsets": ["terminal"]}})
        assert wa.toolsets_for_source(_Src("telegram:123")) is None

    def test_empty_or_non_list_toolsets_returns_none(self):
        wa = _make_adapter(
            {
                "empty": {"secret": "x", "toolsets": []},
                "str": {"secret": "x", "toolsets": "terminal"},
                "blank": {"secret": "x", "toolsets": ["  ", ""]},
            }
        )
        assert wa.toolsets_for_source(_Src("webhook:empty:d")) is None
        assert wa.toolsets_for_source(_Src("webhook:str:d")) is None
        assert wa.toolsets_for_source(_Src("webhook:blank:d")) is None

    def test_base_adapter_default_is_none(self):
        # Non-webhook adapters inherit a None default: no override anywhere.
        wa = _make_adapter({})
        assert (
            BasePlatformAdapter.toolsets_for_source(wa, _Src("webhook:mon:d")) is None
        )


class TestGatewayResolveEnabledToolsetsForSource:
    def test_override_replaces_platform_resolution(self):
        wa = _make_adapter(
            {"mon": {"secret": "x", "toolsets": ["terminal", "file", "web"]}}
        )
        gr = _make_runner(wa)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src("webhook:mon:d"), "webhook"
        )
        assert "terminal" in res and "file" in res and "web" in res
        assert "vision" not in res  # platform list fully replaced, not merged

    def test_override_validated_like_platform_config(self):
        # Contract: resolving with an override is byte-identical to resolving
        # the same list configured as platform_toolsets.webhook.
        override = ["terminal", "file", "web", "discord_admin"]
        wa = _make_adapter({"mon": {"secret": "x", "toolsets": override}})
        gr = _make_runner(wa)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src("webhook:mon:d"), "webhook"
        )
        expected = sorted(
            _get_platform_tools(
                {"platform_toolsets": {"webhook": list(override)}}, "webhook"
            )
        )
        assert res == expected
        # discord_admin is platform-restricted to discord — must be dropped.
        assert "discord_admin" not in res

    def test_no_override_uses_platform_resolution(self):
        wa = _make_adapter({"plain": {"secret": "x"}})
        gr = _make_runner(wa)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src("webhook:plain:d"), "webhook"
        )
        assert res == sorted(_get_platform_tools(BASE_CONFIG, "webhook"))
        assert "terminal" not in res

    def test_adapter_exception_falls_back_to_platform_resolution(self):
        wa = _make_adapter({})
        wa.toolsets_for_source = lambda source: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        gr = _make_runner(wa)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src("webhook:mon:d"), "webhook"
        )
        assert res == sorted(_get_platform_tools(BASE_CONFIG, "webhook"))

    def test_missing_adapter_falls_back_to_platform_resolution(self):
        gr = _make_runner(None)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src("webhook:mon:d"), "webhook"
        )
        assert res == sorted(_get_platform_tools(BASE_CONFIG, "webhook"))

    def test_original_config_not_mutated(self):
        cfg = {"platform_toolsets": {"webhook": ["web"]}}
        wa = _make_adapter({"mon": {"secret": "x", "toolsets": ["terminal"]}})
        gr = _make_runner(wa)
        GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, cfg, _Src("webhook:mon:d"), "webhook"
        )
        assert cfg["platform_toolsets"]["webhook"] == ["web"]
