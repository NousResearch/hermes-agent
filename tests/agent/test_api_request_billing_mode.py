"""Billing classification crosses the core hook boundary without credentials."""

from types import SimpleNamespace

import pytest

from agent.turn_response_intake import _fire_post_api_request_hook
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
from hermes_cli.plugins import get_plugin_manager


@pytest.mark.parametrize(
    "api_key,base_url,billing_mode",
    [
        ("sk-ant-oat01-test", None, "subscription_included"),
        ("sk-ant-oat01-test", "https://api.anthropic.com/v1", "subscription_included"),
        ("sk-ant-api03-test", "https://api.anthropic.com/v1", "official_docs_snapshot"),
        ("sk-ant-oat01-test", "https://proxy.example/v1", "official_docs_snapshot"),
        (None, None, "official_docs_snapshot"),
    ],
    ids=["oauth-default", "oauth-official", "api-key", "oauth-proxy", "no-credential"],
)
def test_post_api_request_delivers_billing_mode_to_external_plugin(
    tmp_path, monkeypatch, api_key, base_url, billing_mode,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [billing-probe]\n")
    plugin_dir = tmp_path / "plugins" / "billing-probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: billing-probe\nversion: 0.1.0\n")
    (plugin_dir / "__init__.py").write_text(
        "received = []\n"
        "legacy_models = []\n"
        "def capture(*, billing_mode, usage, **kwargs):\n"
        "    received.append((billing_mode, usage, kwargs))\n"
        "def legacy(*, model):\n"
        "    legacy_models.append(model)\n"
        "def register(ctx):\n"
        "    ctx.register_hook('post_api_request', capture)\n"
        "    ctx.register_hook('post_api_request', legacy)\n"
    )
    manager = get_plugin_manager()
    manager.discover_and_load()
    loaded = manager._plugins["billing-probe"]
    assert loaded.enabled, loaded.error
    plugin = loaded.module
    usage = {"input_tokens": 100, "output_tokens": 50}
    agent = SimpleNamespace(
        model="claude-sonnet-5", provider="anthropic", base_url=base_url,
        api_key=api_key, api_mode="anthropic_messages", session_id="billing-session",
        platform="cli", client=None,
        _api_response_payload_for_hook=lambda *args, **kwargs: {"model": "claude-sonnet-5"},
        _usage_summary_for_api_request_hook=lambda response: usage,
    )
    try:
        _fire_post_api_request_hook(
            agent, SimpleNamespace(model=agent.model),
            SimpleNamespace(content="done", tool_calls=[]), "stop",
            api_messages=[], api_call_count=1, api_duration=0.1, api_start_time=1.0,
            api_request_id="request-1", effective_task_id="task-1", turn_id="turn-1",
        )

        assert len(plugin.received) == 1
        mode, received_usage, payload = plugin.received[0]
        assert mode == billing_mode
        assert received_usage == usage
        assert "api_key" not in payload
        assert plugin.legacy_models == [agent.model]

        cost = estimate_usage_cost(
            agent.model, CanonicalUsage(**usage), provider=agent.provider,
            base_url=base_url, api_key=api_key,
        )
        if billing_mode == "subscription_included":
            assert cost.status == "included"
            assert cost.amount_usd == 0
        else:
            assert cost.status == "estimated"
            assert cost.amount_usd > 0
    finally:
        manager.unload("billing-probe")
