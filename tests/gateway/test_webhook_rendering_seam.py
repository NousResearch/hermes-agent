"""Seam tests for webhook prompt rendering extraction."""

import json

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from gateway.platforms.webhook_rendering import WebhookRenderingMixin


def _adapter() -> WebhookAdapter:
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": {}},
        )
    )


def test_rendering_methods_resolve_through_webhook_adapter_mro():
    assert WebhookAdapter._render_prompt is WebhookRenderingMixin._render_prompt
    assert (
        WebhookAdapter._render_delivery_extra
        is WebhookRenderingMixin._render_delivery_extra
    )
    assert WebhookRenderingMixin in WebhookAdapter.__mro__


def test_render_prompt_supports_dot_notation_payload_access():
    adapter = _adapter()

    assert (
        adapter._render_prompt(
            "PR: {pull_request.title}",
            {"pull_request": {"title": "Ship the refactor"}},
            "pull_request",
            "github",
        )
        == "PR: Ship the refactor"
    )


def test_render_prompt_raw_payload_is_indented_json_truncated_to_4000_chars():
    adapter = _adapter()
    payload = {"message": "x" * 5000}

    rendered = adapter._render_prompt("{__raw__}", payload, "event", "route")

    assert rendered == json.dumps(payload, indent=2)[:4000]
    assert len(rendered) == 4000


def test_render_delivery_extra_renders_strings_and_preserves_non_strings():
    adapter = _adapter()
    payload = {"pull_request": {"title": "Ship it"}}
    extra = {
        "repo": "org/{pull_request.title}",
        "pr_number": 42,
        "enabled": True,
    }

    assert adapter._render_delivery_extra(extra, payload) == {
        "repo": "org/Ship it",
        "pr_number": 42,
        "enabled": True,
    }


def test_monkeypatching_public_adapter_render_prompt_affects_instances(monkeypatch):
    adapter = _adapter()

    def replacement(self, template, payload, event_type, route_name):
        return "patched"

    monkeypatch.setattr(WebhookAdapter, "_render_prompt", replacement)

    assert adapter._render_prompt("ignored", {}, "", "") == "patched"
    assert adapter._render_delivery_extra({"value": "ignored"}, {}) == {
        "value": "patched"
    }
