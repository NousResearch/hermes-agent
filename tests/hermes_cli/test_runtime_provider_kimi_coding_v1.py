"""Regression test for issue #21297.

`kimi-coding` and `kimi-coding-cn` route through the Anthropic Messages
protocol when the resolved ``base_url`` matches ``api.kimi.com/coding``
(see ``_detect_api_mode_for_url``). The Anthropic SDK appends its own
``/v1/messages`` to the configured ``base_url``, so a base URL ending in
``/v1`` produced ``.../v1/v1/messages`` and HTTP 404 on every request.

The fix adds ``kimi-coding`` / ``kimi-coding-cn`` to the post-resolve
``/v1`` strip list (already in use for ``opencode-zen`` / ``opencode-go``).
"""

from types import SimpleNamespace

from hermes_cli import runtime_provider as rp


def _fake_entry(base_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        access_token="sk-kimi-test",
        source="manual",
        base_url=base_url,
    )


def test_kimi_coding_anthropic_messages_strips_trailing_v1():
    """sk-kimi key → api.kimi.com/coding/v1 base_url. /v1 must be stripped
    so the Anthropic SDK builds .../coding/v1/messages, not .../coding/v1/v1/messages."""
    resolved = rp._resolve_runtime_from_pool_entry(
        provider="kimi-coding",
        entry=_fake_entry("https://api.kimi.com/coding/v1"),
        requested_provider="kimi-coding",
        model_cfg={"default": "kimi-k2.6"},
    )

    assert resolved["provider"] == "kimi-coding"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["base_url"] == "https://api.kimi.com/coding"


def test_kimi_coding_cn_anthropic_messages_strips_trailing_v1():
    """Same fix applies to the CN endpoint when a sk-kimi key resolves to
    api.kimi.com/coding via the CN auth path."""
    resolved = rp._resolve_runtime_from_pool_entry(
        provider="kimi-coding-cn",
        entry=_fake_entry("https://api.kimi.com/coding/v1"),
        requested_provider="kimi-coding-cn",
        model_cfg={"default": "kimi-k2.6"},
    )

    assert resolved["provider"] == "kimi-coding-cn"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["base_url"] == "https://api.kimi.com/coding"


def test_kimi_coding_legacy_moonshot_base_url_not_stripped():
    """Legacy api.moonshot.ai/v1 keys go through chat_completions — the /v1
    suffix is part of the OpenAI-compat path and must NOT be stripped."""
    resolved = rp._resolve_runtime_from_pool_entry(
        provider="kimi-coding",
        entry=_fake_entry("https://api.moonshot.ai/v1"),
        requested_provider="kimi-coding",
        model_cfg={"default": "kimi-k2"},
    )

    assert resolved["provider"] == "kimi-coding"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://api.moonshot.ai/v1"
