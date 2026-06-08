"""Regression tests for Kimi CN /model routing."""

from hermes_cli import model_switch
from hermes_cli.providers import get_label, get_provider


def test_kimi_cn_provider_overlay_has_china_label_and_endpoint():
    """The explicit CN provider must not collapse to the shared models.dev label/URL."""
    pdef = get_provider("kimi-coding-cn")

    assert pdef is not None
    assert get_label("kimi-coding-cn") == "Kimi / Moonshot (China)"
    assert pdef.base_url == "https://api.moonshot.cn/v1"
    assert "KIMI_CN_API_KEY" in pdef.api_key_env_vars


def test_exact_kimi_model_prefers_authenticated_cn_provider(monkeypatch):
    """Bare `/model kimi-k2.6` should use authenticated CN before static global fallback."""
    model_switch.DIRECT_ALIASES.clear()
    monkeypatch.setattr(model_switch, "_load_direct_aliases", lambda: {})
    monkeypatch.setattr(
        model_switch,
        "get_authenticated_provider_slugs",
        lambda **_kwargs: ["kimi-coding-cn"],
    )

    # If the authenticated-provider preference fails, the old last-resort static
    # catalog detection would choose the global provider first.
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda model, _current: ("kimi-coding", model),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested, target_model: {
            "provider": requested,
            "api_key": "sk-test",
            "base_url": "https://api.moonshot.cn/v1"
            if requested == "kimi-coding-cn"
            else "https://api.kimi.com/coding/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "persist": True,
            "recognized": True,
        },
    )

    res = model_switch.switch_model(
        "kimi-k2.6",
        current_provider="openai-codex",
        current_model="gpt-5.5",
        current_base_url="https://chatgpt.com/backend-api/codex",
    )

    assert res.success is True
    assert res.target_provider == "kimi-coding-cn"
    assert res.provider_label == "Kimi / Moonshot (China)"
    assert res.base_url == "https://api.moonshot.cn/v1"
