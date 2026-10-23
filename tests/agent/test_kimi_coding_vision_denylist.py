"""Regression tests for #18990: kimi-coding must not be provider-denylisted for vision.

Live verification (2026-10-08, Kimi Coding Plan key): every model currently served on
https://api.kimi.com/coding (k3, k3-256k, kimi-for-coding, kimi-for-coding-highspeed)
accepts image input on BOTH the OpenAI chat-completions wire (image_url part) and the
Anthropic messages wire (image block). The blanket _PROVIDERS_WITHOUT_VISION entry
predated that support and made vision auto-detect skip the main provider by name
BEFORE any per-model capability check, so even supports_vision: true overrides could
not rescue it.

kimi-coding-cn (api.moonshot.cn) stays excluded: it is a different key surface and
has not had the same live verification.
"""

from __future__ import annotations

import agent.auxiliary_client as aux


def test_kimi_coding_removed_from_providers_without_vision():
    assert "kimi-coding" not in aux._PROVIDERS_WITHOUT_VISION


def test_kimi_coding_cn_stays_excluded_pending_live_verification():
    assert "kimi-coding-cn" in aux._PROVIDERS_WITHOUT_VISION


def test_vision_auto_detect_no_longer_skips_kimi_coding_by_provider_name(monkeypatch):
    """The per-model capability gate (mocked capable here) must now decide, not the name."""
    fake_client = object()
    monkeypatch.setattr(aux, "_main_model_supports_vision", lambda *a, **k: True)
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **k: (fake_client, "k3"))

    client, model = aux._vision_main_provider_client("kimi-coding", "k3", {}, None, None)

    assert client is fake_client
    assert model == "k3"


def test_vision_auto_detect_still_skips_kimi_coding_cn(monkeypatch):
    def _must_not_be_reached(*a, **k):
        raise AssertionError("resolve_provider_client reached for an excluded provider")

    monkeypatch.setattr(aux, "resolve_provider_client", _must_not_be_reached)

    client, model = aux._vision_main_provider_client("kimi-coding-cn", "k3", {}, None, None)

    assert client is None
    assert model is None
