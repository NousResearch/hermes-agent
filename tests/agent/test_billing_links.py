"""Tests for provider-agnostic billing recovery links (agent/billing_links.py).

Behavior/invariant tests — no snapshotting of the exact URL strings beyond the
few that are the whole point of the mapping (the host they must land on).
"""

from __future__ import annotations

from agent.billing_links import (
    BillingBlock,
    build_billing_block,
    is_nous_inference_route,
)


def test_nous_route_by_provider_slug():
    block = build_billing_block(provider="nous", base_url="", model="hermes-4")
    assert block.is_nous is True
    assert block.provider_label == "Nous Portal"
    # Nous always resolves an in-app/portal billing URL as a fallback.
    assert block.billing_url and "nousresearch.com" in block.billing_url


def test_nous_route_by_base_url_host():
    block = build_billing_block(
        provider="openai_compatible",
        base_url="https://inference-api.nousresearch.com/v1",
        model="hermes-4",
    )
    assert block.is_nous is True


def test_is_nous_inference_route_helper():
    assert is_nous_inference_route("nous", "") is True
    assert is_nous_inference_route("", "https://inference-api.nousresearch.com/v1") is True
    assert is_nous_inference_route("openai", "https://api.openai.com/v1") is False


def test_known_provider_by_slug_resolves_label_and_url():
    block = build_billing_block(provider="openai", base_url="", model="gpt-5")
    assert block.is_nous is False
    assert block.provider_label == "OpenAI"
    assert block.billing_url is not None
    assert "openai.com" in block.billing_url


def test_openrouter_resolves_credits_page():
    block = build_billing_block(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude",
    )
    assert block.is_nous is False
    assert block.billing_url is not None
    assert "openrouter.ai" in block.billing_url


def test_unknown_provider_via_base_url_host_fallback():
    # Provider slug is a generic bucket; the host reveals the real upstream.
    block = build_billing_block(
        provider="custom",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    assert block.provider_label == "DeepSeek"
    assert block.billing_url is not None
    assert "deepseek.com" in block.billing_url


def test_unknown_provider_degrades_without_url():
    block = build_billing_block(
        provider="my_local_llm",
        base_url="http://localhost:1234/v1",
        model="llama",
    )
    assert block.is_nous is False
    # No invented URL for an unknown provider — but a readable label survives.
    assert block.billing_url is None
    assert block.provider_label  # non-empty, humanized


def test_message_is_carried_through_unchanged():
    block = build_billing_block(
        provider="openai",
        base_url="",
        model="gpt-5",
        message="You are out of credits.",
    )
    assert block.message == "You are out of credits."


def test_to_dict_round_trips_all_fields():
    block = build_billing_block(provider="openai", base_url="", model="gpt-5")
    data = block.to_dict()
    assert set(data) == {
        "provider",
        "provider_label",
        "model",
        "billing_url",
        "is_nous",
        "message",
    }
    assert isinstance(block, BillingBlock)
