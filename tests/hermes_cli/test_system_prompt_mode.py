"""Tests for the system_prompt_mode custom-provider compatibility option (#76783).

Gemini-backed OpenAI-compatible relays reject Hermes's full runtime prompt as a
``system`` message with HTTP 429 RESOURCE_EXHAUSTED while identical content as a
``user`` message succeeds.  ``system_prompt_mode: user`` is the opt-in fix.
"""

from __future__ import annotations

import pytest

from hermes_cli.config import (
    _normalize_custom_provider_entry,
    get_custom_provider_system_prompt_mode,
)
from providers.base import (
    ProviderProfile,
    VALID_SYSTEM_PROMPT_MODES,
    apply_system_prompt_mode,
)


# ── apply_system_prompt_mode ────────────────────────────────────────────


def _msgs(system: str = "You are Hermes Agent.", user: str = "hi"):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def test_default_mode_is_pass_through():
    msgs = _msgs()
    assert apply_system_prompt_mode(msgs, "system") is msgs
    assert apply_system_prompt_mode(msgs, None) is msgs


def test_unknown_mode_is_pass_through():
    msgs = _msgs()
    assert apply_system_prompt_mode(msgs, "banana") is msgs


def test_valid_modes_constant():
    assert VALID_SYSTEM_PROMPT_MODES == ("system", "developer", "user")


def test_developer_mode_swaps_role_only():
    out = apply_system_prompt_mode(_msgs("BIG PROMPT"), "developer")
    assert out[0]["role"] == "developer"
    assert out[0]["content"] == "BIG PROMPT"
    assert out[1]["role"] == "user"


def test_user_mode_moves_prompt_into_first_user_message():
    out = apply_system_prompt_mode(_msgs("BIG PROMPT", "hi"), "user")
    # Short identity marker survives as a real system message.
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "BIG PROMPT"
    assert "BIG PROMPT" in out[1]["content"]
    assert out[1]["content"].startswith("[Hermes runtime instructions]")
    assert out[1]["content"].rstrip().endswith("hi")


def test_user_mode_keeps_short_identity_not_full_prompt():
    big = "You are Hermes Agent.\n" + ("x" * 20000)
    out = apply_system_prompt_mode(_msgs(big), "user")
    assert len(out[0]["content"]) <= 200
    assert out[0]["content"] == "You are Hermes Agent."


def test_user_mode_prepends_only_to_first_user_message():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
    ]
    out = apply_system_prompt_mode(msgs, "user")
    assert "SYS" in out[1]["content"]
    assert out[2]["content"] == "two"


def test_user_mode_handles_multimodal_content_list():
    msgs = [
        {"role": "system", "content": "SYS"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        },
    ]
    out = apply_system_prompt_mode(msgs, "user")
    parts = out[1]["content"]
    assert isinstance(parts, list)
    assert parts[0]["type"] == "text"
    assert "SYS" in parts[0]["text"]
    # Original parts preserved verbatim after the injected wrapper.
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "look"
    assert parts[2]["type"] == "image_url"
    assert parts[2]["image_url"] == {"url": "data:image/png;base64,AA"}


def test_user_mode_no_leading_system_is_noop():
    msgs = [{"role": "user", "content": "hi"}]
    assert apply_system_prompt_mode(msgs, "user") is msgs


def test_user_mode_empty_system_is_noop():
    msgs = [{"role": "system", "content": "   "}, {"role": "user", "content": "hi"}]
    assert apply_system_prompt_mode(msgs, "user") is msgs


def test_user_mode_no_user_message_is_noop():
    msgs = [{"role": "system", "content": "SYS"}]
    assert apply_system_prompt_mode(msgs, "user") is msgs


def test_does_not_mutate_input():
    msgs = _msgs("BIG PROMPT", "hi")
    apply_system_prompt_mode(msgs, "user")
    assert msgs[0]["content"] == "BIG PROMPT"
    assert msgs[1]["content"] == "hi"


# ── ProviderProfile hook ────────────────────────────────────────────────


def test_profile_default_mode_is_system():
    assert ProviderProfile.system_prompt_mode == "system"


def test_profile_prepare_messages_noop_for_default():
    p = ProviderProfile(name="x", base_url="http://x")
    msgs = _msgs("BIG PROMPT")
    assert p.prepare_messages(msgs) is msgs


def test_profile_prepare_messages_applies_user_mode():
    p = ProviderProfile(name="x", base_url="http://x", system_prompt_mode="user")
    msgs = _msgs("BIG PROMPT", "hi")
    out = p.prepare_messages(msgs)
    assert "[Hermes runtime instructions]" in out[1]["content"]
    assert len(out[0]["content"]) <= 200


# ── Config resolution ───────────────────────────────────────────────────


def test_normalize_keeps_valid_mode():
    entry = _normalize_custom_provider_entry(
        {"name": "Relay", "base_url": "http://localhost:8317/v1", "system_prompt_mode": "USER"}
    )
    assert entry["system_prompt_mode"] == "user"


def test_normalize_drops_invalid_mode():
    entry = _normalize_custom_provider_entry(
        {"name": "Relay", "base_url": "http://localhost:8317/v1", "system_prompt_mode": "bogus"}
    )
    assert "system_prompt_mode" not in entry


def test_resolve_returns_provider_level_mode():
    providers = [
        {"base_url": "http://localhost:8317/v1", "system_prompt_mode": "user"},
    ]
    assert (
        get_custom_provider_system_prompt_mode(
            "gemini-3.1-pro-low", "http://localhost:8317/v1", providers
        )
        == "user"
    )


def test_resolve_model_scoped_mode_wins():
    providers = [
        {
            "base_url": "http://localhost:8317/v1",
            "system_prompt_mode": "system",
            "models": {"gemini-3.1-pro-low": {"system_prompt_mode": "user"}},
        },
    ]
    assert (
        get_custom_provider_system_prompt_mode(
            "gemini-3.1-pro-low", "http://localhost:8317/v1", providers
        )
        == "user"
    )
    assert (
        get_custom_provider_system_prompt_mode(
            "other-model", "http://localhost:8317/v1", providers
        )
        == "system"
    )


def test_resolve_unmatched_url_defaults_to_system():
    providers = [{"base_url": "http://other/v1", "system_prompt_mode": "user"}]
    assert (
        get_custom_provider_system_prompt_mode(
            "m", "http://localhost:8317/v1", providers
        )
        == "system"
    )
    assert get_custom_provider_system_prompt_mode("m", "", providers) == "system"
    assert get_custom_provider_system_prompt_mode("m", "http://x", None) == "system"
    assert get_custom_provider_system_prompt_mode("m", "http://x", []) == "system"


def test_resolve_matches_url_with_trailing_slash_and_case():
    providers = [{"base_url": "http://LocalHost:8317/v1/", "system_prompt_mode": "user"}]
    assert (
        get_custom_provider_system_prompt_mode(
            "m", "http://localhost:8317/v1", providers
        )
        == "user"
    )
