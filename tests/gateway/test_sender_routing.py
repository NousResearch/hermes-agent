"""Tests for per-sender profile routing helper (_resolve_sender_profile)."""
import pytest
from gateway.run import _resolve_sender_profile


def test_empty_config_returns_none():
    assert _resolve_sender_profile("whatsapp", "573104851803", {}) is None


def test_empty_sender_profiles_returns_none():
    config = {"routing": {"sender_profiles": {}}}
    assert _resolve_sender_profile("whatsapp", "573104851803", config) is None


def test_matching_key_returns_profile():
    config = {"routing": {"sender_profiles": {"whatsapp:573104851803": "paco"}}}
    assert _resolve_sender_profile("whatsapp", "573104851803", config) == "paco"


def test_normalization_strips_plus():
    config = {"routing": {"sender_profiles": {"whatsapp:573104851803": "paco"}}}
    assert _resolve_sender_profile("whatsapp", "+573104851803", config) == "paco"


def test_normalization_strips_whatsapp_domain():
    config = {"routing": {"sender_profiles": {"whatsapp:573104851803": "paco"}}}
    assert _resolve_sender_profile("whatsapp", "573104851803@s.whatsapp.net", config) == "paco"


def test_normalization_strips_lid_domain():
    config = {"routing": {"sender_profiles": {"whatsapp:573104851803": "paco"}}}
    assert _resolve_sender_profile("whatsapp", "573104851803@lid", config) == "paco"


def test_nonmatching_sender_returns_none():
    config = {"routing": {"sender_profiles": {"whatsapp:573104851803": "paco"}}}
    assert _resolve_sender_profile("whatsapp", "999999999", config) is None


def test_platform_scoping_whatsapp_key_does_not_match_discord():
    config = {"routing": {"sender_profiles": {"whatsapp:573104851803": "paco"}}}
    assert _resolve_sender_profile("discord", "573104851803", config) is None


def test_discord_routing():
    config = {"routing": {"sender_profiles": {"discord:384435881706127380": "work"}}}
    assert _resolve_sender_profile("discord", "384435881706127380", config) == "work"


def test_telegram_routing():
    config = {"routing": {"sender_profiles": {"telegram:123456789": "myprofile"}}}
    assert _resolve_sender_profile("telegram", "123456789", config) == "myprofile"


def test_empty_sender_returns_none():
    config = {"routing": {"sender_profiles": {"whatsapp:573104851803": "paco"}}}
    assert _resolve_sender_profile("whatsapp", "", config) is None


def test_multiple_mappings():
    config = {
        "routing": {
            "sender_profiles": {
                "whatsapp:573104851803": "paco",
                "discord:384435881706127380": "work",
            }
        }
    }
    assert _resolve_sender_profile("whatsapp", "573104851803", config) == "paco"
    assert _resolve_sender_profile("discord", "384435881706127380", config) == "work"
    assert _resolve_sender_profile("telegram", "573104851803", config) is None


def test_none_routing_key_returns_none():
    config = {"routing": None}
    assert _resolve_sender_profile("whatsapp", "573104851803", config) is None


def test_missing_routing_key_returns_none():
    config = {"model": {"default": "sonnet"}}
    assert _resolve_sender_profile("whatsapp", "573104851803", config) is None
