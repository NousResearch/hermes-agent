"""Anthropic-messages clients honour ``custom_providers[].extra_headers`` (#24293, #9721).

The OpenAI-wire clients apply the per-provider headers; ``build_anthropic_client`` used to skip
them, so a relay behind a WAF that rejects the SDK User-Agent kept 403ing in anthropic_messages mode.
Also covers the routed provider profile's ``default_headers`` (#138112): they ride the same
transport, gap-filling below the custom-provider level.
"""
from types import SimpleNamespace
from unittest.mock import patch

from agent.anthropic_adapter import build_anthropic_client

_ROUTE = "https://proxy.example.com/v1"
_CONFIG = {"custom_providers": [{
    "name": "wafproxy", "base_url": _ROUTE, "api_mode": "anthropic_messages",
    "extra_headers": {"User-Agent": "HermesAgent/1.0", "X-Privacy-Tier": "enterprise"},
}]}


def _build(route):
    with patch("agent.anthropic_adapter._require_sdk") as sdk, patch("hermes_cli.config.load_config", return_value=_CONFIG):
        build_anthropic_client("sk-test", route)
    return sdk.return_value.Anthropic.call_args.kwargs["default_headers"]


def test_matching_route_merges_extra_headers_after_betas():
    headers = _build(_ROUTE)
    assert headers["User-Agent"] == "HermesAgent/1.0"
    assert headers["X-Privacy-Tier"] == "enterprise"
    assert "anthropic-beta" in headers  # provider headers add to, not replace, the beta set


def test_other_route_does_not_inherit_extra_headers():
    headers = _build("https://other.example.com/v1")
    assert "User-Agent" not in headers and "X-Privacy-Tier" not in headers


_GW = "https://gw.example.com"
_GW_CONFIG = {"custom_providers": [{
    "name": "gwrelay", "base_url": _GW + "/v1", "api_mode": "anthropic_messages",
    "extra_headers": {"User-Agent": "HermesAgent/1.0"},
}]}


def _build_with_profiles(route, profiles):
    with patch("agent.anthropic_adapter._require_sdk") as sdk, \
         patch("hermes_cli.config.load_config", return_value=_GW_CONFIG), \
         patch("providers.list_providers", return_value=profiles):
        build_anthropic_client("sk-test", route)
    return sdk.return_value.Anthropic.call_args.kwargs["default_headers"]


def _profile(base_url, headers):
    return SimpleNamespace(base_url=base_url, default_headers=headers)


def test_profile_default_headers_ride_the_messages_transport():
    """#138112: a plugin profile with api_mode=anthropic_messages gets its default_headers on the
    wire (route keeps a trailing /v1, the profile's base_url doesn't), and custom_providers[] still win."""
    headers = _build_with_profiles(_GW + "/v1", [_profile(_GW, {
        "X-Client-Source": "hermes-plugin", "User-Agent": "plugin-agent",
    })])
    assert headers["X-Client-Source"] == "hermes-plugin"
    assert headers["User-Agent"] == "HermesAgent/1.0"  # custom_providers[].extra_headers stay the top level


def test_profile_default_headers_cannot_restore_a_dropped_credential():
    """The Omit() sentinel that keeps a single credential on the wire (#26970) survives the profile merge."""
    headers = _build_with_profiles(_GW + "/v1", [_profile(_GW, {
        "X-Client-Source": "hermes-plugin", "Authorization": "Bearer must-not-ride",
    })])
    assert headers["Authorization"] != "Bearer must-not-ride"  # the Omit() sentinel stays put
    assert headers["X-Client-Source"] == "hermes-plugin"


def test_other_route_profile_headers_do_not_apply():
    headers = _build_with_profiles("https://other.example.com/v1", [_profile(_GW, {"X-Client-Source": "hermes-plugin"})])
    assert "X-Client-Source" not in headers


def test_ambiguous_profiles_on_one_route_apply_no_headers():
    """Two profiles declaring headers on the same route is ambiguous — apply neither."""
    profiles = [
        _profile(_GW, {"X-Client-Source": "first"}),
        _profile(_GW, {"X-Client-Source": "second"}),
    ]
    headers = _build_with_profiles(_GW + "/v1", profiles)
    assert "X-Client-Source" not in headers
