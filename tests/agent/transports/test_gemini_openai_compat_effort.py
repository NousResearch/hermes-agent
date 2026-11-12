"""Google's OpenAI-compat shim rejects ``xhigh``/``max`` with HTTP 400 "Invalid reasoning_effort".

Every build path (a provider profile, the legacy no-profile path, ``request_overrides``) must
clamp the effort it sends there. Other hosts keep the requested effort.
"""
import pytest

from agent.transports import get_transport
from providers import get_provider_profile

GOOGLE_COMPAT = "https://generativelanguage.googleapis.com/v1beta/openai"
_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture
def transport():
    return get_transport("chat_completions")


def _effort(kw):
    top = kw.get("reasoning_effort")
    nested = ((kw.get("extra_body") or {}).get("reasoning") or {}).get("effort")
    return top or nested


@pytest.mark.parametrize("requested,expected", [("max", "high"), ("xhigh", "high"), ("high", "high"),
                                                ("low", "low"), ("minimal", "minimal")])
def test_custom_profile_clamps_top_level_effort(transport, requested, expected):
    kw = transport.build_kwargs(
        model="gemini-3-flash-preview", messages=_MSGS, provider_profile=get_provider_profile("custom"),
        provider_name="custom", base_url=GOOGLE_COMPAT, supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": requested},
    )
    assert kw["reasoning_effort"] == expected


def test_legacy_path_clamps_extra_body_reasoning(transport):
    kw = transport.build_kwargs(
        model="gemini-3-flash-preview", messages=_MSGS, base_url=GOOGLE_COMPAT, supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "max"},
    )
    assert kw["extra_body"]["reasoning"]["effort"] == "high"


def test_request_override_is_clamped_too(transport):
    kw = transport.build_kwargs(
        model="gemini-3-flash-preview", messages=_MSGS, provider_profile=get_provider_profile("custom"),
        provider_name="custom", base_url=GOOGLE_COMPAT, request_overrides={"reasoning_effort": "max"},
    )
    assert kw["reasoning_effort"] == "high"


@pytest.mark.parametrize("base_url", ["https://api.example.com/v1", "https://generativelanguage.googleapis.com/v1beta"])
def test_other_hosts_keep_the_requested_effort(transport, base_url):
    kw = transport.build_kwargs(
        model="some-model", messages=_MSGS, provider_profile=get_provider_profile("custom"),
        provider_name="custom", base_url=base_url, supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "max"},
    )
    assert _effort(kw) in (None, "max")
    if base_url.endswith("/v1"):
        assert kw["reasoning_effort"] == "max"

@pytest.mark.parametrize("with_profile", [False, True])
def test_clamp_does_not_mutate_reused_nested_overrides(transport, with_profile):
    overrides = {"extra_body": {"reasoning": {"effort": "max"}}}
    profile = (
        {"provider_profile": get_provider_profile("custom"), "provider_name": "custom"}
        if with_profile else {}
    )
    google = transport.build_kwargs(
        model="gemini-3-flash-preview", messages=_MSGS, base_url=GOOGLE_COMPAT,
        request_overrides=overrides, **profile,
    )
    assert google["extra_body"]["reasoning"]["effort"] == "high"
    assert overrides["extra_body"]["reasoning"]["effort"] == "max"

    # A shared override must not leak Google's narrower effort limit to another host.
    other = transport.build_kwargs(
        model="some-model", messages=_MSGS, base_url="https://api.example.com/v1",
        request_overrides=overrides, **profile,
    )
    assert other["extra_body"]["reasoning"]["effort"] == "max"
