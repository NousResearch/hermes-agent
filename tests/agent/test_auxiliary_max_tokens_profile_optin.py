"""A provider profile opts its wire into auxiliary ``max_tokens`` forwarding (#139778).

Pinned through the real ``_build_call_kwargs`` entry point: opt-in forwards, the default drops.
"""

from __future__ import annotations

import pytest

import providers as _providers
from providers import register_provider
from providers.base import ProviderProfile

MSGS = [{"role": "user", "content": "summarize"}]


def _cap(kwargs: dict) -> dict:
    return {
        k: kwargs[k] for k in ("max_tokens", "max_completion_tokens") if k in kwargs
    }


@pytest.fixture
def registry(monkeypatch):
    """Snapshot both registries so a test-time registration never leaks."""
    _providers._discover_providers()
    monkeypatch.setattr(_providers, "_REGISTRY", dict(_providers._REGISTRY))
    monkeypatch.setattr(_providers, "_ALIASES", dict(_providers._ALIASES))
    monkeypatch.setattr(_providers, "_PROVIDER_LIST_CACHE", None)


def _build(provider: str, base: str) -> dict:
    from agent.auxiliary_client import _build_call_kwargs

    return _build_call_kwargs(
        provider, "some-model", MSGS, max_tokens=800, base_url=base, task="compression"
    )


def test_opted_in_process_profile_forwards_the_cap(registry):
    register_provider(
        ProviderProfile(
            name="claude-subscription-directsdk-experimental",
            auth_type="external_process",
            base_url="process://claude-subscription-directsdk-experimental",
            forwards_max_tokens=True,
        )
    )
    kwargs = _build(
        "claude-subscription-directsdk-experimental",
        "process://claude-subscription-directsdk-experimental",
    )
    assert _cap(kwargs) == {"max_tokens": 800}


def test_default_process_profile_still_drops_the_cap(registry):
    """Opt-in, not a blanket external_process rule: an unverified subprocess keeps the cap off."""
    register_provider(
        ProviderProfile(
            name="probe-acp-unverified",
            auth_type="external_process",
            base_url="process://probe-acp-unverified",
        )
    )
    kwargs = _build("probe-acp-unverified", "process://probe-acp-unverified")
    assert _cap(kwargs) == {}
