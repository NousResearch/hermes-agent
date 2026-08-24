"""Router catalog-declared reasoning-effort clamping on the codex transport.

Ramp Router (api.router.com) validates ``reasoning.effort`` against each
model's published vocabulary — HTTP 400 ``invalid-argument`` on an
unsupported level, and 400 ``unsupported_parameter`` when a non-reasoning
model receives any reasoning field (both verified live, Aug 2026). The
router profile declares each model's vocabulary from its cached catalog via
``ProviderProfile.supported_reasoning_efforts``; these tests pin how the
codex transport consumes that declaration.

All tests seed the plugin's in-memory cache directly — no network.
"""

import sys

import pytest

from agent.transports import get_transport


def _router_plugin_module():
    from providers import get_provider_profile

    profile = get_provider_profile("router")
    assert profile is not None, "router profile must be registered"
    return profile, sys.modules[type(profile).__module__]


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401
    return get_transport("codex_responses")


@pytest.fixture
def seeded_catalog(monkeypatch):
    """Seed the router efforts cache with catalog-shaped verdicts."""
    profile, mod = _router_plugin_module()
    monkeypatch.setattr(mod, "_efforts_cache", {
        # grok via Router: no "none", no "max" (live catalog shape)
        "grok-4.6": ["minimal", "low", "medium", "high", "xhigh"],
        # non-reasoning model: any reasoning field 400s
        "gpt-4.1-mini": [],
        # full ladder including max
        "accounts/fireworks/models/kimi-k3": [
            "minimal", "low", "medium", "high", "xhigh", "max",
        ],
    })
    monkeypatch.setattr(mod, "_disk_checked", True)
    return profile


class TestProfileContract:
    def test_declared_vocabulary(self, seeded_catalog):
        assert seeded_catalog.supported_reasoning_efforts("grok-4.6") == (
            "minimal", "low", "medium", "high", "xhigh",
        )

    def test_non_reasoning_model_is_definitive_empty(self, seeded_catalog):
        assert seeded_catalog.supported_reasoning_efforts("gpt-4.1-mini") == ()

    def test_unknown_model_is_none(self, seeded_catalog):
        assert seeded_catalog.supported_reasoning_efforts("some-byok-route") is None

    def test_cold_cache_is_none_and_never_blocks(self, monkeypatch):
        profile, mod = _router_plugin_module()
        monkeypatch.setattr(mod, "_efforts_cache", None)
        monkeypatch.setattr(mod, "_disk_checked", True)
        monkeypatch.setattr(mod, "_warm_efforts_async", lambda: None)
        assert profile.supported_reasoning_efforts("grok-4.6") is None

    def test_parse_efforts_catalog_shapes(self):
        _, mod = _router_plugin_module()
        parsed = mod._parse_efforts([
            {
                "id": "grok-4.6",
                "router": {"capabilities": {"reasoning": {
                    "supported": True,
                    "efforts": [{"value": "low"}, {"value": "high"}],
                }}},
            },
            {
                "id": "gpt-4.1",
                "router": {"capabilities": {"reasoning": {"supported": False, "efforts": []}}},
            },
            # reasoning supported but vocabulary unpublished -> omitted (unknown)
            {
                "id": "mystery-model",
                "router": {"capabilities": {"reasoning": {"supported": True, "efforts": []}}},
            },
            # no router metadata at all -> omitted
            {"id": "bare-model"},
        ])
        assert parsed == {"grok-4.6": ["low", "high"], "gpt-4.1": []}


class TestTransportClamp:
    def _kwargs(self, transport, model, reasoning_config=None):
        return transport.build_kwargs(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url="https://api.router.com/v1",
            session_id="sid",
            provider="router",
            reasoning_config=reasoning_config,
        )

    def test_clamps_to_catalog_vocabulary(self, transport, seeded_catalog):
        # grok-4.6 via Router has no "max" — nearest weaker supported is xhigh.
        kw = self._kwargs(transport, "grok-4.6", {"effort": "max"})
        assert kw["reasoning"]["effort"] == "xhigh"

    def test_supported_effort_passes_through(self, transport, seeded_catalog):
        kw = self._kwargs(
            transport, "accounts/fireworks/models/kimi-k3", {"effort": "max"}
        )
        assert kw["reasoning"]["effort"] == "max"

    def test_non_reasoning_model_suppresses_reasoning(self, transport, seeded_catalog):
        # Default reasoning_config is enabled — the () verdict must strip the
        # reasoning field entirely (Router 400s rather than ignoring it).
        kw = self._kwargs(transport, "gpt-4.1-mini")
        assert "reasoning" not in kw
        assert kw.get("include") == []

    def test_unknown_model_falls_back_to_codex_default(self, transport, seeded_catalog):
        # Not in the catalog -> default codex vocabulary applies (legacy has
        # xhigh but no max: max clamps to xhigh, medium is untouched).
        kw = self._kwargs(transport, "some-byok-route", {"effort": "max"})
        assert kw["reasoning"]["effort"] == "xhigh"
        kw = self._kwargs(transport, "some-byok-route", {"effort": "medium"})
        assert kw["reasoning"]["effort"] == "medium"

    def test_cold_cache_keeps_default_behavior(self, transport, monkeypatch):
        _, mod = _router_plugin_module()
        monkeypatch.setattr(mod, "_efforts_cache", None)
        monkeypatch.setattr(mod, "_disk_checked", True)
        monkeypatch.setattr(mod, "_warm_efforts_async", lambda: None)
        kw = self._kwargs(transport, "grok-4.6", {"effort": "xhigh"})
        # Cold cache -> no declaration -> default codex vocabulary (xhigh ok).
        assert kw["reasoning"]["effort"] == "xhigh"

    def test_other_providers_unaffected(self, transport, seeded_catalog):
        kw = transport.build_kwargs(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url="https://generic.example.com/v1",
            session_id="sid",
            provider="some-other-provider",
            reasoning_config={"effort": "medium"},
        )
        # The router catalog's () verdict for gpt-4.1-mini must not leak
        # into other providers' requests.
        assert kw["reasoning"]["effort"] == "medium"
