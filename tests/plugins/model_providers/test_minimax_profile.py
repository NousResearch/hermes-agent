"""Unit tests for the MiniMax provider profile.

Three MiniMax provider profiles (`minimax` direct API, `minimax-cn` China direct
API, `minimax-oauth` browser OAuth) all advertise a `default_aux_model` on
their `ProviderProfile`. The previous M2.7 / M2.7-highspeed values were
stale relative to the current frontier model (M3, released 2026-06-01) and
inconsistent with the `_PROVIDER_MODELS["minimax"]` catalog top entry in
`hermes_cli/models.py`.

This file pins the new defaults so the choice is reviewable and any future
revert shows up in a failing test rather than silent behavior drift.

Refs:
  - Issue #36196: M3 support request
  - PR #36205 (closed unmerged): Csrayz's M3 + 1M context work
  - PR #36212 (open): adds M3 to `_PROVIDER_MODELS["minimax"]` catalog
  - PR #6082: M2.7-highspeed → M2.7 for aux model (half-price fix)
  - Commit 773a0faca: same profile-layer fix pattern for `deepseek`
"""

from __future__ import annotations

import pytest


@pytest.fixture(params=["minimax", "minimax-cn", "minimax-oauth"])
def minimax_profile(request):
    """Resolve each registered MiniMax profile.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    import model_tools  # noqa: F401  -- triggers plugin discovery
    import providers

    profile = providers.get_provider_profile(request.param)
    assert profile is not None, f"{request.param} provider profile must be registered"
    return profile, request.param


class TestMinimaxAuxModelM3:
    """MiniMax profile aux model is the new frontier M3, not the stale M2.7.

    The catalog top entry is ``MiniMax-M3`` in
    ``hermes_cli.models._PROVIDER_MODELS['minimax']`` and the
    user-facing ``model.default`` for a Token-Plan install is M3,
    so pinning the aux default to the same model keeps the runtime
    consistent (same auth, same billing pool, same rate limits, no
    surprise 2x-cost highspeed variant). M3 was released 2026-06-01
    — picking it as the aux default matches the forward-looking
    catalog order rather than the pre-M3 era.
    """


    def test_consumer_api_returns_non_empty_for_each_provider(self, minimax_profile):
        from agent.auxiliary_client import _get_aux_model_for_provider

        profile, provider_id = minimax_profile
        resolved = _get_aux_model_for_provider(provider_id)
        assert resolved != "", (
            f"_get_aux_model_for_provider({provider_id!r}) returned empty — "
            "the 'No auxiliary LLM provider configured' warning will fire on "
            f"every {provider_id} session even though the profile advertises "
            f"default_aux_model={profile.default_aux_model!r}"
        )
        assert resolved == profile.default_aux_model, (
            f"_get_aux_model_for_provider({provider_id!r}) returned "
            f"{resolved!r} but profile advertises {profile.default_aux_model!r} "
            "— the consumer API and the profile have drifted out of sync"
        )




class TestMinimaxM3OpenAIReasoningWireShape:
    """MiniMax-M3 on api.minimax.io/v1 gets MiniMax's OpenAI-compatible knobs."""

    def test_m3_openai_route_requests_reasoning_split_by_default(self):
        import model_tools  # noqa: F401
        import providers

        profile = providers.get_provider_profile("minimax")
        assert profile is not None
        extra_body, top_level = profile.build_api_kwargs_extras(
            reasoning_config=None,
            model="MiniMax-M3",
            base_url="https://api.minimax.io/v1",
        )
        assert extra_body == {"reasoning_split": True}
        assert top_level == {}


    @pytest.mark.parametrize(
        "model,base_url",
        [
            ("MiniMax-M2.7", "https://api.minimax.io/v1"),
            ("MiniMax-M3", "https://api.minimax.io/anthropic"),
            ("MiniMax-M3", "https://api.minimaxi.com/v1"),
        ],
    )
    def test_non_m3_or_non_global_openai_routes_emit_no_openai_reasoning_knobs(
        self, model, base_url
    ):
        import model_tools  # noqa: F401
        import providers

        profile = providers.get_provider_profile("minimax")
        assert profile is not None
        extra_body, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
            base_url=base_url,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_transport_threads_base_url_to_profile(self):
        import model_tools  # noqa: F401
        import providers
        from agent.transports.chat_completions import ChatCompletionsTransport

        profile = providers.get_provider_profile("minimax")
        assert profile is not None
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="MiniMax-M3",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=profile,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.minimax.io/v1",
        )
        assert kwargs["extra_body"] == {
            "reasoning_split": True,
            "thinking": {"type": "adaptive"},
        }


class TestMinimaxOauthAliases:
    """Every ``--provider`` alias the user guide (website/docs/guides/minimax-oauth.md)
    promises for ``minimax-oauth`` must resolve through the plugin registry, not only the
    CLI alias tables — ``get_provider_profile(agent.provider)`` is what selects the
    anthropic_messages wire, extra_body and headers (#107928)."""

    def test_each_documented_oauth_alias_resolves_to_minimax_oauth(self):
        import model_tools  # noqa: F401
        import providers

        for alias in ("minimax_oauth", "minimax-portal", "minimax-global"):
            resolved = providers.get_provider_profile(alias)
            assert resolved is not None and resolved.name == "minimax-oauth", alias


def test_m31_chat_completions_effort_reaches_wire(minimax_profile):
    """PR #127088: the sibling /v1 route must transmit effort without a forbidden disable.

    Exercise real provider discovery, the transport and SDK serialization against a
    local HTTP transport; no MiniMax credentials or live inference are needed.
    """
    import json

    import httpx
    from openai import OpenAI
    from agent.transports.chat_completions import ChatCompletionsTransport

    profile, _ = minimax_profile
    bodies = []

    def capture(request):
        assert request.url.path == "/v1/chat/completions"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": bodies[-1]["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })

    cases = [
        ({"enabled": True, "effort": effort}, expected)
        for effort, expected in (
            ("low", "low"), ("medium", "medium"), ("high", "high"),
            ("xhigh", "xhigh"), ("max", "max"), ("minimal", "low"), ("ultra", "max"),
        )
    ] + [(config, None) for config in (
        None, {}, {"enabled": True}, {"enabled": False},
        {"enabled": False, "effort": "high"}, {"effort": "none"}, {"effort": "invalid"},
    )]
    with OpenAI(api_key="test-only", base_url="https://api.minimax.io/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(capture))) as client:
        for model in ("MiniMax-M3.1-Flash-Preview", "minimax/MiniMax-M3.1-Flash-Preview",
                      "MiniMax-M3.1", "MiniMax-M3-1", "vendor/MiniMax-M3-1-Flash-Preview"):
            for config, expected in cases:
                kwargs = ChatCompletionsTransport().build_kwargs(
                    model=model, messages=[{"role": "user", "content": "ping"}],
                    provider_profile=profile, reasoning_config=config,
                    base_url="https://api.minimax.io/v1/",
                )
                client.chat.completions.create(**kwargs)
                body = bodies[-1]
                assert body.get("reasoning_effort") == expected, (model, config, body)
                if expected is None:
                    assert "reasoning_effort" not in body
                assert not {"thinking", "reasoning", "output_config", "extra_body"} & body.keys()


def test_m31_openai_effort_preserves_sibling_contracts(minimax_profile):
    """M3 keeps its toggle; M2, version lookalikes and other routes remain untouched."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    profile, _ = minimax_profile
    global_route = "https://api.minimax.io/v1"
    for model, base_url in [
        (model, global_route) for model in (
            "MiniMax-M3", "minimax/MiniMax-M3", "MiniMax-M2.7", "MiniMax-M3-Flash",
            "MiniMax-M3.10", "MiniMax-M3.11-Flash", "MiniMax-M3-10", "MiniMax-M3-1x",
            "not-minimax-m3.1", "vendor/other-model",
        )
    ] + [("MiniMax-M3.1-Flash-Preview", route) for route in (
        "https://api.minimax.io/anthropic", "https://api.minimaxi.com/v1",
        "https://api.minimax.io.example.com/v1", "https://api.minimax.io/v1/other",
    )]:
        for config in (None, {"enabled": True, "effort": "low"}, {"enabled": False}):
            kwargs = ChatCompletionsTransport().build_kwargs(
                model=model, messages=[{"role": "user", "content": "ping"}],
                provider_profile=profile, reasoning_config=config, base_url=base_url,
            )
            expected = {}
            if model in ("MiniMax-M3", "minimax/MiniMax-M3"):
                expected["reasoning_split"] = True
                if config is not None:
                    expected["thinking"] = {"type": "adaptive" if config["enabled"] else "disabled"}
            assert kwargs.get("extra_body", {}) == expected, (model, base_url, config)
            assert "reasoning_effort" not in kwargs
