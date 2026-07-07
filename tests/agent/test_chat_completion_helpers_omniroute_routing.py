from types import SimpleNamespace

from agent.chat_completion_helpers import _maybe_route_omniroute_context_bucket_model


def _payload(model: str, tokens: int, **extra):
    payload = {
        "model": model,
        # estimate_request_context_tokens() uses chars//4 for the `input` shape.
        "input": "x" * (tokens * 4),
    }
    payload.update(extra)
    return payload


def test_omniroute_weighted_wide_downshifts_small_requests_to_128k():
    agent = SimpleNamespace()
    routed = _maybe_route_omniroute_context_bucket_model(
        agent, _payload("combo/coding-weighted-wide", 20_000)
    )
    assert routed["model"] == "combo/coding-weighted-128k"
    assert agent._omniroute_effective_model == "combo/coding-weighted-128k"
    assert agent._omniroute_effective_context_length == 128_000


def test_omniroute_weighted_wide_picks_200k_for_mid_size_requests():
    routed = _maybe_route_omniroute_context_bucket_model(
        object(), _payload("combo/coding-weighted-wide", 120_000)
    )
    assert routed["model"] == "combo/coding-weighted-200k"


def test_omniroute_weighted_wide_picks_262k_between_200k_and_400k():
    routed = _maybe_route_omniroute_context_bucket_model(
        object(), _payload("combo/coding-weighted-wide", 220_000)
    )
    assert routed["model"] == "combo/coding-weighted-262k"


def test_omniroute_weighted_wide_picks_400k_for_large_requests():
    routed = _maybe_route_omniroute_context_bucket_model(
        object(), _payload("combo/coding-weighted-wide", 320_000)
    )
    assert routed["model"] == "combo/coding-weighted-400k"


def test_legacy_ranked_1m_bucket_still_downshifts():
    routed = _maybe_route_omniroute_context_bucket_model(
        object(), _payload("combo/coding-extended-ranked-1m", 220_000)
    )
    assert routed["model"] == "combo/coding-extended-ranked-400k"


def test_omniroute_selected_cap_never_silently_upgrades_above_requested_tier():
    routed = _maybe_route_omniroute_context_bucket_model(
        object(), _payload("combo/coding-extended-ranked-400k", 420_000)
    )
    assert routed["model"] == "combo/coding-extended-ranked-400k"


def test_requested_output_reserve_can_bump_to_next_bucket():
    routed = _maybe_route_omniroute_context_bucket_model(
        object(),
        _payload(
            "combo/coding-extended-ranked-200k",
            110_000,
            max_completion_tokens=40_000,
        ),
    )
    assert routed["model"] == "combo/coding-extended-ranked-200k"


def test_non_bucket_models_are_unchanged():
    agent = SimpleNamespace(_omniroute_effective_context_length=128_000)
    payload = _payload("combo/coding-extended-gpt55-400k", 20_000)
    routed = _maybe_route_omniroute_context_bucket_model(agent, payload)
    assert routed is payload
    assert agent._omniroute_effective_model is None
    assert agent._omniroute_effective_context_length is None
