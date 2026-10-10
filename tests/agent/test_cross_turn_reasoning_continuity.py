"""Responses routes whose model renders earlier-turn reasoning keep it on the wire and through compaction.

Live basis (api.openai.com / api.x.ai, PR body): gpt-5.4/5.5 render replayed earlier-turn blobs only with
``reasoning.context: "all_turns"``; o-series / gpt-5-mini 400 on it; xAI renders them by default; sibling
variants of one OpenAI family read each other's blobs.
"""
import pytest

from agent.context_compressor import _prune_stale_reasoning_replay
from agent.transports.codex import ResponsesApiTransport

_OPENAI = "https://api.openai.com/v1"


def _history(model, issuer=f"other:{_OPENAI}"):
    item = {"type": "reasoning", "encrypted_content": "BLOB-1", "id": "rs_1", "_issuer_kind": issuer, "_issuer_model": model}
    return [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "a1", "codex_reasoning_items": [item]},
        {"role": "user", "content": "turn 2"},
    ]


def _build(model, base_url=_OPENAI, history_model=None, **params):
    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        model=model, messages=_history(history_model or model), base_url=base_url,
        reasoning_config={"enabled": True, "effort": "low"}, **params,
    )
    return transport, kwargs


@pytest.mark.parametrize("model, base_url, codex, expected", [
    ("gpt-5.5", _OPENAI, False, "all_turns"),
    ("gpt-6.1-sol", _OPENAI, False, "all_turns"),
    ("o4-mini", _OPENAI, False, None),
    ("gpt-5-mini", _OPENAI, False, None),
    ("gpt-5-nano", _OPENAI, False, None),
    ("o3", _OPENAI, False, None),
    ("gpt-5.5", "https://chatgpt.com/backend-api/codex", True, None),
    ("gpt-5.5", "https://relay.example.com/v1", False, None),
])
def test_reasoning_context_only_on_verified_openai_models(model, base_url, codex, expected):
    _, kwargs = _build(model, base_url, is_codex_backend=codex)
    assert kwargs["reasoning"].get("context") == expected


def test_context_rejection_omits_it_once_and_keeps_effort():
    transport, _ = _build("gpt-5.5")
    err = type("E", (Exception,), {"status_code": 400})("param 'reasoning.context': 'all_turns' is not supported")
    assert transport.reject_all_turns(err) is True
    retry = transport.build_kwargs(model="gpt-5.5", messages=_history("gpt-5.5"), base_url=_OPENAI,
                                   reasoning_config={"enabled": True, "effort": "low"})
    assert "context" not in retry["reasoning"] and retry["reasoning"]["effort"] == "low"
    assert transport.reject_all_turns(err) is False  # never loops


def test_sibling_variant_blob_replays_on_openai_only():
    _, same_family = _build("gpt-5.6-terra", history_model="gpt-5.6-luna")
    _, cross_family = _build("gpt-5.6-terra", history_model="gpt-5.5")
    assert any(i.get("type") == "reasoning" for i in same_family["input"])
    assert not any(i.get("type") == "reasoning" for i in cross_family["input"])


def test_compaction_keeps_prior_turn_reasoning_only_when_the_route_reads_it():
    from types import SimpleNamespace as Route
    from agent.codex_responses_adapter import route_reads_prior_turn_reasoning as reads

    def route(provider, model, base_url):
        return Route(api_mode="codex_responses", provider=provider, model=model, base_url=base_url)
    assert reads(route("xai", "grok-4.3", "https://api.x.ai/v1"))
    assert reads(route("openai-api", "gpt-5.5", _OPENAI))
    assert not reads(route("openai-api", "o4-mini", _OPENAI))
    assert not reads(route("openai-codex", "gpt-5.5", "https://chatgpt.com/backend-api/codex"))
    kept, pruned = _history("grok-4.3"), _history("o4-mini")
    assert _prune_stale_reasoning_replay(kept, keep_prior_turns=True) == 0
    assert kept[1]["codex_reasoning_items"][0]["encrypted_content"] == "BLOB-1"
    assert _prune_stale_reasoning_replay(pruned) == 1 and "codex_reasoning_items" not in pruned[1]


def test_first_invalid_encrypted_rejection_drops_only_unverified_items():
    transport, _ = _build("gpt-5.5")
    own = {"type": "reasoning", "encrypted_content": "OWN", "_issuer_kind": f"other:{_OPENAI}", "_issuer_model": "gpt-5.5"}
    legacy = {"type": "reasoning", "encrypted_content": "LEGACY"}
    messages = [{"role": "assistant", "content": "x", "codex_reasoning_items": [own, legacy]}]
    assert transport.drop_unverified_replay(messages) == 1
    assert messages[0]["codex_reasoning_items"] == [own]
    assert transport.drop_unverified_replay(messages) == 0  # escalation path takes over next time
