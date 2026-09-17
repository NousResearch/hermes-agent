"""Auxiliary ``provider: auto`` fallback walk is multi-hop (#106367).

When a ``fallback_providers`` candidate itself fails with a quota/rate-limit/payment/capacity
error, the walk must advance to the next configured entry instead of letting the candidate's
exception escape after a single hop. Every lane is attempted at most once; when the whole chain
is exhausted the last error still surfaces as a controlled failure. Sync and async share the walk.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm


def _quota_429(lane: str) -> Exception:
    exc = Exception(
        f"Error code: 429 - {{'error': {{'message': 'Weekly usage limit reached', "
        f"'type': 'usage_limit_reached', 'code': 'rate_limit_exceeded', 'lane': '{lane}'}}}}"
    )
    exc.status_code = 429
    return exc


def _client(base_url: str, create):
    client = MagicMock()
    client.base_url = base_url
    client.chat.completions.create = create
    return client


def _ok_response(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].message.tool_calls = None
    return resp


def _walk_patches(primary, main_chain_selections):
    """``provider: auto`` primary; per-task chain empty; main ``fallback_providers`` hands out the
    given (client, model, label) tuples in order and then reports exhaustion; discovery is empty."""
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("auto", None, None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "modelA")),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_fallback_chain",
              side_effect=list(main_chain_selections) + [(None, None, "")]),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
    )


def test_sync_walk_advances_past_quota_limited_candidate_to_next_configured_entry():
    """T3: primary 429 → fallback_providers[0] 429 → fallback_providers[1] serves."""
    primary = _client("http://127.0.0.1:1/v1", MagicMock(side_effect=_quota_429("A")))
    lane_b = _client("http://127.0.0.1:2/v1", MagicMock(side_effect=_quota_429("B")))
    lane_c = _client("http://127.0.0.1:3/v1", MagicMock(return_value=_ok_response("OK")))

    patches = _walk_patches(primary, [(lane_b, "modelB", "custom"), (lane_c, "modelC", "custom")])
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as mark_unhealthy:
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])

    assert result.choices[0].message.content == "OK"
    assert lane_b.chat.completions.create.call_count == 1
    assert lane_c.chat.completions.create.call_count == 1
    # The quota-limited candidate is quarantined so the ordered re-walk skips it.
    assert ("custom",) in {c.args for c in mark_unhealthy.call_args_list}
    assert any(c.kwargs.get("base_url") == "http://127.0.0.1:2/v1" for c in mark_unhealthy.call_args_list)


@pytest.mark.asyncio
async def test_async_walk_exhausts_every_configured_lane_once_then_raises_last_error():
    """T4: primary 429 → fallback[0] 429 → fallback[1] 429 → controlled exhaustion; each lane once."""
    primary = _client("http://127.0.0.1:1/v1", AsyncMock(side_effect=_quota_429("A")))
    lane_b = _client("http://127.0.0.1:2/v1", AsyncMock(side_effect=_quota_429("B")))
    lane_c = _client("http://127.0.0.1:3/v1", AsyncMock(side_effect=_quota_429("C")))

    patches = _walk_patches(primary, [(lane_b, "modelB", "custom"), (lane_c, "modelC", "custom")])
    with patches[0], patches[1], patches[2], patches[3] as main_chain, patches[4] as discovery, patches[5], \
            patch("agent.auxiliary_client._to_async_client", side_effect=lambda c, m, **kw: (c, m)):
        with pytest.raises(Exception, match="usage_limit_reached"):
            await async_call_llm(task="compression", messages=[{"role": "user", "content": "summarize"}])

    assert lane_b.chat.completions.create.call_count == 1
    assert lane_c.chat.completions.create.call_count == 1
    # Walk stopped once the configured chain reported exhaustion — no lane was re-tried.
    assert main_chain.call_count == 3
    # Exhaustion is controlled: discovery was consulted and found nothing, no fourth lane appended.
    assert discovery.call_count >= 1
