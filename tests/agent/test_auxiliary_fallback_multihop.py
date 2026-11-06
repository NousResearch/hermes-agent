"""Auxiliary ``provider: auto`` fallback walk is multi-hop (#106367).

When a ``fallback_providers`` candidate itself fails with a quota/rate-limit/payment/capacity
error, the walk must advance to the next configured entry instead of letting the candidate's
exception escape after a single hop. Every lane is attempted at most once; when the whole chain
is exhausted the primary (narrowed) error still surfaces as a controlled failure. Sync and async
share the walk. A quarantined candidate is hidden for a TTL that matches its failure class: seconds
for a transient 429 / dropped connection, the long payment hold only for depleted credit.
"""
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent.auxiliary_client import async_call_llm, call_llm


@pytest.fixture(autouse=True)
def _fresh_unhealthy_cache():
    ac._reset_aux_unhealthy_cache()
    yield
    ac._reset_aux_unhealthy_cache()


def _quota_429(lane: str) -> Exception:
    exc = Exception(
        f"Error code: 429 - {{'error': {{'message': 'Weekly usage limit reached', "
        f"'type': 'usage_limit_reached', 'code': 'rate_limit_exceeded', 'lane': '{lane}'}}}}"
    )
    exc.status_code = 429
    return exc


def _plain_429(lane: str) -> Exception:
    exc = Exception(f"Error code: 429 - {{'error': {{'message': 'Rate limit exceeded, retry in 20s', 'lane': '{lane}'}}}}")
    exc.status_code = 429
    return exc


def _payment_402(lane: str) -> Exception:
    exc = Exception(f"Error code: 402 - {{'error': {{'message': 'Insufficient credits', 'lane': '{lane}'}}}}")
    exc.status_code = 402
    return exc


def _quarantine_remaining(base_url: str) -> float:
    key = ac._unhealthy_cache_key("custom", base_url)
    return ac._aux_unhealthy_until[key] - time.time()


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
    )


def test_sync_walk_advances_past_quota_limited_candidate_to_next_configured_entry():
    """T3: primary 429 → fallback_providers[0] 429 → fallback_providers[1] serves."""
    primary = _client("http://127.0.0.1:1/v1", MagicMock(side_effect=_quota_429("A")))
    lane_b = _client("http://127.0.0.1:2/v1", MagicMock(side_effect=_quota_429("B")))
    lane_c = _client("http://127.0.0.1:3/v1", MagicMock(return_value=_ok_response("OK")))

    patches = _walk_patches(primary, [(lane_b, "modelB", "custom"), (lane_c, "modelC", "custom")])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])

    assert result.choices[0].message.content == "OK"
    assert lane_b.chat.completions.create.call_count == 1
    assert lane_c.chat.completions.create.call_count == 1
    # The quota-limited candidate is really quarantined (per endpoint) so the ordered re-walk skips it.
    assert ac._is_provider_unhealthy("custom", "http://127.0.0.1:2/v1")
    assert not ac._is_provider_unhealthy("custom", "http://127.0.0.1:3/v1")


@pytest.mark.asyncio
async def test_async_walk_exhausts_every_configured_lane_once_then_raises_primary_error():
    """T4: primary 429 → fallback[0] 429 → fallback[1] 429 → controlled exhaustion; each lane once;
    the primary's own error is what surfaces."""
    primary = _client("http://127.0.0.1:1/v1", AsyncMock(side_effect=_quota_429("A")))
    lane_b = _client("http://127.0.0.1:2/v1", AsyncMock(side_effect=_quota_429("B")))
    lane_c = _client("http://127.0.0.1:3/v1", AsyncMock(side_effect=_quota_429("C")))

    patches = _walk_patches(primary, [(lane_b, "modelB", "custom"), (lane_c, "modelC", "custom")])
    with patches[0], patches[1], patches[2], patches[3] as main_chain, patches[4] as discovery, \
            patch("agent.auxiliary_client._to_async_client", side_effect=lambda c, m, **kw: (c, m)):
        with pytest.raises(Exception, match="'lane': 'A'"):
            await async_call_llm(task="compression", messages=[{"role": "user", "content": "summarize"}])

    assert lane_b.chat.completions.create.call_count == 1
    assert lane_c.chat.completions.create.call_count == 1
    # Walk stopped once the configured chain reported exhaustion — no lane was re-tried.
    assert main_chain.call_count == 3
    # Exhaustion is controlled: discovery was consulted and found nothing, no fourth lane appended.
    assert discovery.call_count >= 1


def test_candidate_quarantine_ttl_is_short_for_transient_429_and_long_for_payment(caplog):
    """A per-minute 429 on a candidate hides the lane for seconds, not the 10-minute payment hold,
    and the warning names the real class; a 402 candidate still gets the long hold."""
    caplog.set_level("WARNING", logger="agent.auxiliary_client")
    primary = _client("http://127.0.0.1:1/v1", MagicMock(side_effect=_plain_429("A")))
    lane_b = _client("http://127.0.0.1:2/v1", MagicMock(side_effect=_plain_429("B")))
    lane_c = _client("http://127.0.0.1:3/v1", MagicMock(side_effect=_payment_402("C")))
    lane_d = _client("http://127.0.0.1:4/v1", MagicMock(return_value=_ok_response("OK")))

    patches = _walk_patches(
        primary, [(lane_b, "modelB", "custom"), (lane_c, "modelC", "custom"), (lane_d, "modelD", "custom")])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])

    assert result.choices[0].message.content == "OK"
    assert 0 < _quarantine_remaining("http://127.0.0.1:2/v1") <= 60
    assert _quarantine_remaining("http://127.0.0.1:3/v1") > 60
    marks = [r.getMessage() for r in caplog.records if "marking local/custom unhealthy" in r.getMessage()]
    assert marks and "payment" not in marks[0] and "rate limit" in marks[0]
