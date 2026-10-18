"""Consecutive full-budget timeouts quarantine the endpoint (timeout circuit breaker).

A full-budget timeout is ambiguous — a slow reasoning model and a hung endpoint look identical
to a single request — so one timeout never hides a provider (the slow-model contract, #89445).
But consecutive timeouts with no success in between are a hung endpoint, and every further call
would otherwise pay the whole timeout budget again: a wedged local llama-server turned each
image of a vision run into timeout + fallback. After ``_AUX_TIMEOUT_STRIKES`` consecutive
timeouts the endpoint is quarantined for ``_AUX_TIMEOUT_QUARANTINE_TTL_SECONDS``: while
quarantined the primary dial is withheld entirely (``_assert_route_dialable``) and the fallback
chain answers; the skip itself is not a fresh strike, so one probe per cooldown window
re-trips on a single timeout; a success on the endpoint clears the streak.

``task="title_generation"`` is deliberate: like ``vision`` (the production case that motivated
this — a wedged local llama-server under ``auxiliary.vision``) it is a ``_TIMEOUT_NO_RETRY_TASKS``
member, so a full-budget timeout goes straight to the ladder with no same-provider retry — one
strike per call — while bypassing the vision client chain, which the ladder semantics under test
do not depend on.
"""
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent import auxiliary_health as ah
from agent.auxiliary_client import async_call_llm, call_llm

_LOCAL_URL = "http://127.0.0.1:8001/v1"
_MESSAGES = [{"role": "user", "content": "Describe the image"}]


@pytest.fixture(autouse=True)
def _fresh_unhealthy_cache():
    ah._reset_aux_unhealthy_cache()
    yield
    ah._reset_aux_unhealthy_cache()


def _timeout(lane: str) -> Exception:
    return Exception(f"Request timed out: {lane} endpoint hung past the full budget")


def _client(base_url, create):
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


def _pinned_custom_patches(primary, fallback):
    """A task pinned to a custom endpoint (the vision-to-local-llama-server shape). The task
    chain has no entries; the main-agent-model fallback hands out the given candidate (the
    ladder tail consults it for a non-auto route)."""
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "modelA", _LOCAL_URL, "k", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "modelA")),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=fallback),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")),
    )


def test_single_timeout_does_not_quarantine_and_next_success_clears_streak():
    """One full-budget timeout is ambiguity, not death: no quarantine after the first strike,
    and the next success on the endpoint forgets the streak entirely."""
    primary = _client(_LOCAL_URL, MagicMock(
        side_effect=[_timeout("A"), _ok_response("OK")]))
    fallback = (_client("http://127.0.0.1:9/v1", MagicMock(return_value=_ok_response("FB"))),
                "modelFB", "custom")
    patches = _pinned_custom_patches(primary, fallback)

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = call_llm(task="title_generation", messages=_MESSAGES)
    assert result.choices[0].message.content == "FB"
    assert not ah._is_provider_unhealthy("custom", _LOCAL_URL)

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = call_llm(task="title_generation", messages=_MESSAGES)
    assert result.choices[0].message.content == "OK"
    assert ah._aux_timeout_streak == {}


def test_builtin_provider_label_quarantines_provider_wide():
    """The breaker covers built-in providers too (no custom base_url → provider-wide key):
    two consecutive timeouts quarantine the label itself."""
    ah._note_aux_timeout("openrouter")
    assert not ah._is_provider_unhealthy("openrouter")
    ah._note_aux_timeout("openrouter")
    assert ah._is_provider_unhealthy("openrouter")


def test_quarantine_skip_is_not_retried_on_the_same_provider():
    """The withheld-dial exception never earns a same-provider transient retry — it must go
    straight to the fallback ladder."""
    assert not ac._should_retry_same_provider(
        "vision", ah._AuxTimeoutQuarantine("endpoint skipped: quarantined (timed out)"), "")


def test_two_consecutive_timeouts_quarantine_endpoint():
    """Two full-budget timeouts with nothing in between trip the quarantine: the endpoint is
    hidden and each call was really dialed (one strike per call)."""
    primary = _client(_LOCAL_URL, MagicMock(side_effect=_timeout("A")))
    fallback = (_client("http://127.0.0.1:9/v1", MagicMock(return_value=_ok_response("FB"))),
                "modelFB", "custom")
    patches = _pinned_custom_patches(primary, fallback)

    for _ in range(2):
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = call_llm(task="title_generation", messages=_MESSAGES)
        assert result.choices[0].message.content == "FB"

    assert primary.chat.completions.create.call_count == 2
    assert ah._is_provider_unhealthy("custom", _LOCAL_URL)


def test_quarantined_endpoint_is_not_dialed_and_skip_is_not_a_fresh_strike():
    """While quarantined the dial is withheld (the fallback answers without the doomed RTT),
    the skip does not add a strike, and the streak survives for single-strike re-trips."""
    primary = _client(_LOCAL_URL, MagicMock(side_effect=_timeout("A")))
    fallback = (_client("http://127.0.0.1:9/v1", MagicMock(return_value=_ok_response("FB"))),
                "modelFB", "custom")
    patches = _pinned_custom_patches(primary, fallback)

    for _ in range(2):
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            call_llm(task="title_generation", messages=_MESSAGES)
    dialed_before = primary.chat.completions.create.call_count

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = call_llm(task="title_generation", messages=_MESSAGES)
    assert result.choices[0].message.content == "FB"
    assert primary.chat.completions.create.call_count == dialed_before
    assert ah._is_provider_unhealthy("custom", _LOCAL_URL)
    # The streak was preserved at the trip threshold: the next real timeout re-trips alone.
    assert list(ah._aux_timeout_streak.values()) == [ah._AUX_TIMEOUT_STRIKES]


def test_expired_quarantine_probes_once_and_single_timeout_re_trips():
    """Cooldown expiry half-opens the endpoint: exactly one probe is dialed, and a single
    timeout re-trips the quarantine (streak survived the trip) — one probe per window."""
    primary = _client(_LOCAL_URL, MagicMock(side_effect=_timeout("A")))
    fallback = (_client("http://127.0.0.1:9/v1", MagicMock(return_value=_ok_response("FB"))),
                "modelFB", "custom")
    patches = _pinned_custom_patches(primary, fallback)

    for _ in range(2):
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            call_llm(task="title_generation", messages=_MESSAGES)
    key = ah._unhealthy_cache_key("custom", _LOCAL_URL)
    ah._aux_unhealthy_until[key] = time.time() - 1  # half-open

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = call_llm(task="title_generation", messages=_MESSAGES)
    assert result.choices[0].message.content == "FB"
    assert primary.chat.completions.create.call_count == 3
    assert ah._is_provider_unhealthy("custom", _LOCAL_URL)


def test_success_during_half_open_clears_streak_and_reopens_endpoint():
    """The probe succeeding is what heals the endpoint: quarantine expires, streak cleared,
    and the following call is dialed normally again."""
    primary = _client(_LOCAL_URL, MagicMock(side_effect=_timeout("A")))
    fallback = (_client("http://127.0.0.1:9/v1", MagicMock(return_value=_ok_response("FB"))),
                "modelFB", "custom")
    patches = _pinned_custom_patches(primary, fallback)

    for _ in range(2):
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            call_llm(task="title_generation", messages=_MESSAGES)
    key = ah._unhealthy_cache_key("custom", _LOCAL_URL)
    ah._aux_unhealthy_until[key] = time.time() - 1  # half-open
    primary.chat.completions.create = MagicMock(return_value=_ok_response("OK"))

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = call_llm(task="title_generation", messages=_MESSAGES)
    assert result.choices[0].message.content == "OK"
    assert ah._aux_timeout_streak == {}
    assert not ah._is_provider_unhealthy("custom", _LOCAL_URL)


@pytest.mark.asyncio
async def test_async_quarantined_endpoint_skips_dial():
    """Async parity (the production vision path is async): the quarantine withholds the dial
    and the fallback answers."""
    primary = _client(_LOCAL_URL, AsyncMock(side_effect=_timeout("A")))
    fallback = (_client("http://127.0.0.1:9/v1", AsyncMock(return_value=_ok_response("FB"))),
                "modelFB", "custom")
    patches = _pinned_custom_patches(primary, fallback)

    for _ in range(2):
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
                patch("agent.auxiliary_client._to_async_client",
                      side_effect=lambda c, m, **kw: (c, m)):
            result = await async_call_llm(task="title_generation", messages=_MESSAGES)
        assert result.choices[0].message.content == "FB"
    dialed_before = primary.chat.completions.create.call_count

    with patches[0], patches[1], patches[2], patches[3], patches[4], \
            patch("agent.auxiliary_client._to_async_client",
                  side_effect=lambda c, m, **kw: (c, m)):
        result = await async_call_llm(task="title_generation", messages=_MESSAGES)
    assert result.choices[0].message.content == "FB"
    assert primary.chat.completions.create.call_count == dialed_before
    assert ah._is_provider_unhealthy("custom", _LOCAL_URL)
