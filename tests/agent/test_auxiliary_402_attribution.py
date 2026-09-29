"""Auxiliary 402 attribution must follow the serving endpoint, not the route label.

A pinned ``openrouter`` task (or an ``auto`` task) can be rescued onto another provider's
endpoint by the auto-detection walk or a fallback chain, e.g. a custom OpenAI-compatible
endpoint after an OpenRouter pool outage. When that endpoint 402s, the failure must be
attributed to the custom endpoint (unhealthy mark scoped to its base_url, logs naming it) and
must NEVER bench the funded OpenRouter lane provider-wide, nor rotate/quarantine the
OpenRouter credential pool. Chain-walk ordering semantics (#106367) are unchanged: the walk
still advances to the next configured entry.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent.auxiliary_client import call_llm


CUSTOM_URL = "https://llm.example.com/v1/"


@pytest.fixture(autouse=True)
def _fresh_unhealthy_cache():
    ac._reset_aux_unhealthy_cache()
    yield
    ac._reset_aux_unhealthy_cache()


def _payment_402() -> Exception:
    exc = Exception("Error code: 402 - {'error': 'Payment required'}")
    exc.status_code = 402
    return exc


def _ok_response(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].message.tool_calls = None
    return resp


def _served_client(base_url: str, effective_provider: str, create) -> MagicMock:
    """A client the route resolver handed back for a label it no longer matches: it carries the
    effective-provider tag and actually talks to another provider's endpoint."""
    client = MagicMock()
    client.base_url = base_url
    client._hermes_aux_effective_provider = effective_provider
    client.chat.completions.create = create
    return client


def _healthy_lane():
    lane = MagicMock()
    lane.base_url = "https://openrouter.ai/api/v1/"
    lane._hermes_aux_effective_provider = "openrouter"
    lane.chat.completions.create = MagicMock(return_value=_ok_response("OK"))
    return lane


def _assert_attribution(caplog):
    """The custom endpoint is benched per endpoint; the OpenRouter lane is not; the log names
    the custom endpoint, never the stale label."""
    assert ac._is_provider_unhealthy("custom", CUSTOM_URL) is True
    assert ac._is_provider_unhealthy("openrouter") is False
    assert ac._is_provider_unhealthy("auto") is False
    messages = [r.getMessage() for r in caplog.records]
    named_endpoint = [m for m in messages if "payment error on custom at " in m and CUSTOM_URL in m]
    assert named_endpoint, f"no log names the custom endpoint: {messages}"
    assert not [m for m in messages if "payment error on openrouter" in m]
    assert not [m for m in messages if "payment error on auto" in m]


def test_pinned_openrouter_label_402_on_custom_endpoint_benches_only_that_endpoint(caplog):
    """Pinned flavour: task label says openrouter, the request is served by the custom
    endpoint, which 402s. The chain still advances; only the custom endpoint is benched."""
    caplog.set_level("INFO", logger="agent.auxiliary_client")
    custom_client = _served_client(CUSTOM_URL, "custom", MagicMock(side_effect=_payment_402()))
    healthy = _healthy_lane()
    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("openrouter", "google/gemini-3-flash-preview", None, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(custom_client, "custom-model")), \
         patch("agent.auxiliary_client._recover_provider_pool") as pool_recovery, \
         patch("agent.auxiliary_client._try_configured_fallback_chain",
               return_value=(healthy, "m2", "fallback_providers[2](openrouter)")):
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])

    assert result.choices[0].message.content == "OK"
    _assert_attribution(caplog)
    # The stale "openrouter" label must not rotate/quarantine the OpenRouter credential pool.
    assert not [c for c in pool_recovery.call_args_list if c.args[0] == "openrouter"]


def test_auto_label_402_on_custom_endpoint_benches_only_that_endpoint(caplog):
    """Auto flavour: resolved label is "auto", the auto walk served the request from the
    custom endpoint, which 402s. Same attribution rules as the pinned flavour."""
    caplog.set_level("INFO", logger="agent.auxiliary_client")
    custom_client = _served_client(CUSTOM_URL, "custom", MagicMock(side_effect=_payment_402()))
    healthy = _healthy_lane()
    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", None, None, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(custom_client, "custom-model")), \
         patch("agent.auxiliary_client._recover_provider_pool") as pool_recovery, \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_main_fallback_chain",
               return_value=(healthy, "m2", "fallback_providers[2](openrouter)")), \
         patch("agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")):
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])

    assert result.choices[0].message.content == "OK"
    _assert_attribution(caplog)
    assert not [c for c in pool_recovery.call_args_list if c.args[0] in ("openrouter", "auto")]


def test_stale_label_402_with_exhausted_chain_still_names_the_endpoint(caplog):
    """Exhaustion flavour: no fallback answers, the narrowed 402 surfaces — and even the
    all-fallbacks-exhausted warning names the serving endpoint, not the stale label."""
    caplog.set_level("INFO", logger="agent.auxiliary_client")
    custom_client = _served_client(CUSTOM_URL, "custom", MagicMock(side_effect=_payment_402()))
    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("openrouter", "google/gemini-3-flash-preview", None, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(custom_client, "custom-model")), \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")):
        with pytest.raises(Exception, match="Payment required"):
            call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])

    _assert_attribution(caplog)
    exhausted = [r.getMessage() for r in caplog.records if "all fallbacks exhausted" in r.getMessage()]
    assert exhausted and "custom at " + CUSTOM_URL in exhausted[0]


def test_genuine_openrouter_402_still_benches_openrouter_provider_wide():
    """Regression guard for the healthy case the fix must preserve: a 402 from a client that
    really is OpenRouter keeps the provider-wide bench (unchanged behaviour)."""
    or_client = _served_client("https://openrouter.ai/api/v1/", "openrouter",
                               MagicMock(side_effect=_payment_402()))
    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("openrouter", "google/gemini-3-flash-preview", None, None, None)), \
         patch("agent.auxiliary_client._get_cached_client",
               return_value=(or_client, "google/gemini-3-flash-preview")), \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")):
        with pytest.raises(Exception, match="Payment required"):
            call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])

    assert ac._is_provider_unhealthy("openrouter") is True


class _Client:
    def __init__(self, base_url):
        self.base_url = base_url


class TestRecoverablePoolProviderHostVerification:
    """_recoverable_pool_provider verifies the failing base_url host before attributing."""

    def test_label_matching_host_keeps_label(self):
        assert ac._recoverable_pool_provider("openrouter", _Client("https://openrouter.ai/api/v1/")) \
            == "openrouter"

    def test_stale_label_on_known_foreign_host_attributes_to_the_host_provider(self):
        # api.x.ai is positively the xai-oauth endpoint: a rejection there speaks to that
        # pool, never to the stale "nous" label the route carried.
        assert ac._recoverable_pool_provider("nous", _Client("https://api.x.ai/v1/")) == "xai-oauth"

    def test_unverifiable_host_keeps_label_for_proxy_safety(self):
        # A proxy / unregistered endpoint cannot be verified; its credentials may be the
        # label provider's own, so the label survives (parity with the session-key shield).
        assert ac._recoverable_pool_provider("openai-api", _Client("https://proxy.example:8443/v1/")) \
            == "openai-api"

    def test_client_without_base_url_keeps_label(self):
        assert ac._recoverable_pool_provider("xai-oauth", None) == "xai-oauth"