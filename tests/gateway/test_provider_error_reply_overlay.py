"""Regression for #102336: the canned provider-failure replies are catalog-backed.

Every row of ``gateway.run._PROVIDER_ERROR_REPLIES`` resolves through ``agent.i18n`` at reply
time, so an operator can restyle what a chat participant sees from
``<HERMES_HOME>/locales/<lang>.yaml`` instead of patching the deployed image — and a category the
operator does not list keeps the shipped wording.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import i18n
from gateway.run import _PROVIDER_ERROR_REPLIES, _gateway_provider_error_reply

# One envelope per row of the pattern table, in resolution order. The two rate-limit rows are
# distinguished by whether the envelope names a reset window (#89401).
SAMPLES: dict[str, str] = {
    "gateway.provider_error.rate_limit": "API call failed after 3 retries: rate limited after 3 retries",
    "gateway.provider_error.rate_limit_reset":
        "API call failed after 3 retries: Error code: 429 - {'error': {'type': 'usage_limit_reached', "
        "'resets_in_seconds': 30995}}",
    "gateway.provider_error.auth": "⚠️ Provider authentication failed: HTTP 401 incorrect api key",
    "gateway.provider_error.policy": "API call failed after 3 retries: request was blocked under the safety policy",
    "gateway.provider_error.connection_interrupted":
        "API call failed after 3 retries: httpx.ReadError: [Errno 104] Connection reset by peer",
    "gateway.provider_error.endpoint_unreachable":
        "API call failed after 3 retries: httpx.ConnectError: [Errno 111] Connection refused",
    "gateway.provider_error.connection": "openai.APIConnectionError: Connection error.",
    "gateway.provider_error.generic": "RuntimeError: model returned empty content",
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fresh temp HERMES_HOME with the catalog cache cleared around the test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    i18n.reset_language_cache()
    try:
        yield home
    finally:
        i18n.reset_language_cache()


def _write_overlay(home: Path, values: dict[str, str]) -> None:
    """Write a partial ``<HERMES_HOME>/locales/en.yaml`` with just the given provider_error keys."""
    locales = home / "locales"
    locales.mkdir(exist_ok=True)
    body = "gateway:\n  provider_error:\n" + "".join(
        f"    {key}: {json.dumps(value)}\n" for key, value in values.items())
    (locales / "en.yaml").write_text(body, encoding="utf-8")


def test_operator_overlay_restyles_the_customer_facing_reply(home):
    """The overlay from #102336 changes what the chat user sees; unlisted categories do not move."""
    _write_overlay(home, {
        "rate_limit": "Sorry — I'm tied up right now. Give me a few minutes and send that again.",
        "generic": "Sorry — something went wrong on my end with that one. Please try again shortly.",
    })
    i18n.reset_language_cache()

    assert _gateway_provider_error_reply("rate limited after 3 retries") == (
        "Sorry — I'm tied up right now. Give me a few minutes and send that again.")
    assert _gateway_provider_error_reply("RuntimeError: model returned empty content") == (
        "Sorry — something went wrong on my end with that one. Please try again shortly.")

    # A category the operator did not list keeps the shipped diagnosis (#86570 wording).
    shipped = _gateway_provider_error_reply(
        "API call failed after 3 retries: httpx.ConnectError: [Errno 111] Connection refused")
    assert "not running or is unreachable" in shipped
    assert shipped != "Sorry — I'm tied up right now. Give me a few minutes and send that again."


def test_every_reply_category_is_a_catalog_key_and_overridable(home):
    """No row may carry its wording inline: each resolves through the catalog, so each is overridable."""
    sentinels = {key: f"overlay wording for {key}" for key in SAMPLES}
    _write_overlay(home, {key.rpartition(".")[2]: value for key, value in sentinels.items()})
    i18n.reset_language_cache()

    for key, envelope in SAMPLES.items():
        assert _gateway_provider_error_reply(envelope) == sentinels[key], envelope

    for _, key in _PROVIDER_ERROR_REPLIES:
        assert i18n.t(key) != key, f"{key} does not resolve in the bundled catalog"
