"""Opt-in Discord mentions make model fallback notices hard to miss."""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.run_turn_runner import _mention_model_fallback


def _source(*, platform=Platform.DISCORD, user_id="123456789"):
    return SimpleNamespace(platform=platform, user_id=user_id)


def test_mentions_discord_sender_for_model_fallback_when_enabled():
    message = "⚠️ Model fallback: primary unavailable; using backup."

    assert _mention_model_fallback(message, _source(), enabled=True) == (
        "<@123456789> ⚠️ Model fallback: primary unavailable; using backup."
    )


def test_does_not_mention_for_unrelated_status_or_disabled_setting():
    fallback = "⚠️ Model fallback: primary unavailable; using backup."

    assert _mention_model_fallback("Retrying request", _source(), enabled=True) == "Retrying request"
    assert _mention_model_fallback(fallback, _source(), enabled=False) == fallback


def test_rejects_non_discord_and_untrusted_user_ids():
    message = "⚠️ Model fallback: primary unavailable; using backup."

    assert _mention_model_fallback(message, _source(platform=Platform.SLACK), enabled=True) == message
    assert _mention_model_fallback(message, _source(user_id="123> @everyone"), enabled=True) == message
