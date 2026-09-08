"""Tests for the outbound silence-narration filter (anti-loop control).

See the gateway delivery path: hallucinated "silence" tokens like ``*(silent)*``
are dropped pre-send so bot-to-bot channels can't mirror them into a token-burning
loop that crashes a model with "no content after all retries".
"""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import (
    DeliveryRouter,
    DeliveryTarget,
    _is_silence_narration,
)


# --- Truth table -----------------------------------------------------------

POSITIVE_CASES = [
    "*(silent)*",
    "*Silence.*",
    "🔇",
    ".",
    "…",
    "...",
    "(silent)",
    "_silent_",
    "silent",
    " *(silent)* ",
    "`silent`",
    "~silent~",
    "Silence",
    "no response",
    "No Reply.",
]

NEGATIVE_CASES = [
    "Silence is golden — here is the plan...",
    "Silent install completed",
    "The deployment ran silently in the background",
    "ok",
    "👍",
    "Here is the result:\n\n- item one\n- item two",
    "I have nothing to add, but here is why: the build is green.",
    "silently",  # word boundary — trailing letters mean it isn't a bare token
    "no responses were collected from the survey",
    # A 64+ char string that opens with a silence token must not be dropped.
    "silent " + "x" * 70,
    "",
    "   ",
]


@pytest.mark.parametrize("content", POSITIVE_CASES)
def test_is_silence_narration_positive(content):
    assert _is_silence_narration(content) is True


def test_length_guard_rejects_long_strings():
    # Exactly 65 chars of dots — over the 64-char guard, so not treated as narration.
    assert _is_silence_narration("." * 65) is False
    assert _is_silence_narration("." * 64) is True


# --- Integration through DeliveryRouter ------------------------------------

class RecordingAdapter:
    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}


@pytest.mark.asyncio
async def test_silence_narration_dropped_pre_send(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_FILTER_SILENCE_NARRATION", raising=False)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(target, "*(silent)*", metadata=None)

    assert adapter.calls == []  # adapter.send never invoked
    assert result == {
        "success": True,
        "filtered": "silence_narration",
        "delivered": False,
    }


@pytest.mark.asyncio
async def test_config_opt_out_lets_silence_through(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_FILTER_SILENCE_NARRATION", raising=False)
    adapter = RecordingAdapter()
    config = GatewayConfig(filter_silence_narration=False)
    router = DeliveryRouter(config, adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(target, "*(silent)*", metadata=None)

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["content"] == "*(silent)*"
    assert result == {"success": True}


@pytest.mark.asyncio
async def test_env_override_enables_filter_over_config(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_FILTER_SILENCE_NARRATION", "1")
    adapter = RecordingAdapter()
    # Config says off, env override forces on.
    config = GatewayConfig(filter_silence_narration=False)
    router = DeliveryRouter(config, adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(target, "*(silent)*", metadata=None)

    assert adapter.calls == []
    assert result["filtered"] == "silence_narration"


# --- Cron artifacts are exempt ----------------------------------------------
#
# The filter exists to stop bot-to-bot mirror loops of *model chatter*. Cron
# output is an artifact: a job that legitimately emits "..." (a quiet script,
# a terse digest) has no loop partner, and dropping it while returning
# {"success": True} produced a cron the scheduler logged as delivered and the
# user never received (#77763). Cron sends carry job_id in metadata.


@pytest.mark.asyncio
async def test_cron_job_id_metadata_bypasses_the_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_FILTER_SILENCE_NARRATION", raising=False)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(
        target, "*(silent)*", metadata={"job_id": "92e639af907f"},
    )

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["content"] == "*(silent)*"
    assert result.get("filtered") is None
    assert result.get("delivered") is not False


@pytest.mark.asyncio
async def test_non_cron_metadata_still_filters(tmp_path, monkeypatch):
    """The exemption keys on job_id alone — everything else is unchanged."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_FILTER_SILENCE_NARRATION", raising=False)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(
        target, "*(silent)*", metadata={"thread_id": "42", "user_id": "u1"},
    )

    assert adapter.calls == []
    assert result["filtered"] == "silence_narration"


# --- Config round-trip ------------------------------------------------------


