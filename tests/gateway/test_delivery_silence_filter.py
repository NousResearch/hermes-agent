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


@pytest.mark.parametrize("content", NEGATIVE_CASES)
def test_is_silence_narration_negative(content):
    assert _is_silence_narration(content) is False


def test_is_silence_narration_none_safe():
    assert _is_silence_narration(None) is False


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
async def test_real_message_is_delivered(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_FILTER_SILENCE_NARRATION", raising=False)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(
        target, "Silence is golden — here is the plan...", metadata=None
    )

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["content"] == "Silence is golden — here is the plan..."
    assert result == {"success": True}


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
async def test_env_override_disables_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_FILTER_SILENCE_NARRATION", "0")
    adapter = RecordingAdapter()
    # Config default is True, but env override wins.
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(target, "🔇", metadata=None)

    assert len(adapter.calls) == 1
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


@pytest.mark.asyncio
async def test_local_delivery_not_filtered(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_FILTER_SILENCE_NARRATION", raising=False)
    router = DeliveryRouter(GatewayConfig(), adapters={})

    results = await router.deliver(
        content="*(silent)*",
        targets=[DeliveryTarget.parse("local")],
        job_id="silence-job",
    )

    # Local path saved the file (no loop risk) and was not filtered.
    local_result = results["local"]
    assert local_result["success"] is True
    saved_path = local_result["result"]["path"]
    assert saved_path.endswith(".md")


# --- Config round-trip ------------------------------------------------------

def test_config_flag_defaults_true():
    assert GatewayConfig().filter_silence_narration is True


def test_config_from_dict_parses_flag():
    cfg = GatewayConfig.from_dict({"filter_silence_narration": False})
    assert cfg.filter_silence_narration is False


def test_config_to_dict_roundtrip():
    cfg = GatewayConfig(filter_silence_narration=False)
    assert cfg.to_dict()["filter_silence_narration"] is False
    restored = GatewayConfig.from_dict(cfg.to_dict())
    assert restored.filter_silence_narration is False
