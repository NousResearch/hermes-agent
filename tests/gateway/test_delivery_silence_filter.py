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
def test_real_replies_are_not_silence_narration(content):
    assert _is_silence_narration(content) is False


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
    adapter = RecordingAdapter()
    config = GatewayConfig(filter_silence_narration=False)
    router = DeliveryRouter(config, adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(target, "*(silent)*", metadata=None)

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["content"] == "*(silent)*"
    assert result == {"success": True}


@pytest.mark.asyncio
async def test_env_does_not_override_config_opt_out(tmp_path, monkeypatch):
    # env shouldn't override config when filter is turned off
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_FILTER_SILENCE_NARRATION", "1")
    adapter = RecordingAdapter()
    config = GatewayConfig(filter_silence_narration=False)
    router = DeliveryRouter(config, adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(target, "*(silent)*", metadata=None)

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["content"] == "*(silent)*"
    assert result == {"success": True}


@pytest.mark.asyncio
async def test_env_does_not_disable_filter_when_config_enabled(tmp_path, monkeypatch):
    # env shouldn't disable filter when config has it on
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_FILTER_SILENCE_NARRATION", "0")
    adapter = RecordingAdapter()
    config = GatewayConfig(filter_silence_narration=True)
    router = DeliveryRouter(config, adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(target, "*(silent)*", metadata=None)

    assert adapter.calls == []
    assert result == {
        "success": True,
        "filtered": "silence_narration",
        "delivered": False,
    }


@pytest.mark.asyncio
async def test_multiplex_profiles_independent_silence_narration_filtering(tmp_path, monkeypatch):
    # secondary profile keeps its own setting under multiplex regardless of process env
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_FILTER_SILENCE_NARRATION", "1")
    adapter_a = RecordingAdapter()
    adapter_b = RecordingAdapter()
    router_a = DeliveryRouter(GatewayConfig(filter_silence_narration=True), adapters={Platform.DISCORD: adapter_a})
    router_b = DeliveryRouter(GatewayConfig(filter_silence_narration=False), adapters={Platform.DISCORD: adapter_b})
    target = DeliveryTarget.parse("discord:99887766")

    res_a = await router_a._deliver_to_platform(target, "*(silent)*", metadata=None)
    res_b = await router_b._deliver_to_platform(target, "*(silent)*", metadata=None)

    assert adapter_a.calls == []
    assert res_a["filtered"] == "silence_narration"
    assert len(adapter_b.calls) == 1
    assert adapter_b.calls[0]["content"] == "*(silent)*"
    assert res_b == {"success": True}


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
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    result = await router._deliver_to_platform(
        target, "*(silent)*", metadata={"thread_id": "42", "user_id": "u1"},
    )

    assert adapter.calls == []
    assert result["filtered"] == "silence_narration"


# --- Config round-trip ------------------------------------------------------
# The vocabulary and the length guard are declared data (#135444): an instance that narrates
# silence in its own language declares its narrations under ``gateway.silence_narration_tokens``
# and the longest line the filter may drop under ``gateway.silence_narration_max_chars``. The
# built-in English vocabulary stays the default, so an install that declares nothing is unchanged.

# The narrations recorded in that report, verbatim.
DECLARED_TOKENS = [
    "tăcere",
    "fără răspuns",
    "fără livrare: no_answer_owed",
    "tăcere — clasa nu poartă răspuns",
    "tăcere — no_answer_owed: clasa nu poartă text și nu se livrează niciun răspuns",
    ("fără răspuns: verdictul no_answer_owed nu poartă text — clasa nu compune niciun răspuns, "
     "iar pe linia asta nu se livrează nimic"),
]
DECLARED_NARRATIONS = [f"_({token})_" for token in DECLARED_TOKENS]
DECLARED_MAX_CHARS = max(len(narration) for narration in DECLARED_NARRATIONS)


@pytest.mark.parametrize("content", DECLARED_NARRATIONS)
def test_declared_vocabulary_drops_the_instances_own_narrations(content):
    assert _is_silence_narration(content, tokens=DECLARED_TOKENS, max_chars=DECLARED_MAX_CHARS) is True


@pytest.mark.parametrize("content", DECLARED_NARRATIONS)
def test_narrations_without_a_declaration_are_delivered(content):
    """The built-in vocabulary is English: an undeclared line was delivered before and still is."""
    assert _is_silence_narration(content) is False


def test_undeclared_vocabulary_falls_back_to_the_built_ins():
    assert _is_silence_narration("*(silent)*", tokens=[], max_chars=None) is True


def test_declared_vocabulary_keeps_the_anchoring():
    assert _is_silence_narration("_(tăcere)_ restul răspunsului", tokens=DECLARED_TOKENS,
                                 max_chars=DECLARED_MAX_CHARS) is False
    assert _is_silence_narration("salut, tăcere nu e ce am spus", tokens=["tăcere"], max_chars=200) is False


def test_declared_max_chars_bounds_the_line():
    long_token = "x" * 100
    assert _is_silence_narration(f"_({long_token})_", tokens=[long_token], max_chars=64) is False
    assert _is_silence_narration(f"_({long_token})_", tokens=[long_token], max_chars=200) is True


def test_declared_multi_word_token_tolerates_whitespace_runs():
    assert _is_silence_narration("_(no   response)_", tokens=["no response"]) is True


def test_gateway_config_reads_the_declaration_from_yaml():
    config = GatewayConfig.from_dict(
        {"gateway": {"silence_narration_tokens": DECLARED_TOKENS, "silence_narration_max_chars": 200}})

    assert config.silence_narration_tokens == DECLARED_TOKENS
    assert config.silence_narration_max_chars == 200


def test_gateway_config_defaults_leave_the_vocabulary_undeclared():
    config = GatewayConfig()

    assert config.silence_narration_tokens is None
    assert config.silence_narration_max_chars is None


def test_gateway_config_ignores_malformed_declarations():
    config = GatewayConfig.from_dict(
        {"gateway": {"silence_narration_tokens": "tăcere", "silence_narration_max_chars": "many"}})

    assert config.silence_narration_tokens is None
    assert config.silence_narration_max_chars is None


def test_declared_vocabulary_survives_the_config_round_trip():
    config = GatewayConfig(silence_narration_tokens=["tăcere"], silence_narration_max_chars=200)

    restored = GatewayConfig.from_dict(config.to_dict())

    assert restored.silence_narration_tokens == ["tăcere"]
    assert restored.silence_narration_max_chars == 200


def test_config_loader_bridges_the_nested_gateway_declaration():
    """The ``gateway:`` YAML form reaches the gateway's own config dict."""
    from gateway.config_loader import bridge_toplevel_keys

    gw_data: dict = {}
    bridge_toplevel_keys({}, {"silence_narration_tokens": DECLARED_TOKENS,
                              "silence_narration_max_chars": 200}, gw_data)

    assert gw_data["silence_narration_tokens"] == DECLARED_TOKENS
    assert gw_data["silence_narration_max_chars"] == 200


@pytest.mark.asyncio
async def test_router_drops_a_declared_narration_pre_send(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    config = GatewayConfig(silence_narration_tokens=DECLARED_TOKENS,
                           silence_narration_max_chars=DECLARED_MAX_CHARS)
    router = DeliveryRouter(config, adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:99887766")

    for narration in DECLARED_NARRATIONS:
        result = await router._deliver_to_platform(target, narration, metadata=None)
        assert result == {"success": True, "filtered": "silence_narration", "delivered": False}

    assert adapter.calls == []


