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


# --- The declared class: vocabulary and guard are data -----------------------
#
# The silence-narration class is language-specific by nature, so the vocabulary and the length
# guard are declared — ``gateway.silence_narration.tokens`` / ``.max_chars`` — and the delivery
# path reads them instead of carrying a second hand-written list. Declaring nothing (or declaring
# something that does not parse) keeps the built-in English/symbol class exactly as it was.
#
# The narration strings below are quoted as literals on purpose: they are the class a firm-facing
# channel actually read (the delivery of the ``no_answer_owed`` verdict), and a literal keeps this
# file collectable against the revision that predates the declaration.

MEASURED_NARRATIONS = [
    "_(tăcere)_",
    "_(fără livrare: no_answer_owed)_",
    "_(fără livrare — no_answer_owed)_",
    "_(tăcere — clasa nu poartă răspuns)_",
    "_(tăcere — `no_answer_owed`: clasa nu poartă text și nu se livrează niciun răspuns)_",
    "_(fără răspuns: verdictul `no_answer_owed` nu poartă text — clasa nu compune niciun răspuns, "
    "iar pe linia asta nu se livrează nimic)_",
]

DECLARED_VOCABULARY = [
    "tăcere",
    "fără livrare",
    "fără răspuns",
    "verdictul",
    "no_answer_owed",
    "clasa nu poartă răspuns",
    "clasa nu poartă text și nu se livrează niciun răspuns",
    "clasa nu compune niciun răspuns",
    "nu poartă text",
    "iar pe linia asta nu se livrează nimic",
]

DECLARED_MAX_CHARS = 200


def _declared_config(**overrides):
    """A config whose profile declares the class the way a deployment would."""
    declaration = {"tokens": DECLARED_VOCABULARY, "max_chars": DECLARED_MAX_CHARS, **overrides}
    return GatewayConfig.from_dict({"gateway": {"silence_narration": declaration}})


def _router(config, adapter):
    return DeliveryRouter(config, adapters={Platform.DISCORD: adapter})


TARGET = "discord:99887766"


@pytest.mark.parametrize("content", MEASURED_NARRATIONS)
@pytest.mark.asyncio
async def test_declared_vocabulary_drops_the_measured_narrations(tmp_path, monkeypatch, content):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = _router(_declared_config(), adapter)

    result = await router._deliver_to_platform(DeliveryTarget.parse(TARGET), content, metadata=None)

    assert adapter.calls == []
    assert result == {"success": True, "filtered": "silence_narration", "delivered": False}


@pytest.mark.asyncio
async def test_declared_token_is_the_one_the_filter_uses(tmp_path, monkeypatch):
    """A placeholder the built-in class never carried is dropped once the profile declares it."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    declared = RecordingAdapter()
    built_in = RecordingAdapter()

    declared_result = await _router(_declared_config(tokens=["无输出"]), declared)._deliver_to_platform(
        DeliveryTarget.parse(TARGET), "（无输出）", metadata=None,
    )
    built_in_result = await _router(GatewayConfig(), built_in)._deliver_to_platform(
        DeliveryTarget.parse(TARGET), "（无输出）", metadata=None,
    )

    assert declared.calls == []                                   # declared ⇒ dropped
    assert declared_result["filtered"] == "silence_narration"
    assert len(built_in.calls) == 1                               # nothing declared ⇒ delivered
    assert built_in.calls[0]["content"] == "（无输出）"
    assert built_in_result.get("filtered") is None


@pytest.mark.asyncio
async def test_declared_length_guard_bounds_the_class(tmp_path, monkeypatch):
    """The 133-byte narration falls under the declared guard, and only under it."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    long_narration = MEASURED_NARRATIONS[-1]
    generous = RecordingAdapter()
    narrow = RecordingAdapter()

    generous_result = await _router(_declared_config(), generous)._deliver_to_platform(
        DeliveryTarget.parse(TARGET), long_narration, metadata=None,
    )
    narrow_result = await _router(_declared_config(max_chars=10), narrow)._deliver_to_platform(
        DeliveryTarget.parse(TARGET), long_narration, metadata=None,
    )

    assert generous.calls == []
    assert generous_result["filtered"] == "silence_narration"
    assert len(narrow.calls) == 1                                 # declared guard refuses it
    assert narrow_result.get("filtered") is None


@pytest.mark.asyncio
async def test_invalid_declaration_keeps_the_built_in_class(tmp_path, monkeypatch):
    """An absent or unparseable declaration is not a widening: the shipped class stands."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    for declaration in (
        None,
        "silence_narration: on",
        {},
        {"tokens": "not-a-list"},
        {"tokens": []},
        {"tokens": [1, 2]},
        {"max_chars": "sixty"},
        {"max_chars": -3},
    ):
        data = {} if declaration is None else {"silence_narration": declaration}
        adapter = RecordingAdapter()
        router = _router(GatewayConfig.from_dict({"gateway": data}), adapter)

        english = await router._deliver_to_platform(
            DeliveryTarget.parse(TARGET), "_(silent)_", metadata=None,
        )
        undeclared = await router._deliver_to_platform(
            DeliveryTarget.parse(TARGET), "（无输出）", metadata=None,
        )

        assert english["filtered"] == "silence_narration", declaration
        assert undeclared.get("filtered") is None, declaration
        assert [call["content"] for call in adapter.calls] == ["（无输出）"], declaration


@pytest.mark.asyncio
async def test_a_message_merely_containing_a_declared_token_is_delivered(tmp_path, monkeypatch):
    """The anchoring is preserved: a declared token inside prose is chat, not a narration."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = _router(_declared_config(), adapter)
    contents = [
        "Tăcere în sală — iată planul de lucru pentru mâine.",
        "Am notat fără livrare pentru marți, restul rămâne cum am stabilit.",
    ]

    for content in contents:
        result = await router._deliver_to_platform(DeliveryTarget.parse(TARGET), content, metadata=None)
        assert result.get("filtered") is None, content

    assert [call["content"] for call in adapter.calls] == contents


@pytest.mark.asyncio
async def test_config_yaml_declaration_reaches_the_filter(tmp_path, monkeypatch):
    """The live path: ``gateway.silence_narration`` in config.yaml → the same two loader steps."""
    from gateway import config_loader

    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  silence_narration:\n"
        "    max_chars: 200\n"
        "    tokens:\n" + "".join(f"      - {token}\n" for token in DECLARED_VOCABULARY),
        encoding="utf-8",
    )
    gw_data: dict = {}
    config_loader.load_yaml_layer(tmp_path, gw_data)
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = _router(GatewayConfig.from_dict(gw_data), adapter)

    result = await router._deliver_to_platform(
        DeliveryTarget.parse(TARGET), MEASURED_NARRATIONS[0], metadata=None,
    )

    assert adapter.calls == []
    assert result["filtered"] == "silence_narration"


# --- Config round-trip ------------------------------------------------------


