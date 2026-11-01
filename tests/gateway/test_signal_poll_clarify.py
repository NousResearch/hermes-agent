"""Signal: multiple-choice clarify rendered as a native poll; votes resolve the clarify."""
import asyncio
import json
import uuid
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from tools import clarify_gateway as cg

ACCOUNT = "+15550000001"
USER = "+15550000002"
POLL_TS = 1792870692665


def _adapter(monkeypatch, allowed=USER):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", allowed)
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": ACCOUNT}
    a = SignalAdapter(config)
    a._POLL_DEBOUNCE_SECONDS = 0.01
    a._stop_typing_indicator = AsyncMock()
    a.send = AsyncMock()
    a.calls = []

    async def rpc(method, params, rpc_id=None, **kw):
        a.calls.append((method, dict(params)))
        if method == "sendPollCreate":
            return {"results": [{"type": "SUCCESS"}], "timestamp": POLL_TS}
        return {}
    a._rpc = rpc

    async def with_target(params, chat_id, *, resolve=True):
        params["recipient"] = [chat_id]
        return params
    a._with_target = with_target
    return a


def _register(choices, multi=False):
    cid = uuid.uuid4().hex
    cg.register(cid, f"sess-{cid}", "Which one?", choices, multi_select=multi)
    return cid


def _vote(indexes, ts=POLL_TS):
    return {"author": ACCOUNT, "targetSentTimestamp": ts, "optionIndexes": indexes, "voteCount": 1}


async def _settle():
    """Wait for every pending poll debounce/resolve task (no wall-clock sleep)."""
    tasks = [x for x in asyncio.all_tasks() if x.get_name().startswith("signal-clarify-poll-")]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_send_clarify_creates_poll_with_other(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"])
    res = await a.send_clarify(USER, "Which one?", ["Red", "Green"], cid, "s")
    assert res.success and res.message_id == str(POLL_TS)
    method, params = a.calls[0]
    assert method == "sendPollCreate"
    assert params["question"] == "Which one?"
    assert params["option"][:2] == ["Red", "Green"] and len(params["option"]) == 4
    assert params["noMulti"] is True and params["recipient"] == [USER]
    assert cg._entries[cid].awaiting_text is True  # typed replies still accepted


@pytest.mark.asyncio
async def test_vote_resolves_clarify_and_terminates_poll(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green", "Blue"])
    await a.send_clarify(USER, "Which one?", ["Red", "Green", "Blue"], cid, "s")
    await a._handle_poll_vote(_vote([1]), USER, "", USER)  # optionIndexes are 0-based
    await _settle()
    assert cg._entries[cid].response == "Green"
    assert any(m == "sendPollTerminate" and p["pollTimestamp"] == POLL_TS for m, p in a.calls)


@pytest.mark.asyncio
async def test_latest_vote_wins_within_debounce(monkeypatch):
    a = _adapter(monkeypatch)
    a._POLL_DEBOUNCE_SECONDS = 0.05
    cid = _register(["Red", "Green", "Blue"])
    await a.send_clarify(USER, "q", ["Red", "Green", "Blue"], cid, "s")
    await a._handle_poll_vote(_vote([0]), USER, "", USER)
    await a._handle_poll_vote(_vote([]), USER, "", USER)   # untap
    await a._handle_poll_vote(_vote([2]), USER, "", USER)
    await _settle()
    assert cg._entries[cid].response == "Blue"


@pytest.mark.asyncio
async def test_deselection_alone_does_not_resolve(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    await a._handle_poll_vote(_vote([0]), USER, "", USER)
    await a._handle_poll_vote(_vote([]), USER, "", USER)
    await _settle()
    assert not cg._entries[cid].event.is_set()


@pytest.mark.asyncio
async def test_other_switches_to_typed_answer(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    await a._handle_poll_vote(_vote([2]), USER, "", USER)
    await _settle()
    assert not cg._entries[cid].event.is_set()
    assert cg._entries[cid].awaiting_text is True
    a.send.assert_awaited()


@pytest.mark.asyncio
async def test_multi_select_returns_json_array(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green", "Blue"], multi=True)
    await a.send_clarify(USER, "q", ["Red", "Green", "Blue"], cid, "s")
    assert a.calls[0][1]["noMulti"] is False
    await a._handle_poll_vote(_vote([0, 2]), USER, "", USER)
    await _settle()
    assert not cg._entries[cid].event.is_set()  # nothing counts until Done
    await a._handle_poll_vote(_vote([0, 2, 5]), USER, "", USER)
    await _settle()
    assert json.loads(cg._entries[cid].response) == ["Red", "Blue"]


@pytest.mark.asyncio
async def test_unauthorized_and_cross_chat_votes_ignored(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    await a._handle_poll_vote(_vote([0]), "+15559999999", "", USER)        # not allowlisted
    await a._handle_poll_vote(_vote([0]), USER, "", "group:abc")           # other chat
    await a._handle_poll_vote(_vote([0], ts=123), USER, "", USER)          # untracked poll
    await _settle()
    assert not cg._entries[cid].event.is_set()


@pytest.mark.asyncio
async def test_late_vote_after_answer_is_swallowed(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    await a._handle_poll_vote(_vote([0]), USER, "", USER)
    await _settle()
    await a._handle_poll_vote(_vote([1]), USER, "", USER)
    await _settle()
    assert cg._entries[cid].response == "Red"


@pytest.mark.asyncio
async def test_fallback_to_numbered_list_when_poll_fails(monkeypatch):
    a = _adapter(monkeypatch)

    async def failing(method, params, rpc_id=None, **kw):
        return None
    a._rpc = failing
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "Which one?", ["Red", "Green"], cid, "s")
    text = a.send.await_args.kwargs.get("content") or a.send.await_args.args[1]
    assert "Red" in text and "Green" in text


@pytest.mark.asyncio
async def test_open_ended_clarify_is_plain_text(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(None)
    await a.send_clarify(USER, "Anything else?", None, cid, "s")
    assert not any(m == "sendPollCreate" for m, _ in a.calls)
    a.send.assert_awaited()


@pytest.mark.asyncio
async def test_long_question_sent_as_text_before_poll(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Yes", "No"])
    long_q = "Context. " * 40
    await a.send_clarify(USER, long_q, ["Yes", "No"], cid, "s")
    assert a.send.await_count == 1
    assert a.calls[0][1]["question"] == "Choose one:"


@pytest.mark.asyncio
async def test_retire_clarify_card_terminates_poll(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    await a.retire_clarify_card(cid, "answered")
    assert any(m == "sendPollTerminate" for m, _ in a.calls)
    await a._handle_poll_vote(_vote([0]), USER, "", USER)
    await _settle()
    assert not cg._entries[cid].event.is_set()


@pytest.mark.asyncio
async def test_poll_vote_envelope_never_reaches_agent(monkeypatch):
    a = _adapter(monkeypatch)
    a.handle_message = AsyncMock()
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    env = {"envelope": {"source": USER, "sourceNumber": USER, "sourceUuid": "u-1", "sourceName": "M",
                        "timestamp": 1, "dataMessage": {"timestamp": 1, "message": None, "pollVote": _vote([1])}},
           "account": ACCOUNT}
    await a._handle_envelope(env)
    await _settle()
    a.handle_message.assert_not_awaited()
    assert cg._entries[cid].response == "Green"


@pytest.mark.asyncio
async def test_vote_acknowledged_immediately_then_checked(monkeypatch):
    a = _adapter(monkeypatch)
    a._POLL_DEBOUNCE_SECONDS = 0.05
    a.send_reaction = AsyncMock(return_value=True)
    a.send_typing = AsyncMock()
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    await a._handle_poll_vote(_vote([0]), USER, "", USER)
    # Before the debounce elapses: 👀 on the poll (authored by the bot) + typing indicator.
    a.send_reaction.assert_awaited_with(USER, "👀", ACCOUNT, POLL_TS)
    a.send_typing.assert_awaited()
    await a._handle_poll_vote(_vote([1]), USER, "", USER)  # changed vote: no second 👀
    assert a.send_reaction.await_count == 1
    await _settle()
    assert cg._entries[cid].response == "Green"
    assert a.send_reaction.await_args.args[1] == "✅"


@pytest.mark.asyncio
async def test_polls_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SIGNAL_CLARIFY_POLLS", "false")
    a = _adapter(monkeypatch)
    assert a.clarify_polls_enabled is False
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "Which one?", ["Red", "Green"], cid, "s")
    assert not any(m == "sendPollCreate" for m, _ in a.calls)
    a.send.assert_awaited()  # numbered-list fallback


def test_config_extra_overrides_env(monkeypatch):
    monkeypatch.setenv("SIGNAL_CLARIFY_POLLS", "false")
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": ACCOUNT, "clarify_polls": True}
    assert SignalAdapter(config).clarify_polls_enabled is True


@pytest.mark.asyncio
async def test_poll_strings_come_from_i18n_catalog(monkeypatch):
    from agent.i18n import t
    a = _adapter(monkeypatch)
    cid = _register(["Yes", "No"])
    await a.send_clarify(USER, "x" * 300, ["Yes", "No"], cid, "s")
    params = a.calls[0][1]
    assert params["option"][-2] == t("gateway.clarify.poll_other") != "gateway.clarify.poll_other"
    assert params["question"] == t("gateway.clarify.poll_title_single") != "gateway.clarify.poll_title_single"
    assert t("gateway.clarify.poll_type_answer") != "gateway.clarify.poll_type_answer"
    assert t("gateway.clarify.poll_title_multi") != "gateway.clarify.poll_title_multi"


def test_signal_platform_hint_steers_choices_to_clarify():
    from plugins.platforms.signal.register import _PLATFORM_HINT as hint
    assert "clarify" in hint and "poll" in hint


@pytest.mark.asyncio
async def test_skip_option_resolves_as_skipped(monkeypatch):
    from agent.i18n import t
    a = _adapter(monkeypatch)
    a.send_reaction = AsyncMock(return_value=True)
    a.send_typing = AsyncMock()
    cid = _register(["Red", "Green"])
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    assert a.calls[0][1]["option"][-1] == t("gateway.clarify.poll_skip")
    await a._handle_poll_vote(_vote([3]), USER, "", USER)  # Red, Green, Other, Skip
    await _settle()
    assert cg._entries[cid].response == cg.SKIPPED
    assert a.send_reaction.await_args.args[1] == "⏭️"
    assert any(m == "sendPollTerminate" for m, _ in a.calls)


@pytest.mark.asyncio
async def test_real_choice_beats_skip_in_multi_select(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green", "Blue"], multi=True)
    await a.send_clarify(USER, "q", ["Red", "Green", "Blue"], cid, "s")
    await a._handle_poll_vote(_vote([1, 4, 5]), USER, "", USER)  # Green + Skip + Done
    await _settle()
    assert json.loads(cg._entries[cid].response) == ["Green"]


@pytest.mark.asyncio
async def test_skip_hint_stripped_from_poll_title(monkeypatch):
    from agent.i18n import t
    a = _adapter(monkeypatch)
    cid = _register(["Yes", "No"])
    await a.send_clarify(USER, f"Proceed?\n{t('gateway.clarify.skip_hint')}", ["Yes", "No"], cid, "s")
    assert a.calls[0][1]["question"] == "Proceed?"


@pytest.mark.asyncio
async def test_nine_choices_fall_back_to_numbered_list(monkeypatch):
    a = _adapter(monkeypatch)
    choices = [f"c{i}" for i in range(9)]  # 9 + Other + Skip > Signal's 10-option cap
    cid = _register(choices)
    await a.send_clarify(USER, "q", choices, cid, "s")
    assert not any(m == "sendPollCreate" for m, _ in a.calls)


@pytest.mark.asyncio
async def test_multi_select_poll_has_done_option(monkeypatch):
    from agent.i18n import t
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"], multi=True)
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    opts = a.calls[0][1]["option"]
    assert opts[-1] == t("gateway.clarify.poll_done") and len(opts) == 5
    single = _register(["Red", "Green"])
    a.calls.clear()
    await a.send_clarify(USER, "q", ["Red", "Green"], single, "s")
    assert t("gateway.clarify.poll_done") not in a.calls[0][1]["option"]


@pytest.mark.asyncio
async def test_multi_select_done_alone_asks_to_pick_first(monkeypatch):
    from agent.i18n import t
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"], multi=True)
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    await a._handle_poll_vote(_vote([4]), USER, "", USER)  # Done only (Red, Green, Other, Skip, Done)
    await _settle()
    assert not cg._entries[cid].event.is_set()
    assert a.send.await_args.args[1] == t("gateway.clarify.poll_pick_first")
    await a._handle_poll_vote(_vote([1, 4]), USER, "", USER)
    await _settle()
    assert json.loads(cg._entries[cid].response) == ["Green"]


@pytest.mark.asyncio
async def test_multi_select_skip_with_done(monkeypatch):
    a = _adapter(monkeypatch)
    cid = _register(["Red", "Green"], multi=True)
    await a.send_clarify(USER, "q", ["Red", "Green"], cid, "s")
    await a._handle_poll_vote(_vote([3, 4]), USER, "", USER)  # Skip + Done
    await _settle()
    assert cg._entries[cid].response == cg.SKIPPED


@pytest.mark.asyncio
async def test_eight_choices_multi_select_falls_back(monkeypatch):
    a = _adapter(monkeypatch)
    choices = [f"c{i}" for i in range(8)]  # 8 + Other + Skip + Done > 10
    cid = _register(choices, multi=True)
    await a.send_clarify(USER, "q", choices, cid, "s")
    assert not any(m == "sendPollCreate" for m, _ in a.calls)
