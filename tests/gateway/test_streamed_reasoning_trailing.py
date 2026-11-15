"""show_reasoning on a streamed reply (#7251).

With streaming on, the answer reaches the chat through the stream consumer, and the gateway's own
final send is skipped (``already_sent``). The reasoning block that ``show_reasoning`` prepends to
the final response only ever rode that skipped send, so a streamed reply never showed it. It now
follows the streamed answer as its own message.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType, SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner


async def _deliver(monkeypatch, *, show_reasoning: bool, already_sent: bool, is_bot: bool = False,
                   uncertain: bool = False, ledger=None, saved: bool = True):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"display": {"show_reasoning": show_reasoning}})
    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    if ledger is not None:
        async def _ledger(event, session_key, response, *, error=None, **_kw):
            ledger.append(response)
            return saved

        adapter.ledger_attempted_final = _ledger
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", is_bot=is_bot)
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")
    result = {"final_response": "the answer is 42", "last_reasoning": "thinking hard about it",
              "already_sent": already_sent}
    if uncertain:
        result["delivery_uncertain"] = True
    response = runner._hmwa_prepend_reasoning(result, "the answer is 42", source, False)

    class _Entry:
        session_id = "s1"

    returned = await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), "agent:main:telegram:dm:42", None, result, [], response, None, False)
    return adapter, returned


@pytest.mark.asyncio
async def test_streamed_reply_is_followed_by_its_reasoning(monkeypatch):
    adapter, returned = await _deliver(monkeypatch, show_reasoning=True, already_sent=True)

    assert returned is None  # the streamed answer is not sent again
    sent = [call["content"] for call in adapter.sent]
    assert len(sent) == 1 and "thinking hard about it" in sent[0]
    assert "the answer is 42" not in sent[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("show_reasoning, already_sent", [(False, True), (True, False)])
async def test_no_trailing_reasoning_when_hidden_or_not_streamed(monkeypatch, show_reasoning, already_sent):
    adapter, returned = await _deliver(monkeypatch, show_reasoning=show_reasoning, already_sent=already_sent)

    assert adapter.sent == []
    if not already_sent:  # the caller's normal send carries the reasoning above the answer
        assert returned.startswith("💭") and returned.endswith("the answer is 42")


@pytest.mark.asyncio
async def test_a_bot_peer_gets_no_trailing_reasoning_message(monkeypatch):
    """A second message to a bot peer is fresh input that can start another paid turn."""
    adapter, returned = await _deliver(monkeypatch, show_reasoning=True, already_sent=True, is_bot=True)

    assert returned is None
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_an_uncertain_streamed_final_carries_its_reasoning_only_once(monkeypatch):
    """The ledgered redelivery of an uncertain streamed final already holds the reasoning block."""
    ledger: list[str] = []
    adapter, returned = await _deliver(monkeypatch, show_reasoning=True, already_sent=True,
                                       uncertain=True, ledger=ledger)

    assert returned is None
    assert adapter.sent == []
    assert len(ledger) == 1 and "thinking hard about it" in ledger[0] and ledger[0].endswith("the answer is 42")


@pytest.mark.asyncio
async def test_an_uncertain_streamed_final_with_the_ledger_off_still_shows_its_reasoning(monkeypatch):
    """``gateway.delivery_ledger: false`` saves no redelivery row, so nothing else will carry the reasoning."""
    ledger: list[str] = []
    adapter, returned = await _deliver(monkeypatch, show_reasoning=True, already_sent=True,
                                       uncertain=True, ledger=ledger, saved=False)

    assert returned is None
    sent = [call["content"] for call in adapter.sent]
    assert len(sent) == 1 and "thinking hard about it" in sent[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_the_base_ledger_reports_whether_it_saved_a_redelivery(monkeypatch, tmp_path, enabled):
    """The real adapter seam: with ``gateway.delivery_ledger`` off nothing is saved, and it says so."""
    from gateway import delivery_ledger as dl

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: enabled)
    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")

    saved = await adapter.ledger_attempted_final(event, "agent:main:telegram:dm:42", "the answer is 42",
                                                 error="Timed out")

    assert saved is enabled
