"""A user row merged into the one before it replays the bytes its turn was sent with.

``repair_message_sequence`` folds consecutive user rows into one. A cron delivery mirrored into the chat
is a user-role row on purpose (``cron/scheduler_delivery.py``), so the next message in that chat always
merges into it. That message's injected context (memory prefetch, ``pre_llm_call`` notes) lives only in
its ``api_content`` sidecar. The turn itself sends it, but the next turn rebuilds the history from the
session DB and merges the rows again, so the merge has to carry the sidecar or the request stops
matching at that row.
"""

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.turn_context import compose_user_api_content
from tests.agent import test_in_place_preflight_rewind as _rewind
from tests.agent.test_in_place_preflight_rewind import _turn

session = _rewind.session  # shared fixture: AIAgent with a mocked client over a temp SessionDB

CRON_MIRROR = "[Cron delivery: Morning brief]\nThree meetings today, the first at 10."


def _recall(_agent, message, *_args, **_kwargs):
    return f"Recalled while answering: {message}"


def _user_row_sent(request, text):
    rows = [m["content"] for m in request if m.get("role") == "user" and text in str(m.get("content"))]
    assert len(rows) == 1, rows
    return rows[0]


def test_reply_after_a_cron_delivery_replays_what_it_sent(session, monkeypatch):
    db, agent = session
    monkeypatch.setattr("agent.turn_context._memory_turn_start_and_prefetch", _recall)
    chat = SimpleNamespace(conversation_history=[])
    _turn(db, agent, chat, "gateway", 1, 5_000)
    db.append_message("sid", "user", CRON_MIRROR)  # what gateway.mirror stores for an attached cron job
    _turn(db, agent, chat, "gateway", 2, 5_000)
    _turn(db, agent, chat, "gateway", 3, 5_000)

    requests = [c.kwargs["messages"] for c in agent.client.chat.completions.create.call_args_list]
    prompt = "U2 please continue with the next step"
    sent = _user_row_sent(requests[1], prompt)
    assert CRON_MIRROR in sent and _recall(None, prompt) in sent
    assert _user_row_sent(requests[2], prompt) == sent


def _stamped(text, recall, **fields):
    """A prompt row as the turn prologue leaves it: the sidecar carries the fenced recalled context."""
    return {"role": "user", "content": text, "api_content": compose_user_api_content(text, recall, ""), **fields}


_PERSISTED = {_DB_PERSISTED_MARKER: True}


@pytest.mark.parametrize(("rows", "sent", "rewritten"), [
    pytest.param(
        [{"role": "user", "content": CRON_MIRROR, **_PERSISTED}, _stamped("ask", "M1")],
        compose_user_api_content(CRON_MIRROR + "\n\nask", "M1", ""), True, id="mirror-then-prompt"),
    pytest.param(
        [_stamped("first", "M0", **_PERSISTED), _stamped("second", "M1")],
        compose_user_api_content("first", "M0", "") + "\n\n" + compose_user_api_content("second", "M1", ""),
        True, id="unanswered-prompt-then-prompt"),
    pytest.param(
        [{"role": "user", "content": "unanswered", **_PERSISTED}, _stamped("", "M1")],
        compose_user_api_content("unanswered", "M1", ""), True, id="empty-turn-with-context"),
    pytest.param(
        [{"role": "user", "content": CRON_MIRROR, **_PERSISTED}, {"role": "user", "content": ""}],
        CRON_MIRROR, False, id="empty-turn-nothing-injected"),
])
def test_merged_row_carries_what_each_side_sent(rows, sent, rewritten):
    assert repair_message_sequence(None, rows) == 1
    merged = rows[0]
    assert merged.get("api_content", merged["content"]) == sent
    assert (_DB_PERSISTED_MARKER not in merged) is rewritten
