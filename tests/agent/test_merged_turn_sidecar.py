"""A merged user row replays the bytes both of its turns were sent with (regression for #121836).

The pre-call alternation repair folds a role=user notice row (the model-switch marker, or the
/reload-mcp notice) and the next prompt into one row. That prompt's injected context (memory
prefetch, ``pre_llm_call`` notes) lives only in its ``api_content`` sidecar, so the merge has to
carry the sidecar onto the surviving row. Otherwise the turn itself still sends the context, but
every later request replays the row without it, and the request prefix stops matching there.
"""

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.turn_context import compose_user_api_content
from tests.agent import test_in_place_preflight_rewind as _rewind
from tests.agent.test_in_place_preflight_rewind import _turn

session = _rewind.session  # shared fixture: AIAgent with a mocked client over a temp SessionDB

SWITCH = {"role": "user", "display_kind": "model_switch",
          "content": "[System: The active model for this chat has changed to X via provider Y.]"}
RELOAD = {"role": "user",  # /reload-mcp appends a plain user row: no display_kind
          "content": "[IMPORTANT: MCP servers have been reloaded. Added servers: example.]"}


def _recall(_agent, message, *_args, **_kwargs):
    return f"Recalled while answering: {message}"


def _row_sent_with(request, text):
    rows = [m["content"] for m in request if m.get("role") == "user" and text in str(m.get("content"))]
    assert len(rows) == 1, rows
    return rows[0]


@pytest.mark.parametrize("surface", ["cli", "gateway"])
@pytest.mark.parametrize("notice", [SWITCH, RELOAD], ids=["model-switch", "reload-mcp"])
def test_turn_after_a_notice_replays_what_it_sent(session, surface, notice, monkeypatch):
    db, agent = session
    monkeypatch.setattr("agent.turn_context._memory_turn_start_and_prefetch", _recall)
    cli = SimpleNamespace(conversation_history=[])
    _turn(db, agent, cli, surface, 1, 5_000)
    if surface == "gateway":  # the gateway reloads the transcript, which holds the notice row
        db.append_message("sid", "user", notice["content"], display_kind=notice.get("display_kind"))
    else:  # the classic CLI appends it to its in-memory history
        cli.conversation_history.append(dict(notice))
    _turn(db, agent, cli, surface, 2, 5_000)
    _turn(db, agent, cli, surface, 3, 5_000)

    requests = [c.kwargs["messages"] for c in agent.client.chat.completions.create.call_args_list]
    prompt = "U2 please continue with the next step"
    sent = _row_sent_with(requests[1], prompt)
    assert notice["content"] in sent and _recall(None, prompt) in sent
    assert _row_sent_with(requests[2], prompt) == sent
    durable = [m["content"] for m in db.get_messages_as_conversation("sid", repair_alternation=True)
               if m.get("role") == "user" and prompt in m["content"]]
    assert len(durable) == 1 and durable[0].count(prompt) == 1


def _stamped(text, recall, **fields):
    """A prompt row as the turn prologue leaves it: the sidecar carries the fenced recalled context."""
    return {"role": "user", "content": text, "api_content": compose_user_api_content(text, recall, ""), **fields}


_PERSISTED = {_DB_PERSISTED_MARKER: True}
# rows before the merge -> (bytes the merged row replays, whether its durable row must be re-written)
MERGES = {
    "notice-then-prompt": (
        [dict(RELOAD, **_PERSISTED), _stamped("ask", "M1")],
        compose_user_api_content(RELOAD["content"] + "\n\nask", "M1", ""), True),
    "unanswered-prompt-then-prompt": (
        [_stamped("first", "M0", **_PERSISTED), _stamped("second", "M1")],
        compose_user_api_content("first", "M0", "") + "\n\n" + compose_user_api_content("second", "M1", ""), True),
    "empty-turn-with-context": (
        [{"role": "user", "content": "unanswered", **_PERSISTED}, _stamped("", "M1")],
        compose_user_api_content("unanswered", "M1", ""), True),
    "empty-turn-without-context": (
        [{"role": "user", "content": "unanswered", **_PERSISTED}, {"role": "user", "content": ""}],
        "unanswered", False),
}


@pytest.mark.parametrize("rows, wire, rewritten", list(MERGES.values()), ids=list(MERGES))
def test_merged_user_row_keeps_what_each_turn_sent(rows, wire, rewritten):
    messages = [{"role": "assistant", "content": "reply"}, *(dict(row) for row in rows)]

    assert repair_message_sequence(None, messages) == 1

    merged = messages[1]
    assert merged.get("api_content", merged["content"]) == wire
    # The marker claims the durable row matches the dict, so a row whose bytes changed must lose it.
    assert (_DB_PERSISTED_MARKER not in merged) is rewritten
