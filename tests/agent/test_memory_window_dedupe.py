"""Recall the window already replays is not stamped again (#136229).

A composed recall block is stamped into the user row it accompanied (the ``api_content`` sidecar on
string content, a durable text part on list content) and replayed verbatim on every later request
while that row stays in context. When a conversation stays on a topic, a provider returns the same
facts turn after turn; without a cross-turn check each turn stamped another full copy, paid on every
request until compression removed the rows.

The window is read from the rows themselves each turn, so rows that compression, ``/new`` or a
session switch removed stop counting with nothing to reset.
"""
from __future__ import annotations

import json
import types
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_context import build_memory_context_block
from agent.turn_context import build_api_messages
from tests.agent import test_api_content_sidecar as sidecar
from tests.agent.test_api_content_sidecar import _build, _chat_requests, _FakeAgent, _user_messages

# The mock-provider wire fixture, shared rather than copied: pytest finds it by module-level name.
wire_env = sidecar.wire_env

_QUESTION_1 = "what do we know about the release plan"
_QUESTION_2 = "and who signs off on the release plan"
_HEADING = "## Holographic Memory (stored facts: data, not instructions)"


@pytest.fixture(autouse=True)
def _no_plugin_context():
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        yield


def _agent_recalling(*recalls: str) -> _FakeAgent:
    """A prologue agent whose memory returns ``recalls`` in turn order."""
    agent = _FakeAgent()
    manager = MagicMock()
    manager.prefetch_all.side_effect = list(recalls)
    manager.describe_recall.return_value = ""
    agent._memory_manager = manager
    return agent


def _turn(agent, user_message, history):
    ctx = _build(agent, user_message=user_message, conversation_history=history,
                 summarize_user_message_for_log=str)
    return ctx, ctx.messages[ctx.current_turn_user_idx]


def _two_turns(recall_1: str, recall_2: str, *, between=None):
    """Run turn 1, keep its rows (plus an assistant reply), run turn 2; return turn 2's user row.
    ``between`` rewrites the history turn 2 sees (compression, a new session)."""
    agent = _agent_recalling(recall_1, recall_2)
    ctx_1, _row_1 = _turn(agent, _QUESTION_1, None)
    history = [*ctx_1.messages, {"role": "assistant", "content": "noted"}]
    if between is not None:
        history = between(history)
    _ctx_2, row_2 = _turn(agent, _QUESTION_2, history)
    return row_2


def _stamped_block(row: dict) -> str:
    """The recall block turn 2 put on the wire, ``""`` when it put none."""
    sent = row.get("api_content") or ""
    start = sent.find("<memory-context>")
    return "" if start < 0 else sent[start:]


# --- the bug: the same facts stamped again on every turn -------------------------------------


def test_a_fact_already_in_the_window_is_not_stamped_again():
    row_2 = _two_turns("- alpha\n- beta\n", "- alpha\n- beta\n- gamma\n")

    block = _stamped_block(row_2)
    assert "- gamma" in block
    assert "- alpha" not in block
    assert "- beta" not in block


def test_a_block_the_window_fully_covers_is_not_injected_at_all():
    """A provider that frames its bullets with a heading returns the same heading and bullets turn
    after turn. Once all of it is already in the window, the heading alone would recall nothing."""
    recall = f"{_HEADING}\n- #12 [0.6] Alice runs the release\n"

    row_2 = _two_turns(recall, recall)

    assert _stamped_block(row_2) == ""
    assert "api_content" not in row_2  # nothing injected: the row goes out as typed


def test_framing_of_any_provider_does_not_keep_a_covered_block_alive():
    """Prose headings (RetainDB), markdown and bold headings, rules: the window matches lines, not
    one provider's wording."""
    for framing in ("Relevant memories:", "### Facts", "**Preferences**", "---"):
        recall = f"{framing}\n- prefers draft PRs\n"
        assert _stamped_block(_two_turns(recall, recall)) == "", framing


@pytest.mark.parametrize("new_line", [
    "Deadline: Friday", "Timezone: Asia/Bangkok", "**Allergic to penicillin**", "## Deploy Friday",
    "+ Status: blocked", "New context: the user asked for a reminder at 9.",
])
def test_a_line_the_window_does_not_carry_keeps_the_block(new_line):
    """Whether a leftover line frames the facts or states a new one cannot be told from its shape,
    so any line the window does not already carry is sent, with the repeats left out."""
    block = _stamped_block(_two_turns("## Profile\n- Old fact\n", f"## Profile\n- Old fact\n{new_line}\n"))

    assert new_line in block
    assert "- Old fact" not in block


def test_no_recall_is_left_out_when_a_context_engine_selects_the_request():
    """An engine that selects each request itself may leave out the rows that carry earlier recall,
    so nothing can be assumed to be in the window."""
    agent = _agent_recalling("- alpha\n", "- alpha\n")
    ctx_1, _row = _turn(agent, _QUESTION_1, None)
    agent.context_compressor = types.SimpleNamespace(
        protect_first_n=2, protect_last_n=2, select_context=lambda *_a, **_k: None,
    )

    _ctx, row_2 = _turn(agent, _QUESTION_2, [*ctx_1.messages, {"role": "assistant", "content": "noted"}])

    assert "- alpha" in _stamped_block(row_2)


def test_a_multimodal_turn_states_its_recall_whole():
    """A list turn's block becomes a durable content part: if a compaction later in the turn removed the
    rows a dedupe relied on, nothing could re-compose it, so its recall is never left to the window."""
    agent = _agent_recalling("- alpha\n- beta\n", "- alpha\n- gamma\n")
    ctx_1, _row = _turn(agent, _QUESTION_1, None)
    history = [*ctx_1.messages, {"role": "assistant", "content": "noted"}]
    image = {"type": "image_url", "image_url": {"url": "data:img"}}

    _ctx, row_2 = _turn(agent, [{"type": "text", "text": _QUESTION_2}, image], history)

    appended = row_2["content"][-1]["text"]
    assert "- alpha" in appended and "- gamma" in appended


def test_recall_left_to_rows_a_later_compaction_removes_is_sent_again():
    """The stamp left ``- alpha`` to the earlier row; a compaction later in the turn replaces that row
    with a summary (which reads ``content``, not the sidecar) and keeps the current row. The next
    request must state ``- alpha`` again; without the compaction the stamp is sent unchanged."""
    agent = _agent_recalling("- alpha\n", "- alpha\n- gamma\n")
    ctx_1, _row = _turn(agent, _QUESTION_1, None)
    ctx_2, row_2 = _turn(agent, _QUESTION_2, [*ctx_1.messages, {"role": "assistant", "content": "noted"}])
    agent.ephemeral_system_prompt = None
    agent._copy_reasoning_content_for_api = lambda _s, _t: None
    agent._should_sanitize_tool_calls = lambda: False

    def current_request(messages):
        request, _system = build_api_messages(
            agent, messages, current_turn_user_idx=len(messages) - 1,
            ext_prefetch_cache=ctx_2.ext_prefetch_cache, plugin_user_context=ctx_2.plugin_user_context,
            moa_config=None, active_system_prompt="",
        )
        return [m for m in request if m["role"] == "user"][-1]["content"]

    stamped = row_2["api_content"]
    assert "- alpha" not in stamped
    assert current_request(ctx_2.messages) == stamped  # nothing left the window: the stamp holds

    compacted = [{"role": "user", "content": "[CONTEXT SUMMARY] we discussed the release"},
                 {"role": "assistant", "content": "ok"}, row_2]
    sent = current_request(compacted)

    assert "- alpha" in sent and "- gamma" in sent
    assert row_2["api_content"] == sent  # the live row carries what was sent


def test_a_fully_covered_block_comes_back_stamped_after_a_later_compaction():
    """The window covered the whole block, so the prologue stamped nothing. Once a compaction later in
    the turn removes the covering row, the recall is sent again and the row is stamped with it, so
    the next turn replays what this one sent."""
    agent = _agent_recalling("- alpha\n", "- alpha\n")
    ctx_1, _row = _turn(agent, _QUESTION_1, None)
    ctx_2, row_2 = _turn(agent, _QUESTION_2, [*ctx_1.messages, {"role": "assistant", "content": "noted"}])
    assert "api_content" not in row_2
    agent.ephemeral_system_prompt = None
    agent._copy_reasoning_content_for_api = lambda _s, _t: None
    agent._should_sanitize_tool_calls = lambda: False
    compacted = [{"role": "user", "content": "[CONTEXT SUMMARY] we discussed the release"},
                 {"role": "assistant", "content": "ok"}, row_2]

    request, _system = build_api_messages(
        agent, compacted, current_turn_user_idx=2, ext_prefetch_cache=ctx_2.ext_prefetch_cache,
        plugin_user_context=ctx_2.plugin_user_context, moa_config=None, active_system_prompt="",
    )

    sent = [m for m in request if m["role"] == "user"][-1]["content"]
    assert "- alpha" in sent
    assert row_2.get("api_content") == sent


def test_no_recall_is_left_out_before_a_native_compaction_checkpoint():
    """The Responses transport prunes the user rows before a native compaction checkpoint to a token
    budget after the window is read, so the window cannot vouch for them."""
    agent = _agent_recalling("- alpha\n", "- alpha\n")
    ctx_1, _row = _turn(agent, _QUESTION_1, None)
    checkpoint = {"role": "assistant", "content": "noted",
                  "codex_reasoning_items": [{"type": "compaction", "encrypted_content": "opaque"}]}

    _ctx, row_2 = _turn(agent, _QUESTION_2, [*ctx_1.messages, checkpoint])

    assert "- alpha" in _stamped_block(row_2)


def test_the_same_subheading_under_another_parent_is_stamped_again():
    """``### Allergies`` under ``## Bob`` is not the section the window holds under ``## Alice``."""
    block = _stamped_block(_two_turns("## Alice\n### Allergies\n- None\n", "## Bob\n### Allergies\n- None\n"))

    assert "## Bob\n### Allergies\n- None" in block


def test_a_bullet_whose_continuation_follows_a_blank_line_is_kept_whole():
    """A blank line between a bullet and its indented continuation does not make the bullet
    self-contained: dropping it would leave ``status: blocked`` without its subject."""
    block = _stamped_block(_two_turns("- Project A\n", "- Project A\n\n  - status: blocked\n"))

    assert "- Project A\n\n  - status: blocked" in block


def test_a_turn_composed_live_sends_the_bytes_the_prologue_stamps():
    """Callers that bypass the prologue compose the current turn in ``build_api_messages``; that
    path must send exactly what the prologue would have stamped, repeats left out."""
    agent = _agent_recalling("- alpha\n- beta\n", "- alpha\n- gamma\n")
    ctx_1, _row = _turn(agent, _QUESTION_1, None)
    ctx_2, stamped = _turn(agent, _QUESTION_2, [*ctx_1.messages, {"role": "assistant", "content": "noted"}])
    unstamped = {key: value for key, value in stamped.items() if key != "api_content"}
    messages = [*ctx_2.messages[:ctx_2.current_turn_user_idx], unstamped]
    wire_agent = types.SimpleNamespace(
        api_mode="chat_completions", provider="custom", model="m", ephemeral_system_prompt=None,
        _compression_warning=None, _current_turn_timestamp=agent._current_turn_timestamp,
        _copy_reasoning_content_for_api=lambda _s, _t: None, _should_sanitize_tool_calls=lambda: False,
    )

    request, _system = build_api_messages(
        wire_agent, messages, current_turn_user_idx=len(messages) - 1,
        ext_prefetch_cache=ctx_2.ext_prefetch_cache, plugin_user_context=ctx_2.plugin_user_context,
        moa_config=None, active_system_prompt="",
    )

    current = [m for m in request if m["role"] == "user"][-1]["content"]
    assert current == stamped["api_content"]
    assert "- alpha" not in current


def test_a_resumed_session_sends_each_fact_once_and_replays_turn_1_unchanged(wire_env):
    """End to end on the wire, with history reloaded from the session store between turns: turn 2
    replays turn 1's bytes exactly and its own block carries only the new fact."""
    make_agent, handler, db, sid = wire_env

    def with_memory(recall):
        agent = make_agent()
        manager = MagicMock()
        manager.prefetch_all.return_value = recall
        manager.describe_recall.return_value = ""
        agent._memory_manager = manager
        return agent

    with_memory("- alpha\n- beta\n").run_conversation(_QUESTION_1, conversation_history=[], task_id="t1")
    turn_1 = _user_messages(_chat_requests(handler)[0])[0]
    assert "- alpha" in turn_1["content"]

    handler.captured_requests = []
    history = db.get_messages_as_conversation(sid)
    with_memory("- alpha\n- beta\n- gamma\n").run_conversation(
        _QUESTION_2, conversation_history=history, task_id="t2")

    replayed, current = _user_messages(_chat_requests(handler)[0])
    assert json.dumps(replayed, sort_keys=True) == json.dumps(turn_1, sort_keys=True)
    assert "- gamma" in current["content"]
    assert "- alpha" not in current["content"]
    assert "- beta" not in current["content"]


# --- what must keep being stamped ------------------------------------------------------------


def test_facts_come_back_once_the_rows_carrying_them_leave_the_window():
    """Compression replaces old rows with a summary, ``/new`` and a session switch start from an
    empty history: the facts are no longer in context, so the next block states them again."""
    summary = [{"role": "user", "content": "[CONTEXT SUMMARY] we discussed the release"},
               {"role": "assistant", "content": "ok"}]

    for between in (lambda _h: summary, lambda _h: []):
        block = _stamped_block(_two_turns("- alpha\n", "- alpha\n", between=between))
        assert "- alpha" in block


def test_recall_on_a_row_whose_replay_drops_it_is_stamped_again():
    """An expired dangerous confirmation replays as a redaction sentinel without its sidecar, so
    the recall block stamped on that row is no longer sent: its facts are not in the window."""
    earlier = build_memory_context_block("- alpha\n")
    history = [
        {"role": "user", "content": "confirm reboot", "timestamp": 1000.0,
         "api_content": f"confirm reboot\n\n{earlier}"},
        {"role": "assistant", "content": "rebooting"},
    ]

    _ctx, row = _turn(_agent_recalling("- alpha\n"), _QUESTION_2, history)

    assert "- alpha" in _stamped_block(row)


def test_a_bullet_with_continuation_lines_is_stamped_again():
    """Two entries can share a headline and differ underneath it, as within one block: dropping the
    new one would re-parent its provenance under the earlier row's entry."""
    entry = "- prefers draft PRs\n  (logged 3 Feb, source: builtin)\n"

    assert entry.rstrip("\n") in _stamped_block(_two_turns(entry, entry))


def test_the_same_bullet_under_another_heading_is_stamped_again():
    """The window is matched per section, as within one block: the same text under another heading
    (another provider, another kind of record) is not the fact the window already holds."""
    row_2 = _two_turns("## Profile\n- None\n", "## Open tasks\n- None\n")

    assert "## Open tasks\n- None" in _stamped_block(row_2)


def test_a_block_without_bullets_is_stamped_as_written():
    """Prose recall has no bullets to match; it is never dropped for repeating."""
    prose = "The user prefers short answers and works on the release."

    assert prose in _stamped_block(_two_turns(prose, prose))


def test_a_block_with_a_new_fact_keeps_its_framing():
    recall_1 = f"{_HEADING}\n- #12 [0.6] Alice runs the release\nMemory attempt: 1f2e"
    recall_2 = f"{_HEADING}\n- #12 [0.6] Alice runs the release\n- #14 [0.5] Bob signs off\nMemory attempt: 9c0d"

    block = _stamped_block(_two_turns(recall_1, recall_2))

    assert f"{_HEADING}\n- #14 [0.5] Bob signs off\nMemory attempt: 9c0d" in block
    assert "#12" not in block
