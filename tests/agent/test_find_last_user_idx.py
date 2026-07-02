"""Regression tests for ``find_last_user_idx``.

Context: ``current_turn_user_idx`` is snapshotted once at turn-start
(``agent/turn_context.py``), right after the user message is appended.
Two mechanisms mutate ``messages`` afterwards, before the glue site in
``agent/conversation_loop.py`` injects ephemeral plugin/memory context onto
the current turn's user message: preflight context compression (replaces
``messages`` with a shorter list) and ``repair_message_sequence_with_cursor``
(in-place alternation repair — its docstring fixes the SessionDB flush
cursor for #44837, but not this injection cursor). After either mutation,
the stale ``current_turn_user_idx`` matches no element in the final list,
so the glue condition never fires and plugin/memory context is silently
dropped from the request.

``find_last_user_idx`` must be recomputed from the *final* message list at
glue time (a plain reverse scan for the last ``role == "user"`` entry) so
the index is always in-bounds and correct.
"""

from agent.agent_runtime_helpers import find_last_user_idx


def test_normal_tail_returns_last_user_message():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what's up"},
    ]
    assert find_last_user_idx(messages) == 2


def test_stale_index_scenario_after_compression_shrinks_list():
    """Simulates the exact bug: an old index computed against a longer,
    pre-compaction list would point past the end (or at the wrong
    message) once compression replaces ``messages`` with a shorter list.
    The function must return the fresh, correct index for whatever list
    it's given — never rely on a precomputed value."""
    pre_compaction_messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "turn 3 (current)"},
    ]
    stale_idx = find_last_user_idx(pre_compaction_messages)
    assert stale_idx == 4

    # Preflight compression drops/summarizes older turns and rebuilds the
    # list — the current turn's user message survives but at a new index.
    post_compaction_messages = [
        {"role": "user", "content": "[compressed summary]"},
        {"role": "user", "content": "turn 3 (current)"},
    ]
    assert stale_idx not in range(len(post_compaction_messages))  # would be out of bounds
    assert find_last_user_idx(post_compaction_messages) == 1


def test_synthetic_user_nudge_in_tail_is_selected():
    """When scaffolding appends a synthetic nudge user message after the
    real one (e.g. after an interrupted tool sequence), the nudge is what
    actually reaches the API as the outgoing user turn — glue must target
    it, not the earlier real user message."""
    messages = [
        {"role": "user", "content": "real question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "user", "content": "[nudge] please continue"},
    ]
    assert find_last_user_idx(messages) == 3


def test_no_user_message_returns_none():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "assistant", "content": "hello"},
    ]
    assert find_last_user_idx(messages) is None


def test_empty_messages_returns_none():
    assert find_last_user_idx([]) is None


def test_non_dict_elements_do_not_crash():
    messages = [None, "garbage", 42, {"role": "user", "content": "ok"}]
    assert find_last_user_idx(messages) == 3


def test_non_dict_elements_after_last_user_do_not_crash():
    messages = [{"role": "user", "content": "ok"}, None, "trailing garbage"]
    assert find_last_user_idx(messages) == 0
