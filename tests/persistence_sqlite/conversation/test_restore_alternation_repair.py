"""get_messages_as_conversation(repair_alternation=True): heal durable
alternation violations at the restore boundary.

A turn that persists a user row but no assistant row (e.g. its reply was
suppressed, or two concurrent turns interleaved their flushes) leaves a
``user;user`` pair in state.db. Without repair at restore, the defensive
pre-request ``repair_message_sequence`` re-fires on EVERY request for the
rest of the session's life, because it mutates only the per-request list.

Default (``repair_alternation=False``) must stay verbatim: inspection and
export consumers (trace upload, context guard) read the transcript as-is.
"""

import pytest

from persistence.sqlite.conversation._store.messages import append_message, get_messages_as_conversation
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture()
def holder(backend, scope_a):
    return backend.open(scope_a).holder()


def _seed_wedged_session(holder, session_id="s1"):
    """assistant → user → user (no assistant row between): the durable wedge."""
    create_session(holder, session_id, "system prompt")
    append_message(holder, session_id=session_id, role="user", content="first ask")
    append_message(holder, session_id=session_id, role="assistant", content="first reply")
    append_message(holder, session_id=session_id, role="user", content="unanswered turn")
    append_message(holder, session_id=session_id, role="user", content="next turn")
    append_message(holder, session_id=session_id, role="assistant", content="next reply")


def test_repair_alternation_merges_user_pair(holder):
    _seed_wedged_session(holder)
    messages = get_messages_as_conversation(holder, "s1", repair_alternation=True)
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    # Both user texts survive, merged in order: no user input is lost.
    merged = messages[2]["content"]
    assert "unanswered turn" in merged and "next turn" in merged
    assert merged.index("unanswered turn") < merged.index("next turn")


def test_repaired_load_is_stable_under_prerequest_repair(holder):
    """The restored list must yield ZERO further repairs: this is the whole
    point: the pre-request defensive repair stops firing every turn."""
    from agent.agent_runtime_helpers import repair_message_sequence

    _seed_wedged_session(holder)
    messages = get_messages_as_conversation(holder, "s1", repair_alternation=True)
    assert repair_message_sequence(None, messages) == 0
