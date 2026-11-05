"""A tool-calling assistant row's display identity is its call ids, so a continuation's copy folds into the ancestor's whatever the key order of its calls."""

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession


@pytest.mark.parametrize("reorder", [False, True], ids=["identical", "reordered-keys"])
def test_ancestor_display_identity_folds_tool_call_copies_by_call_id(backend, scope_a, reorder):
    scope = scope_a
    original = [{"id": "call", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    copied = [{"function": {"arguments": "{}", "name": "f"}, "type": "function", "id": "call"}] if reorder else original
    for sid, parent, calls in [("parent", None, original), ("child", "parent", copied)]:
        backend.open(scope).sessions.create(NewSession(SessionId(sid), "cli", parent_session_id=parent))
        backend.open(scope).messages.append(sid, NewMessage("assistant", "answer", timestamp=40.125, tool_calls=calls))
        if parent is None:
            backend.open(scope).sessions.end(SessionId(sid), "compression")
    assert backend.open(scope).lineage.ancestor_display_prefix(SessionId("child")) == ()
