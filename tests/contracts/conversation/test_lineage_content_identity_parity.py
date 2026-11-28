"""A compression continuation's copy of an ancestor message folds into it only when the stored content matches."""

from copy import deepcopy

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession


@pytest.mark.parametrize(
    "role,reordered_retained",
    [
        ("user", False),
        ("assistant", True),
        ("system", True),
        ("developer", True),
        ("tool", True),
        ("function", True),
        ("session_meta", True),
        ("custom", True),
    ],
)
@pytest.mark.parametrize("kind", ["identical", "reordered", "list-order"])
def test_prefix_keeps_role_specific_content_identity(backend, scope_a, role, reordered_retained, kind):
    store_a = backend.open(scope_a)
    original = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/picture", "detail": "low"}},
    ]
    reordered = [
        {"text": "look", "type": "text"},
        {"image_url": {"detail": "low", "url": "https://example.invalid/picture"}, "type": "image_url"},
    ]
    copies = {"identical": deepcopy(original), "reordered": reordered, "list-order": list(reversed(original))}
    for sid, parent, content in [("parent", None, original), ("child", "parent", copies[kind])]:
        store_a.sessions.create(NewSession(SessionId(sid), "cli", parent_session_id=parent))
        store_a.messages.append(sid, NewMessage(role, content, timestamp=40.125))
        if parent is None:
            store_a.sessions.end(SessionId(sid), "compression")

    prefix = store_a.lineage.ancestor_display_prefix(SessionId("child"))

    retained = kind == "list-order" or (kind == "reordered" and reordered_retained)
    assert [(row.role, row.content) for row in prefix] == ([(role, original)] if retained else [])
