"""Exports include appended rows and the stored message counters."""

import pytest

from persistence import ExportTooLarge
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession

AT = 1672531200.0


@pytest.mark.parametrize("imported", [False, True])
def test_export_reads_appended_messages(backend, scope_a, imported):
    store_a = backend.open(scope_a)
    sid = "export-appended"
    expected = []
    if imported:
        result = store_a.maintenance.import_sessions([{"id": sid, "source": "cli", "started_at": AT, "messages": [{"role": "user", "content": "imported", "timestamp": AT}]}])
        assert result.imported == 1
        expected.append("imported")
    else:
        store_a.sessions.create(NewSession(sid, "cli"))
    tool_id = store_a.messages.append(sid, NewMessage("assistant", "tool call", timestamp=AT, tool_calls=({"id": "call", "function": {"name": "echo", "arguments": "{}"}},)))
    store_a.messages.append(sid, NewMessage("user", "appended", timestamp=AT))
    expected.extend(["tool call", "appended"])
    exported = store_a.maintenance.export_session(sid)
    assert (exported.session.message_count, exported.session.tool_call_count) == (len(expected), 1)
    assert [message.content for message in exported.messages] == expected
    with pytest.raises(ExportTooLarge):
        store_a.maintenance.assert_export_safe(sid, 1)
    # Deactivating a row leaves the stored counters as they were; the default export reads live rows only.
    assert store_a.messages.deactivate_message(sid, tool_id)
    exported = store_a.maintenance.export_session(sid)
    assert (exported.session.message_count, exported.session.tool_call_count) == (len(expected), 1)
    assert [message.content for message in exported.messages] == [content for content in expected if content != "tool call"]
