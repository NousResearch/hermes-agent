"""Session exports keep in-place-compacted turns; verified delete stays honest (#119933)."""

import hermes_state_portability as hsp
import hermes_cli.sessions_cmd as sc


class _FakeDB:
    def __init__(self, messages):
        self._messages = messages
        self.calls = []

    def get_messages(self, session_id, **kwargs):
        self.calls.append((session_id, kwargs))
        return list(self._messages)

    def get_session(self, session_id):
        return {"id": session_id}

    def export_session(self, session_id, **kwargs):
        return {"id": session_id, "messages": list(self._messages)}


def test_with_messages_can_include_compacted_rows():
    db = _FakeDB([{"role": "user", "content": "old"}])
    out = hsp.SessionPortabilityMixin._with_messages(db, {"id": "s1"}, include_compacted=True)
    assert db.calls[-1][1] == {"include_compacted": True}
    assert out["messages"] == [{"role": "user", "content": "old"}]


def test_default_export_still_excludes_compacted_rows():
    db = _FakeDB([])
    hsp.SessionPortabilityMixin._with_messages(db, {"id": "s1"})
    assert db.calls[-1][1] == {"include_compacted": False}
