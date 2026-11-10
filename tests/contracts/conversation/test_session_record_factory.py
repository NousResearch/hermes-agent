"""The shared session-record factory of ``tests/contracts/_records.py`` builds the record a backend returns for a fresh session, field by field, on every backend."""

from __future__ import annotations

from dataclasses import fields

import pytest

from persistence import SessionId
from persistence.conversation.sessions import NewSession, SessionRecord, TitleSource
from tests.contracts._records import SESSION_RECORD_DEFAULTS, session_record


def test_defaults_name_every_record_field():
    assert set(SESSION_RECORD_DEFAULTS) == {f.name for f in fields(SessionRecord)}


def test_factory_equals_a_created_session_field_by_field(backend, scope_a):
    store = backend.open(scope_a)
    sid = SessionId("sess")
    store.sessions.create(NewSession(session_id=sid, source="cli"))
    stored = store.sessions.get(sid)
    built = session_record(started_at=stored.started_at, profile_name=stored.profile_name)
    for f in fields(SessionRecord):
        assert getattr(built, f.name) == getattr(stored, f.name), f.name
    assert built == stored


def test_overrides_replace_only_the_named_fields():
    record = session_record(session_id=SessionId("other"), title="t", title_source=TitleSource.USER, archived=True)
    assert (record.session_id, record.title, record.title_source, record.archived) == ("other", "t", TitleSource.USER, True)
    assert record.source == "cli" and record.model_config == {}
    with pytest.raises(TypeError):
        session_record(no_such_field=1)
