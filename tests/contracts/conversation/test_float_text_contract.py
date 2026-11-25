"""Contract case for a float written as message content: it reads back as text that parses to the same float; the exact spelling is the backend's own.

Every case runs each registered backend through public operations only.
"""

from __future__ import annotations

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.identity import SessionId

S = SessionId("float-text-s1")
# Values whose text spelling differs between SQLite builds and between backends (15 significant digits, or the shortest round-trip text).
_VALUES = (0.1 + 0.2, 1e16, 5e-324, 100.0, 1 / 3, 2.2250738585072014e-308, 1.7976931348623157e308, 9007199254740993.0)


@pytest.mark.parametrize("value", _VALUES)
def test_float_content_reads_back_as_text_for_the_same_float(backend, scope_a, value):
    if value > 1e308:
        pytest.skip("a 15-digit spelling of the largest float parses back past its range")
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=S, source="cli"))
    store.messages.append(S, NewMessage(role="user", content=value, timestamp=1.0))
    (row,) = store.messages.read_transcript(S)
    assert isinstance(row.content, str)
    assert float(row.content) == pytest.approx(value, rel=1e-14, abs=0)
