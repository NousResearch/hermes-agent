"""SessionEntry.reasoning_tokens serialization round-trip.

Standalone (no gateway runtime / async) so it runs even where the wider
gateway suite can't be collected.
"""
from datetime import datetime, timezone

from gateway.session import SessionEntry


def _entry(**kw):
    now = datetime.now(timezone.utc)
    return SessionEntry(
        session_key="k", session_id="s", created_at=now, updated_at=now, **kw
    )


def test_reasoning_tokens_default_zero():
    assert _entry().reasoning_tokens == 0


def test_reasoning_tokens_round_trips_through_dict():
    e = _entry(reasoning_tokens=4242)
    d = e.to_dict()
    assert d["reasoning_tokens"] == 4242
    assert SessionEntry.from_dict(d).reasoning_tokens == 4242


def test_from_dict_without_reasoning_tokens_defaults_zero():
    d = _entry(reasoning_tokens=7).to_dict()
    d.pop("reasoning_tokens")
    # Legacy persisted state (pre-field) must load without error.
    assert SessionEntry.from_dict(d).reasoning_tokens == 0
