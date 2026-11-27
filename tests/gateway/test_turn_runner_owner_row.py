"""The turn runner reads the owner's execution generation from its ``sessions`` row; a live authority session with
no row fails the turn with the ``TypeError`` reading a missing row raises."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner


def _runner(row):
    store = SimpleNamespace(sessions=SimpleNamespace(get=lambda sid: row))
    authority = SimpleNamespace(sessions={"owner": object()}, persistence=store, runner=None, profile_id="p",
                                logical_owner=lambda sid: sid)
    ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL, chat_id="owner"), session_id="owner")
    return authority, lambda: TurnRunner(SimpleNamespace(session_authority=authority), ctx)


def test_the_owner_row_gives_the_generation():
    authority, build = _runner(SimpleNamespace(runtime_generation=4))
    assert build()._approval_owner == (authority, "owner", 4)


def test_a_missing_owner_row_raises_type_error():
    _authority, build = _runner(None)
    with pytest.raises(TypeError, match=r"^'NoneType' object is not subscriptable$"):
        build()
