"""A turn's completion is published with the reply objects the turn produced; a str subclass whose
``__new__`` needs more than the text could not be deep-copied, so ``SessionEvents.publish`` raised and
the turn's ``message.complete`` never went out (#156709). Each such reply type must publish and copy
back as itself, fields included."""
import copy
import pickle
import sqlite3

import pytest

from agent.prepared_compaction import _Miss
from gateway.runtime_footer import FooteredReply
from gateway.session_events import SessionEvents
from hermes_cli.active_sessions import ActiveSessionRefusal
from hermes_cli.cli_backtrack import BacktrackResend
from hermes_cli.plugins_cmd_install import _ConsentRefusal
from persistence.sqlite._state.repair import _ProbeFailure
from tools.process_registry_notifications import TimelineNotification

_db_error = sqlite3.OperationalError("disk I/O error")
_db_error.sqlite_errorcode = 778
REPLIES = [
    FooteredReply("Here it is", "gpt-5.4 · 68% · ~/projects/hermes"),
    TimelineNotification("job done", "Job finished", "background_result", "diagnostic"),
    _ConsentRefusal("dependency install declined", "deps_declined"),
    _Miss("head_moved", "head moved (3 -> 5)"),
    _ProbeFailure("fts5 read probe failed", _db_error),
    BacktrackResend("edited prompt", ordinal=2, expected="old prompt", session_id="s1", number=4),
    ActiveSessionRefusal("session is busy", "busy"),
]


def _fields(value):
    names = getattr(type(value), "__slots__", ()) or vars(value)
    return {name: getattr(value, name) for name in names}


@pytest.mark.parametrize("reply", REPLIES, ids=lambda r: type(r).__name__)
def test_reply_publishes_and_round_trips_as_itself(reply):
    events = SessionEvents()
    events.publish("s1", {"text": reply, "messages": [{"content": reply}]})
    for clone in (copy.deepcopy(reply), pickle.loads(pickle.dumps(reply))):
        assert type(clone) is type(reply)
        assert str(clone) == str(reply)
        assert _fields(clone) == _fields(reply)
