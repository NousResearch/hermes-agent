"""Copy contract for dashboard chat start failures (``/api/pty`` close 1011).

The terminal pane used to show the raw exception (``Chat unavailable: [Errno 2]
No such file or directory: 'node'``, ``Chat unavailable: 1``, or an empty
``Chat unavailable:`` for a full registry). Each failure must now say what
happened and what to do, with no errno or exception class in the lead.
"""

from fastapi import HTTPException

from hermes_cli.pty_session import RegistryFull
from hermes_cli.web_routers.chat_ws_errors import chat_start_failure_message


def test_registry_full_names_the_fix_and_carries_a_message():
    exc = RegistryFull()
    assert str(exc)  # was empty before; the SPA printed "Chat unavailable:"
    msg = chat_start_failure_message(exc)
    assert "too many chat terminals" in msg
    assert "Start new session" in msg


def test_missing_node_points_at_installing_node_not_errno():
    for exc in (FileNotFoundError(2, "No such file or directory", "node"), SystemExit(1)):
        msg = chat_start_failure_message(exc)
        assert "Node" in msg and "nodejs.org" in msg
        assert "Errno" not in msg and "SystemExit" not in msg
        assert not msg.endswith(": 1")


def test_unknown_profile_keeps_detail_and_adds_next_step():
    msg = chat_start_failure_message(HTTPException(status_code=404, detail="Profile 'nope' not found"))
    assert msg.startswith("Chat could not start: Profile 'nope' not found.")
    assert "profile" in msg.lower()
