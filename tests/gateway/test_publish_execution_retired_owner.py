"""A late execution callback whose owning live session was retired is dropped, not raised.

``publish_execution`` is the one sink every local-viewer callback reaches: ``hermes_agent.gateway.local_viewer_events.wire``
installs the parent turn's closures on the agent, and ``tools/delegate_tool`` copies the parent's ``_print_fn``
onto a background child by reference. Once the parent's turn ends and ``session_retirement.retire_idle`` drops
the live entry, the child's next status print (the first one after the parent goes idle is usually the
``⟳ compacting context…`` line) indexed ``self.sessions[session_id]`` and died with ``KeyError``, which
``StatusOutputMixin._safe_print`` does not swallow — so it travelled out of the child's ``run_conversation``
and lost the child's work (#148644). The generation fence right below the index already documents that a
callback outliving its claim is dropped, so the entry has to be read without indexing.
"""
from types import SimpleNamespace
from typing import cast

from agent.status_output import StatusOutputMixin
from hermes_agent.gateway import local_viewer_events
from tests.gateway.test_session_authority_cancel_settlement import _authority

PARENT = 's'


def test_publish_execution_drops_a_callback_for_a_retired_session(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        authority.executing[PARENT] = 7  # the stamp the child inherited from the parent's turn
        del authority.sessions[PARENT]  # retire_idle: the parent's turn ended, nothing pinned it
        assert authority.publish_execution(PARENT, 7, 'agent.print', {'text': 'x'}) is False


def test_a_background_delegate_child_print_survives_the_parent_turn_ending(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    frames = []
    with db:
        authority.executing[PARENT] = 7
        authority.sessions[PARENT].event_stream.observers.add(frames.append)
        turn = SimpleNamespace(_publish_execution=lambda event_type, payload:
                               authority.publish_execution(PARENT, 7, event_type, payload))
        parent_agent = SimpleNamespace()
        local_viewer_events.wire(turn, parent_agent)  # installs this turn's `_print_fn`
        child = cast(StatusOutputMixin, SimpleNamespace(_print_fn=parent_agent._print_fn,
                                                       _mute_notification_reply=False))

        StatusOutputMixin._safe_print(child, '  ⟳ compacting context…')
        assert [f['params']['payload']['text'] for f in frames if f['params']['type'] == 'agent.print'] == [
            '  ⟳ compacting context…'], 'a live owner still receives its child prints'

        del authority.sessions[PARENT]
        StatusOutputMixin._safe_print(child, '  ⟳ compacting context…')  # must be dropped, never raise
