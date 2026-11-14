"""A canonical conversation owns its children across parent-agent cache rebuilds."""
import weakref
from types import SimpleNamespace

from gateway.session_ancillary import _owner_objects
from tools import delegate_tool_registry as registry


class Agent:
    pass


def test_roster_and_controls_keep_children_after_parent_rebuild(monkeypatch):
    old_parent, replacement = Agent(), Agent()
    old_parent.session_id = replacement.session_id = 'same-conversation'
    live = SimpleNamespace()
    child = Agent()
    child._delegate_parent_ref = weakref.ref(old_parent)
    other = Agent()
    other._delegate_parent_ref = weakref.ref(old_parent)
    records = {
        'owned': {'subagent_id': 'owned', 'agent': child, 'owner_session_record': live},
        'foreign': {'subagent_id': 'foreign', 'agent': other, 'owner_session_record': SimpleNamespace()},
        'unstamped': {'subagent_id': 'unstamped', 'agent': other},
    }
    monkeypatch.setattr(registry, '_active_subagents', records)
    from gateway import session_managed_worker
    monkeypatch.setattr(session_managed_worker, 'managed_policy', lambda authority, ref: None)
    authority = SimpleNamespace(sessions={'logical': live}, agent=lambda ref: replacement)
    ref = SimpleNamespace(session_id='logical')
    agent, visible = _owner_objects(authority, ref)
    assert agent is replacement
    assert [r['subagent_id'] for r in visible] == ['owned']
    authority.agent = lambda ref: None
    assert [r['subagent_id'] for r in _owner_objects(authority, ref)[1]] == ['owned']
    # Reusing the same durable ID with another LiveSession must not transfer access.
    authority.sessions['logical'] = SimpleNamespace()
    assert _owner_objects(authority, ref)[1] == []


def test_child_registration_inherits_the_canonical_owner(monkeypatch):
    from tools import delegate_tool_child_run as child_run
    parent, child = Agent(), Agent()
    live = SimpleNamespace()
    # The child captured the token before its parent turn restored its callbacks.
    child._canonical_session_owner = live
    parent._canonical_session_owner = None
    parent.session_id = 'same-conversation'
    child._subagent_id = 'child-fixture'
    saved = []
    monkeypatch.setattr(child_run, '_register_subagent', saved.append)
    child_run._register_child(child, parent, 'fixture', owner_session_id=None,
                              owner_transport=None, owner_session_record=None)
    assert saved[0]['owner_session_record'] is live
    assert child._canonical_session_owner is live
