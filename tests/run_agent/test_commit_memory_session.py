"""Regression tests for AIAgent.commit_memory_session() (#22394).

The function is called when the session_id rotates without tearing
providers down (CLI ``/new``, gateway session expiry, in-process
compression). Both the memory manager AND the context engine must
receive ``on_session_end`` so plugin engines like LCM persist their
final turns and finalize the rotating session.
"""

from run_agent import AIAgent


class _Compressor:
    def __init__(self):
        self.session_end_calls = []

    def on_session_end(self, session_id, messages):
        self.session_end_calls.append((session_id, list(messages or [])))


class _Manager:
    def __init__(self):
        self.session_end_calls = []

    def on_session_end(self, messages):
        self.session_end_calls.append(list(messages or []))


def _agent(memory_manager=None, context_compressor=None, session_id="sess-1"):
    a = AIAgent.__new__(AIAgent)
    a._memory_manager = memory_manager
    a.context_compressor = context_compressor
    a.session_id = session_id
    return a


def test_commit_memory_session_notifies_context_compressor():
    """Primary regression: with both manager and compressor configured,
    ``commit_memory_session`` must fan out to both. Before the fix the
    compressor never saw session-end on /new, gateway expiry, or
    compression-driven rotation."""
    mgr = _Manager()
    comp = _Compressor()
    agent = _agent(mgr, comp)
    msgs = [{"role": "user", "content": "hi"}]

    agent.commit_memory_session(msgs)

    assert mgr.session_end_calls == [msgs]
    assert comp.session_end_calls == [("sess-1", msgs)]


def test_commit_memory_session_compressor_called_with_empty_when_no_messages():
    mgr = _Manager()
    comp = _Compressor()
    agent = _agent(mgr, comp)

    agent.commit_memory_session(None)

    assert mgr.session_end_calls == [[]]
    assert comp.session_end_calls == [("sess-1", [])]


def test_commit_memory_session_uses_empty_session_id_when_unset():
    """``self.session_id`` may be transiently None during early init or
    test stubs — the compressor call must not raise on that."""
    mgr = _Manager()
    comp = _Compressor()
    agent = _agent(mgr, comp, session_id=None)

    agent.commit_memory_session([])

    assert comp.session_end_calls == [("", [])]


def test_commit_memory_session_swallows_compressor_exception():
    """A misbehaving compressor must not break session rotation; the
    memory manager call must still have happened."""

    class _BadComp:
        def on_session_end(self, session_id, messages):
            raise RuntimeError("boom")

    mgr = _Manager()
    agent = _agent(mgr, _BadComp())

    agent.commit_memory_session([{"role": "user", "content": "x"}])

    assert mgr.session_end_calls == [[{"role": "user", "content": "x"}]]


def test_commit_memory_session_no_compressor_attribute_is_no_op():
    """An agent built without a compressor (older test stubs, plugin
    engine never selected) must not raise."""
    mgr = _Manager()
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = mgr
    agent.session_id = "sess-1"
    # Deliberately no context_compressor attribute.

    agent.commit_memory_session([])

    assert mgr.session_end_calls == [[]]


def test_commit_memory_session_compressor_set_to_none_is_no_op():
    """Compressor explicitly None (no plugin engine selected) is a
    common configuration; no compressor call must fire."""
    mgr = _Manager()
    agent = _agent(mgr, context_compressor=None)

    agent.commit_memory_session([])

    assert mgr.session_end_calls == [[]]
