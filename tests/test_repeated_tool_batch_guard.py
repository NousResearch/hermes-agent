import types

from run_agent import AIAgent


def _tc(name, args):
    return types.SimpleNamespace(function=types.SimpleNamespace(name=name, arguments=args))


def _agent():
    return AIAgent.__new__(AIAgent)


def test_third_identical_batch_appends_loop_guard_note():
    agent = _agent()
    batch = [_tc("process_manage", '{"action":"kill","id":"p1"}')]
    for i in range(3):
        msgs = [{"role": "tool", "content": "already_exited"}]
        agent._note_repeated_tool_batch(batch, msgs)
        assert ("Loop guard" in msgs[0]["content"]) == (i == 2)


def test_changed_batch_resets_counter():
    agent = _agent()
    a = [_tc("t", "{}")]
    b = [_tc("t", '{"x":1}')]
    for batch in (a, a, b, a, a):
        msgs = [{"role": "tool", "content": "r"}]
        agent._note_repeated_tool_batch(batch, msgs)
        assert "Loop guard" not in msgs[0]["content"]
