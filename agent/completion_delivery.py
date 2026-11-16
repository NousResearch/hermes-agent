"""Fresh tool-boundary exposure of child evidence; no system-prompt or old-context mutation."""
from contextlib import contextmanager


@contextmanager
def tool_result_boundary(agent):
    previous = getattr(agent, '_completion_tool_boundary', False)
    agent._completion_tool_boundary = True
    try:
        yield
    finally:
        agent._completion_tool_boundary = previous


def remember_exposure(agent, message, block):
    """Keep the bounded evidence through this tool batch's aggregate budget pass."""
    pending = getattr(agent, '_completion_result_blocks', None)
    if pending is None:
        pending = agent._completion_result_blocks = {}
    pending[id(message)] = (message, block)


def keep_after_turn_budget(agent, batch):
    pending = getattr(agent, '_completion_result_blocks', None)
    if not pending:
        return
    for message in batch:
        queued = pending.pop(id(message), None)
        if queued is None or queued[0] is not message:
            continue
        block = queued[1]
        content = message.get('content')
        if isinstance(content, list):
            if not any(isinstance(part, dict) and block in str(part.get('text', '')) for part in content):
                message['content'] = [*content, {'type': 'text', 'text': block}]
        elif block not in str(content or ''):
            message['content'] = str(content or '') + block
