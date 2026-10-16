"""Bounded lifecycle projection of delegated work; no child transcript or reasoning."""


def publish_subagent_progress(turn, event, name, identity):
    if event not in {'subagent.start', 'subagent.tool', 'subagent.progress', 'subagent.complete'}:
        return
    payload = {key: identity[key] for key in ('subagent_id', 'parent_id', 'goal', 'task_index',
               'task_count', 'model', 'tool_count', 'status') if key in identity}
    if name:
        payload['tool_name'] = name
    turn._publish_execution(event, payload)
