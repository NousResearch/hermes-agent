"""Bind already-read inbound file slices to the agent that consumes the turn."""

from pathlib import Path

# Private bootstrap bounds must be available before any runtime/provider modules load. The
# consuming registry independently enforces its own file/grant limits when it binds these facts.
_MAX_WORKER_SOURCE_SLICES = 64
_MAX_WORKER_SOURCE_LINES = 2_000
_MAX_WORKER_SOURCE_BYTES = 262_144


def pack_worker_source_slices(slices):
    """Only the owner's expansion result enters this private pipe; never admission payload fields.

    These are read facts, not grants: the child revalidates the bytes and supplies its own identity.
    """
    packed = [{'path': str(s.path), 'line_start': s.line_start, 'line_end': s.line_end,
               'content': s.content.decode('utf-8')} for s in slices]
    validate_worker_source_slices(packed)
    return packed


def validate_worker_source_slices(value):
    """Validate only inert bootstrap facts; safe to import before the worker policy is bound."""
    if not isinstance(value, list) or len(value) > _MAX_WORKER_SOURCE_SLICES:
        raise ValueError('invalid_managed_source_slices')
    total = 0
    for item in value:
        if (not isinstance(item, dict) or set(item) != {'path', 'line_start', 'line_end', 'content'}
                or not isinstance(item['path'], str) or not Path(item['path']).is_absolute()
                or not isinstance(item['content'], str)
                or type(item['line_start']) is not int or type(item['line_end']) is not int
                or not 1 <= item['line_start'] <= item['line_end']
                or item['line_end'] - item['line_start'] + 1 > _MAX_WORKER_SOURCE_LINES):
            raise ValueError('invalid_managed_source_slices')
        content = item['content'].encode('utf-8')
        total += len(content)
        if not content or len(content) > _MAX_WORKER_SOURCE_BYTES or total > _MAX_WORKER_SOURCE_BYTES:
            raise ValueError('invalid_managed_source_slices')


def restore_worker_source_slices(value):
    """Construct runtime slice types only after the child's worker policy has been bound."""
    validate_worker_source_slices(value)
    from agent.context_references import ContextSourceSlice
    return [ContextSourceSlice(Path(item['path']), item['line_start'], item['line_end'],
                               item['content'].encode('utf-8')) for item in value]


def bind_worker_source_slices(agent, frame):
    """Bind after adopted-worker construction, never from task text or a serialized grant."""
    slices = restore_worker_source_slices(frame.get('source_slices_v1', []))
    if slices and getattr(agent, 'session_id', None) != frame['scope']['session_id']:
        raise ValueError('invalid_managed_source_identity')
    bind_inbound_source_slices(agent, slices)


def bind_inbound_source_slices(agent, slices):
    if not slices:
        return
    from agent.source_provenance import clear_agent_source_provenance, provenance_kwargs_for_agent
    identity = provenance_kwargs_for_agent(agent, establish_turn=True)
    if not identity:
        raise ValueError("Inbound source context has no consuming session identity")
    registry = identity.pop("source_provenance_registry")
    try:
        for source in slices:
            registry.issue_file_slice(
                path=source.path, line_start=source.line_start, line_end=source.line_end,
                content=source.content, **identity,
            )
    except BaseException:
        clear_agent_source_provenance(agent)
        raise
