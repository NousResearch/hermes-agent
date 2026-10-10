"""Bind already-read inbound file slices to the agent that consumes the turn."""

from agent.source_provenance import clear_agent_source_provenance, provenance_kwargs_for_agent


def bind_inbound_source_slices(agent, slices):
    if not slices:
        return
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
