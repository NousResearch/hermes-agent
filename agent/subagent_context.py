"""Detached fork capture and persistence; history format belongs to the plugin."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class _ForkSource:
    owner: int
    messages: list | None
    snapshot: list | None = None
    lock: Lock = field(default_factory=Lock)


_SOURCE: ContextVar[_ForkSource | None] = ContextVar("subagent_fork_source", default=None)


@contextmanager
def bind_fork_source(parent, messages):
    """One lazy snapshot for all launches in a hook/tool round, including worker threads."""
    source = _ForkSource(id(parent), messages)
    token = _SOURCE.set(source)
    try:
        yield
    finally:
        # Copied Contexts may outlive a tool round; never let them pin its transcript.
        with source.lock:
            source.messages = source.snapshot = None
        _SOURCE.reset(token)


def _snapshot(messages):
    from agent.session_persistence import _is_ephemeral_scaffolding
    from agent.context_compressor import is_compaction_summary_message
    # Recovery scaffolding is never durable history, even for an identity formatter.
    snapshot = deepcopy([m for m in messages if not _is_ephemeral_scaffolding(m)])
    for message in snapshot:
        if is_compaction_summary_message(message):
            message["_compressed_summary"] = True
    return snapshot


def capture_fork_boundary(parent):
    """Copy unformatted history once per round, detached from parent and siblings."""
    source = _SOURCE.get()
    if source is not None and source.owner == id(parent):
        with source.lock:
            if source.messages is not None:
                if source.snapshot is None:
                    source.snapshot = _snapshot(source.messages)
                return deepcopy(source.snapshot)
    messages = getattr(parent, "_session_messages", None)
    return _snapshot(messages) if isinstance(messages, list) else []


def format_fork_history(messages, *, task=None, extension=None):
    """Apply plugin policy to a private snapshot, then enforce transcript integrity."""
    from agent.subagent_extension import get_subagent_extension
    from agent.subagent_fork_validation import validate_fork_history
    extension = extension if extension is not None else get_subagent_extension()
    if extension is None:
        raise ValueError("fork_context requires an enabled subagent extension with a fork formatter.")
    history = extension.format_fork_history(deepcopy(messages), task=deepcopy(task or {}))
    history = validate_fork_history(history)
    if history and history[-1]["role"] == "user":
        # Provider role repair must not merge an inherited request into the new assignment.
        # Persist the same boundary sent on the wire so resume sees identical history.
        history.append({"role": "assistant", "content":
                        "[End of inherited conversation. The next message is your delegated assignment.]"})
    return history


def fork_history(parent, *, task=None, extension=None):
    return format_fork_history(capture_fork_boundary(parent), task=task, extension=extension)


def attach_fork_history(child, history):
    if history:
        child._delegate_fork_history = deepcopy(history)
        child._delegate_fork_needs_persist = True


def persist_fork_history(agent, conversation_history):
    """Seed the child's own session before compaction; ordinary history is already durable."""
    if vars(agent).get("_delegate_fork_needs_persist") is not True:
        return
    if agent._flush_messages_to_session_db(conversation_history, None) is False:
        raise RuntimeError("Could not persist inherited subagent context; refusing to start the child.")
    agent._delegate_fork_needs_persist = False
