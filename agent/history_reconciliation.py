"""Reconcile an addressed model projection while its durable turn lease is held.

Only the caller knows that this is a session-backed model context. Persistence
markers are deduplication hints, not provenance. Never use them to claim an
arbitrary explicit API history, or use display history (including inactive rows).
"""
from agent.session_persistence import _durable_content

_MODEL_FIELDS = (
    "role", "tool_call_id", "tool_name", "tool_calls", "reasoning", "reasoning_content",
    "reasoning_details", "codex_reasoning_items", "codex_message_items", "effect_disposition",
)


def _row_id(message):
    value = message.get("_row_id")
    return value if type(value) is int and value > 0 else None


def _model_projection(message):
    # Compare what persistence can retain, not a lossy DB string to a live image.
    content = message.get("api_content")
    if not isinstance(content, str):
        content = message.get("content")
    return (
        _durable_content(content), tuple(message.get(key) for key in _MODEL_FIELDS),
        bool(message.get("_compressed_summary")),
    )


def reconcile_model_history(durable_history, model_history):
    """Retain enrichments only on the same active row with an equal durable projection.

    This does not read or mutate storage. The surface invokes it under its native
    lease using active-only rows and excludes its already-staged input first.
    A free lease alone does not prove that an earlier snapshot is current.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT

    addressed, pending = {}, []
    for message in model_history:
        rid = _row_id(message)
        if rid is not None:
            if rid in addressed or pending:
                raise ValueError("ambiguous or interleaved model history")
            addressed[rid] = message
        elif message.get(_PERSIST_AFTER_ADMISSION_INTERRUPT) or (
            message.get("role") == "user" and not message.get(_DB_PERSISTED_MARKER)
        ):
            pending.append(message)
        else:
            raise ValueError("unaddressed model context cannot be safely reconciled")

    reconciled = []
    seen = set()
    for durable in durable_history:
        rid = _row_id(durable)
        if rid is None or rid in seen:
            raise ValueError("durable context must have unique row addresses")
        seen.add(rid)
        existing = addressed.get(rid)
        reconciled.append(existing if existing is not None and
                          _model_projection(existing) == _model_projection(durable) else durable)
    reconciled.extend(pending)
    if len(reconciled) == len(model_history) and all(a is b for a, b in zip(reconciled, model_history)):
        return model_history
    return reconciled
