"""Execution-scoped original-message context for native plugins.

Native plugins receive tool arguments after the gateway has normalized and
coalesced inbound messages. A reply anchor or task/session id cannot prove
which original inbound messages produced a tool call, especially when one
turn merged several originals. This module exposes that provenance without
routing identity, authorization tokens, or raw bodies through model-authored
tool arguments.

Ownership:

* The gateway owns authorization and execution leases (:func:`bind_execution`
  / :func:`clear_execution`, called from the turn handoff).
* :meth:`Registry.dispatch` establishes the narrow tool-call scope
  (:func:`scoped_tool_call`); :func:`get_tool_source_context` only resolves
  inside it.
* A copied worker context becomes unusable after cancellation, run
  replacement, reset, or completion because every read re-validates the
  execution id against the live lease table.

Failure behavior (fail-closed): missing/contradictory source metadata,
altered presentation, a foreign or stale worker context, internal events, and
model-forged fields never authorize a mutation. Quote text without a
documented transport reference is explicitly unavailable. Local FIFO order is
not promoted to a transport-order guarantee; ordinal positions are presentation
order only.
"""

import hashlib
import threading
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "SourceFragment",
    "ToolSourceContext",
    "bind_execution",
    "bind_execution_for_event",
    "clear_execution",
    "scoped_tool_call",
    "get_tool_source_context",
    "render_source_fragments",
    "verify_source_quote",
    "source_context_allows_mutation",
    "note_single_source",
    "merge_append_fragments",
    "rebase_event_fragments",
    "shift_event_fragments",
    "invalidate_source_fragments",
]


@dataclass(frozen=True)
class SourceFragment:
    """Immutable provenance of one original inbound message inside merged text."""

    namespace: str
    message_id: Optional[str] = None
    seq: Optional[str] = None
    reference: Optional[str] = None
    start: int = 0
    end: int = 0
    complete: bool = True


@dataclass(frozen=True)
class ToolSourceContext:
    """Immutable per-execution source record handed to native plugins.

    ``abort_cancelled`` is a per-execution :class:`threading.Event` set
    when the execution lease is cleared — mirrors the ``abortSignal`` other
    harnesses pass to tool hooks for cooperative cancellation.
    """

    execution_id: str
    run_generation: int
    session_key: str
    scope: str
    fragments: Tuple[SourceFragment, ...] = ()
    text_hash: str = ""
    complete: bool = False
    internal: bool = False
    abort_cancelled: Any = None  # threading.Event set on clear, None for forgeries

    @property
    def fragment_count(self) -> int:
        """How many originals were coalesced into this presentation."""
        return len(self.fragments)


# --- Execution leases (gateway-owned) ---------------------------------------

_LEASES_LOCK = threading.Lock()
_LEASES: Dict[str, Any] = {}  # execution_id -> (run_generation, abort_cancelled)


def _register_lease(execution_id: str, run_generation: int, abort_cancelled: Any) -> None:
    with _LEASES_LOCK:
        _LEASES[execution_id] = (int(run_generation), abort_cancelled)


def _release_lease(execution_id: str) -> None:
    with _LEASES_LOCK:
        record = _LEASES.pop(execution_id, None)
    # Outside the lock — setting the event cannot deadlock.
    if record is not None:
        _, abort_cancelled = record
        if abort_cancelled is not None:
            try:
                abort_cancelled.set()
            except Exception:
                pass


def _lease_generation(execution_id: str) -> Optional[int]:
    with _LEASES_LOCK:
        record = _LEASES.get(execution_id)
    if record is None:
        return None
    generation, _ = record
    return generation


# --- Context state ------------------------------------------------------------

# Session-level binding installed by the gateway turn handoff; never readable
# directly by plugins (only the call scope below is).
_SESSION_BIND: ContextVar[Optional[ToolSourceContext]] = ContextVar(
    "hermes_tool_source_session", default=None
)
# Narrow tool-call scope installed by Registry.dispatch; the only var the
# public getter reads.
_CALL_SCOPE: ContextVar[Optional[ToolSourceContext]] = ContextVar(
    "hermes_tool_source_call", default=None
)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_fragments(text: str, fragments: Tuple[SourceFragment, ...]) -> bool:
    """True when every fragment span is ordered, non-overlapping, and in bounds."""
    text_len = len(text or "")
    cursor = 0
    for frag in fragments or ():
        try:
            start, end = int(frag.start), int(frag.end)
        except (TypeError, ValueError):
            return False
        if start < cursor or end < start or end > text_len:
            return False
        cursor = end
    return True


def bind_execution(
    *,
    text: str,
    fragments: Tuple[SourceFragment, ...],
    session_key: str,
    run_generation: int,
    scope: Optional[str] = None,
    internal: bool = False,
    execution_id: Optional[str] = None,
) -> Tuple[Token, str]:
    """Register a turn execution lease and bind its source record.

    Returns the reset token plus the execution id. Contradictory metadata
    binds a best-effort record flagged ``complete=False`` instead of raising,
    so reads stay available while mutations stay refused.
    """
    eid = execution_id or uuid.uuid4().hex
    frags = tuple(fragments or ())
    # Provenance may carry an overall-complete flag via per-fragment
    # ``complete`` (see rebase_event_fragments).  A merged turn with a
    # text-bearing side lacking provenance must never re-authorize from the
    # surviving span, so ``all(f.complete ...)`` participates, not just
    # “nonempty + spans in bounds”.
    complete = bool(frags) and validate_fragments(text or "", frags) and all(
        getattr(f, "complete", True) for f in frags
    )
    abort_cancelled: Any = threading.Event()
    record = ToolSourceContext(
        execution_id=eid,
        run_generation=int(run_generation),
        session_key=str(session_key or ""),
        scope=str(scope or session_key or ""),
        fragments=frags,
        text_hash=_hash_text(text or ""),
        complete=complete and not internal,
        internal=bool(internal),
        abort_cancelled=abort_cancelled,
    )
    _register_lease(eid, int(run_generation), abort_cancelled)
    return _SESSION_BIND.set(record), eid


def bind_execution_for_event(
    *,
    event: Any,
    session_key: str,
    run_generation: int,
    scope: Optional[str] = None,
) -> Tuple[Any, str]:
    """Bind the execution record for a turn's inbound event (gateway handoff).

    Fragments come from the event's coalescer-preserved provenance; internal
    events bind a read-only record that never authorizes mutations.
    """
    try:
        text = event.text or ""
    except AttributeError:
        text, fragments, internal = "", (), True
    else:
        fragments = tuple(getattr(event, "source_fragments", None) or ())
        internal = bool(getattr(event, "internal", False))
    return bind_execution(
        text=text,
        fragments=fragments,
        session_key=session_key,
        run_generation=run_generation,
        scope=scope,
        internal=internal,
    )


def clear_execution(token: Any) -> None:
    """Unregister the lease and clear the session binding (turn end/reset)."""
    try:
        record = _SESSION_BIND.get()
    except LookupError:
        record = None
    try:
        _SESSION_BIND.reset(token)
    except (ValueError, RuntimeError):
        try:
            _SESSION_BIND.set(None)
        except LookupError:
            pass
    if record is not None:
        _release_lease(record.execution_id)
    try:
        _CALL_SCOPE.set(None)
    except LookupError:
        pass


class scoped_tool_call:
    """Narrow tool-call scope: snapshot the session binding for one dispatch.

    Usable as ``with scoped_tool_call(): ...``. Outside it,
    :func:`get_tool_source_context` returns None.
    """

    def __init__(self) -> None:
        self._token: Any = None

    def __enter__(self) -> Optional[ToolSourceContext]:
        try:
            record = _SESSION_BIND.get()
        except LookupError:
            record = None
        self._token = _CALL_SCOPE.set(record)
        return record

    def __exit__(self, *exc: Any) -> None:
        try:
            _CALL_SCOPE.reset(self._token)
        except (ValueError, RuntimeError):
            try:
                _CALL_SCOPE.set(None)
            except LookupError:
                pass


def get_tool_source_context() -> Optional[ToolSourceContext]:
    """Return the execution-scoped source record, or None.

    Resolves only inside an authorized native registry dispatch with a live
    execution lease. A copied worker context from a cancelled, replaced,
    reset, or completed run fails the lease check and yields None.
    """
    try:
        record = _CALL_SCOPE.get()
    except LookupError:
        return None
    if record is None:
        return None
    if _lease_generation(record.execution_id) != record.run_generation:
        return None
    return record


def render_source_fragments(
    text: str, fragments: Tuple[SourceFragment, ...]
) -> Tuple[str, List[Dict[str, int]]]:
    """Render ordinal-only boundaries and return rebased presentation spans.

    Ordinals are presentation order (1-based); no namespace, message id, or
    transport reference ever enters the rendered text. Each span is
    ``{"ordinal": n, "start": i, "end": j}`` over the rendered string.
    """
    frags = list(fragments or ())
    if not frags:
        return text or "", []
    parts: List[str] = []
    spans: List[Dict[str, int]] = []
    cursor = 0
    for ordinal, frag in enumerate(frags, start=1):
        try:
            start, end = int(frag.start), int(frag.end)
        except (TypeError, ValueError):
            continue
        start = max(0, min(start, len(text or "")))
        end = max(start, min(end, len(text or "")))
        opening = f"[{ordinal}]"
        parts.append((text or "")[cursor:start])
        base = sum(len(p) for p in parts)
        parts.append(opening)
        parts.append((text or "")[start:end])
        spans.append({"ordinal": ordinal, "start": base + len(opening),
                      "end": base + len(opening) + (end - start)})
        cursor = end
    parts.append((text or "")[cursor:])
    return "".join(parts), spans


def _transport_reference(frag: SourceFragment) -> Optional[str]:
    return frag.reference or frag.message_id


def verify_source_quote(
    ctx: Optional[ToolSourceContext],
    text: str,
    ordinal: int,
    quote: str,
) -> bool:
    """True when *quote* matches the ordinal fragment of the bound presentation.

    Fails closed on: no record, stale lease, internal events, incomplete
    records, altered presentation (hash mismatch), unknown ordinals, text
    mismatches, and fragments without a documented transport reference.
    """
    live = get_tool_source_context()
    if ctx is None or live is None or ctx != live:
        return False
    if ctx.internal or not ctx.complete:
        return False
    if _hash_text(text or "") != ctx.text_hash:
        return False
    try:
        frag = ctx.fragments[int(ordinal) - 1]
    except (IndexError, TypeError, ValueError):
        return False
    if _transport_reference(frag) is None:
        return False
    try:
        start, end = int(frag.start), int(frag.end)
    except (TypeError, ValueError):
        return False
    if start < 0 or end < start or end > len(text or ""):
        return False
    return (text or "")[start:end] == (quote or "")


def source_context_allows_mutation(
    ctx: Optional[ToolSourceContext], *, text: Optional[str] = None
) -> bool:
    """Whether *ctx* authorizes a source-grounded mutation.

    An explicit authorized read scope without original ids never grants write
    authority: records with no fragments, internal events, incomplete merges,
    stale leases, or (when *text* is given) altered presentation all refuse.
    """
    live = get_tool_source_context()
    if ctx is None or live is None or ctx != live:
        return False
    if ctx.internal or not ctx.complete or not ctx.fragments:
        return False
    if not all(_transport_reference(frag) for frag in ctx.fragments):
        return False
    if text is not None and _hash_text(text) != ctx.text_hash:
        return False
    return True


# --- Adapter/coalescer helpers -------------------------------------------------

def note_single_source(
    event: Any,
    *,
    namespace: str,
    message_id: Optional[str] = None,
    seq: Optional[str] = None,
    reference: Optional[str] = None,
) -> None:
    """Record a freshly built event as a single original covering its text."""
    try:
        text = event.text or ""
    except AttributeError:
        return
    if not text:
        event.source_fragments = ()
        return
    event.source_fragments = (
        SourceFragment(
            namespace=str(namespace),
            message_id=str(message_id) if message_id else None,
            seq=str(seq) if seq else None,
            reference=str(reference) if reference else None,
            start=0,
            end=len(text),
            complete=True,
        ),
    )


def _shifted(
    fragments: Tuple[SourceFragment, ...], delta: int
) -> Tuple[SourceFragment, ...]:
    return tuple(
        SourceFragment(
            namespace=f.namespace, message_id=f.message_id, seq=f.seq,
            reference=f.reference, start=f.start + delta, end=f.end + delta,
            complete=f.complete,
        )
        for f in fragments
    )


def merge_append_fragments(
    existing_text: str,
    existing_fragments: Tuple[SourceFragment, ...],
    incoming_text: str,
    incoming_fragments: Tuple[SourceFragment, ...],
    new_text: str,
) -> Tuple[Tuple[SourceFragment, ...], bool]:
    """Rebase fragment spans for a coalescer text join.

    ``new_text`` is the already-joined presentation text (``existing``,
    ``existing + sep + incoming``, or either side alone when the other is
    empty or deduplicated away); the separator is derived from the real
    lengths, so ``"\\n"`` appends and ``"\\n\\n"`` caption merges share one
    path. Returns the merged tuple plus whether provenance is complete.
    Missing metadata on either side keeps the attributable spans but flags
    the merge incomplete instead of inventing attribution; contradictory
    spans fail closed to ``((), False)``.
    """
    existing_text = existing_text or ""
    incoming_text = incoming_text or ""
    new_text = new_text or ""
    existing_frags = tuple(existing_fragments or ())
    incoming_frags = tuple(incoming_fragments or ())
    if not new_text:
        return (), False
    if new_text == existing_text and not incoming_text:
        merged = existing_frags
    elif new_text == incoming_text and not existing_text:
        merged = incoming_frags
    elif new_text == existing_text:
        merged = existing_frags
    else:
        sep_len = len(new_text) - len(existing_text) - len(incoming_text)
        if sep_len < 0:
            return (), False
        merged = existing_frags + _shifted(incoming_frags, len(existing_text) + sep_len)
    complete = True
    if (existing_text and not existing_frags) or (incoming_text and not incoming_frags):
        complete = False
    if not all(f.complete for f in merged):
        complete = False
    if not validate_fragments(new_text, merged):
        return (), False
    if not merged:
        complete = False
    return merged, complete


def rebase_event_fragments(
    existing_event: Any, incoming_event: Any, new_text: str, *, old_text: str = ""
) -> None:
    """Fold *incoming_event* provenance into *existing_event* for *new_text*.

    *old_text* is the pre-join presentation text (call sites join first, so
    it must be captured before the assignment). The merged tuple is
    fail-closed (contradictory spans clear provenance entirely).
    Missing provenance on a text-bearing side is propagated as
    ``complete=False`` on the surviving fragments so a later bind cannot
    re-authorize from the attributable span alone.
    Completeness is re-derived from spans at bind time.
    """
    import dataclasses

    try:
        merged, merge_complete = merge_append_fragments(
            old_text,
            tuple(getattr(existing_event, "source_fragments", None) or ()),
            incoming_event.text or "",
            tuple(getattr(incoming_event, "source_fragments", None) or ()),
            new_text,
        )
    except AttributeError:
        return
    if not merge_complete and merged:
        try:
            merged = tuple(dataclasses.replace(f, complete=False) for f in merged)
        except Exception:
            merged = ()
    try:
        existing_event.source_fragments = merged
    except AttributeError:
        pass


def shift_event_fragments(event: Any, delta: int) -> None:
    """Shift all fragment spans of *event* by *delta* (injected prefix text).

    Used when the gateway prepends non-original content (e.g. auto-loaded
    skill payloads) ahead of the user's original text: the user spans stay
    attributable at their new offsets. No-op for empty provenance.
    """
    try:
        frags = tuple(getattr(event, "source_fragments", None) or ())
    except AttributeError:
        return
    if not frags or not delta:
        return
    try:
        event.source_fragments = _shifted(frags, int(delta))
    except AttributeError:
        pass


def invalidate_source_fragments(event: Any) -> None:
    """Drop fragment provenance after the presentation text was rewritten."""
    try:
        event.source_fragments = ()
    except AttributeError:
        pass
