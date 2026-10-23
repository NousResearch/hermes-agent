"""Per-turn client presentation (hidden widget intents, large-paste title preview) through canonical admission.

The Desktop composer types some sends ``display_kind="hidden"`` (goal control, inline-preview decisions,
first-run kickoffs): model-facing text no client may render as a user bubble. A large paste carries a
``title_preview`` that only the session titler reads. Both are facts about ONE admitted turn, committed with
the admission row as ``display_v1`` so a retry, a queued drain or a managed child applies exactly what
was admitted, and bound to that admission's own event, never to a later turn on the same task.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from hermes_state_runtime import RuntimeStoreError

HIDDEN = 'hidden'
_TITLE_PREVIEW_LIMIT = 1000
_SUBMIT_FIELDS = ('display_kind', 'title_preview')
_display_turn: ContextVar[tuple | None] = ContextVar('display_turn', default=None)


def submit_display_fields(params):
    """The raw per-turn presentation fields of a ``prompt.submit`` request, forwarded to admission."""
    return {key: params[key] for key in _SUBMIT_FIELDS if key in params}


def admit_display(params):
    """Wire ``display_kind`` / ``title_preview`` -> committed ``display_v1`` (``{}`` when absent).

    Only ``hidden`` is a client-authorable kind; anything else is refused rather than rendered,
    so a typo can never turn machine text into a visible user row."""
    kind, preview = params.get('display_kind'), params.get('title_preview')
    if kind is not None and kind != HIDDEN:
        raise RuntimeStoreError('invalid_params')
    if preview is not None and not isinstance(preview, str):
        raise RuntimeStoreError('invalid_params')
    committed = {}
    if kind:
        committed['kind'] = HIDDEN
    if preview and preview.strip():
        committed['title_preview'] = preview[:_TITLE_PREVIEW_LIMIT]
    return {'display_v1': committed} if committed else {}


def restore_display(committed):
    """A committed ``display_v1`` arriving across a process boundary (managed worker), re-checked by the
    admission validator: only an object admission itself could have committed is accepted."""
    if not isinstance(committed, dict) or not committed or set(committed) - {'kind', 'title_preview'}:
        raise RuntimeStoreError('invalid_params')
    wire = {'display_kind': committed.get('kind'), 'title_preview': committed.get('title_preview')}
    if admit_display(wire).get('display_v1') != committed:
        raise RuntimeStoreError('invalid_params')
    return committed


@contextmanager
def display_turn_scope(admission_id, committed):
    token = _display_turn.set((admission_id, committed) if committed else None)
    try:
        yield
    finally:
        _display_turn.reset(token)


def admission_display(event):
    """The committed presentation of the admission ``event`` executes (its ``message_id`` is the
    admission id), ``{}`` for any other event: a follow-up on the same task never inherits it."""
    bound = _display_turn.get()
    if bound is None or event is None or getattr(event, 'message_id', None) != bound[0]:
        return {}
    return bound[1]


def worker_display_kwargs():
    """``run_conversation`` persistence kwargs of the managed child's single admitted turn."""
    bound = _display_turn.get()
    committed = bound[1] if bound is not None else {}
    return {**({'persist_user_display_kind': HIDDEN} if committed.get('kind') == HIDDEN else {}),
            **({'persist_user_display_metadata': {'title_preview': committed['title_preview']}}
               if committed.get('title_preview') else {})}
