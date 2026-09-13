"""Per-turn client surface (floating HUD, live voice) carried through canonical admission.

A surface is a fact about ONE turn, never about the session: the same session alternates
between the app window, the HUD and a spoken delegation. It is committed with the admission
row and reaches the model as a note on the user message (api_content sidecar), so the
persisted user row stays the words the user typed and the frozen system prefix is untouched.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from hermes_state_runtime import RuntimeStoreError

SURFACES = frozenset({'hud', 'voice-live'})
_VOICE_CONTEXT_LIMIT = 6000
_SUBMIT_FIELDS = ('surface', 'voice_context', 'interrupted')
_surface_turn = ContextVar('surface_turn', default=None)


def submit_surface_fields(params):
    """The raw per-turn surface fields of a ``prompt.submit`` request, forwarded to admission."""
    return {key: params[key] for key in _SUBMIT_FIELDS if key in params}


def admit_surface(params):
    """Wire ``surface`` / ``voice_context`` / ``interrupted`` -> committed ``surface_v1`` (``{}`` when absent)."""
    surface, context, interrupted = (params.get(key) for key in _SUBMIT_FIELDS)
    if surface is not None and surface not in SURFACES:
        raise RuntimeStoreError('invalid_params')
    # The spoken transcript only makes sense for a live-voice delegation; anywhere else it is
    # a client smuggling model input past the persisted row.
    if context is not None and (surface != 'voice-live' or not isinstance(context, str)):
        raise RuntimeStoreError('invalid_params')
    if interrupted is not None and type(interrupted) is not bool:
        raise RuntimeStoreError('invalid_params')
    committed = {}
    if surface:
        committed['surface'] = surface
    if context:
        committed['voice_context'] = context[:_VOICE_CONTEXT_LIMIT]
    if interrupted:
        committed['interrupted'] = True
    return {'surface_v1': committed} if committed else {}


def _hud_note(committed, valid_tool_names):
    from agent.prompt_builder import hud_surface_note
    return hud_surface_note(valid_tool_names)


def _voice_live_note(committed, valid_tool_names):
    from tools.voice_live import voice_live_turn_note
    return voice_live_turn_note(committed.get('voice_context') or '')


_SURFACE_NOTES = {'hud': _hud_note, 'voice-live': _voice_live_note}


def surface_note(committed, valid_tool_names=None):
    """The model-bound note for a committed surface (``""`` for the plain app window): barge-in
    first, then the HUD read-the-window-below prior or the spoken-delegation contract."""
    notes = []
    if committed.get('interrupted'):
        from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE
        notes.append(SPEECH_INTERRUPTED_NOTE)
    render = _SURFACE_NOTES.get(committed.get('surface'))
    if render is not None:
        notes.append(render(committed, valid_tool_names))
    return '\n\n'.join(note for note in notes if note)


@contextmanager
def surface_turn_scope(committed):
    token = _surface_turn.set(committed)
    try:
        yield
    finally:
        _surface_turn.reset(token)


def surface_turn_note(agent):
    """The executing admission's surface note, gated on the tools this agent actually has."""
    committed = _surface_turn.get()
    if not committed:
        return ''
    return surface_note(committed, getattr(agent, 'valid_tool_names', None))
