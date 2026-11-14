"""A messaging ``/moa <prompt>`` as a durable FIFO turn of its own (#146026).

Under a session authority the prompt is admitted like text, with its preset committed on the
admission (``native_text_v1.moa_turn_v1``). The one-turn MoA route is armed only when that row
executes, so it never lands on another queued turn, and a queued or crash-recovered ``/moa`` row
still runs on MoA. LOCAL sessions keep their own path (their frozen launch policy).
"""
from gateway.config import Platform


def moa_override(preset):
    """The session-override shape of the virtual MoA provider for *preset*."""
    return {"provider": "moa", "model": preset, "base_url": "moa://local",
            "api_key": "moa-virtual-provider", "api_mode": "chat_completions"}


def admit_moa_turn(runner, event, source, session_key, preset):
    """Mark *event* (its text already the bare prompt) to be admitted with *preset*. False when
    it stays in-process: no authority admits it, a LOCAL session, or another one-turn model is
    already armed for the next turn."""
    if source.platform == Platform.LOCAL or not runner._command_turn_admitted(event, session_key):
        return False
    event._moa_preset = preset
    return True


def stamp_moa_turn(event, envelope):
    if preset := getattr(event, "_moa_preset", None):
        envelope["moa_turn_v1"] = {"preset": preset}


def restore_moa_turn(envelope, event):
    if "moa_turn_v1" in envelope:
        event._moa_preset = envelope["moa_turn_v1"]["preset"]


def arm_admitted_moa(runner, session_key, event):
    """The executing ``/moa`` admission's one-turn route, restored by the turn's own finalizer."""
    preset = getattr(event, "_moa_preset", None)
    if not preset:
        return
    runner._claim_one_turn_restore(session_key)
    runner._session_state(session_key).conversation.model_override = moa_override(preset)
    runner._evict_cached_agent(session_key)
