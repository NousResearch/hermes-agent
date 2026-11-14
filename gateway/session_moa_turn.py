"""A messaging ``/moa <prompt>`` as a durable FIFO turn of its own (#146026).

Under a session authority the prompt is admitted like text, with its preset committed on the
admission (``native_text_v1.moa_turn_v1``). The one-turn MoA route is armed only when that row
executes, so it never lands on another queued turn, and a queued or crash-recovered ``/moa`` row
still runs on MoA. LOCAL sessions keep their own path (their frozen launch policy).
"""
import asyncio
import logging

from gateway.config import Platform

logger = logging.getLogger("gateway.run")


async def default_preset():
    """The configured default MoA preset (the built-in one when config is unreadable)."""
    from hermes_cli.config import load_config
    from hermes_cli.moa_config import normalize_moa_config
    try:
        cfg = await asyncio.to_thread(load_config)
        return normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})["default_preset"]
    except Exception:  # an unreadable config still runs /moa on the built-in default preset
        logger.warning("/moa: config unreadable, using the default MoA preset", exc_info=True)
        return normalize_moa_config({})["default_preset"]


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


async def queue_busy_moa(runner, event, session_key, source):
    """A ``/moa <prompt>`` while a turn runs: its own queued admission with the preset, behind the
    running turn (whose model is untouched). Without an authority to admit it, the old refusal."""
    from agent.i18n import t
    from gateway.run_busy import _busy_accepted
    from hermes_cli.moa_config import moa_usage
    prompt = event.get_command_args().strip()
    if not prompt:
        return moa_usage()
    adapter = runner._delivery_adapter_for(source)
    queued = runner._queued_turn_event(event, prompt, False)
    epoch = runner._route_stop_epoch(session_key)  # before the config read: a /stop during it drops this
    if adapter is None or not admit_moa_turn(runner, queued, source, session_key, await default_preset()):
        return t("gateway.busy.reject_moa")
    await _busy_accepted()
    await runner._queue_command_turn(session_key, event, queued, adapter, t("gateway.queue.queued"),
                                     stop_epoch=epoch)
    return None  # its commit sends the ack
