"""A messaging ``+<alias> rest`` turn: ``/model <alias> --once``, then ``rest`` as the prompt.

Opt-in (``display.model_alias_prefix``); the parse is ``hermes_cli.model_alias_prefix``, shared with
the classic CLI. The switch goes through the real ``/model`` handler inside the turn's claimed slot,
so its one-turn restore is the ``/model --once`` snapshot the ``_handle_message`` finalizer settles
on every exit path, a refused or cancelled turn included.

An alias message is always its own turn: never steered, redirected or merged into another input,
never read as the answer to a pending clarify (``is_alias_turn``). Under a session authority the raw
text is admitted like any message and the prefix is applied when its row EXECUTES, so the one-turn
model never lands on another queued turn. LOCAL sessions (Desktop, Ink TUI) run a frozen launch
route with no one-turn override tier, so their text is left alone.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Optional

from gateway.config import Platform
from hermes_cli.model_alias_prefix import AliasPrefix, split_alias_prefix


def _alias_prefix_sync(runner, event) -> Optional[AliasPrefix]:
    from gateway.run import _load_gateway_config
    source, text = event.source, event.text
    if (not isinstance(text, str) or not text.lstrip().startswith("+") or event.internal
            or not event.allow_gateway_control or source is None or source.platform == Platform.LOCAL):
        return None
    config_path = None
    if getattr(getattr(runner, "config", None), "multiplex_profiles", False):
        config_path = runner._resolve_profile_home_for_source(source) / "config.yaml"
    return split_alias_prefix(text, lambda: _load_gateway_config(config_path))


async def alias_prefix_for(runner, event) -> Optional[AliasPrefix]:
    """The ``+<alias>`` prefix *event* carries for its owning profile, else None."""
    if not isinstance(event.text, str) or not event.text.lstrip().startswith("+"):
        return None
    return await asyncio.to_thread(_alias_prefix_sync, runner, event)


def is_alias_turn(runner, event) -> bool:
    """Synchronous twin for the adapter's burst merge (config is cached; only ``+word`` text reads it)."""
    return runner is not None and _alias_prefix_sync(runner, event) is not None


def _send_after_confirm(runner, event, prefix, session_key) -> None:
    """A cost-confirmed alias keeps its question: approving commits the switch, then sends ``rest``."""
    from tools import slash_confirm
    pending = slash_confirm.get_pending(session_key)
    if not pending or pending.get("command") != "model":
        return
    inner = pending["handler"]

    async def _then_send(choice: str) -> Optional[str]:
        reply = await inner(choice)
        if choice == "cancel":
            return reply
        if reply:
            await runner._send_command_ack(event.source, reply, "model")
        return await runner._handle_message(dataclasses.replace(event, text=prefix.rest))

    slash_confirm.register(session_key, pending["confirm_id"], "model", _then_send, authority=pending.get("authority"))


async def apply_alias_prefix(runner, event, source, session_key) -> tuple[bool, Optional[str]]:
    """Inside the claimed turn: switch this turn's model for ``+<alias> rest``, rewrite it to ``rest``.

    ``(False, None)``: ordinary text, or switched (the ``/model`` reply is sent as an ack) and the
    turn proceeds on ``rest``. ``(True, reply)``: refused, failed or awaiting a cost confirmation;
    no turn runs now, so ``rest`` never reaches a model the user did not pick.
    """
    prefix = await alias_prefix_for(runner, event)
    if prefix is None:
        return False, None
    denied = runner._check_slash_access(source, "model")
    if denied is not None:
        return True, denied
    override_before = runner._session_model_override(session_key)
    async with runner._async_profile_scope_for_source(source):
        reply = await runner._handle_model_command(dataclasses.replace(event, text=prefix.model_command))
    # A committed switch installs a fresh override dict and arms the --once restore; an error, a
    # cost confirmation or a selection-guard refusal leaves both as they were.
    state = runner._peek_session_state(session_key)
    if (state is None or not state.conversation.one_turn_restore
            or runner._session_model_override(session_key) is override_before):
        _send_after_confirm(runner, event, prefix, session_key)
        return True, reply
    if reply:
        await runner._send_command_ack(source, reply, "model")
    event.text = prefix.rest
    return False, None


async def prepare_admitted_turn(runner, event, source, session_key, is_internal) -> tuple[bool, Optional[str]]:
    """The claimed turn's one-turn model and plugin consume, in order; ``(True, reply)`` ends it."""
    from gateway.session_moa_turn import arm_admitted_moa
    arm_admitted_moa(runner, session_key, event)  # restored by the turn's finalizer
    if is_internal:
        return False, None
    # Fail-open plugin consume inside the claimed slot (#129958); once per message.
    consumed, reply = await runner._hm_post_admission_consume(event, source, session_key)
    if consumed:
        return True, reply
    return await apply_alias_prefix(runner, event, source, session_key)
