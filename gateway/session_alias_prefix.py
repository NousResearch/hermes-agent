"""A messaging ``+<alias> rest`` turn: ``/model <alias> --once``, then ``rest`` as the prompt.

Opt-in (``display.model_alias_prefix``); the parse is ``hermes_cli.model_alias_prefix``, shared with
the classic CLI. The switch goes through the real ``/model`` handler, so its one-turn restore is the
``/model --once`` snapshot the turn finalizer in ``_handle_message`` settles on every exit path.

Under a session authority the raw text is admitted like any message and the prefix is applied when
its row EXECUTES, so the one-turn model never lands on another queued turn. A follow-up typed while
a turn runs is never steered or redirected into that turn: it becomes its own admission. LOCAL
sessions (Desktop, Ink TUI) run a frozen launch route with no one-turn override tier, so their
text is left alone.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Optional

from gateway.config import Platform
from hermes_cli.model_alias_prefix import AliasPrefix, split_alias_prefix


async def alias_prefix_for(runner, event, source) -> Optional[AliasPrefix]:
    """The ``+<alias>`` prefix *event* carries for its owning profile, else None."""
    from gateway.run import _load_gateway_config
    text = event.text
    if (not isinstance(text, str) or not text.lstrip().startswith("+") or event.internal
            or not event.allow_gateway_control or source.platform == Platform.LOCAL):
        return None
    config_path = None
    if getattr(getattr(runner, "config", None), "multiplex_profiles", False):
        config_path = runner._resolve_profile_home_for_source(source) / "config.yaml"
    return await asyncio.to_thread(split_alias_prefix, text, lambda: _load_gateway_config(config_path))


async def apply_alias_prefix(runner, event, source, session_key) -> tuple[bool, Optional[str]]:
    """Switch this turn's model for a ``+<alias> rest`` message and rewrite it to ``rest``.

    ``(False, None)``: ordinary text, or switched (the ``/model`` reply is sent as an ack) and the
    turn proceeds on ``rest``. ``(True, reply)``: the switch was refused or failed; no turn runs,
    so ``rest`` never reaches a model the user did not pick.
    """
    prefix = await alias_prefix_for(runner, event, source)
    if prefix is None:
        return False, None
    denied = runner._check_slash_access(source, "model")
    if denied is not None:
        return True, denied
    override_before = runner._session_model_override(session_key)
    command = dataclasses.replace(event, text=prefix.model_command)
    async with runner._async_profile_scope_for_source(source):
        reply = await runner._handle_model_command(command)
    # A committed switch installs a fresh override dict and arms the --once restore; an error, a
    # cost confirmation or a selection-guard refusal leaves both as they were.
    state = runner._peek_session_state(session_key)
    if (state is None or not state.conversation.one_turn_restore
            or runner._session_model_override(session_key) is override_before):
        return True, reply
    if reply:
        await runner._send_command_ack(source, reply, "model")
    event.text = prefix.rest
    return False, None
