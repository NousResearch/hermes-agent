"""Where a key-less ``PluginContext.inject_message(text)`` goes when plugin code runs for the gateway.

Before the one-gateway runtime each classic CLI loaded its own plugins, so a key-less inject meant that
CLI's own chat. The gateway's copy of a plugin (or its plugin host's) serves every session of its
profile, so the target is picked here, in order:

1. The local session the call is working for: the session bound in the calling context (its slash
   command, a hook fired for it, a tool or hook in its turn). Never a guess, however many are open.
2. No session in the calling context (a timer, the plugin's own thread): the one local session of the
   plugin's profile launched by the classic CLI (frozen policy source ``cli``) with a viewer attached.
3. Otherwise refused by name: nothing is injected anywhere.

A calling context that is not a local session of the plugin's profile (a messaging chat) keeps main's
refusal. A resolved target is committed like a keyed plugin inject under the session authority: a
queued producer admission in that session's FIFO, run once, in order.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
import threading
import time

from gateway.config import Platform

logger = logging.getLogger("gateway.run")

# One refusal warning per (plugin, kind) per window: a plugin retrying on a timer must not flood the log.
# (plugin, kind) -> when it last warned; a mark past the window is dropped at the next refusal.
_WARN_WINDOW_S = 300.0
_warned: dict[tuple[str, str], float] = {}
_warned_lock = threading.Lock()  # refusals run on plugin threads


def _refuse(plugin_id: str, kind: str, detail: str) -> None:
    now = time.monotonic()
    with _warned_lock:
        for key in [key for key, at in _warned.items() if now - at >= _WARN_WINDOW_S]:
            del _warned[key]
        repeated = (plugin_id, kind) in _warned
        if not repeated:
            _warned[(plugin_id, kind)] = now
    if repeated:
        logger.debug("inject_message from plugin %s refused again: %s", plugin_id, detail)
        return
    logger.warning("inject_message from plugin %s has no session key and %s; nothing was injected. Pass "
                   "session_key= (or origin=) to choose the chat.", plugin_id, detail)


def _local(runner, live) -> bool:
    from gateway.session_local import authorize_local_source
    source = live.source
    return source is not None and source.platform == Platform.LOCAL and authorize_local_source(runner, source) is True


def _launched_by_classic_cli(runner, live) -> bool:
    from gateway.session_policy import policy_for_source
    from hermes_state_runtime import RuntimeStoreError
    try:
        policy = policy_for_source(runner, live.source)
    except RuntimeStoreError:
        return False
    return policy is not None and policy.source == "cli"


def resolve(runner, *, plugin_home, calling_session_key: str, plugin_id: str):
    """``(authority, live session)`` the key-less inject goes to, or ``None`` (logged). Runs on the
    plugin's calling thread so a refusal is the plugin's own ``False``; it only reads the authority's
    live map, never the store."""
    from gateway.session_authorities import authority_for_home
    authority = authority_for_home(runner, plugin_home) if plugin_home is not None else None
    if authority is None:
        _refuse(plugin_id, "unserved", f"its profile home {plugin_home} is not served by this gateway")
        return None
    sessions = list(authority.sessions.values())
    if calling_session_key:
        live = next((s for s in sessions if s.route == calling_session_key and _local(runner, s)), None)
        if live is None:
            # Main's answer for a key-less inject from a messaging chat's command, hook or tool.
            logger.warning("inject_message: gateway mode requires an existing session_key or an origin")
            return None
        return authority, live
    attached = [s for s in sessions if s.subscribers and _local(runner, s) and _launched_by_classic_cli(runner, s)]
    if len(attached) == 1:
        return authority, attached[0]
    if not attached:
        _refuse(plugin_id, "none", "no classic CLI session is attached in its profile")
        return None
    _refuse(plugin_id, "several", f"{len(attached)} classic CLI sessions are attached in its profile")
    return None


async def admit(runner, authority, live, *, content: str, plugin_id: str) -> bool:
    """Commit the text to *live*'s durable FIFO through the keyed path's producer admission, under the
    plugin's profile (whose authority resolved it)."""
    from gateway.session_authorities import active_authority
    if not runner._plugin_injection_accepting():
        return False
    entry = await runner.async_session_store.lookup_by_session_key(live.route)
    if entry is None or not runner._plugin_injection_accepting():
        return False
    scope = contextlib.nullcontext()
    if getattr(getattr(runner, "config", None), "multiplex_profiles", False) is True:
        from gateway.run import _async_profile_runtime_scope
        scope = _async_profile_runtime_scope(Path(authority.profile_id))
    async with scope:
        if active_authority(runner) is not authority:
            return False
        return bool(await runner._admit_plugin_injection(
            runner._adapter_for_source(live.source), live.source, session_key=live.route,
            session_id=entry.session_id, content=content, plugin_id=plugin_id))
