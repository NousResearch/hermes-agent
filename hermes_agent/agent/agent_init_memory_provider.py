"""Memory provider at agent init: resolve the configured one (telling the user once why it will not run)
and build the kwargs ``MemoryProvider.initialize`` receives."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from hermes_agent.agent.agent_init_params import _GATEWAY_IDENTITY_PARAMS
from hermes_agent.hermes_constants import get_hermes_home

logger = logging.getLogger("hermes_agent.run_agent")


# Deduped: the gateway builds a fresh AIAgent per message, so it would warn every turn.
_warned_unavailable_providers: set[tuple[str, str, str]] = set()


def _unavailable_warning_key(name: str, kind: str = "unavailable") -> tuple[str, str, str]:
    """Once per profile home, provider and kind of trouble: one multiplexed gateway/Desktop backend
    serves several profiles, each one running without its memory must be told, and a provider the
    user turns back on that is then unavailable is news even after the "turned off" notice."""
    from hermes_agent.hermes_constants import get_hermes_home, hermes_home_key
    return hermes_home_key(get_hermes_home()), name, kind


def _warn_once(name: str, message: str, say=None, kind: str = "unavailable") -> None:
    """Log *message* and hand it to the user's sink once per home, provider and kind."""
    key = _unavailable_warning_key(name, kind)
    if key in _warned_unavailable_providers:
        return
    _warned_unavailable_providers.add(key)
    logger.warning(message)
    if say is not None:
        say(message)


def _warn_memory_provider_unavailable(name: str, reason: str = "", say=None) -> None:
    """Warn once per home and provider that a configured memory provider is unavailable.

    ``is_available()`` is a side-effect-free hot-path check and can't log itself; without this
    the provider is silently dropped. ``reason`` (the provider's ``unavailable_reason()`` hint)
    can only reach the user here, so it is appended when present. *say* is the agent's
    user-facing sink: a log line alone leaves the user running without memory unaware.
    """
    _warn_once(name, (
        f"⚠ Memory provider {name!r} is selected but reports unavailable — external memory "
        "is disabled for this session (built-in memory still works). Check the "
        "provider's credentials/config with 'hermes memory status'. Note: "
        "systemd/gateway services do not inherit ~/.hermes/.env automatically; set "
        f"any required variables in the service environment.{f' {reason}' if reason else ''}"
    ), say=say)


def resolve_memory_provider(name: str, say=None):
    """The configured provider ready to add, or None after telling the user why not (once per home)."""
    from hermes_agent.plugins.memory import find_provider_dir, load_memory_provider, parked_provider_notice

    provider = load_memory_provider(name)
    if provider is None:
        parked = parked_provider_notice(name)
        if parked:
            # Installed but turned off (by the user or by an update): say so, never re-install it.
            _warn_once(name, parked, say=say, kind="parked")
            return None
        if find_provider_dir(name) is None:
            # The provider left core for the catalog (or was never installed): fetch it once.
            from hermes_agent.hermes_cli.memory_provider_migration import recover_at_startup

            if recover_at_startup(name, say=say):
                provider = load_memory_provider(name)
        # Installed (now or before) yet not loading: neither "not installed" nor an install notice explains that.
        if provider is None and find_provider_dir(name) is not None:
            _warn_once(name, f"⚠ Memory provider '{name}' is installed but failed to load, so external memory is off "
                             "for this session (built-in memory still works). The log names the error; "
                             "`hermes memory status` shows the plugin's state.", say=say, kind="load")
    if provider is None or provider.is_available():
        return provider
    if _unavailable_warning_key(name) not in _warned_unavailable_providers:
        # unavailable_reason() reads config/probes importlib — skip it once warned.
        reason = ""
        try:
            reason = provider.unavailable_reason()
        except Exception:  # health: allow BLE001 -- a provider's own hint must never break agent init
            logger.debug("unavailable_reason() failed for %s", name, exc_info=True)
        _warn_memory_provider_unavailable(name, reason, say=say)
    return None


def _stored_transcript_kwargs(agent) -> dict[str, Any]:
    """``resumed`` / ``message_count`` for ``MemoryProvider.initialize``. ``message_count`` is how many
    messages the session store holds for this session id when the agent is built; ``resumed`` is
    whether one of them is an assistant reply, i.e. an earlier turn ran. TUI/Desktop store the prompt
    being submitted before the build, so a brand-new chat can count 1 and still be not resumed."""
    db, sid = agent._session_db, agent.session_id
    count, resumed = 0, False
    if db:
        with suppress(Exception):
            count = int((db.get_session_list_row(sid) or {}).get("message_count") or 0)
            resumed = count > 0 and db.latest_message_row_id(sid, role="assistant", require_text=False) is not None
    return {"resumed": resumed, "message_count": count}


def _memory_provider_init_kwargs(agent, platform) -> dict[str, Any]:
    """Scoping kwargs for ``MemoryManager.initialize_all`` (status_callback is CLI-only:
    gateway status travels a different path and the indicator no-ops without it)."""
    kwargs = {
        "session_id": agent.session_id,
        "platform": platform or "cli",
        "hermes_home": str(get_hermes_home()),
        # platform="cron" (scheduler) / "subagent" (delegate_task) → providers skip writes (MemoryProvider.initialize);
        # a fork that loads a provider only for the checkpoint gate names its scope (memory_agent_context).
        "agent_context": getattr(agent, "memory_agent_context", None)
        or (platform if platform in ("cron", "subagent") else "primary"),
    }
    # Documented in MemoryProvider.initialize: forks tell their shape apart by platform + parent.
    if getattr(agent, "_parent_session_id", None):
        kwargs["parent_session_id"] = agent._parent_session_id
    if kwargs["platform"] == "cli":
        kwargs["warning_callback"] = agent._emit_warning
        kwargs["status_callback"] = agent._emit_status
    kwargs.update(_stored_transcript_kwargs(agent))
    # Session title (e.g. honcho derives chat-scoped session keys from it).
    if agent._session_db:
        with suppress(Exception):
            _st = agent._session_db.get_session_title(agent.session_id)
            if _st:
                kwargs["session_title"] = _st
                _source = agent._session_db.get_session_title_source(agent.session_id)
                if _source:
                    kwargs["session_title_source"] = _source
    # Gateway user/chat identity for per-user scoping (gateway_session_key: stable per-chat
    # Honcho session isolation).
    for _ident in _GATEWAY_IDENTITY_PARAMS:
        _val = getattr(agent, f"_{_ident}")
        if _val:
            kwargs[_ident] = _val
    if agent.session_cwd:
        kwargs["cwd"] = agent.session_cwd
    # Profile identity for per-profile provider scoping
    with suppress(Exception):
        from hermes_agent.hermes_cli.profiles import get_active_profile_name
        kwargs["agent_identity"] = get_active_profile_name()
        kwargs["agent_workspace"] = "hermes"
    return kwargs
