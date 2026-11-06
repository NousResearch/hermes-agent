"""Resolve the configured memory provider at agent init, telling the user once why it will not run."""

from __future__ import annotations

import logging

logger = logging.getLogger("run_agent")


# Deduped: the gateway builds a fresh AIAgent per message, so it would warn every turn.
_warned_unavailable_providers: set[tuple[str, str, str]] = set()


def _unavailable_warning_key(name: str, kind: str = "unavailable") -> tuple[str, str, str]:
    """Once per profile home, provider and kind of trouble: one multiplexed gateway/Desktop backend
    serves several profiles, each one running without its memory must be told, and a provider the
    user turns back on that is then unavailable is news even after the "turned off" notice."""
    from hermes_constants import get_hermes_home, hermes_home_key
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
    from plugins.memory import find_provider_dir, load_memory_provider, parked_provider_notice

    provider = load_memory_provider(name)
    if provider is None:
        parked = parked_provider_notice(name)
        if parked:
            # Installed but turned off (by the user or by an update): say so, never re-install it.
            _warn_once(name, parked, say=say, kind="parked")
            return None
        if find_provider_dir(name) is None:
            # The provider left core for the catalog (or was never installed): fetch it once.
            from hermes_cli.memory_provider_migration import recover_at_startup

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
