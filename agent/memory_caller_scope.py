"""Which profile a memory-provider call or hook belongs to (#140058).

On a multiplexed gateway every served profile's turns run in one process. Each agent's manager owns
its own provider instance, bound by ``initialize_all`` to one profile home; the provider-side half of
the rule is the multiplexing contract in ``agent/memory_provider.py``. The manager's half lives here:

- a tool call or lifecycle hook runs only when the caller explicitly names the profile the manager was
  initialized for (the context's HERMES_HOME override, or the home its secret scope was built for).
  On a multiplexed process a caller that names no profile is refused, never resolved to the process's
  own profile; a single-profile process falls back to its home as before;
- one provider instance serves one profile: a plugin that hands several agents the same instance is
  refused for every profile after the first, instead of being rebound to it.

Refusals log the check's code and both homes (errors.log on every host, gateway.log on a gateway); the
model/chat only ever sees a generic refusal.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CALLER_PROFILE_REFUSAL = "memory_caller_profile_unresolved"
SHARED_INSTANCE_REFUSAL = "memory_provider_instance_shared"

# id(provider) -> (weak reference to that very instance, profile home key). Keyed by identity, never
# by the provider's own __hash__/__eq__: an unhashable or value-equal provider is still tracked.
_claims: dict[int, tuple[weakref.ref, str]] = {}
_claims_lock = threading.RLock()  # reentrant: a weakref callback can fire from GC inside the lock


def refused_tool_message(tool_name: str) -> str:
    """What the model/chat sees for a refused memory tool call: no check name, no home paths."""
    return f"Memory tool '{tool_name}' refused. Ask the operator to check `hermes logs errors`."


def _home_key(home: Any) -> Optional[str]:
    if not home:
        return None
    from hermes_constants import hermes_home_key
    try:
        return hermes_home_key(home)
    except (OSError, RuntimeError, ValueError):  # unresolvable path: no key, so the call is refused
        return None


def _explicit_caller_home() -> Optional[str]:
    """The profile the caller's context names: its HERMES_HOME override, else the home its secret
    scope was built for. None when the context names no profile (``get_hermes_home()`` would only fall
    back to the process home, which proves nothing on a multiplexed process)."""
    from agent.secret_scope import current_secret_scope_home
    from hermes_constants import get_hermes_home_override
    return get_hermes_home_override() or current_secret_scope_home()


def _refuse(label: str, reason: str, caller_home: Any, bound_home: Any, multiplexed: bool) -> None:
    logger.warning("Memory %s refused (%s): %s; caller profile home %r, provider bound to %r, multiplexed=%s",
                   label, CALLER_PROFILE_REFUSAL, reason, caller_home, bound_home, multiplexed)


def caller_scope(bound: Optional[dict[str, Any]], label: str) -> Optional[dict[str, Any]]:
    """Provider kwargs naming the calling profile, or None when the call/hook must be refused.

    ``bound`` is what ``initialize_all`` recorded (``hermes_home``, ``agent_identity``); ``label``
    names the call in the log (``tool <name>`` / ``hook <name>``)."""
    from agent.secret_scope import is_multiplex_active
    bound = bound or {}
    bound_home = bound.get("hermes_home")
    multiplexed = is_multiplex_active()
    caller_home = _explicit_caller_home()
    if caller_home is None:
        if multiplexed:
            _refuse(label, "the caller names no profile on a process serving several", None, bound_home, multiplexed)
            return None
        from hermes_constants import get_hermes_home
        caller_home = str(get_hermes_home())  # single-profile process: its own home is the only one
    if not bound_home:
        if multiplexed:
            _refuse(label, "the memory manager recorded no profile", caller_home, None, multiplexed)
            return None
        return {"hermes_home": str(caller_home)}
    caller_key = _home_key(caller_home)
    if caller_key is None or caller_key != _home_key(bound_home):
        _refuse(label, "the caller's profile is not the one its provider was initialized for",
                caller_home, bound_home, multiplexed)
        return None
    scope: dict[str, Any] = {"hermes_home": str(bound_home)}
    if bound.get("agent_identity"):
        scope["agent_identity"] = bound["agent_identity"]
    return scope


def _forget(pid: int, ref: weakref.ref) -> None:
    with _claims_lock:
        if _claims.get(pid, (None,))[0] is ref:
            del _claims[pid]


def claim_instance(provider: Any, home: Any) -> bool:
    """Record that ``provider`` (the instance) is bound to ``home``. False on a multiplexed process
    when another profile already bound this same instance (initializing it again would rebind the
    other profile's store), or when the instance can't be tracked at all (no weak reference)."""
    from agent.secret_scope import is_multiplex_active
    key = _home_key(home) or str(home)
    multiplexed = is_multiplex_active()
    pid = id(provider)
    with _claims_lock:
        entry = _claims.get(pid)
        owner = entry[1] if entry is not None and entry[0]() is provider else None
        if owner is not None and owner != key and multiplexed:
            logger.warning("Memory provider '%s' refused for profile home %r (%s): this same provider instance "
                           "is already bound to %r; a provider must give every agent its own instance",
                           getattr(provider, "name", "?"), str(home), SHARED_INSTANCE_REFUSAL, owner)
            return False
        try:
            ref = weakref.ref(provider, lambda dead, pid=pid: _forget(pid, dead))
        except TypeError:
            if multiplexed:
                logger.warning("Memory provider '%s' refused for profile home %r (%s): the instance can't be "
                               "tracked (no weak reference support), so sharing it can't be ruled out",
                               getattr(provider, "name", "?"), str(home), SHARED_INSTANCE_REFUSAL)
                return False
            return True
        _claims[pid] = (ref, key)
    return True


def in_owner_home(bound: Optional[dict[str, Any]], fn: Callable[[], Any]) -> Callable[[], Any]:
    """``fn`` run under the owning profile's HERMES_HOME override, set explicitly: a background
    worker must not depend on the submitter's context having carried it."""
    home = (bound or {}).get("hermes_home")
    if not home:
        return fn

    def _run() -> Any:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        token = set_hermes_home_override(str(home))
        try:
            return fn()
        finally:
            reset_hermes_home_override(token)
    return _run
