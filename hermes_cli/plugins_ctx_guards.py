"""Guards wrapped around every mutating :class:`hermes_cli.plugins.PluginContext` entry point.

``hermes_cli.plugins`` applies them by name to every ``register_*`` method and to the other mutating
methods listed here, so a registrar added later is guarded without anyone remembering to.
"""

from __future__ import annotations

import logging
from functools import wraps

from registry_freeze import refuse_if_frozen

logger = logging.getLogger("hermes_cli.plugins")

# Mutating entry points without the ``register_`` prefix. ``spawn_task`` is freeze-guarded only: a
# task spawned by a load that timed out is still cancelled when that load's registrations are swept.
ABANDONED_LOAD_GUARDED = frozenset({"subscribe", "on_unload"})
FREEZE_GUARDED = ABANDONED_LOAD_GUARDED | {"spawn_task"}


def ignore_after_abandoned_load(method):
    """Turn a registrar into a no-op once the context's load timed out: the abandoned worker thread may
    still be executing register(), and a late registration would land in registries that the failure
    path already swept (#108139)."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "_load_abandoned", False):
            logger.warning(
                "Plugin '%s' called %s() after its load timed out; ignored", self.manifest.name,
                method.__name__,
            )
            return None
        return method(self, *args, **kwargs)

    return wrapped


def refuse_after_freeze(method):
    """Raise ``RegistryFrozen`` before the method's first side effect once the hosted boot froze the
    registries. A plugin can keep its ``ctx`` past ``register()`` and call it during any later turn;
    that call would change what every tenant served by the process sees."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        refuse_if_frozen("plugin registry")
        return method(self, *args, **kwargs)

    return wrapped
