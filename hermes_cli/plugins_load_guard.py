"""The abandoned-load guard behind :class:`hermes_cli.plugins.PluginContext`.

A plugin's ``register()`` runs on a worker bounded by ``plugins.load_timeout_seconds``. When it
overruns, the loader marks the context abandoned (``PluginContext._abandon_load``) and sweeps what it
registered, but the worker thread may still be executing ``register()``; every registrar it reaches
afterwards must be a no-op, or the late registration lands in registries the failure path already
swept (#108139). ``hermes_cli.plugins`` applies :func:`ignore_registrars_after_abandoned_load` to
``PluginContext`` once, after its last registrar is defined.
"""

from __future__ import annotations

import logging
from functools import wraps

from hermes_cli.plugins_ledger import undo_abandoned_registration

logger = logging.getLogger("hermes_cli.plugins")


def _ignore_after_abandoned_load(method):
    """No-op a registrar once its load timed out, and undo one that was running when that happened.

    The guard's first check covers a registrar called after the deadline. A registrar that was
    already running when the deadline fired passed that check, so its registration lands after the
    failure path snapshotted the ledger and is never disposed (#149326): re-check the flag once the
    registrar returns and undo everything the abandoned load registered.
    """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "_load_abandoned", False):
            logger.warning(
                "Plugin '%s' called %s() after its load timed out; ignored", self.manifest.name,
                method.__name__,
            )
            return None
        handle = method(self, *args, **kwargs)
        if getattr(self, "_load_abandoned", False):
            return undo_abandoned_registration(self, method.__name__)
        return handle

    return wrapped


def ignore_registrars_after_abandoned_load(cls: type) -> None:
    """Guard every mutating entry point plugins reach through ``ctx`` during register(): each
    ``register_*``, ``subscribe`` and ``on_unload`` that ``cls`` defines or inherits from a mixin.
    Applied by name over the MRO, so the guard cannot drift from the surface as registrars or
    mixins are added."""
    names = {name for klass in cls.__mro__ for name in vars(klass)
             if name.startswith("register_") or name in {"subscribe", "on_unload"}}
    for name in sorted(names):
        if callable(method := getattr(cls, name)):
            setattr(cls, name, _ignore_after_abandoned_load(method))
