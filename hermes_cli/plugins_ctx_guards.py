"""The freeze guard wrapped around every mutating :class:`hermes_cli.plugins.PluginContext` entry point.

``hermes_cli.plugins_context`` applies :func:`refuse_registrars_after_freeze` to ``PluginContext`` once,
after its last registrar is defined (and after the abandoned-load guard in ``plugins_load_guard``), to
every ``register_*`` method and to the other mutating methods listed here, mixins included, so a
registrar added later is guarded without anyone remembering to.
"""

from __future__ import annotations

from functools import wraps

from registry_freeze import refuse_if_frozen

# Mutating entry points without the ``register_`` prefix. ``spawn_task`` is freeze-guarded only: a
# task spawned by a load that timed out is still cancelled when that load's registrations are swept.
FREEZE_GUARDED = frozenset({"subscribe", "on_unload", "spawn_task"})


def refuse_after_freeze(method):
    """Raise ``RegistryFrozen`` before the method's first side effect once the hosted boot froze the
    registries. A plugin can keep its ``ctx`` past ``register()`` and call it during any later turn;
    that call would change what every tenant served by the process sees."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        refuse_if_frozen("plugin registry")
        return method(self, *args, **kwargs)

    return wrapped


def refuse_registrars_after_freeze(cls: type) -> None:
    """Freeze-guard each ``register_*`` and :data:`FREEZE_GUARDED` method ``cls`` defines or inherits."""
    names = {name for klass in cls.__mro__ for name in vars(klass)
             if name.startswith("register_") or name in FREEZE_GUARDED}
    for name in sorted(names):
        if callable(method := getattr(cls, name)):
            setattr(cls, name, refuse_after_freeze(method))
