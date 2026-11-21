"""Request-origin fence shared by the ordinary loader and catalog publication.

Ordinary startup has no origin and preserves its existing activation policy.
HTTP cold activation never downgrades an isolation choice while awaiting locks.
"""
import contextvars

_origin: contextvars.ContextVar[tuple | None] = contextvars.ContextVar('catalog_bootstrap_origin', default=None)


def current(manager):
    origin = _origin.get()
    if origin is None:
        return True
    from hermes_cli.plugins import get_plugin_manager
    return origin == (manager, getattr(manager, '_catalog_epoch', 0)) and get_plugin_manager() is manager


def allowed(manager):
    if _origin.get() is None:
        return True
    from hermes_cli.plugin_isolation import isolation_mode
    return current(manager) and isolation_mode() == 'host'
