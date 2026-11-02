"""Platform adapters for messaging integrations (receive, send, auth, media)."""

from importlib import import_module

# Exports resolve on first use, so importing a light helper module (``gateway.platforms._shared``,
# which plugin client tools load in CLI processes) does not pull in the adapter base and sessions.
_EXPORTS = {
    "BasePlatformAdapter": ".base",
    "SendResult": ".base",
    "MessageEvent": ".event",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)
