"""Picker catalogs stay cached until an explicit refresh (scoped per request)."""
from contextlib import contextmanager
from contextvars import ContextVar

_manual = ContextVar('manual_model_catalog', default=False)


def catalog_refresh_is_manual() -> bool:
    return _manual.get()


@contextmanager
def manual_catalog_refresh():
    token = _manual.set(True)
    try:
        yield
    finally:
        _manual.reset(token)
