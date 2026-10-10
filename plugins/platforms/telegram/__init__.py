from . import adapter as _adapter
from .retry_after_compat import install_retry_after_compat

install_retry_after_compat(_adapter)

register = _adapter.register

__all__ = ["register"]
