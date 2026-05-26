"""
WeChat / iLink Bot platform package.

Re-exports the main adapter symbols from ``adapter.py`` (the original
``weixin.py``) so that all existing import paths remain unchanged::

    from gateway.platforms.wxbot import WeixinAdapter       # new — works
    from gateway.platforms.weixin import WeixinAdapter      # old — works (shim)
"""

from .adapter import (  # noqa: F401
    WeixinAdapter,
    check_weixin_requirements,
    send_weixin_direct,
    qr_login,
    ContextTokenStore,
)

__all__ = [
    "WeixinAdapter",
    "check_weixin_requirements",
    "send_weixin_direct",
    "qr_login",
    "ContextTokenStore",
]
