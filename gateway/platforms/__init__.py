"""
Platform adapters for messaging integrations.

Each adapter handles:
- Receiving messages from a platform
- Sending messages/responses back
- Platform-specific authentication
- Message formatting and media handling
"""

from .base import BasePlatformAdapter, MessageEvent, SendResult

# QQAdapter was previously imported eagerly here, but nothing in the codebase
# consumes ``from gateway.platforms import QQAdapter`` (every real call site
# uses the long-form path ``from gateway.platforms.qqbot import QQAdapter``).
# The eager import pulled in qqbot's chunked-upload + keyboards + onboard
# machinery — about 48 ms wall and ~8 MB RSS on every CLI invocation, even
# ones that never touch a gateway adapter.
#
# YuanbaoAdapter migrated to a bundled plugin (plugins/platforms/yuanbao/);
# its re-export is repointed there so any external code that imported
# ``from gateway.platforms import YuanbaoAdapter`` keeps working.
#
# Use PEP 562 module ``__getattr__`` to keep the public re-export working
# while deferring the actual import to first attribute access. This is
# 100% backward-compatible for any external code that still imports the
# adapters from the package root.
__all__ = [
    "BasePlatformAdapter",
    "MessageEvent",
    "SendResult",
    "QQAdapter",
    "YuanbaoAdapter",
]


def __getattr__(name):
    if name == "QQAdapter":
        from .qqbot import QQAdapter  # noqa: F401
        return QQAdapter
    if name == "YuanbaoAdapter":
        from plugins.platforms.yuanbao.adapter import YuanbaoAdapter  # noqa: F401
        return YuanbaoAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
