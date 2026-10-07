"""How a gateway turn presents dangerous-command approvals: native buttons or text, or not at all.

An approval prompt nobody permitted will ever see should fail closed at once instead of blocking the
agent thread until ``approvals.timeout``. Adapters report that per chat through an optional
``exec_approval_unanswerable(source)`` method returning the reason (shown to the agent) or None.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.run")


def _renders_exec_approval_buttons(adapter_cls: type) -> bool:
    """True when the adapter class renders native approval buttons. BasePlatformAdapter subclasses
    say so through ``supports_exec_approval_buttons``; anything else (test doubles, relay-style
    duck types) counts when it defines ``send_exec_approval`` itself."""
    probe = getattr(adapter_cls, "supports_exec_approval_buttons", None)
    if callable(probe) and issubclass(adapter_cls, BasePlatformAdapter):
        return bool(probe())
    return getattr(adapter_cls, "send_exec_approval", None) is not None


def unanswerable_approval_reason(adapter: Any, source: Any) -> Optional[str]:
    """Why no permitted person can answer an approval prompt for *source*'s chat, or None.

    The adapter hook is looked up on the class: MagicMock adapters in tests invent any instance
    attribute. A hook that raises leaves the prompt answerable (the normal wait), never blocks it."""
    if not callable(getattr(type(adapter), "exec_approval_unanswerable", None)):
        return None
    try:
        reason = adapter.exec_approval_unanswerable(source)
    except Exception:
        logger.warning("exec_approval_unanswerable failed; prompting as usual", exc_info=True)
        return None
    return reason if isinstance(reason, str) and reason else None
