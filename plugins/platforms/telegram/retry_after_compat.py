"""Compatibility glue for Telegram RetryAfter duration values.

python-telegram-bot 22.2 allows ``RetryAfter.retry_after`` to be a
``datetime.timedelta`` (and documents that timedelta will become the default
in a future major version). The legacy Telegram adapter still converts those
values through the module-global ``float`` name and builds ``SendResult``
objects after exhausted inline retries.

Keep the compatibility boundary outside the large adapter module so rebases do
not overwrite unrelated Telegram work while the adapter is being decomposed.
"""

from __future__ import annotations

import builtins
import re
from typing import Any

_RETRY_AFTER_RE = re.compile(
    r"(?:retry\s+(?:after|in)\s+|flood_control:)(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def install_retry_after_compat(adapter_module: Any) -> None:
    """Teach the Telegram adapter to preserve timedelta retry delays."""
    if getattr(adapter_module, "_retry_after_compat_installed", False):
        return

    real_float = builtins.float
    base_send_result = adapter_module.SendResult

    class DurationFloat(real_float):
        def __new__(cls, value: Any = 0.0):
            total_seconds = getattr(value, "total_seconds", None)
            if callable(total_seconds):
                value = total_seconds()
            return real_float.__new__(cls, value)

    class TelegramSendResult(base_send_result):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if getattr(self, "retry_after", None) is not None:
                return
            error = getattr(self, "error", None)
            if not error:
                return
            match = _RETRY_AFTER_RE.search(str(error))
            if match is not None:
                self.retry_after = real_float(match.group(1))

    DurationFloat.__name__ = "float"
    TelegramSendResult.__name__ = "SendResult"

    # Adapter functions resolve these names from module globals at call time.
    # Normal numeric conversions still behave exactly like builtins.float;
    # timedelta-compatible values additionally use total_seconds().
    adapter_module.float = DurationFloat
    adapter_module.SendResult = TelegramSendResult
    adapter_module._retry_after_compat_installed = True
