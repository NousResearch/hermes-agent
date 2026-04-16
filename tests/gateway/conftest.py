"""Shared fixtures for gateway tests.

The ``_ensure_telegram_mock`` helper guarantees that a minimal mock of
the ``telegram`` package is registered in :data:`sys.modules` **before**
any test file triggers ``from gateway.platforms.telegram import ...``.

Without this, ``pytest-xdist`` workers that happen to collect
``test_telegram_caption_merge.py`` (bare top-level import, no per-file
mock) first will cache ``ChatType = None`` from the production
ImportError fallback, causing 30+ downstream test failures wherever
``ChatType.GROUP`` / ``ChatType.SUPERGROUP`` is accessed.

Individual test files may still call their own ``_ensure_telegram_mock``
— it short-circuits when the mock is already present.
"""

import sys
from unittest.mock import MagicMock


def _ensure_telegram_mock() -> None:
    """Install a comprehensive telegram mock in sys.modules.

    Idempotent — skips when the real library is already imported.
    Uses ``sys.modules[name] = mod`` (overwrite) instead of
    ``setdefault`` so it wins even if a partial/broken import
    already cached a module with ``ChatType = None``.
    """
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return  # Real library is installed — nothing to mock

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"

    # Real exception classes so ``except (NetworkError, ...)`` clauses
    # in production code don't blow up with TypeError.
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    mod.error.Forbidden = type("Forbidden", (Exception,), {})
    mod.error.InvalidToken = type("InvalidToken", (Exception,), {})
    mod.error.RetryAfter = type("RetryAfter", (Exception,), {"retry_after": 1})
    mod.error.Conflict = type("Conflict", (Exception,), {})

    # Update.ALL_TYPES used in start_polling()
    mod.Update.ALL_TYPES = []

    for name in (
        "telegram",
        "telegram.ext",
        "telegram.constants",
        "telegram.request",
    ):
        sys.modules[name] = mod
    sys.modules["telegram.error"] = mod.error


# Run at collection time — before any test file's module-level imports.
_ensure_telegram_mock()
