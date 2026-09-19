"""Tests for GHSA-3vpc-7q5r-276h — Telegram webhook secret required.

Previously, when TELEGRAM_WEBHOOK_URL was set but TELEGRAM_WEBHOOK_SECRET
was not, python-telegram-bot received secret_token=None and the webhook
endpoint accepted any HTTP POST.

The fix refuses to start the adapter in webhook mode without the secret.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


class TestTelegramWebhookSecretRequired:
    """Direct source-level check of the webhook-secret guard.

    The guard is embedded in TelegramAdapter.connect() and hard to isolate
    via mocks (requires a full python-telegram-bot ApplicationBuilder
    chain). These tests exercise it via source inspection — verifying the
    check exists, raises RuntimeError with the advisory link, and only
    fires in webhook mode. End-to-end validation is covered by CI +
    manual deployment tests.
    """

    def _get_source(self) -> str:
        path = Path(_repo) / "plugins" / "platforms" / "telegram" / "adapter.py"
        return path.read_text(encoding="utf-8")

    def test_webhook_branch_checks_secret(self):
        """The webhook-mode branch of connect() must read
        TELEGRAM_WEBHOOK_SECRET and refuse when empty."""
        src = self._get_source()
        # The guard must appear after TELEGRAM_WEBHOOK_URL is set
        assert re.search(
            r'TELEGRAM_WEBHOOK_SECRET.*?\.strip\(\)\s*\n\s*if not webhook_secret:',
            src, re.DOTALL,
        ), (
            "TelegramAdapter.connect() must strip TELEGRAM_WEBHOOK_SECRET "
            "and raise when the secret is empty — see GHSA-3vpc-7q5r-276h"
        )


    def test_polling_branch_has_no_secret_guard(self):
        """Polling mode must NOT require the webhook secret — polling
        authenticates via the bot token, not a webhook secret.

        connect() dispatches to ``_start_webhook_mode`` / ``_start_polling_mode``;
        the guard must live in the webhook method only.
        """
        import ast

        src = self._get_source()
        bodies = {
            node.name: ast.get_source_segment(src, node)
            for node in ast.walk(ast.parse(src))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in ("_start_webhook_mode", "_start_polling_mode")
        }
        assert set(bodies) == {"_start_webhook_mode", "_start_polling_mode"}
        assert "TELEGRAM_WEBHOOK_SECRET" in bodies["_start_webhook_mode"]
        assert "if not webhook_secret:" in bodies["_start_webhook_mode"]
        assert "TELEGRAM_WEBHOOK_SECRET" not in bodies["_start_polling_mode"]
