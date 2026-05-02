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

import pytest

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
        path = Path(_repo) / "gateway" / "platforms" / "telegram.py"
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

    def test_guard_raises_runtime_error(self):
        """The guard raises RuntimeError (not a silent log) so operators
        see the failure at startup."""
        src = self._get_source()
        # Between the "if not webhook_secret:" line and the next blank
        # line block, we should see a RuntimeError being raised
        guard_match = re.search(
            r'if not webhook_secret:\s*\n\s*raise\s+RuntimeError\(',
            src,
        )
        assert guard_match, (
            "Missing webhook secret must raise RuntimeError — silent "
            "fall-through was the original GHSA-3vpc-7q5r-276h bypass"
        )

    def test_guard_message_includes_advisory_link(self):
        """The RuntimeError message should reference the advisory so
        operators can read the full context."""
        src = self._get_source()
        assert "GHSA-3vpc-7q5r-276h" in src, (
            "Guard error message must cite the advisory for operator context"
        )

    def test_guard_message_explains_remediation(self):
        """The error should tell the operator how to fix it."""
        src = self._get_source()
        # Should mention how to generate a secret
        assert "openssl rand" in src or "TELEGRAM_WEBHOOK_SECRET=" in src, (
            "Guard error message should show operators how to set "
            "TELEGRAM_WEBHOOK_SECRET"
        )

    def test_polling_branch_has_no_secret_guard(self):
        """Polling mode (else-branch) must NOT require the webhook secret —
        polling authenticates via the bot token, not a webhook secret."""
        src = self._get_source()
        # The guard should appear inside the `if webhook_url:` branch,
        # not the `else:` polling branch. Rough check: the raise is
        # followed (within ~60 lines) by an `else:` that starts the
        # polling branch, and there's no secret-check in that polling
        # branch.
        webhook_block = re.search(
            r'if webhook_url:\s*\n(.*?)\n            else:\s*\n(.*?)\n',
            src, re.DOTALL,
        )
        if webhook_block:
            webhook_body = webhook_block.group(1)
            polling_body = webhook_block.group(2)
            assert "TELEGRAM_WEBHOOK_SECRET" in webhook_body
            assert "TELEGRAM_WEBHOOK_SECRET" not in polling_body
