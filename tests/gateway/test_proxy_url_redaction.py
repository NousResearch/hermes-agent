"""Regression tests for #58994 — platform adapters must never log a proxy
URL containing embedded ``user:pass@`` credentials at INFO.

The vulnerability surfaced when ``HTTPS_PROXY`` was set to an Infisical
Agent Vault MITM endpoint (``http://<agent_token>:hermes@host:14322``)
and the adapter printed the raw URL at INFO on every gateway restart,
leaking the agent-vault bearer token into ``gateway.log``.

The fix:

  1. Route the proxy URL through :func:`safe_url_for_log` so the
     ``user:pass@`` userinfo is stripped (preserving host:port).
  2. Lower the log level to DEBUG so the URL doesn't surface in default
     INFO+ output. Operators who need to debug a proxy path can enable
     DEBUG on the relevant logger.
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from gateway.platforms.base import safe_url_for_log


class SafeUrlForLogTests(unittest.TestCase):
    """``safe_url_for_log`` strips userinfo — verify the helper used by the
    adapter fix is itself behaving correctly.
    """

    def test_strips_userinfo_with_password(self) -> None:
        url = "http://agent_token:hermes@127.0.0.1:14322"
        redacted = safe_url_for_log(url)
        self.assertNotIn("agent_token", redacted)
        self.assertNotIn("hermes", redacted)
        self.assertIn("127.0.0.1", redacted)
        self.assertIn("14322", redacted)

    def test_strips_userinfo_without_password(self) -> None:
        url = "http://supersecrettoken@proxy.example.com:8080"
        redacted = safe_url_for_log(url)
        self.assertNotIn("supersecrettoken", redacted)
        self.assertIn("proxy.example.com", redacted)
        self.assertIn("8080", redacted)

    def test_no_userinfo_passes_host_through(self) -> None:
        url = "http://127.0.0.1:58309"
        self.assertEqual(safe_url_for_log(url), "http://127.0.0.1:58309")

    def test_empty_url_returns_empty(self) -> None:
        self.assertEqual(safe_url_for_log(""), "")
        self.assertEqual(safe_url_for_log(None), "")  # type: ignore[arg-type]

    def test_truncates_long_paths(self) -> None:
        # safe_url_for_log also collapses long paths to ``.../<basename>``;
        # verify the userinfo strip works even when the path is long.
        url = "http://u:p@host.example.com:8080/very/long/path/to/endpoint"
        redacted = safe_url_for_log(url)
        self.assertNotIn("u:p", redacted)
        self.assertNotIn("p@host", redacted)
        self.assertIn("host.example.com", redacted)
        self.assertIn("8080", redacted)


class TelegramAdapterProxyLogStatementTests(unittest.TestCase):
    """Static-source test: assert the adapter's proxy-detected log line
    is at DEBUG and uses ``safe_url_for_log`` so userinfo is stripped.
    """

    def _adapter_source(self) -> str:
        from pathlib import Path
        adapter_path = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "telegram" / "adapter.py"
        return adapter_path.read_text(encoding="utf-8")

    def test_logs_at_debug_not_info(self) -> None:
        source = self._adapter_source()
        # The exact (post-fix) line must use ``logger.debug`` and pass
        # ``safe_url_for_log(proxy_url)`` to the formatter.
        needle = (
            'logger.debug(\n'
            '                    "[%s] Proxy detected; passing explicitly to HTTPXRequest: %s",\n'
            '                    self.name,\n'
            '                    safe_url_for_log(proxy_url),\n'
            '                )'
        )
        self.assertIn(needle, source)
        # Guard against the prior buggy INFO log line being reintroduced.
        self.assertNotIn(
            'logger.info("[%s] Proxy detected; passing explicitly to HTTPXRequest: %s"',
            source,
        )

    def test_imports_safe_url_for_log(self) -> None:
        source = self._adapter_source()
        self.assertIn("safe_url_for_log", source)


class ResolveProxyUrlSmokeTest(unittest.TestCase):
    """Confirm the real :func:`resolve_proxy_url` returns the env value
    verbatim so callers can apply their own redaction in the log layer
    (the redaction is the caller's responsibility, not the resolver's).
    """

    def test_returns_env_value_with_userinfo(self) -> None:
        import os
        from gateway.platforms.base import resolve_proxy_url
        url = "http://agent_token:hermes@127.0.0.1:14322"
        with patch.dict(os.environ, {"HTTPS_PROXY": url}, clear=True):
            self.assertEqual(resolve_proxy_url("TELEGRAM_PROXY"), url)


class LoggingBehaviorTests(unittest.TestCase):
    """Exercise the post-fix logging shape directly: when a caller logs at
    DEBUG with ``safe_url_for_log`` applied, an INFO-level handler does
    NOT see the credential.
    """

    def _emit(self, url: str, level: int) -> list[logging.LogRecord]:
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.DEBUG)
        logger = logging.getLogger("test_proxy_url_redaction")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            logger.log(level, "[%s] Proxy detected; passing explicitly to HTTPXRequest: %s",
                       "Telegram", safe_url_for_log(url))
        finally:
            logger.handlers = []
        return records

    def test_url_with_creds_only_visible_at_debug(self) -> None:
        url = "http://agent_token:hermes@127.0.0.1:14322"
        records = self._emit(url, level=logging.DEBUG)
        self.assertEqual(len(records), 1)
        msg = records[0].getMessage()
        self.assertIn("127.0.0.1", msg)
        self.assertIn("14322", msg)
        self.assertNotIn("agent_token", msg)
        self.assertNotIn("hermes", msg)

    def test_url_without_creds_logged_unchanged(self) -> None:
        url = "http://127.0.0.1:58309"
        records = self._emit(url, level=logging.DEBUG)
        self.assertEqual(len(records), 1)
        msg = records[0].getMessage()
        self.assertIn("127.0.0.1:58309", msg)
        self.assertNotIn("***", msg)

    def test_log_level_is_debug_not_info(self) -> None:
        url = "http://agent_token:hermes@127.0.0.1:14322"
        records = self._emit(url, level=logging.DEBUG)
        self.assertEqual(len(records), 1)
        self.assertLess(records[0].levelno, logging.INFO)


if __name__ == "__main__":
    unittest.main()