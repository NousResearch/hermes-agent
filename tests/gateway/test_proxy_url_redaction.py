"""Regression tests for #58994 — nothing may log a proxy URL containing
embedded ``user:pass@`` credentials.

The vulnerability surfaced when ``HTTPS_PROXY`` was set to an Infisical
Agent Vault MITM endpoint (``http://<agent_token>:hermes@host:14322``) and
adapters printed the raw URL at INFO on every gateway restart, leaking the
agent-vault bearer token into ``gateway.log``.

This file consolidates the coverage from #59647, #58999 and #71717:

  1. ``safe_url_for_log`` strips userinfo on the normal parse path.
  2. It also strips userinfo on both *fallback* paths, which previously
     returned the raw string — the ``urlsplit`` exception path (a malformed
     bracketed host raises ``ValueError: Invalid IPv6 URL``) and the
     no-scheme path (``user:pass@host`` with no scheme).
  3. A repo-wide sweep asserts no logging call anywhere passes a raw proxy
     URL variable, so a new adapter cannot silently reintroduce the leak.

The sweep in (3) replaces the exact-source-text assertion from #59647,
which pinned both the formatting and the log level of a single call site
and would break on any reformatting.
"""

from __future__ import annotations

import logging
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.platforms.base import resolve_proxy_url, safe_url_for_log

REPO_ROOT = Path(__file__).resolve().parents[2]


class SafeUrlForLogTests(unittest.TestCase):
    """The helper every call site relies on."""

    def test_strips_userinfo_with_password(self) -> None:
        redacted = safe_url_for_log("http://agent_token:hermes@127.0.0.1:14322")
        self.assertNotIn("agent_token", redacted)
        self.assertNotIn("hermes", redacted)
        self.assertIn("127.0.0.1", redacted)
        self.assertIn("14322", redacted)

    def test_strips_userinfo_without_password(self) -> None:
        redacted = safe_url_for_log("http://supersecrettoken@proxy.example.com:8080")
        self.assertNotIn("supersecrettoken", redacted)
        self.assertIn("proxy.example.com", redacted)

    def test_strips_userinfo_with_empty_password(self) -> None:
        # A regex of the ``://[^:@]+:[^@]+@`` shape misses this one, because
        # the password component is empty. Splitting on the last '@' does not.
        redacted = safe_url_for_log("socks5h://tok:@host.example:9050")
        self.assertNotIn("tok", redacted)
        self.assertIn("host.example:9050", redacted)

    def test_no_userinfo_passes_host_through(self) -> None:
        self.assertEqual(safe_url_for_log("http://127.0.0.1:58309"), "http://127.0.0.1:58309")

    def test_empty_url_returns_empty(self) -> None:
        self.assertEqual(safe_url_for_log(""), "")
        self.assertEqual(safe_url_for_log(None), "")  # type: ignore[arg-type]

    def test_strips_query_and_fragment(self) -> None:
        redacted = safe_url_for_log("http://u:p@host.example:8080/route?token=secret#frag")
        self.assertNotIn("secret", redacted)
        self.assertNotIn("u:p", redacted)
        self.assertIn("host.example", redacted)


class SafeUrlForLogFallbackTests(unittest.TestCase):
    """Both fallback paths previously returned the raw string (#71717, #58999).

    ``urlsplit`` rejects some malformed URLs outright and yields no
    scheme/netloc for others; each fell through to ``return raw``.
    """

    def test_malformed_url_does_not_leak(self) -> None:
        # urlsplit raises ValueError("Invalid IPv6 URL") on this input.
        redacted = safe_url_for_log("http://agent-vault-token:hermes@[bad")
        self.assertNotIn("agent-vault-token", redacted)
        self.assertNotIn("hermes", redacted)

    def test_schemeless_userinfo_does_not_leak(self) -> None:
        redacted = safe_url_for_log("proxyuser:proxypass@10.0.0.9:8080")
        self.assertNotIn("proxyuser", redacted)
        self.assertNotIn("proxypass", redacted)
        self.assertIn("10.0.0.9", redacted)

    def test_unstrippable_userinfo_fails_closed(self) -> None:
        # If a userinfo marker survives the fallback strip, emit nothing
        # rather than something that might still carry a credential.
        self.assertEqual(safe_url_for_log("a@b@host"), "<invalid-url>")

    def test_max_len_is_honoured_on_fallback(self) -> None:
        redacted = safe_url_for_log("user:pass@" + "h" * 200, max_len=20)
        self.assertLessEqual(len(redacted), 20)
        self.assertNotIn("pass", redacted)


class NoRawProxyUrlInLogsTests(unittest.TestCase):
    """Repo-wide sweep: no logging call may pass a raw proxy URL variable.

    This is the regression guard for the whole class of bug rather than for
    the individual call sites that happened to be found in #58994.
    """

    LOG_CALL = re.compile(r"logger\.(?:info|warning|error|debug|critical)\s*\(", re.S)
    PROXY_VAR = re.compile(r"\b(\w*proxy\w*(?:_url)?|_tg_proxy)\b", re.I)
    # Names that are proxy-ish but are not URLs.
    ALLOWED = {"proxy", "proxies", "_proxy_err", "_IRON_PROXY_VERSION", "proxy_kwargs"}

    def _sources(self):
        for path in REPO_ROOT.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel.startswith(("tests/", "build/", "node_modules/", ".venv/")):
                continue
            try:
                yield rel, path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

    @staticmethod
    def _call_text(text: str, start: int) -> str:
        depth = 0
        for i, ch in enumerate(text[start : start + 600]):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start : start + i + 1]
        return text[start : start + 300]

    def test_no_logging_call_passes_a_raw_proxy_url(self) -> None:
        offenders = []
        for rel, text in self._sources():
            for match in self.LOG_CALL.finditer(text):
                call = self._call_text(text, match.start())
                if "safe_url_for_log" in call or "redact" in call:
                    continue
                args = call.split(",", 1)[1] if "," in call else ""
                for name in self.PROXY_VAR.findall(args):
                    if name in self.ALLOWED:
                        continue
                    line = text[: match.start()].count("\n") + 1
                    offenders.append(f"{rel}:{line} passes {name!r} to a logging call")
                    break
        self.assertEqual(
            offenders,
            [],
            "raw proxy URL reaching a log call - route it through "
            "safe_url_for_log():\n  " + "\n  ".join(offenders),
        )


class ResolveProxyUrlSmokeTest(unittest.TestCase):
    """The resolver returns the env value verbatim — redaction is the
    caller's responsibility at the log layer, not the resolver's."""

    def test_returns_env_value_with_userinfo(self) -> None:
        url = "http://agent_token:hermes@127.0.0.1:14322"
        with patch.dict(os.environ, {"HTTPS_PROXY": url}, clear=True):
            self.assertEqual(resolve_proxy_url("TELEGRAM_PROXY"), url)


class LoggingBehaviorTests(unittest.TestCase):
    """The emitted record carries no credential, at the level actually used.

    The call sites log at INFO: operators need to know a proxy is active,
    and the redaction — not the level — is what closes the leak. #59647
    additionally lowered these to DEBUG as defence in depth; that is a
    one-line change if maintainers prefer it.
    """

    def _emit(self, url: str, level: int = logging.INFO) -> list[logging.LogRecord]:
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("test_proxy_url_redaction")
        logger.handlers = [_Capture(level=logging.DEBUG)]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            logger.log(level, "[%s] Using proxy: %s", "Telegram", safe_url_for_log(url))
        finally:
            logger.handlers = []
        return records

    def test_credential_absent_from_emitted_record(self) -> None:
        records = self._emit("http://agent_token:hermes@127.0.0.1:14322")
        self.assertEqual(len(records), 1)
        message = records[0].getMessage()
        self.assertIn("127.0.0.1", message)
        self.assertIn("14322", message)
        self.assertNotIn("agent_token", message)
        self.assertNotIn("hermes", message)

    def test_url_without_creds_logged_unchanged(self) -> None:
        records = self._emit("http://127.0.0.1:58309")
        self.assertIn("127.0.0.1:58309", records[0].getMessage())


if __name__ == "__main__":
    unittest.main()
