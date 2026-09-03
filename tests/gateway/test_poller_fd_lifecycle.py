"""Poller socket lifecycle — proxied fd-leak regressions (#79889).

On macOS (256 soft fd limit) a gateway routing weixin/email pollers through
a local HTTP proxy leaked one TCP socket per failed poll/connect cycle until
``[Errno 24] Too many open files`` crashed the gateway. Two code-side gaps:

1. email: ``imaplib.IMAP4.logout()`` only swallows ``OSError``; on a broken
   connection ``LOGOUT`` raises ``IMAP4.abort`` *before* the internal
   ``shutdown()``, so the socket stayed open. And ``connect()`` had no
   try/finally at all — a login/select failure abandoned the connected
   socket entirely.
2. weixin: repeated poll failures through a proxy strand sockets in the
   aiohttp connector; the poll session was never recycled, so they
   accumulated for the life of the process.
"""

import asyncio
import imaplib
import os
import unittest
from unittest.mock import MagicMock, patch


def _make_email_adapter(address="hermes@test.com"):
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": address,
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True))


class TestCloseImap(unittest.TestCase):
    """_close_imap must guarantee socket teardown."""

    def test_logout_success_no_shutdown_needed(self):
        from plugins.platforms.email.adapter import _close_imap

        imap = MagicMock()
        _close_imap(imap)
        imap.logout.assert_called_once()
        imap.shutdown.assert_not_called()

    def test_logout_abort_falls_back_to_shutdown(self):
        from plugins.platforms.email.adapter import _close_imap

        imap = MagicMock()
        imap.logout.side_effect = imaplib.IMAP4.abort("socket error: EOF")
        _close_imap(imap)
        imap.shutdown.assert_called_once()

    def test_shutdown_failure_is_swallowed(self):
        from plugins.platforms.email.adapter import _close_imap

        imap = MagicMock()
        imap.logout.side_effect = imaplib.IMAP4.abort("broken")
        imap.shutdown.side_effect = OSError("already closed")
        _close_imap(imap)  # must not raise


class TestEmailConnectClosesSocket(unittest.TestCase):
    """connect() must close the IMAP socket on every path, incl. failures."""

    def test_login_failure_still_closes_socket(self):
        adapter = _make_email_adapter()
        mock_imap = MagicMock()
        mock_imap.login.side_effect = imaplib.IMAP4.error("AUTHENTICATIONFAILED")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            result = asyncio.run(adapter.connect())

        self.assertFalse(result)
        # The failed handle must have been torn down (logout attempted;
        # abort fallback covered by TestCloseImap).
        mock_imap.logout.assert_called_once()

    def test_select_failure_still_closes_socket(self):
        adapter = _make_email_adapter()
        mock_imap = MagicMock()
        mock_imap.select.side_effect = imaplib.IMAP4.abort("connection lost")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            result = asyncio.run(adapter.connect())

        self.assertFalse(result)
        mock_imap.logout.assert_called_once()


class TestFetchClosesSocketOnBrokenLogout(unittest.TestCase):
    def test_fetch_logout_abort_falls_back_to_shutdown(self):
        adapter = _make_email_adapter()
        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b""])
        mock_imap.logout.side_effect = imaplib.IMAP4.abort("EOF")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(results, [])
        mock_imap.shutdown.assert_called_once()
        # A teardown failure is not a fetch failure.
        self.assertFalse(adapter._last_fetch_failed)


class TestWeixinPollSessionRecycle(unittest.TestCase):
    """The weixin poll loop must recycle its session after a failure streak."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.weixin import WeixinAdapter

        return WeixinAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"account_id": "test-account"},
            )
        )

    def test_recycle_closes_old_and_installs_fresh_session(self):
        from gateway.platforms import weixin as weixin_mod

        adapter = self._make_adapter()
        adapter._running = True

        old_session = MagicMock()
        old_session.closed = False

        async def _close():
            old_session.close_called = True

        old_session.close = MagicMock(side_effect=lambda: _close())
        # close() must return an awaitable
        closed = {"v": False}

        async def _aclose():
            closed["v"] = True

        old_session.close = _aclose
        adapter._poll_session = old_session

        new_session = MagicMock()
        with patch.object(
            weixin_mod.aiohttp, "ClientSession", return_value=new_session
        ) as mk:
            asyncio.run(adapter._recycle_poll_session())

        mk.assert_called_once()
        self.assertIs(adapter._poll_session, new_session)
        self.assertTrue(closed["v"])

    def test_recycle_noop_when_not_running(self):
        adapter = self._make_adapter()
        adapter._running = False
        sentinel = MagicMock()
        adapter._poll_session = sentinel
        asyncio.run(adapter._recycle_poll_session())
        self.assertIs(adapter._poll_session, sentinel)

    def test_poll_loop_recycles_after_max_consecutive_failures(self):
        from gateway.platforms import weixin as weixin_mod

        adapter = self._make_adapter()
        adapter._running = True
        adapter._poll_session = MagicMock()

        calls = {"n": 0, "recycled": 0}

        async def _failing_get_updates(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > weixin_mod.MAX_CONSECUTIVE_FAILURES:
                adapter._running = False
                raise asyncio.CancelledError
            raise ConnectionError("Cannot connect to host via proxy")

        async def _fake_recycle():
            calls["recycled"] += 1

        async def _no_sleep(_secs):
            return None

        with patch.object(weixin_mod, "_get_updates", _failing_get_updates), \
                patch.object(weixin_mod, "_load_sync_buf", return_value=""), \
                patch.object(weixin_mod.asyncio, "sleep", _no_sleep), \
                patch.object(adapter, "_recycle_poll_session", _fake_recycle):
            asyncio.run(adapter._poll_loop())

        self.assertEqual(calls["recycled"], 1)


if __name__ == "__main__":
    unittest.main()
