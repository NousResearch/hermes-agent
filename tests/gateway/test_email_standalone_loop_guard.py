"""The out-of-process (cron/startup) sender must mint the Message-ID the loop guard recognises.

``_standalone_send`` delivers cron/startup notices over SMTP without going through the adapter, so
the adapter's structural loop guard is the only thing that can recognise such a notice when it comes
back. The guard (``_sender_accepted``) drops any mail whose ``Message-ID`` has the shape the adapter
mints for its own outbound mail, ``<hermes-<12 hex>@domain>``.

That guard is load-bearing on a gated config (``EMAIL_RECIPIENTS`` set): the identity guard there is
deliberately skipped, because under plus addressing the operator's own mail and Hermes' mail share one
``From:`` identity. So when ``EMAIL_HOME_ADDRESS`` points at the channel mailbox, a notice that does
not carry a minted id is read straight back in as a prompt — the channel answers its own startup mail.

Hermetic by construction: every identity/authorization env var the adapter reads is pinned or removed
for the duration of each test, so an ambient ``EMAIL_*`` in the developer's shell (or a CI job that
exports the live identity) cannot decide the outcome.
"""

import asyncio
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# The gated live shape: EMAIL_ADDRESS is the LOGIN (Fastmail requires the account address to
# authenticate — an alias cannot), EMAIL_FROM is the visible From:, and the channel mailbox IS the
# From: address, so a notice Hermes sends lands in the mailbox the channel polls.
LOGIN_ADDRESS = "nca@fastmail.com"
FROM_ADDRESS = "hermes.nca@fastmail.com"
CHANNEL_MAILBOX = "hermes.nca@fastmail.com"

# Every env var that decides identity, the recipients gate, or the sender authorization gates.
_IDENTITY_ENV = (
    "EMAIL_ADDRESS", "EMAIL_FROM", "EMAIL_RECIPIENTS", "EMAIL_MAILBOXES",
    "EMAIL_ALLOW_ALL_USERS", "EMAIL_ALLOWED_USERS", "EMAIL_TRUST_FROM_HEADER",
    "EMAIL_TRUST_INTERNAL_SENDER", "EMAIL_AUTHSERV_ID", "EMAIL_HOME_ADDRESS",
    "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
)

_GATED_ENV = {
    "EMAIL_ADDRESS": LOGIN_ADDRESS,
    "EMAIL_PASSWORD": "secret",
    "EMAIL_SMTP_HOST": "smtp.fastmail.com",
    "EMAIL_SMTP_PORT": "587",
    "EMAIL_FROM": FROM_ADDRESS,
    "EMAIL_RECIPIENTS": CHANNEL_MAILBOX,
    # The tail gates (allowlist / From: authentication) are not what this card is about; allow-all
    # clears them so the loop guard is the only thing left that can drop the mail.
    "EMAIL_ALLOW_ALL_USERS": "true",
}


@contextmanager
def _gated_env():
    """The gated live shape with every identity var pinned — nothing inherited from the shell."""
    env = {k: v for k, v in os.environ.items() if k not in _IDENTITY_ENV}
    env.update(_GATED_ENV)
    with patch.dict(os.environ, env, clear=True):
        yield


def _sent_notice():
    """Drive the real ``_standalone_send`` over a mocked SMTP; return the message it built."""
    from plugins.platforms.email.adapter import _standalone_send

    pconfig = SimpleNamespace(token=None, api_key=None, extra={
        "address": LOGIN_ADDRESS, "from_address": FROM_ADDRESS, "smtp_host": "smtp.fastmail.com",
    })
    with patch("smtplib.SMTP") as mock_smtp:
        server = MagicMock()
        mock_smtp.return_value = server
        result = asyncio.run(_standalone_send(pconfig, CHANNEL_MAILBOX, "Hermes Agent started"))
    assert result.get("success"), result
    return server.send_message.call_args[0][0]


def _arriving_mail(message_id):
    """The fetched-mail view of that notice landing back in the channel mailbox."""
    return {
        "uid": b"1", "sender_addr": FROM_ADDRESS, "sender_name": "Hermes",
        "subject": "Hermes Agent", "message_id": message_id, "in_reply_to": "",
        "thread_key": "", "recipients": [CHANNEL_MAILBOX], "body": "Hermes Agent started",
        "attachments": [], "date": "", "sender_authenticated": True, "auth_reason": "",
    }


class TestStandaloneNoticeLoopGuard(unittest.TestCase):
    """A cron/startup notice must be recognisable as ours when it returns to the channel mailbox."""

    def test_standalone_notice_mints_a_self_message_id(self):
        """The out-of-process builder mints the same Message-ID shape the guard matches."""
        from plugins.platforms.email.adapter import _SELF_MESSAGE_ID_RE

        with _gated_env():
            msg = _sent_notice()

        self.assertEqual(msg["From"], FROM_ADDRESS)
        self.assertEqual(msg["To"], CHANNEL_MAILBOX)
        msg_id = str(msg["Message-ID"] or "").strip()
        self.assertTrue(msg_id, "the standalone notice carries no Message-ID at all")
        self.assertRegex(msg_id, _SELF_MESSAGE_ID_RE)

    def test_gated_config_drops_the_notice_when_it_returns(self):
        """On the gated shape the notice is dropped — and the control proves the id is the cause."""
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        with _gated_env():
            adapter = EmailAdapter(PlatformConfig(enabled=True))
            # The gated shape, i.e. the one where the identity guard is skipped by design.
            self.assertEqual(adapter._recipients, {CHANNEL_MAILBOX})
            self.assertNotEqual(adapter._from_address.lower(), adapter._address.lower())

            notice = _sent_notice()
            dropped = adapter._sender_accepted(FROM_ADDRESS, _arriving_mail(notice["Message-ID"]))
            # Control: the same mail WITHOUT the minted id. If this is also dropped the test proves
            # nothing about the id — some other gate would be doing the work.
            control = adapter._sender_accepted(FROM_ADDRESS, _arriving_mail(""))

        self.assertFalse(dropped, "our own cron/startup notice came back in as a prompt")
        self.assertTrue(control, "control failed: the recipients gate alone already drops this mail")


if __name__ == "__main__":
    unittest.main()
