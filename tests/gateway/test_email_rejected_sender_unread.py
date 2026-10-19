"""A sender rejected by the gate must not be marked read on the server (#141637).

``_mark_uid_consumed`` keeps a rejected UID out of the next ``UNSEEN`` search by
adding it to the in-memory ``_seen_uids`` set, which is enough on its own: the
``if uid in self._seen_uids: continue`` at the top of ``_fetch_new_messages``
skips it. The extra ``STORE +FLAGS.SILENT (\\Seen)`` therefore bought nothing and
destroyed the user's unread flag for every message from an unknown sender.

These tests pin both halves of that contract: no server-side write, and the UID
still remembered in memory.
"""

import os
import unittest
import uuid
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch


def _make_adapter(address="hermes@test.com"):
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": address,
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        adapter = EmailAdapter(PlatformConfig(enabled=True))
    return adapter


def _raw_email(sender="unknown@test.com", subject="Hello"):
    msg = MIMEText("Test body", "plain", "utf-8")
    msg["From"] = sender
    msg["Subject"] = subject
    msg["Message-ID"] = f"<{uuid.uuid4().hex[:8]}@test.com>"
    return msg.as_bytes()


class TestRejectedSenderKeepsUnread(unittest.TestCase):
    """The sender gate rejecting a message must not consume its UNSEEN flag."""

    def _run(self, aceptado):
        adapter = _make_adapter()
        llamadas = []

        def uid_handler(command, *args):
            llamadas.append((command, args))
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(args[0], _raw_email())])
            if command == "store":
                return ("OK", [])
            return ("NO", [])

        mock_imap = MagicMock()
        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            resultados = adapter._fetch_new_messages(lambda _c: aceptado)
        return adapter, resultados, llamadas

    def test_rejected_sender_is_not_marked_seen_on_the_server(self):
        adapter, resultados, llamadas = self._run(aceptado=False)

        self.assertEqual(resultados, [], "un remitente rechazado no se despacha")
        stores = [args for (cmd, args) in llamadas if cmd == "store"]
        self.assertEqual(
            stores, [],
            f"no debe escribirse el flag Seen en el servidor, pero hubo: {stores}",
        )

    def test_rejected_uid_is_remembered_in_memory(self):
        adapter, _resultados, _llamadas = self._run(aceptado=False)

        self.assertIn(
            b"1", adapter._seen_uids,
            "el UID rechazado debe recordarse para no volver a pedirlo",
        )

    def test_authorized_sender_still_dispatches(self):
        adapter, resultados, _llamadas = self._run(aceptado=True)

        self.assertEqual(len(resultados), 1)
        self.assertEqual(resultados[0]["sender_addr"], "unknown@test.com")

    def test_second_pass_does_not_refetch_the_headers(self):
        """A rejected UID is consumed once; the next poll must not ask again."""
        adapter = _make_adapter()
        peticiones = []

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                if "BODY.PEEK" in args[1]:
                    peticiones.append(args[0])
                return ("OK", [(args[0], _raw_email())])
            if command == "store":
                return ("OK", [])
            return ("NO", [])

        mock_imap = MagicMock()
        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            adapter._fetch_new_messages(lambda _c: False)
            adapter._fetch_new_messages(lambda _c: False)

        self.assertEqual(
            peticiones.count(b"1"), 1,
            f"el sobre del UID rechazado debe pedirse una sola vez, y se pidió "
            f"{peticiones.count(b'1')} veces",
        )


if __name__ == "__main__":
    unittest.main()
