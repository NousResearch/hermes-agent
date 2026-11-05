"""Email adapter INBOX → Working → Done folder lifecycle (working_folder / done_folder)."""

import imaplib
import os
import unittest
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch


class TestFolderLifecycle(unittest.TestCase):
    """Tests for the INBOX → Working → Done two-stage folder lifecycle."""

    # ------------------------------------------------------------------
    # Helper: build a minimal EmailAdapter with controlled folder settings
    # ------------------------------------------------------------------

    def _make_adapter(self, extra=None, env=None):
        """Build an EmailAdapter.

        ``extra`` maps to ``config.yaml`` ``platforms.email`` (working_folder /
        done_folder live here now). ``env`` adds/overrides environment variables
        (e.g. EMAIL_ALLOWED_USERS, which the dispatch path still reads from env).
        """
        from gateway.config import PlatformConfig
        base_env = {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_IMAP_PORT": "993",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SMTP_PORT": "587",
            "EMAIL_POLL_INTERVAL": "15",
        }
        if env:
            base_env.update(env)
        with patch.dict(os.environ, base_env, clear=True):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True, extra=extra or {}))
        return adapter

    def _make_raw_email(self, sender="user@test.com", subject="Test", message_id="<msg@test.com>"):
        """Build a minimal RFC822 message as bytes."""
        msg = MIMEText("Hello Hermes", "plain", "utf-8")
        msg["From"] = sender
        msg["Subject"] = subject
        msg["Message-ID"] = message_id
        return msg.as_bytes()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    def test_imap_move_uses_uid_move_when_available(self):
        """_imap_move should call UID MOVE and NOT fall back to COPY when MOVE works."""
        from plugins.platforms.email.adapter import EmailAdapter

        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b"1"])

        result = EmailAdapter._imap_move(mock_imap, b"42", "Hermes_Working")

        self.assertTrue(result)
        # First uid call must be MOVE
        first_call = mock_imap.uid.call_args_list[0]
        self.assertEqual(first_call.args[0], "MOVE")
        # COPY must NOT have been called
        copy_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "COPY"]
        self.assertEqual(len(copy_calls), 0)

    @staticmethod
    def _uidplus_imap(uid_handler):
        """A mock IMAP handle that advertises UIDPLUS (needed for the fallback)."""
        mock_imap = MagicMock()
        mock_imap.capabilities = ("IMAP4REV1", "UIDPLUS")
        mock_imap.uid.side_effect = uid_handler
        return mock_imap

    def test_imap_move_falls_back_to_copy_expunge(self):
        """_imap_move should fall back to COPY+STORE+EXPUNGE when MOVE returns NO."""
        from plugins.platforms.email.adapter import EmailAdapter

        def uid_handler(command, *args):
            if command == "MOVE":
                return ("NO", [b"MOVE not supported"])
            if command == "COPY":
                return ("OK", [b"1"])
            return ("OK", [b""])

        mock_imap = self._uidplus_imap(uid_handler)

        result = EmailAdapter._imap_move(mock_imap, b"42", "Hermes_Done")

        self.assertTrue(result)
        commands = [c.args[0] for c in mock_imap.uid.call_args_list]
        self.assertIn("COPY", commands)
        self.assertIn("STORE", commands)
        self.assertIn("EXPUNGE", commands)
        # UID EXPUNGE only — a bare EXPUNGE would hit the whole folder
        mock_imap.expunge.assert_not_called()

    def test_imap_move_falls_back_when_move_raises(self):
        """_imap_move should fall back to COPY+EXPUNGE when MOVE raises an exception."""
        from plugins.platforms.email.adapter import EmailAdapter

        def uid_handler(command, *args):
            if command == "MOVE":
                raise imaplib.IMAP4.error("command unknown: UID MOVE")
            if command == "COPY":
                return ("OK", [b"1"])
            return ("OK", [b""])

        mock_imap = self._uidplus_imap(uid_handler)

        result = EmailAdapter._imap_move(mock_imap, b"7", "Hermes_Done")

        self.assertTrue(result)
        commands = [c.args[0] for c in mock_imap.uid.call_args_list]
        self.assertIn("COPY", commands)

    def test_imap_move_refuses_fallback_without_uidplus(self):
        """Without MOVE and without UIDPLUS, nothing is copied, deleted or expunged.

        A global EXPUNGE would permanently remove every \\Deleted message in
        the folder, including mail another client flagged. Bail out instead.
        """
        from plugins.platforms.email.adapter import EmailAdapter

        def uid_handler(command, *args):
            if command == "MOVE":
                return ("NO", [b"MOVE not supported"])
            return ("OK", [b"1"])

        mock_imap = MagicMock()
        mock_imap.capabilities = ("IMAP4REV1",)   # no UIDPLUS
        mock_imap.uid.side_effect = uid_handler

        result = EmailAdapter._imap_move(mock_imap, b"42", "Hermes_Done")

        self.assertFalse(result)
        commands = [c.args[0] for c in mock_imap.uid.call_args_list]
        self.assertNotIn("COPY", commands)
        self.assertNotIn("STORE", commands)
        mock_imap.expunge.assert_not_called()

    def test_imap_move_survives_uid_expunge_failure(self):
        """A refused UID EXPUNGE still counts as moved — and never escalates to EXPUNGE."""
        from plugins.platforms.email.adapter import EmailAdapter

        def uid_handler(command, *args):
            if command == "MOVE":
                return ("NO", [b"MOVE not supported"])
            if command == "COPY":
                return ("OK", [b"1"])
            if command == "EXPUNGE":
                raise imaplib.IMAP4.error("UID EXPUNGE refused")
            return ("OK", [b""])

        mock_imap = self._uidplus_imap(uid_handler)

        result = EmailAdapter._imap_move(mock_imap, b"42", "Hermes_Done")

        # The copy reached the destination; only the source cleanup failed.
        self.assertTrue(result)
        mock_imap.expunge.assert_not_called()

    def test_search_message_id_quotes_the_literal(self):
        """The attacker-controlled Message-ID is passed as ONE quoted IMAP literal."""
        from plugins.platforms.email.adapter import EmailAdapter

        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b"5"])

        uids = EmailAdapter._search_message_id(mock_imap, '<a" OR ALL@evil>')

        self.assertEqual(uids, [b"5"])
        # Quoted, with the embedded quote escaped — not injected as extra keys
        self.assertEqual(
            mock_imap.uid.call_args.args,
            ("SEARCH", None, "HEADER", "Message-ID", '"<a\\" OR ALL@evil>"'),
        )

    def test_search_message_id_refuses_control_characters(self):
        """A Message-ID with CR/LF or non-ASCII is refused, not escaped."""
        from plugins.platforms.email.adapter import EmailAdapter

        for bad in ("<a\r\nA001 SELECT INBOX@x>", "<a\x00@x>", "<ä@x>"):
            with self.subTest(message_id=bad):
                mock_imap = MagicMock()
                self.assertEqual(EmailAdapter._search_message_id(mock_imap, bad), [])
                mock_imap.uid.assert_not_called()

    def test_ensure_folder_swallows_errors(self):
        """_ensure_folder must not propagate exceptions from imap.create()."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.create.side_effect = imaplib.IMAP4.error("unexpected server error")

        # Should not raise
        adapter._ensure_folder(mock_imap, "Hermes_Working")
        mock_imap.create.assert_called_once_with("Hermes_Working")

    def test_ensure_folder_warns_once_per_folder(self):
        """A CREATE failure warns the operator once, then drops to debug."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.create.side_effect = imaplib.IMAP4.error("permission denied")

        with self.assertLogs("plugins.platforms.email.adapter", level="WARNING") as logs:
            adapter._ensure_folder(mock_imap, "Hermes_Working")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("Hermes_Working", logs.output[0])

        # Second attempt (e.g. next reconnect) must not repeat the warning
        with self.assertLogs("plugins.platforms.email.adapter", level="DEBUG") as logs:
            adapter._ensure_folder(mock_imap, "Hermes_Working")
        self.assertFalse([line for line in logs.output if line.startswith("WARNING")])

    def test_ensure_folder_no_warning_when_folder_exists(self):
        """A NO response (already exists) is the normal path and stays quiet."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.create.return_value = ("NO", [b"Mailbox already exists"])

        with self.assertLogs("plugins.platforms.email.adapter", level="DEBUG") as logs:
            adapter._ensure_folder(mock_imap, "Hermes_Done")
        self.assertFalse([line for line in logs.output if line.startswith("WARNING")])

    def test_ensure_folder_skips_empty_name(self):
        """_ensure_folder must do nothing when name is empty."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        adapter._ensure_folder(mock_imap, "")
        mock_imap.create.assert_not_called()

    # ------------------------------------------------------------------
    # Finalize connection reuse
    # ------------------------------------------------------------------

    def test_finalize_reuses_one_connection(self):
        """Finalizing a batch must not reconnect per message."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        mock_imap = MagicMock()
        mock_imap.capabilities = ("IMAP4REV1", "UIDPLUS")
        mock_imap.noop.return_value = ("OK", [b"NOOP completed"])

        def uid_handler(command, *args):
            if command == "SEARCH":
                return ("OK", [b"3"])
            return ("OK", [b"1"])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap) as ctor:
            for i in range(3):
                adapter._finalize_message(f"<m{i}@test.com>", "Hermes_Working")

        self.assertEqual(ctor.call_count, 1, "expected one IMAP connection for the batch")
        # Subsequent calls probe the cached handle instead of reconnecting
        self.assertEqual(mock_imap.noop.call_count, 2)

    def test_finalize_reconnects_after_a_dropped_connection(self):
        """A connection the server dropped mid-batch is reopened, not lost."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        dead = MagicMock()
        dead.capabilities = ("IMAP4REV1", "UIDPLUS")
        dead.noop.return_value = ("OK", [b"NOOP completed"])
        dead.select.side_effect = imaplib.IMAP4.abort("socket error: EOF")

        live = MagicMock()
        live.capabilities = ("IMAP4REV1", "UIDPLUS")
        live.noop.return_value = ("OK", [b"NOOP completed"])
        live.uid.side_effect = lambda command, *a: (
            ("OK", [b"3"]) if command == "SEARCH" else ("OK", [b"1"])
        )

        with patch("imaplib.IMAP4_SSL", side_effect=[dead, live]) as ctor:
            adapter._finalize_message("<m@test.com>", "Hermes_Working")

        self.assertEqual(ctor.call_count, 2)
        live.select.assert_called_once_with("Hermes_Working")

    def test_disconnect_closes_finalize_connection(self):
        """disconnect() releases the cached handle so no socket is left behind."""
        import asyncio

        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        mock_imap = MagicMock()
        adapter._finalize_conn = mock_imap

        asyncio.run(adapter.disconnect())

        self.assertIsNone(adapter._finalize_conn)
        mock_imap.logout.assert_called_once()

    # ------------------------------------------------------------------
    # _fetch_new_messages: Working-folder move
    # ------------------------------------------------------------------

    def test_fetch_moves_to_working_when_configured(self):
        """_fetch_new_messages should MOVE mail to Hermes_Working and set source_folder."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        raw = self._make_raw_email()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"10"])
            if command == "fetch":
                return ("OK", [(b"10", raw)])
            if command == "MOVE":
                return ("OK", [b"1"])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(lambda _c: True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "Hermes_Working")

        # Verify a MOVE to Hermes_Working was issued
        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "MOVE"]
        self.assertTrue(move_calls, "Expected UID MOVE to be called")
        self.assertEqual(move_calls[0].args[2], "Hermes_Working")

    def test_fetch_skips_move_when_working_empty(self):
        """_fetch_new_messages should NOT issue any MOVE when working_folder is empty."""
        adapter = self._make_adapter({
            "working_folder": "",
            "done_folder": "Hermes_Done",
        })

        raw = self._make_raw_email()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"11"])
            if command == "fetch":
                return ("OK", [(b"11", raw)])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(lambda _c: True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "INBOX")

        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] in ("MOVE", "COPY")]
        self.assertEqual(len(move_calls), 0, "No MOVE/COPY should be issued when working_folder is empty")

    def test_fetch_source_folder_inbox_when_move_fails(self):
        """source_folder should remain 'INBOX' when the MOVE to Working fails."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        raw = self._make_raw_email()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"12"])
            if command == "fetch":
                return ("OK", [(b"12", raw)])
            if command == "MOVE":
                return ("NO", [b"failed"])
            if command == "COPY":
                return ("NO", [b"failed"])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(lambda _c: True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "INBOX")

    def test_fetch_skips_move_when_done_empty(self):
        """done_folder='' disables ALL moves, even when working_folder is set."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "",
        })

        raw = self._make_raw_email()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"13"])
            if command == "fetch":
                return ("OK", [(b"13", raw)])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(lambda _c: True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "INBOX")
        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] in ("MOVE", "COPY")]
        self.assertEqual(len(move_calls), 0, "No move should happen when done_folder is empty")

    def test_fetch_skips_move_when_no_message_id(self):
        """Mail with no Message-ID must NOT be moved to Working (it could not be
        re-located there for the final MOVE → Done) — it stays in INBOX."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        raw = self._make_raw_email(message_id="")
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"14"])
            if command == "fetch":
                return ("OK", [(b"14", raw)])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(lambda _c: True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "INBOX")
        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] in ("MOVE", "COPY")]
        self.assertEqual(len(move_calls), 0, "No Working move without a Message-ID")

    def test_fetch_rejected_uid_goes_to_done_without_body_fetch(self):
        """A UID the pre-fetch gate rejects is moved INBOX → Done (UID still
        valid there), never RFC822-fetched and never returned for dispatch;
        with done_folder='' it only gets \\Seen, as upstream does."""
        for done_folder, expect_move in (("Hermes_Done", True), ("", False)):
            with self.subTest(done_folder=done_folder):
                adapter = self._make_adapter({
                    "working_folder": "Hermes_Working",
                    "done_folder": done_folder,
                })

                raw = self._make_raw_email(sender="stranger@test.com")
                mock_imap = MagicMock()

                def uid_handler(command, *args):
                    if command == "search":
                        return ("OK", [b"15"])
                    if command == "fetch":
                        return ("OK", [(b"15", raw)])
                    if command == "MOVE":
                        return ("OK", [b"1"])
                    return ("OK", [b""])

                mock_imap.uid.side_effect = uid_handler

                with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                    results = adapter._fetch_new_messages(lambda _c: False)

                self.assertEqual(results, [])
                fetch_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "fetch"]
                self.assertEqual(len(fetch_calls), 1, "Only the bounded header fetch, no RFC822")
                self.assertNotIn("RFC822", fetch_calls[0].args[2])
                store_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "store"]
                self.assertTrue(store_calls, "Rejected UID must be marked \\Seen")
                move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] in ("MOVE", "COPY")]
                if expect_move:
                    self.assertEqual(len(move_calls), 1)
                    self.assertEqual(move_calls[0].args[1], b"15")
                    self.assertEqual(move_calls[0].args[2], "Hermes_Done")
                else:
                    self.assertEqual(move_calls, [], "No move when done_folder is empty")

    # ------------------------------------------------------------------
    # _finalize_message
    # ------------------------------------------------------------------

    def test_finalize_noop_when_done_empty(self):
        """_finalize_message must do nothing when done_folder is empty."""
        adapter = self._make_adapter({"done_folder": ""})

        with patch("imaplib.IMAP4_SSL") as mock_cls:
            adapter._finalize_message("<msg@test.com>", "Hermes_Working")
            mock_cls.assert_not_called()

    def test_finalize_noop_when_already_in_done(self):
        """_finalize_message must do nothing when source_folder == _done_folder."""
        adapter = self._make_adapter({"done_folder": "Hermes_Done"})

        with patch("imaplib.IMAP4_SSL") as mock_cls:
            adapter._finalize_message("<msg@test.com>", "Hermes_Done")
            mock_cls.assert_not_called()

    def test_finalize_moves_to_done(self):
        """_finalize_message should open IMAP, SEARCH by Message-ID, and MOVE to Done."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "SEARCH":
                return ("OK", [b"42"])
            if command == "MOVE":
                return ("OK", [b"1"])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            adapter._finalize_message("<msg@test.com>", "Hermes_Working")

        mock_imap.select.assert_called_once_with("Hermes_Working")
        search_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "SEARCH"]
        self.assertTrue(search_calls)
        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "MOVE"]
        self.assertTrue(move_calls)
        self.assertEqual(move_calls[0].args[2], "Hermes_Done")

    # ------------------------------------------------------------------
    # Full dispatch flow
    # ------------------------------------------------------------------

    def test_finalize_still_runs_when_handle_message_raises(self):
        """_finalize_message must be called even if handle_message raises."""
        import asyncio
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        finalize_called = []

        def fake_finalize(message_id, source_folder):
            finalize_called.append((message_id, source_folder))

        adapter._finalize_message = fake_finalize

        async def raising_handler(event):
            raise RuntimeError("agent crashed")

        adapter.handle_message = raising_handler

        msg_data = {
            "uid": b"50",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Crash test",
            "message_id": "<crash@test.com>",
            "in_reply_to": "",
            "body": "Will crash",
            "attachments": [],
            "date": "",
            "sender_authenticated": True,
            "auth_reason": "dmarc=pass",
            "source_folder": "Hermes_Working",
        }

        # user@test.com must clear the allowlist gate AND the From:
        # authentication check to reach handle_message (upstream default-denies
        # when EMAIL_ALLOWED_USERS is unset and drops unauthenticated From:),
        # so the finalize-on-raise invariant is actually exercised. Env is read
        # at dispatch time, so patch around the call (not just at construction).
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            with self.assertRaises(RuntimeError):
                asyncio.run(adapter._dispatch_message(msg_data))

        self.assertEqual(len(finalize_called), 1)
        self.assertEqual(finalize_called[0][0], "<crash@test.com>")
        self.assertEqual(finalize_called[0][1], "Hermes_Working")

    def test_finalize_moves_to_done_after_dispatch(self):
        """Full dispatch flow: after handle_message returns, mail is moved to Done."""
        import asyncio
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        finalize_calls = []

        def fake_finalize(message_id, source_folder):
            finalize_calls.append((message_id, source_folder))

        adapter._finalize_message = fake_finalize

        async def noop_handler(event):
            pass

        adapter.handle_message = noop_handler

        msg_data = {
            "uid": b"60",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Done test",
            "message_id": "<done@test.com>",
            "in_reply_to": "",
            "body": "Process me",
            "attachments": [],
            "date": "",
            "source_folder": "Hermes_Working",
        }

        # Authorize the sender so it reaches handle_message (upstream default-
        # denies an unset allowlist); env is read at dispatch time.
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._dispatch_message(msg_data))

        self.assertEqual(len(finalize_calls), 1)
        self.assertEqual(finalize_calls[0][0], "<done@test.com>")
        self.assertEqual(finalize_calls[0][1], "Hermes_Working")

    def test_dispatch_skips_finalize_when_no_source_folder(self):
        """Mail dispatched without source_folder never went through the
        lifecycle fetch (nothing was moved), so _dispatch_message must not
        finalize it — a bare dispatch must not open an IMAP connection."""
        import asyncio
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        finalize_calls = []

        def fake_finalize(message_id, source_folder):
            finalize_calls.append(source_folder)

        adapter._finalize_message = fake_finalize

        async def noop_handler(event):
            pass

        adapter.handle_message = noop_handler

        # Omit source_folder — a message dict that did not come from _fetch_new_messages
        msg_data = {
            "uid": b"70",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Fallback test",
            "message_id": "<fallback@test.com>",
            "in_reply_to": "",
            "body": "No source_folder key",
            "attachments": [],
            "date": "",
        }

        # Authorize the sender so it reaches handle_message (upstream default-
        # denies an unset allowlist); env is read at dispatch time.
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._dispatch_message(msg_data))

        self.assertEqual(finalize_calls, [])

    # ------------------------------------------------------------------
    # Early-drop paths must still finalize (no mail stranded in Working)
    # ------------------------------------------------------------------

    def _dropping_adapter(self):
        """Adapter whose handle_message records calls and finalize records folders."""
        adapter = self._make_adapter(
            {"working_folder": "Hermes_Working", "done_folder": "Hermes_Done"},
        )
        adapter._handled = []
        adapter._finalized = []

        async def recording_handler(event):
            adapter._handled.append(event)

        def recording_finalize(message_id, source_folder):
            adapter._finalized.append((message_id, source_folder))

        adapter.handle_message = recording_handler
        adapter._finalize_message = recording_finalize
        return adapter

    def _drop_msg(self, sender):
        return {
            "uid": b"80",
            "sender_addr": sender,
            "sender_name": "Someone",
            "subject": "Hi",
            "message_id": "<drop@test.com>",
            "in_reply_to": "",
            "body": "body",
            "attachments": [],
            "date": "",
            "source_folder": "Hermes_Working",
        }

    def test_dispatch_finalizes_dropped_self_message(self):
        """A self-message is dropped (handler not called) but still finalized → Done."""
        import asyncio
        adapter = self._dropping_adapter()

        asyncio.run(adapter._dispatch_message(self._drop_msg("hermes@test.com")))

        self.assertEqual(adapter._handled, [], "self-message must not be handled")
        self.assertEqual(adapter._finalized, [("<drop@test.com>", "Hermes_Working")],
                         "dropped self-message must still be finalized out of Working")

    def test_dispatch_finalizes_non_allowlisted_sender(self):
        """A non-allowlisted sender is dropped but still finalized → Done."""
        import asyncio
        adapter = self._dropping_adapter()

        # EMAIL_ALLOWED_USERS is read from env at dispatch time, so patch it
        # around the dispatch call (not just at construction).
        with patch.dict(os.environ, {"EMAIL_ALLOWED_USERS": "boss@test.com"}):
            asyncio.run(adapter._dispatch_message(self._drop_msg("stranger@test.com")))

        self.assertEqual(adapter._handled, [], "non-allowlisted sender must not be handled")
        self.assertEqual(adapter._finalized, [("<drop@test.com>", "Hermes_Working")],
                         "dropped non-allowlisted mail must still be finalized out of Working")

    def test_dispatch_finalizes_unauthenticated_from_drop(self):
        """An allowlisted sender dropped for an unauthenticated From: is still finalized."""
        import asyncio
        adapter = self._dropping_adapter()
        msg = dict(self._drop_msg("boss@test.com"), sender_authenticated=False)

        with patch.dict(os.environ, {"EMAIL_ALLOWED_USERS": "boss@test.com"}):
            asyncio.run(adapter._dispatch_message(msg))

        self.assertEqual(adapter._handled, [])
        self.assertEqual(adapter._finalized, [("<drop@test.com>", "Hermes_Working")])

    def test_dispatch_finalizes_unknown_sender_handed_to_gateway_once(self):
        """A stranger under unauthorized_dm_behavior=pair reaches the gateway and is finalized exactly once."""
        import asyncio
        adapter = self._dropping_adapter()
        adapter.config.extra = {"unauthorized_dm_behavior": "pair"}
        msg = dict(self._drop_msg("stranger@test.com"), sender_authenticated=True)

        with patch.dict(os.environ, {"EMAIL_ALLOWED_USERS": "boss@test.com"}):
            asyncio.run(adapter._dispatch_message(msg))

        self.assertEqual(len(adapter._handled), 1)
        self.assertEqual(adapter._finalized, [("<drop@test.com>", "Hermes_Working")])


if __name__ == "__main__":
    unittest.main()
