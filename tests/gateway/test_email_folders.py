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

    def test_fetch_rejected_uid_goes_to_rejected_without_body_fetch(self):
        """A UID the pre-fetch gate rejects is moved INBOX → Rejected (UID still
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
                    self.assertEqual(move_calls[0].args[2], "Hermes_Rejected")
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

        def fake_finalize(message_id, source_folder, target_folder=None):
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
        """A handle_message that does not accept the event finalizes the mail at once."""
        import asyncio
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        finalize_calls = []

        def fake_finalize(message_id, source_folder, target_folder=None):
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

        def fake_finalize(message_id, source_folder, target_folder=None):
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

        def recording_finalize(message_id, source_folder, target_folder=None):
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


class TestDeferredFinalize(unittest.TestCase):
    """The Done move waits for on_processing_complete when the gateway accepted the event."""

    _make_adapter = TestFolderLifecycle._make_adapter

    def _lifecycle_adapter(self, extra=None):
        adapter = self._make_adapter(extra or {"working_folder": "Hermes_Working", "done_folder": "Hermes_Done"})
        adapter._finalized = []
        adapter._targets = []
        adapter._handled = []
        adapter._finalize_message = lambda mid, folder, target=None: (
            adapter._finalized.append((mid, folder)), adapter._targets.append(target))

        async def accepting_handler(event):
            adapter._handled.append(event)
            event._gateway_accepted = True

        adapter.handle_message = accepting_handler
        return adapter

    def _msg(self, message_id="<m1@test.com>", sender="user@test.com", folder="Hermes_Working"):
        return {
            "uid": b"90", "sender_addr": sender, "sender_name": "User", "subject": "Hi",
            "message_id": message_id, "in_reply_to": "", "body": "body", "attachments": [], "date": "",
            "sender_authenticated": True, "auth_reason": "dmarc=pass",
            "source_folder": folder,
        }

    def _dispatch(self, adapter, msg):
        import asyncio
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._dispatch_message(msg))

    def test_accepted_event_is_not_finalized_by_dispatch(self):
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        self.assertEqual(adapter._finalized, [])
        self.assertIn("<m1@test.com>", adapter._open_mails)

    def _complete(self, adapter, outcome):
        import asyncio
        asyncio.run(adapter.on_processing_complete(adapter._handled[0], outcome))

    def test_success_moves_to_done(self):
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        self._complete(adapter, ProcessingOutcome.SUCCESS)
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])
        self.assertEqual(len(adapter._open_mails), 0)

    def test_failure_moves_to_done(self):
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        self._complete(adapter, ProcessingOutcome.FAILURE)
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])

    def test_cancelled_leaves_mail_in_working(self):
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        self._complete(adapter, ProcessingOutcome.CANCELLED)
        self.assertEqual(adapter._finalized, [])
        self.assertEqual(len(adapter._open_mails), 0)

    def test_second_completion_does_not_move_again(self):
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        self._complete(adapter, ProcessingOutcome.SUCCESS)
        self._complete(adapter, ProcessingOutcome.SUCCESS)
        self.assertEqual(len(adapter._finalized), 1)

    def test_working_folder_empty_moves_from_inbox_after_completion(self):
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter({"working_folder": "", "done_folder": "Hermes_Done"})
        self._dispatch(adapter, self._msg(folder="INBOX"))
        self.assertEqual(adapter._finalized, [])
        self._complete(adapter, ProcessingOutcome.SUCCESS)
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "INBOX")])

    def test_completion_swallows_finalize_errors(self):
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())

        def boom(mid, folder, target=None):
            raise RuntimeError("imap down")

        adapter._finalize_message = boom
        self._complete(adapter, ProcessingOutcome.SUCCESS)

    def test_completion_calls_base_hook(self):
        from gateway.platforms.base import BasePlatformAdapter
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        with patch.object(BasePlatformAdapter, "on_processing_complete", side_effect=RuntimeError("x")) as base:
            self._complete(adapter, ProcessingOutcome.SUCCESS)
        base.assert_called_once()
        self.assertEqual(len(adapter._finalized), 1, "base hook failure must not block the move")

    def test_completion_before_handle_message_returns_finalizes_once(self):
        import asyncio
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()

        async def fast_handler(event):
            event._gateway_accepted = True
            await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

        adapter.handle_message = fast_handler
        self._dispatch(adapter, self._msg())
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])

    def test_handler_exception_finalizes_immediately(self):
        adapter = self._lifecycle_adapter()

        async def raising(event):
            event._gateway_accepted = True
            raise RuntimeError("boom")

        adapter.handle_message = raising
        with self.assertRaises(RuntimeError):
            self._dispatch(adapter, self._msg())
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])
        self.assertEqual(len(adapter._open_mails), 0)

    def test_rejected_sender_finalizes_immediately(self):
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg(sender="hermes@test.com"))
        self.assertEqual(adapter._handled, [])
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])

    def test_done_folder_empty_never_tracks_or_moves(self):
        adapter = self._lifecycle_adapter({"working_folder": "Hermes_Working", "done_folder": ""})
        self._dispatch(adapter, self._msg())
        self.assertEqual(len(adapter._open_mails), 0)

    def test_open_mail_map_is_bounded(self):
        import plugins.platforms.email.adapter as mod
        adapter = self._lifecycle_adapter()
        with patch.object(mod, "_OPEN_MAILS_MAX", 2):
            for i in range(3):
                adapter._track_open_mail(f"<{i}@t>", "Hermes_Working", MagicMock())
        self.assertEqual(list(adapter._open_mails), ["<1@t>", "<2@t>"])

    def test_absorbed_ids_finalize_with_the_turn_without_the_reaper(self):
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg("<a@test.com>"))
        self._dispatch(adapter, self._msg("<b@test.com>"))
        self._dispatch(adapter, self._msg("<c@test.com>"))
        head, second, third = adapter._handled
        second.absorb_reply_expected(third)
        head.absorb_reply_expected(second)
        import asyncio
        asyncio.run(adapter.on_processing_complete(head, ProcessingOutcome.SUCCESS))
        self.assertEqual(sorted(m for m, _ in adapter._finalized), ["<a@test.com>", "<b@test.com>", "<c@test.com>"])
        self.assertEqual(len(adapter._open_mails), 0)

    def test_overwritten_head_id_is_still_finalized(self):
        """The text debounce rewrites the head's message_id; the original must still move."""
        import asyncio
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg("<a@test.com>"))
        self._dispatch(adapter, self._msg("<b@test.com>"))
        head, second = adapter._handled
        head.absorb_reply_expected(second)
        head.message_id = "<b@test.com>"
        asyncio.run(adapter.on_processing_complete(head, ProcessingOutcome.SUCCESS))
        self.assertEqual(sorted(m for m, _ in adapter._finalized), ["<a@test.com>", "<b@test.com>"])

    def test_cancelled_drops_own_and_absorbed_but_not_other_mails(self):
        import asyncio
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._lifecycle_adapter()
        for mid in ("<a@test.com>", "<b@test.com>", "<c@test.com>"):
            self._dispatch(adapter, self._msg(mid))
        head, second, _third = adapter._handled
        head.absorb_reply_expected(second)
        asyncio.run(adapter.on_processing_complete(head, ProcessingOutcome.CANCELLED))
        self.assertEqual(adapter._finalized, [])
        self.assertEqual(list(adapter._open_mails), ["<c@test.com>"])

    def test_on_processing_start_is_overridden(self):
        from gateway.platforms.base import BasePlatformAdapter
        from plugins.platforms.email.adapter import EmailAdapter
        self.assertIsNot(EmailAdapter.on_processing_start, BasePlatformAdapter.on_processing_start)

    def _reap(self, adapter):
        import asyncio
        asyncio.run(adapter._reap_orphaned_mails())

    def _age(self, adapter):
        import plugins.platforms.email.adapter as mod
        return patch.object(mod, "_ORPHAN_GRACE_SECONDS", 0.0)

    def test_reaper_respects_grace_period(self):
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        self._reap(adapter)
        self.assertEqual(adapter._finalized, [])

    def test_reaper_skips_busy_sessions(self):
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        key = adapter._event_session_key(adapter._handled[0])
        with self._age(adapter):
            for holder in (adapter._active_sessions, adapter._pending_messages, adapter._text_debounce_store()):
                holder[key] = object()
                self._reap(adapter)
                del holder[key]
            self.assertEqual(adapter._finalized, [])
            runner = MagicMock()
            runner._is_session_running.return_value = True
            adapter.gateway_runner = runner
            self._reap(adapter)
            self.assertEqual(adapter._finalized, [])
            runner._is_session_running.return_value = False
            runner._peek_session_state.return_value.conversation.queued_events = [object()]
            self._reap(adapter)
            self.assertEqual(adapter._finalized, [])

    def test_reaper_treats_idle_check_errors_as_not_idle(self):
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        runner = MagicMock()
        runner._is_session_running.side_effect = RuntimeError("x")
        adapter.gateway_runner = runner
        with self._age(adapter):
            self._reap(adapter)
        self.assertEqual(adapter._finalized, [])
        self.assertEqual(len(adapter._open_mails), 1)

    def test_reaper_finalizes_idle_orphan(self):
        adapter = self._lifecycle_adapter()
        self._dispatch(adapter, self._msg())
        runner = MagicMock()
        runner._is_session_running.return_value = False
        runner._peek_session_state.return_value = None
        adapter.gateway_runner = runner
        with self._age(adapter):
            self._reap(adapter)
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])

    def _refusing_adapter(self):
        adapter = self._lifecycle_adapter()

        async def refusing(event):
            adapter._handled.append(event)

        adapter.handle_message = refusing
        return adapter

    def _busy(self, adapter, event=None, busy=True):
        key = adapter._event_session_key(event or adapter._handled[0])
        if busy:
            adapter._active_sessions[key] = object()
        else:
            adapter._active_sessions.pop(key, None)

    def _dispatch_busy(self, adapter, message_id="<m1@test.com>"):
        import asyncio
        from gateway.config import Platform
        from gateway.platforms.event import MessageEvent
        from gateway.session import SessionSource
        probe = MessageEvent(text="x", source=SessionSource(platform=Platform.EMAIL, chat_id="user@test.com", chat_type="dm", user_id="user@test.com"))
        adapter._active_sessions[adapter._event_session_key(probe)] = object()
        self._dispatch(adapter, self._msg(message_id))
        return probe

    def test_busy_refusal_keeps_mail_in_working(self):
        adapter = self._refusing_adapter()
        self._dispatch_busy(adapter)
        self.assertEqual(adapter._finalized, [])
        self.assertIn("<m1@test.com>", adapter._retry)
        self.assertNotIn("<m1@test.com>", adapter._open_mails)

    def test_queue_full_notice_keeps_mail_and_is_not_sent(self):
        import asyncio
        from gateway.session_ingress import QUEUE_FULL_REPLY
        adapter = self._refusing_adapter()

        async def refusing_with_notice(event):
            adapter._handled.append(event)
            await adapter.send("user@test.com", QUEUE_FULL_REPLY)

        adapter.handle_message = refusing_with_notice
        with patch.object(adapter, "_send_email") as smtp:
            self._dispatch(adapter, self._msg())
        smtp.assert_not_called()
        self.assertEqual(adapter._finalized, [])
        self.assertIn("<m1@test.com>", adapter._retry)

    def test_no_retry_while_busy_then_retry_after_idle_and_finalize_after_turn(self):
        import asyncio
        from gateway.platforms.event import ProcessingOutcome
        adapter = self._refusing_adapter()
        probe = self._dispatch_busy(adapter)
        asyncio.run(adapter._retry_deferred())
        self.assertEqual(len(adapter._handled), 1, "still busy: no re-dispatch")
        self._busy(adapter, probe, busy=False)

        async def accepting(event):
            adapter._handled.append(event)
            event._gateway_accepted = True

        adapter.handle_message = accepting
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._retry_deferred())
        self.assertEqual(len(adapter._handled), 2)
        self.assertEqual(adapter._retry, {})
        self.assertEqual(adapter._finalized, [])
        asyncio.run(adapter.on_processing_complete(adapter._handled[1], ProcessingOutcome.SUCCESS))
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])

    def test_reaper_skips_mails_in_retry_state(self):
        adapter = self._refusing_adapter()
        probe = self._dispatch_busy(adapter)
        self._busy(adapter, probe, busy=False)
        with self._age(adapter):
            self._reap(adapter)
        self.assertEqual(adapter._finalized, [])
        self.assertIn("<m1@test.com>", adapter._retry)

    def test_retry_bound_moves_mail_to_failed_with_warning(self):
        import asyncio
        import plugins.platforms.email.adapter as mod
        adapter = self._refusing_adapter()
        probe = self._dispatch_busy(adapter)
        self._busy(adapter, probe, busy=False)
        with patch.object(mod, "_RETRY_MAX_ATTEMPTS", 2), patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            async def always_busy_refusal(event):
                adapter._handled.append(event)
                adapter._queue_full_chats.add("user@test.com")

            adapter.handle_message = always_busy_refusal
            with self.assertLogs(mod.logger, level="WARNING") as logs:
                for _ in range(4):
                    asyncio.run(adapter._retry_deferred())
        self.assertEqual(adapter._retry, {})
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])
        self.assertEqual(adapter._targets, ["Hermes_Failed"])
        self.assertTrue(any("Giving up" in r for r in logs.output))

    def test_retry_age_bound_moves_mail_to_failed(self):
        import asyncio
        import plugins.platforms.email.adapter as mod
        adapter = self._refusing_adapter()
        self._dispatch_busy(adapter)
        with patch.object(mod, "_RETRY_MAX_AGE_SECONDS", -1.0), self.assertLogs(mod.logger, level="WARNING"):
            asyncio.run(adapter._retry_deferred())
        self.assertEqual(adapter._retry, {})
        self.assertEqual(adapter._targets, ["Hermes_Failed"])

    def test_missing_attachment_moves_mail_to_failed(self):
        import asyncio
        import plugins.platforms.email.adapter as mod
        adapter = self._refusing_adapter()
        probe = self._dispatch_busy(adapter)
        adapter._retry["<m1@test.com>"][0]["attachments"] = [{"path": "/nonexistent/x.bin", "type": "document", "media_type": "x"}]
        self._busy(adapter, probe, busy=False)
        with self.assertLogs(mod.logger, level="WARNING"):
            asyncio.run(adapter._retry_deferred())
        self.assertEqual(adapter._retry, {})
        self.assertEqual(len(adapter._handled), 1)
        self.assertEqual(adapter._targets, ["Hermes_Failed"])

    def test_debounced_and_pending_mails_are_left_to_the_hook(self):
        for holder in ("debounce", "pending"):
            adapter = self._refusing_adapter()
            probe = self._busy_probe(adapter)
            key = adapter._event_session_key(probe)
            (adapter._text_debounce_store() if holder == "debounce" else adapter._pending_messages)[key] = object()
            self._dispatch(adapter, self._msg())
            self.assertEqual(adapter._retry, {}, holder)
            self.assertEqual(adapter._finalized, [], holder)
            self.assertIn("<m1@test.com>", adapter._open_mails, holder)

    def _busy_probe(self, adapter):
        from gateway.config import Platform
        from gateway.platforms.event import MessageEvent
        from gateway.session import SessionSource
        probe = MessageEvent(text="x", source=SessionSource(platform=Platform.EMAIL, chat_id="user@test.com", chat_type="dm", user_id="user@test.com"))
        adapter._active_sessions[adapter._event_session_key(probe)] = object()
        return probe

    def test_rejected_sender_is_done_even_when_session_busy(self):
        adapter = self._refusing_adapter()
        self._busy_probe(adapter)
        self._dispatch(adapter, self._msg(sender="hermes@test.com"))
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])
        self.assertEqual(adapter._retry, {})

    def test_command_is_not_deferred(self):
        adapter = self._refusing_adapter()
        self._busy_probe(adapter)
        self._dispatch(adapter, dict(self._msg(), subject="", body="/stop"))
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])

    def test_idle_refusal_is_finalized_immediately(self):
        adapter = self._refusing_adapter()
        self._dispatch(adapter, self._msg())
        self.assertEqual(adapter._finalized, [("<m1@test.com>", "Hermes_Working")])

    # Routing targets

    def _targets_after(self, extra, scenario):
        adapter = self._lifecycle_adapter({"working_folder": "Hermes_Working", "done_folder": "Hermes_Done", **extra})
        scenario(adapter)
        return adapter._targets

    def _complete_with(self, adapter, outcome):
        import asyncio
        asyncio.run(adapter.on_processing_complete(adapter._handled[0], outcome))

    def test_success_goes_to_done_failure_to_failed(self):
        from gateway.platforms.event import ProcessingOutcome

        def run(outcome):
            def scenario(adapter):
                self._dispatch(adapter, self._msg())
                self._complete_with(adapter, outcome)
            return scenario

        self.assertEqual(self._targets_after({}, run(ProcessingOutcome.SUCCESS)), [None])
        self.assertEqual(self._targets_after({}, run(ProcessingOutcome.FAILURE)), ["Hermes_Failed"])
        self.assertEqual(self._targets_after({"failed_folder": ""}, run(ProcessingOutcome.FAILURE)), ["Hermes_Done"])
        self.assertEqual(self._targets_after({"failed_folder": "F"}, run(ProcessingOutcome.FAILURE)), ["F"])

    def test_rejected_sender_goes_to_rejected(self):
        def scenario(adapter):
            self._dispatch(adapter, self._msg(sender="hermes@test.com"))

        self.assertEqual(self._targets_after({}, scenario), ["Hermes_Rejected"])
        self.assertEqual(self._targets_after({"rejected_folder": ""}, scenario), ["Hermes_Done"])
        self.assertEqual(self._targets_after({"rejected_folder": "R"}, scenario), ["R"])

    def test_unaccepted_event_and_exception_go_to_failed(self):
        def refused(adapter):
            async def refusing(event):
                return None
            adapter.handle_message = refusing
            self._dispatch(adapter, self._msg())

        def raising(adapter):
            async def boom(event):
                raise RuntimeError("x")
            adapter.handle_message = boom
            with self.assertRaises(RuntimeError):
                self._dispatch(adapter, self._msg())

        for scenario in (refused, raising):
            self.assertEqual(self._targets_after({}, scenario), ["Hermes_Failed"])
            self.assertEqual(self._targets_after({"failed_folder": ""}, scenario), ["Hermes_Done"])

    def test_inline_command_goes_to_done(self):
        def scenario(adapter):
            async def refusing(event):
                return None
            adapter.handle_message = refusing
            self._dispatch(adapter, dict(self._msg(), subject="", body="/stop"))

        self.assertEqual(self._targets_after({}, scenario), ["Hermes_Done"])

    def test_reaper_goes_to_done_with_warning(self):
        import plugins.platforms.email.adapter as mod

        def scenario(adapter):
            self._dispatch(adapter, self._msg())
            with patch.object(mod, "_ORPHAN_GRACE_SECONDS", 0.0), self.assertLogs(mod.logger, level="WARNING") as logs:
                self._reap(adapter)
            self.assertTrue(any("<m1@test.com>" in line for line in logs.output))

        self.assertEqual(self._targets_after({}, scenario), [None])

    def test_retry_bound_without_failed_folder_stays_in_working(self):
        import asyncio
        import plugins.platforms.email.adapter as mod
        adapter = self._lifecycle_adapter({"working_folder": "Hermes_Working", "done_folder": "Hermes_Done", "failed_folder": ""})
        adapter.handle_message = lambda event: asyncio.sleep(0)
        self._dispatch_busy(adapter)
        with patch.object(mod, "_RETRY_MAX_AGE_SECONDS", -1.0), self.assertLogs(mod.logger, level="WARNING"):
            asyncio.run(adapter._retry_deferred())
        self.assertEqual(adapter._retry, {})
        self.assertEqual(adapter._finalized, [])

    def test_done_folder_empty_disables_all_routing(self):
        adapter = self._make_adapter({"done_folder": "", "working_folder": ""})
        with patch.object(adapter, "_finalize_conn", None), patch.object(adapter, "_finalize_imap") as imap:
            adapter._finalize_message("<m@t>", "INBOX", "Hermes_Failed")
        imap.assert_not_called()

    def test_finalize_message_moves_to_given_target(self):
        adapter = self._make_adapter({"working_folder": "Hermes_Working", "done_folder": "Hermes_Done"})
        imap = MagicMock()
        with patch.object(adapter, "_finalize_imap", return_value=imap), \
                patch.object(adapter, "_search_message_id", return_value=[b"3"]), \
                patch.object(adapter, "_imap_move") as move:
            adapter._finalize_message("<m@t>", "Hermes_Working", "Hermes_Failed")
        move.assert_called_once_with(imap, b"3", "Hermes_Failed")

    def test_probe_creates_failed_and_rejected_folders(self):
        adapter = self._make_adapter({"working_folder": "W", "done_folder": "D"})
        created = []
        with patch.object(adapter, "_ensure_folder", side_effect=lambda imap, f: created.append(f)), \
                patch.object(adapter, "_inbox") as inbox:
            inbox.return_value.__enter__.return_value = MagicMock(uid=MagicMock(return_value=("OK", [b""])))
            adapter._probe_imap(False)
        self.assertEqual(created, ["W", "D", "Hermes_Failed", "Hermes_Rejected"])

    def test_real_base_flow_finalizes_merged_follow_up_with_its_turn(self):
        import asyncio
        adapter = self._make_adapter({"working_folder": "Hermes_Working", "done_folder": "Hermes_Done"})
        finalized = []
        adapter._finalize_message = lambda mid, folder, target=None: finalized.append(mid)
        release = asyncio.Event
        seen = []

        async def scenario():
            gate = release()

            async def handler(event):
                seen.append(event.message_id)
                if len(seen) == 1:
                    await gate.wait()
                return "ok"

            adapter.set_message_handler(handler)

            adapter.send = lambda *a, **k: asyncio.sleep(0)
            await adapter._dispatch_message(self._msg("<a@test.com>"))
            await asyncio.sleep(0.05)
            self.assertEqual(finalized, [], "turn still running")
            await adapter._dispatch_message(self._msg("<b@test.com>"))
            await adapter._dispatch_message(self._msg("<c@test.com>"))
            await asyncio.sleep(0.05)
            self.assertEqual(finalized, [], "follow-ups are queued behind the running turn")
            gate.set()
            await asyncio.sleep(0.5)

        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(scenario())
        self.assertEqual(seen, ["<a@test.com>", "<b@test.com>"], "the second follow-up merged into the first")
        self.assertEqual(sorted(finalized), ["<a@test.com>", "<b@test.com>", "<c@test.com>"])
        self.assertEqual(len(finalized), 3, "each mail finalized exactly once")
        self.assertEqual(len(adapter._open_mails), 0)


if __name__ == "__main__":
    unittest.main()
