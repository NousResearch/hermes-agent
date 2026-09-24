"""Tests for polling more than one IMAP mailbox (``EMAIL_MAILBOXES`` / ``platforms.email.mailboxes``).

The operator's mail server files channel mail into a folder of its own, and the channel used to read
INBOX only — so the moment a rule moved a message out of the Inbox the channel went deaf. These tests
cover the multi-mailbox poller: the configured mailboxes are all polled, each keeps its own UID
high-water cursor in its own state file, one message reachable through two mailboxes is dispatched
once, and an unset setting behaves exactly like the single-mailbox adapter it replaced.

No network or live mailbox: a small in-process IMAP stand-in holds a per-mailbox message store.
"""

import asyncio
import imaplib
import os
import tempfile
import unittest
from contextlib import contextmanager
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import MagicMock, patch

ADAPTER_LOGGER = "plugins.platforms.email.adapter"


def _raw(subject, message_id, sender="user@test.com"):
    """One RFC822 message: subject + Message-ID are what the poller keys on."""
    msg = MIMEText(f"body of {subject}", "plain", "utf-8")
    msg["From"] = sender
    msg["To"] = "hermes@test.com"
    msg["Subject"] = subject
    msg["Message-ID"] = message_id
    return msg.as_bytes()


class FakeIMAP:
    """Minimal IMAP peer: per-mailbox message stores, SELECT bookkeeping and UID SEARCH/FETCH.

    A message's UID is its 1-based position in its mailbox, so two mailboxes have independent UID
    spaces exactly like a real server (which is what makes a shared seen-UID set wrong). Every
    command is recorded, so a test can prove the poll never wrote to the mailbox.
    """

    def __init__(self, mailboxes):
        self.mailboxes = {name: list(messages) for name, messages in mailboxes.items()}
        self.selected = []
        self.searches = []
        self.fetches = []
        self.store_calls = []
        self.expunge_calls = []
        self.capabilities = ()

    # -- connection / mailbox management --
    def login(self, user, password):
        return ("OK", [b"LOGIN"])

    def logout(self):
        return ("OK", [b"BYE"])

    def shutdown(self):
        pass

    def select(self, mailbox="INBOX", readonly=False):
        if mailbox not in self.mailboxes:
            raise imaplib.IMAP4.error(f"{mailbox}: no such mailbox")
        self.selected.append(mailbox)
        return ("OK", [b"1"])

    def store(self, *args, **kwargs):
        self.store_calls.append(args)
        return ("OK", [b""])

    def expunge(self, *args, **kwargs):
        self.expunge_calls.append(args)
        return ("OK", [b""])

    # -- UID commands --
    def uid(self, command, *args):
        mailbox = self.selected[-1] if self.selected else "INBOX"
        query = args[-1]
        if command == "search":
            self.searches.append((mailbox, query))
            uids = [str(i + 1).encode() for i in range(len(self.mailboxes.get(mailbox, [])))]
            if isinstance(query, str) and query.startswith("UID "):
                floor = int(query.split()[1].split(":")[0])
                uids = [uid for uid in uids if int(uid) >= floor]
            return ("OK", [b" ".join(uids)])
        if command == "fetch":
            self.fetches.append((mailbox, args[0], query))
            return ("OK", [(args[0], self.mailboxes[mailbox][int(args[0]) - 1])])
        return ("NO", [])


@contextmanager
def _email_env(home, **settings):
    """Hermetic EMAIL_* environment with HERMES_HOME pointed at a temp dir (no live mailbox, no real home)."""
    env = {
        "HERMES_HOME": str(home),
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }
    env.update(settings)
    with patch.dict(os.environ, env, clear=True):
        yield


def _construct_adapter():
    """Build an adapter from the ambient (patched) environment."""
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter

    return EmailAdapter(PlatformConfig(enabled=True))


class TestMailboxSelection(unittest.TestCase):
    """The setting's shape: comma-separated (or a YAML list), INBOX always included."""

    def _parse(self, raw):
        from plugins.platforms.email.adapter import _parse_mailboxes

        return _parse_mailboxes(raw)

    def test_unset_means_inbox_alone(self):
        for raw in ("", "   ", None, "INBOX", "inbox", " INBOX ,, "):
            with self.subTest(raw=raw):
                self.assertEqual(self._parse(raw), ["INBOX"])

    def test_configured_mailboxes_are_appended_to_inbox(self):
        self.assertEqual(self._parse("INBOX,Hermes"), ["INBOX", "Hermes"])
        self.assertEqual(self._parse("Hermes"), ["INBOX", "Hermes"])  # INBOX is never dropped
        self.assertEqual(self._parse(" INBOX , Hermes ,Archive, "), ["INBOX", "Hermes", "Archive"])
        self.assertEqual(self._parse(["INBOX", "Hermes"]), ["INBOX", "Hermes"])  # YAML list

    def test_duplicates_are_collapsed(self):
        self.assertEqual(self._parse("INBOX,Hermes,hermes,INBOX"), ["INBOX", "Hermes"])


class TestMailboxConfiguration(unittest.TestCase):
    """Adapter-level configuration: which mailboxes, and where each one's cursor lives."""

    def test_unset_setting_polls_inbox_only_with_the_historic_state_file(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home):
            adapter = _construct_adapter()
            self.assertEqual(adapter._mailboxes, ["INBOX"])
            self.assertEqual(adapter._uid_state_files["INBOX"], Path(home) / "email_state" / "hermes_test.com.uid")
            # INBOX-only deployments keep the historic log line untouched.
            with self.assertLogs(ADAPTER_LOGGER, level="INFO") as logs:
                _construct_adapter()
            self.assertIn("[Email] Adapter initialized for hermes@test.com", "\n".join(logs.output))

    def test_selected_mailboxes_are_logged_at_init(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            with self.assertLogs(ADAPTER_LOGGER, level="INFO") as logs:
                _construct_adapter()
            self.assertIn("[Email] Polling mailboxes: INBOX, Hermes", "\n".join(logs.output))

    def test_each_mailbox_gets_its_own_state_file(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes,Rule Folder"):
            adapter = _construct_adapter()
            state = Path(home) / "email_state"
            self.assertEqual(adapter._uid_state_files["INBOX"], state / "hermes_test.com.uid")
            # <sanitised address>.<sanitised mailbox>.uid
            self.assertEqual(adapter._uid_state_files["Hermes"], state / "hermes_test.com.hermes.uid")
            self.assertEqual(adapter._uid_state_files["Rule Folder"], state / "hermes_test.com.rule_folder.uid")


class TestMailboxPolling(unittest.TestCase):
    """The poll itself: which mailboxes are read, and what is collected from each."""

    def test_unset_setting_only_selects_and_reads_inbox(self):
        """Unchanged behaviour without the setting: one INBOX poll, nothing from another folder."""
        with tempfile.TemporaryDirectory() as home, _email_env(home):
            fake = FakeIMAP({"INBOX": [_raw("inbox mail", "<inbox@test.com>")],
                             "Hermes": [_raw("folder mail", "<folder@test.com>")]})
            adapter = _construct_adapter()
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                results = adapter._fetch_new_messages()
            self.assertEqual(fake.selected, ["INBOX"])
            self.assertEqual([m["subject"] for m in results], ["inbox mail"])
            self.assertEqual(adapter._seen_uids_for("Hermes"), set())

    def test_a_second_mailbox_is_polled_with_the_same_fetch_path(self):
        """Mail a server-side rule moved out of INBOX is still read (the deaf-channel fix)."""
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            fake = FakeIMAP({"INBOX": [], "Hermes": [_raw("moved by rule", "<moved@test.com>")]})
            adapter = _construct_adapter()
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                results = adapter._fetch_new_messages()
            self.assertEqual(fake.selected, ["INBOX", "Hermes"])
            self.assertEqual([m["subject"] for m in results], ["moved by rule"])

    def test_folder_message_whose_uid_collides_with_an_inbox_uid_is_still_read(self):
        """Seen UIDs are per mailbox: UID 1 of a folder is not UID 1 of the Inbox."""
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            fake = FakeIMAP({"INBOX": [_raw("inbox one", "<i1@test.com>")],
                             "Hermes": [_raw("folder one", "<f1@test.com>")]})
            adapter = _construct_adapter()
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                results = adapter._fetch_new_messages()
            # INBOX's UID 1 is now seen; the folder's own UID 1 must not be swallowed by that.
            self.assertIn(b"1", adapter._seen_uids)
            self.assertEqual(sorted(m["subject"] for m in results), ["folder one", "inbox one"])

    def test_single_mailbox_keeps_the_historic_no_dedupe_behaviour(self):
        """With the setting unset nothing is deduped — a same-id redelivery in INBOX is still two messages."""
        with tempfile.TemporaryDirectory() as home, _email_env(home):
            fake = FakeIMAP({"INBOX": [_raw("first", "<same@test.com>"), _raw("again", "<same@test.com>")]})
            adapter = _construct_adapter()
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                results = adapter._fetch_new_messages()
            self.assertEqual(len(results), 2)

    def test_polling_never_writes_to_the_mailbox(self):
        """Read-only guarantee: BODY.PEEK[] fetches only, no STORE/EXPUNGE/COPY/move."""
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            fake = FakeIMAP({"INBOX": [_raw("inbox", "<i@test.com>")],
                             "Hermes": [_raw("folder", "<f@test.com>")]})
            adapter = _construct_adapter()
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                adapter._fetch_new_messages()
            self.assertEqual(fake.store_calls, [])
            self.assertEqual(fake.expunge_calls, [])
            self.assertEqual([query for _mailbox, _uid, query in fake.fetches], ["(BODY.PEEK[])"] * 2)

    def test_a_mailbox_that_cannot_be_read_does_not_stop_inbox(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            fake = FakeIMAP({"INBOX": [_raw("inbox mail", "<i@test.com>")]})  # no Hermes folder on the server
            adapter = _construct_adapter()
            with self.assertLogs(ADAPTER_LOGGER, level="ERROR") as logs, \
                 patch("imaplib.IMAP4_SSL", return_value=fake):
                results = adapter._fetch_new_messages()
            self.assertEqual([m["subject"] for m in results], ["inbox mail"])
            self.assertIn("IMAP poll failed for mailbox Hermes", "\n".join(logs.output))
            # A single unreadable folder is not a connection failure: no reconnect escalation.
            self.assertFalse(adapter._last_fetch_failed)


class TestCrossMailboxDedupe(unittest.TestCase):
    """One message reachable through two mailboxes is dispatched exactly once."""

    def test_same_message_id_in_two_mailboxes_is_dispatched_once(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            # The same mail, filed in both places at once — the case a naive "poll both" answers twice.
            fake = FakeIMAP({"INBOX": [_raw("both places", "<dup@test.com>")],
                             "Hermes": [_raw("both places", "<dup@test.com>")]})
            adapter = _construct_adapter()
            dispatched = []

            async def collect(msg_data):
                dispatched.append(msg_data)

            adapter._dispatch_message = collect
            with self.assertLogs(ADAPTER_LOGGER, level="DEBUG") as logs, \
                 patch("imaplib.IMAP4_SSL", return_value=fake):
                asyncio.run(adapter._check_inbox())

            self.assertEqual([m["subject"] for m in dispatched], ["both places"])
            self.assertTrue(any("Skipping duplicate Message-ID <dup@test.com>" in line for line in logs.output),
                            logs.output)

    def test_message_id_dedupe_ignores_case_and_padding(self):
        """The same message keeps its Message-ID byte-for-byte, but the key compare is normalised."""
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            fake = FakeIMAP({"INBOX": [_raw("one", "<Dup@Test.com>")],
                             "Hermes": [_raw("filler", "<filler@test.com>"),
                                        _raw("two", "  <dup@test.com>  ")]})
            adapter = _construct_adapter()
            dispatched = []

            async def collect(msg_data):
                dispatched.append(msg_data)

            adapter._dispatch_message = collect
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                asyncio.run(adapter._check_inbox())
            self.assertEqual([m["subject"] for m in dispatched], ["one", "filler"])


class TestPerMailboxUidCursors(unittest.TestCase):
    """Each mailbox keeps its own UID high-water cursor, in its own state file, across a restart."""

    def _state(self, home, name):
        return Path(home) / "email_state" / name

    def test_cursors_are_per_mailbox_and_resume_after_a_restart(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes", EMAIL_POLL_MODE="uid"):
            fake = FakeIMAP({"INBOX": [_raw("inbox one", "<i1@test.com>")],
                             "Hermes": [_raw("folder one", "<f1@test.com>")]})
            inbox_state, folder_state = self._state(home, "hermes_test.com.uid"), self._state(home, "hermes_test.com.hermes.uid")

            first = _construct_adapter()
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                self.assertEqual(first._fetch_new_messages(), [])  # first run baselines, dispatches nothing
            self.assertEqual(fake.selected, ["INBOX", "Hermes"])
            self.assertEqual(inbox_state.read_text().strip(), "1")
            self.assertEqual(folder_state.read_text().strip(), "1")

            # New mail in the folder only — nowhere else, exactly what a filing rule produces.
            fake.mailboxes["Hermes"].append(_raw("folder two", "<f2@test.com>"))

            restarted = _construct_adapter()  # fresh adapter, same HERMES_HOME: cursors come from the files
            self.assertEqual(restarted._last_uids, {"INBOX": 1, "Hermes": 1})
            dispatched = []

            async def collect(msg_data):
                dispatched.append(msg_data)

            restarted._dispatch_message = collect
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                asyncio.run(restarted._check_inbox())

            self.assertEqual([m["subject"] for m in dispatched], ["folder two"])
            self.assertEqual(inbox_state.read_text().strip(), "1")   # INBOX cursor stayed put
            self.assertEqual(folder_state.read_text().strip(), "2")  # the folder's cursor advanced on its own

            # And it stays consumed: a third adapter replays nothing.
            third = _construct_adapter()
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                self.assertEqual(third._fetch_new_messages(), [])

    def test_corrupt_extra_mailbox_state_file_does_not_break_the_channel(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes", EMAIL_POLL_MODE="uid"):
            state = Path(home) / "email_state"
            state.mkdir(parents=True, exist_ok=True)
            (state / "hermes_test.com.hermes.uid").write_text("not-a-number\n")
            adapter = _construct_adapter()
            self.assertEqual(adapter._last_uids["Hermes"], 0)


class TestConnectBaselines(unittest.TestCase):
    """connect() baselines every configured mailbox and keeps INBOX's historic behaviour."""

    def setUp(self):
        from plugins.platforms.email.adapter import EmailAdapter

        EmailAdapter._seen_uids_snapshot.clear()

    tearDown = setUp

    def test_connect_baselines_every_configured_mailbox(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            fake = FakeIMAP({"INBOX": [_raw("a", "<a@t.com>"), _raw("b", "<b@t.com>")],
                             "Hermes": [_raw("c", "<c@t.com>")]})
            adapter = _construct_adapter()
            adapter._connect_smtp = MagicMock(return_value=MagicMock())
            with patch("imaplib.IMAP4_SSL", return_value=fake):
                self.assertTrue(asyncio.run(adapter.connect()))
                asyncio.run(adapter.disconnect())
            self.assertEqual(fake.selected[:2], ["INBOX", "Hermes"])  # the probe, before any poll
            self.assertEqual(adapter._seen_uids, {b"1", b"2"})        # INBOX, as before
            self.assertEqual(adapter._seen_uids_for("Hermes"), {b"1"})  # the folder baselined on its own

    def test_connect_skips_a_mailbox_it_cannot_baseline(self):
        with tempfile.TemporaryDirectory() as home, _email_env(home, EMAIL_MAILBOXES="INBOX,Hermes"):
            fake = FakeIMAP({"INBOX": [_raw("a", "<a@t.com>")]})  # Hermes does not exist on the server
            adapter = _construct_adapter()
            adapter._connect_smtp = MagicMock(return_value=MagicMock())
            with self.assertLogs(ADAPTER_LOGGER, level="ERROR") as logs, \
                 patch("imaplib.IMAP4_SSL", return_value=fake):
                self.assertTrue(asyncio.run(adapter.connect()))
                asyncio.run(adapter.disconnect())
            self.assertEqual(adapter._mailboxes, ["INBOX"])  # unbaselined mailboxes are not polled blind
            self.assertIn("IMAP probe failed for mailbox Hermes", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
