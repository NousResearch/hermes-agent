"""Tests for email threads: thread roots, subject, the HTML part, per-thread
sessions and cold-start catch-up.

Everything here is offline: SMTP and IMAP are mocks, and addresses use test.com
or example.com. This module holds the shared fixtures; the thread tests build
on them.
"""

import asyncio
import json
import os
import unittest
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


ENV = {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
}


def make_adapter(**extra):
    """Build an EmailAdapter with mocked env vars and ``extra`` platform settings."""
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, ENV):
        from plugins.platforms.email.adapter import EmailAdapter

        adapter = EmailAdapter(PlatformConfig(enabled=True, extra=dict(extra)))
    return adapter


def make_inbound(
    *,
    sender="User <user@test.com>",
    subject="Hello",
    body="Hi",
    message_id="<in-1@test.com>",
    in_reply_to=None,
    references=None,
) -> bytes:
    """Raw bytes of an inbound plain-text mail; threading headers only when given."""
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = sender
    msg["To"] = "hermes@test.com"
    msg["Subject"] = subject
    msg["Message-ID"] = message_id
    if in_reply_to is not None:
        msg["In-Reply-To"] = in_reply_to
    if references is not None:
        msg["References"] = references
    return msg.as_bytes()


def _as_bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode()


def fake_imap(uids_all=b"", uids_unseen=b"", messages=None, uidvalidity=b"42"):
    """A MagicMock IMAP4_SSL connection.

    ``uid("search", None, "ALL")`` returns *uids_all*, ``uid("search", None,
    "UNSEEN")`` returns *uids_unseen*, ``uid("fetch", uid, ...)`` returns the
    raw bytes in *messages* (keys are bytes like ``b"4"``) and ``uid("store", ...)``
    is OK. Anything else is NO.
    ``response("UIDVALIDITY")`` gives *uidvalidity* (``[None]`` when it is None).
    """
    messages = dict(messages or {})
    imap = MagicMock()

    def uid_handler(command, *args):
        command = str(command).lower()
        if command == "search":
            criteria = " ".join(str(a) for a in args if a is not None).upper()
            if "UNSEEN" in criteria:
                return ("OK", [uids_unseen])
            if "ALL" in criteria:
                return ("OK", [uids_all])
            return ("NO", [])
        if command == "fetch" and args:
            uid = _as_bytes(args[0])
            if uid in messages:
                return ("OK", [(uid, messages[uid])])
        if command == "store":
            return ("OK", [])
        return ("NO", [])

    imap.uid.side_effect = uid_handler
    imap.response.return_value = ("UIDVALIDITY", [uidvalidity])
    imap.select.return_value = ("OK", [b"1"])
    imap.login.return_value = ("OK", [b""])
    imap.logout.return_value = ("OK", [b""])
    return imap


def sent_message(mock_smtp_cls):
    """The message object passed to ``send_message`` on a patched ``smtplib.SMTP``."""
    return mock_smtp_cls.return_value.send_message.call_args[0][0]


class DispatchCase(unittest.TestCase):
    """Base for tests that drive ``_dispatch_message`` / the poll path.

    Opts into allow-all and From-header trust (the dispatch mechanics are under
    test, not the authorization gate), and clears the class-level seen-UID
    snapshot so one test's connect never leaks into the next.
    """

    _ENV_FLAGS = {"EMAIL_ALLOW_ALL_USERS": "true", "EMAIL_TRUST_FROM_HEADER": "true"}

    def setUp(self):
        from plugins.platforms.email.adapter import EmailAdapter

        self._prev_env = {name: os.environ.get(name) for name in self._ENV_FLAGS}
        os.environ.update(self._ENV_FLAGS)
        self._prev_snapshot = dict(EmailAdapter._seen_uids_snapshot)
        EmailAdapter._seen_uids_snapshot.clear()

    def tearDown(self):
        from plugins.platforms.email.adapter import EmailAdapter

        for name, value in self._prev_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        EmailAdapter._seen_uids_snapshot.clear()
        EmailAdapter._seen_uids_snapshot.update(self._prev_snapshot)

    def capture(self, adapter):
        """Replace ``adapter.handle_message`` with a recorder; returns the event list."""
        events = []

        async def _capture(event):
            events.append(event)

        adapter.handle_message = _capture
        return events


class TestScaffold(unittest.TestCase):
    def test_scaffold_adapter_store_dir_is_sandboxed(self):
        """The thread store's directory stays inside the per-test HERMES_HOME."""
        from plugins.plugin_storage import plugin_data_dir

        adapter = make_adapter()
        self.assertIsNotNone(adapter)
        home = Path(os.environ["HERMES_HOME"]).resolve()
        self.assertIn(home, plugin_data_dir("email").resolve().parents)


class TestSubjectRule(unittest.TestCase):
    """MAIL-01: a leading ``Subject:`` line names the mail, else the job name, else "Hermes Agent"."""

    def test_subject_from_leading_subject_line(self):
        from plugins.platforms.email.adapter import _split_subject

        self.assertEqual(_split_subject("Subject: Weekly report — Nov 9\n\n# Body\n- a", "Fallback"),
                         ("Weekly report — Nov 9", "# Body\n- a"))
        self.assertEqual(_split_subject("\n\nSubject: Late start\nbody", "F"), ("Late start", "body"))
        self.assertEqual(_split_subject("subject: lower case\nbody", "F"), ("lower case", "body"))

    def test_subject_after_cron_failure_marker(self):
        from plugins.platforms.email.adapter import _split_subject

        self.assertEqual(_split_subject("[CRON_FAILURE]\nSubject: Job broke\nDetails", "F"), ("Job broke", "Details"))

    def test_subject_not_taken_from_later_lines(self):
        from plugins.platforms.email.adapter import _split_subject

        self.assertEqual(_split_subject("Intro\nSubject: x", "F"), ("F", "Intro\nSubject: x"))
        self.assertEqual(_split_subject("Subject: \nbody", "F"), ("F", "Subject: \nbody"))

    def test_subject_strips_control_chars_and_caps_length(self):
        from plugins.platforms.email.adapter import _split_subject

        self.assertEqual(_split_subject("Subject: A\x07B\r\nbody", "F"), ("AB", "body"))
        subject, body = _split_subject("Subject: " + "x" * 400 + "\nbody", "F")
        self.assertEqual(subject, "x" * 250)
        self.assertEqual(body, "body")

    def test_subject_falls_back_to_job_name(self):
        from plugins.platforms.email.adapter import _subject_from_thread_name

        self.assertEqual(_subject_from_thread_name("Hermes — Weekly report"), "Weekly report")
        self.assertEqual(_subject_from_thread_name(None), "Hermes Agent")
        self.assertEqual(_subject_from_thread_name(""), "Hermes Agent")
        self.assertEqual(_subject_from_thread_name("  Plain name  "), "Plain name")

    def test_standalone_send_uses_subject_rule(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import _standalone_send

        with patch("smtplib.SMTP") as mock_smtp, patch.dict(os.environ, ENV):
            result = asyncio.run(_standalone_send(PlatformConfig(enabled=True), "user@test.com", "Subject: Status\n\nAll good"))
            self.assertTrue(result["success"])
            msg = sent_message(mock_smtp)
            self.assertEqual(msg["Subject"], "Status")
            body = msg.get_payload(decode=True).decode("utf-8")
            self.assertEqual(body, "All good")
            self.assertNotIn("Subject:", body)

        with patch("smtplib.SMTP") as mock_smtp, patch.dict(os.environ, ENV):
            asyncio.run(_standalone_send(PlatformConfig(enabled=True), "user@test.com", "No subject line here"))
            self.assertEqual(sent_message(mock_smtp)["Subject"], "Hermes Agent")


def _html_of(msg):
    """Decoded text/html payload of *msg*, or None."""
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            return part.get_payload(decode=True).decode("utf-8")
    return None


class TestHtmlPart(unittest.TestCase):
    """MAIL-02: ``platforms.email.html`` adds a text/html alternative rendered from Markdown."""

    def setUp(self):
        import plugins.platforms.email.adapter as adapter_module

        self.adapter_module = adapter_module
        adapter_module._MD_RENDERER = None

    def tearDown(self):
        self.adapter_module._MD_RENDERER = None

    def _send(self, adapter, body):
        with patch("smtplib.SMTP") as mock_smtp:
            adapter._send_email("user@test.com", body, None)
            return sent_message(mock_smtp)

    def test_html_part_is_multipart_alternative_when_enabled(self):
        body = "# Title\n\n- one\n- two"
        msg = self._send(make_adapter(html=True), body)
        types = [p.get_content_type() for p in msg.walk()]
        self.assertEqual(types, ["multipart/mixed", "multipart/alternative", "text/plain", "text/html"])
        plain = next(p for p in msg.walk() if p.get_content_type() == "text/plain")
        self.assertEqual(plain.get_payload(decode=True).decode("utf-8"), body)
        html = _html_of(msg)
        self.assertIn("<h1>Title</h1>", html)
        self.assertIn("<li>one</li>", html)

    def test_html_escapes_raw_html(self):
        html = _html_of(self._send(make_adapter(html=True), "<script>alert(1)</script> and [x](javascript:alert(1))"))
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script", html)
        self.assertNotIn('href="javascript:', html)

    def test_html_image_rendered_as_link(self):
        html = _html_of(self._send(make_adapter(html=True), "![pixel](http://tracker.example.com/p.gif)"))
        self.assertNotIn("<img", html)
        self.assertIn('<a href="http://tracker.example.com/p.gif">pixel</a>', html)  # a stray "!" before it is fine

    def test_no_html_part_when_flag_off(self):
        msg = self._send(make_adapter(), "# Title\n\n- one")
        self.assertEqual([p.get_content_type() for p in msg.walk()], ["multipart/mixed", "text/plain"])

    def test_html_nested_in_mixed_with_attachment(self):
        import tempfile

        adapter = make_adapter(html=True)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"doc")
            tmp_path = f.name
        try:
            with patch("smtplib.SMTP") as mock_smtp:
                adapter._send_email_with_attachment("user@test.com", "**See** the file", tmp_path)
                msg = sent_message(mock_smtp)
        finally:
            os.unlink(tmp_path)
        self.assertEqual(msg.get_content_type(), "multipart/mixed")
        top = msg.get_payload()
        self.assertEqual([p.get_content_type() for p in top], ["multipart/alternative", "application/octet-stream"])
        self.assertEqual([p.get_content_type() for p in top[0].get_payload()], ["text/plain", "text/html"])
        self.assertIn("attachment", top[1]["Content-Disposition"])

    def test_html_flag_accepts_top_level_and_extra_spelling(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        with patch.dict(os.environ, ENV):
            top = EmailAdapter(PlatformConfig.from_dict({"enabled": True, "html": True}))
            nested = EmailAdapter(PlatformConfig.from_dict({"enabled": True, "extra": {"html": "true"}}))
            default = EmailAdapter(PlatformConfig.from_dict({"enabled": True}))
        self.assertIs(top._html, True)
        self.assertIs(nested._html, True)
        self.assertIs(default._html, False)

    def test_html_falls_back_to_plain_when_markdown_it_missing(self):
        import sys

        adapter = make_adapter(html=True)
        with patch.dict(sys.modules, {"markdown_it": None}), \
                self.assertLogs("plugins.platforms.email.adapter", "WARNING") as logs:
            first = self._send(adapter, "# Title")
            second = self._send(adapter, "# Again")
        for msg in (first, second):
            self.assertEqual([p.get_content_type() for p in msg.walk()], ["multipart/mixed", "text/plain"])
        warnings = [line for line in logs.output if "markdown-it-py" in line]
        self.assertEqual(len(warnings), 1)

    def test_standalone_send_html_when_enabled(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import _standalone_send

        with patch("smtplib.SMTP") as mock_smtp, patch.dict(os.environ, ENV):
            result = asyncio.run(_standalone_send(PlatformConfig(enabled=True, extra={"html": True}), "user@test.com",
                                                  "Subject: Status\n\n# All good"))
            self.assertTrue(result["success"])
            msg = sent_message(mock_smtp)
        self.assertEqual(msg["Subject"], "Status")
        self.assertEqual([p.get_content_type() for p in msg.walk()], ["multipart/alternative", "text/plain", "text/html"])
        self.assertIn("<h1>All good</h1>", _html_of(msg))


def _store_path() -> Path:
    from plugins.plugin_storage import plugin_data_dir

    return plugin_data_dir("email") / "threads.json"


class TestThreadStore(unittest.TestCase):
    """The persistent thread store and the one root sanitiser both ends share."""

    def test_root_sanitised_for_session_key(self):
        from gateway.session import _is_path_unsafe
        from plugins.platforms.email.adapter import _root_key

        self.assertEqual(_root_key("<hermes-abc@test.com>"), "hermes-abc@test.com")
        self.assertEqual(_root_key("  hermes-abc@test.com "), "hermes-abc@test.com")
        hashed = ["<a/../b@x.com>", "<a b@x.com>", "<" + "x" * 200 + "@x.com>", "<\\evil@x.com>", "<c:x@y>",
                  "<x@[IPv6:::1]>"]
        for raw in hashed:
            key = _root_key(raw)
            self.assertTrue(key.startswith("h-"), raw)
            self.assertEqual(len(key), 26, raw)
            self.assertEqual(_root_key(raw), key, raw)
        self.assertEqual(len({_root_key(raw) for raw in hashed}), len(hashed))
        for raw in hashed + ["<hermes-abc@test.com>"]:
            self.assertFalse(_is_path_unsafe(_root_key(raw), strict=False), raw)

    def test_message_ids_parses_folded_header(self):
        from plugins.platforms.email.adapter import _message_ids

        self.assertEqual(_message_ids("<a@x>\r\n\t<b@y> junk <c@z>"), ["<a@x>", "<b@y>", "<c@z>"])
        self.assertEqual(_message_ids(None), [])
        self.assertEqual(_message_ids(""), [])

    def test_thread_store_survives_new_adapter_instance(self):
        first = make_adapter()
        first._store.add_pending_root("k1", "<k1>", "Hermes — Report", "user@test.com")
        second = make_adapter()
        root = second._store.get_root("k1")
        self.assertEqual(root["name"], "Hermes — Report")
        self.assertIs(root["pending"], True)
        self.assertTrue(second._store.is_pending("k1"))

    def test_corrupt_store_resets_to_empty(self):
        path = _store_path()
        path.write_text("{not json", encoding="utf-8")
        with self.assertLogs("plugins.platforms.email.adapter", "WARNING") as logs:
            adapter = make_adapter()
        self.assertTrue(any("threads.json" in line for line in logs.output))
        self.assertIsNone(adapter._store.get_root("k1"))
        adapter._store.add_pending_root("k1", "<k1>", "Hermes — Report", "user@test.com")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("k1", data["roots"])

    def test_store_prunes_old_roots(self):
        import time

        now = time.time()
        root = {"header": "", "name": "n", "subject": "", "job_id": "", "chat_id": "user@test.com", "pending": False}
        _store_path().write_text(json.dumps({
            "version": 1,
            "roots": {"old": dict(root, header="<old>", created=now - 91 * 86400),
                      "new": dict(root, header="<new>", created=now - 10 * 86400)},
            "index": {"old": "old", "m-old@x": "old", "new": "new", "m-new@x": "new"},
            "mailboxes": {},
        }), encoding="utf-8")
        store = make_adapter()._store
        self.assertIsNone(store.get_root("old"))
        self.assertIsNone(store.root_for(["<m-old@x>"]))
        self.assertEqual(store.get_root("new")["header"], "<new>")
        self.assertEqual(store.root_for(["<m-new@x>"]), "new")
        store.index_message("<m2@x>", "new")  # the next write persists the pruned state
        on_disk = json.loads(_store_path().read_text(encoding="utf-8"))
        self.assertEqual(set(on_disk["roots"]), {"new"})
        self.assertNotIn("m-old@x", on_disk["index"])
        self.assertEqual(on_disk["index"]["m2@x"], "new")

    def test_store_index_resolves_first_known_id(self):
        store = make_adapter()._store
        store.add_pending_root("k1", "<k1>", "Hermes — Report", "user@test.com")
        store.index_message("<m2@x>", "k1")
        self.assertEqual(store.root_for(["<unknown@x>", "<m2@x>"]), "k1")
        self.assertEqual(store.root_for(["<k1>"]), "k1")
        self.assertIsNone(store.root_for([]))
        self.assertIsNone(store.root_for(["<unknown@x>"]))

    def test_store_concurrent_writes_keep_valid_json(self):
        import threading

        store = make_adapter()._store
        store.add_pending_root("k1", "<k1>", "Hermes — Report", "user@test.com")

        def worker(n):
            for i in range(50):
                store.index_message(f"<t{n}-{i}@x>", "k1")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        data = json.loads(_store_path().read_text(encoding="utf-8"))
        ids = {f"t{n}-{i}@x" for n in range(4) for i in range(50)}
        self.assertEqual(ids - set(data["index"]), set())
        self.assertEqual(len(ids), 200)


class TestOutboundThreads(unittest.TestCase):
    """MAIL-01 end to end and the outbound half of MAIL-03: minted roots, fresh first mail, References chain."""

    def _mint(self, adapter, name="Hermes — Report"):
        return asyncio.run(adapter.create_handoff_thread("user@test.com", name))

    def _send(self, adapter, content, **kwargs):
        with patch("smtplib.SMTP") as mock_smtp:
            result = asyncio.run(adapter.send("user@test.com", content, **kwargs))
            msg = sent_message(mock_smtp) if mock_smtp.return_value.send_message.called else None
        return result, msg

    @staticmethod
    def _plain(msg):
        return next(p for p in msg.walk() if p.get_content_type() == "text/plain").get_payload(decode=True).decode("utf-8")

    def test_create_handoff_thread_mints_pending_root(self):
        import re

        adapter = make_adapter()
        key = asyncio.run(adapter.create_handoff_thread("User@Test.com", "Hermes — Report"))
        self.assertRegex(key, r"^hermes-[0-9a-f]{12}@test\.com$")
        root = adapter._store.get_root(key)
        self.assertIs(root["pending"], True)
        self.assertEqual(root["name"], "Hermes — Report")
        self.assertEqual(root["chat_id"], "user@test.com")
        self.assertEqual(root["header"], "<" + key + ">")
        self.assertIsNotNone(re.fullmatch(r"[^<>\s/\\]+", key))

    def test_pending_root_sent_as_fresh_mail_with_minted_message_id(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        result, msg = self._send(adapter, "Subject: Report — Nov 9\n\n- item", metadata={"thread_id": key, "job_id": "j1"})
        self.assertTrue(result.success)
        self.assertEqual(msg["Message-ID"], "<" + key + ">")
        self.assertEqual(msg["Subject"], "Report — Nov 9")
        self.assertIsNone(msg["In-Reply-To"])
        self.assertEqual(msg["References"], "<" + key + ">")
        self.assertNotIn("Subject:", self._plain(msg))
        self.assertEqual(self._plain(msg), "- item")
        self.assertEqual(result.message_id, "<" + key + ">")
        root = adapter._store.get_root(key)
        self.assertIs(root["pending"], False)
        self.assertEqual(root["job_id"], "j1")
        self.assertEqual(root["subject"], "Report — Nov 9")

    def test_pending_root_subject_falls_back_to_job_name(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        _, msg = self._send(adapter, "- item", metadata={"thread_id": key, "job_id": "j1"})
        self.assertEqual(msg["Subject"], "Report")
        self.assertEqual(msg["Message-ID"], "<" + key + ">")

    def test_pending_root_ignores_per_sender_context(self):
        adapter = make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Old", "message_id": "<old@test.com>"}
        key = self._mint(adapter)
        _, msg = self._send(adapter, "- item", metadata={"thread_id": key, "job_id": "j1"})
        self.assertIsNone(msg["In-Reply-To"])
        self.assertEqual(msg["References"], "<" + key + ">")
        self.assertNotIn("old@test.com", msg["References"])
        self.assertNotIn("Old", msg["Subject"])
        self.assertEqual(msg["Subject"], "Report")

    def test_pending_root_references_its_own_minted_id(self):
        # Some providers replace the Message-ID of submitted mail; References survives, so the root names itself there.
        adapter = make_adapter()
        key = self._mint(adapter)
        _, msg = self._send(adapter, "Subject: Report\n\n- item", metadata={"thread_id": key, "job_id": "j1"})
        self.assertEqual(msg["References"], "<" + key + ">")
        self.assertEqual(msg["Message-ID"], "<" + key + ">")
        self.assertIsNone(msg["In-Reply-To"])
        self.assertEqual(len(msg.get_all("References")), 1)

    def test_second_chunk_threads_under_root(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        meta = {"thread_id": key, "job_id": "j1"}
        self._send(adapter, "Subject: Report — Nov 9\n\n- part one", metadata=meta)
        result, msg = self._send(adapter, "- part two", metadata=meta)
        self.assertEqual(msg["Subject"], "Re: Report — Nov 9")
        self.assertEqual(msg["In-Reply-To"], "<" + key + ">")
        self.assertEqual(msg["References"], "<" + key + ">")
        self.assertNotEqual(msg["Message-ID"], "<" + key + ">")
        self.assertEqual(result.message_id, msg["Message-ID"])
        self.assertEqual(adapter._store.root_for([msg["Message-ID"]]), key)
        self.assertEqual(adapter._store.get_root(key)["job_id"], "j1")

    def test_reply_sends_full_references_chain(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        self._send(adapter, "Subject: Report — Nov 9\n\n- item", metadata={"thread_id": key, "job_id": "j1"})
        root_id = "<" + key + ">"
        adapter._root_context[("user@test.com", key)] = {
            "subject": "Report — Nov 9", "message_id": "<r2@test.com>", "references": [root_id, "<h1@test.com>"]}
        _, msg = self._send(adapter, "answer", reply_to="<r2@test.com>", metadata={"thread_id": key})
        self.assertEqual(msg["In-Reply-To"], "<r2@test.com>")
        self.assertEqual(msg["References"], f"{root_id} <h1@test.com> <r2@test.com>")
        self.assertEqual(msg["Subject"], "Re: Report — Nov 9")

    def test_references_chain_capped(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        self._send(adapter, "- item", metadata={"thread_id": key, "job_id": "j1"})
        root_id = "<" + key + ">"
        refs = [root_id] + [f"<m{i}@test.com>" for i in range(1, 40)]
        self.assertEqual(len(refs), 40)
        adapter._root_context[("user@test.com", key)] = {"subject": "Report", "message_id": "<m39@test.com>",
                                                         "references": refs}
        _, msg = self._send(adapter, "answer", reply_to="<p@test.com>", metadata={"thread_id": key})
        ids = msg["References"].split()
        self.assertEqual(len(ids), 20)
        self.assertEqual(ids[0], root_id)
        self.assertEqual(ids[1:], (refs + ["<p@test.com>"])[-19:])
        self.assertEqual(ids[-1], "<p@test.com>")
        self.assertEqual(msg["In-Reply-To"], "<p@test.com>")

    def test_send_without_thread_unchanged(self):
        adapter = make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Project question", "message_id": "<original@test.com>"}
        result, msg = self._send(adapter, "x")
        self.assertTrue(result.success)
        self.assertEqual(msg["Subject"], "Re: Project question")
        self.assertEqual(msg["In-Reply-To"], "<original@test.com>")
        self.assertEqual(msg["References"], "<original@test.com>")
        self.assertEqual(adapter._store.root_for([msg["Message-ID"]]), None)

    def test_failed_smtp_leaves_root_pending(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.send_message.side_effect = OSError("smtp down")
            result = asyncio.run(adapter.send("user@test.com", "Subject: Report\n\nbody",
                                              metadata={"thread_id": key, "job_id": "j1"}))
        self.assertIs(result.success, False)
        root = adapter._store.get_root(key)
        self.assertIs(root["pending"], True)
        self.assertEqual(root["job_id"], "")
        self.assertIsNone(adapter._store.root_for(["<unrelated@test.com>"]))

    def test_attachments_thread_under_root(self):
        """A document, an image batch or an image link sent in a thread replies inside it, as text does, and is
        indexed to the root; it never threads on the sender's latest mail from another conversation."""
        import tempfile

        adapter = make_adapter()
        key = self._mint(adapter)
        self._send(adapter, "Subject: Report — Nov 9\n\n- item", metadata={"thread_id": key, "job_id": "j1"})
        adapter._thread_context["user@test.com"] = {"subject": "Other question", "message_id": "<other@test.com>"}
        meta = {"thread_id": key}
        with tempfile.TemporaryDirectory() as tmp:
            document, image = Path(tmp) / "report.txt", Path(tmp) / "chart.png"
            document.write_text("figures", encoding="utf-8")
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            calls = {
                "send_document": lambda: adapter.send_document("user@test.com", str(document), caption="The file",
                                                               metadata=meta),
                "send_multiple_images": lambda: adapter.send_multiple_images(
                    "user@test.com", [(f"file://{image}", "Chart")], metadata=meta),
                "send_image": lambda: adapter.send_image("user@test.com", "https://example.com/c.png",
                                                         caption="Chart", metadata=meta),
            }
            for method, call in calls.items():
                with self.subTest(method=method), patch("smtplib.SMTP") as mock_smtp:
                    result = asyncio.run(call())
                    msg = sent_message(mock_smtp)
                    self.assertTrue(result.success)
                    self.assertEqual(msg["Subject"], "Re: Report — Nov 9")
                    self.assertEqual(msg["In-Reply-To"], f"<{key}>")
                    self.assertEqual(msg["References"].split()[0], f"<{key}>")
                    self.assertNotIn("<other@test.com>", msg["References"])
                    self.assertEqual(adapter._store.root_for([msg["Message-ID"]]), key)


class TestInboundThreads(DispatchCase):
    """The inbound half of MAIL-03 and MAIL-04: replies resolve to their thread's root and session key."""

    FOREIGN_REFS = "<foreign-root@mail.example.com> <foreign-2@mail.example.com>"
    REWRITTEN = "<0100018f0a1b2c3d-example@email.example.com>"

    def _mint_and_send(self, adapter, content="Subject: Report\n\n- item"):
        key = asyncio.run(adapter.create_handoff_thread("user@test.com", "Hermes — Report"))
        self._send(adapter, content, metadata={"thread_id": key, "job_id": "j1"})
        return key

    def _send(self, adapter, content, **kwargs):
        with patch("smtplib.SMTP") as mock_smtp:
            result = asyncio.run(adapter.send("user@test.com", content, **kwargs))
            msg = sent_message(mock_smtp) if mock_smtp.return_value.send_message.called else None
        return result, msg

    def _dispatch(self, adapter, **inbound):
        events = self.capture(adapter)
        asyncio.run(adapter._dispatch_message(adapter._parse_fetched_message(b"1", make_inbound(**inbound))))
        return events[-1] if events else None

    @staticmethod
    def _key(source):
        from gateway.session import build_session_key

        return build_session_key(source)

    def test_references_resolved_to_thread_root(self):
        adapter = make_adapter()
        key = self._mint_and_send(adapter)
        event = self._dispatch(adapter, subject="Re: Report", in_reply_to=f"<{key}>", references=f"<{key}>",
                               message_id="<r1@test.com>")
        self.assertEqual(event.source.thread_id, key)
        self.assertEqual(event.source.chat_id, "user@test.com")
        self.assertEqual(event.source.chat_type, "dm")

    def test_reply_to_rewritten_root_id_routes_by_minted_reference(self):
        # The provider delivered the root under REWRITTEN; the reader's client replies to that id and copies the
        # root's References before it, as mail clients do.
        for n, extra in enumerate(({}, {"thread_sessions": True})):
            with self.subTest(extra=extra):
                adapter = make_adapter(**extra)
                key = asyncio.run(adapter.create_handoff_thread("user@test.com", "Hermes — Report"))
                _, root_msg = self._send(adapter, "Subject: Report\n\n- item",
                                         metadata={"thread_id": key, "job_id": "j1"})
                references = " ".join(filter(None, [root_msg["References"], self.REWRITTEN]))
                self.assertEqual(references, f"<{key}> {self.REWRITTEN}")
                event = self._dispatch(adapter, subject="Re: Report", in_reply_to=self.REWRITTEN,
                                       references=references, message_id=f"<r{n}-1@test.com>")
                self.assertEqual(event.source.thread_id, key)
                direct = self._dispatch(adapter, subject="Re: Report", in_reply_to=f"<{key}>",
                                        references=f"<{key}>", message_id=f"<r{n}-2@test.com>")
                self.assertEqual(self._key(event.source), self._key(direct.source))

    def test_reply_without_minted_reference_gets_own_root(self):
        from plugins.platforms.email.adapter import _root_key

        adapter = make_adapter(thread_sessions=True)
        key = self._mint_and_send(adapter)
        event = self._dispatch(adapter, subject="Re: Report", in_reply_to=self.REWRITTEN, references=self.REWRITTEN,
                               message_id="<r1@test.com>")
        self.assertNotEqual(event.source.thread_id, key)
        self.assertEqual(event.source.thread_id, _root_key(self.REWRITTEN))

    def test_reply_to_hermes_reply_resolves_via_index(self):
        adapter = make_adapter()
        key = self._mint_and_send(adapter)
        result, _ = self._send(adapter, "- part two", metadata={"thread_id": key})
        hermes_reply = result.message_id
        event = self._dispatch(adapter, subject="Re: Report", in_reply_to=hermes_reply,
                               references=f"<other@elsewhere> {hermes_reply}", message_id="<r1@test.com>")
        self.assertEqual(event.source.thread_id, key)

    def test_seed_and_reply_share_session_key(self):
        from cron.scheduler_delivery import _seed_cron_thread_session

        adapter = make_adapter()
        key = self._mint_and_send(adapter)
        seed_adapter, store = MagicMock(), MagicMock()
        seed_adapter._session_store = store
        with patch("gateway.mirror.mirror_to_session", return_value=True):
            _seed_cron_thread_session({"id": "j1", "name": "Report"}, seed_adapter, "email", "user@test.com", key,
                                      "report")
        store.get_or_create_session.assert_called_once()
        seed_key = self._key(store.get_or_create_session.call_args[0][0])

        first = self._dispatch(adapter, subject="Re: Report", in_reply_to=f"<{key}>", references=f"<{key}>",
                               message_id="<r1@test.com>")
        self.assertEqual(self._key(first.source), seed_key)
        answer, _ = self._send(adapter, "answer", reply_to="<r1@test.com>", metadata={"thread_id": key})
        second = self._dispatch(adapter, subject="Re: Report", in_reply_to=answer.message_id,
                                references=f"<{key}> <r1@test.com> {answer.message_id}", message_id="<r2@test.com>")
        self.assertEqual(self._key(second.source), seed_key)
        # Restart: a new adapter in the same HERMES_HOME knows the thread from the persisted store alone.
        restarted = make_adapter()
        third = self._dispatch(restarted, subject="Re: Report", in_reply_to="<r2@test.com>",
                               references="<r2@test.com>", message_id="<r3@test.com>")
        self.assertEqual(self._key(third.source), seed_key)

    def test_unknown_root_gets_own_session_with_subject(self):
        adapter = make_adapter(thread_sessions=True)
        topic = "Daily summary (thread <foreign-root@mail.example.com>)"
        first = self._dispatch(adapter, subject="Re: Fwd: Daily summary", references=self.FOREIGN_REFS,
                               in_reply_to="<foreign-2@mail.example.com>", message_id="<u1@test.com>")
        self.assertEqual(first.source.thread_id, "foreign-root@mail.example.com")
        self.assertEqual(first.source.chat_topic, topic)
        self.assertEqual(adapter._store.get_root("foreign-root@mail.example.com")["subject"], "Daily summary")
        second = self._dispatch(adapter, subject="Re: something else", references=self.FOREIGN_REFS,
                                message_id="<u2@test.com>")
        self.assertEqual(second.source.thread_id, "foreign-root@mail.example.com")
        self.assertEqual(second.source.chat_topic, topic)

    def test_own_message_id_session_for_new_mail(self):
        adapter = make_adapter(thread_sessions=True)
        event = self._dispatch(adapter, subject="Plan for Friday", message_id="<new-1@test.com>")
        self.assertEqual(event.source.thread_id, "new-1@test.com")
        self.assertTrue(event.source.chat_topic.startswith("Plan for Friday"))

    def test_flag_off_keeps_flat_session(self):
        adapter = make_adapter()
        event = self._dispatch(adapter, subject="Re: Fwd: Daily summary", references=self.FOREIGN_REFS,
                               in_reply_to="<foreign-2@mail.example.com>", message_id="<u1@test.com>")
        self.assertIsNone(event.source.thread_id)
        self.assertIsNone(event.source.chat_topic)
        self.assertTrue(self._key(event.source).endswith(":email:dm:user@test.com"))
        self.assertIsNone(adapter._store.get_root("foreign-root@mail.example.com"))

    def test_hermes_root_routes_with_flag_off(self):
        adapter = make_adapter()
        key = self._mint_and_send(adapter)
        event = self._dispatch(adapter, subject="Re: Report", in_reply_to=f"<{key}>", message_id="<r1@test.com>")
        self.assertEqual(event.source.thread_id, key)
        self.assertEqual(event.source.chat_topic, f"Report (thread <{key}>)")

    def test_thread_from_other_sender_gets_different_key(self):
        adapter = make_adapter()
        key = self._mint_and_send(adapter)
        own = self._dispatch(adapter, subject="Re: Report", references=f"<{key}>", message_id="<r1@test.com>")
        other = self._dispatch(adapter, sender="Other <other@test.com>", subject="Re: Report",
                               references=f"<{key}>", message_id="<o1@test.com>")
        self.assertEqual(other.source.thread_id, key)
        self.assertNotEqual(self._key(other.source), self._key(own.source))
        self.assertIn(":dm:other@test.com:", self._key(other.source))

    def test_long_root_id_capped_in_topic(self):
        adapter = make_adapter(thread_sessions=True)
        long_id = "<" + "x" * 900 + "@test.com>"
        event = self._dispatch(adapter, subject="Re: Report", references=long_id, message_id="<l1@test.com>")
        self.assertTrue(event.source.thread_id.startswith("h-"))
        self.assertLess(len(event.source.chat_topic), 260)
        self.assertTrue(event.source.chat_topic.startswith("Report (thread <xxx"))

    def test_crafted_message_id_cannot_escape_session_key(self):
        from gateway.session import _is_path_unsafe

        adapter = make_adapter(thread_sessions=True)
        event = self._dispatch(adapter, subject="Re: x", references="<../../etc/passwd@x>", message_id="<c1@test.com>")
        self.assertTrue(event.source.thread_id.startswith("h-"))
        self.assertFalse(_is_path_unsafe(self._key(event.source), strict=False))

    def test_sender_case_normalised(self):
        adapter = make_adapter()
        event = self._dispatch(adapter, sender="User <USER@Test.COM>")
        self.assertEqual(event.source.chat_id, "user@test.com")

    def test_reply_in_thread_chains_references_from_inbound(self):
        adapter = make_adapter()
        key = self._mint_and_send(adapter)
        self._dispatch(adapter, subject="Re: Report", in_reply_to=f"<{key}>", references=f"<{key}>",
                       message_id="<r1@test.com>")
        _, msg = self._send(adapter, "answer", reply_to="<r1@test.com>", metadata={"thread_id": key})
        self.assertEqual(msg["References"], f"<{key}> <r1@test.com>")
        self.assertEqual(msg["In-Reply-To"], "<r1@test.com>")
        self.assertEqual(msg["Subject"], "Re: Report")

    def test_rejected_sender_never_touches_store(self):
        adapter = make_adapter(thread_sessions=True)
        key = self._mint_and_send(adapter)
        index_before, file_before = dict(adapter._store._index), _store_path().read_text(encoding="utf-8")
        with patch.dict(os.environ, {}):
            for name in ("EMAIL_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "EMAIL_ALLOWED_USERS",
                         "GATEWAY_ALLOWED_USERS"):
                os.environ.pop(name, None)
            event = self._dispatch(adapter, sender="Stranger <stranger@elsewhere.com>", subject="Re: Report",
                                   in_reply_to=f"<{key}>", references=f"<{key}>", message_id="<s1@elsewhere.com>")
        self.assertIsNone(event)
        self.assertEqual(adapter._store._index, index_before)
        self.assertEqual(_store_path().read_text(encoding="utf-8"), file_before)
        self.assertEqual(adapter._root_context, {})

    def test_answered_stranger_never_touches_store(self):
        """With pair or decline the gateway still gets a stranger's mail so it can answer, but the mail is not
        threaded: no root, no index entry, no thread context. A granted sender still threads."""
        for behavior in ("pair", "decline"):
            with self.subTest(behavior=behavior):
                adapter = make_adapter(thread_sessions=True, unauthorized_dm_behavior=behavior)
                key = self._mint_and_send(adapter)
                index_before, file_before = dict(adapter._store._index), _store_path().read_text(encoding="utf-8")
                with patch.dict(os.environ, {}):
                    for name in ("EMAIL_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "EMAIL_ALLOWED_USERS",
                                 "GATEWAY_ALLOWED_USERS"):
                        os.environ.pop(name, None)
                    with patch.object(adapter, "_is_sender_authorized", return_value=False):
                        new = self._dispatch(adapter, sender="Stranger <stranger@elsewhere.com>", subject="Hello",
                                             message_id="<s1@elsewhere.com>")
                        reply = self._dispatch(adapter, sender="Stranger <stranger@elsewhere.com>",
                                               subject="Re: Report", in_reply_to=f"<{key}>", references=f"<{key}>",
                                               message_id="<s2@elsewhere.com>")
                    self.assertIsNotNone(new)
                    self.assertIsNotNone(reply)
                    self.assertIsNone(new.source.thread_id)
                    self.assertIsNone(reply.source.thread_id)
                    self.assertEqual(adapter._store._index, index_before)
                    self.assertEqual(_store_path().read_text(encoding="utf-8"), file_before)
                    self.assertEqual(adapter._root_context, {})
                    with patch.object(adapter, "_is_sender_authorized", return_value=True):
                        granted = self._dispatch(adapter, sender="Paired <paired@elsewhere.com>",
                                                 subject="Re: Report", references=f"<{key}>",
                                                 message_id="<p1@elsewhere.com>")
                    self.assertEqual(granted.source.thread_id, key)

    def test_inbound_message_id_never_repoints_an_indexed_id(self):
        """An inbound mail whose Message-ID is already indexed (a root's own id, or an earlier mail's) leaves that
        entry alone, so later replies keep reaching the original thread."""
        adapter = make_adapter(thread_sessions=True)
        key = self._mint_and_send(adapter)
        other = self._dispatch(adapter, subject="Re: other", references="<other-root@test.com>",
                               message_id=f"<{key}>")
        self.assertEqual(other.source.thread_id, "other-root@test.com")
        self.assertEqual(adapter._store.root_for([f"<{key}>"]), key)
        self._dispatch(adapter, subject="Re: Report", references=f"<{key}>", message_id="<r1@test.com>")
        self._dispatch(adapter, subject="Re: other", references="<other-root@test.com>", message_id="<r1@test.com>")
        self._dispatch(adapter, subject="Fwd: Report", message_id="<r1@test.com>")  # a resent copy, no References
        self.assertEqual(adapter._store.root_for(["<r1@test.com>"]), key)
        reply = self._dispatch(adapter, subject="Re: Report", references=f"<{key}>", message_id="<r2@test.com>")
        self.assertEqual(reply.source.thread_id, key)
        self.assertEqual(make_adapter()._store.root_for([f"<{key}>", "<r1@test.com>"]), key)


class TestColdStart(DispatchCase):
    """MAIL-05: mail that arrived while the gateway was down is handled after a restart.

    A restart is a new adapter instance with the class-level seen-UID snapshot cleared and the same
    HERMES_HOME, so the only thing it carries over is the thread store on disk.
    """

    ADDRESS = "hermes@test.com"

    def _connect(self, adapter, imap, *, is_reconnect=False):
        # The poll loop is stubbed: each test drives _check_inbox itself, so connect alone never consumes mail.
        with patch.object(adapter, "_connect_imap", return_value=imap), patch.object(
            adapter, "_connect_smtp", return_value=MagicMock()
        ), patch.object(adapter, "_poll_loop", new=AsyncMock()):
            self.assertTrue(asyncio.run(adapter.connect(is_reconnect=is_reconnect)))
        asyncio.run(adapter.disconnect())

    def _poll(self, adapter, imap):
        events = self.capture(adapter)
        with patch.object(adapter, "_connect_imap", return_value=imap):
            asyncio.run(adapter._check_inbox())
        return events

    @staticmethod
    def _restart():
        from plugins.platforms.email.adapter import EmailAdapter

        EmailAdapter._seen_uids_snapshot.clear()
        return make_adapter()

    def _seed(self, uidvalidity, uid_mark):
        from plugins.platforms.email.adapter import _EmailThreadStore

        _EmailThreadStore(_store_path()).set_uid_state(self.ADDRESS, uidvalidity, uid_mark)

    def _state(self):
        """The UID state as a fresh process would read it from disk."""
        from plugins.platforms.email.adapter import _EmailThreadStore

        return _EmailThreadStore(_store_path()).uid_state(self.ADDRESS)

    @staticmethod
    def _mail(n, sender="User <user@test.com>"):
        return make_inbound(sender=sender, subject=f"Mail {n}", body=f"body {n}", message_id=f"<m{n}@test.com>")

    @staticmethod
    def _bodies(events):
        return [event.text.rsplit("\n", 1)[-1] for event in events]

    def test_no_mark_marks_all_seen(self):
        adapter = make_adapter()
        self._connect(adapter, fake_imap(uids_all=b"1 2 3"))
        self.assertEqual(adapter._seen_uids, {b"1", b"2", b"3"})
        self.assertEqual(self._state(), ("42", 3))

    def test_empty_mailbox_sets_zero_mark(self):
        self._connect(make_adapter(), fake_imap(uids_all=b""))
        self.assertEqual(self._state(), ("42", 0))
        # The first mail arrives while the gateway is down: the next start still handles it.
        adapter = self._restart()
        imap = fake_imap(uids_all=b"1", uids_unseen=b"1", messages={b"1": self._mail(1)})
        self._connect(adapter, imap)
        self.assertEqual(adapter._seen_uids, set())
        self.assertEqual(self._bodies(self._poll(adapter, imap)), ["body 1"])
        self.assertEqual(self._state(), ("42", 1))

    def test_mail_after_mark_dispatched_after_restart(self):
        first = make_adapter()
        self._connect(first, fake_imap(uids_all=b"1 2 3"))
        self.assertEqual(self._state(), ("42", 3))
        # Gateway down; mail 4 and 5 arrive. A new process starts with the same HERMES_HOME.
        second = self._restart()
        imap = fake_imap(uids_all=b"1 2 3 4 5", uids_unseen=b"4 5",
                         messages={b"4": self._mail(4), b"5": self._mail(5)})
        self._connect(second, imap)
        self.assertEqual(second._seen_uids, {b"1", b"2", b"3"})
        events = self._poll(second, imap)
        self.assertEqual(self._bodies(events), ["body 4", "body 5"])
        self.assertEqual(self._state(), ("42", 5))

    def test_uidvalidity_change_resets_mark(self):
        self._seed("42", 3)
        adapter = self._restart()
        self._connect(adapter, fake_imap(uids_all=b"1 2 3 4 5", uidvalidity=b"99"))
        self.assertEqual(adapter._seen_uids, {b"1", b"2", b"3", b"4", b"5"})
        self.assertEqual(self._state(), ("99", 5))

    def test_missing_uidvalidity_means_no_mark(self):
        self._seed("42", 3)
        adapter = self._restart()
        self._connect(adapter, fake_imap(uids_all=b"1 2 3 4 5", uidvalidity=None))
        self.assertEqual(adapter._seen_uids, {b"1", b"2", b"3", b"4", b"5"})
        self.assertEqual(self._state(), ("42", 3))

    def test_mark_advances_after_dispatch(self):
        self._seed("42", 3)
        adapter = self._restart()
        imap = fake_imap(uids_all=b"1 2 3 4 5", uids_unseen=b"4 5",
                         messages={b"4": self._mail(4), b"5": self._mail(5)})
        self._connect(adapter, imap)
        marks_seen_by_handler = []

        async def _record(event):
            marks_seen_by_handler.append((event.text.rsplit("\n", 1)[-1], self._state()[1]))

        adapter.handle_message = _record
        with patch.object(adapter, "_connect_imap", return_value=imap):
            asyncio.run(adapter._check_inbox())
        # Mail 5 is handled after mail 4's dispatch moved the mark, not only at the end of the batch.
        self.assertEqual(marks_seen_by_handler, [("body 4", 3), ("body 5", 4)])
        self.assertEqual(self._state(), ("42", 5))

    def test_eight_bit_thread_headers_do_not_drop_the_batch(self):
        """Raw 8-bit bytes in Message-ID / References / In-Reply-To parse to ``Header`` objects (compat32); the
        mail is still handled, and the next mail in the same poll too."""
        self._seed("42", 3)
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        EmailAdapter._seen_uids_snapshot.clear()
        with patch.dict(os.environ, ENV):
            adapter = EmailAdapter(PlatformConfig(enabled=True, extra={"thread_sessions": True}))
        eight_bit = (b"From: User <user@test.com>\r\nTo: hermes@test.com\r\nSubject: Mail 4\r\n"
                     b"Message-ID: <m4\xe9@test.com>\r\nIn-Reply-To: <r\xe9@test.com>\r\n"
                     b"References: <root@test.com> <r\xe9@test.com>\r\n"
                     b"Content-Type: text/plain; charset=utf-8\r\n\r\nbody 4\r\n")
        imap = fake_imap(uids_all=b"1 2 3 4 5", uids_unseen=b"4 5", messages={b"4": eight_bit, b"5": self._mail(5)})
        self._connect(adapter, imap)
        events = self._poll(adapter, imap)
        self.assertEqual(self._bodies(events), ["body 4", "body 5"])
        self.assertTrue(all(isinstance(e.message_id, str) for e in events))
        self.assertEqual(self._state(), ("42", 5))

    def test_one_failing_dispatch_does_not_drop_the_batch(self):
        self._seed("42", 3)
        adapter = self._restart()
        imap = fake_imap(uids_all=b"1 2 3 4 5", uids_unseen=b"4 5",
                         messages={b"4": self._mail(4), b"5": self._mail(5)})
        self._connect(adapter, imap)
        real_dispatch, handled = adapter._dispatch_message, []

        async def _dispatch(msg_data):
            if msg_data["uid"] == b"4":
                raise TypeError("poison")
            handled.append(msg_data["uid"])
            await real_dispatch(msg_data)

        with patch.object(adapter, "_dispatch_message", side_effect=_dispatch), \
                self.assertLogs("plugins.platforms.email.adapter", level="ERROR"):
            events = self._poll(adapter, imap)
        self.assertEqual((handled, self._bodies(events)), ([b"5"], ["body 5"]))
        self.assertEqual(self._state(), ("42", 5))

    def test_filtered_uid_still_advances_mark(self):
        self._seed("42", 3)
        adapter = self._restart()
        robot = "Robot <noreply@test.com>"
        messages = {b"4": self._mail(4, sender=robot), b"5": self._mail(5), b"6": self._mail(6, sender=robot)}
        imap = fake_imap(uids_all=b"1 2 3 4 5 6", uids_unseen=b"4 5 6", messages=messages)
        self._connect(adapter, imap)
        self.assertEqual(self._bodies(self._poll(adapter, imap)), ["body 5"])
        # 6 was dropped by the automated-sender filter and is the batch's last UID: the mark still covers it.
        self.assertEqual(self._state(), ("42", 6))
        later = self._restart()
        self._connect(later, imap)
        self.assertEqual(later._seen_uids, {b"1", b"2", b"3", b"4", b"5", b"6"})
        self.assertEqual(self._poll(later, imap), [])

    def test_reconnect_snapshot_path_unchanged(self):
        first = make_adapter()
        self._connect(first, fake_imap(uids_all=b"1 2 3"))
        file_before = _store_path().read_bytes()
        # Same-process reconnect: the snapshot wins, even against a UIDVALIDITY that would reset the mark.
        second = make_adapter()
        imap = fake_imap(uids_all=b"1 2 3 4", uidvalidity=b"99")
        self._connect(second, imap, is_reconnect=True)
        self.assertEqual(second._seen_uids, {b"1", b"2", b"3"})
        imap.response.assert_not_called()
        self.assertEqual(_store_path().read_bytes(), file_before)
        self.assertEqual(self._state(), ("42", 3))


def fake_folders(folders, *, inbox_unseen=None, list_lines=None):
    """A MagicMock IMAP connection with several folders.

    *folders* maps a mailbox name to ``{"uidvalidity": bytes, "messages": {uid bytes: raw bytes}}``; ``select``
    switches the folder that ``uid("search"|"fetch", ...)`` answers for. ``uid("search", None, "UNSEEN")``
    returns *inbox_unseen* (INBOX only), ``"ALL"`` every UID, ``"UID n:*"`` the UIDs from n (or the last UID).
    A fetch takes a comma-separated UID set and returns the stored bytes. ``list()`` returns *list_lines*
    (default: one line per folder, ``\\Sent`` on the one named ``Sent``). ``imap.selected`` records each
    selected name in order.
    """
    imap = MagicMock()
    imap.selected = []
    state = {"box": None}

    def unquote(name):
        name = str(name)
        return name[1:-1].replace('\\"', '"').replace("\\\\", "\\") if name.startswith('"') else name

    def select(name="INBOX", readonly=False):
        name = unquote(name)
        imap.selected.append(name)
        if name not in folders:
            state["box"] = None
            return ("NO", [b"no such mailbox"])
        state["box"] = name
        return ("OK", [str(len(folders[name].get("messages", {}))).encode()])

    def uid_handler(command, *args):
        box = folders.get(state["box"])
        if box is None:
            return ("NO", [])
        messages = box.get("messages", {})
        uids = sorted(messages, key=int)
        command = str(command).lower()
        if command == "search":
            criteria = " ".join(str(a) for a in args if a is not None).upper()
            if "UNSEEN" in criteria:
                return ("OK", [inbox_unseen if state["box"] == "INBOX" and inbox_unseen is not None else b""])
            if criteria.startswith("UID "):
                low = int(criteria.split()[1].split(":")[0])
                hits = [u for u in uids if int(u) >= low] or uids[-1:]
                return ("OK", [b" ".join(hits)])
            if "ALL" in criteria:
                return ("OK", [b" ".join(uids)])
            return ("NO", [])
        if command == "fetch" and args:
            wanted = [_as_bytes(u) for u in _as_bytes(args[0]).split(b",")]
            data = []
            for uid in wanted:
                if uid in messages:
                    data.append((b"%s (UID %s BODY[HEADER] {1}" % (uid, uid), messages[uid]))
                    data.append(b")")
            return ("OK", data) if data else ("NO", [])
        return ("NO", [])

    imap.select.side_effect = select
    imap.uid.side_effect = uid_handler
    imap.response.side_effect = lambda name: (name, [folders.get(state["box"], {}).get("uidvalidity", b"7")])
    if list_lines is None:
        list_lines = [(b'(\\HasNoChildren \\Sent) "/" "%s"' if name == "Sent" else b'(\\HasNoChildren) "/" "%s"')
                      % name.encode() for name in folders]
    imap.list.return_value = ("OK", list_lines)
    imap.login.return_value = ("OK", [b""])
    imap.logout.return_value = ("OK", [b""])
    return imap


def delivered_copy(sent, message_id):
    """The header bytes a provider files in the Sent folder for *sent*: its own Message-ID replaced by
    *message_id*, In-Reply-To and References as the adapter wrote them."""
    copy = MIMEText("", "plain", "utf-8")
    copy["From"] = sent["From"]
    copy["To"] = sent["To"]
    copy["Subject"] = sent["Subject"]
    copy["Message-ID"] = message_id
    for name in ("In-Reply-To", "References"):
        if sent[name] is not None:
            copy[name] = sent[name]
    return copy.as_bytes()


class TestDeliveredMessageIds(DispatchCase):
    """A provider that replaces the Message-ID of every mail Hermes sends (Amazon SES and services built on it)
    still keeps the thread: the adapter reads each delivered id back from the Sent folder and indexes it to the
    thread the mail was sent in, so a reply that names only that id (In-Reply-To, short References) routes."""

    def _send(self, adapter, content, **kwargs):
        with patch("smtplib.SMTP") as mock_smtp:
            result = asyncio.run(adapter.send("user@test.com", content, **kwargs))
            msg = sent_message(mock_smtp)
        return result, msg

    def _dispatch(self, adapter, **inbound):
        events = self.capture(adapter)
        asyncio.run(adapter._dispatch_message(adapter._parse_fetched_message(b"1", make_inbound(**inbound))))
        return events[-1] if events else None

    def _poll(self, adapter, imap):
        events = self.capture(adapter)
        with patch.object(adapter, "_connect_imap", return_value=imap):
            asyncio.run(adapter._check_inbox())
        return events

    def _thread(self, adapter, tag=""):
        """A minted thread with its root mail and one answer to a reply, as the provider delivered them.
        Returns ``(key, root copy id, answer copy id, sent folder messages)``."""
        key = asyncio.run(adapter.create_handoff_thread("user@test.com", "Hermes — Report"))
        _, root_msg = self._send(adapter, "Subject: Report\n\n- item", metadata={"thread_id": key, "job_id": "j1"})
        root_copy = f"<0100-root{tag}-000000@email.example.com>"
        self._dispatch(adapter, subject="Re: Report", in_reply_to=root_copy, references=f"<{key}> {root_copy}",
                       message_id=f"<r1{tag}@test.com>")
        _, answer = self._send(adapter, "answer", reply_to=f"<r1{tag}@test.com>", metadata={"thread_id": key})
        answer_copy = f"<0100-answer{tag}-000000@email.example.com>"
        sent = {b"7": delivered_copy(root_msg, root_copy), b"8": delivered_copy(answer, answer_copy)}
        return key, root_copy, answer_copy, sent

    def test_poll_learns_delivered_ids_of_sent_mail(self):
        adapter = make_adapter()
        key, root_copy, answer_copy, sent = self._thread(adapter)
        self.assertIsNone(adapter._store.root_for([answer_copy]))
        imap = fake_folders({"INBOX": {}, "Sent": {"messages": sent}})
        self._poll(adapter, imap)
        self.assertEqual(adapter._store.root_for([answer_copy]), key)
        self.assertEqual(adapter._store.root_for([root_copy]), key)
        # Read-only: the Sent folder is examined, then INBOX is selected again for the UNSEEN search.
        imap.select.assert_any_call('"Sent"', readonly=True)
        self.assertEqual(imap.selected[-1], "INBOX")
        on_disk = json.loads(_store_path().read_text(encoding="utf-8"))
        self.assertEqual(on_disk["index"][answer_copy.strip("<>")], key)

    def test_reply_naming_only_delivered_id_routes_to_thread(self):
        for tag, extra in (("a", {}), ("b", {"thread_sessions": True})):
            with self.subTest(extra=extra):
                adapter = make_adapter(**extra)
                key, _, answer_copy, sent = self._thread(adapter, tag)
                seed = self._dispatch(adapter, subject="Re: Report", in_reply_to=f"<{key}>", references=f"<{key}>",
                                      message_id=f"<seed{tag}@test.com>")
                self._poll(adapter, fake_folders({"INBOX": {}, "Sent": {"messages": sent}}))
                for n, headers in enumerate(({"in_reply_to": answer_copy},
                                             {"in_reply_to": answer_copy, "references": answer_copy})):
                    event = self._dispatch(adapter, subject="Re: Re: Report", message_id=f"<short{tag}{n}@test.com>",
                                           **headers)
                    self.assertEqual(event.source.thread_id, key)
                    self.assertEqual(build_key(event.source), build_key(seed.source))

    def test_same_poll_reply_routes(self):
        # The Sent copy and the reply to it are both new in one poll: the Sent folder is read first.
        adapter = make_adapter()
        key, _, answer_copy, sent = self._thread(adapter)
        reply = make_inbound(subject="Re: Report", in_reply_to=answer_copy, references=answer_copy,
                             message_id="<late@test.com>")
        imap = fake_folders({"INBOX": {"messages": {b"3": reply}}, "Sent": {"messages": sent}}, inbox_unseen=b"3")
        events = self._poll(adapter, imap)
        self.assertEqual([e.source.thread_id for e in events], [key])

    def test_startup_backfill_is_bounded_and_idempotent(self):
        from plugins.platforms.email import adapter as mod

        first = make_adapter()
        key, _, answer_copy, sent = self._thread(first)
        older = {str(n).encode(): delivered_copy(MIMEText("x"), f"<old-{n}@email.example.com>") for n in range(1, 7)}
        # A restart: the new adapter has never seen the Sent folder and reads back its newest copies.
        restarted = make_adapter()
        imap = fake_folders({"INBOX": {}, "Sent": {"messages": {**older, **sent}}})
        with patch.object(mod, "_SENT_BACKFILL_MAX", 3):
            self._poll(restarted, imap)
        fetched = [c.args[1] for c in imap.uid.call_args_list if str(c.args[0]).lower() == "fetch"]
        self.assertEqual(fetched, ["6,7,8"])
        self.assertEqual(restarted._store.root_for([answer_copy]), key)
        before = _store_path().read_bytes()
        # The next polls only look at UIDs above the last one read, and learning again changes nothing.
        imap.uid.reset_mock()
        self._poll(restarted, imap)
        self.assertFalse([c for c in imap.uid.call_args_list if str(c.args[0]).lower() == "fetch"])
        self.assertEqual(_store_path().read_bytes(), before)
        again = make_adapter()
        self._poll(again, fake_folders({"INBOX": {}, "Sent": {"messages": sent}}))
        self.assertEqual(_store_path().read_bytes(), before)

    def test_backfill_learns_mail_sent_outside_a_thread_send(self):
        # A mail sent without the thread id (e.g. a document) that replies inside a known thread joins that thread.
        adapter = make_adapter()
        key, _, _, sent = self._thread(adapter)
        with patch("smtplib.SMTP") as mock_smtp:
            asyncio.run(adapter._dispatch_message(adapter._parse_fetched_message(b"2", make_inbound(
                subject="Re: Report", in_reply_to="<r1@test.com>", references=f"<{key}> <r1@test.com>",
                message_id="<r2@test.com>"))))
            adapter._send_email("user@test.com", "file follows")
            doc = sent_message(mock_smtp)
        doc_copy = "<0100-doc-000000@email.example.com>"
        sent[b"9"] = delivered_copy(doc, doc_copy)
        self._poll(adapter, fake_folders({"INBOX": {}, "Sent": {"messages": sent}}))
        self.assertEqual(adapter._store.root_for([doc_copy]), key)

    def test_learned_id_wins_over_root_minted_from_it(self):
        from plugins.platforms.email.adapter import _root_key

        adapter = make_adapter(thread_sessions=True)
        key, _, answer_copy, sent = self._thread(adapter)
        # Before the id is known, a short reply opens its own root keyed by that id (the fork).
        fork = self._dispatch(adapter, subject="Re: Report", in_reply_to=answer_copy, references=answer_copy,
                              message_id="<f1@test.com>")
        fork_key = _root_key(answer_copy)
        self.assertEqual(fork.source.thread_id, fork_key)
        with self.assertLogs("plugins.platforms.email.adapter", level="INFO") as logs:
            self._poll(adapter, fake_folders({"INBOX": {}, "Sent": {"messages": sent}}))
        self.assertTrue(any(fork_key in line and key in line for line in logs.output), logs.output)
        self.assertEqual(adapter._store.root_for([answer_copy]), key)
        later = self._dispatch(adapter, subject="Re: Report", in_reply_to=answer_copy, references=answer_copy,
                               message_id="<f2@test.com>")
        self.assertEqual(later.source.thread_id, key)
        # The fork's root and its own mail stay: nothing is deleted.
        self.assertIsNotNone(adapter._store.get_root(fork_key))
        self.assertEqual(adapter._store.root_for(["<f1@test.com>"]), fork_key)

    def test_send_record_beats_sent_folder(self):
        # An id the adapter itself indexed when sending (a provider that keeps Message-IDs) is not re-pointed.
        adapter = make_adapter()
        _, _, _, sent = self._thread(adapter)
        other = asyncio.run(adapter.create_handoff_thread("user@test.com", "Hermes — Other"))
        self._send(adapter, "Subject: Other\n\nx", metadata={"thread_id": other})
        other_reply, _ = self._send(adapter, "more", metadata={"thread_id": other})
        kept = other_reply.message_id
        self.assertEqual(adapter._store.root_for([kept]), other)
        # The copy keeps its Message-ID but claims to reply into the first thread.
        copy = MIMEText("")
        copy["Message-ID"], copy["In-Reply-To"] = kept, "<r1@test.com>"
        sent[b"9"] = copy.as_bytes()
        self._poll(adapter, fake_folders({"INBOX": {}, "Sent": {"messages": sent}}))
        self.assertEqual(adapter._store.root_for([kept]), other)

    def test_in_reply_to_decides_before_references(self):
        store = make_adapter()._store
        store.add_pending_root("a", "<a>", "A", "user@test.com")
        store.add_pending_root("b", "<b>", "B", "user@test.com")
        store.index_message("<in-b@test.com>", "b")
        copies = [("<d1@x>", ["<in-b@test.com>"], ["<a>", "<in-b@test.com>"]),
                  ("<d2@x>", [], ["<a>", "<in-b@test.com>"]),
                  ("<d3@x>", ["<unknown@x>"], []),
                  ("<a>", [], ["<a>"])]
        self.assertEqual(store.learn_delivered_ids(copies), 2)
        self.assertEqual(store.root_for(["<d1@x>"]), "b")
        self.assertEqual(store.root_for(["<d2@x>"]), "a")
        self.assertIsNone(store.root_for(["<d3@x>"]))
        self.assertEqual(store.learn_delivered_ids(copies), 0)

    def test_no_thread_roots_no_sent_folder_traffic(self):
        adapter = make_adapter()
        imap = fake_folders({"INBOX": {}, "Sent": {"messages": {}}})
        self._poll(adapter, imap)
        imap.list.assert_not_called()
        self.assertEqual(imap.selected, ["INBOX"])

    def test_sent_folder_found_by_name_without_special_use(self):
        adapter = make_adapter()
        key, _, answer_copy, sent = self._thread(adapter)
        imap = fake_folders({"INBOX": {}, "Sent Items": {"messages": sent}},
                            list_lines=[b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Sent Items"'])
        self._poll(adapter, imap)
        imap.select.assert_any_call('"Sent Items"', readonly=True)
        self.assertEqual(adapter._store.root_for([answer_copy]), key)

    def test_sent_folder_setting(self):
        for tag, (setting, expected) in enumerate((("Archive/Out", "Archive/Out"), (False, None), ("off", None))):
            with self.subTest(setting=setting):
                adapter = make_adapter(sent_folder=setting)
                key, _, answer_copy, sent = self._thread(adapter, str(tag))
                imap = fake_folders({"INBOX": {}, "Sent": {"messages": sent}, "Archive/Out": {"messages": sent}})
                self._poll(adapter, imap)
                if expected is None:
                    imap.list.assert_not_called()
                    self.assertEqual(imap.selected, ["INBOX"])
                else:
                    self.assertIn(expected, imap.selected)
                    self.assertEqual(adapter._store.root_for([answer_copy]), key)

    def test_sent_folder_inbox_refused(self):
        for setting in ("INBOX", " inbox "):
            with self.subTest(setting=setting), self.assertLogs("plugins.platforms.email.adapter", "WARNING"):
                adapter = make_adapter(sent_folder=setting)
            self._thread(adapter)
            imap = fake_folders({"INBOX": {}, "Sent": {"messages": {}}})
            self._poll(adapter, imap)
            self.assertEqual(imap.selected, ["INBOX"])
            imap.list.assert_not_called()

    def test_unselectable_sent_folder_warns_once(self):
        adapter = make_adapter(sent_folder="Missing")
        self._thread(adapter)
        imap = fake_folders({"INBOX": {}})
        with self.assertLogs("plugins.platforms.email.adapter", "WARNING") as logs:
            self._poll(adapter, imap)
            self._poll(adapter, imap)
        self.assertEqual(sum("Missing" in line for line in logs.output), 1, logs.output)
        self.assertEqual(imap.selected.count("Missing"), 2)

    def test_no_sent_folder_is_quiet_and_not_retried(self):
        adapter = make_adapter()
        self._thread(adapter)
        imap = fake_folders({"INBOX": {}}, list_lines=[b'(\\HasNoChildren) "/" "INBOX"'])
        self._poll(adapter, imap)
        self._poll(adapter, imap)
        self.assertEqual(imap.list.call_count, 1)
        self.assertFalse(adapter._last_fetch_failed)

    def test_sent_folder_failure_never_blocks_inbox(self):
        adapter = make_adapter()
        self._thread(adapter)
        mail = make_inbound(subject="Hello", message_id="<h1@test.com>")
        imap = fake_folders({"INBOX": {"messages": {b"3": mail}}, "Sent": {"messages": {}}}, inbox_unseen=b"3")
        imap.list.side_effect = OSError("boom")
        events = self._poll(adapter, imap)
        self.assertEqual(len(events), 1)
        self.assertFalse(adapter._last_fetch_failed)
        self.assertEqual(imap.selected[-1], "INBOX")


def build_key(source):
    from gateway.session import build_session_key

    return build_session_key(source)


if __name__ == "__main__":
    unittest.main()
