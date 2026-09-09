"""Deterministic in-process IMAP/SMTP protocol fakes for email tests.

Stateful stand-ins for ``imaplib.IMAP4/IMAP4_SSL`` and ``smtplib.SMTP/
SMTP_SSL``: real RFC822 bytes flow through ``append``/``send_message`` into
an in-memory store the test can assert on, and every command is recorded so
sequence regressions (e.g. an unsupported command desyncing the session)
are observable. Connections are cheap objects bound to a shared
``FakeMailStore`` so reconnect-per-operation adapters keep one mailbox
state.

Usage::

    store = FakeMailStore()
    store.add_inbox_message(raw_bytes)
    with patch("imaplib.IMAP4_SSL", store.imap_factory), \
         patch("smtplib.SMTP", store.smtp_factory):
        ...

"""

from typing import Any, Dict, List, Optional, Tuple


class FakeMailStore:
    """Shared mailbox + wire state behind any number of fake connections."""

    def __init__(self, capabilities: Tuple[str, ...] = ("IMAP4REV1", "UIDPLUS")):
        self.capabilities = capabilities
        # mailbox name -> list of (flags, raw_bytes)
        self.mailboxes: Dict[str, List[Tuple[str, bytes]]] = {"INBOX": []}
        self.smtp_messages: List[Any] = []  # Message objects passed to send_message
        self.command_log: List[Tuple[str, tuple]] = []
        self.fail_append_status: Optional[str] = None  # e.g. "NO" to fail appends
        self.append_exception: Optional[Exception] = None
        self._next_uid = 1
        self._inbox_uids: Dict[int, bytes] = {}

    def add_inbox_message(self, raw: bytes) -> int:
        uid = self._next_uid
        self._next_uid += 1
        self._inbox_uids[uid] = raw
        self.mailboxes["INBOX"].append(("", raw))
        return uid

    def imap_factory(self, *args, **kwargs) -> "FakeIMAP4":
        return FakeIMAP4(self)

    def smtp_factory(self, *args, **kwargs) -> "FakeSMTP":
        return FakeSMTP(self)

    def messages_in(self, mailbox: str) -> List[Tuple[str, bytes]]:
        return list(self.mailboxes.get(mailbox, []))


class FakeIMAP4:
    def __init__(self, store: FakeMailStore):
        self.store = store
        self.capabilities = store.capabilities

    def _log(self, name: str, *args) -> None:
        self.store.command_log.append((name, args))

    def login(self, user: str, password: str):
        self._log("login", user)
        return ("OK", [b"LOGIN completed"])

    def capability(self):
        self._log("capability")
        return ("OK", [" ".join(self.store.capabilities).encode("ascii")])

    def xatom(self, name: str, *args):
        self._log("xatom", name, *args)
        if name.upper() == "ID" and "ID" not in self.store.capabilities:
            # Mimic Purelymail: unsupported command answers BAD.
            raise Exception("BAD Unknown command")
        return ("OK", [b"ID completed"])

    def select(self, mailbox: str = "INBOX"):
        self._log("select", mailbox)
        count = len(self.store.mailboxes.get(mailbox, []))
        return ("OK", [str(count).encode("ascii")])

    def uid(self, command: str, *args):
        self._log("uid", command, *args)
        command = command.lower()
        if command == "search":
            uids = b" ".join(
                str(u).encode("ascii") for u in sorted(self.store._inbox_uids)
            )
            return ("OK", [uids])
        if command == "fetch":
            uid = int(args[0])
            raw = self.store._inbox_uids.get(uid)
            if raw is None:
                return ("NO", [b"no such message"])
            return ("OK", [(str(uid).encode("ascii") + b" (RFC822)", raw)])
        return ("NO", [b"unsupported"])

    def append(self, mailbox: str, flags: str, date_time, message_bytes: bytes):
        self._log("append", mailbox, flags)
        if self.store.append_exception is not None:
            raise self.store.append_exception
        if self.store.fail_append_status is not None:
            return (self.store.fail_append_status, [b"append refused"])
        self.store.mailboxes.setdefault(mailbox, []).append(
            (flags or "", bytes(message_bytes))
        )
        return ("OK", [b"APPEND completed"])

    def logout(self):
        self._log("logout")
        return ("BYE", [b"logging out"])

    def shutdown(self):
        self._log("shutdown")

    def starttls(self, ssl_context=None):
        self._log("starttls")
        return ("OK", [b"TLS negotiation successful"])


class FakeSMTP:
    def __init__(self, store: FakeMailStore):
        self.store = store

    def login(self, user: str, password: str):
        self.store.command_log.append(("smtp_login", (user,)))

    def starttls(self, context=None):
        self.store.command_log.append(("smtp_starttls", ()))

    def send_message(self, msg):
        self.store.command_log.append(("smtp_send_message", (msg["To"],)))
        self.store.smtp_messages.append(msg)

    def quit(self):
        self.store.command_log.append(("smtp_quit", ()))

    def close(self):
        self.store.command_log.append(("smtp_close", ()))
