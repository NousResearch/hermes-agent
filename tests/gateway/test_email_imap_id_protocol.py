"""Exercise IMAP ID negotiation with real imaplib and a local protocol peer.

No external mail service or credentials are used. Unsupported ID gets BAD
followed by BYE: swallowing the ID error still leaves SELECT unable to proceed.
"""

import imaplib
import socketserver
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


@contextmanager
def imap_peer(id_mode):
    commands = []

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(5)
            try:
                self.exchange()
            except (TimeoutError, ConnectionError):
                # Bound cleanup even if the client fails before LOGOUT.
                return

        def exchange(self):
            self.wfile.write(b"* OK test IMAP ready\r\n")
            while raw := self.rfile.readline():
                tag, command, *_ = raw.decode("ascii").strip().split()
                command = command.upper()
                commands.append(command)
                prefix = tag.encode("ascii")
                if command == "CAPABILITY":
                    caps = b"IMAP4rev1" + (b" ID" if id_mode != "absent" else b"")
                    self.wfile.write(b"* CAPABILITY " + caps + b"\r\n")
                elif command == "ID":
                    if id_mode == "absent":
                        self.wfile.write(
                            prefix
                            + b" BAD ID unsupported\r\n* BYE Unknown command.\r\n"
                        )
                        return
                    if id_mode == "reject":
                        self.wfile.write(prefix + b" BAD ID rejected\r\n")
                        continue
                    self.wfile.write(b"* ID NIL\r\n")
                elif command == "SELECT":
                    self.wfile.write(b"* 0 EXISTS\r\n")
                elif command == "UID":
                    self.wfile.write(b"* SEARCH\r\n")
                elif command == "LOGOUT":
                    self.wfile.write(
                        b"* BYE logging out\r\n" + prefix + b" OK LOGOUT\r\n"
                    )
                    return
                elif command != "LOGIN":
                    self.wfile.write(prefix + b" BAD unexpected command\r\n")
                    continue
                self.wfile.write(prefix + b" OK completed\r\n")

    # Non-daemon handler threads are joined by server_close on context exit;
    # socket timeouts also bound teardown when the client fails prematurely.
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        thread.start()
        try:
            yield ("127.0.0.1", int(server.server_address[1])), commands
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("id_mode", ["absent", "accept", "reject"])
def test_id_negotiation_preserves_inbox_connection(monkeypatch, id_mode):
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter

    for key, value in {
        "EMAIL_ADDRESS": "agent@example.test",
        "EMAIL_PASSWORD": "test-only",
        "EMAIL_IMAP_HOST": "imap.example.test",
        "EMAIL_SMTP_HOST": "smtp.example.test",
    }.items():
        monkeypatch.setenv(key, value)
    adapter = EmailAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_connect_smtp", MagicMock(return_value=MagicMock()))

    with imap_peer(id_mode) as (address, commands):
        # TLS is out of scope: replace only the transport with real imaplib on
        # loopback, keeping the actual IMAP parser and command API under test.
        monkeypatch.setattr(
            imaplib,
            "IMAP4_SSL",
            lambda *args, **kwargs: imaplib.IMAP4(*address, timeout=5),
        )

        assert adapter._fetch_new_messages() == []
        assert adapter._last_fetch_failed is False

    assert commands.count("ID") == (0 if id_mode == "absent" else 1)
    assert commands.index("LOGIN") < commands.index("SELECT") < commands.index("UID")
    if id_mode != "absent":
        assert commands.index("LOGIN") < commands.index("ID") < commands.index("SELECT")
    assert commands[-1] == "LOGOUT"
