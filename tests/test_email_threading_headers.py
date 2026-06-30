from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import EmailAdapter


class FakeSMTP:
    def __init__(self):
        self.sent_message = None

    def login(self, address, password):
        self.login_args = (address, password)

    def send_message(self, msg):
        self.sent_message = msg

    def quit(self):
        pass


def _adapter(monkeypatch):
    monkeypatch.setenv("EMAIL_ADDRESS", "hermes@example.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setenv("EMAIL_IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("EMAIL_SMTP_HOST", "smtp.example.com")
    return EmailAdapter(PlatformConfig(enabled=True))


def test_email_replies_preserve_references_chain(monkeypatch):
    adapter = _adapter(monkeypatch)
    fake = FakeSMTP()
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: fake)

    adapter._thread_context["sender@example.com"] = {
        "subject": "Re: Existing thread",
        "message_id": "<current-inbound@example.com>",
        "references": "<root@example.com> <previous-agent@example.com>",
    }

    adapter._send_email("sender@example.com", "reply body")

    assert fake.sent_message is not None
    assert fake.sent_message["In-Reply-To"] == "<current-inbound@example.com>"
    assert (
        fake.sent_message["References"]
        == "<root@example.com> <previous-agent@example.com> <current-inbound@example.com>"
    )


def test_email_replies_fallback_to_current_message_id_without_references(monkeypatch):
    adapter = _adapter(monkeypatch)
    fake = FakeSMTP()
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: fake)

    adapter._thread_context["sender@example.com"] = {
        "subject": "New thread",
        "message_id": "<current-inbound@example.com>",
    }

    adapter._send_email("sender@example.com", "reply body")

    assert fake.sent_message is not None
    assert fake.sent_message["In-Reply-To"] == "<current-inbound@example.com>"
    assert fake.sent_message["References"] == "<current-inbound@example.com>"


def test_email_replies_unfold_and_deduplicate_references(monkeypatch):
    adapter = _adapter(monkeypatch)
    fake = FakeSMTP()
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: fake)

    adapter._thread_context["sender@example.com"] = {
        "subject": "Re: Existing thread",
        "message_id": "<current-inbound@example.com>",
        "references": (
            "<root@example.com>\r\n"
            " <previous-agent@example.com>\r\n"
            " <current-inbound@example.com>"
        ),
    }

    adapter._send_email("sender@example.com", "reply body")

    assert fake.sent_message is not None
    assert fake.sent_message["References"] == (
        "<root@example.com> <previous-agent@example.com> <current-inbound@example.com>"
    )
