"""Hermetic security contracts for the optional unbroker skill."""
import sys
from pathlib import Path
import smtplib
import ssl
import json
from unittest.mock import MagicMock

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "optional-skills/security/unbroker/scripts"
sys.path.insert(0, str(SCRIPTS))

import config
import dossier
import ledger
import pdd
import emailer
import email_modes
import autopilot
import brokers


@pytest.fixture(autouse=True)
def isolated_capability_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for key in config.CAPABILITY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_dotenv_only_exposes_declared_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("EMAIL_ADDRESS", "shell@example.com")
    monkeypatch.setenv("UNRELATED_SECRET", "shell-secret")
    (tmp_path / ".env").write_text(
        "EMAIL_ADDRESS=file@example.com\nEMAIL_PASSWORD=mail-secret\n"
        "BROWSERBASE_API_KEY=browser-key\nAGENTMAIL_API_KEY=mail-key\n"
        "UNRELATED_SECRET=file-secret\nEMAIL_UNUSED_SECRET=not-needed\n",
        encoding="utf-8",
    )
    env = config.dotenv_env()
    assert env["EMAIL_ADDRESS"] == "shell@example.com"
    assert env["EMAIL_PASSWORD"] == "mail-secret"
    assert env["BROWSERBASE_API_KEY"] == "browser-key"
    assert env["AGENTMAIL_API_KEY"] == "mail-key"
    assert "UNRELATED_SECRET" not in env
    assert "EMAIL_UNUSED_SECRET" not in env
    assert "PATH" not in env


@pytest.mark.parametrize("failure", [smtplib.SMTPNotSupportedError("no TLS"),
                                    ssl.SSLCertVerificationError("untrusted")])
def test_smtp_tls_failure_never_authenticates(failure):
    smtp = MagicMock()
    smtp.__enter__.return_value = smtp
    smtp.starttls.side_effect = failure
    with pytest.raises((smtplib.SMTPNotSupportedError, ssl.SSLCertVerificationError)):
        emailer.send({"optout": {"email": "privacy@example.com"}}, "request",
                     env={"EMAIL_ADDRESS": "owner@gmail.com", "EMAIL_PASSWORD": "dummy"},
                     _smtp_factory=lambda *a, **k: smtp, _sleep=lambda _: None)
    smtp.login.assert_not_called()
    smtp.send_message.assert_not_called()
    context = smtp.starttls.call_args.kwargs["context"]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


@pytest.mark.parametrize("port", [465, 587])
def test_smtp_transport_uses_verifying_context(port, monkeypatch):
    factory = MagicMock()
    smtp = factory.return_value.__enter__.return_value
    other = "SMTP" if port == 465 else "SMTP_SSL"
    monkeypatch.setattr(emailer.smtplib, other, MagicMock(side_effect=AssertionError("wrong transport")))
    monkeypatch.setattr(emailer.smtplib, "SMTP_SSL" if port == 465 else "SMTP", factory)
    emailer.send({"optout": {"email": "privacy@example.com"}}, "request",
                 env={"EMAIL_ADDRESS": "owner@gmail.com", "EMAIL_PASSWORD": "dummy",
                      "EMAIL_SMTP_PORT": str(port)})
    context = (factory.call_args.kwargs["context"] if port == 465
               else smtp.starttls.call_args.kwargs["context"])
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    if port == 465:
        smtp.starttls.assert_not_called()
    smtp.login.assert_called_once()
    smtp.send_message.assert_called_once()


def test_imap_requires_verifying_context(monkeypatch):
    factory = MagicMock()
    factory.return_value.search.return_value = ("OK", [b""])
    monkeypatch.setattr(emailer.imaplib, "IMAP4_SSL", factory)
    emailer.fetch_recent(env={"EMAIL_ADDRESS": "owner@gmail.com", "EMAIL_PASSWORD": "dummy"})
    context = factory.call_args.kwargs["ssl_context"]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    factory.return_value.select.assert_called_once_with("INBOX", readonly=True)


@pytest.mark.parametrize("link", [
    "http://broker.example/verify", "https://evil.example/verify?broker=broker.example",
    "https://broker.example.evil.example/verify", "https://broker.example@evil.example/verify",
    "https://evil.example/broker.example/confirm", "https://broker.example:444/verify",
])
def test_verification_rejects_untrusted_origins(link):
    broker = {"optout": {"url": "https://www.broker.example/optout"}}
    assert email_modes.extract_verification_link(link, broker) is None


def test_verification_requires_broker_and_exact_sender_domain():
    link = "https://broker.example/verify/request"
    broker = {"optout": {"url": "https://www.broker.example/optout"}}
    assert email_modes.extract_verification_link(link) is None
    assert email_modes.extract_verification_link(link, broker) == link
    forged = {"from": "Broker <mail@broker.example.evil.example>", "text": link}
    assert emailer.link_from_messages([forged], broker) is None
    assert emailer.link_from_messages([{"from": "mail@broker.example", "text": link}], broker)["link"] == link


@pytest.fixture
def pending_case(tmp_path, monkeypatch):
    monkeypatch.setenv("PDD_DATA_DIR", str(tmp_path / "data"))
    dossier.save({"subject_id": "sub_test", "consent": {"authorized": True, "method": "self"},
                  "identity": {"full_name": "Test Person", "emails": ["test@example.com"]}})
    ledger.transition("sub_test", "radaris", "found", found=True)
    return ["send-email", "sub_test", "radaris", "--listing", "https://radaris.com/test"]


def test_assisted_send_requires_explicit_confirmation(pending_case, monkeypatch):
    config.save_config({"autonomy": "assisted", "email_mode": "programmatic"})
    send = MagicMock(return_value={"message_id": "test"})
    monkeypatch.setattr(pdd.emailer, "send", send)
    with pytest.raises(SystemExit, match="--confirm"):
        pdd.main(pending_case)
    send.assert_not_called()
    assert ledger.get_case("sub_test", "radaris")["state"] == "found"
    pdd.main([*pending_case, "--confirm"])
    send.assert_called_once()


def test_browser_compose_does_not_claim_submission(pending_case, capsys):
    config.save_config({"autonomy": "full", "email_mode": "browser"})
    pdd.main(pending_case)
    result = json.loads(capsys.readouterr().out)
    assert result["send_via"] == "browser"
    case = ledger.get_case("sub_test", "radaris")
    assert case["state"] == "found"
    assert case["disclosure_log"] == []


def test_poll_candidate_does_not_advance_request(pending_case, monkeypatch, capsys):
    ledger.transition("sub_test", "radaris", "submitted")
    monkeypatch.setattr(pdd.emailer, "find_verification_link", lambda *a, **k: {
        "link": "https://radaris.com/verify/request", "from": "privacy@radaris.com"})
    pdd.main(["poll-verification", "sub_test", "--broker", "radaris"])
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["requires_review"] is True
    assert ledger.get_case("sub_test", "radaris")["state"] == "submitted"


def test_mail_settings_read_allowlisted_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in config.CAPABILITY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "EMAIL_ADDRESS=test@gmail.com\nEMAIL_PASSWORD=dummy\nEMAIL_SMTP_PORT=465\n"
        "EMAIL_IMAP_PORT=993\nUNRELATED_SECRET=do-not-use\n", encoding="utf-8")
    assert emailer.smtp_settings()["port"] == 465
    assert emailer.imap_settings()["port"] == 993


def test_setup_auto_preserves_explicit_safety_choices(tmp_path, monkeypatch):
    monkeypatch.setenv("PDD_DATA_DIR", str(tmp_path))
    config.save_config({"autonomy": "assisted", "email_mode": "browser"})
    result = config.auto_configure({"EMAIL_ADDRESS": "test@gmail.com", "EMAIL_PASSWORD": "dummy"})
    assert result["autonomy"] == "assisted"
    assert result["email_mode"] == "browser"


@pytest.mark.parametrize("mode", ["browser", "programmatic"])
def test_verification_queue_requires_provenance_review(pending_case, mode):
    ledger.transition("sub_test", "radaris", "submitted")
    cfg = {**config.DEFAULT_CONFIG, "email_mode": mode}
    env = {} if mode == "browser" else {"EMAIL_ADDRESS": "test@gmail.com", "EMAIL_PASSWORD": "dummy"}
    result = autopilot.next_actions(dossier.load("sub_test"), [brokers.get("radaris")],
                                    cfg, ledger.load("sub_test"), env=env)
    action = next(a for a in result["actions"] if a["type"] == "poll_verification")
    instruction = action.get("then") or action["how"]
    assert "provenance" in instruction
    assert "pending request" in instruction
    assert "final official destination" in instruction


def test_verify_link_requires_subject_authorization(pending_case, monkeypatch):
    require = MagicMock(side_effect=ValueError("consent required"))
    monkeypatch.setattr(pdd.dossier_mod, "require_authorized", require)
    with pytest.raises(SystemExit, match="consent required"):
        pdd.main(["verify-link", "sub_test", "radaris", "--text", "https://radaris.com/verify"])
    require.assert_called_once()


def test_invalid_submission_state_never_sends(pending_case, monkeypatch):
    config.save_config({"autonomy": "full", "email_mode": "programmatic"})
    ledger.transition("sub_test", "radaris", "blocked")
    send = MagicMock(return_value={"message_id": "test"})
    monkeypatch.setattr(pdd.emailer, "send", send)
    with pytest.raises(SystemExit):
        pdd.main(pending_case)
    send.assert_not_called()
