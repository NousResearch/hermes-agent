"""Programmatic email (Mode B) via stdlib smtplib/imaplib - no human in the loop.

This is what turns email opt-outs autonomous: `send()` delivers the rendered
request straight to the broker's known opt-out address, and `find_verification_link()`
polls the inbox for the broker's confirmation email and extracts the link (scored
by email_modes.extract_verification_link, so arbitrary/phishing links are ignored).
The agent still OPENS the link with its own browser - several brokers bind the
verification session to the browser that opens it (see the intelius record).

Configuration comes from the same env vars the Hermes email gateway uses:
  EMAIL_ADDRESS / EMAIL_PASSWORD              (required for Mode B)
  EMAIL_SMTP_HOST / EMAIL_SMTP_PORT           (optional; inferred for common providers)
  EMAIL_IMAP_HOST / EMAIL_IMAP_PORT           (optional; inferred for common providers)

Anti-misuse: `send()` refuses a recipient that is not the broker record's own
opt-out/privacy address - this module cannot be repurposed to email arbitrary people.
All network calls live behind small functions that the hermetic tests monkeypatch.
"""
from __future__ import annotations

import email as _email
import email.utils
import imaplib
import json
import os
import re
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

import email_modes
import paths

# provider domain -> (smtp_host, smtp_port, imap_host, imap_port)
PROVIDERS = {
    "gmail.com": ("smtp.gmail.com", 587, "imap.gmail.com", 993),
    "googlemail.com": ("smtp.gmail.com", 587, "imap.gmail.com", 993),
    "outlook.com": ("smtp-mail.outlook.com", 587, "outlook.office365.com", 993),
    "hotmail.com": ("smtp-mail.outlook.com", 587, "outlook.office365.com", 993),
    "live.com": ("smtp-mail.outlook.com", 587, "outlook.office365.com", 993),
    "yahoo.com": ("smtp.mail.yahoo.com", 587, "imap.mail.yahoo.com", 993),
    "icloud.com": ("smtp.mail.me.com", 587, "imap.mail.me.com", 993),
    "me.com": ("smtp.mail.me.com", 587, "imap.mail.me.com", 993),
    "fastmail.com": ("smtp.fastmail.com", 587, "imap.fastmail.com", 993),
}


def _domain(address: str) -> str:
    return address.rsplit("@", 1)[-1].lower() if "@" in address else ""


def smtp_settings(env: dict | None = None) -> dict | None:
    """SMTP connection settings, or None when sending is not configured."""
    env = os.environ if env is None else env
    address, password = env.get("EMAIL_ADDRESS"), env.get("EMAIL_PASSWORD")
    if not (address and password):
        return None
    inferred = PROVIDERS.get(_domain(address))
    host = env.get("EMAIL_SMTP_HOST") or (inferred[0] if inferred else None)
    if not host:
        return None  # unknown provider and no explicit host
    port = int(env.get("EMAIL_SMTP_PORT") or (inferred[1] if inferred else 587))
    return {"host": host, "port": port, "address": address, "password": password}


def imap_settings(env: dict | None = None) -> dict | None:
    """IMAP connection settings, or None when inbox reading is not configured."""
    env = os.environ if env is None else env
    address, password = env.get("EMAIL_ADDRESS"), env.get("EMAIL_PASSWORD")
    if not (address and password):
        return None
    inferred = PROVIDERS.get(_domain(address))
    host = env.get("EMAIL_IMAP_HOST") or (inferred[2] if inferred else None)
    if not host:
        return None
    port = int(env.get("EMAIL_IMAP_PORT") or (inferred[3] if inferred else 993))
    return {"host": host, "port": port, "address": address, "password": password}


def available(env: dict | None = None) -> dict:
    return {"smtp": smtp_settings(env) is not None, "imap": imap_settings(env) is not None}


# --- sending ------------------------------------------------------------------

def broker_addresses(broker: dict) -> list[str]:
    """Every address the broker record itself declares (the ONLY valid recipients).

    Includes the primary opt-out email, the right-to-delete lane's email
    (optout.deletion.email), and any mailto: links parsed from BADBOOL.
    """
    opt = broker.get("optout") or {}
    out = [a for a in [opt.get("email"), (opt.get("deletion") or {}).get("email")] if a]
    for link in opt.get("links") or []:
        url = (link.get("url") or "")
        if url.lower().startswith("mailto:"):
            out.append(url[7:].split("?")[0])
    seen: set[str] = set()
    deduped = []
    for a in out:
        if a.lower() not in seen:
            seen.add(a.lower())
            deduped.append(a)
    return deduped


def _split_subject_body(text: str) -> tuple[str, str]:
    """Templates start with a 'Subject: ...' line; split it out for the MIME header."""
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("subject:"):
        return lines[0].split(":", 1)[1].strip(), "\n".join(lines[1:]).lstrip("\n")
    return "Data removal request", text


def browser_send_payload(broker: dict, body_text: str, to: str | None = None) -> dict:
    """Build a recipient-locked {to, subject, body} for the agent to send via browser webmail.

    No network and no credentials: the deterministic part (recipient-lock to the broker's own
    declared address, subject/body split) happens here; the agent then composes and sends it in
    the operator's logged-in webmail with browser_* tools. Same recipient guard as `send()`, so
    the browser lane cannot be pointed at an arbitrary person either.
    """
    allowed = broker_addresses(broker)
    if not allowed:
        raise RuntimeError(f"broker {broker.get('id')!r} declares no opt-out email address")
    recipient = to or allowed[0]
    if recipient.lower() not in {a.lower() for a in allowed}:
        raise PermissionError(
            f"refusing to target {recipient!r}: not an address the broker record declares "
            f"(allowed: {allowed})"
        )
    subject, body = _split_subject_body(body_text)
    return {"to": recipient, "subject": subject, "body": body}


def _rate_limit_path() -> Path:
    return paths.data_dir() / "email-rate.json"


def _respect_rate_limit(min_interval: float, sleep, now, state_path=None) -> None:
    """Pace sends across CLI invocations so a run can't torch the sending account.

    Persists the last-send wall-clock time; if the next send is too soon, sleep the
    remainder. Cross-process because each `send-email` is a separate invocation.
    """
    if min_interval <= 0:
        return
    p = state_path or _rate_limit_path()
    last = 0.0
    try:
        last = float(json.loads(p.read_text(encoding="utf-8")).get("last", 0.0))
    except (OSError, ValueError, TypeError):
        last = 0.0
    wait = min_interval - (now() - last)
    if wait > 0:
        sleep(min(wait, min_interval))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"last": now()}), encoding="utf-8")
    except OSError:
        pass


# SMTP errors that are permanent (don't retry) vs transient (retry with backoff).
_SMTP_PERMANENT = (smtplib.SMTPAuthenticationError, smtplib.SMTPRecipientsRefused,
                   smtplib.SMTPSenderRefused, smtplib.SMTPDataError)


def send(broker: dict, body_text: str, to: str | None = None,
         env: dict | None = None, _smtp_factory=None,
         min_interval: float = 0.0, max_retries: int = 3,
         _sleep=time.sleep, _now=time.time, _rate_state=None) -> dict:
    """Send an opt-out/legal request to the broker's own opt-out address.

    Recipient is locked to an address the broker record declares (PermissionError
    otherwise). `min_interval` paces sends across invocations (deliverability /
    account-safety); transient SMTP/socket failures retry with exponential backoff,
    permanent ones (auth, recipient refused) raise immediately. NOTE: a successful
    SMTP handoff is NOT proof of delivery - real bounces arrive later as inbound mail;
    in programmatic mode `poll-verification`/inbox review surfaces them, and the
    due-queue re-scan is the true confirmation. Returns send metadata.
    """
    settings = smtp_settings(env)
    if not settings:
        raise RuntimeError(
            "programmatic email not configured (need EMAIL_ADDRESS + EMAIL_PASSWORD, and "
            "EMAIL_SMTP_HOST for non-mainstream providers); fall back to `render-email` drafts"
        )
    allowed = broker_addresses(broker)
    if not allowed:
        raise RuntimeError(f"broker {broker.get('id')!r} declares no opt-out email address")
    recipient = to or allowed[0]
    if recipient.lower() not in {a.lower() for a in allowed}:
        raise PermissionError(
            f"refusing to send to {recipient!r}: not an address the broker record declares "
            f"(allowed: {allowed})"
        )

    subject, body = _split_subject_body(body_text)
    msg = EmailMessage()
    msg["From"] = settings["address"]
    msg["To"] = recipient
    msg["Subject"] = subject
    msg["Date"] = email.utils.formatdate(localtime=True)
    msg["Message-ID"] = email.utils.make_msgid()
    msg.set_content(body)

    _respect_rate_limit(min_interval, _sleep, _now, _rate_state)

    factory = _smtp_factory or smtplib.SMTP
    attempts = 0
    while True:
        attempts += 1
        try:
            with factory(settings["host"], settings["port"], timeout=30) as smtp:
                smtp.ehlo()
                try:
                    smtp.starttls()
                    smtp.ehlo()
                except smtplib.SMTPNotSupportedError:
                    pass  # already-TLS ports / test doubles
                smtp.login(settings["address"], settings["password"])
                smtp.send_message(msg)
            break
        except _SMTP_PERMANENT:
            raise  # auth / recipient refused: retrying won't help
        except (smtplib.SMTPException, OSError) as exc:
            if attempts > max_retries:
                raise RuntimeError(f"SMTP send failed after {attempts} attempts: {exc}") from exc
            _sleep(min(2 ** (attempts - 1), 30))  # 1s, 2s, 4s... capped
    return {"to": recipient, "subject": subject, "message_id": msg["Message-ID"],
            "from": settings["address"], "attempts": attempts,
            "delivery_note": "SMTP accepted; not proof of delivery - a bounce would arrive as "
                             "inbound mail. The due-queue re-scan is the real confirmation."}


# --- inbox polling ------------------------------------------------------------

def _decode_part(part) -> str:
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    except Exception:  # noqa: BLE001 - malformed MIME must not kill the poll
        return ""


def message_text(msg) -> str:
    """All text/plain + text/html content of a parsed email message."""
    chunks: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() in ("text/plain", "text/html"):
                chunks.append(_decode_part(part))
    else:
        chunks.append(_decode_part(msg))
    return "\n".join(c for c in chunks if c)


def _broker_domains(broker: dict) -> list[str]:
    """Domains this broker legitimately mails from (site domains + optout email domain)."""
    domains: list[str] = []
    for section in ("optout", "search"):
        url = ((broker.get(section) or {}).get("url")) or ""
        m = re.search(r"https?://([^/]+)", url)
        if m:
            domains.append(m.group(1).lower().removeprefix("www."))
    opt_email = (broker.get("optout") or {}).get("email")
    if opt_email and "@" in opt_email:
        domains.append(_domain(opt_email))
    # strip subdomains to the registrable-ish tail (mailer.intelius.com -> intelius.com)
    tails = {".".join(d.split(".")[-2:]) for d in domains if d}
    return sorted(tails)


def fetch_recent(env: dict | None = None, since_days: int = 3, limit: int = 30,
                 _imap_factory=None) -> list[dict]:
    """Fetch recent inbox messages: [{from, subject, date, text}], newest first."""
    settings = imap_settings(env)
    if not settings:
        raise RuntimeError("IMAP not configured (need EMAIL_ADDRESS + EMAIL_PASSWORD, and "
                           "EMAIL_IMAP_HOST for non-mainstream providers)")
    import datetime as _dt
    since = (_dt.date.today() - _dt.timedelta(days=max(0, since_days))).strftime("%d-%b-%Y")

    factory = _imap_factory or imaplib.IMAP4_SSL
    conn = factory(settings["host"], settings["port"])
    try:
        conn.login(settings["address"], settings["password"])
        conn.select("INBOX", readonly=True)
        _typ, data = conn.search(None, "SINCE", since)
        ids = (data[0].split() if data and data[0] else [])[-limit:]
        out: list[dict] = []
        for mid in reversed(ids):  # newest first
            _typ, msg_data = conn.fetch(mid, "(RFC822)")
            raw = next((p[1] for p in msg_data or [] if isinstance(p, tuple)), None)
            if not raw:
                continue
            msg = _email.message_from_bytes(raw)
            out.append({
                "from": msg.get("From", ""),
                "subject": msg.get("Subject", ""),
                "date": msg.get("Date", ""),
                "text": message_text(msg),
            })
        return out
    finally:
        try:
            conn.logout()
        except Exception:  # noqa: BLE001
            pass


def link_from_messages(messages: list[dict], broker: dict) -> dict | None:
    """Pure: find the broker's verification link in already-fetched messages.

    A message is only considered if its From domain OR any contained link matches
    the broker's own domains; the link itself must pass the anti-phishing scorer.
    """
    domains = _broker_domains(broker)
    for m in messages:
        sender = (m.get("from") or "").lower()
        text = m.get("text") or ""
        sender_match = any(d in sender for d in domains)
        body_match = any(d in text.lower() for d in domains)
        if not (sender_match or body_match):
            continue
        link = email_modes.extract_verification_link(text, broker)
        if link:
            return {"link": link, "from": m.get("from"), "subject": m.get("subject"),
                    "date": m.get("date")}
    return None


def find_verification_link(broker: dict, env: dict | None = None, since_days: int = 3,
                           _imap_factory=None) -> dict | None:
    """Poll the inbox and return the broker's verification link (or None yet)."""
    messages = fetch_recent(env, since_days=since_days, _imap_factory=_imap_factory)
    return link_from_messages(messages, broker)
