"""Alias-aware override of the bundled Email platform, without patching core modules.

Only authentication seams are specialized. Inbound parsing, authorization,
threading, attachments, TLS transport, lifecycle and error policies continue to
come from the bundled adapter installed with the user's Hermes version.
"""

from contextlib import contextmanager, suppress
import smtplib

from gateway.platforms._shared import get_scoped_secret
from plugins.platforms.email import adapter as email_base


def resolve_login_user(extra: dict, address: str) -> str:
    """Non-blank scoped env > config.yaml extra > public From address."""
    return (str(get_scoped_secret("EMAIL_LOGIN_USER", "") or "").strip()
            or str(extra.get("login_user") or "").strip()
            or address)


def _release_smtp(server) -> None:
    """Release a connected SMTP handle even when QUIT fails."""
    try:
        server.quit()
    except Exception:
        with suppress(Exception):
            server.close()


class EmailAliasAdapter(email_base.EmailAdapter):
    """Keep the bundled Email behavior; change only the login identity."""

    def __init__(self, config):
        super().__init__(config)
        self._login_user = resolve_login_user(config.extra or {}, self._address)

    @contextmanager
    def _inbox(self):
        """The shared connect/poll inbox seam, with guaranteed socket teardown."""
        imap = self._connect_imap()
        try:
            imap.login(self._login_user, self._password)
            email_base._send_imap_id(imap)
            imap.select("INBOX")
            yield imap
        finally:
            email_base._close_imap(imap)

    def _probe_smtp(self) -> bool:
        try:
            smtp = self._connect_smtp()
            try:
                smtp.login(self._login_user, self._password)
            finally:
                _release_smtp(smtp)
            email_base.logger.info("[Email] SMTP connection test passed.")
            return True
        except smtplib.SMTPAuthenticationError as exc:
            return self._fail("[Email] SMTP authentication failed: %s", exc, "email_auth_error",
                              f"SMTP authentication failed for {self._address}: {exc}. Check EMAIL_PASSWORD (for Gmail/Outlook "
                              "this must be an app password, not the account password).", retryable=False)
        except Exception as exc:
            return self._fail("[Email] SMTP connection failed: %s", exc, "email_smtp_connect_error",
                              f"SMTP connection to {self._smtp_host} failed: {exc}", retryable=True)

    def _smtp_send(self, msg) -> None:
        """Replies and attachments share this one authentication seam."""
        smtp = self._connect_smtp()
        try:
            smtp.login(self._login_user, self._password)
            smtp.send_message(msg)
        finally:
            _release_smtp(smtp)


async def standalone_send(pconfig, chat_id, message, *, thread_id=None,
                          media_files=None, force_document=False):
    """Out-of-process/cron SMTP delivery with the same login resolution."""
    extra = getattr(pconfig, "extra", {}) or {}
    address = (get_scoped_secret("EMAIL_ADDRESS", "") or extra.get("address") or "").strip()
    password = get_scoped_secret("EMAIL_PASSWORD", "")
    login_user = resolve_login_user(extra, address)
    smtp_host = (get_scoped_secret("EMAIL_SMTP_HOST", "") or extra.get("smtp_host") or "").strip()
    smtp_port = email_base._esecret_int("EMAIL_SMTP_PORT", 587)
    smtp_security = email_base._normalize_security(
        get_scoped_secret("EMAIL_SMTP_SECURITY", "") or extra.get("smtp_security"),
        default="tls" if smtp_port == 465 else "starttls",
    )
    smtp_tls_verify = email_base._esecret_bool(
        "EMAIL_SMTP_TLS_VERIFY",
        email_base.is_truthy_value(extra.get("smtp_tls_verify"), default=True),
    )
    if not all([address, password, smtp_host]):
        return email_base.send_error("Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)")
    try:
        msg = email_base.MIMEText(message, "plain", "utf-8")
        for key, value in (("From", address), ("To", chat_id),
                           ("Subject", email_base.t("platform.email.standalone_subject")),
                           ("Date", email_base.formatdate(localtime=True))):
            msg[key] = value
        server = email_base._open_smtp(
            smtp_host, smtp_port, smtp_security,
            email_base._tls_context(smtp_tls_verify, smtp_host),
            smtplib.SMTP, smtplib.SMTP_SSL,
        )
        try:
            server.login(login_user, password)
            server.send_message(msg)
        finally:
            _release_smtp(server)
        return {"success": True, "platform": "email", "chat_id": chat_id}
    except Exception as exc:
        try:
            from tools.send_message_tool import _error
            return _error(f"Email send failed: {exc}")
        except Exception:
            return email_base.send_error(f"Email send failed: {exc}")


def register(ctx) -> None:
    """Re-register `email` for this profile, retaining bundled metadata."""
    class EmailRegistration:
        def register_platform(self, **fields):
            required = list(fields.get("required_env") or [])
            if "EMAIL_IMAP_HOST" not in required:
                required.append("EMAIL_IMAP_HOST")
            fields.update(adapter_factory=EmailAliasAdapter,
                          standalone_sender_fn=standalone_send,
                          required_env=required)
            return ctx.register_platform(**fields)

    email_base.register(EmailRegistration())
