"""Tests for the SMTP security policy in the Email platform adapter.

Covers:
1. normalize_smtp_security - canonicalization + alias handling.
2. normalize_smtp_security - rejection of ambiguous/invalid values.
3. resolve_smtp_security - port + mode resolution.
4. open_smtp_connection - STARTTLS failure closes the connection and raises.
5. _standalone_send - end-to-end mode selection (implicit_tls vs starttls),
   override semantics, and validation-before-connect.

Behavior-contract tests only (no source-text reading, no snapshot values).
Stdlib + pytest + unittest.mock, no live network.
"""

import asyncio
import types
from unittest.mock import MagicMock, patch

import pytest

from plugins.platforms.email.adapter import (
    SMTP_SECURITY_ENV,
    _standalone_send,
    normalize_smtp_security,
    open_smtp_connection,
    resolve_smtp_security,
)


# ── 1. normalize_smtp_security: canonicalization ────────────────────────────


class TestNormalizeSmtpSecurity:
    def test_none_is_auto(self):
        assert normalize_smtp_security(None) == "auto"

    @pytest.mark.parametrize("raw", ["", "   ", "\t", "\n  "])
    def test_empty_or_whitespace_is_auto(self, raw):
        assert normalize_smtp_security(raw) == "auto"

    def test_auto_case_insensitive(self):
        assert normalize_smtp_security("AUTO") == "auto"
        assert normalize_smtp_security("Auto") == "auto"

    def test_starttls_canonical_and_case(self):
        assert normalize_smtp_security("StartTLS") == "starttls"
        assert normalize_smtp_security("STARTTLS") == "starttls"
        assert normalize_smtp_security("starttls") == "starttls"

    @pytest.mark.parametrize("raw", ["start_tls", "start-tls", "Start_TLS", "START-TLS"])
    def test_starttls_aliases(self, raw):
        assert normalize_smtp_security(raw) == "starttls"

    def test_implicit_tls_canonical_and_case(self):
        assert normalize_smtp_security("IMPLICIT_TLS") == "implicit_tls"
        assert normalize_smtp_security("Implicit_Tls") == "implicit_tls"

    @pytest.mark.parametrize("raw", ["implicit-tls", "SMTPS", "smtp_ssl", "SMTP_SSL", "smtps"])
    def test_implicit_tls_aliases(self, raw):
        assert normalize_smtp_security(raw) == "implicit_tls"


# ── 2. normalize_smtp_security: rejection ───────────────────────────────────


class TestNormalizeSmtpSecurityRejection:
    @pytest.mark.parametrize(
        "raw",
        ["tls", "ssl", "none", "plain", "plaintext", "no_tls", "off", "false", "true", "bogus"],
    )
    def test_invalid_values_raise_value_error(self, raw):
        with pytest.raises(ValueError) as exc_info:
            normalize_smtp_security(raw)
        msg = str(exc_info.value)
        # The error must reference the env var name, the offending raw value,
        # and all three canonical options so the operator can self-correct.
        assert SMTP_SECURITY_ENV in msg
        assert repr(raw) in msg
        assert "auto" in msg
        assert "starttls" in msg
        assert "implicit_tls" in msg

    def test_whitespace_then_invalid_still_rejects(self):
        with pytest.raises(ValueError):
            normalize_smtp_security("  tls  ")


# ── 3. resolve_smtp_security ────────────────────────────────────────────────


class TestResolveSmtpSecurity:
    def test_port_465_auto_is_implicit_tls(self):
        assert resolve_smtp_security(465, None) == "implicit_tls"

    def test_port_465_explicit_auto_is_implicit_tls(self):
        assert resolve_smtp_security(465, "auto") == "implicit_tls"

    def test_port_587_none_is_starttls(self):
        assert resolve_smtp_security(587, None) == "starttls"

    def test_nonstandard_port_auto_is_starttls(self):
        assert resolve_smtp_security(2525, "auto") == "starttls"

    def test_explicit_starttls_wins_over_port_465(self):
        assert resolve_smtp_security(465, "starttls") == "starttls"

    def test_explicit_implicit_tls_wins_over_port_587(self):
        assert resolve_smtp_security(587, "implicit_tls") == "implicit_tls"

    def test_alias_resolves_through_resolve(self):
        assert resolve_smtp_security(2525, "smtps") == "implicit_tls"

    def test_invalid_mode_raises_before_returning(self):
        with pytest.raises(ValueError):
            resolve_smtp_security(587, "bogus")


# ── 4. open_smtp_connection ─────────────────────────────────────────────────


class TestOpenSmtpConnection:
    def _make_mock_module(self):
        """Build a fake smtplib module whose SMTP/SMTP_SSL return Magics."""
        module = MagicMock()
        server = MagicMock()
        module.SMTP.return_value = server
        module.SMTP_SSL.return_value = MagicMock()
        return module, server

    def test_implicit_tls_uses_smtp_ssl_and_no_starttls(self):
        module, _ = self._make_mock_module()
        ctx_factory = MagicMock()
        open_smtp_connection(
            "smtp.test.com", 465, "implicit_tls",
            smtp_module=module, context_factory=ctx_factory,
        )
        assert module.SMTP_SSL.call_count == 1
        assert module.SMTP.call_count == 0
        # SMTP_SSL should be constructed with host/port/timeout/context.
        args, kwargs = module.SMTP_SSL.call_args
        assert args[0] == "smtp.test.com"
        assert args[1] == 465
        assert kwargs["context"] is ctx_factory.return_value

    def test_starttls_uses_smtp_then_starttls(self):
        module, server = self._make_mock_module()
        ctx_factory = MagicMock()
        result = open_smtp_connection(
            "smtp.test.com", 587, "starttls",
            smtp_module=module, context_factory=ctx_factory,
        )
        assert module.SMTP.call_count == 1
        assert module.SMTP_SSL.call_count == 0
        server.starttls.assert_called_once()
        assert result is server

    def test_auto_port_465_uses_smtp_ssl(self):
        module, _ = self._make_mock_module()
        open_smtp_connection(
            "smtp.test.com", 465, None,
            smtp_module=module, context_factory=MagicMock(),
        )
        assert module.SMTP_SSL.call_count == 1
        assert module.SMTP.call_count == 0

    def test_auto_port_587_uses_smtp_plus_starttls(self):
        module, server = self._make_mock_module()
        open_smtp_connection(
            "smtp.test.com", 587, None,
            smtp_module=module, context_factory=MagicMock(),
        )
        assert module.SMTP.call_count == 1
        assert module.SMTP_SSL.call_count == 0
        server.starttls.assert_called_once()

    def test_invalid_mode_raises_without_constructing(self):
        module, _ = self._make_mock_module()
        with pytest.raises(ValueError):
            open_smtp_connection(
                "smtp.test.com", 587, "bogus",
                smtp_module=module, context_factory=MagicMock(),
            )
        # No SMTP object constructed for an invalid mode.
        assert module.SMTP.call_count == 0
        assert module.SMTP_SSL.call_count == 0

    def test_starttls_failure_closes_connection_and_raises(self):
        """If STARTTLS fails, the connection must be closed and the error
        propagated - never left half-open."""
        module = MagicMock()
        server = MagicMock()
        server.starttls.side_effect = RuntimeError("STARTTLS unsupported")
        module.SMTP.return_value = server
        module.SMTP_SSL.return_value = MagicMock()

        with pytest.raises(RuntimeError):
            open_smtp_connection(
                "smtp.test.com", 587, "starttls",
                smtp_module=module, context_factory=MagicMock(),
            )

        # SMTP constructed once, SMTP_SSL never, starttls attempted once,
        # and the connection torn down (quit) on failure.
        assert module.SMTP.call_count == 1
        assert module.SMTP_SSL.call_count == 0
        server.starttls.assert_called_once()
        server.quit.assert_called_once()


# ── 5. _standalone_send end-to-end ──────────────────────────────────────────


def _make_pconfig(extra=None):
    """Build a PlatformConfig-like object with the keys _standalone_send reads."""
    merged = {
        "address": "hermes@test.com",
        "smtp_host": "smtp.test.com",
    }
    if extra:
        merged.update(extra)
    return types.SimpleNamespace(extra=merged)


class TestStandaloneSend:
    @patch("plugins.platforms.email.adapter.smtplib.SMTP")
    @patch("plugins.platforms.email.adapter.smtplib.SMTP_SSL")
    def test_port_465_no_security_uses_smtp_ssl(self, mock_ssl_cls, mock_smtp_cls):
        with patch.dict("os.environ", {"EMAIL_PASSWORD": "secret", "EMAIL_SMTP_PORT": "465"}, clear=False):
            result = asyncio.run(
                _standalone_send(_make_pconfig(), "user@test.com", "hi")
            )
        assert result.get("success") is True
        assert mock_ssl_cls.call_count == 1
        assert mock_smtp_cls.call_count == 0
        # The implicit-TLS server must still login + send + quit.
        ssl_server = mock_ssl_cls.return_value
        ssl_server.login.assert_called_once()
        ssl_server.send_message.assert_called_once()
        ssl_server.quit.assert_called_once()

    @patch("plugins.platforms.email.adapter.smtplib.SMTP")
    @patch("plugins.platforms.email.adapter.smtplib.SMTP_SSL")
    def test_port_587_no_security_uses_smtp_plus_starttls(self, mock_ssl_cls, mock_smtp_cls):
        with patch.dict("os.environ", {"EMAIL_PASSWORD": "secret", "EMAIL_SMTP_PORT": "587"}, clear=False):
            result = asyncio.run(
                _standalone_send(_make_pconfig(), "user@test.com", "hi")
            )
        assert result.get("success") is True
        assert mock_smtp_cls.call_count == 1
        assert mock_ssl_cls.call_count == 0
        smtp_server = mock_smtp_cls.return_value
        smtp_server.starttls.assert_called_once()
        smtp_server.login.assert_called_once()
        smtp_server.send_message.assert_called_once()
        smtp_server.quit.assert_called_once()

    @patch("plugins.platforms.email.adapter.smtplib.SMTP")
    @patch("plugins.platforms.email.adapter.smtplib.SMTP_SSL")
    def test_port_465_with_starttls_override_uses_smtp(self, mock_ssl_cls, mock_smtp_cls):
        # Explicit starttls on port 465 must force STARTTLS, not SMTP_SSL.
        pconfig = _make_pconfig({"smtp_security": "starttls"})
        with patch.dict("os.environ", {"EMAIL_PASSWORD": "secret", "EMAIL_SMTP_PORT": "465"}, clear=False):
            result = asyncio.run(
                _standalone_send(pconfig, "user@test.com", "hi")
            )
        assert result.get("success") is True
        assert mock_smtp_cls.call_count == 1
        assert mock_ssl_cls.call_count == 0
        mock_smtp_cls.return_value.starttls.assert_called_once()

    @patch("plugins.platforms.email.adapter.smtplib.SMTP")
    @patch("plugins.platforms.email.adapter.smtplib.SMTP_SSL")
    def test_port_587_with_implicit_tls_override_uses_smtp_ssl(self, mock_ssl_cls, mock_smtp_cls):
        # Explicit implicit_tls on port 587 must force SMTP_SSL.
        pconfig = _make_pconfig({"smtp_security": "implicit_tls"})
        with patch.dict("os.environ", {"EMAIL_PASSWORD": "secret", "EMAIL_SMTP_PORT": "587"}, clear=False):
            result = asyncio.run(
                _standalone_send(pconfig, "user@test.com", "hi")
            )
        assert result.get("success") is True
        assert mock_ssl_cls.call_count == 1
        assert mock_smtp_cls.call_count == 0
        mock_ssl_cls.return_value.starttls.assert_not_called()

    @patch("plugins.platforms.email.adapter.smtplib.SMTP")
    @patch("plugins.platforms.email.adapter.smtplib.SMTP_SSL")
    def test_bogus_security_fails_without_connecting(self, mock_ssl_cls, mock_smtp_cls):
        pconfig = _make_pconfig({"smtp_security": "bogus"})
        with patch.dict("os.environ", {"EMAIL_PASSWORD": "secret", "EMAIL_SMTP_PORT": "587"}, clear=False):
            result = asyncio.run(
                _standalone_send(pconfig, "user@test.com", "hi")
            )
        # Invalid mode -> failure result, and no SMTP object constructed.
        assert result.get("success") is not True
        assert "error" in result or "status" in result
        assert mock_smtp_cls.call_count == 0
        assert mock_ssl_cls.call_count == 0

    @patch("plugins.platforms.email.adapter.smtplib.SMTP")
    @patch("plugins.platforms.email.adapter.smtplib.SMTP_SSL")
    def test_env_security_bridge_override_uses_smtp_ssl(self, mock_ssl_cls, mock_smtp_cls):
        # EMAIL_SMTP_SECURITY env (bridge) should force implicit_tls on 587.
        with patch.dict(
            "os.environ",
            {"EMAIL_PASSWORD": "secret", "EMAIL_SMTP_PORT": "587", "EMAIL_SMTP_SECURITY": "smtps"},
            clear=False,
        ):
            result = asyncio.run(
                _standalone_send(_make_pconfig(), "user@test.com", "hi")
            )
        assert result.get("success") is True
        assert mock_ssl_cls.call_count == 1
        assert mock_smtp_cls.call_count == 0

    @patch("plugins.platforms.email.adapter.smtplib.SMTP")
    @patch("plugins.platforms.email.adapter.smtplib.SMTP_SSL")
    def test_send_message_failure_still_quits(self, mock_ssl_cls, mock_smtp_cls):
        # If login/send raises, quit must still be called (try/finally).
        ssl_server = mock_ssl_cls.return_value
        ssl_server.send_message.side_effect = RuntimeError("relay denied")
        with patch.dict("os.environ", {"EMAIL_PASSWORD": "secret", "EMAIL_SMTP_PORT": "465"}, clear=False):
            result = asyncio.run(
                _standalone_send(_make_pconfig(), "user@test.com", "hi")
            )
        assert result.get("success") is not True
        # quit called even on failure (cleanup).
        ssl_server.quit.assert_called_once()
