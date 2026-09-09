"""Detect SSL/TLS failure patterns during Codex device login and produce a user-facing hint.

Codex device-code login can fail with SSL errors on Python/OpenSSL 3.5 when a middlebox rejects
a TLS 1.3 handshake containing post-quantum hybrid groups (e.g. X25519MLKEM768 in ClientHello
`key_share`/`supported_groups`). The resulting transport error is indistinguishable from a genuine
network outage, but its fix is a client-side OpenSSL configuration that restricts offered groups
to classic elliptic curves. See issue #106384, PR #44392.
"""

from __future__ import annotations

import ssl
from typing import Any

_DOCS_URL = (
    "https://hermes-agent.nousresearch.com/docs/guides/codex-device-login-tls-failure"
)


def _ssl_tls_middlebox_hint(exc: Any) -> str | None:
    """Detect SSL/TLS-flavored transport failures and return a hint, or None for unrelated errors.

    Recognized patterns:
    - ssl.SSLError subclasses (SSLEOFError, SSLSyscallError, ...)
    - Message substrings: 'UNEXPECTED_EOF_WHILE_READING', 'handshake operation timed out', '[SSL:',
      'ssl handshake', 'tlsv1 alert', 'sslv3 alert', 'tls alert', 'bad record mac'

    Non-SSL transport errors (ConnectError, ReadTimeout, HTTPStatusError) return None — no false
    positives.
    """
    # Type match: Python SSL exception classes (avoid string-based type-name in case of SDK wrapping)
    if isinstance(
        exc,
        (ssl.SSLError, ssl.SSLEOFError, ssl.SSLSyscallError, ssl.SSLZeroReturnError),
    ):
        return _PQ_MIDDLEBOX_HINT

    # Message-based signatures (works for httpx.ConnectError that wraps SSL, SDK-wrapped errors, ...)
    msg = str(exc).lower()
    ssl_indicators = (
        "unexpected_eof_while_reading",
        "unexpected eof while reading",
        "handshake operation timed out",
        "ssl handshake",
        "tls handshake",
        "[ssl:",
        "tlsv1 alert",
        "sslv3 alert",
        "tls alert",
        "bad record mac",
    )
    if any(sig in msg for sig in ssl_indicators):
        return _PQ_MIDDLEBOX_HINT

    return None


_PQ_MIDDLEBOX_HINT = (
    "\n\n"
    "This failure pattern often indicates an SSL/TLS middlebox (corporate proxy, firewall, "
    "TLS-intercepting gateway) that rejects TLS 1.3 handshakes containing post-quantum hybrid "
    "groups. Python/OpenSSL 3.5 advertises such groups by default (e.g. X25519MLKEM768 in ClientHello "
    "`key_share`), and some middleboxes respond with a connection reset or timeout.\n\n"
    "curl works but Python fails because curl's TLS stack does not advertise PQ groups by default.\n\n"
    "Try one of these workarounds:\n\n"
    "1. **Classic groups OpenSSL config** — create a file with:\n"
    "   ```\n"
    "   openssl_conf = openssl_init\n\n"
    "   [openssl_init]\n"
    "   ssl_conf = ssl_sect\n\n"
    "   [ssl_sect]\n"
    "   system_default = system_default_sect\n\n"
    "   [system_default_sect]\n"
    "   Groups = x25519:secp256r1:secp384r1:x448\n"
    "   ```\n"
    "   and set `OPENSSL_CONF=/path/to/that/file` before running `hermes model`.\n\n"
    "2. **Force TLS 1.2** (when #44392 lands): `HERMES_TLS_MAX_VERSION=1.2 hermes model` (TLS 1.2 "
    "does not carry PQ `key_share` extensions).\n\n"
    f"Full troubleshooting guide: {_DOCS_URL}"
)
