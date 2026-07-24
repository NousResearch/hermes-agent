"""Central redaction, public error classification, and private diagnostics.

Nothing written through this module contains credentials or network identity.
Detailed diagnostics are disabled unless ``HERMES_TOR_DEBUG=1`` and are then
written to a local, owner-only file.  The debug file is never an MCP output.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


REDACTED = "[REDACTED]"
_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_HOME = re.compile(r"(?<!\w)(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)(?:[/\\][^\s,;:)]+)*")
_URL = re.compile(r"\b(?:https?|socks5h?|ftp)://[^\s<>\"']+")
_BRIDGE = re.compile(r"(?im)^\s*(?:Bridge\s+)?(?:obfs\d|webtunnel|snowflake|meek|scramblesuit)\s+.*$")
_CERT = re.compile(r"(?i)\bcert=\S+")
_ENV = re.compile(r"(?i)\b([A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASS|KEY|PROXY))=\S+")


def _safe_url(match: re.Match[str]) -> str:
    value = match.group(0)
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        if parts.port:
            host += f":{parts.port}"
        # Userinfo and query/fragment are deliberately never retained.
        return urlunsplit((parts.scheme, host, parts.path, "", "")) + (
            "?[REDACTED]" if parts.query else ""
        )
    except (ValueError, UnicodeError):
        return REDACTED


def redact(value: object, *, redact_ips: bool = True) -> str:
    """Return a serialization-safe representation of potentially secret text."""
    text = str(value)
    text = _BRIDGE.sub("[REDACTED BRIDGE]", text)
    text = _CERT.sub("cert=[REDACTED]", text)
    text = _ENV.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _URL.sub(_safe_url, text)
    text = _HOME.sub("[REDACTED HOME]", text)
    if redact_ips:
        text = _IP.sub("[REDACTED IP]", text)
    return text


class RedactingFilter(logging.Filter):
    """Redact both the format string and arguments before any handler sees them."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(redact(v) for v in record.args) if isinstance(record.args, tuple) else redact(record.args)
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger


@dataclass(frozen=True)
class PublicError:
    code: str
    message: str


def classify_error(error: object) -> PublicError:
    """Map implementation failures to stable, deliberately terse public errors."""
    text = str(error).lower()
    if "not installed" in text or "binary not found" in text:
        return PublicError("TOR_NOT_INSTALLED", "Install the Tor bundle and retry.")
    if "timeout" in text or "bootstrap" in text:
        return PublicError("TOR_BOOTSTRAP_FAILED", "Check connectivity or refresh bridges, then retry.")
    if "not running" in text or "connection refused" in text:
        return PublicError("TOR_NOT_RUNNING", "Start Tor and retry.")
    if "bridge" in text:
        return PublicError("TOR_BRIDGE_INVALID", "Replace the bridge configuration and retry.")
    if "http" in text or "network" in text or "connect" in text:
        return PublicError("TOR_NETWORK_ERROR", "Check network access and retry.")
    if isinstance(error, (ValueError, TypeError)):
        return PublicError("INVALID_REQUEST", "Check the supplied values and retry.")
    return PublicError("TOR_INTERNAL_ERROR", "Retry; contact the local administrator if it persists.")


def private_diagnostic(component: str, error: object) -> None:
    """Opt-in owner-only diagnostic logging; command arguments remain redacted."""
    if os.environ.get("HERMES_TOR_DEBUG", "").lower() not in {"1", "true", "yes"}:
        return
    path = Path(os.environ.get("HERMES_TOR_DEBUG_LOG", Path.home() / ".hermes/tor/debug.log"))
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, f"{component}: {redact(error)}\n".encode("utf-8", "replace"))
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def require_local_admin(token: str | None) -> None:
    """Authorize a separately invoked, local administrative interface."""
    import hmac
    expected = os.environ.get("HERMES_TOR_ADMIN_TOKEN")
    if not expected or not token or not hmac.compare_digest(token, expected):
        raise PermissionError("Local administrative authorization required")
