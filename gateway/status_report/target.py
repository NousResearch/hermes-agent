"""Where status reports go: the launch environment's report URL, with its credential split off."""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

# Stamped by the hosting deployment and carrying its own credential in the query string, the shape
# of GATEWAY_RELAY_SLEEP_URL. Launch-only (hermes_constants.LAUNCH_ONLY_ENV_KEYS): no ``.env`` and no
# secret source can set it, so reports are on only where the deployment turned them on.
STATUS_REPORT_URL_ENV = "GATEWAY_STATUS_REPORT_URL"
_CREDENTIAL_PARAM = "t"
# What may follow ``Bearer `` (RFC 6750). A credential that decodes to anything else (a space, CR,
# LF) would end the header early or start another one.
_BEARER_TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/-]+=*")


@dataclass(frozen=True)
class ReportTarget:
    """The URL to POST to, without its credential, and the credential for the ``Authorization``
    header. Neither shows in the repr, so a target is safe to log."""

    url: str = field(repr=False)
    credential: str = field(repr=False)


def report_target_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[ReportTarget]:
    """The report target of the launch environment, or None when reports are off.

    Unset or blank is off, silently. A malformed value is off with a warning that names the key
    and never the value. Asked once per gateway start, so the warning needs no latch."""
    env = os.environ if environ is None else environ  # launch profile, pre-identity
    stamped = str(env.get(STATUS_REPORT_URL_ENV, "")).strip()
    if not stamped:
        return None
    target = _parse(stamped)
    if target is None:
        logger.warning(
            "status report: ignoring malformed %s (want an absolute https URL whose `%s` query "
            "parameter holds the credential)", STATUS_REPORT_URL_ENV, _CREDENTIAL_PARAM)
    return target


def _parse(stamped: str) -> Optional[ReportTarget]:
    try:
        parts = urllib.parse.urlsplit(stamped)
        # ``.port`` raises ValueError unless it is a number.
        addressed = (parts.scheme == "https" and bool(parts.hostname) and parts.username is None
                     and parts.password is None and parts.port != 0)
    except ValueError:
        return None
    # Split by hand rather than ``parse_qsl``: the other parameters go back out byte for byte, and
    # a ``+`` in the credential stays a ``+``.
    pairs = [pair for pair in parts.query.split("&") if pair]
    credentials = [pair.partition("=")[2] for pair in pairs if pair.partition("=")[0] == _CREDENTIAL_PARAM]
    if not addressed or len(credentials) != 1:
        return None
    credential = urllib.parse.unquote(credentials[0])
    if not _BEARER_TOKEN_RE.fullmatch(credential):
        return None
    rest = "&".join(pair for pair in pairs if pair.partition("=")[0] != _CREDENTIAL_PARAM)
    return ReportTarget(urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, rest, "")), credential)
