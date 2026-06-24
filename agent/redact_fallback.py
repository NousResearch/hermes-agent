"""Small last-ditch secret redactor for hard safety boundaries.

This module intentionally has no dependency on :mod:`agent.redact`.  Callers use
it only when the full shared redactor fails; it preserves a minimal set of
credential guards so model-facing/tool-facing boundaries do not fail open.
"""

from __future__ import annotations

import re

_SENSITIVE_FIELD_RE = re.compile(
    r'("(?:api_?key|token|secret|password|access_token|refresh_token|auth_token|bearer|authorization|private_key)")'
    r'\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)

_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"\b([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
)

_AUTH_HEADER_RE = re.compile(r"(\bAuthorization:\s*Bearer\s+)(\S+)", re.IGNORECASE)
_BEARER_RE = re.compile(r"(\bBearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE)

_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-[A-Za-z0-9_-]{10,}"
    r"|gh[pousr]_[A-Za-z0-9_]{10,}"
    r"|github_pat_[A-Za-z0-9_]{10,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|hf_[A-Za-z0-9]{10,}"
    r"|glpat-[A-Za-z0-9_-]{10,}"
    r"|AIza[A-Za-z0-9_-]{30,}"
    r"|pplx-[A-Za-z0-9]{10,}"
    r"|fal_[A-Za-z0-9_-]{10,}"
    r"|fc-[A-Za-z0-9]{10,}"
    r"|bb_live_[A-Za-z0-9_-]{10,}"
    r"|gAAAA[A-Za-z0-9_=-]{20,}"
    r"|AKIA[A-Z0-9]{16}"
    r"|sk_live_[A-Za-z0-9]{10,}"
    r"|sk_test_[A-Za-z0-9]{10,}"
    r"|rk_live_[A-Za-z0-9]{10,}"
    r"|SG\.[A-Za-z0-9_-]{10,}"
    r"|npm_[A-Za-z0-9]{10,}"
    r"|pypi-[A-Za-z0-9_-]{10,}"
    r"|dop_v1_[A-Za-z0-9]{10,}"
    r"|doo_v1_[A-Za-z0-9]{10,}"
    r"|am_[A-Za-z0-9_-]{10,}"
    r"|sk_[A-Za-z0-9_]{10,}"
    r"|tvly-[A-Za-z0-9]{10,}"
    r"|exa_[A-Za-z0-9]{10,}"
    r"|gsk_[A-Za-z0-9]{10,}"
    r"|hsk-[A-Za-z0-9]{10,}"
    r"|xai-[A-Za-z0-9]{30,}"
    r")"
    r"(?![A-Za-z0-9_-])"
)


def redact_sensitive_text_fallback(text: object) -> str:
    """Return ``text`` with high-risk secret shapes replaced by ``[REDACTED]``.

    This is deliberately narrower than the canonical redactor.  It catches
    sensitive JSON fields, env-style assignments, bearer tokens, and common
    vendor prefixes without trying to preserve debuggability.
    """
    redacted = "" if text is None else str(text)
    if not redacted:
        return redacted
    redacted = _SENSITIVE_FIELD_RE.sub(lambda m: f'{m.group(1)}: "[REDACTED]"', redacted)
    redacted = _ENV_ASSIGN_RE.sub(lambda m: f"{m.group(1)}={m.group(2)}[REDACTED]{m.group(2)}", redacted)
    redacted = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}[REDACTED]", redacted)
    redacted = _BEARER_RE.sub(lambda m: f"{m.group(1)}[REDACTED]", redacted)
    redacted = _PREFIX_RE.sub("[REDACTED]", redacted)
    return redacted
