"""Secret and personal-data redaction.

Every string this pipeline is about to persist (to the candidate database),
send to a model, write to a target file, or put in a Slack notification
passes through :func:`redact_text` first. This module has no dependency on
the hermes-agent package — it is a standalone, conservative reimplementation
of the same idea as ``agent/redact.py`` and the
``self-improvement-notify`` plugin's ``_SECRET_PATTERNS``, sized for this
pipeline's much narrower job: masking credential-shaped and personal-data-
shaped substrings inside short lesson snippets, not full HTTP traffic.

Two independent guards live here:

* :func:`redact_text` — masks matches in place. Used on every candidate
  snippet before it is stored, sent to a model, or written anywhere.
* :func:`find_secret_leftovers` — a second, read-only pass used right before
  a target file write (see ``writers.py``). If it finds anything, the write
  is refused outright rather than trusting that redaction already caught
  everything — defense in depth for the one place a miss would land in a
  file George reads every session.
"""

from __future__ import annotations

import re

# --- Secret / credential patterns -------------------------------------------
# Adapted from agent/redact.py and the self-improvement-notify plugin's
# _SECRET_PATTERNS: vendor-prefixed API tokens, bearer tokens, key=value
# assignments, and long opaque blobs.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "vendor-token",
        re.compile(
            r"(?i)\b(?:xox[abposr]|ghp|gho|ghu|ghs|ghr|github_pat|lin_api|"
            r"sk-ant|sk-proj|sk-live|sk-test|sk|dp\.st|dp\.ct|rk_live|pk_live)"
            r"[_\-][A-Za-z0-9_\-]{8,}"
        ),
    ),
    ("aws-key", re.compile(r"\bAKIA[0-9A-Z]{12,}")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}")),
    (
        "key-value-secret",
        re.compile(
            r"(?i)\b(?:api[_\-]?key|token|secret|password|passwd|access[_\-]?key)"
            r"\b\s*[:=]\s*\S{6,}"
        ),
    ),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("opaque-blob", re.compile(r"\b[A-Za-z0-9_\-]{40,}\b")),
]

# --- Personal / customer data patterns --------------------------------------
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "credit-card",
        re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    ),
    (
        "phone",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    ),
    ("ip-address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

_ALL_PATTERNS = _SECRET_PATTERNS + _PII_PATTERNS


def redact_text(text: str) -> str:
    """Return *text* with every recognized secret/PII pattern masked.

    Never raises: a redaction bug must never crash a pipeline pass, and a
    crash-to-empty-string here is safer than crash-to-unhandled-exception
    (an empty candidate is simply dropped downstream).
    """
    try:
        out = str(text or "")
    except Exception:
        return ""
    for name, pattern in _ALL_PATTERNS:
        try:
            out = pattern.sub(f"[redacted:{name}]", out)
        except Exception:
            continue
    return out


def find_secret_leftovers(text: str) -> list[str]:
    """Return the names of any secret pattern still present in *text*.

    Only checks the *secret* patterns (not the broader PII set) — this is
    the last-chance gate before a write, and it should refuse a write on
    "this looks like a live credential", not on "this looks like an email
    address a rule legitimately needed to mention".
    """
    try:
        haystack = str(text or "")
    except Exception:
        return ["unreadable-content"]
    found = []
    for name, pattern in _SECRET_PATTERNS:
        try:
            if pattern.search(haystack):
                found.append(name)
        except Exception:
            continue
    return found


def truncate(text: str, limit: int) -> str:
    """Collapse whitespace and cut *text* to *limit* characters."""
    try:
        flat = " ".join(str(text or "").split())
    except Exception:
        return ""
    if len(flat) > limit:
        flat = flat[: limit].rstrip() + "…"
    return flat
