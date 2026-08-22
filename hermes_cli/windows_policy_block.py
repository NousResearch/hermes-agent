"""Detect Windows application-control policy blocks (Smart App Control / WDAC).

When Smart App Control or a WDAC/Application-Control policy blocks a spawn or
file-copy Hermes needs for install/update/launch, Windows fails the underlying
syscall instead of raising anything Hermes-specific. The only signal is a raw
``OSError``/``CalledProcessError`` whose message looks like:

    An Application Control policy has blocked this file. (os error 4551)

or a Win32 error code of 1260 (``ERROR_ACCESS_DISABLED_BY_POLICY``), the
documented code for the same class of block under Group Policy / WDAC. Field
reports (issue #87789) also show the code 4551, which isn't in the standard
Win32 system-error table but is what ``FormatMessage``/``std::io::Error``
surface for this exact block on affected machines — so both codes, plus the
stable text signatures Windows prints for either, are treated as equivalent.

Detection is guidance-only (see #87789): it never changes policy or retries
with elevated rights, it only recognizes the block and attaches an actionable
message next to the original error. The raw error is always preserved.
"""

from __future__ import annotations

import re

# Win32 error 1260 — ERROR_ACCESS_DISABLED_BY_POLICY. Documented code for
# Group Policy / WDAC application-control blocks.
WINERROR_ACCESS_DISABLED_BY_POLICY = 1260

# Observed in the wild (issue #87789 support logs) for Smart App Control
# blocking a spawn/copy. Not in the standard Win32 system-error table, but
# reproducible enough across reports to treat as a second known code.
WINERROR_SMART_APP_CONTROL_BLOCK = 4551

_KNOWN_WINERRORS = frozenset(
    {WINERROR_ACCESS_DISABLED_BY_POLICY, WINERROR_SMART_APP_CONTROL_BLOCK}
)

# Text Windows/antivirus stacks are known to emit for this block class.
# Matched case-insensitively against the exception's string form (and, for
# subprocess failures, its captured stdout/stderr) since the winerror
# attribute is not always populated — e.g. when the block surfaces inside a
# child process's own stderr rather than as a Python-level OSError.
_TEXT_SIGNATURES = (
    "application control policy has blocked this file",
    "blocked by group policy",
    "this program is blocked by group policy",
    "smart app control",
)

_OS_ERROR_SUFFIX_RE = re.compile(r"\(os error (\d+)\)")


def _winerror_code(exc: BaseException) -> int | None:
    """Best-effort Win32 error code for *exc*, walking the exception chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "winerror", None)
        if isinstance(code, int):
            return code
        text = str(current)
        match = _OS_ERROR_SUFFIX_RE.search(text)
        if match:
            return int(match.group(1))
        current = current.__cause__ or current.__context__
    return None


def _matching_text(*texts: str | None) -> str | None:
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        for signature in _TEXT_SIGNATURES:
            if signature in lowered:
                return text
    return None


def detect_policy_block(exc: BaseException, *extra_text: str | None) -> bool:
    """Whether *exc* (optionally plus *extra_text*, e.g. captured stderr) looks
    like a Windows application-control policy block rather than an ordinary
    I/O failure.
    """
    code = _winerror_code(exc)
    if code in _KNOWN_WINERRORS:
        return True
    return _matching_text(str(exc), *extra_text) is not None


def policy_block_guidance(context: str) -> str:
    """Actionable guidance block for a confirmed policy-block failure.

    *context* names the operation that was blocked (e.g. ``"update"``,
    ``"launch"``) so the message reads naturally next to the raw error.
    """
    return (
        "  This looks like Windows Smart App Control or an application-control\n"
        f"  policy blocking Hermes' Python during {context} — not a corrupted\n"
        "  install. A Windows update can silently re-enable Smart App Control.\n"
        "  See https://aka.ms/smartappcontrol to check its state, or ask your\n"
        "  IT administrator for an exemption if it's managed by policy.\n"
        "  This message is guidance only — Hermes has not changed any policy."
    )
