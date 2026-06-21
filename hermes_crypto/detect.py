"""Dependency-free detection of hermes_crypto envelopes.

Kept separate from :mod:`hermes_crypto.envelope` (which imports
``cryptography``) so a reader can cheaply ask "is this file encrypted?"
without pulling in the optional ``encryption`` extra. When encryption is not
in use this is the only code path that runs — every credential reader checks
:func:`is_encrypted` first and only decrypts on a match.
"""

from __future__ import annotations

MAGIC = b"HRMSENC"
ENV_MARKER = "#HERMES-ENCRYPTED-V1"
ENV_MARKER_BYTES = ENV_MARKER.encode("ascii")


def is_encrypted(data: bytes) -> bool:
    """Return True when *data* is a hermes_crypto envelope (binary or .env text)."""
    if not isinstance(data, (bytes, bytearray)):
        return False
    data = bytes(data)
    return data.startswith(MAGIC) or data.startswith(ENV_MARKER_BYTES)


def is_encrypted_env(data: bytes) -> bool:
    """Return True specifically for the text-framed encrypted ``.env`` form."""
    return isinstance(data, (bytes, bytearray)) and bytes(data).startswith(ENV_MARKER_BYTES)
