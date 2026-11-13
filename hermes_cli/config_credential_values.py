"""Credential VALUE checks for the ``.env`` / ``config.yaml`` writers.

The agent only ever sees credentials through the redactor (``agent.redact``) and the display
maskers (``mask_secret``): ``sk-pro...WXYZ``, ``***``, ``«redacted:sk-…»``. A writer that persists
one of those as the credential turns a working key into a dead string and the next provider call
fails with 401 (#42727). Recognising Hermes's own mask outputs at the write edge keeps a redacted
display value from ever becoming the stored secret. This is hardening at the edge of a heuristic
(the redactor), not a boundary.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Callable, Iterator

# ``mask_secret`` (``sk-p...7890``, ``_mask_token``'s ``sk-pro...WXYZ``) and hand-truncated
# previews of it (``sk-...``). Short on purpose: no real key is a few chars around ``...``.
_ELLIPSIS_MASK_RE = re.compile(r"\S{0,8}(?:\.\.\.|…)\S{0,8}")
_BARE_MASKS = frozenset({"***", "[redacted]", "<redacted>", "(redacted)", "redacted"})
_ENV_REF_RE = re.compile(r"\$\{[^}]+\}")


def is_redacted_credential_value(value: Any) -> bool:
    """True when ``value`` is a redaction/display mask rather than a usable secret."""
    if not isinstance(value, str):
        return False
    candidate = value.strip().strip("\"'")
    if not candidate or _ENV_REF_RE.fullmatch(candidate):
        return False
    lowered = candidate.lower()
    if lowered in _BARE_MASKS or lowered.startswith(("«redacted", "[redacted")):
        return True
    if "redacted" in lowered and len(candidate) <= 40:  # ``****redacted****`` and kin
        return True
    return _ELLIPSIS_MASK_RE.fullmatch(candidate) is not None


def _is_credential_env_name(key: str) -> bool:
    from hermes_cli.config import OPTIONAL_ENV_VARS, _is_secret_config_key

    meta = OPTIONAL_ENV_VARS.get(key.upper())
    return _is_secret_config_key(key) or (isinstance(meta, dict) and meta.get("password") is True)


def refuse_redacted_env_value(key: str, value: Any) -> None:
    """Raise ``ValueError`` when a credential ``.env`` entry would store a redacted mask."""
    if is_redacted_credential_value(value) and _is_credential_env_name(key):
        raise ValueError(
            f"Refusing to save {key.upper()}: the value is a redacted display mask, not the "
            "credential. Enter the real secret (or leave it unchanged).")


def _redacted_credential_paths(new: Any, old: Any, path: tuple[str, ...] = ()) -> Iterator[str]:
    from hermes_cli.config import _is_secret_config_key

    if isinstance(new, dict):
        items = ((str(k), v, old.get(k) if isinstance(old, dict) else None) for k, v in new.items())
    elif isinstance(new, list):
        items = ((str(i), v, old[i] if isinstance(old, list) and i < len(old) else None)
                 for i, v in enumerate(new))
    else:
        return
    for key, child, prior in items:
        # A mask already on disk is reported by the reader, not re-refused on every unrelated save.
        if _is_secret_config_key(key) and child != prior and is_redacted_credential_value(child):
            yield ".".join(path + (key,))
        else:
            yield from _redacted_credential_paths(child, prior, path + (key,))


def refuse_redacted_config_credentials(new: Any, read_old: Callable[[], Any]) -> None:
    """Raise ``ConfigValueError`` when ``new`` sets a credential key to a redacted mask that the
    config on disk (``read_old()``, read only when a mask is present) did not already hold."""
    if not any(_redacted_credential_paths(new, None)):
        return
    bad = list(_redacted_credential_paths(new, read_old()))
    if bad:
        from hermes_cli.config_backend import ConfigValueError

        shown = ", ".join(bad[:5]) + (f", +{len(bad) - 5} more" if len(bad) > 5 else "")
        raise ConfigValueError(
            f"Refusing to save redacted credential mask(s) at {shown}: a masked display value is "
            "not the credential. Enter the real secret or a ${ENV_VAR} reference.")


def _check_non_ascii_credential(key: str, value: str) -> str:
    """Strip non-ASCII characters from a credential (HTTP header values must be ASCII) and warn.
    Lookalike glyphs typically come from copy-pasting out of a PDF or rich-text editor."""
    if value.isascii():
        return value

    bad_chars = [f"  position {i}: {ch!r} (U+{ord(ch):04X})" for i, ch in enumerate(value) if ord(ch) > 127]
    sanitized = value.encode("ascii", errors="ignore").decode("ascii")

    print(
        f"\n  Warning: {key} contains non-ASCII characters that will break API requests.\n"
        f"  This usually happens when copy-pasting from a PDF, rich-text editor,\n"
        f"  or web page that substitutes lookalike Unicode glyphs for ASCII letters.\n\n"
        + "\n".join(f"  {line}" for line in bad_chars[:5])
        + ("\n  ... and more" if len(bad_chars) > 5 else "")
        + "\n\n  The non-ASCII characters have been stripped automatically.\n"
        "  If authentication fails, re-copy the key from the provider's dashboard.\n",
        file=sys.stderr)
    return sanitized
