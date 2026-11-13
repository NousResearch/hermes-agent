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

# ``mask_secret`` (``sk-p...7890``, ``_mask_token``'s ``sk-pro...WXYZ``: a short head, ``...``,
# then exactly four tail chars) and hand-truncated previews of it (``sk-...``).
_ELLIPSIS_MASK_RE = re.compile(r"\S{0,8}(?:\.\.\.|…)(?:\S{4})?")
_STARS_RE = re.compile(r"\*{3,}")
_BARE_MASKS = frozenset({"[redacted]", "<redacted>", "(redacted)", "redacted"})
_ENV_REF_RE = re.compile(r"\$\{[^}]+\}")
# The redactor keeps context around the masked part: ``Bearer ***`` (auth header), ``123456789:***``
# (Telegram bot id in file_read mode), ``xoxb-***`` (vendor prefix). Strip it before matching.
_AUTH_SCHEME_RE = re.compile(r"(?i)(?:bearer|basic|token|digest)\s+")
_KEPT_PREFIX_RE = re.compile(r"\d+:|[A-Za-z][A-Za-z0-9]{0,7}[-_]")


def _is_bare_mask(candidate: str) -> bool:
    lowered = candidate.lower()
    if lowered in _BARE_MASKS or lowered.startswith(("«redacted", "[redacted")):
        return True
    if "redacted" in lowered and len(candidate) <= 40:  # ``****redacted****`` and kin
        return True
    return bool(_STARS_RE.fullmatch(candidate) or _ELLIPSIS_MASK_RE.fullmatch(candidate))


def is_redacted_credential_value(value: Any) -> bool:
    """True when ``value`` is a redaction/display mask rather than a usable secret."""
    if not isinstance(value, str):
        return False
    candidate = value.strip().strip("\"'")
    scheme = _AUTH_SCHEME_RE.match(candidate)
    if scheme:
        candidate = candidate[scheme.end():]
    if not candidate or _ENV_REF_RE.fullmatch(candidate):
        return False
    if _is_bare_mask(candidate):
        return True
    prefix = _KEPT_PREFIX_RE.match(candidate)
    return bool(prefix) and _is_bare_mask(candidate[prefix.end():])


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


def _masked_credential_leaves(doc: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    from hermes_cli.config import _is_secret_config_key

    if isinstance(doc, dict):
        items = ((str(k), v) for k, v in doc.items())
    elif isinstance(doc, list):
        items = ((str(i), v) for i, v in enumerate(doc))
    else:
        return
    for key, child in items:
        if _is_secret_config_key(key) and is_redacted_credential_value(child):
            yield ".".join(path + (key,)), child
        else:
            yield from _masked_credential_leaves(child, path + (key,))


def refuse_redacted_config_credentials(new: Any, read_old: Callable[[], Any]) -> None:
    """Raise ``ConfigValueError`` when ``new`` sets a credential key to a redacted mask that the
    config on disk (``read_old()``, read only when a mask is present) did not already hold.

    Pre-existing masks are matched by value, not position: deleting or reordering a list entry
    (``custom_providers``) moves an old mask to a new index without it being newly written."""
    masked = list(_masked_credential_leaves(new))
    if not masked:
        return
    on_disk = {value for _, value in _masked_credential_leaves(read_old())}
    bad = [path for path, value in masked if value not in on_disk]
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
