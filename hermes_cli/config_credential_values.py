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
from typing import Any, Callable, Iterable, Iterator, Mapping

# ``mask_secret`` (``sk-p...7890``, ``_mask_token``'s ``sk-pro...WXYZ``: a short head, ``...``,
# then exactly four tail chars) and hand-truncated previews of it (``sk-...``).
_ELLIPSIS_MASK_RE = re.compile(r"\S{0,8}(?:\.\.\.|…)(?:\S{4})?")
_STARS_RE = re.compile(r"\*{3,}\S{0,4}")  # ``***``, ``sk-****WXYZ`` after the prefix strip
_BARE_MASKS = frozenset({"[redacted]", "<redacted>", "(redacted)", "redacted"})
# ``****redacted****`` and kin: the whole value is the word fenced by filler, so a real password
# that merely contains the word (``MyRedactedPassword123!``) is not a mask.
_FENCED_REDACTED_RE = re.compile(r"(?i)([*#=_-])\1+\s*redacted\s*\1+")
_ENV_REF_RE = re.compile(r"\$\{[^}]+\}")
# The redactor keeps context around the masked part: ``Bearer ***`` (auth header), ``123456789:***``
# (Telegram bot id in file_read mode), ``xoxb-***`` (vendor prefix). Strip it before matching.
_AUTH_SCHEME_RE = re.compile(r"(?i)(?:bearer|basic|token|digest)\s+")
_KEPT_PREFIX_RE = re.compile(r"\d+:|[A-Za-z][A-Za-z0-9]{0,7}[-_]")


def _is_marker_word(candidate: str) -> bool:
    lowered = candidate.lower()
    return (lowered in _BARE_MASKS or lowered.startswith(("«redacted", "[redacted"))
            or bool(_FENCED_REDACTED_RE.fullmatch(candidate)))


def _is_bare_mask(candidate: str, *, shapes: bool = True) -> bool:
    if _is_marker_word(candidate):
        return True
    return shapes and bool(_STARS_RE.fullmatch(candidate) or _ELLIPSIS_MASK_RE.fullmatch(candidate))


def _mask_candidate(value: Any) -> str:
    """``value`` with quotes and the context the redactor keeps in front (``Bearer ``) removed;
    ``""`` for a non-string, an empty value or a ``${VAR}`` reference (never a mask)."""
    if not isinstance(value, str):
        return ""
    candidate = value.strip().strip("\"'")
    scheme = _AUTH_SCHEME_RE.match(candidate)
    if scheme:
        candidate = candidate[scheme.end():]
    return "" if _ENV_REF_RE.fullmatch(candidate) else candidate


def is_redacted_credential_value(value: Any, *, shapes: bool = True) -> bool:
    """True when ``value`` is a redaction/display mask rather than a usable secret. With
    ``shapes=False`` only the marker words (``[REDACTED]``, ``«redacted:…»``) count, not the
    ellipsis/star shapes a human-chosen password can legitimately take."""
    candidate = _mask_candidate(value)
    if not candidate:
        return False
    # ``*** sk-pro...WXYZ``: the redactor masks a multi-token header value token by token.
    return all(_is_masked_token(token, shapes) for token in candidate.split())


def _is_masked_token(candidate: str, shapes: bool) -> bool:
    if _is_bare_mask(candidate, shapes=shapes):
        return True
    prefix = _KEPT_PREFIX_RE.match(candidate)
    return bool(prefix) and _is_bare_mask(candidate[prefix.end():], shapes=shapes)


# Human-chosen secrets (``SUDO_PASSWORD``, ``dashboard.basic_auth.password``) have no prescribed
# alphabet, so ``Hello...2026`` or ``****`` can be the real password. For these the mask SHAPES
# are refused only when the value is what the maskers show for the credential stored now: that is
# the write-back #42727 is about, and it is the only way the agent can have seen such a value.
_PASSWORD_NAME_RE = re.compile(r"(?:^|_)(?:password|passwd|passphrase)$")


def _is_password_name(key: str) -> bool:
    return bool(_PASSWORD_NAME_RE.search(key.rsplit(".", 1)[-1].lower().replace("-", "_")))


def _masks_of(stored: Any) -> set[str]:
    """Every display value Hermes's redactor/maskers produce for the secret ``stored``."""
    if not isinstance(stored, str) or not stored or _ENV_REF_RE.fullmatch(stored.strip()):
        return set()
    from agent.redact import _mask_token, _mask_token_nonreusable, mask_secret

    return {"***", mask_secret(stored), _mask_token(stored), _mask_token_nonreusable(stored)}


def _is_mask_write_back(key: str, value: Any, stored: Iterable[Any]) -> bool:
    """``value`` would replace a credential with a display mask: any mask for a provider-style
    credential name; for a password name, a marker word or a mask of a ``stored`` value."""
    if not _is_password_name(key):
        return is_redacted_credential_value(value)
    if is_redacted_credential_value(value, shapes=False):
        return True
    candidate = _mask_candidate(value)
    olds = [old for old in stored if isinstance(old, str) and old != value]
    return bool(candidate) and any(candidate in _masks_of(old) for old in olds)


def _is_credential_env_name(key: str) -> bool:
    from hermes_cli.config import OPTIONAL_ENV_VARS, _is_secret_config_key

    meta = OPTIONAL_ENV_VARS.get(key.upper())
    return _is_secret_config_key(key) or (isinstance(meta, dict) and meta.get("password") is True)


def _stored_env_values(key: str) -> tuple[Any, ...]:
    """What ``key`` holds now (``.env``, then this profile's environment): the values whose
    masks the agent can have been shown."""
    from agent.secret_scope import UnscopedSecretError
    from hermes_cli.config import _scoped_environ_get, load_env

    try:
        scoped = _scoped_environ_get(key)
    except UnscopedSecretError:  # multiplexed with no scope: no ambient env to compare with
        scoped = None
    return load_env().get(key), scoped


def refuse_redacted_env_value(key: str, value: Any) -> None:
    """Raise ``ValueError`` when a credential ``.env`` entry would store a redacted mask."""
    if not (is_redacted_credential_value(value) and _is_credential_env_name(key)):
        return
    if _is_mask_write_back(key, value, _stored_env_values(key) if _is_password_name(key) else ()):
        raise ValueError(
            f"Refusing to save {key.upper()}: the value is a redacted display mask, not the "
            "credential. Enter the real secret (or leave it unchanged).")


def _list_item_identity(item: Any) -> str:
    """A list entry's location, stable across deletes/reorders: its ``name``/``id``/``key``."""
    if isinstance(item, dict):
        for field in ("name", "id", "key"):
            if isinstance(item.get(field), (str, int)):
                return f"[{field}={item[field]}]"
    return "[*]"


def _credential_leaves(doc: Any, path: tuple[str, ...] = (),
                       where: tuple[str, ...] = ()) -> Iterator[tuple[str, tuple[str, ...], Any]]:
    """Yield ``(display_path, location, value)`` for each scalar credential leaf, where
    ``location`` replaces list indexes with the entry's identity."""
    from hermes_cli.config import _is_secret_config_key

    if isinstance(doc, dict):
        items = ((str(k), str(k), v) for k, v in doc.items())
    elif isinstance(doc, list):
        items = ((str(i), _list_item_identity(v), v) for i, v in enumerate(doc))
    else:
        return
    for key, loc, child in items:
        if _is_secret_config_key(key) and not isinstance(child, (dict, list)):
            yield ".".join(path + (key,)), where + (loc,), child
        else:
            yield from _credential_leaves(child, path + (key,), where + (loc,))


def refuse_redacted_config_credentials(new: Any, read_old: Callable[[], Any]) -> None:
    """Raise ``ConfigValueError`` when ``new`` sets a credential key to a redacted mask that the
    config on disk (``read_old()``, read only when a mask is present) did not already hold there.

    Pre-existing masks are matched by (location, value), with list indexes replaced by the entry's
    ``name``/``id``/``key``: deleting or reordering ``custom_providers`` keeps an old mask's
    location, while the same mask string at any other credential path is still a new write. A
    password leaf is refused only for a mask of the value stored at its location (see
    ``_PASSWORD_NAME_RE``)."""
    masked = [leaf for leaf in _credential_leaves(new) if is_redacted_credential_value(leaf[2])]
    if not masked:
        return
    on_disk: dict[tuple[str, ...], list[Any]] = {}
    for _, where, value in _credential_leaves(read_old()):
        on_disk.setdefault(where, []).append(value)
    bad = [path for path, where, value in masked
           if value not in on_disk.get(where, ()) and _is_mask_write_back(path, value, on_disk.get(where, ()))]
    if bad:
        from hermes_cli.config_backend import ConfigValueError

        shown = ", ".join(bad[:5]) + (f", +{len(bad) - 5} more" if len(bad) > 5 else "")
        raise ConfigValueError(
            f"Refusing to save redacted credential mask(s) at {shown}: a masked display value is "
            "not the credential. Enter the real secret or a ${ENV_VAR} reference.")


def _holds_mask(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_holds_mask(v) for v in value.values())
    if isinstance(value, list):
        return any(_holds_mask(v) for v in value)
    return is_redacted_credential_value(value)


def refuse_redacted_config_sets(sets: Mapping[str, Any], read_old: Callable[[], Any]) -> None:
    """:func:`refuse_redacted_config_credentials` for a keyed write (``write_config_keys``,
    ``Changes.set``): the dotted ``sets`` are checked as the nested document they write."""
    if not any(_holds_mask(value) for value in sets.values()):
        return  # no hermes_cli.config import for ordinary keyed writes (PM runtime)
    from hermes_cli.config import _split_key_path

    doc: dict[str, Any] = {}
    for dotted, value in sets.items():
        *parents, leaf = _split_key_path(dotted) or [dotted]
        node = doc
        for part in parents:
            child = node.setdefault(part, {})
            node = child if isinstance(child, dict) else {}
        node[leaf] = value
    refuse_redacted_config_credentials(doc, read_old)


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
