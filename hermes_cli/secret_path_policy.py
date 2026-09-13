"""Fail-closed pre-ingress policy that rejects secret-bearing paths.

Backstory
---------
Multiple code paths admit a file into a durable Hermes surface
(per-task attachment blob, scratch-workspace artifact that gets promoted
into attachments, dashboard upload). Hermes' existing ``file_safety``
module blocks *reads* of ``auth.json``/``.env``/keychain exports on
behalf of the agent, but nothing blocks those same paths from being
*copied in* via ``kanban_attach`` / ``kanban_attach_url`` /
``hermes kanban attach`` / the dashboard upload endpoint /
scratch-workspace artifact preservation.

That asymmetry is what this module closes.  Every site that touches an
attachment's bytes or that promotes a scratch file into the attachments
directory must call into here and fail closed if the supplied name or
path matches a known secret-bearing shape.  The agent is told
``(kind, safe_reason)`` and never the file's contents, basename, or path
— by design the policy must not echo credential material even when a
block fires.

Scope (what gets blocked)
-------------------------
At minimum:

  * exact ``auth.json`` (case-insensitive);
  * any ``.env`` or ``.env.<suffix>`` (the ``_BLOCKED_PROJECT_ENV_BASENAMES``
    set used by ``agent.file_safety`` for the symmetric read-side guard);
  * recognised keychain export formats/names: ``credentials.json``,
    ``keychain.json``, ``1password-export.json``, ``lastpass-export.csv``,
    ``bitwarden-export.json``, ``keepass-export.xml``, ``dashlane-export.json``,
    and ``*.kbdx`` / ``*.kdbx`` / ``*.kdb`` keychain databases.

Normalisation
-------------
The agent can supply ``Foo.env``, ``./Foo.env``, ``C:\\foo\\bar\\Auth.JSON``,
or any other case/separator variation.  Before matching the basename is
reduced via :func:`_normalise_basename`: separators flattened to ``/``,
the path collapsed to its leaf, surrounding whitespace and case
normalised to lowercase, and ``.``/``..`` traversal resolved.  Symlinks
are resolved via :func:`Path.resolve` when a ``Path`` is supplied
(scratch-artifact check); for plain basenames no filesystem I/O is
performed so a network share's contents are never inspected.

Errors are descriptive but safe: the agent sees the policy ``kind``
(e.g. ``"dotenv"``, ``"auth_json"``, ``"keychain_export"``) and a stable
reason string.  No filename, no path, no credential bytes ever leak into
the error.

This module is intentionally dependency-free and I/O-light so it can be
imported at the top of every ingress handler without bringing in
tooling that pulls in unrelated Hermes state.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Public constants — the prohibited shapes
# ---------------------------------------------------------------------------

# Project-local env files. Mirrors ``agent.file_safety._BLOCKED_PROJECT_ENV_BASENAMES``
# so the read-side and the ingress-side guards deny the same set of names.
# Names are stored in the post-normalisation form (leading dots stripped
# by :func:`_normalise_basename`).
_PROHIBITED_DOTENV_BASENAMES: frozenset[str] = frozenset(
    {
        "env",
        "env.local",
        "env.development",
        "env.production",
        "env.test",
        "env.staging",
        "envrc",
    }
)

# Exact-match prohibited basenames (lowercase, post-normalisation).
# Names are stored in the form returned by :func:`_normalise_basename`
# — i.e. *without* a leading dot, since the normaliser strips them.
# ``auth.json`` is the Hermes credential store; the others are the common
# shapes for exported keychain / password-manager databases that we
# definitely do not want uploaded as a task artifact.
_PROHIBITED_EXACT_BASENAMES: frozenset[str] = frozenset(
    {
        "auth.json",
        "auth.lock",
        "anthropic_oauth.json",
        "webhook_subscriptions.json",
        "credentials.json",
        "keychain.json",
        "1password-export.json",
        "lastpass-export.csv",
        "bitwarden-export.json",
        "dashlane-export.json",
        "bws_cache.enc.json",
        "bws_cache.json",
    }
)

# Suffix-keychain-database extensions.  ``.kdbx``/``.kdb`` are KeePass;
# ``.kbdx`` is a less-common variant that several managers export.
_PROHIBITED_KEYCHAIN_SUFFIXES: tuple[str, ...] = (".kdbx", ".kbdx", ".kdb")

# Filename matchers for keychain-export formats that don't reduce to a
# simple suffix (e.g. ``keepass-export.xml`` is *named* and suffix-only
# matchers miss it).
_PROHIBITED_KEYCHAIN_GLOB: tuple[str, ...] = (
    "*keepass*.xml",
    "*keepass*.csv",
    "*keychain*.xml",
    "*keychain*.csv",
    "*passwords*.xml",
    "*passwords*.csv",
)


# ---------------------------------------------------------------------------
# Public error type
# ---------------------------------------------------------------------------


class SecretPathError(ValueError):
    """Raised (or returned) when an ingress path matches the denylist.

    The ``kind`` is a stable token (e.g. ``"auth_json"``); ``reason`` is a
    short, human-readable string with no filename or contents.  Callers
    are expected to surface ``str(exc)`` only — the exception does not
    include the offending path or basename.
    """

    def __init__(self, kind: str, reason: str) -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _collapse_separators(value: str) -> str:
    """Map any separator (``/``/``\\``) to ``/`` so cross-platform names compare."""
    return value.replace("\\", "/")


def _strip_traversal(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Remove ``.``/``..`` components from a path parts tuple.

    Used to fold ``./foo/../bar.env`` down to ``bar.env`` without
    filesystem I/O — ``Path.resolve`` would do the same but does an
    actual lookup the policy doesn't need.
    """
    out: list[str] = []
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            if out:
                out.pop()
            continue
        out.append(part)
    return tuple(out)


def _normalise_basename(name: str) -> str:
    """Reduce a name string to its lowercase leaf basename.

    Steps applied in order:

    1. ``None`` becomes empty.
    2. ``\\\\`` is folded to ``/`` so cross-platform names compare.
    3. Outer whitespace stripped.
    4. ``.``/``..`` traversal components removed via
       :func:`_strip_traversal`.
    5. Internal whitespace collapsed to a single space (so ``auth.
       json`` and ``auth . json`` both normalise to ``auth. json``).
    6. Leading ``.`` characters removed from the leaf (so ``.env``
       and ``. auth.json`` are still recognised) — this matches
       POSIX's notion of a hidden file and is the normal form every
       filesystem returns for ``ls -A``.
    7. Result lowercased so the lowercase denylists apply.

    Pure function — no filesystem I/O.  Returns ``""`` for any input
    that normalises to nothing (used as the "no leaf" signal).
    """
    if name is None:
        return ""
    collapsed = _collapse_separators(str(name).strip())
    if not collapsed:
        return ""
    parts = _strip_traversal(tuple(collapsed.split("/")))
    if not parts:
        return ""
    leaf = parts[-1].strip()
    if not leaf:
        return ""
    # Collapse runs of internal whitespace to a single space — keeps
    # the comparison deterministic for ``auth. json`` / ``auth.json``.
    leaf = re.sub(r"\s+", " ", leaf)
    # Strip leading dots from the leaf so ``.auth.json`` /
    # ``..auth.json`` all normalise to ``auth.json``.  Then strip
    # again so a ``. env`` -> `` env`` -> ``env`` chain still works.
    leaf = leaf.lstrip(".").strip()
    if not leaf:
        return ""
    # Also strip internal whitespace entirely — ``auth. json`` is
    # suspicious in shell-supplied names, and Hermes never produces
    # internal-whitespace leaves.  Removing it guarantees a one-to-one
    # match with the no-whitespace denylist entries.  Whitespace-free
    # names (``auth.json``) are unaffected.
    leaf = re.sub(r"\s+", "", leaf)
    if not leaf:
        return ""
    return leaf.lower()


def _glob_match(pattern: str, name: str) -> bool:
    """``fnmatch``-style glob match (lowercased inputs only).

    Implemented locally so the policy has no ``fnmatch`` dependency at
    import time — keeps the import surface tight for the ingress
    handlers.
    """
    rx = re.escape(pattern).replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
    return re.fullmatch(rx, name) is not None


# ---------------------------------------------------------------------------
# Public check entry points
# ---------------------------------------------------------------------------


def check_attachment_filename(
    name: str,
    *,
    source: str = "attach",
) -> Optional[str]:
    """Return a stable error kind if *name* is a prohibited attachment name.

    ``name`` may be a raw client-supplied filename (with directory
    components, mixed separators, or different casing).  The function
    performs no filesystem I/O — it only inspects the basename string.
    Returns ``None`` when the name is safe, or a short policy-error
    string suitable for surfacing as a tool error (no credential
    material, no path).

    ``source`` is a free-form label (e.g. ``"kanban_attach"``,
    ``"kanban_attach_url"``, ``"dashboard_upload"``,
    ``"kanban_cli_attach"``) included in the error so the audit trail
    attributes the block to the entry point that fired.  It is not part
    of the policy decision.
    """
    if not name or not str(name).strip():
        # Empty filenames are a separate problem (the dashboard upload
        # rejects them upstream).  Not a secret-path concern; let the
        # upstream caller surface its own message.
        return None

    leaf = _normalise_basename(name)
    if not leaf:
        return None

    # 1. exact auth.json / .anthropic_oauth.json / credentials.json / ...
    if leaf in _PROHIBITED_EXACT_BASENAMES:
        return _format_error("credential_file", source)

    # 2. dotenv shapes — both the leading-dot form (``<name>``) and the
    #    bare form after leading-dot stripping (``<name>.<env>``).
    #    ``_normalise_basename`` strips leading dots, so ``.env`` is
    #    matched here as the bare ``env`` and ``.env.production`` as the
    #    bare ``env.production``.  Both the fixed denylist and the
    #    generic ``env.<suffix>`` pattern are checked.
    if leaf in _PROHIBITED_DOTENV_BASENAMES:
        return _format_error("dotenv", source)
    if leaf == "env" or leaf.startswith("env."):
        return _format_error("dotenv", source)
    if leaf == "envrc" or leaf.startswith("envrc."):
        return _format_error("dotenv", source)

    # 3. keychain databases: foo.kdbx / foo.kbdx / foo.kdb
    for suffix in _PROHIBITED_KEYCHAIN_SUFFIXES:
        if leaf.endswith(suffix):
            return _format_error("keychain_database", source)

    # 4. keychain-export globs: keepass-export.xml / keychain-export.csv / ...
    for pattern in _PROHIBITED_KEYCHAIN_GLOB:
        if _glob_match(pattern, leaf):
            return _format_error("keychain_export", source)

    return None


def check_scratch_artifact_path(
    path: Union[str, os.PathLike[str]],
    *,
    source: str = "scratch_artifact",
) -> Optional[str]:
    """Return a stable error kind if *path*'s basename is prohibited.

    Used by the scratch-workspace → attachment promotion path
    (``_persist_scratch_completion_artifacts``).  Resolves symlinks via
    :func:`Path.resolve` when possible so an attacker can't rename a
    malicious file to point at ``auth.json`` via a relative symlink and
    then promote it under a benign leaf name.  Resolution failures
    fall back to basename-only inspection (we'd rather block on a
    basename we *can* classify than silently let an unresolvable path
    through).
    """
    try:
        resolved = Path(path).expanduser().resolve()
        leaf = resolved.name.lower()
    except (OSError, RuntimeError, ValueError):
        # Path.resolve failed — fall back to literal basename.
        leaf = _normalise_basename(os.fspath(path))

    if not leaf:
        return None

    return check_attachment_filename(leaf, source=source)


def raise_if_blocked(
    name: str,
    *,
    source: str = "attach",
) -> None:
    """Raise :class:`SecretPathError` if *name* matches the denylist.

    Convenience wrapper for ingress sites that want a single line of
    guarding code instead of the ``err = check(...); if err: return ...``
    dance.
    """
    err = check_attachment_filename(name, source=source)
    if err is not None:
        kind = _kind_for_error(err)
        raise SecretPathError(kind=kind, reason=err)


def raise_if_blocked_scratch(
    path: Union[str, os.PathLike[str]],
    *,
    source: str = "scratch_artifact",
) -> None:
    """Raise :class:`SecretPathError` for a prohibited scratch artifact path."""
    err = check_scratch_artifact_path(path, source=source)
    if err is not None:
        kind = _kind_for_error(err)
        raise SecretPathError(kind=kind, reason=err)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_KIND_BY_TOKEN = {
    "credential_file": "credential_file",
    "dotenv": "dotenv",
    "keychain_database": "keychain_export",
    "keychain_export": "keychain_export",
}


_REASON_BY_TOKEN: dict[str, str] = {
    "credential_file": (
        "this shape is a known credential store and must not be uploaded "
        "as a task artifact"
    ),
    "dotenv": (
        "project environment files routinely contain API keys / database "
        "passwords and must not be uploaded as a task artifact"
    ),
    "keychain_database": (
        "this is a keychain / password-manager database and must not be "
        "uploaded as a task artifact"
    ),
    "keychain_export": (
        "this is a keychain / password-manager export and must not be "
        "uploaded as a task artifact"
    ),
}


def _format_error(token: str, source: str) -> str:
    """Build the user-visible error string.

    Deliberately includes only the policy ``kind`` and a stable,
    filename-free reason.  ``source`` is shown so the audit trail can
    tell which ingress fired the block.  The reason text is a fixed
    lookup (no string interpolation of the offending input) so the
    policy never echoes the basename or contents.
    """
    kind = _KIND_BY_TOKEN.get(token, "credential")
    reason = _REASON_BY_TOKEN.get(token, "this shape is a credential-like artifact")
    return f"{source}: refused to admit '{kind}' attachment — {reason}."


def _kind_for_error(err: str) -> str:
    """Extract the ``kind`` token out of a formatted error string.

    The error string starts with ``"<source>: refused to admit '<kind>'"``
    so we can recover the kind without threading it through the public
    API.  Falls back to ``"credential"`` if the format ever changes.
    """
    try:
        # Find the single-quoted token between ``'<kind>'``.
        start = err.index("'") + 1
        end = err.index("'", start)
        return err[start:end]
    except ValueError:
        return "credential"


__all__ = [
    "SecretPathError",
    "check_attachment_filename",
    "check_scratch_artifact_path",
    "raise_if_blocked",
    "raise_if_blocked_scratch",
]


# ---------------------------------------------------------------------------
# Self-check (intentionally minimal — only run by tests via import).
# ---------------------------------------------------------------------------


def _self_check() -> list[Optional[str]]:
    """Return a list of policy-error strings for the canonical cases.

    Used by the bundled tests; not invoked at import time.
    """
    cases = (
        "auth.json",
        "AUTH.JSON",
        "./secrets/auth.json",
        "C:\\Users\\me\\auth.json",
        ".env",
        ".env.local",
        ".env.production",
        ".envrc",
        "credentials.json",
        "1password-export.json",
        "bitwarden-export.json",
        "lastpass-export.csv",
        "vault.kdbx",
        "team.kdb",
        "keepass-export.xml",
        "Keychain-Export.csv",
        ".anthropic_oauth.json",
        "bws_cache.enc.json",
    )
    return [check_attachment_filename(c, source="self_check") for c in cases]