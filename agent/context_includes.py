"""@-include expansion for context files (CAAMP-style transitive includes).

This module is the single source of truth for how Hermes resolves
``@<path>`` directives inside context files (``.hermes.md``, ``AGENTS.md``,
``CLAUDE.md``, ``.cursorrules``, ``SOUL.md``, ``.cursor/rules/*.mdc``).

Design goals
============

* **Provider-neutral** — works the same way Claude Code, Cursor, and other
  agent harnesses treat ``@`` references, so a project's instruction files
  behave consistently across runtimes.
* **Bounded** — every expansion has a hard depth cap and a per-include
  size cap so a runaway chain or a single huge file cannot blow out the
  system prompt.
* **Cycle-safe** — file inclusion graphs may form cycles
  (A → B → A) and we must terminate without infinite recursion.
* **Injection-safe** — every expanded chunk is run through the same
  ``_scan_context_content`` pipeline as a top-level context file so
  malicious payloads in an included file are blocked before they reach
  the model.
* **Inert in code blocks** — ``@<path>`` written inside a fenced code
  block (```` ``` ```` or ``~~~``) is left as literal text so docs that
  *describe* the syntax don't accidentally trigger expansion.
* **Pure / testable** — the module exposes a small surface
  (:func:`expand_includes`) that callers can unit-test in isolation.

Public API
==========

* :func:`expand_includes` — recursive expander.  Single entry point
  consumers should call.
* :data:`CONTEXT_INCLUDE_MAX_DEPTH` — depth cap (default ``5``).
* :data:`INCLUDE_PATTERN` — the regex used to detect include directives.

Callers in :mod:`agent.prompt_builder` inject a ``scanner`` and a
``truncator`` so this module stays free of upstream-specific behavior:
the scanner is the prompt-injection guard and the truncator is the
head/tail size limiter.  Both default to identity functions when not
supplied — useful for tests and for callers that don't need either pass.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Maximum recursion depth for nested @-includes.  Includes beyond this depth
# are replaced with a "<!-- @max-depth -->" marker so the prompt can never
# blow up from a runaway chain or accidental recursion.
CONTEXT_INCLUDE_MAX_DEPTH = 5

# Pattern: a line that contains ONLY an @<path> reference (with optional
# leading/trailing whitespace).  This intentionally does NOT match inline
# @mentions in prose (e.g. "see @bob") to avoid false positives with email
# addresses, social handles, or doc references.
#
# Allowed path characters cover absolute paths, relative paths, ~, env
# vars (${VAR}), and common punctuation in filenames.  We deliberately
# exclude whitespace and shell metacharacters.
INCLUDE_PATTERN = re.compile(
    r"^[ \t]*@([\w./~$\-+:{}]+)[ \t]*$",
    re.MULTILINE,
)

# Sentinel format used to swap out fenced code blocks during expansion so
# their contents are not scanned for @-includes.  The NUL byte makes
# accidental collisions with real content impossible.
_FENCE_SENTINEL = "\x00CONTEXT_INCLUDE_FENCE{}\x00"
_FENCE_PATTERN = re.compile(r"(?ms)^([ \t]*)(```|~~~)[^\n]*\n.*?^\1\2[ \t]*$")


# ---------------------------------------------------------------------------
# Type aliases for the injectable hooks
# ---------------------------------------------------------------------------

# scanner(content, label) -> sanitized_content
ContentScanner = Callable[[str, str], str]
# truncator(content, label) -> possibly-truncated_content
ContentTruncator = Callable[[str, str], str]


def _identity_scanner(content: str, _label: str) -> str:
    return content


def _identity_truncator(content: str, _label: str) -> str:
    return content


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_include_path(raw: str, base_dir: Path) -> Path:
    """Resolve an @-include path: env vars, ~, then relative to *base_dir*.

    Absolute paths (after ~ / env-var expansion) are kept absolute.
    Relative paths resolve against the *including* file's directory so
    nested includes behave intuitively when files move.
    """
    expanded = os.path.expandvars(os.path.expanduser(raw))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError):
        return candidate


# ---------------------------------------------------------------------------
# Code-fence masking (so @ inside ``` blocks is inert)
# ---------------------------------------------------------------------------


def _mask_code_fences(content: str) -> tuple[str, list[tuple[int, str]]]:
    """Replace fenced code blocks with sentinel placeholders.

    Returns ``(masked_content, fences)`` where ``fences`` lets callers
    restore the original blocks via :func:`_unmask_code_fences`.
    """
    fences: list[tuple[int, str]] = []

    def _replace(match: "re.Match[str]") -> str:
        idx = len(fences)
        fences.append((idx, match.group(0)))
        return _FENCE_SENTINEL.format(idx)

    return _FENCE_PATTERN.sub(_replace, content), fences


def _unmask_code_fences(content: str, fences: list[tuple[int, str]]) -> str:
    """Inverse of :func:`_mask_code_fences`."""
    for idx, original in fences:
        content = content.replace(_FENCE_SENTINEL.format(idx), original)
    return content


# ---------------------------------------------------------------------------
# Recursive expander
# ---------------------------------------------------------------------------


def expand_includes(
    content: str,
    base_dir: Path,
    *,
    depth: int = 0,
    visited: Optional[set[str]] = None,
    scanner: ContentScanner = _identity_scanner,
    truncator: ContentTruncator = _identity_truncator,
    max_depth: int = CONTEXT_INCLUDE_MAX_DEPTH,
) -> str:
    """Recursively expand ``@<path>`` directives in *content*.

    Parameters
    ----------
    content
        The text to scan.  Typically the body of a context file
        (``AGENTS.md``, ``.hermes.md``, etc.).
    base_dir
        Directory used to resolve *relative* include paths.  Should be
        the directory of the file *content* came from.
    depth
        Current recursion depth.  Callers should leave this at ``0``;
        the function increments it on recursive calls.
    visited
        Set of resolved absolute paths already on the include stack.
        Used for cycle detection.  Callers should leave this ``None``;
        the function seeds it on the first call.
    scanner
        Optional content-sanitization hook applied to every expanded
        chunk.  Defaults to a no-op.  Hermes injects its
        ``_scan_context_content`` function here so prompt-injection
        attempts in included files are caught.
    truncator
        Optional size-limiting hook applied to every expanded chunk.
        Defaults to a no-op.  Hermes injects its ``_truncate_content``
        function here so a single huge include cannot dominate the
        prompt.
    max_depth
        Hard cap on recursion depth.  Includes beyond this depth are
        replaced with a ``<!-- @max-depth -->`` marker.

    Returns
    -------
    str
        The expanded content with each ``@<path>`` directive replaced
        by the wrapped contents of the target file.

    Notes
    -----
    Markers emitted into the output:

    * ``<!-- @include-begin: <raw> -->`` … ``<!-- @include-end: <raw> -->``
      surround successfully expanded chunks.
    * ``<!-- @missing: <raw> ... -->`` when the target file does not exist.
    * ``<!-- @cycle: already-included <raw> -->`` when the include would
      reintroduce a file already on the stack.
    * ``<!-- @max-depth: <raw> ... -->`` when the depth cap is hit.
    * ``<!-- @unreadable: <raw> (<error>) -->`` for I/O errors.
    """
    if visited is None:
        visited = set()

    if depth >= max_depth:
        # Strip include directives at the depth boundary so they don't
        # appear as raw "@..." text in the rendered prompt.
        masked, fences = _mask_code_fences(content)
        masked = INCLUDE_PATTERN.sub(
            lambda m: f"<!-- @max-depth: {m.group(1)} (max_depth={max_depth}) -->",
            masked,
        )
        return _unmask_code_fences(masked, fences)

    masked, fences = _mask_code_fences(content)

    def _expand_one(match: "re.Match[str]") -> str:
        raw = match.group(1)
        target = resolve_include_path(raw, base_dir)
        target_str = str(target)

        if not target.exists() or not target.is_file():
            return f"<!-- @missing: {raw} (resolved: {target_str}, file not found) -->"

        if target_str in visited:
            return f"<!-- @cycle: already-included {raw} -->"

        try:
            included_text = target.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not read @-include %s: %s", target, exc)
            return f"<!-- @unreadable: {raw} ({exc}) -->"

        # Mark visited BEFORE recursing so cycles A→B→A terminate.
        new_visited = visited | {target_str}
        expanded = expand_includes(
            included_text,
            target.parent,
            depth=depth + 1,
            visited=new_visited,
            scanner=scanner,
            truncator=truncator,
            max_depth=max_depth,
        )
        scanned = scanner(expanded, raw)
        capped = truncator(scanned, raw)
        return (
            f"<!-- @include-begin: {raw} -->\n"
            f"{capped}\n"
            f"<!-- @include-end: {raw} -->"
        )

    expanded = INCLUDE_PATTERN.sub(_expand_one, masked)
    return _unmask_code_fences(expanded, fences)


__all__ = [
    "CONTEXT_INCLUDE_MAX_DEPTH",
    "INCLUDE_PATTERN",
    "expand_includes",
    "resolve_include_path",
]
