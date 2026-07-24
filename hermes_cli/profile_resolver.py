"""Canonical profile-name resolver used by the kanban CLI (and any other
surface that needs to validate an ``--assignee`` value).

This module is the single source of truth for "is this string a registered
profile?". Other surfaces (the dispatcher daemon, the dashboard, programmatic
task creators) should call :func:`resolve_assignee` rather than reimplementing
their own matching against ``hermes_cli.profiles``.

Why a dedicated module
----------------------

Before this module existed, profile validation lived inline in
``hermes_cli.kanban._cmd_create`` (the CLI path) but **also** implicitly in
the dispatcher's claim-time silent self-heal. Three kanban cards were filed
with assignee names that aren't real profiles; the dispatcher silently
routed them to whatever profile happened to match, hiding the typo from the
operator. Catching the typo at the CLI entry point closes the root cause.

A dedicated resolver lets the dispatcher reuse the same canonicalization
rules the CLI uses, so a future daemon-side fix can be expressed as
"replace my bespoke matcher with ``profile_resolver.resolve_assignee``"
rather than a second copy of the matching logic.

Scope
-----

* Resolve a free-form name to its canonical on-disk form (lowercase,
  whitespace-trimmed, ``Default`` -> ``default``).
* Return the canonical name if a profile directory exists for it.
* Return ``None`` for ``None``/empty input and for unknown names; the caller
  decides which case to treat as an error.
* Accept the documented bypass value ``__any__`` without resolving — that's
  a magic string the dispatcher recognises, not a profile name.
"""

from __future__ import annotations

from typing import Optional

from hermes_cli.profiles import normalize_profile_name, profile_exists


# Literal --assignee values that must be accepted without resolving against
# the installed profile roster. The "no --assignee flag" case (None / "") is
# handled inline by resolve_assignee itself.
#
# "__any__": documented magic value meaning "I don't care which profile
#            picks this up". Used by automated/orchestrator tooling that
#            doesn't know the host's roster ahead of time. The dispatcher
#            claims tasks with this assignee onto whatever profile is free.
_BYPASS_VALUES: frozenset[str] = frozenset({"__any__"})


def resolve_assignee(name: Optional[str]) -> Optional[str]:
    """Resolve an ``--assignee`` value to its canonical profile name, or ``None``.

    Resolution rules (in order):

    1. ``None`` or empty/whitespace-only → ``None``. The caller should treat
       this as "no assignee was set; persist the card unassigned".
    2. Any value in :data:`_BYPASS_VALUES` (currently just ``"__any__"``)
       → returned unchanged. These are pass-through magic strings, not
       profile names.
    3. Otherwise: normalize the name (``normalize_profile_name``) and check
       whether a profile directory exists for it.

       * Profile exists → return the **canonical** (normalized) name.
         Mixed-case input (``Code-Craftsman``) collapses to the on-disk
         lowercase form (``code-craftsman``) so the card's persisted
         ``assignee`` field matches the directory name on disk.
       * No such profile → return ``None``. The caller should reject the
         create with an error.

    Returns ``None`` for both "no assignee set" and "unknown profile" —
    callers must distinguish these by checking ``name`` themselves
    (see ``hermes_cli.kanban._cmd_create``).
    """
    if name is None:
        return None
    if not isinstance(name, str):
        # Non-string input (e.g. an int sneaking through argparse misuse) is
        # treated the same as None. Refuse silently rather than crashing
        # mid-CLI; the caller will still error if ``name`` was meant to be a
        # real profile.
        return None
    stripped = name.strip()
    if not stripped:
        return None
    if stripped in _BYPASS_VALUES:
        return stripped
    try:
        canon = normalize_profile_name(stripped)
    except ValueError:
        # normalize_profile_name rejects empty input (already handled above)
        # and ``None`` (already handled). Any other ValueError here means the
        # input is unusable as a profile name — treat as unknown.
        return None
    if profile_exists(canon):
        return canon
    return None


__all__ = ["resolve_assignee"]
