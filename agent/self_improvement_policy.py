"""Canonical self-improvement policy for post-turn writes.

Closes the gap (issue: end-of-session self-improvement writes) where the
background review fork could generate or apply writes (skill, memory,
cron-suggestions) after the user's session ended.

This module is **pure**: it does not read prompt text, does not touch the
filesystem, does not import heavy modules, and does not have any side
effects at import time. The two L1/L2 call sites (turn_finalizer and the
per-tool write handlers) construct the inputs from session signals they
already have and read the decision back.

Two boolean knobs are honoured:

* ``HERMES_DISABLE_SELF_IMPROVEMENT`` — when activated, every self-
  improvement write is denied regardless of session or origin. This is
  the global kill switch.
* ``HERMES_READ_ONLY_SESSION`` — when activated, self-improvement writes
  from the background_review origin are denied. Foreground (user-driven)
  writes are not in scope; this policy only governs autonomous writes
  that flow through the background review thread.

Both knobs are interpreted by ``_normalize_bool``. A value that cannot be
normalised is treated fail-closed: environment disabled is assumed; a
read-only session is assumed; a DENY_UNKNOWN_OPERATION is returned with
an auditable reason. The decision never raises an exception, so the
call-site can log it inline and continue the normal session-close flow.

Decisions:

* ``ALLOW`` — write is allowed to proceed.
* ``DENY_ENV_DISABLED`` — ``HERMES_DISABLE_SELF_IMPROVEMENT`` activated.
* ``DENY_READ_ONLY_SESSION`` — ``HERMES_READ_ONLY_SESSION`` activated and
  the origin is background review.
* ``DENY_UNKNOWN_OPERATION`` — neither knob is set, but the requested
  operation kind is not in the catalogue; or a knob value could not be
  normalised. Fail-closed for self-improvement writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Decision result sentinels. Strings (not enums) so log records print
# readably and downstream test assertions don't need a class import.
ALLOW = "ALLOW"
DENY_ENV_DISABLED = "DENY_ENV_DISABLED"
DENY_READ_ONLY_SESSION = "DENY_READ_ONLY_SESSION"
DENY_UNKNOWN_OPERATION = "DENY_UNKNOWN_OPERATION"

# Operation kinds explicitly recognised by this policy. Anything not in
# this frozenset is treated as unknown and produces
# ``DENY_UNKNOWN_OPERATION``. Keep the set narrow and audit-friendly.
#
# PHASE 2 (P0 containment): the foreground skill-management action
# kinds (skill_create/skill_edit/skill_patch/skill_delete/skill_write_file/
# skill_remove_file) are now first-class catalogue entries so the
# foreground self-improvement gate returns ``DENY_ENV_DISABLED`` when
# ``HERMES_DISABLE_SELF_IMPROVEMENT`` is activated, instead of the
# unrelated ``DENY_UNKNOWN_OPERATION`` sentinel. This closes the
# policy-control gap documented in
# ``06_POLICY_CONTROL_GAP.md`` of the design package.
SELF_IMPROVEMENT_OPERATIONS = frozenset({
    "skill_write",
    "skill_create",
    "skill_edit",
    "skill_patch",
    "skill_delete",
    "skill_write_file",
    "skill_remove_file",
    "memory_write",
    "memory_delete",
    "suggestions_write",
    "background_review_spawn",
})

# Background-review origin sentinel. Mirrors
# ``tools.skill_provenance.BACKGROUND_REVIEW`` so the two layers agree on
# the string used to identify the autonomous fork.
BACKGROUND_REVIEW_ORIGIN = "background_review"


def _normalize_bool(value: Any) -> Optional[bool]:
    """Best-effort boolean coercion.

    Returns:
      * ``True`` — value is one of ``"1"``, ``"true"``, ``"yes"``, ``"on"``
        (case-insensitive, surrounding whitespace ignored).
      * ``False`` — value is empty, ``"0"``, ``"false"``, ``"no"``, ``"off"``,
        or ``False``/``None``.
      * ``None`` — value is a non-empty string that does not match any
        known token. Callers must treat None as fail-closed for
        self-improvement decisions.

    Never raises. Lists, dicts and other types return False.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # Only 0 / 1 are normalised; other integers are not booleans.
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return False
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
        return None
    return False


def normalize_env_disabled(value: Any) -> bool:
    """Return True iff ``HERMES_DISABLE_SELF_IMPROVEMENT`` is activated.

    PHASE 1 (P0 containment): missing/empty/unset values are treated as
    ACTIVATED (DENY) -- fail-closed. Unknown values are also treated as
    activated. Only the explicit positive opt-in ("0"/"false"/"no"/"off")
    returns False.
    """
    # PHASE 1: explicit missing/empty = DENY (was ALLOW pre-Phase-1).
    # Local guard so we do NOT touch _normalize_bool (shared with
    # normalize_read_only_session, whose semantics must remain unchanged).
    if isinstance(value, str) and not value.strip():
        return True
    coerced = _normalize_bool(value)
    if coerced is None:
        return True
    return coerced


def normalize_read_only_session(value: Any) -> bool:
    """Return True iff ``HERMES_READ_ONLY_SESSION`` is activated.

    Unknown values are treated as activated (fail-closed).
    """
    coerced = _normalize_bool(value)
    if coerced is None:
        return True
    return coerced


def _map_foreground_skill_operation(operation_kind: str) -> str:
    """Collapse a foreground skill-management ``operation_kind`` to its
    catalogue entry.

    ``tools/skill_manager_tool.py`` emits the more specific labels
    ``skill_create``, ``skill_edit``, ``skill_patch``, ``skill_delete``,
    ``skill_write_file`` and ``skill_remove_file``. They are all members
    of ``SELF_IMPROVEMENT_OPERATIONS`` after Phase 2, but historically
    only ``skill_write`` was catalogued. Callers that need a stable
    identity for logging or audit can keep using the more specific
    label; this helper exists so a future policy pass can collapse
    them without breaking imports.
    """
    if operation_kind in SELF_IMPROVEMENT_OPERATIONS:
        return operation_kind
    return operation_kind


@dataclass(frozen=True)
class Decision:
    """Result of a policy evaluation. Frozen so callers cannot mutate."""
    result: str
    reason: str

    @property
    def allow(self) -> bool:
        return self.result == ALLOW


def evaluate(
    *,
    environment_disabled: Any,
    session_read_only: Any,
    operation_kind: str,
    origin: Optional[str] = None,
    target_path: Optional[str] = None,
    explicit_opt_in: Any = None,
) -> Decision:
    """Evaluate the self-improvement policy.

    Parameters
    ----------
    environment_disabled:
        Raw value of ``HERMES_DISABLE_SELF_IMPROVEMENT`` (str/int/bool/None).
    session_read_only:
        Raw value of ``HERMES_READ_ONLY_SESSION`` (str/int/bool/None).
    operation_kind:
        One of ``SELF_IMPROVEMENT_OPERATIONS``. Anything else returns
        ``DENY_UNKNOWN_OPERATION``. PHASE 2 now accepts the foreground
        ``skill_*`` labels as first-class entries; this is the seam that
        ``tools/skill_manager_tool.py`` consults to close the gap.
    origin:
        Optional string identifying the call site. Used only to decide
        whether a read-only session applies (it must — the foreground
        user is never in scope here; this is the L1/L2 layer for
        autonomous post-turn writes).
    target_path:
        Optional path the write targets. Audited in the reason string
        when the decision is a DENY, so a log scanner sees what was
        refused without leaking secrets.
    explicit_opt_in:
        Reserved for callers that already enforce an explicit opt-in
        gate (e.g. curator running under a verified host approval).
        When ``False`` and the knobs would otherwise allow, the
        decision is still ``DENY_UNKNOWN_OPERATION`` — the conservative
        default. ``True`` is ignored here because this module cannot
        verify the opt-in's provenance; see L2 callers.

    Returns
    -------
    Decision
        Either ``ALLOW`` or one of three DENY variants. Never raises.
    """
    operation_kind = _map_foreground_skill_operation(operation_kind)
    env_disabled = normalize_env_disabled(environment_disabled)
    read_only = normalize_read_only_session(session_read_only)

    # Precedence (strongest deny first) — directive §11 hierarchy:
    #   1. read-only or deny-all SessionWritePolicy (strongest DENY)
    #   2. explicit self-improvement disablement (env_disabled)
    #   3. invalid authorization (operation_kind not in catalogue)
    #   4. missing / empty authorization (explicit_opt_in is False)
    #   5. explicit valid enablement (ALLOW)
    #
    # Read-only must be checked BEFORE env_disabled so a missing/empty
    # HERMES_DISABLE_SELF_IMPROVEMENT does not mask the strongest deny
    # when HERMES_READ_ONLY_SESSION is activated for a background_review
    # origin. Both are fail-closed; the precedence only changes which
    # reason code is surfaced.
    if read_only and origin == BACKGROUND_REVIEW_ORIGIN:
        return Decision(
            result=DENY_READ_ONLY_SESSION,
            reason=(
                "HERMES_READ_ONLY_SESSION is activated; refusing "
                f"background_review operation_kind={operation_kind!r}"
                + (f" target={target_path!r}" if target_path else "")
            ),
        )

    if env_disabled:
        return Decision(
            result=DENY_ENV_DISABLED,
            reason=(
                "HERMES_DISABLE_SELF_IMPROVEMENT is activated; "
                f"refusing operation_kind={operation_kind!r} "
                f"origin={origin!r}"
                + (f" target={target_path!r}" if target_path else "")
            ),
        )

    if operation_kind not in SELF_IMPROVEMENT_OPERATIONS:
        return Decision(
            result=DENY_UNKNOWN_OPERATION,
            reason=(
                f"operation_kind={operation_kind!r} is not in the "
                "self-improvement catalogue; fail-closed"
            ),
        )

    if explicit_opt_in is False:
        return Decision(
            result=DENY_UNKNOWN_OPERATION,
            reason=(
                "explicit_opt_in=False; refusing autonomous write "
                f"operation_kind={operation_kind!r}"
            ),
        )

    return Decision(
        result=ALLOW,
        reason=(
            f"self-improvement policy allows operation_kind={operation_kind!r} "
            f"origin={origin!r}"
        ),
    )


__all__ = [
    "ALLOW",
    "DENY_ENV_DISABLED",
    "DENY_READ_ONLY_SESSION",
    "DENY_UNKNOWN_OPERATION",
    "SELF_IMPROVEMENT_OPERATIONS",
    "BACKGROUND_REVIEW_ORIGIN",
    "Decision",
    "evaluate",
    "normalize_env_disabled",
    "normalize_read_only_session",
]
