"""Single-writer guard: the legacy append-style food helper must be gone.

The pre-migration deployment carried an untracked host-state helper
(``~/.hermes/scripts/food_log_commit.py`` + ``food_nudge.py``) that
appended food entries directly. The migration contract requires exactly
one reviewed Health writer, with the old helper DISABLED before cutover
and never restorable by rollback.

This guard makes that a *source-enforced design property* rather than an
operational promise: the Sol food plugin refuses to activate while any
legacy helper artifact is present under the active Hermes home. Because
the guard lives in reviewed source, a runtime rollback of the gateway
cannot re-enable the helper path — rolling back simply removes the new
plugin too, and re-deploying it re-asserts the guard. The plugin has no
code path that shells out to, imports, or falls back to the helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

__all__ = ["LegacyHelperPresent", "assert_legacy_helper_disabled"]

REASON_LEGACY_PRESENT = "sol_food_legacy_helper_present"

#: Relative artifact paths (under the Hermes home) whose presence proves
#: the legacy writer could still be invoked. Exact names, no globbing.
LEGACY_HELPER_ARTIFACTS: Tuple[str, ...] = (
    "scripts/food_log_commit.py",
    "scripts/food_nudge.py",
)


class LegacyHelperPresent(Exception):
    def __init__(self) -> None:
        super().__init__(REASON_LEGACY_PRESENT)
        self.reason_code = REASON_LEGACY_PRESENT


def assert_legacy_helper_disabled(hermes_home: Path) -> None:
    """Raise :class:`LegacyHelperPresent` if any legacy artifact exists.

    Fail-closed: an unreadable home directory is treated as unknown state
    and also refuses activation.
    """
    home = Path(hermes_home)
    try:
        for relative in LEGACY_HELPER_ARTIFACTS:
            if (home / relative).exists():
                raise LegacyHelperPresent()
    except OSError:
        raise LegacyHelperPresent() from None
