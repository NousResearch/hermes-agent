"""validate_env.py — validate HERMES_* environment variables at startup.

Plan 002-D: Cloud Env-Var Routing.

Called from AIAgent.__init__ (non-strict mode) to surface operator
misconfiguration early.  Also usable as a standalone script:

    python3 scripts/validate_env.py

Exit codes:
  0 — all checked vars are valid (may have warnings)
  1 — at least one unsupported URI scheme detected (strict error)

Design:
  - Warns on missing paths (lazy creation is valid — e.g. runtime/ is created
    on first session, not at startup)
  - Errors on URI schemes that are not yet backed by a storage abstraction
    (e.g. s3://) — these would silently fail at pathlib.Path() time
  - Non-strict mode (default): logs errors but does not raise/exit
  - Strict mode: raises ValueError (programmatic use) or exits non-zero (CLI)
"""

from __future__ import annotations

import logging
import os
import sys
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

# Env vars expected to hold local filesystem paths.
# Session/runtime vars (HERMES_SESSION_ID, HERMES_USER_ID) are set by the
# agent process itself, not by operators — they are not validated here.
_LOCAL_PATH_VARS: list[str] = [
    "HERMES_USERS_ROOT",
    "HERMES_RUNTIME_ROOT",
    "HERMES_MCP_SERVERS_CONFIG",
]

# URI schemes that are NOT yet backed by a storage abstraction.
# Operators setting these would get a silent runtime failure at pathlib.Path()
# level; we surface the error here instead.
_UNSUPPORTED_SCHEMES: frozenset[str] = frozenset({
    "s3", "gs", "az", "gcs",
    "dynamodb", "postgres", "postgresql",
    "neon", "supabase", "redis",
})


def validate_hermes_env(*, strict: bool = False) -> list[str]:
    """Validate all HERMES_* operator env vars.

    Args:
        strict: If True, raise ValueError when any unsupported URI scheme is
                found.  If False (default), log the error and continue.

    Returns:
        List of error message strings (empty if everything is valid).
        Warnings are only emitted via logger; they are not returned.

    Raises:
        ValueError: Only when strict=True and at least one error is detected.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for var in _LOCAL_PATH_VARS:
        val = os.environ.get(var, "").strip()
        if not val:
            continue  # unset = use default; always fine

        # Check for URI scheme (unsupported without backend abstraction)
        parsed = urllib.parse.urlparse(val)
        if parsed.scheme and len(parsed.scheme) > 1:
            # urlparse treats Windows paths (C:\...) as scheme "c" — skip those
            if parsed.scheme.lower() in _UNSUPPORTED_SCHEMES:
                errors.append(
                    f"{var}={val!r}: URI scheme {parsed.scheme!r} is not yet "
                    f"supported.  Hermes path functions use pathlib.Path() which "
                    f"requires a local filesystem path.  Use a FUSE mount or "
                    f"wait for a cloud storage backend abstraction."
                )
                continue
            # Unknown scheme — warn but don't block (could be a valid path on
            # some systems, e.g. a relative path with a colon in it)
            if parsed.scheme not in ("", "file"):
                warnings.append(
                    f"{var}={val!r}: unrecognized URI scheme {parsed.scheme!r} — "
                    f"treating as filesystem path.  If this is a remote URI, "
                    f"it will fail at runtime."
                )

        # Check path exists (warn only — lazy creation is valid)
        p = Path(val)
        if not p.exists():
            warnings.append(
                f"{var}={val!r}: path does not exist yet "
                f"(will be created on first use — this is normal for runtime/)"
            )

    for w in warnings:
        logger.warning("hermes env: %s", w)

    if errors:
        for e in errors:
            logger.error("hermes env: %s", e)
        if strict:
            raise ValueError(
                f"Invalid HERMES_* env vars ({len(errors)} error(s)): "
                + "; ".join(errors)
            )
        else:
            logger.warning(
                "hermes env: %d error(s) above — proceeding anyway (non-strict mode). "
                "Set strict=True or run `python3 scripts/validate_env.py` to hard-fail.",
                len(errors),
            )

    return errors


def _cli_main() -> int:
    """Standalone CLI entry point.

    Prints a report and exits 0 (all clean) or 1 (errors found).
    """
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    print("Validating HERMES_* environment variables...")
    errors = validate_hermes_env(strict=False)
    if errors:
        print(f"\nFailed: {len(errors)} error(s) found.")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("OK: all HERMES_* env vars are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
