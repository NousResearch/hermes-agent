#!/usr/bin/env python3
"""Diagnose why Hermes Dashboard uses a fixed vs random session token.

Mirrors ``hermes_cli.main`` bootstrap (profile override + ``load_hermes_dotenv``)
before inspecting ``HERMES_DASHBOARD_SESSION_TOKEN``. Secrets are masked in output.

Usage::

    cd /path/to/hermes-agent
    source .venv/bin/activate
    python scripts/diagnose_dashboard_session_token.py
    python scripts/diagnose_dashboard_session_token.py -p myprofile
    python scripts/diagnose_dashboard_session_token.py --import-web-server

Exit codes:
    0 — Diagnostic finished (and optional web_server token matches fixed secret).
    1 — With ``--import-web-server``: fixed branch expected but ``_SESSION_TOKEN`` mismatch.
    2 — Usage / bootstrap error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

KEY = "HERMES_DASHBOARD_SESSION_TOKEN"
ALT = "HERMES_DASHBOARD_TOKEN"


def _strip_own_flags(argv: list[str]) -> tuple[list[str], bool]:
    """Remove ``--import-web-server`` from args forwarded to profile parsing."""
    import_ws = False
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--import-web-server":
            import_ws = True
            i += 1
            continue
        rest.append(a)
        i += 1
    return rest, import_ws


def _mask(secret: str) -> str:
    if not secret:
        return "(empty)"
    if len(secret) <= 8:
        return f"*** (len={len(secret)})"
    return f"{secret[:4]}…{secret[-4:]} (len={len(secret)})"


def _ascii_only(secret: str) -> bool:
    if not secret:
        return True
    try:
        secret.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _reject_reason(value: str | None, *, min_length: int = 16) -> str | None:
    """Mirror ``has_usable_secret`` failure reasons (dashboard uses min_length=16)."""
    from hermes_cli.auth import _PLACEHOLDER_SECRET_VALUES  # noqa: SLF001

    if value is None:
        return "unset"
    if not isinstance(value, str):
        return "not_a_string"
    cleaned = value.strip()
    if not cleaned:
        return "empty_after_strip"
    if len(cleaned) < min_length:
        return f"too_short(len={len(cleaned)},need={min_length})"
    if cleaned.lower() in _PLACEHOLDER_SECRET_VALUES:
        return "placeholder"
    return None


def main() -> int:
    argv_full = sys.argv[:]
    if len(argv_full) > 1 and argv_full[1] in ("-h", "--help"):
        print(__doc__)
        return 0

    passthrough, import_web_server = _strip_own_flags(argv_full[1:])
    sys.argv = ["hermes_diagnose", *passthrough]
    sys.path.insert(0, str(_REPO_ROOT))

    try:
        import hermes_cli.main  # noqa: F401 — profile + dotenv (same as ``hermes`` CLI)
    except Exception as exc:
        print(f"Bootstrap failed while importing hermes_cli.main: {exc}", file=sys.stderr)
        return 2

    from hermes_cli.auth import has_usable_secret
    from hermes_cli.config import get_env_path, get_hermes_home

    raw = os.environ.get(KEY)
    stripped = (raw or "").strip()
    usable = has_usable_secret(stripped, min_length=16)
    if usable:
        reason: str | None = None
    elif KEY not in os.environ:
        reason = "unset"
    else:
        reason = _reject_reason(raw if isinstance(raw, str) else "", min_length=16)

    home = get_hermes_home()
    env_path = get_env_path()
    alt_set = bool((os.environ.get(ALT) or "").strip())

    print("=== Hermes Dashboard session token diagnostic ===")
    print(f"HERMES_HOME:           {home}")
    print(f".env path:             {env_path} (exists={env_path.exists()})")
    print(f"os.environ[{KEY!r}]:    {'present' if KEY in os.environ else 'absent'}")
    print(f"  stripped_len:         {len(stripped)}")
    print(f"  ascii_only:           {_ascii_only(stripped)}")
    print(f"  masked_preview:       {_mask(stripped)}")
    print(f"has_usable_secret(..., min_length=16): {usable}")
    print(f"reject_reason:          {reason or '(none — OK for fixed token)'}")

    branch = "fixed" if usable else "random"
    print(
        f"branch:                {branch}\n"
        f"                        # fixed  → web_server uses your env value\n"
        f"                        # random → secrets.token_urlsafe(32) at import time",
    )

    if alt_set and not usable:
        print(
            f"\nNote: {ALT!r} is set (Workspace client name). "
            f"Hermes Dashboard reads {KEY!r} only — "
            "set that key in $HERMES_HOME/.env on the agent host.",
        )

    exit_code = 0
    if import_web_server:
        print("\n=== hermes_cli.web_server snapshot (--import-web-server) ===")
        try:
            from hermes_cli.web_server import _SESSION_TOKEN  # noqa: PLC0415
        except Exception as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 2

        print(f"_SESSION_TOKEN masked: {_mask(_SESSION_TOKEN)}")
        if usable:
            if _SESSION_TOKEN == stripped:
                print("_SESSION_TOKEN matches stripped fixed secret: True")
            else:
                print("_SESSION_TOKEN matches stripped fixed secret: False", file=sys.stderr)
                exit_code = 1
        else:
            print(
                "(fixed comparison skipped — branch is random; value differs each process)",
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
