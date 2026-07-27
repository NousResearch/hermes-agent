#!/usr/bin/env python3
"""Boot-time re-seed of a terminally-dead Nous bootstrap session.

Background
----------
A Nous bootstrap session (client_id ``hermes-cli-vps``) can take a terminal
``invalid_grant`` and be quarantined locally — the refresh path clears the dead
tokens from ``auth.json`` and stamps
``providers.nous.last_auth_error.relogin_required = true``. From then on every
inference turn hard-fails with a provider-auth error until the credential is
replaced, even though the gateway and dashboard otherwise look healthy.

``stage2-hook.sh`` seeds ``auth.json`` from ``HERMES_AUTH_JSON_BOOTSTRAP`` only
on a *blank* volume (``[ ! -f auth.json ]``) — that guard is load-bearing: it
stops a container restart from clobbering a healthy, rotated refresh token. So a
plain restart with a fresh seed env can NOT recover a container whose volume
already has an auth.json.

This script is the narrow, safe exception. An orchestrator that manages the
container can supply a freshly-issued bootstrap session via
``HERMES_AUTH_JSON_REBOOTSTRAP`` (plus a restart). On boot we re-seed the Nous
provider entry from that env when the on-disk entry is provably terminal, or
when the orchestrator seed's ``obtained_at`` is newer than the local session.
The latter matters because an orchestrator may revoke the previous session
before restart while its still-present local tokens look healthy. Older or
incomparable seeds remain no-ops, so a retained env cannot roll auth backward.

Design constraints
------------------
- Pure stdlib, no hermes_cli imports: runs early in the boot hook, before the
  app venv/modules are guaranteed importable, as its own subprocess.
- Surgical: replaces ONLY ``providers.nous`` in the existing auth.json, leaving
  every other provider, the version, and any other top-level state untouched.
- Fail-safe: any parse/IO error leaves auth.json exactly as-is and exits 0 (a
  failed re-seed must never take the container further down than it already is).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

# Env var the orchestrator sets to the re-seed payload. Deliberately DISTINCT
# from HERMES_AUTH_JSON_BOOTSTRAP (create-only, blank-volume seed) so the two
# paths can never be confused: BOOTSTRAP seeds a fresh volume; REBOOTSTRAP
# overwrites a terminally-dead Nous entry on an existing volume.
REBOOTSTRAP_ENV = "HERMES_AUTH_JSON_REBOOTSTRAP"
BOOTSTRAP_CLIENT_ID = "hermes-cli-vps"


def _nous_entry_is_terminal(nous_state: Any) -> bool:
    """True iff the on-disk Nous provider entry is in the terminal/quarantined
    state AND holds no usable credential.

    Mirrors the ``terminal`` predicate in ``hermes_cli.auth.get_nous_session_validity``:
    a persisted ``last_auth_error.relogin_required`` with the token material
    already cleared. Keeping this in lockstep is what guarantees we only re-seed
    a session that is genuinely dead.
    """
    if not isinstance(nous_state, dict):
        return False
    last_err = nous_state.get("last_auth_error")
    if not (isinstance(last_err, dict) and last_err.get("relogin_required")):
        return False
    # Only terminal while there is no usable credential left. If a live token is
    # somehow present, treat it as healthy and do NOT clobber it.
    if nous_state.get("access_token") or nous_state.get("refresh_token"):
        return False
    return True


def _extract_nous_from_seed(seed_raw: str) -> Optional[dict]:
    """Pull the ``providers.nous`` block out of a HERMES_AUTH_JSON_REBOOTSTRAP
    payload. The payload is a full auth.json document (same shape as
    HERMES_AUTH_JSON_BOOTSTRAP). Returns None unless it carries the expected VPS
    bootstrap client plus non-empty access and refresh tokens — caller treats
    None as "nothing to do"."""
    try:
        seed = json.loads(seed_raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(seed, dict):
        return None
    providers = seed.get("providers")
    if not isinstance(providers, dict):
        return None
    nous = providers.get("nous")
    if not isinstance(nous, dict) or not nous:
        return None
    if nous.get("client_id") != BOOTSTRAP_CLIENT_ID:
        return None
    if not (
        isinstance(nous.get("access_token"), str)
        and nous["access_token"].strip()
        and isinstance(nous.get("refresh_token"), str)
        and nous["refresh_token"].strip()
    ):
        return None
    return nous


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an OAuth timestamp without guessing when either side is malformed."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return None
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _seed_is_newer(local_nous: Any, seed_nous: dict) -> bool:
    """Whether NAS supplied a bootstrap session newer than the local one.

    NAS mints the replacement before restarting the machine and revokes the
    previous session.  A healthy-looking local entry can therefore already be
    stale.  ``obtained_at`` is the server-issued ordering signal that lets boot
    apply a genuinely newer replacement without allowing an old retained env
    value to roll credentials back on later restarts.
    """
    if not isinstance(local_nous, dict):
        return False
    local_obtained = _parse_timestamp(local_nous.get("obtained_at"))
    seed_obtained = _parse_timestamp(seed_nous.get("obtained_at"))
    return bool(
        local_obtained is not None
        and seed_obtained is not None
        and seed_obtained > local_obtained
    )


def reseed_if_terminal(auth_path: str, seed_raw: str) -> str:
    """Core logic. Returns a short status string for logging/testing:

      - "no_seed"          — seed env empty/absent
      - "bad_seed"         — seed present but unparseable / no nous entry
      - "no_auth_file"     — auth.json absent (blank volume → let the normal
                             HERMES_AUTH_JSON_BOOTSTRAP path handle it)
      - "auth_unreadable"  — auth.json present but unparseable (leave as-is)
      - "not_terminal"     — local entry is healthy and at least as new → no-op
      - "reseeded"         — terminal entry replaced from seed
      - "reseeded_newer"   — healthy-but-stale entry replaced by a newer seed
    """
    if not seed_raw:
        return "no_seed"

    seed_nous = _extract_nous_from_seed(seed_raw)
    if seed_nous is None:
        return "bad_seed"

    if not os.path.exists(auth_path):
        # Blank volume — this is the normal first-boot case, not a re-seed.
        return "no_auth_file"

    try:
        with open(auth_path, "r", encoding="utf-8") as fh:
            store = json.load(fh)
    except (OSError, ValueError):
        # Corrupt/unreadable auth.json: do NOT overwrite blindly. A separate
        # concern; leave it for the operator / other recovery paths.
        return "auth_unreadable"

    if not isinstance(store, dict):
        return "auth_unreadable"

    providers = store.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        store["providers"] = providers

    local_nous = providers.get("nous")
    terminal = _nous_entry_is_terminal(local_nous)
    newer_seed = _seed_is_newer(local_nous, seed_nous)
    if not terminal and not newer_seed:
        # Healthy and at least as new as the seed, or incomparable. Never roll a
        # session back merely because an old rebootstrap env remains configured.
        return "not_terminal"

    # Surgical replacement: swap ONLY providers.nous, preserve everything else.
    providers["nous"] = seed_nous

    tmp_path = f"{auth_path}.rebootstrap.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh)
    os.replace(tmp_path, auth_path)
    try:
        os.chmod(auth_path, 0o600)
    except OSError:
        pass
    return "reseeded" if terminal else "reseeded_newer"


def main() -> int:
    auth_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not auth_path:
        home = os.environ.get("HERMES_HOME", "")
        auth_path = os.path.join(home, "auth.json") if home else "auth.json"
    seed_raw = os.environ.get(REBOOTSTRAP_ENV, "")

    try:
        result = reseed_if_terminal(auth_path, seed_raw)
    except Exception as exc:  # never let a re-seed error fail the boot
        print(f"[rebootstrap] error (ignored): {exc!r}", file=sys.stderr)
        return 0

    if result == "reseeded":
        print("[rebootstrap] Nous bootstrap session was terminal; re-seeded auth.json from "
              f"{REBOOTSTRAP_ENV}")
    elif result == "reseeded_newer":
        print("[rebootstrap] Applied newer orchestrator-issued Nous bootstrap session from "
              f"{REBOOTSTRAP_ENV}")
    else:
        # Quiet by default for the common no-op cases; still emit a breadcrumb.
        print(f"[rebootstrap] no-op ({result})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
