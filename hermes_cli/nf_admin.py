"""North Forge in-session admin trigger.

A Full-tier operator can type the drive's admin passcode as a bare message in a
running North Forge conversation to open the SAME Setup Run reconfiguration
``scripts/nf-setup.ps1`` already provides (tier / pin / edition), without a
re-provision-from-scratch cycle.

Design (see logs/ledger/CHANGELOG.md CHG-2026-09-10-001):

* **Reuse, no second system.** Verification is ``nf_tier.verify_admin_passcode``
  (the same PBKDF2 check ``nf-setup.ps1`` uses at drive-build time). The
  reconfiguration flow is ``scripts/nf-setup.ps1`` itself, run interactively — no
  tier/pin/edition logic is duplicated into the CLI.
* **Full tier only.** On a Basic-tier drive, an unprovisioned drive, or plain
  upstream Hermes, the recognizer returns ``None`` before doing anything. There
  is nothing registered in any command table, completion, or help to find — the
  trigger is the operator's own passcode, stored only as a PBKDF2 hash on the
  drive.
* **Silent.** A wrong or blank attempt, or the phrase typed on Basic, produces
  zero observable difference: the recognizer returns ``None`` and the input
  routes to the model exactly as normal chat.
* **Logged.** Every recognized attempt (a hit, or a plausible miss on a Full
  drive) is appended to ``<nf-root>/north-forge/admin-attempts.log`` via
  ``nf_tier.log_admin_attempt`` — for the owner's review only. The passcode is
  never recorded.

The recognizer only matches a **whitespace-free** passcode (6-128 chars): a bare
token is the shape of a passcode attempt, and it lets normal multi-word chat be
skipped without hashing every message. ``nf-setup.ps1`` still accepts any
>=6-char passcode at build time; only this in-session trigger needs the token
shape.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("hermes_cli.nf_admin")

_MIN_LEN = 6
_MAX_LEN = 128


def maybe_recognize_admin_phrase(text: str, root: Path | str | None = None) -> str | None:
    """Classify a submitted message line.

    Returns:
      ``"open"``  - a Full-tier drive, an admin passcode is set, and *text* is it.
                    The caller should open the Setup Run reconfiguration.
      ``None``    - anything else (not Full tier / unprovisioned / no passcode set
                    / not a bare token / wrong passcode). The caller routes the
                    input normally; there is no observable difference.

    A plausible-but-wrong attempt on a Full drive (right shape, wrong value) is
    logged as ``passcode-mismatch`` before returning ``None``. A hit is logged as
    ``reconfig-opened``.

    *root* is an nf-root override for tests (mirrors every ``nf_tier`` entry
    point); the caller passes nothing, so the live drive's provisioning applies.
    """
    cand = (text or "").strip()
    if not (_MIN_LEN <= len(cand) <= _MAX_LEN):
        return None
    if any(ch.isspace() for ch in cand):
        return None

    try:
        from hermes_cli import nf_tier
    except Exception:
        return None

    try:
        prov = nf_tier.load(root)
    except Exception:
        return None
    if prov.state != nf_tier.STATE_ACTIVE or prov.tier != nf_tier.TIER_FULL:
        return None
    if not nf_tier.admin_passcode_is_set(root):
        return None

    try:
        ok = nf_tier.verify_admin_passcode(cand, root)
    except Exception:
        return None

    if ok:
        nf_tier.log_admin_attempt("reconfig-opened", source="cli", root=root)
        return "open"
    nf_tier.log_admin_attempt("passcode-mismatch", source="cli", root=root)
    return None


def _nf_setup_script() -> Path | None:
    try:
        from hermes_cli.config import get_project_root
        root = Path(get_project_root())
    except Exception:
        root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "nf-setup.ps1"
    return script if script.is_file() else None


def run_nf_reconfig_and_resume() -> None:
    """Run ``scripts/nf-setup.ps1`` interactively with the real terminal, then
    re-exec ``hermes`` so the operator lands back in a session (no
    re-provision-from-scratch). Call ONLY from the main thread after
    prompt_toolkit has torn down (next to the ``_pending_relaunch`` handling in
    ``cli.run``).
    """
    script = _nf_setup_script()
    if script is None:
        print("  north forge: scripts/nf-setup.ps1 not found in this checkout — cannot reconfigure.")
        _log("reconfig-launch-failed: nf-setup.ps1 missing")
        _resume()
        return

    if os.name != "nt":
        # nf-setup.ps1 is a Windows PowerShell wizard; there is no non-Windows
        # equivalent of the interactive flow. Point the operator at it.
        print(f"  north forge: run  {script}  to reconfigure this drive (Windows).")
        _log("reconfig-launch-skipped: non-windows")
        _resume()
        return

    print()
    print("  north forge: opening Setup Run (tier / pin / edition) — the drive is not re-provisioned from scratch.")
    print()
    rc = 1
    try:
        # -Force is required: this trigger only ever fires on an already-provisioned
        # Full-tier drive, and nf-setup.ps1 without it hands nf_tier.write_provisioning
        # overwrite=False, which refuses with "already exists — pass --force". The
        # admin passcode is still prompted for and verified by nf-setup.ps1 itself
        # (it is set on this drive) — -Force only lifts the overwrite guard, it does
        # not bypass authentication. This is nf-setup.ps1's own documented
        # RE-PROVISION path (its confirm prompt already reads "RE-PROVISION …").
        rc = subprocess.call(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Force"],
            cwd=str(script.parent.parent),
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  north forge: could not launch Setup Run: {exc}")
        _log(f"reconfig-launch-failed: {exc!r}")
        _resume()
        return

    _log(f"reconfig-completed exit={rc}")
    print()
    print(f"  north forge: Setup Run finished (exit {rc}) — resuming the session…")
    _resume()


def _log(event: str) -> None:
    try:
        from hermes_cli import nf_tier
        nf_tier.log_admin_attempt(event, source="cli")
    except Exception:
        pass


def _resume() -> None:
    """Re-exec ``hermes`` so the operator is back in an interactive session with
    the fresh provisioning in effect. Mirrors the ``/update`` relaunch, minus the
    update args."""
    try:
        from hermes_cli.relaunch import relaunch
        relaunch([], preserve_inherited=False)
    except Exception:
        # Last resort: a plain re-exec of the same interpreter entry point.
        try:
            os.execv(sys.executable, [sys.executable, "-m", "hermes_cli.main"])
        except Exception:
            print("  north forge: reconfiguration done — restart North Forge to continue.")
