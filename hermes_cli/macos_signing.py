"""Managed interpreter identity, available before PM or application imports."""
from __future__ import annotations

import logging
from pathlib import Path
import platform
import shutil
import subprocess

logger = logging.getLogger(__name__)
_IDENTIFIER = "com.nousresearch.hermes.managed-python"


def sign_managed_python(python: Path) -> bool:
    """Pin the designated requirement across downloaded Python generations.

    PBS's ad-hoc cdhash changes on upgrades. A stable identifier preserves
    TCC identity without a Developer ID certificate. Signing remains best
    effort so unavailable codesign cannot prevent a bootable interpreter.
    """
    if platform.system() != "Darwin":
        return False
    codesign = shutil.which("codesign")
    if not codesign:
        logger.info("macOS codesign is unavailable; using the downloaded Python signature")
        return False
    try:
        signed = subprocess.run(
            [codesign, "--force", "--deep", "--sign", "-", "--timestamp=none",
             "--identifier", _IDENTIFIER, "--requirements",
             f'=designated => identifier "{_IDENTIFIER}"', str(python)],
            check=False, capture_output=True, text=True,
        )
        if signed.returncode != 0:
            logger.warning("could not stably sign managed Python %s: %s", python,
                           (signed.stderr or signed.stdout or "codesign failed").strip())
            return False
        verified = subprocess.run(
            [codesign, "--verify", "--deep", "--strict", str(python)],
            check=False, capture_output=True, text=True,
        )
        if verified.returncode != 0:
            logger.warning("macOS signature verification failed for managed Python %s: %s", python,
                           (verified.stderr or verified.stdout or "verification failed").strip())
            return False
        return True
    except Exception as exc:
        logger.warning("could not sign managed Python %s: %s", python, exc)
        return False
