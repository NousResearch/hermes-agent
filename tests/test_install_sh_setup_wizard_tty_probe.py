"""Regression for #16746: setup-wizard tty gate must actually open /dev/tty.

In a Docker build, ``/dev/tty`` exists as a device node (so a bare ``-e``
existence test returns true) but opening it fails with ``ENXIO: No such
device or address``. Under the old gate the wizard proceeded past the "no
terminal available" skip and then crashed on the ``< /dev/tty`` redirect a
few lines later, aborting the entire image build. The fix replaces the
existence check with an open-based probe so the skip kicks in correctly.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_run_setup_wizard() -> str:
    """Return the body of ``run_setup_wizard()`` as a single string.

    Anchored to ``run_setup_wizard()`` and a top-of-line ``}`` so the helper
    keeps working if neighbouring functions are renamed.
    """
    text = INSTALL_SH.read_text()
    match = re.search(
        r"^run_setup_wizard\(\)\s*\{\s*\n(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "run_setup_wizard() not found in scripts/install.sh"
    return match["body"]


def test_run_setup_wizard_does_not_use_existence_only_tty_check() -> None:
    """The bare ``-e`` test is the bug — no spelling of it should remain."""
    body = _extract_run_setup_wizard()
    # Cover ``[ -e /dev/tty ]``, ``[ -e "/dev/tty" ]``, ``test -e /dev/tty``
    # and friends, with arbitrary surrounding whitespace.
    pattern = re.compile(
        r"""(
            \[\s*-e\s+["']?/dev/tty["']?\s*\]
            |
            \btest\s+-e\s+["']?/dev/tty["']?
        )""",
        re.VERBOSE,
    )
    match = pattern.search(body)
    assert match is None, (
        "run_setup_wizard contains an existence-only check on /dev/tty "
        f"({match.group(0)!r}). Bare `-e` tests pass in Docker builds "
        "where the device node is in the mount namespace but cannot be "
        "opened (ENXIO). Use an open-based probe (e.g. "
        "`(: </dev/tty) 2>/dev/null` or `exec 3</dev/tty`) so the skip "
        "kicks in before the wizard tries to read from /dev/tty. "
        "See #16746."
    )


def test_run_setup_wizard_gates_on_open_based_tty_probe() -> None:
    """The gate must actually attempt to open ``/dev/tty``.

    Any ``if !`` (or ``if``) whose condition opens ``/dev/tty`` for input
    counts: ``(: </dev/tty)``, ``exec 3</dev/tty``, ``{ exec 3</dev/tty; }``,
    etc. Asserting the higher-level invariant rather than a specific spelling
    so equivalent refactors stay green.
    """
    body = _extract_run_setup_wizard()
    gate = re.compile(r"^\s*if\s+!?\s+[^\n]*<\s*/dev/tty[^\n]*;\s*then", re.MULTILINE)
    assert gate.search(body), (
        "run_setup_wizard must gate on an open-based probe of /dev/tty "
        "(an `if`/`if !` whose test redirects stdin from /dev/tty), not a "
        "mere existence check. See #16746."
    )
