"""Regression for #16746: setup-wizard tty gate must actually open /dev/tty.

In a Docker build, ``/dev/tty`` exists as a device node (so ``[ -e /dev/tty ]``
returns true) but opening it fails with ``ENXIO: No such device or address``.
Under the old gate the wizard proceeded past the "no terminal available" skip
and then crashed on the ``< /dev/tty`` redirect a few lines later, aborting
the entire image build. The fix replaces the bare existence check with an
open-based probe so the skip kicks in correctly.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_run_setup_wizard() -> str:
    """Return the body of run_setup_wizard() as a single string."""
    text = INSTALL_SH.read_text()
    start = text.index("run_setup_wizard()")
    # The next top-level function follows immediately; use it as the end marker.
    end = text.index("\nmaybe_start_gateway()", start)
    return text[start:end]


def test_run_setup_wizard_does_not_use_bare_existence_check() -> None:
    body = _extract_run_setup_wizard()
    assert "[ -e /dev/tty ]" not in body, (
        "run_setup_wizard guards on `[ -e /dev/tty ]`, which is true in Docker "
        "builds where the device node exists but cannot be opened (ENXIO). "
        "Use an open-based probe such as `(: </dev/tty) 2>/dev/null` so the "
        "skip kicks in before the wizard tries to read from /dev/tty. See #16746."
    )


def test_run_setup_wizard_uses_open_based_tty_probe() -> None:
    body = _extract_run_setup_wizard()
    assert "(: </dev/tty)" in body, (
        "run_setup_wizard must probe /dev/tty by actually opening it before "
        "running the wizard. See #16746."
    )
