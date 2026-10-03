"""Shared leaf helpers for the ``hermes update`` modules (no Hermes imports; no cycle)."""

import logging
import os
from contextlib import contextmanager

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


@contextmanager
def _best_effort(message: str):
    """Run a non-critical update step; swallow ``Exception`` and log it at debug.

    The updater must never die on bookkeeping (receipt, notices, cache seeds):
    ``message`` is the ``%s``-style debug line the inline ``try/except`` used.
    """
    try:
        yield
    except Exception as exc:
        logger.debug(message, exc)


def foreign_install_shadow_warning(project_root=None) -> str | None:
    """Warn when this process is a NON-live checkout that would shadow the live install.

    ``hermes update`` operates on ``PROJECT_ROOT`` — whichever checkout the invoking interpreter
    imports ``hermes_cli`` from. When a dev clone's venv is active, bare ``hermes`` resolves to the
    DEV clone: the whole update runs there and prints a success line carrying a sha, which reads as
    "already deployed" while the user's live install silently stays on the old commit. Observed live
    2026-09-26 — the live install sat one commit behind after two updater runs reported success.

    Deliberately narrow, so a legitimately non-standard install is never nagged:
      * only fires when BOTH this checkout and the default live layout exist and look like real
        Hermes checkouts (checkout markers shipped with the repo, probed by file presence not by path spelling);
      * silenced under pytest (a suite running in the dev clone is normal; mirrors
        ``_early_recovery._pytest_owns_live_checkout``);
      * returns ``None`` on any failure — this must never block an update.

    The corp-style layout (launcher pointing straight at ``~/repos/hermes-agent``) is unaffected:
    there the live layout at ``~/.hermes/hermes-agent`` does not exist.
    """
    from pathlib import Path

    try:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return None
        current_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
        current_root = Path(os.path.realpath(current_root))
        live_root = Path(os.path.realpath(
            Path(os.environ.get("HERMES_REAL_HOME") or str(Path.home())) / ".hermes" / "hermes-agent"
        ))
        if current_root == live_root:
            return None
        if not live_root.is_dir():
            return None
        # Content probe: only a real checkout carries these files.
        markers = (Path("pyproject.toml"), Path("hermes_cli") / "__init__.py")
        if not all((live_root / m).exists() for m in markers) or not all((current_root / m).exists() for m in markers):
            return None
        return (
            f"⚠ This is NOT the live install — the update would run against:\n"
            f"    {current_root}\n"
            f"  Your live install is:\n"
            f"    {live_root}\n"
            f"  (a dev-clone venv on PATH is shadowing the live launcher; direnv + the repo's\n"
            f"   .envrc `use flake` sets VIRTUAL_ENV, so bare `hermes` resolves to the dev clone.)\n"
            f"  Update the live install explicitly instead:\n"
            f"    env -u VIRTUAL_ENV {live_root}/venv/bin/hermes update\n"
        )
    except Exception:
        return None
