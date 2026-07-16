"""Adoption offer at launch (hop 2).

See docs/updater-world.md §2.13 and
docs/plans/updater-rework/03-phase2-compat-and-adoption.md task 2.4.

Runs on every launch. Detects legacy layout, offers adoption if
appropriate. Crash-proof: any internal error logs and continues to
normal startup. Runs before anything that could trip on a stale venv.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Snooze duration: 7 days (in seconds)
SNOOZE_SECONDS = 7 * 24 * 60 * 60

# The config key: updates.adopt = auto|prompt|never
ADOPT_PROMPT_COPY = """⚕ Hermes can switch this install to managed releases (faster, atomic,
  rollbackable updates — no local building). Your current checkout is
  kept untouched as a fallback.
    hermes adopt         # switch now
    hermes adopt --help  # details
  (configure: updates.adopt = auto|prompt|never)"""


def _snooze_path(hermes_home: Path) -> Path:
    """Path to the adoption snooze stamp."""
    state_dir = hermes_home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "adoption-snooze"


def _is_snoozed(hermes_home: Path) -> bool:
    """Check if the adoption offer is snoozed (within the snooze window)."""
    path = _snooze_path(hermes_home)
    if not path.exists():
        return False
    try:
        last_shown = float(path.read_text().strip())
        return (time.time() - last_shown) < SNOOZE_SECONDS
    except (ValueError, OSError):
        return False


def _mark_shown(hermes_home: Path) -> None:
    """Record that the offer was shown."""
    path = _snooze_path(hermes_home)
    try:
        path.write_text(str(time.time()))
    except OSError:
        pass


def should_offer(
    hermes_home: Path,
    project_root: Path,
    *,
    adopt_mode: str = "prompt",
    is_interactive: bool = True,
) -> bool:
    """Determine whether to show the adoption offer.

    Args:
        hermes_home: The HERMES_HOME directory.
        project_root: The running code's root (checkout or slot).
        adopt_mode: One of "auto", "prompt", "never" (from config).
        is_interactive: Whether stdin is a terminal.

    Returns:
        True if the offer should be shown, False otherwise.
    """
    if adopt_mode == "never":
        return False

    if _is_snoozed(hermes_home):
        return False

    # Detect legacy layout — crash-proof
    try:
        from hermes_cli.adoption import detect_legacy_install

        info = detect_legacy_install(project_root, hermes_home)
    except Exception as e:
        logger.debug("Adoption detection failed: %s", e)
        return False

    if info is None:
        return False  # Not a legacy install (slot, docker, nix, etc.)

    # For auto mode with pristine, auto-invoke adopt (interactive or not).
    # For prompt mode, show the offer text.
    if adopt_mode == "auto" and info.pristine:
        return True  # The caller will invoke adopt

    if not is_interactive:
        return False  # Don't show prompts in non-interactive mode

    return True


def offer_adoption(
    hermes_home: Path,
    project_root: Path,
    *,
    adopt_mode: str = "prompt",
    is_interactive: bool = True,
) -> None:
    """Show the adoption offer (or auto-invoke adopt). Never raises.

    This is the entry point called from main() at startup.
    """
    try:
        if not should_offer(
            hermes_home,
            project_root,
            adopt_mode=adopt_mode,
            is_interactive=is_interactive,
        ):
            return

        # Mark shown (even if we auto-invoke, so we don't loop)
        _mark_shown(hermes_home)

        # Detect again for the pristine check (crash-proof)
        try:
            from hermes_cli.adoption import detect_legacy_install

            info = detect_legacy_install(project_root, hermes_home)
        except Exception:
            return

        if info is None:
            return

        if adopt_mode == "auto" and info.pristine:
            # Auto-invoke adopt (interactive or not)
            import subprocess

            print("→ Auto-adopting to managed release bundles...")
            subprocess.Popen(
                ["hermes", "adopt", "--yes"],
                start_new_session=True,
            )
        else:
            # Show the offer text
            print(ADOPT_PROMPT_COPY)
    except Exception as e:
        logger.debug("Adoption offer failed: %s", e)
