"""Launch-time staleness refusal for app surfaces.

Phase 3 task 3.3: in a checkout, launching `hermes desktop` / `hermes web` /
`hermes --tui` checks the ArtifactStamp and refuses with instructions when
stale or missing. `--build` preserves today's build-then-launch. In a SLOT,
no staleness check at all (bundle artifacts are always current by construction).

See docs/updater-world.md §2.9 and
docs/plans/updater-rework/04-phase3-ejected-dev.md task 3.3.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def is_slot_install(project_root: Path) -> bool:
    """Check if running from a slot (has manifest.json)."""
    return (project_root / "manifest.json").is_file()


def check_staleness(
    project_root: Path,
    surface: str,
    stamp_file: str,
    source_globs: list[str],
    dist_sentinel: str,
    *,
    has_build_flag: bool = False,
) -> bool:
    """Check if a build artifact is stale. Returns True if OK to launch.

    Args:
        project_root: The checkout root.
        surface: Human-readable surface name (e.g., "desktop", "TUI", "web").
        stamp_file: Path to the content-hash stamp (relative to project_root).
        source_globs: Source file globs to hash.
        dist_sentinel: Path to the dist output sentinel (relative to project_root).
        has_build_flag: If True, skip the check (user passed --build).

    Returns:
        True if the surface is current (or in a slot, or --build was passed).
        False if stale — caller should exit 4 with the refusal message.
    """
    # In a slot, no staleness check
    if is_slot_install(project_root):
        return True

    # --build flag bypasses the check
    if has_build_flag:
        return True

    # Try to use ArtifactStamp from dev_sync
    try:
        from hermes_cli.dev_sync import ArtifactStamp

        stamp = ArtifactStamp(
            stamp_file=project_root / stamp_file,
            source_globs=source_globs,
            project_root=project_root,
            dist_sentinel=project_root / dist_sentinel,
        )
        if stamp.needs_build():
            _print_staleness_message(surface)
            return False
    except Exception:
        # If ArtifactStamp isn't available or fails, don't block the launch
        pass

    return True


def _print_staleness_message(surface: str) -> None:
    """Print the staleness refusal message."""
    print(
        f"{surface} build is behind the source tree.\n"
        f"  run: hermes dev sync            # rebuild what changed\n"
        f"  or:  hermes {surface} --build   # build now and launch",
        file=sys.stderr,
    )
