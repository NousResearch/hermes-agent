"""Default configuration values for ``updates``."""

UPDATES_DEFAULTS = {
    # Passive version/banner checks only; explicit `hermes update --check` remains enabled.
    "check": True,
    # Bound network Git operations while allowing large partial-clone backfills more time.
    "fetch_timeout": 300,
    # Pre-update backup. quick = snapshot small critical state (pairing JSONs, cron jobs,
    # config.yaml, .env, auth.json, profile DBs) into <HERMES_HOME>/state-snapshots/, skipping
    # files >1 GiB; restore via ``/snapshot``. full = quick PLUS a ``hermes backup`` zip in
    # <HERMES_HOME>/backups/ (``hermes import`` restores; slow on large homes; ``--backup``
    # forces once). off = none (``--no-backup`` forces once). Legacy booleans: true -> full,
    # false -> off.
    # Pre-update safety backup — ONE consolidated mechanism, three modes: Files over 1 GiB (e.g. a
    # bloated state.db) are skipped with a warning so the snapshot stays fast. This is the #48200
    # (wrong-path wipe) safety net.
    "pre_update_backup": "quick",
    # Full backup zips to retain (older pruned after each success; floored to 1 so the newest is
    # always kept). The quick snapshot always keeps exactly 1.
    "backup_keep": 5,
    # Uncommitted source-tree changes during NON-interactive updates (desktop, gateway — no TTY;
    # interactive updates always stash and ask). stash = stash, pull, restore on top (conflicts
    # stay in a git stash). discard = stash and drop after the pull (stash-and-drop, not reset
    # --hard + clean -fd, so ignored paths like node_modules/venv are never touched).
    "non_interactive_local_changes": "stash",
    # If the checkout is parked on a feature branch and the tree is clean, switch to the update
    # target (commits stay on the branch; a loud notice names it) so non-interactive updates
    # keep working. A DIRTY tree blocks the switch and the code update is SKIPPED with a loud
    # warning. False = never auto-switch.
    "auto_switch_parked_branch": True,
    # Clean parked branch with unmerged commits: switch = move to the update target, commits
    # stay on the branch (never conflicts). update_in_place = for a maintained custom branch:
    # merge origin/<target> INTO it after leaving a pre-update-<stamp> tag; a conflict stops the
    # update cleanly. `hermes update --switch-branch` overrides to switch for one run.
    "parked_branch_strategy": "switch",
    # Refresh an installed cua-driver during `hermes update` (best-effort, macOS only). Turn off
    # e.g. on non-admin accounts where /Applications isn't writable.
    "refresh_cua_driver": True,
}
