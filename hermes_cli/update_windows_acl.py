"""``hermes update``'s owed ``windows_home_acl`` step: after an elevated update, give the same
user back access to the home tree that run may have left Administrators-only (#139870, #134889 W8).
Split from ``update_cmd_maint.py``.

The repair is a whole-home grant, and on a large home it is minutes to half an hour of work
(#143124). It runs inside the update's completion tail, which every later launch re-enters while
the tail is owed — so a repair that never finishes used to hold the gateway hostage. It therefore
runs under a wall-clock budget: what it grants stands (the ACE is idempotent), and a repair that
cannot finish stops, says what is left, and hands the operator the one command that finishes it,
instead of keeping the machine unable to start.
"""
import logging
import sys
import time

logger = logging.getLogger("hermes_cli.update_cmd")

#: How long the post-update tail may spend on the repair before it defers the rest. The tail sits
#: between the update and the gateway restart, so it gets a window rather than the whole job.
REPAIR_BUDGET_SECONDS = 10 * 60

#: Mirrors ``hermes_dir_acl._TREE_GRANT_PROGRESS_EVERY`` for the start line only.
_PROGRESS_EVERY = 10_000


def repair_windows_home_acl(*, elevated: bool | None = None, budget_s: float | None = None) -> None:
    """Restore access for the filtered UAC token after an elevated update.

    An elevated process can create home-tree entries whose DACL only grants
    Administrators and SYSTEM.  Administrators is deny-only in the normal UAC
    token, so grant the actual user SID explicitly on every entry under the home.

    The walk never descends into or grants through a junction, symlink or hard link
    (``hermes_dir_acl.grant_tree_user_access``): ``icacls /T`` follows junctions, so a
    same-user junction inside the home would have turned the elevated grant into full
    control of e.g. ``C:\\Program Files\\X``.

    *elevated* is the host fact (default: probed; False on every host but an elevated Windows one).
    *budget_s* bounds the whole repair (default :data:`REPAIR_BUDGET_SECONDS`; ``0`` or less runs it
    unbounded). A budget that runs out is not a failure: the grants already written stay, the step
    reports what is left and the command that finishes it, and the tail completes so the next launch
    is not made to repeat the same half-hour.
    """
    from hermes_dir_acl import AclGrantDeferred, grant_tree_user_access, process_is_elevated
    if not (process_is_elevated() if elevated is None else elevated):
        # Only an elevated update leaves Administrators-only entries, and only an elevated token
        # can rewrite them; a recursive grant over the whole home on every update costs minutes.
        return
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    budget = REPAIR_BUDGET_SECONDS if budget_s is None else budget_s
    deadline = None if budget <= 0 else time.monotonic() + budget
    window = f"{budget / 60:.0f} min" if budget >= 60 else f"{budget:.1f}s"
    if deadline is not None:
        # The count is only known after the walk, so the start line promises a window, not a size:
        # the point is that silence is now explained and bounded.
        print(f"  ⏳ Windows ACL repair: restoring your user access under {home} "
              f"(up to {window}, progress every {_PROGRESS_EVERY:,} entries; "
              f"`hermes update` re-runs it)", file=sys.stderr, flush=True)

    def progress(granted: int, skipped: list, failed: list) -> None:
        print(f"     {granted:,} granted, {len(skipped):,} links skipped, {len(failed):,} failed ...",
              file=sys.stderr, flush=True)

    try:
        # The home itself is the operator's choice and may be a link: pin its real directory and walk
        # below it (links beneath are still never followed).
        granted, skipped, failed = grant_tree_user_access(
            home, pin_root=True, elevated=True, deadline=deadline, progress=progress)
    except AclGrantDeferred as deferred:
        print(f"  ⚠ Windows ACL repair stopped at its {window} budget after "
              f"{deferred.granted:,} entries; the rest keeps its current ACLs. Finish it once from an "
              f"elevated prompt, then the next `hermes update` has nothing to do:\n"
              f"    icacls \"{home}\" /grant \"%USERNAME%\":(OI)(CI)F /T /C /Q",
              file=sys.stderr, flush=True)
        return
    if failed and not granted:
        path, reason = failed[0]
        raise RuntimeError(f"could not restore user access to {home}: {path}: {reason}")
    if not granted:
        # Nothing took (the root itself was skipped or unreadable): the step stays owed, never a
        # silent success that leaves stranded state stranded.
        why = f"{skipped[0]} is a junction, symlink or hard link" if skipped else "nothing was walkable"
        raise RuntimeError(f"could not restore user access to {home}: nothing was repaired ({why})")
    print(f"  ✅ Windows ACL repair: {granted:,} entries under {home}", file=sys.stderr, flush=True)
    if failed:
        # Like icacls /C: keep going, report the first few; the step stays owed only when nothing took.
        logger.warning("Restoring user access under %s failed for %d entr%s, e.g. %s: %s", home, len(failed),
                       "y" if len(failed) == 1 else "ies", failed[0][0], failed[0][1])
