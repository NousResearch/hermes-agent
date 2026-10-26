"""``hermes update``'s owed ``windows_home_acl`` step: after an elevated update, give the same
user back access to the home tree that run may have left Administrators-only (#139870, #134889 W8).
Split from ``update_cmd_maint.py``."""
import logging

logger = logging.getLogger("hermes_cli.update_cmd")


def repair_windows_home_acl(*, elevated: bool | None = None) -> None:
    """Restore access for the filtered UAC token after an elevated update.

    An elevated process can create home-tree entries whose DACL only grants
    Administrators and SYSTEM.  Administrators is deny-only in the normal UAC
    token, so grant the actual user SID explicitly on every entry under the home.

    The walk never descends into or grants through a junction, symlink or hard link
    (``hermes_dir_acl.grant_tree_user_access``): ``icacls /T`` follows junctions, so a
    same-user junction inside the home would have turned the elevated grant into full
    control of e.g. ``C:\\Program Files\\X``.

    *elevated* is the host fact (default: probed; False on every host but an elevated Windows one).
    """
    from hermes_dir_acl import grant_tree_user_access, process_is_elevated
    if not (process_is_elevated() if elevated is None else elevated):
        # Only an elevated update leaves Administrators-only entries, and only an elevated token
        # can rewrite them; a recursive grant over the whole home on every update costs minutes.
        return
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    # The home itself is the operator's choice and may be a link: pin its real directory and walk
    # below it (links beneath are still never followed).
    granted, skipped, failed = grant_tree_user_access(home, pin_root=True, elevated=True)
    if failed and not granted:
        path, reason = failed[0]
        raise RuntimeError(f"could not restore user access to {home}: {path}: {reason}")
    if not granted:
        # Nothing took (the root itself was skipped or unreadable): the step stays owed, never a
        # silent success that leaves stranded state stranded.
        why = f"{skipped[0]} is a junction, symlink or hard link" if skipped else "nothing was walkable"
        raise RuntimeError(f"could not restore user access to {home}: nothing was repaired ({why})")
    if failed:
        # Like icacls /C: keep going, report the first few; the step stays owed only when nothing took.
        logger.warning("Restoring user access under %s failed for %d entr%s, e.g. %s: %s", home, len(failed),
                       "y" if len(failed) == 1 else "ies", failed[0][0], failed[0][1])
