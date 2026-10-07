"""ZIP-download fallback for ``hermes update`` (Windows with broken git): two-phase stage/commit swap, dirty-tree guard.

Split out of ``update_cmd.py``; every name is re-imported there so ``hermes_cli.update_cmd.<name>`` keeps
resolving/monkeypatching. Origin helpers are imported lazily per function (no cycle; test patches stay effective).
"""

import logging
from contextlib import suppress
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Collection, Optional

from hermes_cli._early_recovery import _keep_aside
from hermes_cli._early_recovery_zip import (
    ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap, write_zip_swap_journal, zip_entry_identity, zip_swap_owner_lock)

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")

_ZIP_STAGING_ARTIFACT_SUFFIXES = ".hermes-update-staging", ".hermes-update-old"
# Hermes' own root breadcrumbs: the swap journal + its owner lock, and the update/refresh markers.
# They are never user work, so the dirty-tree guard must not refuse (and wedge) on them.
_ZIP_HERMES_ROOT_ARTIFACTS = frozenset({
    ".hermes-update-zip-swap", ".hermes-update-zip-swap.lock", ".update-incomplete", ".lazy-refresh-incomplete"})
# A journal temp a killed writer left (``write_durable_text``'s unpredictable name). The swap never
# touches it (no ZIP ships it) and nothing deletes it, so admitting it risks no user bytes. A file at
# the old fixed ``.hermes-update-zip-swap.tmp`` name is no longer Hermes': it blocks (review Z5).
_ZIP_JOURNAL_TEMP = re.compile(r"\.hermes-update-zip-swap\.[0-9a-f]{12}\.tmp")

# Single source of truth for entries the ZIP swap preserves — used by the dirty-tree filter and the swap loop.
_ZIP_PRESERVED_TOP_LEVEL = {"venv", ".venv", "node_modules", ".git", ".env"}

# Gitignored build outputs the source ZIP never ships, nested under top-level entries the swap replaces
# (so `_ZIP_PRESERVED_TOP_LEVEL` cannot shield them): the packaged Desktop app, its renderer bundle and
# its own node_modules (electron itself), and the dashboard assets. The dirty-tree guard admits them and
# `_stage_entries` grafts the live copies into the staged tree so the swap keeps them (#90495).
_ZIP_PRESERVED_NESTED = {
    "apps": ("desktop/release", "desktop/dist", "desktop/node_modules", "desktop/build"),
    "hermes_cli": ("web_dist",),
    "scripts": ("whatsapp-bridge/node_modules",),
    "ui-tui": ("dist", "node_modules", "packages/hermes-ink/dist"),
    "web": ("node_modules",),
}

_STASH_HINT = "  Stash or commit your changes, then rerun `hermes update`."


def _remove_path(path: str, *, ignore_errors: bool = False) -> None:
    """Remove a dir or file; missing paths are a no-op."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        if ignore_errors:
            with suppress(OSError):
                os.remove(path)
        else:
            os.remove(path)


def _atomic_replace_dir(src: str, dst: str) -> None:
    """Replace *dst* with *src* without a half-deleted window (naive ``rmtree; copytree`` loses the old
    tree when the copy fails partway — likely here, since the ZIP path only runs when file I/O is flaky).
    Thin alias over the two-phase helpers; retained for the ``hermes_cli.main`` re-export surface.

    The naive ``rmtree(dst); copytree(src, dst)`` has a destructive window: if the copy fails partway
    (common on the Windows ZIP-update path, which only runs because file I/O is already flaky on that
    machine), the old directory is already gone and nothing replaced it — the install is left with a deleted
    tree (issue #49145, where ``ui-tui/`` vanished and broke the TUI).
    Now a thin single-entry alias over the two-phase helpers below, which generalise the same
    stage-then-swap discipline across every entry the ZIP update touches (#76104).
    """
    _commit_staged_replacements([(_stage_replacement(src, dst), dst)])


def _stage_replacement(src: str, dst: str, on_created=None) -> str:
    """Phase 1: copy *src* (dir or file) to a sibling staging path for *dst*; return it.
    Touches nothing live, so a failure here leaves the install untouched. ``on_created(staging)`` runs
    once the staging entry exists (a directory: before it is filled), so the journal can record it."""
    staging = f"{dst}.hermes-update-staging"
    backup = f"{dst}.hermes-update-old"
    # A prior run may have died mid-swap leaving the backup as the ONLY copy. Restore it BEFORE
    # clearing leftovers, else deleting it then failing to stage (disk exhaustion) leaves a hole.
    if not os.path.lexists(dst) and os.path.lexists(backup):  # a dangling symlink backup is still one (Z3)
        os.rename(backup, dst)
    # Nothing this run made is there yet: an entry at either suffix is not provably the update's (a
    # user's file created after the clean-tree preflight, or a pre-journal crash's), so it is kept aside,
    # never deleted (F78). Neither may stay: a backup would later be taken for this swap's own.
    for leftover in (staging, backup):
        if os.path.lexists(leftover):
            print(f"  ⚠ Kept {leftover} aside as {_keep_aside(Path(leftover)).name}: nothing proves it is "
                  "this update's.")
    # Never through the reusable pathname: copy2 opens it following a symlink planted after the sweep
    # (review Z2). copytree's own os.mkdir is exclusive and never follows; a file is created the same way.
    if os.path.isdir(src):
        os.mkdir(staging)
    else:
        _copy_file_exclusive(src, staging)
    made = zip_entry_identity(staging)
    try:
        if on_created is not None:
            on_created(staging)
        if zip_entry_identity(staging) != made:  # swapped while the journal recorded it: never fill that
            raise OSError(f"the staging entry {staging} was replaced while it was being created")
        if os.path.isdir(src):
            shutil.copytree(src, staging, dirs_exist_ok=True)
    except BaseException:
        # A failed stage leaves nothing of its own (a full disk must not shrink further on the retry).
        # Best effort, like _discard_staged: the journal's recovery drops what this cannot.
        with suppress(OSError):
            if zip_entry_identity(staging) == made:
                _remove_path(staging)
        raise
    return staging


def _copy_file_exclusive(src: str, dst: str, *, sync: bool = False, on_created=None) -> None:
    """``copy2`` that only ever creates ``dst``: O_CREAT|O_EXCL (+O_NOFOLLOW) fails on any entry there
    (symlink, hardlink, file) instead of writing through it, and the bytes and mode go through the held
    fd. Times are copied only where that can be done without following ``dst``. ``on_created(dst)``
    runs once ``dst`` exists, before a byte is written, so a journal can record what this run made."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    st = os.stat(src)
    fd = os.open(dst, flags, 0o600)
    try:
        if on_created is not None:
            on_created(dst)
        with os.fdopen(fd, "wb") as out, open(src, "rb") as source:
            shutil.copyfileobj(source, out, 1 << 20)
            out.flush()
            if os.chmod in os.supports_fd:
                os.chmod(out.fileno(), st.st_mode & 0o7777)
            if os.utime in os.supports_fd:
                os.utime(out.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
            if sync:
                os.fsync(out.fileno())
    except BaseException:
        with suppress(OSError):  # our own half-written create: never leave it for the swap to install
            os.remove(dst)
        raise


def _discard_staged(staged) -> None:
    """Remove staging paths for never-committed entries; otherwise a phase-1 failure (disk exhaustion)
    orphans up to a full second tree and the advised retry fails harder with less free space."""
    for staging, _dst in staged:
        try:
            _remove_path(staging)
        except OSError as exc:  # best-effort cleanup, never fatal
            logger.warning("could not remove staging path %s: %s", staging, exc)


def _hardlink_backup(path: str, backup: str) -> bool:
    """Hardlink ``path`` to ``backup``; False where links are unsupported. A link is all-or-nothing, so a
    killed swap never leaves a short backup the restore would put back."""
    try:
        os.link(path, backup)
    except OSError:
        return False
    return True


def _file_backup(path: str, backup: str, tag: str = "", on_temp=None) -> None:
    """Back up the regular file ``path`` as a complete ``backup`` while ``path`` stays in place: a hardlink,
    or where links are unsupported (FAT32/exFAT/SMB) a synced copy landed whole by ``os.replace``. The
    copy's temp carries the journal's random ``tag``, and ``on_temp(temp)`` journals it as it comes into
    existence: recovery deletes a temp only on that record (F78-R)."""
    if _hardlink_backup(path, backup):
        return
    tmp = f"{backup}.{tag or os.urandom(6).hex()}.tmp"
    try:
        _copy_file_exclusive(path, tmp, sync=True, on_created=on_temp)  # never through a planted link (Z2)
        os.replace(tmp, backup)
    except OSError:
        with suppress(OSError):
            os.remove(tmp)
        raise


def _commit_staged_replacements(staged, *, on_committed=None, tag: str = "", on_temp=None) -> None:
    """Phase 2: swap every staged entry into place, rolling back all on failure.

    Per-entry safety wasn't enough: a partway failure over ~90 entries left a mixed-version tree (every
    file valid, combination unbootable). Covers plain files too (repo root holds 20 first-party modules).
    Each swap is an ``os.rename`` onto a just-moved-aside path — atomic on POSIX and NTFS, unlike
    ``copy2`` onto a live path. Stage-all-then-swap-all shrinks the failure window to N renames and a
    failed swap restores every entry already swapped, so the tree lands wholly new or wholly old.

    ``_atomic_replace_dir`` makes each *individual* directory swap safe, but the ZIP update replaces ~90
    top-level entries in a loop, and nothing made the loop atomic *as a whole*. See #63717, #76091, #76104.

    A plain file is never absent, not even for one rename: its backup is a hardlink (or, without
    hardlinks, a complete copy) and ``os.replace`` lands the new bytes over it; only directories move
    aside. The root modules every launcher imports first
    (``hermes_constants``, ``hermes_bootstrap``) therefore always import, and ``hermes_bootstrap`` runs the
    journal-driven restore even while a killed swap left a directory (``hermes_cli/`` included) moved aside.
    """
    swapped: list[tuple[str, str]] = []  # (dst, backup) in swap order; "" = absent
    try:
        for staging, dst in staged:
            backup = f"{dst}.hermes-update-old"
            if os.path.lexists(backup):  # appeared since staging set leftovers aside: not ours to replace (F78)
                raise FileExistsError(f"{backup} appeared during the update; it was left in place")
            if os.path.isfile(dst) and not os.path.islink(dst):
                _file_backup(dst, backup, tag, on_temp)
                swapped.append((dst, backup))
                os.replace(staging, dst)
                continue
            if os.path.lexists(dst):
                os.rename(dst, backup)
                swapped.append((dst, backup))
            else:
                swapped.append((dst, ""))
            os.rename(staging, dst)
    except OSError:
        for dst, backup in reversed(swapped):  # undo every swap already made so the install stays self-consistent
            try:
                if backup and os.path.isfile(backup) and not os.path.islink(backup):
                    os.replace(backup, dst)
                    continue
                _remove_path(dst)
                if backup and os.path.lexists(backup):
                    os.rename(backup, dst)
            except OSError as exc:
                # Keep restoring the rest; a silent failure here turns a recoverable rollback into a mixed tree.
                logger.warning("rollback failed for %s: %s", dst, exc)
        raise
    if on_committed is not None:
        on_committed()  # every rename landed: an interrupted cleanup now finishes, never rolls back
    for _dst, backup in swapped:  # all swaps succeeded — drop the backups (best-effort, never fatal)
        if backup:
            _remove_path(backup, ignore_errors=True)


def _zip_overlay_block_reason(
    root: Path, *, staged: Collection[str] = (), shipped: Optional[Collection[str]] = None,
) -> Optional[str]:
    """Why overlaying a ZIP onto ``root`` would destroy work, or None if safe.

    The swap replaces every top-level entry (minus a tiny preserve set) and deletes backups, so uncommitted
    edits and untracked files are gone. Fails closed when git status cannot run. ``staged`` is for the
    pre-swap re-check: phase 1 leaves ``<entry>.hermes-update-staging`` for those entries, which git
    reports as untracked; without the filter the re-check always refuses. ``shipped`` is the extracted
    ZIP's top-level entry set once known (the re-check); before the download the tracked root entries stand
    in for it. A gitignored path under a root entry the ZIP does not ship is never touched by the swap.

    Fail closed when git status cannot run: unknown dirtiness is not a license to clobber the tree (#87304).
    """
    if not (root / ".git").exists():
        return None
    from hermes_cli.update_custody import run_git

    git_cmd = ["git", "-c", "windows.appendAtomically=false"] if sys.platform == "win32" else ["git"]
    result = run_git(
        # -uall: a user-level ``status.showUntrackedFiles = no`` must not blind this guard. --ignored=matching:
        # gitignored files are still USER DATA the overlay would delete; ``matching`` reports an ignored dir
        # as one ``dir/`` line. ``--ignored=all`` is NOT a valid git mode (exits 128, would fail-close every update).
        # ``matching`` reports an ignored directory as one ``dir/`` line instead of enumerating its contents
        # (cheaper, same verdict for the top-level filter below). See #87392.
        git_cmd, ["status", "--porcelain", "--untracked-files=all", "--ignored=matching"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode == 0 and shipped is None:
        # Before the download the ZIP's entry set is unknown; the tracked root entries stand in for it (the
        # pre-swap re-check gets the real set), so an ignored root entry the swap never touches cannot refuse.
        tracked = run_git(
            git_cmd, ["ls-tree", "--name-only", "HEAD"],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        shipped = set(tracked.stdout.splitlines()) if tracked.returncode == 0 else None
        result.returncode = tracked.returncode
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return f"could not check the working tree{f' ({detail[0]})' if detail else ''}"
    # Preserved entries (venv, node_modules are gitignored on every normal install) are never touched by the
    # swap, so they must not cause a false refusal. Everything else — including ignored files — blocks.
    dirty = any(
        line.strip()
        and not _is_zip_preserved_entry_status_line(line, shipped)
        and not _is_zip_staging_artifact_status_line(line, staged)
        for line in (result.stdout or "").splitlines()
    )
    return "the working tree has uncommitted changes or untracked files" if dirty else None


def _status_top_level(path: str) -> str:
    return path.strip().strip('"').replace("\\", "/").rstrip("/").split("/", 1)[0]


def _is_zip_preserved_entry_status_line(line: str, shipped: Optional[Collection[str]] = None) -> bool:
    """True when the swap would not destroy what a porcelain status line names: every path sits under a
    preserved top-level entry; or the line is gitignored (``!!``) and under a root entry the ZIP does not
    ship (``.bytecode-fingerprint``, ``.hermes-bootstrap-complete``, ``hermes_agent.egg-info/`` — the swap
    replaces ``shipped`` entries only); or a ``!!`` build output nested under a shipped dir that the swap
    keeps (`_ZIP_PRESERVED_NESTED`) or regenerates (``__pycache__``, ``node_modules``). Every real install
    has all of these, and blocking on them made the ZIP fallback refuse every install. Tracked edits,
    renames and other untracked/ignored user files still block.

    The ``" -> "`` split applies ONLY to R/C codes: porcelain v1 doesn't quote plain names with spaces, so
    ``venv -> node_modules`` on a ``!!``/``??`` line is ONE path and splitting would fail-open. Requiring
    EVERY path preserved keeps renames out of a preserved dir (``R venv/x -> src/x``) blocking.
    """
    status, payload = (line[:2], line[3:]) if len(line) >= 3 else ("", line)
    paths = payload.split(" -> ") if any(code in "RC" for code in status) else [payload]
    if all(_status_top_level(path) in _ZIP_PRESERVED_TOP_LEVEL for path in paths):
        return True
    if status != "!!":
        return False
    path = payload.strip().strip('"').replace("\\", "/").rstrip("/")
    top, _, nested = path.partition("/")
    if shipped is not None and top not in shipped:
        return True
    return path.rsplit("/", 1)[-1] in ("__pycache__", "node_modules") or any(
        nested == keep or nested.startswith(f"{keep}/") for keep in _ZIP_PRESERVED_NESTED.get(top, ()))


def _is_zip_staging_artifact_status_line(line: str, staged: Collection[str] = ()) -> bool:
    """True when a porcelain status line is Hermes' own: a root breadcrumb, or the staging copy of an
    entry this run staged. Any other ``*.hermes-update-staging``/``-old`` may be a user's file (a manual
    ``cp -r tools tools.hermes-update-old``) that staging would delete, so it blocks."""
    top = _status_top_level(line[3:] if len(line) >= 3 else line)
    return (top in _ZIP_HERMES_ROOT_ARTIFACTS or bool(_ZIP_JOURNAL_TEMP.fullmatch(top))
            or top in {item + _ZIP_STAGING_ARTIFACT_SUFFIXES[0] for item in staged})


def _abort_zip_update_if_dirty_tree() -> None:
    """Refuse to overlay a ZIP onto a dirty git checkout.

    See #87304.
    """
    from hermes_cli.update_cmd import _m
    # Settle an interrupted swap first: only its journal tells its staging/backup siblings apart from a
    # user's lookalike file, and once settled none are left to refuse the retry on.
    restore_interrupted_zip_swap(_m().PROJECT_ROOT)
    reason = _zip_overlay_block_reason(_m().PROJECT_ROOT)
    if reason is None:
        return
    print(f"✗ ZIP fallback refused: {reason}.")
    print("  Overlaying the ZIP would overwrite uncommitted edits and permanently delete untracked files.")
    print(_STASH_HINT)
    print("  To inspect: git status --porcelain")
    _m().sys.exit(1)


def _extract_zip_safely(zip_path: str, tmp_dir: str) -> None:
    """Extract, rejecting zip-slip AND symlink members: a source ZIP never legitimately contains
    symlinks, and a compromised mirror could use them to plant files anywhere."""
    import stat as _stat
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        tmp_dir_real = os.path.realpath(tmp_dir)
        for member in zf.infolist():
            member_path = os.path.realpath(os.path.join(tmp_dir, member.filename))
            if not member_path.startswith(tmp_dir_real + os.sep) and member_path != tmp_dir_real:
                raise ValueError(f"Zip-slip detected: {member.filename} escapes extraction directory")
            # Unix mode lives in the upper 16 bits of external_attr; mask to the file-type bits.
            if _stat.S_ISLNK((member.external_attr >> 16) & 0o170000):
                raise ValueError(f"ZIP contains unsupported symlink member: {member.filename}")
        zf.extractall(tmp_dir)


def _archive_commit(zip_path: str) -> Optional[str]:
    """The commit a GitHub archive was made from: ``git archive`` stores it as the ZIP comment."""
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        comment = zf.comment.decode("ascii", "replace").strip()
    return comment if re.fullmatch(r"[0-9a-f]{40}", comment) else None


def _extracted_root(tmp_dir: str, branch: str) -> str:
    """GitHub ZIPs extract to ``hermes-agent-<branch>/``; fall back to the first non-``__MACOSX`` dir."""
    extracted = os.path.join(tmp_dir, f"hermes-agent-{branch}")
    if not os.path.isdir(extracted):
        for d in os.listdir(tmp_dir):
            candidate = os.path.join(tmp_dir, d)
            if os.path.isdir(candidate) and d != "__MACOSX":
                return candidate
    return extracted


def _require_staging_space(extracted: str, entries: list[str], project_root: str) -> None:
    """Staging costs one extra tree copy; swaps are renames, so require the copy plus 20% headroom — not 2x,
    which would block updates on exactly the space-constrained machines that hit this path."""
    need = sum(
        # Two-phase replace (#76104). Phase 1 copies every entry — directories AND top-level files — to a
        # sibling staging path without touching anything live; phase 2 swaps them all in with
        # same-filesystem renames and rolls back every swap if any one fails. Replacing entries
        # one-at-a-time (the previous shape) meant an interruption partway left `agent/` new and `tools/`
        # stale — all files valid, the tree unbootable. Files matter as much as directories here: the repo
        # root holds 20 first-party modules (run_agent.py, cli.py, hermes_constants.py, ...). Check up front
        # so we fail with a clear message instead of running out mid-copy.
        os.path.getsize(os.path.join(dirpath, f))
        for entry in entries
        for dirpath, _dirs, files in os.walk(os.path.join(extracted, entry))
        for f in files
    ) + sum(os.path.getsize(os.path.join(extracted, e)) for e in entries if os.path.isfile(os.path.join(extracted, e)))
    required = int(need * 1.2)
    free = shutil.disk_usage(project_root).free
    if free < required:
        raise RuntimeError(
            f"not enough free disk space to stage the update safely "
            f"(need ~{required // (1024 * 1024)} MB, have {free // (1024 * 1024)} MB)"
        )


def _link_or_copy_artifact(source: str, destination: str) -> None:
    """Hardlink where the filesystem allows (apps/desktop/node_modules is hundreds of MB, and a link stays
    valid after the swap unlinks the old tree; on Windows a link also succeeds on a locked Hermes.exe
    where copy2 raises); byte copy otherwise."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _graft_nested_artifacts(item: str, live: str, staging: str) -> None:
    """Clone the live build outputs under *item* into its staged copy so the swap keeps them.
    A path the ZIP ships wins over the live copy; a never-built install has nothing to graft."""
    for nested in _ZIP_PRESERVED_NESTED.get(item, ()):
        source = os.path.join(live, *nested.split("/"))
        destination = os.path.join(staging, *nested.split("/"))
        if os.path.lexists(destination) or not os.path.isdir(source):
            continue
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copytree(source, destination, symlinks=True, copy_function=_link_or_copy_artifact)


def _stage_entries(extracted: str, entries: list[str], project_root: str,
                   on_created=None) -> list[tuple[str, str]]:
    """Phase 1 for every entry; on failure nothing is live yet, so drop partial staging copies so a retry
    starts from the same free space. An entry is listed only once its staging copy is this run's: a copy
    that fails partway drops its own partial tree (``_stage_replacement``), and a pathname this run never
    created (a user's file there) is never ``_discard_staged``'s to delete (F78).
    ``on_created(index, staging)``: the journal's hook as each staging entry comes into existence."""
    staged: list[tuple[str, str]] = []
    try:
        for index, item in enumerate(entries):
            dst = os.path.join(project_root, item)
            staged.append((_stage_replacement(
                os.path.join(extracted, item), dst,
                None if on_created is None else (lambda path, i=index: on_created(i, path))), dst))
            # The source ZIP carries only source; the built outputs (#70337/#87331 release/, then
            # dist/, apps/desktop/node_modules and web_dist — #90495) exist only in the LIVE tree. Graft
            # them into the staged copy BEFORE the swap so the commit preserves them atomically.
            _graft_nested_artifacts(item, dst, staged[-1][0])
    except Exception:
        _discard_staged(staged)
        raise
    return staged


def _drop_journal_if_clean(root: Path, entries: list[str], suffixes=(".hermes-update-staging",)) -> None:
    """The journal is the only record of the swap's siblings (``<entry><suffix>``): it goes only once
    every one of them is gone, else the next launch's recovery could not tell them from a user's file."""
    left = [item for item in entries if any(os.path.lexists(os.path.join(root, item + x)) for x in suffixes)]
    if left:
        print(f"  ⚠ Could not remove the update's own copies of {', '.join(left)}; the next `hermes` launch "
              "removes them.")
        return
    (root / ZIP_SWAP_JOURNAL).unlink(missing_ok=True)


def _journaled_stage_and_swap(extracted: str, entries: list[str], root: Path, target_sha) -> list:
    """Stage + swap under the ZIP swap journal and its owner lock (``_early_recovery``), so a kill at
    any point is finished or rolled back by the next launch and no staging/backup sibling leaks."""
    from hermes_cli import update_cmd_commit as _commit
    from hermes_cli.update_cmd import _UPDATE_CRITICAL_FILES, _m

    # A previous run killed mid-swap: settle it before staging over its leftovers. A journal it could
    # not settle is that swap's only record: never overwrite it with this run's.
    restore_interrupted_zip_swap(root)
    if (root / ZIP_SWAP_JOURNAL).exists():
        raise RuntimeError("an interrupted ZIP update could not be settled; see the warning above")
    with zip_swap_owner_lock(root, wait=10.0) as owned:
        if not owned:
            raise RuntimeError("another `hermes update` is swapping this install right now"
                               if owned.reason == "busy" else f"no ZIP swap lock, no swap ({owned.reason})")
        # A leftover backup counts: _stage_replacement puts it back as the entry before the swap.
        # Each entry also carries the identities recovery needs before it may delete anything (review
        # Z1): its staging copy's, recorded as the copy comes into existence, and the live entry's (the
        # one its backup links or renames), recorded at the swapping phase.
        journal_entries = [[item, any(os.path.lexists(os.path.join(root, item + suffix))
                                      for suffix in ("", ".hermes-update-old")), "", ""] for item in entries]
        gen = os.urandom(6).hex()
        write_zip_swap_journal(root, "staging", journal_entries, gen)

        def record_staged(index: int, staging: str) -> None:
            # A directory's id must be durable before copytree fills it: a kill mid-fill then leaves a
            # partial tree recovery can prove is its own and delete. A file is already copied when this
            # runs, so its id rides the next directory's write (or the last entry's) instead of one
            # fsync'd rewrite per file (~9 ms each on ext4: ~1 s of a 123-entry stage, review K132361).
            journal_entries[index][2] = zip_entry_identity(staging)
            if index == len(entries) - 1 or os.path.isdir(os.path.join(extracted, entries[index])):
                write_zip_swap_journal(root, "staging", journal_entries, gen)

        try:
            staged = _stage_entries(extracted, entries, str(root), record_staged)
        except BaseException:
            _drop_journal_if_clean(root, entries)  # _stage_entries dropped its copies, if it could
            raise
        # A failure past here leaves the journal: the caller's restore_interrupted_zip_swap (or the next
        # launch) settles it and drops every staging copy, so the retry's free-space check finds no litter.
        # Never discard staging first: a staging copy still on disk is what proves its entry was never
        # installed, so recovery must not delete whatever (a user's new file) now sits at that path.
        # TOCTOU re-check right before the swap: download + extract + staging can take minutes and
        # work created meanwhile would be destroyed. Our own staging siblings are filtered out.
        recheck_reason = _zip_overlay_block_reason(root, staged=entries, shipped=entries)
        if recheck_reason is not None:
            _discard_staged(staged)
            _drop_journal_if_clean(root, entries)
            print(f"✗ ZIP fallback aborted before the swap: {recheck_reason}.")
            print("  Files appeared in the checkout while the update was downloading; committing the swap would delete them.")
            print(_STASH_HINT)
            _m().sys.exit(1)
        # Pre-commit gate: a target whose startup modules do not compile is refused untouched, judged
        # under the interpreter the target admits (``startup_syntax_error``, review N15).
        if broken := _commit.tree_syntax_error(Path(extracted), _UPDATE_CRITICAL_FILES):
            raise SyntaxError(f"{broken[0]}: {broken[1]}")
        # The commit point: tail + fleet restart owed before the first live rename.
        _commit.arm_commit_obligations(root, target_sha or "")
        for entry in journal_entries:
            entry[3] = zip_entry_identity(root / entry[0])
        write_zip_swap_journal(root, "swapping", journal_entries, gen)
        temps: dict[str, str] = {}

        def record_temp(tmp: str) -> None:  # a backup copy's temp, journaled before its first byte
            temps[os.path.basename(tmp)] = zip_entry_identity(tmp, filling=True)
            write_zip_swap_journal(root, "swapping", journal_entries, gen, temps)

        # Committed: every temp already became its backup, so the record drops them (a backup the
        # cleanup then frees must never leave an inode that vouches for a later file at a temp name).
        _commit_staged_replacements(
            staged, on_committed=lambda: write_zip_swap_journal(root, "committed", journal_entries, gen), tag=gen,
            on_temp=record_temp)
        # Committed: the new tree is whole. A journal that cannot go now (AV/indexer holding it) says
        # "committed", which the next launch's recovery settles by keeping the new tree. It stays, too,
        # while a backup (or backup temp) the best-effort cleanup could not remove is still on disk:
        # recovery's own terminal check, else that backup is orphaned and wedges the retry (Q1/F79).
        with suppress(OSError):
            _drop_journal_if_clean(root, entries, (".hermes-update-staging", ".hermes-update-old",
                                                   f".hermes-update-old.{gen}.tmp"))
    return staged


def _download_and_swap_zip(branch: str, zip_url: str, target_sha: str | None = None) -> Optional[str]:
    """Download the source ZIP for *branch* and two-phase swap it into the checkout; return the commit it
    installed: the archive's own, which must equal ``target_sha`` when one is pinned.
    ``sys.exit(1)`` on any failure; the install ends fully updated or fully rolled back.
    Two-phase: stage every entry (dirs AND top-level files) beside its target, then swap all in with
    same-filesystem renames, rolling back on failure — one-at-a-time replacement left a mixed, unbootable
    tree on interruption."""
    from hermes_cli.update_cmd import _m

    import tempfile
    from urllib.request import urlretrieve
    print("→ Downloading latest version...")
    tmp_dir = tempfile.mkdtemp(prefix="hermes-update-")
    try:
        zip_path = os.path.join(tmp_dir, f"hermes-agent-{branch}.zip")
        urlretrieve(zip_url, zip_path)
        print("→ Extracting...")
        _extract_zip_safely(zip_path, tmp_dir)
        # The installed identity comes from the downloaded bytes (``git archive``'s comment), pinned or
        # not: without it the restart debt is armed for '' and can never be proven discharged (S4).
        archived = _archive_commit(zip_path)
        if archived is None or (target_sha and archived != target_sha):
            raise ValueError(f"the archive names commit {archived or 'none'}, not "
                             f"{target_sha or 'a full commit SHA'}; refusing to install unidentified bytes")
        target_sha = archived
        extracted = _extracted_root(tmp_dir, branch)
        entries = [i for i in os.listdir(extracted) if i not in _ZIP_PRESERVED_TOP_LEVEL]
        project_root = str(_m().PROJECT_ROOT)
        _require_staging_space(extracted, entries, project_root)
        staged = _journaled_stage_and_swap(extracted, entries, Path(project_root), target_sha)
        print(f"✓ Updated {len(staged)} items from ZIP")
        return target_sha
    except Exception as e:  # health: allow BLE001 -- all-or-nothing boundary: any failure settles the journal, exits 1
        # The swap rolled itself back; a rollback that could not finish leaves the journal, which
        # this settles now (or the next launch's _early_recovery does) instead of leaking siblings.
        with suppress(Exception):
            restore_interrupted_zip_swap(_m().PROJECT_ROOT)
            if not (Path(_m().PROJECT_ROOT) / ZIP_SWAP_JOURNAL).exists():
                from hermes_cli import update_cmd_commit as _commit
                _commit.disarm_commit_obligations()  # the old tree is whole again: nothing is owed
        print(f"✗ ZIP update failed: {e}")
        # Two-phase replace commits all or rolls all back, so no mixed tree here — don't push a needless reinstall.
        print("  Your existing install was left in place.")
        print("  Re-run `hermes update` to retry; if the agent won't start, reinstall from https://hermes-agent.nousresearch.com")
        _m().sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



def _update_via_zip(args, *, had_desktop_app_before_update: bool = False,
                   target_sha: str | None = None, target_repository: str | None = None,
                   completion_request=None) -> bool:
    """Update via ZIP when Windows git file I/O fails; dependency/build failures propagate.

    A supplied commit keeps the archive on the target selected before Git failed.
    """
    from hermes_cli.update_cmd import _m, _complete_source_update
    # The static archive would silently ignore --branch — the exact silent-divergence bug it exists to
    # prevent. Refuse rather than lie.
    branch = _m()._resolve_update_branch(args)
    if branch != "main":
        print(f"✗ --branch={branch} is not supported on the Windows ZIP-fallback update path.")
        print(
            "  This path runs when git file I/O is broken on the system. "
            "Either resolve the git-side breakage (typically an antivirus "
            "or NTFS filter holding files open) and rerun `hermes update "
            f"--branch {branch}`, or update against main with `hermes update`."
        )
        _m().sys.exit(1)
    _abort_zip_update_if_dirty_tree()
    # Older callers lack the snapshot/receipt/lifecycle handoff. Refuse before swap.
    if completion_request is None:
        from hermes_cli._old_updater import stop_for_relaunch
        stop_for_relaunch(incomplete=True)
    if target_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", target_sha):
        raise ValueError("ZIP update requires an exact full commit SHA")
    ref = target_sha if target_sha is not None else f"refs/heads/{branch}"
    repository = target_repository or "NousResearch/hermes-agent"
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
            or any(part in (".", "..") for part in repository.split("/"))):
        raise ValueError("ZIP update requires a GitHub owner/repository")
    installed = _download_and_swap_zip(branch, f"https://github.com/{repository}/archive/{ref}.zip", target_sha)
    completion_request["expected_sha"] = installed or target_sha
    completion_request["apply_mode"] = "zip"
    _complete_source_update(completion_request)
    return True
