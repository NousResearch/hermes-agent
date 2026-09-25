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

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")

_ZIP_STAGING_ARTIFACT_SUFFIXES = ".hermes-update-staging", ".hermes-update-old"

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


def _stage_replacement(src: str, dst: str) -> str:
    """Phase 1: copy *src* (dir or file) to a sibling staging path for *dst*; return it.
    Touches nothing live, so a failure here leaves the install untouched."""
    staging = f"{dst}.hermes-update-staging"
    backup = f"{dst}.hermes-update-old"
    # A prior run may have died mid-swap leaving the backup as the ONLY copy. Restore it BEFORE
    # clearing leftovers, else deleting it then failing to stage (disk exhaustion) leaves a hole.
    if not os.path.exists(dst) and os.path.exists(backup):
        os.rename(backup, dst)
    for leftover in (staging, backup):
        _remove_path(leftover)
    (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, staging)
    return staging


def _discard_staged(staged) -> None:
    """Remove staging paths for never-committed entries; otherwise a phase-1 failure (disk exhaustion)
    orphans up to a full second tree and the advised retry fails harder with less free space."""
    for staging, _dst in staged:
        try:
            _remove_path(staging)
        except OSError as exc:  # best-effort cleanup, never fatal
            logger.warning("could not remove staging path %s: %s", staging, exc)


def _commit_staged_replacements(staged) -> None:
    """Phase 2: swap every staged entry into place, rolling back all on failure.

    Per-entry safety wasn't enough: a partway failure over ~90 entries left a mixed-version tree (every
    file valid, combination unbootable). Covers plain files too (repo root holds 20 first-party modules).
    Each swap is an ``os.rename`` onto a just-moved-aside path — atomic on POSIX and NTFS, unlike
    ``copy2`` onto a live path. Stage-all-then-swap-all shrinks the failure window to N renames and a
    failed swap restores every entry already swapped, so the tree lands wholly new or wholly old.

    ``_atomic_replace_dir`` makes each *individual* directory swap safe, but the ZIP update replaces ~90
    top-level entries in a loop, and nothing made the loop atomic *as a whole*. See #63717, #76091, #76104.
    """
    swapped: list[tuple[str, str]] = []  # (dst, backup) in swap order; "" = absent
    try:
        for staging, dst in staged:
            backup = f"{dst}.hermes-update-old"
            if os.path.exists(dst):
                os.rename(dst, backup)
                swapped.append((dst, backup))
            else:
                swapped.append((dst, ""))
            os.rename(staging, dst)
    except OSError:
        for dst, backup in reversed(swapped):  # undo every swap already made so the install stays self-consistent
            try:
                _remove_path(dst)
                if backup and os.path.exists(backup):
                    os.rename(backup, dst)
            except OSError as exc:
                # Keep restoring the rest; a silent failure here turns a recoverable rollback into a mixed tree.
                logger.warning("rollback failed for %s: %s", dst, exc)
        raise
    for _dst, backup in swapped:  # all swaps succeeded — drop the backups (best-effort, never fatal)
        if backup:
            _remove_path(backup, ignore_errors=True)


def _zip_overlay_block_reason(
    root: Path, *, ignore_staging_artifacts: bool = False, shipped: Optional[Collection[str]] = None,
) -> Optional[str]:
    """Why overlaying a ZIP onto ``root`` would destroy work, or None if safe.

    The swap replaces every top-level entry (minus a tiny preserve set) and deletes backups, so uncommitted
    edits and untracked files are gone. Fails closed when git status cannot run. ``ignore_staging_artifacts``
    is for the pre-swap re-check: phase 1 leaves our own ``*.hermes-update-staging`` siblings that git
    reports as untracked; without the filter the re-check always refuses. ``shipped`` is the extracted
    ZIP's top-level entry set once known (the re-check); before the download the tracked root entries stand
    in for it. A gitignored path under a root entry the ZIP does not ship is never touched by the swap.

    Fail closed when git status cannot run: unknown dirtiness is not a license to clobber the tree (#87304).
    """
    if not (root / ".git").exists():
        return None
    git_cmd = ["git", "-c", "windows.appendAtomically=false"] if sys.platform == "win32" else ["git"]
    result = subprocess.run(
        # -uall: a user-level ``status.showUntrackedFiles = no`` must not blind this guard. --ignored=matching:
        # gitignored files are still USER DATA the overlay would delete; ``matching`` reports an ignored dir
        # as one ``dir/`` line. ``--ignored=all`` is NOT a valid git mode (exits 128, would fail-close every update).
        # ``matching`` reports an ignored directory as one ``dir/`` line instead of enumerating its contents
        # (cheaper, same verdict for the top-level filter below). See #87392.
        git_cmd + ["status", "--porcelain", "--untracked-files=all", "--ignored=matching"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode == 0 and shipped is None:
        # Before the download the ZIP's entry set is unknown; the tracked root entries stand in for it (the
        # pre-swap re-check gets the real set), so an ignored root entry the swap never touches cannot refuse.
        tracked = subprocess.run(
            git_cmd + ["ls-tree", "--name-only", "HEAD"],
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
        and not (ignore_staging_artifacts and _is_zip_staging_artifact_status_line(line))
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


def _is_zip_staging_artifact_status_line(line: str) -> bool:
    """True when a porcelain status line is our own two-phase-swap artifact."""
    return _status_top_level(line[3:] if len(line) >= 3 else line).endswith(_ZIP_STAGING_ARTIFACT_SUFFIXES)


def _abort_zip_update_if_dirty_tree() -> None:
    """Refuse to overlay a ZIP onto a dirty git checkout.

    See #87304.
    """
    from hermes_cli.update_cmd import _m
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


def _stage_entries(extracted: str, entries: list[str], project_root: str) -> list[tuple[str, str]]:
    """Phase 1 for every entry; on failure nothing is live yet, so drop partial staging copies so a retry
    starts from the same free space."""
    staged: list[tuple[str, str]] = []
    try:
        for item in entries:
            dst = os.path.join(project_root, item)
            staged.append((_stage_replacement(os.path.join(extracted, item), dst), dst))
            # The source ZIP carries only source; the built outputs (#70337/#87331 release/, then
            # dist/, apps/desktop/node_modules and web_dist — #90495) exist only in the LIVE tree. Graft
            # them into the staged copy BEFORE the swap so the commit preserves them atomically.
            _graft_nested_artifacts(item, dst, staged[-1][0])
    except Exception:
        _discard_staged(staged)
        raise
    return staged


def _download_and_swap_zip(branch: str, zip_url: str) -> None:
    """Download the source ZIP for *branch* and two-phase swap it into the checkout.
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
        extracted = _extracted_root(tmp_dir, branch)
        entries = [i for i in os.listdir(extracted) if i not in _ZIP_PRESERVED_TOP_LEVEL]
        project_root = str(_m().PROJECT_ROOT)
        _require_staging_space(extracted, entries, project_root)
        staged = _stage_entries(extracted, entries, project_root)
        try:
            # TOCTOU re-check right before the swap: download + extract + staging can take minutes and
            # work created meanwhile would be destroyed. Our own staging siblings are filtered out.
            recheck_reason = _zip_overlay_block_reason(
                _m().PROJECT_ROOT, ignore_staging_artifacts=True, shipped=entries)
            if recheck_reason is not None:
                _discard_staged(staged)
                print(f"✗ ZIP fallback aborted before the swap: {recheck_reason}.")
                print("  Files appeared in the checkout while the update was downloading; committing the swap would delete them.")
                print(_STASH_HINT)
                _m().sys.exit(1)
            _commit_staged_replacements(staged)
        except Exception:
            # Rollback restored swapped entries but staging copies for the rest remain; drop them or the
            # retry's up-front free-space check (runs BEFORE per-entry leftover cleanup) fails on our litter.
            # Safe post-rollback: _discard_staged skips paths that no longer exist.
            _discard_staged(staged)
            raise
        print(f"✓ Updated {len(staged)} items from ZIP")
    except Exception as e:
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
    _download_and_swap_zip(branch, f"https://github.com/{repository}/archive/{ref}.zip")
    completion_request["expected_sha"] = target_sha
    _complete_source_update(completion_request)
    return True
