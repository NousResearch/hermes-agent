"""Native local-change preservation lifecycle for ``hermes update``.

Opt-in, built on native Git refs/stashes plus updater locks and
interruption recovery. Separate from the merge-based
``updates.parked_branch_strategy: update_in_place`` setting, which it does
not redefine.

When ``updates.local_change_preservation: preserve`` (or
``--preserve-local-changes``) is enabled, a dirty parked-branch checkout
configured for ``update_in_place`` no longer exits before autostashing.
Instead:

1. Before source mutation, preserve HEAD (committed history) behind a
   ``refs/hermes-local-preservation/`` ref, and tracked + untracked
   working-tree content in two new stash entries. Pre-existing stash
   entries are never dropped or rewritten; preservation fails closed
   before mutation when it cannot be verified.
2. The update advances the base installation to the selected upstream
   revision with a clean tree, so conflicting groups cannot block it.
3. Under the selected restoration policy (``never`` by default,
   ``safe`` when explicitly selected) compatible groups are reapplied
   with syntax + critical-import validation. Conflicting or uncertain
   groups stay parked with exact recovery handles.
4. A persistent receipt names the upstream revision, recovery refs,
   active groups, inactive groups and reasons. CLI prints
   "updated with inactive customizations" (exit 0) to distinguish it
   from a real failure (exit 1); the same receipt fact flows to Desktop
   via the update receipt.

``--keep-stash`` always forces the ``never`` policy for the run, so the
desktop updater's explicit "park, never silently re-apply" contract is
unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PRESERVATION_REF_PREFIX = "refs/hermes-local-preservation/"
PRESERVATION_STASH_PREFIX = "hermes-local-preservation-"
PRESERVATION_MARKER_NAME = "hermes-local-preservation-in-progress"
PRESERVATION_RECEIPT_DIRNAME = "hermes-local-preservation"
PRESERVATION_SCHEMA = 1

GROUP_COMMITTED = "committed"
GROUP_TRACKED = "tracked"
GROUP_UNTRACKED = "untracked"
_GROUPS = (GROUP_COMMITTED, GROUP_TRACKED, GROUP_UNTRACKED)


@dataclass
class PreservationGroup:
    """One coherent customization group and its recovery handle."""

    name: str
    present: bool = False
    ref: str = ""
    status: str = "absent"
    reason: str = ""


@dataclass
class PreservationState:
    """Durable preservation captured before source mutation."""

    preservation_id: str
    pre_sha: str
    branch: str
    target_ref: str
    base_ref: str = ""
    groups: dict = field(default_factory=dict)
    stash_before: tuple = ()
    created_at: str = ""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _updates_config() -> dict:
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("updates", {})
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def preservation_enabled(args=None) -> bool:
    """True when the opt-in preservation lifecycle applies to this run."""
    if args is not None:
        if bool(getattr(args, "no_preserve_local_changes", False)):
            return False
        if bool(getattr(args, "preserve_local_changes", False)):
            return True
    try:
        value = _updates_config().get("local_change_preservation", "off")
    except Exception:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"preserve", "on", "true", "yes", "1"}


def restore_policy(args=None) -> str:
    """Selected restoration policy: ``"never"`` (default) or ``"safe"``.

    ``--keep-stash`` always forces ``"never"`` so explicit park semantics
    survive regardless of configured policy.
    """
    if args is not None and bool(getattr(args, "keep_stash", False)):
        return "never"
    raw = getattr(args, "restore_policy", None) if args is not None else None
    if raw in ("never", "safe"):
        return raw
    try:
        configured = str(
            _updates_config().get("local_change_restore_policy", "never")
        ).strip().lower()
    except Exception:
        return "never"
    return "safe" if configured == "safe" else "never"


def _git_run(git_cmd, args, cwd):
    from hermes_cli._subprocess_compat import windows_hide_flags

    return subprocess.run(
        list(git_cmd) + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=windows_hide_flags(),
    )


def _git_dir(repo_root: Path) -> Path:
    dot_git = Path(repo_root) / ".git"
    if dot_git.is_file():
        try:
            text = dot_git.read_text(encoding="utf-8-sig").strip()
        except OSError:
            return dot_git
        if text.startswith("gitdir:"):
            return Path(repo_root) / text[len("gitdir:"):].strip()
    return dot_git


def _marker_path(repo_root: Path) -> Path:
    return _git_dir(repo_root) / PRESERVATION_MARKER_NAME


def _receipt_dir(repo_root: Path) -> Path:
    return _git_dir(repo_root) / PRESERVATION_RECEIPT_DIRNAME


def _head_sha(git_cmd, repo_root) -> str:
    result = _git_run(git_cmd, ["rev-parse", "HEAD"], repo_root)
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _current_branch(git_cmd, repo_root) -> str:
    result = _git_run(git_cmd, ["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _status_porcelain(git_cmd, repo_root) -> str:
    result = _git_run(git_cmd, ["status", "--porcelain"], repo_root)
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def _stash_commit_shas(git_cmd, repo_root) -> tuple:
    result = _git_run(git_cmd, ["stash", "list", "--format=%H"], repo_root)
    if result.returncode != 0:
        return ()
    return tuple(line.strip() for line in (result.stdout or "").splitlines() if line.strip())


def _stash_ref_for_commit(git_cmd, repo_root, commit_sha: str) -> Optional[str]:
    result = _git_run(git_cmd, ["stash", "list", "--format=%gd %H"], repo_root)
    if result.returncode != 0:
        return None
    for line in (result.stdout or "").splitlines():
        selector, _, commit = line.partition(" ")
        if commit.strip() == commit_sha:
            return selector.strip() or None
    return None


def _resolve_stash_commit(git_cmd, repo_root, commit_sha: str) -> str:
    """Commit SHA for a stash entry; empty when unresolvable."""
    result = _git_run(git_cmd, ["rev-parse", "--verify", "--quiet", f"{commit_sha}^{{commit}}"], repo_root)
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _tracked_dirty(git_cmd, repo_root) -> bool:
    """True when staged or unstaged tracked content differs from HEAD."""
    for args in (["diff", "--quiet", "HEAD", "--"], ["diff", "--cached", "--quiet"]):
        result = _git_run(git_cmd, args, repo_root)
        if result.returncode not in (0, 1):
            return True
        if result.returncode == 1:
            return True
    return False


def _untracked_paths(git_cmd, repo_root) -> Optional[set]:
    result = _git_run(
        git_cmd, ["ls-files", "--others", "--exclude-standard", "-z"], repo_root
    )
    if result.returncode != 0:
        return None
    return {path for path in (result.stdout or "").split("\0") if path}


def _local_commit_count(git_cmd, repo_root, target_ref: str) -> int:
    result = _git_run(
        git_cmd, ["rev-list", "--count", f"{target_ref}..HEAD", "--"], repo_root
    )
    if result.returncode != 0:
        return -1
    try:
        return int((result.stdout or "").strip())
    except ValueError:
        return -1


def _promote_intent_to_add(git_cmd, repo_root, porcelain_z: str) -> None:
    """Promote ``git add -N`` entries so ``git stash push`` accepts them."""
    paths = tuple(
        record[3:]
        for record in (porcelain_z or "").split("\0")
        if len(record) > 3 and record[0] == " " and record[1] == "A"
    )
    if paths:
        print(f"→ Making {len(paths)} intent-to-add file(s) preservable...")
        _git_run(git_cmd, ["add", "--", *paths], repo_root)


def _clear_unmerged_index(git_cmd, repo_root) -> None:
    result = _git_run(git_cmd, ["ls-files", "--unmerged"], repo_root)
    if (result.stdout or "").strip():
        print("→ Clearing unmerged index entries from a previous conflict...")
        subprocess.run(
            list(git_cmd) + ["reset"], cwd=repo_root, capture_output=True
        )


def _stash_push(git_cmd, repo_root, message: str, include_untracked: bool) -> str:
    """Push one preservation stash; return its commit SHA or empty."""
    before = _git_run(git_cmd, ["rev-parse", "--verify", "refs/stash"], repo_root)
    before_sha = (before.stdout or "").strip() if before.returncode == 0 else ""
    args = ["stash", "push", "-m", message]
    if include_untracked:
        args.append("--include-untracked")
    push = _git_run(git_cmd, args, repo_root)
    after = _git_run(git_cmd, ["rev-parse", "--verify", "refs/stash"], repo_root)
    after_sha = (after.stdout or "").strip() if after.returncode == 0 else ""
    if not after_sha or after_sha == before_sha:
        return ""
    if push.returncode != 0:
        # Entry created but untracked cleanup failed (permission-denied
        # class): still durable, tree needs a reset to reach clean.
        _git_run(git_cmd, ["reset", "--hard", "HEAD"], repo_root)
    return after_sha


def preserve_local_changes(
    git_cmd, repo_root, target_ref: str, *, stamp: Optional[str] = None
) -> PreservationState:
    """Durably preserve HEAD, tracked and untracked content before mutation.

    Creates a base ref plus up to two new stash entries; never drops or
    rewrites pre-existing stashes. Raises ``SystemExit(1)`` before any
    source mutation when preservation cannot be verified.
    """
    repo_root = Path(repo_root)
    stamp = stamp or _utc_stamp()
    pre_sha = _head_sha(git_cmd, repo_root)
    if not pre_sha:
        print("✗ Could not resolve HEAD — refusing to preserve local changes.")
        raise SystemExit(1)
    branch = _current_branch(git_cmd, repo_root) or "HEAD"
    stash_before = _stash_commit_shas(git_cmd, repo_root)
    porcelain = _git_run(git_cmd, ["status", "--porcelain", "-z"], repo_root)
    if porcelain.returncode != 0:
        print("✗ Could not read the working tree — refusing to preserve local changes.")
        raise SystemExit(1)
    _clear_unmerged_index(git_cmd, repo_root)
    _promote_intent_to_add(git_cmd, repo_root, porcelain.stdout or "")

    preservation_id = f"{stamp}-{pre_sha[:12]}"
    # Never silently overwrite an unfinished preservation: its marker is the
    # only index of which stash belongs to which interrupted run. Overwriting
    # it would orphan the previous handles while their stashes remain.
    existing = read_in_progress_marker(repo_root)
    if existing is not None and existing.get("pre_sha") != pre_sha:
        pending_groups = [
            name for name, group in (existing.get("groups") or {}).items()
            if isinstance(group, dict) and group.get("present")
        ]
        if pending_groups:
            print("✗ A previous preservation run did not finish "
                  f"(id {existing.get('id')}); refusing to overwrite its marker before mutation.")
            print(f"  Pending groups: {', '.join(pending_groups)}; base ref: {existing.get('base_ref')}")
            print("  Resolve it first (restore the stashes above or clear the marker manually), then re-run.")
            raise SystemExit(1)
    committed_count = _local_commit_count(git_cmd, repo_root, target_ref)
    tracked = _tracked_dirty(git_cmd, repo_root)
    untracked = _untracked_paths(git_cmd, repo_root)
    if untracked is None:
        print("✗ Could not enumerate untracked files — update stopped before mutation.")
        raise SystemExit(1)
    if committed_count <= 0 and not tracked and not untracked:
        # Nothing to preserve: no base ref, no marker, no receipt spam.
        # Callers treat base_ref == "" as "lifecycle ran, nothing stored".
        groups: dict[str, PreservationGroup] = {
            GROUP_COMMITTED: PreservationGroup(name=GROUP_COMMITTED),
            GROUP_TRACKED: PreservationGroup(name=GROUP_TRACKED),
            GROUP_UNTRACKED: PreservationGroup(name=GROUP_UNTRACKED),
        }
        return PreservationState(
            preservation_id=preservation_id, pre_sha=pre_sha, branch=branch,
            target_ref=target_ref, base_ref="", groups=groups,
            stash_before=tuple(stash_before),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    base_ref = f"{PRESERVATION_REF_PREFIX}base-{branch}-{preservation_id}"
    if _git_run(git_cmd, ["update-ref", base_ref, pre_sha], repo_root).returncode != 0:
        print(f"✗ Could not preserve HEAD behind {base_ref} — update stopped before mutation.")
        raise SystemExit(1)
    verify = _git_run(git_cmd, ["rev-parse", "--verify", "--quiet", base_ref], repo_root)
    if verify.returncode != 0 or (verify.stdout or "").strip() != pre_sha:
        print(f"✗ Preservation of HEAD behind {base_ref} could not be verified — update stopped.")
        raise SystemExit(1)

    groups: dict[str, PreservationGroup] = {}
    if committed_count > 0:
        committed_ref = f"{PRESERVATION_REF_PREFIX}committed-{branch}-{preservation_id}"
        if _git_run(git_cmd, ["update-ref", committed_ref, pre_sha], repo_root).returncode != 0:
            print("✗ Could not preserve local commits — update stopped before mutation.")
            raise SystemExit(1)
        groups[GROUP_COMMITTED] = PreservationGroup(
            name=GROUP_COMMITTED, present=True, ref=committed_ref,
            status="preserved", reason=f"{committed_count} commit(s) not on {target_ref}",
        )
    else:
        groups[GROUP_COMMITTED] = PreservationGroup(name=GROUP_COMMITTED)

    if tracked:
        stash_sha = _stash_push(
            git_cmd, repo_root,
            f"{PRESERVATION_STASH_PREFIX}tracked-{preservation_id}", False,
        )
        if not stash_sha:
            print("✗ Could not stash tracked changes — update stopped before mutation.")
            raise SystemExit(1)
        groups[GROUP_TRACKED] = PreservationGroup(
            name=GROUP_TRACKED, present=True, ref=stash_sha,
            status="preserved", reason="staged and unstaged tracked content stashed",
        )
    else:
        groups[GROUP_TRACKED] = PreservationGroup(name=GROUP_TRACKED)
    # Re-read untracked after the tracked stash: the tracked push never
    # touches untracked paths, but a permission-denied tree may still hold
    # occupants that must not be mistaken for a clean tree.
    remaining_untracked = _untracked_paths(git_cmd, repo_root)
    if remaining_untracked is None:
        print("✗ Could not re-enumerate untracked files — update stopped before mutation.")
        raise SystemExit(1)
    if remaining_untracked:
        stash_sha = _stash_push(
            git_cmd, repo_root,
            f"{PRESERVATION_STASH_PREFIX}untracked-{preservation_id}", True,
        )
        if not stash_sha:
            print("✗ Could not stash untracked files — update stopped before mutation.")
            raise SystemExit(1)
        groups[GROUP_UNTRACKED] = PreservationGroup(
            name=GROUP_UNTRACKED, present=True, ref=stash_sha,
            status="preserved",
            reason=f"{len(remaining_untracked)} untracked file(s) stashed",
        )
    else:
        groups[GROUP_UNTRACKED] = PreservationGroup(name=GROUP_UNTRACKED)

    # Verify before mutation: clean tree, new refs resolve, old stashes intact.
    if _status_porcelain(git_cmd, repo_root).strip():
        # A permission-denied occupant can survive even a successful stash
        # push; it is durable in the stash, but the tree is not clean enough
        # to advance the base installation.
        print("✗ Working tree still dirty after preservation — update stopped before mutation.")
        print("  Your changes are preserved in the stashes named below; clean up the")
        print("  remaining paths manually, then re-run `hermes update`.")
        raise SystemExit(1)
    stash_after = _stash_commit_shas(git_cmd, repo_root)
    if any(sha not in stash_after for sha in stash_before):
        print("✗ A pre-existing stash entry went missing during preservation — update stopped.")
        raise SystemExit(1)
    for group in groups.values():
        if group.present and group.name != GROUP_COMMITTED and group.ref not in stash_after:
            print(f"✗ Preservation stash for '{group.name}' could not be verified — update stopped.")
            raise SystemExit(1)

    state = PreservationState(
        preservation_id=preservation_id, pre_sha=pre_sha, branch=branch,
        target_ref=target_ref, base_ref=base_ref, groups=groups,
        stash_before=tuple(stash_before),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_in_progress_marker(git_cmd, repo_root, state)
    return state


def _write_in_progress_marker(git_cmd, repo_root, state: PreservationState) -> None:
    """Record preservation so an interruption before/after source movement recovers."""
    try:
        payload = {
            "schema": PRESERVATION_SCHEMA, "id": state.preservation_id,
            "pre_sha": state.pre_sha, "branch": state.branch,
            "target_ref": state.target_ref, "base_ref": state.base_ref,
            "created_at": state.created_at, "pid": os.getpid(),
            "groups": {
                name: {"present": group.present, "ref": group.ref,
                       "status": group.status, "reason": group.reason}
                for name, group in state.groups.items()
            },
            "stash_before": list(state.stash_before),
        }
        marker = _marker_path(Path(repo_root))
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(marker)
    except OSError as exc:
        print(f"✗ Could not write the preservation marker ({exc}) — update stopped before mutation.")
        raise SystemExit(1)


def read_in_progress_marker(repo_root) -> Optional[dict]:
    """Pending preservation marker, or None; never raises."""
    try:
        marker = _marker_path(Path(repo_root))
        if not marker.is_file():
            return None
        payload = json.loads(marker.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def clear_in_progress_marker(repo_root) -> None:
    with suppress(OSError):
        _marker_path(Path(repo_root)).unlink(missing_ok=True)


def check_pending_preservation(git_cmd, repo_root) -> Optional[dict]:
    """Surface an interrupted preservation run without losing newer edits.

    Returns the pending marker when one exists, else None. A dirty tree on
    top of a pending marker fails closed: the caller must not mutate until
    the operator resolves it, or newer edits could be stranded.
    """
    marker = read_in_progress_marker(repo_root)
    if marker is None:
        return None
    head = _head_sha(git_cmd, repo_root)
    dirty = bool(_status_porcelain(git_cmd, repo_root).strip())
    print()
    print("⚠ A previous update's local-change preservation did not finish:")
    print(f"  id: {marker.get('id')}")
    print(f"  preserved HEAD: {(marker.get('pre_sha') or '')[:12]}  current HEAD: {(head or '')[:12]}")
    for name, group in (marker.get("groups") or {}).items():
        if isinstance(group, dict) and group.get("present"):
            print(f"  group '{name}': {group.get('ref')} ({group.get('reason')})")
    print(f"  base ref: {marker.get('base_ref')}")
    if dirty:
        print("  The working tree has newer edits — refusing to mutate until this is resolved.")
        print("  Recover with: git stash list --format='%gd %H %s' | grep hermes-local-preservation")
        raise SystemExit(1)
    if head and head != marker.get("pre_sha"):
        print("  The base installation already moved; preserved groups stay recoverable.")
    return marker


def _committed_still_active(git_cmd, repo_root, pre_sha: str) -> bool:
    if not pre_sha:
        return False
    result = _git_run(git_cmd, ["merge-base", "--is-ancestor", pre_sha, "HEAD"], repo_root)
    return result.returncode == 0


def _apply_one_stash(git_cmd, repo_root, stash_sha: str) -> tuple[bool, str]:
    """Apply one stash SHA; (ok, stderr). Never drops the entry."""
    apply_result = _git_run(git_cmd, ["stash", "apply", stash_sha], repo_root)
    unmerged = _git_run(
        git_cmd, ["diff", "--name-only", "--diff-filter=U"], repo_root
    )
    conflicted = (unmerged.stdout or "").strip()
    if apply_result.returncode == 0 and not conflicted:
        return True, apply_result.stderr or ""
    return False, (apply_result.stderr or "") + ("\n" + conflicted if conflicted else "")


def _restored_python_paths(git_cmd, repo_root) -> Optional[tuple]:
    diff = _git_run(git_cmd, ["diff", "--name-only", "-z", "HEAD", "--", "*.py"], repo_root)
    if diff.returncode != 0:
        return None
    paths = {path for path in (diff.stdout or "").split("\0") if path}
    untracked = _untracked_paths(git_cmd, repo_root)
    if untracked is None:
        return None
    paths.update(path for path in untracked if path.endswith(".py"))
    return tuple(sorted(paths))


def _validate_restored_tree(repo_root, git_cmd, clean_import_failures: dict) -> Optional[str]:
    """None when the restored tree passes syntax + import health, else a reason.

    Import health is differential against the clean updated tree: a fixture
    repo that already fails the probe before restoration must not park every
    group. Only a NEW failure counts, mirroring ``_restore_stashed_changes``.
    """
    from hermes_cli.update_cmd import _validate_python_files_syntax

    restored = _restored_python_paths(git_cmd, repo_root)
    if restored is None:
        return "restored Python source discovery failed"
    if restored:
        ok, failing_path, syntax_error = _validate_python_files_syntax(repo_root, restored)
        if not ok:
            detail = str(syntax_error).splitlines()[0] if syntax_error else "syntax error"
            return f"{failing_path or 'restored Python source'}: {detail}"
    try:
        from hermes_cli.update_cmd_validation import _critical_module_import_failures

        failures = _critical_module_import_failures(repo_root, report_runtime_errors=True)
    except Exception:
        return None
    for module, error in failures.items():
        if clean_import_failures.get(module) != error:
            return f"agent import {module}: {error[1]}"
    return None


def _reset_after_failed_apply(git_cmd, repo_root, untracked_before: set) -> None:
    _git_run(git_cmd, ["reset", "--hard", "HEAD"], repo_root)
    current_untracked = _untracked_paths(git_cmd, repo_root) or set()
    added = sorted(current_untracked - set(untracked_before or ()))
    if added:
        _git_run(git_cmd, ["clean", "-fd", "--", *added], repo_root)
    _git_run(git_cmd, ["reset", "--hard", "HEAD"], repo_root)


def _tracked_dirty_paths(git_cmd, repo_root) -> Optional[set]:
    """Tracked paths differing from HEAD (staged, unstaged or unmerged)."""
    result = _git_run(
        git_cmd, ["status", "--porcelain=v1", "-z", "--untracked-files=no"], repo_root
    )
    if result.returncode != 0:
        return None
    dirty = set()
    for entry in (result.stdout or "").split("\0"):
        if not entry:
            continue
        path = entry[3:] if len(entry) > 3 else ""
        if " -> " in path:  # rename/copy: the new path holds the content
            path = path.split(" -> ", 1)[1]
        if path:
            dirty.add(path)
    return dirty


def _snapshot_group_prestate(git_cmd, repo_root) -> Optional[tuple]:
    """Capture the tree before one group's apply: tracked-dirty paths, the
    untracked set, and untracked file contents (to restore files the failed
    group overwrote or deleted). None when git itself errors."""
    dirty = _tracked_dirty_paths(git_cmd, repo_root)
    untracked = _untracked_paths(git_cmd, repo_root)
    if dirty is None or untracked is None:
        return None
    contents = {}
    for path in untracked:
        full = Path(repo_root) / path
        try:
            contents[path] = full.read_bytes() if full.is_file() and not full.is_symlink() else None
        except OSError:
            contents[path] = None
    return (dirty, set(untracked), contents)


def _revert_group_apply(git_cmd, repo_root, prestate) -> None:
    """Surgically revert ONE group's apply, leaving earlier groups intact.

    A whole-tree ``reset --hard`` would also wipe groups that already applied
    cleanly (and their dropped stashes), turning their ``active`` receipt into
    a dangling pointer (#130902). Instead only paths this group touched are
    rolled back: tracked paths it dirtied (reset collapses unmerged stages
    first, so conflicted applies revert too), untracked files it added, and
    pre-existing untracked files it overwrote or deleted.
    """
    pre_dirty, pre_untracked, pre_contents = prestate
    current_dirty = _tracked_dirty_paths(git_cmd, repo_root) or set()
    touched = sorted(current_dirty - pre_dirty)
    if touched:
        _git_run(git_cmd, ["reset", "-q", "HEAD", "--", *touched], repo_root)
        _git_run(git_cmd, ["checkout", "-q", "HEAD", "--", *touched], repo_root)
    current_untracked = _untracked_paths(git_cmd, repo_root) or set()
    added = sorted(current_untracked - pre_untracked)
    if added:
        _git_run(git_cmd, ["clean", "-fd", "--", *added], repo_root)
    for path, content in pre_contents.items():
        if content is None:
            continue
        full = Path(repo_root) / path
        try:
            if not full.is_file() or full.read_bytes() != content:
                full.write_bytes(content)
        except OSError:
            print(f"! Could not restore pre-apply untracked file {path}.")


def _drop_stash(git_cmd, repo_root, stash_sha: str) -> None:
    selector = _stash_ref_for_commit(git_cmd, repo_root, stash_sha)
    if selector is None:
        return
    # Never drop by bare SHA: resolve to the current index first.
    index = selector.replace("stash@{", "").replace("}", "")
    _git_run(git_cmd, ["stash", "drop", index if index.isdigit() else selector], repo_root)


def restore_preserved_groups(
    git_cmd, repo_root, state: PreservationState, *, policy: str, keep_stash: bool = False
) -> dict:
    """Reapply preserved groups under *policy*; conflicting groups stay parked.

    Returns ``{group: {"status": ..., "reason": ...}}``. The base
    installation is never rolled back: a failed group resets the tree to
    the clean updated revision and keeps its stash.
    """
    repo_root = Path(repo_root)
    outcome: dict[str, dict] = {}
    effective = "never" if keep_stash else policy
    committed = state.groups.get(GROUP_COMMITTED)
    if committed is not None and committed.present:
        if _committed_still_active(git_cmd, repo_root, state.pre_sha):
            committed.status, committed.reason = "active", "local commits preserved by the update merge"
        else:
            committed.status = "inactive"
            committed.reason = (
                f"local commits not on {state.target_ref}; recoverable at {committed.ref} "
                f"(git log {state.target_ref}..{committed.ref})"
            )
        outcome[GROUP_COMMITTED] = {"status": committed.status, "reason": committed.reason}
    try:
        from hermes_cli.update_cmd_validation import _critical_module_import_failures

        clean_import_failures = _critical_module_import_failures(
            repo_root, report_runtime_errors=True
        )
    except Exception:
        clean_import_failures = {}
    for name in (GROUP_TRACKED, GROUP_UNTRACKED):
        group = state.groups.get(name)
        if group is None or not group.present:
            outcome[name] = {"status": "absent", "reason": ""}
            continue
        if effective == "never":
            group.status = "inactive"
            group.reason = (
                f"preserved only (--keep-stash); recoverable at {group.ref}"
                if keep_stash
                else f"preserved only (restore policy never; re-apply manually); recoverable at {group.ref}"
            )
            outcome[name] = {"status": group.status, "reason": group.reason}
            continue
        untracked_before = _untracked_paths(git_cmd, repo_root) or set()
        prestate = _snapshot_group_prestate(git_cmd, repo_root)
        ok, detail = _apply_one_stash(git_cmd, repo_root, group.ref)
        if not ok:
            if prestate is None:
                _reset_after_failed_apply(git_cmd, repo_root, untracked_before)
            else:
                _revert_group_apply(git_cmd, repo_root, prestate)
            group.status = "inactive"
            first_line = (detail or "").strip().splitlines()[:1]
            group.reason = (
                f"stash apply conflicted; recoverable at {group.ref}"
                + (f" ({first_line[0][:160]})" if first_line else "")
            )
            outcome[name] = {"status": group.status, "reason": group.reason}
            continue
        failure = _validate_restored_tree(repo_root, git_cmd, clean_import_failures)
        if failure is not None:
            if prestate is None:
                _reset_after_failed_apply(git_cmd, repo_root, untracked_before)
            else:
                _revert_group_apply(git_cmd, repo_root, prestate)
            group.status = "inactive"
            group.reason = f"restored tree failed validation ({failure}); recoverable at {group.ref}"
            outcome[name] = {"status": group.status, "reason": group.reason}
            continue
        group.status = "active"
        group.reason = "reapplied cleanly and passed validation"
        outcome[name] = {"status": group.status, "reason": group.reason}
        _drop_stash(git_cmd, repo_root, group.ref)
    return outcome


def write_preservation_receipt(
    repo_root, state: PreservationState, upstream_revision: str,
    outcome: dict, *, policy: str, keep_stash: bool,
) -> Path:
    """Persist the preservation receipt inside the repo's git dir; return its path."""
    active = sorted(name for name, row in outcome.items() if row.get("status") == "active")
    inactive = sorted(name for name, row in outcome.items() if row.get("status") == "inactive")
    payload = {
        "schema": PRESERVATION_SCHEMA, "id": state.preservation_id,
        "created_at": state.created_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "branch": state.branch, "pre_sha": state.pre_sha,
        "upstream_revision": upstream_revision, "target_ref": state.target_ref,
        "base_ref": state.base_ref, "restore_policy": policy,
        "keep_stash": bool(keep_stash),
        "groups": {
            name: {
                "present": group.present, "ref": group.ref,
                "status": outcome.get(name, {}).get("status", group.status),
                "reason": outcome.get(name, {}).get("reason", group.reason),
            }
            for name, group in state.groups.items()
        },
        "active_groups": active, "inactive_groups": inactive,
    }
    directory = _receipt_dir(Path(repo_root))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"receipt-{state.preservation_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    with suppress(Exception):
        from hermes_cli.update_receipt import record_fact, record_step

        record_fact("local_preservation", payload)
        if inactive:
            record_step(
                "local_preservation", True,
                f"updated with inactive customizations: {', '.join(inactive)} "
                f"(receipt {path.name}; base {state.base_ref})",
            )
        else:
            record_step(
                "local_preservation", True,
                f"preserved {len(active)} active group(s)"
                + (f": {', '.join(active)}" if active else " (clean tree)")
                + f" (receipt {path.name})",
            )
    return path


def print_preservation_summary(receipt_path: Path, payload: Optional[dict] = None) -> None:
    """CLI/Desktop-visible distinction between parked customizations and failure."""
    try:
        data = payload
        if data is None:
            data = json.loads(Path(receipt_path).read_text(encoding="utf-8-sig"))
    except Exception:
        data = None
    if not isinstance(data, dict):
        print(f"  Preservation receipt: {receipt_path}")
        return
    inactive = data.get("inactive_groups") or []
    active = data.get("active_groups") or []
    upstream = (data.get("upstream_revision") or "")[:12]
    if inactive:
        print()
        print("⚠ Updated with inactive customizations — the upstream installation completed,")
        print("  but compatible local changes were NOT reapplied:")
        for name in inactive:
            reason = (data.get("groups") or {}).get(name, {}).get("reason", "")
            ref = (data.get("groups") or {}).get(name, {}).get("ref", "")
            print(f"    • {name}: {reason}")
            if ref:
                print(f"      recover with: git stash apply {ref}" if "refs/" not in ref
                      else f"      recover with: git log {data.get('target_ref')}..{ref}")
        print(f"  Upstream revision: {upstream}")
        print(f"  Base recovery ref: {data.get('base_ref')}")
        print(f"  Receipt: {receipt_path}")
        if active:
            print(f"  Active groups: {', '.join(active)}")
    else:
        if active:
            print(f"  ✓ Local customizations active: {', '.join(active)}")
        print(f"  Preservation receipt: {receipt_path}")


def already_on_target(git_cmd, repo_root, target_sha: str) -> bool:
    """True when HEAD already equals the selected upstream revision."""
    if not target_sha:
        return False
    head = _head_sha(git_cmd, repo_root)
    return bool(head) and head == target_sha
