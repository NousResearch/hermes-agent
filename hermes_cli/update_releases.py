"""Opt-in release ledger behind ``hermes update --rollback``.

With ``updates.rollback_releases: N`` (N > 0) every ``hermes update`` that moves the code records
the commit it left as ``refs/worktree/hermes/releases/<seq>-<sha12>``, keeping the newest N. The
entry is written once the move is verified (``update_cmd._apply_pulled_update``, still in the
running updater, so a target without this module records too), so an update that fails or finds
nothing new never rotates the ledger. ``refs/worktree/`` refs are private to the checkout that
wrote them (a linked worktree install keeps its own ledger) yet still reachability roots for
``git gc``, so the commits (and, in a blobless clone, the blobs their checkout already fetched)
stay on disk and a rollback needs no network.

``hermes update --rollback`` selects the entry after the running commit (the newest when the
running commit is not in the ledger) and hands it to the normal update pipeline as a pinned,
already-local release: the same pause, commit point, syntax and downgrade guards, dependency
sync and gateway fleet restart as any other update, landing detached at that commit. The run
records nothing, so repeated rollbacks step further back and a plain ``hermes update`` moves
forward again.

The install root stays one checkout: the venv, launchers, service units and per-install keys
(``pm.environments.install_key`` hashes the resolved root) all name it, so a symlinked
``current`` root would give every release its own environment and channel record.
"""

from __future__ import annotations

import re
from pathlib import Path

RELEASE_PREFIX = "refs/worktree/hermes/releases/"
_RELEASE_REF = re.compile(r"^refs/worktree/hermes/releases/(\d+)-[0-9a-f]+$")


def keep_count(updates_config: dict) -> int:
    """``updates.rollback_releases`` as a non-negative int; anything unreadable is 0 (off)."""
    value = updates_config.get("rollback_releases", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _git(git_cmd, root: Path, args: list[str]):
    from hermes_cli.update_cmd_check import _git as run

    return run(list(git_cmd), root, args)


def list_releases(git_cmd, root: Path) -> list[tuple[int, str, str]]:
    """``(seq, ref, sha)`` for every ledger entry, newest first."""
    result = _git(git_cmd, root, ["for-each-ref", "--format=%(refname) %(objectname)", RELEASE_PREFIX])
    entries = []
    for line in (result.stdout or "").splitlines() if result.returncode == 0 else ():
        ref, _, sha = line.partition(" ")
        match = _RELEASE_REF.match(ref)
        if match and sha:
            entries.append((int(match.group(1)), ref, sha))
    return sorted(entries, reverse=True)


def record_release(git_cmd, root: Path, sha: str, keep: int) -> str | None:
    """Record ``sha`` as the newest entry (dropping an older entry of the same commit), prune to
    ``keep``. Returns the new ref, or None when nothing was written (off, or already newest)."""
    entries = list_releases(git_cmd, root)
    if keep <= 0 or (entries and entries[0][2] == sha):
        return None
    ref = f"{RELEASE_PREFIX}{(entries[0][0] + 1) if entries else 1:06d}-{sha[:12]}"
    if _git(git_cmd, root, ["update-ref", ref, sha]).returncode != 0:
        return None
    older = [old_ref for _seq, old_ref, old_sha in entries if old_sha == sha]
    survivors = [old_ref for _seq, old_ref, old_sha in entries if old_sha != sha]
    for stale in older + survivors[keep - 1:]:
        _git(git_cmd, root, ["update-ref", "-d", stale])
    return ref


def rollback_target(git_cmd, root: Path, head: str | None) -> str | None:
    """The ledger entry after ``head`` (the newest when ``head`` is not recorded), else None."""
    shas = [sha for _seq, _ref, sha in list_releases(git_cmd, root)]
    start = shas.index(head) + 1 if head in shas else 0
    return next((sha for sha in shas[start:] if sha != head), None)


def select(args, git_cmd, root: Path, request: dict) -> tuple | None:
    """``select_apply_target``'s ledger step: the rollback target tuple for ``--rollback``, else
    None after noting on ``request`` the commit a ledger-on update leaves (``record_left``)."""
    from hermes_cli.update_cmd import _capture_head_sha, _record_update_step, _updates_config

    head = _capture_head_sha(list(git_cmd), root)
    if not getattr(args, "rollback", False):
        keep = keep_count(_updates_config())
        if keep and head:
            request["release_ledger"] = {"leaving": head, "keep": keep}
        return None
    target = rollback_target(git_cmd, root, head)
    if target is None:
        raise LookupError(
            "no earlier release is recorded for this install. Set `updates.rollback_releases` "
            "(e.g. 3) in config.yaml; each later `hermes update` then records the commit it leaves.")
    print(f"→ Rolling back: {head[:12] if head else 'unknown'} → {target[:12]} (recorded release)")
    _record_update_step("rollback", True, f"{(head or '')[:12]} -> {target[:12]}")
    request["expected_sha"] = target
    request["local_target"] = True  # already on disk: no fetch, no unshallow, no channel hold
    return target, target, True, None


def record_left(request: dict) -> None:
    """Record the commit this update moved off, once the move landed on ``expected_sha``."""
    ledger = request.get("release_ledger")
    landed = request.get("expected_sha")
    if not ledger or not landed or ledger["leaving"] == landed:
        return
    ref = record_release(["git"], Path(request["source"]), ledger["leaving"], ledger["keep"])
    if ref:
        print(f"  ✓ Kept {ledger['leaving'][:12]} for `hermes update --rollback` ({ref.rsplit('/', 1)[1]})")
