"""Atomically publish identity for a mutable source checkout."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from hermes_cli.version_info import (
    _derived_version,
    _git_version_info,
    _parse_nonnegative,
    _reset_version_info_cache,
    _run_git,
)


def _prior_release_identity(root: Path) -> tuple[str, int | None] | None:
    """The release identity of the stamp this publish replaces, if resolvable.

    Consulted only when the live read cannot name a release: a checkout that
    already had one must not lose it to a single failed git read.
    """
    try:
        data = json.loads((Path(root) / "install-stamp.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    base = data.get("baseVersion")
    if not isinstance(base, str) or not base or base == "unknown":
        return None
    distance = data.get("distance")
    if isinstance(distance, str):
        distance = _parse_nonnegative(distance)
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        distance = None
    return base, distance


def write_source_stamp(root: Path) -> dict | None:
    """Replace ``install-stamp.json`` with identity read from ``root`` itself.

    A root git cannot identify -- the ZIP update fallback runs precisely because
    git is unusable -- publishes no identity: the old stamp is removed rather
    than left naming the commit that was just replaced. Returns None then.

    A root git identifies but whose *release* it cannot name keeps the release
    recorded by the stamp it replaces: every reader prefers the stamp, so one
    failed git read must not permanently downgrade a checkout that already had
    a version.
    """
    root = Path(root).resolve()
    info = _git_version_info(root, include_untracked=True)
    if info.commit is None:
        with suppress(FileNotFoundError):
            (root / "install-stamp.json").unlink()
        _reset_version_info_cache()
        return None
    base_version = info.base_version
    distance = info.distance
    display_version = info.derived_version
    if base_version == "unknown":
        # ``unknown`` reports a read that failed, not a checkout without a
        # release: the resolver's git calls are best-effort (3s timeout,
        # failures swallowed). Keep the last resolvable release and refresh
        # only what this read did answer.
        prior = _prior_release_identity(root)
        if prior is not None:
            base_version, distance = prior
            recomputed = _parse_nonnegative(
                _run_git(root, "rev-list", "--count", f"v{base_version}..HEAD")
            )
            if recomputed is not None:
                distance = recomputed
            display_version = _derived_version(
                base_version, distance, info.dirty, info.commit[:7]
            )
    stamp = {
        "schemaVersion": 2,
        "commit": info.commit,
        "commitDate": info.commit_date,
        "branch": info.branch,
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "dirty": info.dirty,
        "source": "git",
        "distribution": None,
        "updateMechanism": "self",
        "baseVersion": base_version,
        "displayVersion": display_version,
        "distance": distance,
        "payload": "bootstrap",
        "tag": None,
    }
    stamp_path = root / "install-stamp.json"
    fd, tmp_name = tempfile.mkstemp(dir=root, prefix=".install-stamp.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(stamp, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, stamp_path)
        with suppress(OSError):
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        with suppress(OSError):
            os.unlink(tmp_name)
    _reset_version_info_cache()
    # Every path that publishes a new checkout identity (completion handoff, the PM
    # updater's finish, boot-time adoption) moves the installers' receipt with it.
    refresh_bootstrap_receipt(root, stamp)
    return stamp


BOOTSTRAP_RECEIPT = ".hermes-bootstrap-complete"


def refresh_bootstrap_receipt(root: Path, stamp: dict) -> None:
    """Move the installers' bootstrap receipt to the commit ``stamp`` just published.

    install.sh / install.ps1 write the receipt once, at the commit they installed;
    an update that moves the checkout has to move it too, or it keeps naming the
    replaced release. Only an existing receipt is refreshed: its presence is what
    marks a script install, so a manual clone never grows one.
    """
    receipt_path = Path(root) / BOOTSTRAP_RECEIPT
    try:
        # install.ps1 writes it with Windows PowerShell's UTF-8 BOM.
        previous = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    receipt = {
        **previous,
        "schemaVersion": 1,
        "pinnedCommit": stamp["commit"],
        # A detached checkout keeps the branch it was installed from.
        "pinnedBranch": stamp["branch"] or previous.get("pinnedBranch"),
        "completedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    }
    fd, tmp_name = tempfile.mkstemp(dir=root, prefix=".hermes-bootstrap-complete.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2) + "\n")
        os.replace(tmp_name, receipt_path)
    finally:
        with suppress(OSError):
            os.unlink(tmp_name)
