"""Atomically publish identity for a mutable source checkout."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone, UTC
import json
import os
from pathlib import Path
import tempfile

from hermes_cli.version_info import _git_version_info, _reset_version_info_cache


def write_source_stamp(root: Path, *, adopted: bool = False) -> dict | None:
    """Replace ``install-stamp.json`` with identity read from ``root`` itself.

    *adopted* marks a boot-time adoption (``adoptedAt``): identity only, nothing
    built, so it must never read as a completed install/update of this commit.

    A root git cannot identify -- the ZIP update fallback runs precisely because
    git is unusable -- publishes no identity: the old stamp is removed rather
    than left naming the commit that was just replaced. Returns None then.
    """
    root = Path(root).resolve()
    info = _git_version_info(root, include_untracked=True)
    if info.commit is None:
        with suppress(FileNotFoundError):
            (root / "install-stamp.json").unlink()
        _reset_version_info_cache()
        return None
    stamp = {
        "schemaVersion": 2,
        "commit": info.commit,
        "commitDate": info.commit_date,
        "branch": info.branch,
        "builtAt": datetime.now(UTC).isoformat(),
        "dirty": info.dirty,
        "source": "git",
        "distribution": None,
        "updateMechanism": "self",
        "baseVersion": info.base_version,
        "displayVersion": info.derived_version,
        "distance": info.distance,
        "payload": "bootstrap",
        "tag": None,
    }
    if adopted:
        stamp["adoptedAt"] = stamp["builtAt"]
    stamp_path = root / "install-stamp.json"
    runtime_dir = None
    if stamp_path.is_file():
        try:
            previous_stamp = json.loads(stamp_path.read_text(encoding="utf-8-sig"))
            if isinstance(previous_stamp, dict):
                prev_val = previous_stamp.get("runtimeDir")
                if prev_val and isinstance(prev_val, str) and Path(prev_val).resolve().is_dir():
                    runtime_dir = str(Path(prev_val).resolve())
        except (OSError, ValueError):
            pass

    if runtime_dir is None:
        try:
            from pm.environments import store_root
            from hermes_constants import get_default_hermes_root
            candidate = store_root(root, honor_runtime_override=True)
            default_tools = (get_default_hermes_root() / "tools").resolve()
            if candidate.resolve() != default_tools and candidate.is_dir():
                runtime_dir = str(candidate.resolve())
        except Exception:
            pass

    if runtime_dir:
        stamp["runtimeDir"] = runtime_dir
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
        "completedAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    }
    fd, tmp_name = tempfile.mkstemp(dir=root, prefix=".hermes-bootstrap-complete.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2) + "\n")
        os.replace(tmp_name, receipt_path)
    finally:
        with suppress(OSError):
            os.unlink(tmp_name)