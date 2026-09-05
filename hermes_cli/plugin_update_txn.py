"""Staged plugin-update security transaction (G1 / HookPry) — the single update owner.

``hermes plugins update`` and the Dashboard update route both ride the
transaction in this module: a private quarantine candidate is fetched (the
live checkout is never mutated before authorization), the review is bound to
``(plugin key, old revision, candidate revision, candidate artifact
identity)``, and the candidate is promoted only after a token-bound accept,
under a per-plugin commit lock, through one durable settlement.

Scope (per maintainer review of #103497 — andrexibiza, round 2; staged
transaction shape adapted from #37977 / coygeek):

* **Artifact identity + consent records** — the immutable reviewed artifact is
  the canonical git tree id (``HEAD^{tree}``; bytes + mode + path + type) for
  git checkouts, the noise-excluded whole-tree sha256 for non-git/manual
  trees. Consent records bind the trust anchor to that artifact, not to the
  declared ``name``/``version``.
* **Staging/quarantine** — stage fetches a FRESH checkout of the candidate
  revision into ``~/.hermes/.plugin-updates/`` (chmod 700); it never copies
  the live directory, so mutable local state is not smuggled into the
  artifact. The old revision's git objects are fetched from the live checkout
  (already verified == the recorded consent) purely so the review diff can be
  rendered.
* **Activation/promotion/settlement** — the accept step runs entirely under a
  process-safe per-plugin lock (``fcntl.flock``/``msvcrt`` sibling-file lock):
  re-read + revalidate live and staged identities, run the plugin security
  scan on the exact staged candidate, promote (two ``os.replace`` renames),
  replay the *allowed* mutable untracked state from the live tree into the
  promoted artifact, settle the metadata sidecar + consent record, and only
  then release. One terminal outcome is durable; every earlier failure rolls
  the tree and sidecar back.
* **Mutable-state split** — the artifact is immutable; local mutable state is
  a separate coordinate. ``*.example``-derived files (the
  ``_copy_example_files`` class) are replayed from the live tree at commit
  time with their exact current bytes; a file deleted from untracked state
  after staging is NOT resurrected. Everything else stays artifact-only.
* **Post-accept policy** — the security scan and the capability-delta
  settlement live here, not in caller-specific tails. One surface-neutral
  :class:`PluginUpdatePolicy` (granted / pending / refused) comes out of the
  commit step; CLI and Dashboard only render it. There is no Dashboard path
  that promotes without the scan/consent policy.
* **Update-diff review** — the review gate, stable-version tripwire, and
  static (no-import) registration inventory used by the drift paths.

This module is a leaf sibling of ``hermes_cli.plugins_cmd.py``: it never
imports the facade at module level (the facade imports this module); every
facade helper it needs (metadata sidecar, manifest reader, git runner,
enable/disable set, scan config) is late-imported inside the function that
uses it, so ``monkeypatch.setattr(plugins_cmd, ...)`` seams keep working.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from hermes_cli.plugin_treehash import git_tree_id, tree_sha256
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_CONSENT_SCOPES = ("install", "update", "reinstall")
_ARTIFACT_KIND_GIT_TREE = "git_tree"
_ARTIFACT_KIND_SHA256 = "sha256"

_TD_WARNING = "possible unauthorized update (code changed under a stable version)"

# Surface-neutral policy outcomes of one accept/commit decision. CLI and
# Dashboard render the same value for the same candidate; neither runs a
# divergent policy path.
_OUTCOME_GRANTED = "granted"     # committed; declared capability set fully covered
_OUTCOME_PENDING = "pending"     # committed; capability additions need consent (fail closed until granted)
_OUTCOME_REFUSED = "refused"     # NOT committed (scan blocked / stale / superseded / declined)
_OUTCOME_DISABLED = "disabled"   # NOT committed and the plugin was disabled (fail-closed drift decline)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class PluginUpdatePolicy:
    """One surface-neutral policy decision from the commit step.

    ``outcome`` is granted / pending / refused. ``committed`` says whether the
    candidate replaced the live tree. Scan and capability fields let the
    surfaces render the same decision without re-deriving it.
    """

    outcome: str
    committed: bool
    candidate_revision: Optional[str] = None
    candidate_artifact: Optional[str] = None
    reason: str = ""
    scan_verdict: Optional[str] = None
    scan_blocked: bool = False
    scan_findings: List[dict] = field(default_factory=list)
    pending_capabilities: List[str] = field(default_factory=list)
    capabilities_changed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "committed": self.committed,
            "candidate_revision": self.candidate_revision,
            "candidate_artifact": self.candidate_artifact,
            "reason": self.reason,
            "scan_verdict": self.scan_verdict,
            "scan_blocked": self.scan_blocked,
            "scan_findings": self.scan_findings,
            "pending_capabilities": self.pending_capabilities,
            "capabilities_changed": self.capabilities_changed,
        }


# ── Artifact (content) consent records ──────────────────────────────────────
# ``consent`` binds the plugin's trust anchor to its *artifact identity*, not
# to its declared identity. Written at install/reinstall and at every
# successful update consent, so a baseline always exists. Legacy records
# without a consent key are treated as "no baseline" by the update gate (one
# re-consent).


def _granted_at() -> str:
    """ISO-8601 UTC timestamp for a consent record."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _consent_record(artifact_kind: str, artifact_id: str, revision: Optional[str], *, scope: str) -> dict:
    """One artifact-consent record; *revision* is the git SHA or 'manual'."""
    if scope not in _CONSENT_SCOPES:
        raise ValueError(f"consent scope must be one of {_CONSENT_SCOPES}, got {scope!r}")
    if artifact_kind not in (_ARTIFACT_KIND_GIT_TREE, _ARTIFACT_KIND_SHA256):
        raise ValueError(f"artifact identity kind must be git_tree or sha256, got {artifact_kind!r}")
    return {
        "identity": artifact_kind,
        "artifact_id": artifact_id,
        "revision": revision or "manual",
        "granted_at": _granted_at(),
        "scope": scope,
    }


def _plugin_artifact_identity(target: Path, *, is_git: bool, git_exe: Optional[str]) -> Tuple[str, str]:
    """``(kind, id)`` artifact identity of an installed plugin tree.

    Git checkouts use the canonical git tree id (``HEAD^{tree}``) — tamper
    evident over every tracked entry's bytes + mode + path + type. When git is
    unavailable (or the tree is not a usable checkout) the identity falls back
    to the noise-excluded whole-tree sha256 — the identity for non-git/manual
    trees, where there is no index separating artifact from noise.
    """
    if is_git:
        tree_id = git_tree_id(target, git_exe)
        if tree_id:
            return (_ARTIFACT_KIND_GIT_TREE, tree_id)
    return (_ARTIFACT_KIND_SHA256, tree_sha256(target))


def _consent_artifact_matches(target: Path, consent: dict, *, git_exe: Optional[str]) -> Optional[bool]:
    """True when *target*'s live artifact equals the recorded consent artifact.

    ``None`` when *consent* carries no usable artifact (no baseline). The
    comparison re-derives the identity exactly as the consent write did, so
    the ``live tree == last consented artifact`` invariant is checked with the
    same function that recorded it.
    """
    artifact_id = consent.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        return None
    if consent.get("identity") == _ARTIFACT_KIND_GIT_TREE:
        live_id = git_tree_id(target, git_exe)
        return live_id is not None and live_id == artifact_id.lower()
    if consent.get("identity") == _ARTIFACT_KIND_SHA256:
        return tree_sha256(target) == artifact_id
    return None


def _record_accepted_tree_consent(plugin_key: str, artifact_kind: str, artifact_id: str, revision: str, *, scope: str) -> None:
    """Persist the accepted artifact + revision for *plugin_key* (read-modify-write).

    Called only after a reviewed update was promoted (or a drifted live tree
    was reviewed in place), so the consent baseline always describes the tree
    the operator just authorized.
    """
    from hermes_cli.plugins_cmd import _read_install_metadata, _write_install_metadata
    metadata = _read_install_metadata()
    record = metadata.setdefault(plugin_key, {})
    record["revision"] = revision
    record["consent"] = _consent_record(artifact_kind, artifact_id, revision, scope=scope)
    _write_install_metadata(metadata)


def _consent_from_record(record: dict) -> dict:
    """The ``consent`` sub-record of an install-metadata record, or {}."""
    consent = record.get("consent")
    return consent if isinstance(consent, dict) else {}


class PluginConsentDrift(Exception):
    """The live tree no longer matches the recorded consent artifact.

    Raised by the stage step before anything is fetched: the update cannot be
    reviewed against a baseline the live tree does not actually sit on. The
    caller reviews the drifted live state itself (re-consent or fail closed).
    ``dirty`` distinguishes uncommitted tracked edits (the operator's own local
    changes — abort without disabling so they are never destroyed) from a clean
    content/revision drift (out-of-band commit or crashed promote — review).

    Deliberately NOT a ``plugins_cmd.PluginOperationError`` subclass: this
    module is a leaf sibling of the facade (no module-level facade import, so
    no cycle). The facade entry points catch it by type BEFORE their generic
    ``PluginOperationError`` handlers, which is the only place it can surface.
    """

    def __init__(self, message: str, *, consent: dict, old_record: dict, git_exe: Optional[str], dirty: bool = False):
        super().__init__(message)
        self.consent = consent
        self.old_record = old_record
        self.git_exe = git_exe
        self.dirty = dirty


# ── Per-plugin commit lock ───────────────────────────────────────────────────
# The plugin key is the mutation authority: revalidate → promote (two renames)
# → settlement → release all happen under one process-safe lock, so two
# accepts of different candidates staged from the same revision cannot both
# pass the old-generation gate and race the tree/metadata commit. Shape
# mirrors plugins_state._locked_plugin_state (sibling .lock file because
# atomic replacement changes the target's inode; fcntl.flock on POSIX, msvcrt
# on Windows per the repo support matrix). The lock file lives in the chmod
# 700 quarantine root; flock dies with its process, so a wedged holder cannot
# leak the lock.

_LOCK_GUARD = threading.Lock()
_IN_PROCESS_LOCKS: Dict[str, threading.RLock] = {}


@contextlib.contextmanager
def _per_plugin_update_lock(plugin_key: str, root: Path) -> Iterator[None]:
    """Serialize one plugin's commit transaction across threads and processes."""
    safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", plugin_key)
    lock_path = root / f".{safe_key}.lock"
    with _LOCK_GUARD:
        thread_lock = _IN_PROCESS_LOCKS.setdefault(
            str(lock_path.resolve(strict=False)), threading.RLock())
    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as handle:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt
                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ── Quarantine / staging ─────────────────────────────────────────────────────


def _plugin_update_root() -> Path:
    """Private quarantine root for staged plugin updates (never the live tree).

    Lives under HERMES_HOME so it is profile-local and on the same filesystem
    as the plugin dir (atomic ``os.replace`` promotion). chmod 700: staged
    candidates can hold an unreviewed upstream tree.
    """
    root = get_hermes_home() / ".plugin-updates"
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root


def _live_remote_url(target: Path, git_exe: str) -> str:
    """Scrubbed ``origin`` URL of the installed checkout (sanitized at install)."""
    from hermes_cli.plugins_cmd import PluginOperationError, _run_plugin_git
    result = _run_plugin_git(git_exe, target, "remote", "get-url", "origin", timeout=15)
    url = result.stdout.strip() if result.returncode == 0 else ""
    if not url:
        raise PluginOperationError(
            f"Plugin '{target.name}' has no 'origin' remote; cannot stage an update.")
    return url


def _live_tracked_branch(target: Path, git_exe: str) -> Optional[str]:
    """Current branch short name of the live checkout, or None when detached."""
    from hermes_cli.plugins_cmd import _run_plugin_git
    result = _run_plugin_git(git_exe, target, "symbolic-ref", "--short", "-q", "HEAD", timeout=15)
    branch = result.stdout.strip() if result.returncode == 0 else ""
    return branch or None


def _remote_default_branch(candidate: Path, git_exe: str) -> str:
    """Default branch of the candidate's origin (``refs/heads/<name>`` of HEAD)."""
    from hermes_cli.plugins_cmd import PluginOperationError, _run_plugin_git
    result = _run_plugin_git(git_exe, candidate, "ls-remote", "--symref", "origin", "HEAD", timeout=30)
    if result.returncode != 0:
        raise PluginOperationError("Could not determine the plugin remote's default branch.")
    for line in result.stdout.splitlines():
        if "ref:" in line and "\tHEAD" in line:
            ref = line.split("ref:", 1)[1].split("\t", 1)[0].strip()
            return ref.removeprefix("refs/heads/")
    raise PluginOperationError("Could not determine the plugin remote's default branch.")


def _checkout_remote_candidate(
    live_target: Path, git_exe: str, old_revision: str, candidate_dir: Path,
) -> str:
    """Fresh checkout of the remote candidate revision into empty *candidate_dir*.

    Returns the candidate revision SHA. The candidate starts empty
    (``git init`` + fetch); the live directory is only ever READ (origin URL,
    current branch). Fetching the tracked branch WITHOUT ``--depth`` keeps the
    full branch history in the candidate so the review-diff helpers can render
    ``old_revision..HEAD``; when the live checkout is detached the remote's
    default branch is resolved via ``ls-remote --symref``.

    Because the live tree was verified to equal its recorded consent before
    this runs, its own git object store is a trustworthy source of the old
    revision's commit/tree objects for review display — a depth-1 fetch of the
    live checkout's advertised HEAD ref (== old_revision at stage time) brings
    them in without needing unadvertised-object permissions.
    """
    from hermes_cli.plugins_cmd import PluginOperationError, _run_plugin_git
    remote_url = _live_remote_url(live_target, git_exe)
    branch = _live_tracked_branch(live_target, git_exe)

    init = _run_plugin_git(git_exe, candidate_dir, "init", "-q", timeout=15)
    if init.returncode != 0:
        raise PluginOperationError("Could not initialize the staged update candidate.")
    add_remote = _run_plugin_git(
        git_exe, candidate_dir, "remote", "add", "origin", remote_url, timeout=15)
    if add_remote.returncode != 0:
        raise PluginOperationError("Could not attach the staged update candidate to its origin.")

    if branch is None:
        branch = _remote_default_branch(candidate_dir, git_exe)

    fetch = _run_plugin_git(git_exe, candidate_dir, "fetch", "origin", branch, timeout=120)
    if fetch.returncode != 0:
        raise PluginOperationError(
            "Could not fetch the plugin update from the remote:\n"
            + ((fetch.stderr or fetch.stdout or "").strip() or "git fetch failed."))
    resolve = _run_plugin_git(git_exe, candidate_dir, "rev-parse", "FETCH_HEAD", timeout=15)
    new_revision = resolve.stdout.strip().lower() if resolve.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{40}", new_revision):
        raise PluginOperationError(
            f"Could not determine the candidate revision for plugin '{live_target.name}' "
            "after fetching.")

    # Bring the OLD revision's objects in from the (consent-verified) live
    # checkout so the review can diff old..HEAD and show the old manifest.
    old_fetch = _run_plugin_git(
        git_exe, candidate_dir, "fetch", "--depth", "1", str(live_target.resolve()), "HEAD", timeout=60)
    if old_fetch.returncode != 0:
        raise PluginOperationError(
            "Could not read the current revision's git objects from the installed checkout:\n"
            + ((old_fetch.stderr or old_fetch.stdout or "").strip() or "git fetch failed."))

    checkout = _run_plugin_git(git_exe, candidate_dir, "checkout", "-q", "--detach", new_revision, timeout=60)
    if checkout.returncode != 0:
        raise PluginOperationError(
            "Could not check out the staged update candidate:\n"
            + ((checkout.stderr or checkout.stdout or "").strip() or "git checkout failed."))
    verify = _run_plugin_git(git_exe, candidate_dir, "rev-parse", "HEAD", timeout=15)
    head = verify.stdout.strip().lower() if verify.returncode == 0 else ""
    if head != new_revision:
        raise PluginOperationError(
            f"Staged candidate HEAD '{head}' does not match the fetched revision '{new_revision}'.")
    return new_revision


def _stage_plugin_update(name: str, target: Path) -> dict:
    """Fetch an update into a private quarantine tree; bind review to the candidate.

    The live tree is never mutated and never copied: the candidate is a FRESH
    checkout of the remote at its current branch head, so mutable local state
    (untracked files, ``*.example``-derived configs) cannot ride into the
    artifact snapshot. Returns an ``unchanged`` payload when the remote is
    current (only after the live tree was verified to equal the recorded
    consent — the no-op path cannot launder), or a ``review_required`` payload
    carrying the review token bound to ``(key, old revision, candidate
    revision, candidate artifact identity)`` plus the diff/registration review
    content. Raises :class:`PluginOperationError` on refusals and
    :class:`PluginConsentDrift` when the live tree drifted from its consent.
    """
    from hermes_cli.plugins_cmd import (
        PluginOperationError, _check_manifest_version, _git_head_revision,
        _read_install_metadata, _read_manifest, _resolve_git_executable)
    from rich.markup import escape

    git_exe = _resolve_git_executable()
    if not git_exe:
        raise PluginOperationError("git is not installed or not in PATH.")
    if not (target / ".git").is_dir():
        raise PluginOperationError(
            f"Plugin '{name}' was not installed from git (no .git directory). Cannot update.")

    metadata = _read_install_metadata()
    old_record = dict(metadata.get(name) or {})
    if old_record.get("pinned") is True:
        raise PluginOperationError(
            f"Plugin '{name}' is pinned to {old_record.get('revision')}. To move it, run "
            f"`hermes plugins install {escape(str(old_record.get('source', '<source>')))} "
            "--force --ref <40-character commit SHA>`.")

    consent = _consent_from_record(old_record)
    consent_artifact = consent.get("artifact_id")
    if isinstance(consent_artifact, str) and consent_artifact:
        # No-op / stale-remote safety: refuse to stage from a live tree that is
        # not the consented artifact (out-of-band commit/edit, crashed prior
        # promote). The caller reviews that drifted state instead.
        if _consent_artifact_matches(target, consent, git_exe=git_exe) is not True:
            raise PluginConsentDrift(
                f"Plugin '{name}' live tree no longer matches its recorded consent "
                f"(consented {consent_artifact[:12]}…). Review the drifted state before "
                "updating again — run the update in an interactive session or pass "
                "--accept-update after reviewing the diff below.",
                consent=consent, old_record=old_record, git_exe=git_exe)
        if _plugin_checkout_tracked_dirty(target, git_exe):
            raise PluginConsentDrift(
                f"Plugin '{name}' has uncommitted tracked changes; the live executable "
                "tree differs from the consented artifact. Commit or stash the local "
                "changes, or accept re-consenting the current live tree with "
                "`hermes plugins update <name>` in an interactive session.",
                consent=consent, old_record=old_record, git_exe=git_exe, dirty=True)
    elif _plugin_checkout_tracked_dirty(target, git_exe):
        raise PluginConsentDrift(
            f"Plugin '{name}' has uncommitted tracked changes and no consent baseline; "
            "review the tree before updating it.",
            consent=consent, old_record=old_record, git_exe=git_exe, dirty=True)

    old_revision = str(old_record.get("revision") or "").strip().lower() or None
    if old_revision is None:
        old_revision = _git_head_revision(target, git_exe)  # legacy record without revision
    old_manifest = _read_manifest(target)
    # The live artifact the review is based on. With a consent baseline the
    # drift checks above already proved live == consent; legacy records (no
    # baseline) are anchored to the live tree as it is. Recorded so the commit
    # step can revalidate that the live tree did not change between review and
    # promote.
    live_kind, live_artifact = _plugin_artifact_identity(target, is_git=True, git_exe=git_exe)
    # Mutable-state snapshot: untracked files present at stage time. The
    # commit step replays the *allowed* ones from the live tree and never
    # resurrects files that disappear from this set after staging.
    live_untracked = _untracked_relpaths(target, git_exe) or []

    root = _plugin_update_root()
    work_dir = Path(tempfile.mkdtemp(prefix="stage-", dir=str(root)))
    candidate = work_dir / "candidate"
    try:
        candidate.mkdir()
        new_revision = _checkout_remote_candidate(target, git_exe, old_revision, candidate)
        if new_revision == old_revision:
            shutil.rmtree(work_dir, ignore_errors=True)
            return {
                "ok": True,
                "name": name,
                "output": "Already up to date.",
                "unchanged": True,
                "review_required": False,
            }

        new_manifest = _read_manifest(candidate)
        declared_name = str(new_manifest.get("name") or name)
        if declared_name != str(old_manifest.get("name") or name):
            shutil.rmtree(work_dir, ignore_errors=True)
            raise PluginOperationError(
                f"Plugin manifest name changed from '{old_manifest.get('name') or name}' "
                f"to '{declared_name}'; refusing the staged update.")
        _check_manifest_version(new_manifest, declared_name)

        candidate_kind, candidate_artifact = _plugin_artifact_identity(
            candidate, is_git=True, git_exe=git_exe)
        changed_lines = _update_changed_file_lines(candidate, git_exe, old_revision)
        version_unchanged = ((old_manifest.get("version") or None)
                             == (new_manifest.get("version") or None))
        td_signature = version_unchanged and _update_touched_code(candidate, git_exe, old_revision)
        reg_lines = _update_registration_review_lines(
            candidate, git_exe, old_revision, old_manifest, new_manifest)

        # Review token bound to (plugin key, old revision, candidate revision,
        # candidate artifact identity). ``name`` is the metadata/directory key
        # (== target.name for CLI; dashboard passes the same discovered key).
        token = hashlib.sha256(
            f"{name}\0{old_revision}\0{new_revision}\0{candidate_artifact}".encode("utf-8")
        ).hexdigest()
        staged_path = root / token
        metadata_path = root / f"{token}.json"
        if staged_path.exists():
            shutil.rmtree(staged_path, ignore_errors=True)
        candidate.rename(staged_path)
        shutil.rmtree(work_dir, ignore_errors=True)
        stage_meta = {
            "name": name,
            "target": str(target.resolve()),
            "old_revision": old_revision,
            "old_revision_artifact": live_artifact,
            "old_artifact_kind": live_kind,
            "candidate_revision": new_revision,
            "candidate_artifact": candidate_artifact,
            "candidate_artifact_kind": candidate_kind,
            "old_manifest": old_manifest,
            "new_manifest": new_manifest,
            "changed_files": changed_lines,
            "td_signature": td_signature,
            "review_lines": reg_lines,
            "live_untracked": live_untracked,
        }
        metadata_path.write_text(json.dumps(stage_meta, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "ok": True,
            "name": name,
            "output": "Fetched update.",
            "unchanged": False,
            "review_required": True,
            "review_token": token,
            "old_revision": old_revision,
            "candidate_revision": new_revision,
            "candidate_artifact": candidate_artifact,
            "candidate_artifact_kind": candidate_kind,
            "old_artifact_id": live_artifact,
            "old_artifact_kind": live_kind,
            "changed_files": changed_lines,
            "td_signature": td_signature,
            "review_lines": reg_lines,
            "old_manifest": old_manifest,
            "new_manifest": new_manifest,
        }
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        raise


def _read_stage_metadata(token: str) -> dict:
    """Metadata sidecar of a staged update; raises when the stage is gone."""
    from hermes_cli.plugins_cmd import PluginOperationError
    root = _plugin_update_root()
    staged_path = root / token
    metadata_path = root / f"{token}.json"
    if not staged_path.is_dir() or not metadata_path.is_file():
        raise PluginOperationError(
            "The staged plugin update no longer exists (it may have been superseded "
            "by a newer stage); review the update again.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise PluginOperationError("The staged plugin update record is corrupt.")
    return metadata


def _discard_staged_update(token: str) -> None:
    """Remove a staged candidate + its metadata (decline / superseded). Never raises."""
    root = _plugin_update_root()
    shutil.rmtree(root / token, ignore_errors=True)
    (root / f"{token}.json").unlink(missing_ok=True)


# ── Git helpers for the update-diff review ───────────────────────────────────


def _plugin_checkout_tracked_dirty(target: Path, git_exe: Optional[str]) -> bool:
    """True when tracked files differ from HEAD (porcelain lines other than ``??``)."""
    from hermes_cli.plugins_cmd import _run_plugin_git
    if not git_exe:
        return True  # cannot tell → treat as dirty (decline disables instead of resetting)
    status = _run_plugin_git(git_exe, target, "status", "--porcelain", timeout=15)
    if status.returncode != 0:
        return True  # cannot tell → treat as dirty (decline disables instead of resetting)
    return any(line and not line.startswith("??") for line in status.stdout.splitlines())


def _git_capture(target: Path, git_exe: Optional[str], *args: str) -> Optional[str]:
    """stdout of a best-effort git read command inside a plugin checkout, else None."""
    from hermes_cli.plugins_cmd import _run_plugin_git
    if not git_exe:
        return None
    try:
        result = _run_plugin_git(git_exe, target, *args, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _update_changed_file_lines(target: Path, git_exe: Optional[str], old_rev: Optional[str]) -> Optional[List[str]]:
    """``git diff --name-status old..HEAD`` lines, or None when unavailable."""
    if not old_rev:
        return None
    out = _git_capture(target, git_exe, "diff", "--name-status", f"{old_rev}..HEAD")
    if out is None:
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines or None


def _update_diff_stat(target: Path, git_exe: Optional[str], old_rev: Optional[str]) -> Optional[List[str]]:
    """``git diff --stat old..HEAD`` lines, or None when unavailable."""
    if not old_rev:
        return None
    out = _git_capture(target, git_exe, "diff", "--stat", f"{old_rev}..HEAD")
    if out is None:
        return None
    lines = [line for line in out.splitlines() if line.strip()]
    return lines or None


def _update_log_lines(target: Path, git_exe: Optional[str], old_rev: Optional[str]) -> Optional[List[str]]:
    """``git log --oneline old..HEAD`` lines, or None when unavailable."""
    if not old_rev:
        return None
    out = _git_capture(target, git_exe, "log", "--oneline", "-n", "20", f"{old_rev}..HEAD")
    if out is None:
        return None
    lines = [line for line in out.splitlines() if line.strip()]
    return lines or None


def _update_blob_at(target: Path, git_exe: Optional[str], rev: Optional[str], relpath: str) -> Optional[str]:
    """Text of one tracked file at *rev* (``git show rev:path``), else None."""
    if not rev:
        return None
    return _git_capture(target, git_exe, "show", f"{rev}:{relpath}")


def _update_touched_code(target: Path, git_exe: Optional[str], old_rev: Optional[str]) -> bool:
    """True when the pulled diff touches a code file (conservative when unknown)."""
    from tools.plugin_guard import CODE_FILE_EXTENSIONS

    changed = _update_changed_file_lines(target, git_exe, old_rev)
    if changed is None:
        return True  # no previous revision / git failure → assume code changed
    for line in changed:
        parts = line.split(None, 1)
        if len(parts) == 2 and Path(parts[1]).suffix.lower() in CODE_FILE_EXTENSIONS:
            return True
    return False


def _update_registration_review_lines(
    target: Path, git_exe: Optional[str], old_rev: Optional[str],
    old_manifest: dict, new_manifest: dict,
) -> List[str]:
    """Review lines for hook/tool/command registration changes in the pull.

    Combines the manifest ``provides_hooks`` / ``hooks:`` declaration diff with a
    static AST scan (no import) of every changed ``.py`` file between *old_rev*
    and HEAD. Best-effort: an unavailable previous revision yields only the
    declarations diff, never an error.
    """
    from hermes_cli.plugin_treehash import diff_manifest_hook_declarations, diff_registration_inventories
    lines: List[str] = []
    decl_lines = diff_manifest_hook_declarations(old_manifest or {}, new_manifest or {})
    lines.extend(decl_lines)
    changed = _update_changed_file_lines(target, git_exe, old_rev)
    if not changed:
        return lines
    old_sources: Dict[str, str] = {}
    new_sources: Dict[str, str] = {}
    for line in changed:
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[1].endswith(".py"):
            continue
        rel = parts[1]
        old_text = _update_blob_at(target, git_exe, old_rev, rel)
        new_text = _update_blob_at(target, git_exe, "HEAD", rel)
        if old_text is not None:
            old_sources[rel] = old_text
        if new_text is not None:
            new_sources[rel] = new_text
    lines.extend(diff_registration_inventories(old_sources, new_sources))
    return lines


def _run_plugin_update_diff_gate(
    console,
    target: Path,
    name: str,
    old_revision: Optional[str],
    consent_tree_hash: Optional[str],
    post_tree_hash: str,
    old_manifest: dict,
    new_manifest: dict,
    *,
    td_signature: bool,
    accept_update: bool,
    accept_caution: bool = False,
) -> Tuple[bool, bool]:
    """Review-and-accept gate for a staged-but-unconsented plugin candidate.

    Prints the update diff (changed files, ``--stat``, commits, hook
    registrations) and the stable-version tripwire, then requires explicit
    consent: TTY ``y``, or ``--accept-update`` anywhere. Returns
    ``(accepted, caution_accepted)``: *accepted* True only when the caller may
    adopt the new tree and record the consent baseline; *caution_accepted*
    True only when the scan returned a caution verdict AND an explicit
    keep-enabled decision resolved it (``--accept-caution`` or a TTY ``y``),
    which the commit step re-checks under its lock.
    """
    from hermes_cli.plugins_cmd import _ask_yes, _is_tty, _resolve_git_executable, _scan_on_install_enabled
    from rich.markup import escape

    console.print()
    if consent_tree_hash:
        console.print(
            f"[yellow]Plugin '{escape(name)}' content changed since the last consent "
            f"(tree {consent_tree_hash[:12]}… → {post_tree_hash[:12]}…). "
            f"Review the update before accepting:[/yellow]")
    else:
        console.print(
            f"[yellow]Plugin '{escape(name)}' content changed and has no recorded consent "
            f"baseline (installed before content consent existed). "
            f"Review the update before accepting:[/yellow]")

    if td_signature:
        console.print()
        console.print(f"[red bold]⚠ {_TD_WARNING}[/red bold]")
        console.print(
            f"[red]The plugin's code changed but its declared version stayed "
            f"{old_manifest.get('version')!r}. A code change under a stable version "
            f"is the exact signature of a trojanized update. Decline unless you "
            f"expected this update from a source you trust.[/red]")

    git_exe = _resolve_git_executable()
    changed_lines = _update_changed_file_lines(target, git_exe, old_revision)
    stat_lines = _update_diff_stat(target, git_exe, old_revision)
    log_lines = _update_log_lines(target, git_exe, old_revision)
    if stat_lines:
        console.print("\n  [bold]Diff stat:[/bold]")
        for line in stat_lines:
            console.print(f"    {line}")
    if changed_lines:
        console.print("\n  [bold]Changed files:[/bold]")
        for line in changed_lines:
            console.print(f"    {line}")
    if log_lines:
        console.print("\n  [bold]Commits:[/bold]")
        for line in log_lines:
            console.print(f"    {line}")
    if old_revision is None and changed_lines is None:
        console.print("\n  [dim](git history is unavailable for this checkout — "
                      "review the plugin tree above)[/dim]")

    reg_lines = _update_registration_review_lines(
        target, git_exe, old_revision, old_manifest, new_manifest)
    if reg_lines:
        console.print("\n  [bold]Hook / tool / command registrations changed by the update:[/bold]")
        for line in reg_lines:
            console.print(f"    {line}")
    elif old_revision:
        console.print("\n  [dim]No added or removed hook / tool / command "
                      "registrations detected in the changed files.[/dim]")

    console.print()
    if accept_update:
        console.print("[green]✓[/green] --accept-update given; adopting the reviewed update.")
        content_accepted = True
    elif not _is_tty():
        console.print(
            "[red]Non-interactive session: update NOT accepted (fail closed).[/red] "
            "Review the diff above, then re-run with "
            "`hermes plugins update <name> --accept-update` to adopt it.")
        return False, False
    else:
        content_accepted = _ask_yes(
            "\n  Accept this update and record consent for the new content? [y/N] ", console.input)
    if not content_accepted:
        return False, False

    # ── Post-review security scan presentation (HookPry G4-2) ─────────────────
    # The commit step re-scans the exact staged candidate under the per-plugin lock
    # and refuses a dangerous verdict before promotion. This scan exists so the
    # operator's acceptance is INFORMED by the verdict — the report is shown BEFORE
    # the keep decision, mirroring the install path's scan_decision_cb. A caution
    # verdict on content being adopted requires an explicit keep-enabled decision:
    # `--accept-caution`, or a TTY `y` at the prompt below. Non-TTY without the flag
    # fails closed (the update is declined and nothing is adopted — the live tree
    # stays at the last consented revision, so no disable is needed; unlike the old
    # pull-then-scan shape, the caution tree never reaches the live namespace).
    if _scan_on_install_enabled():
        from tools.plugin_guard import format_scan_report, scan_plugin
        scan_result = scan_plugin(target, source=name)
        if scan_result.verdict == "dangerous":
            console.print()
            console.print(
                f"[yellow]⚠ Security scan flagged the updated plugin:[/yellow] "
                f"{scan_result.summary}")
            console.print(
                "[red]The security scan returned a dangerous verdict: the commit step "
                "refuses such candidates, so this update cannot be adopted.[/red]")
            # Content stays accepted; the commit step refuses the candidate under its
            # lock and renders the refusal + findings — the SAME surface-neutral
            # outcome the Dashboard returns (parity test pins "Update refused").
            return True, False
        if scan_result.verdict == "caution":
            console.print()
            console.print(
                f"[yellow]⚠ Security scan flagged the updated plugin:[/yellow] "
                f"{scan_result.summary}")
            console.print(format_scan_report(scan_result))
            if accept_caution:
                console.print(
                    "[green]✓[/green] --accept-caution given; keeping the plugin "
                    "enabled despite the caution findings.")
                return True, True
            if not _is_tty():
                console.print(
                    "[red bold]✗ Update NOT adopted (fail closed).[/red bold] "
                    "The updated tree carries a caution verdict, which requires an "
                    "explicit keep-enabled decision. Re-run with "
                    f"`hermes plugins update {name} --accept-caution` after reviewing "
                    "the findings above — nothing was changed on disk.")
                return False, False
            keep = _ask_yes(
                "  Keep the plugin enabled despite the caution findings? "
                "Only continue if you trust the source. [y/N] ", console.input)
            if keep:
                console.print(
                    "[green]✓[/green] Caution verdict accepted by user; plugin stays enabled.")
                return True, True
            console.print(
                "[yellow]✗ Update declined: the caution findings were not accepted.[/yellow] "
                "The plugin stays on the previously consented revision — "
                "no changes were adopted.")
            return False, False
    return True, False


def _parse_yaml_text(text: str) -> dict:
    """``yaml.safe_load`` of manifest text; {} when empty."""
    import yaml
    return yaml.safe_load(text) or {}


# ── Drifted-live review / fail-closed decline ────────────────────────────────


def _review_live_drift(
    console, target: Path, key: str, name: str, drift: PluginConsentDrift, *,
    accept_update: bool, accept_caution: bool = False,
) -> None:
    """Review a live tree that drifted from its recorded consent (out-of-band edit/commit).

    The staged transaction cannot review an update against a baseline the live
    tree no longer sits on, so the drifted live state itself is reviewed: the
    diff since the consented revision is shown and an explicit accept re-consents
    the tree as it is. Decline restores the consented revision when the checkout
    is clean; otherwise the plugin is disabled (fail closed) — the unreviewed
    live state is never left silently active under a stale consent. A caution
    verdict on the drifted tree follows the same owner-side rule as updates
    (G4-2): re-consenting requires an explicit keep decision.
    """
    from hermes_cli.plugins_cmd import (
        _git_head_revision, _native_manifest_file, _read_manifest, _resolve_git_executable)
    git_exe = drift.git_exe
    consent = drift.consent
    old_revision = str(consent.get("revision") or "").strip().lower() or None
    old_record = drift.old_record
    pre_drift_dirty = _plugin_checkout_tracked_dirty(target, git_exe) if git_exe else True
    live_manifest = _read_manifest(target)
    live_kind, live_artifact = _plugin_artifact_identity(target, is_git=True, git_exe=git_exe)

    # Best-effort tripwire against the manifest at the consented revision.
    old_manifest: dict = {}
    version_unchanged = False
    if git_exe and old_revision:
        rel = _native_manifest_file(target)
        rel = rel.relative_to(target).as_posix() if rel else ("plugin.json" if (target / "plugin.json").exists() else None)
        old_text = _update_blob_at(target, git_exe, old_revision, rel) if rel else None
        if old_text is not None:
            try:
                old_manifest = _parse_yaml_text(old_text)
            except Exception:
                old_manifest = {}
            version_unchanged = ((old_manifest.get("version") or None)
                                 == (live_manifest.get("version") or None))

    td_signature = version_unchanged and _update_touched_code(target, git_exe, old_revision)
    accepted, _caution_accepted = _run_plugin_update_diff_gate(
        console, target, name, old_revision,
        str(consent.get("artifact_id") or "") or None,
        live_artifact, old_manifest or {}, live_manifest,
        td_signature=td_signature, accept_update=accept_update,
        accept_caution=accept_caution)
    if not accepted:
        _decline_drifted_live(
            console, target, key, old_record, old_revision, pre_drift_dirty,
            consent, git_exe)
        return
    live_revision = _git_head_revision(target, git_exe) if git_exe else None
    _record_accepted_tree_consent(key, live_kind, live_artifact, live_revision or "", scope="update")
    console.print(
        f"[green]✓[/green] Reviewed the drifted tree of [bold]{key}[/bold] and recorded "
        f"consent for revision {live_revision or 'HEAD'} — no update was fetched.")


def _decline_drifted_live(
    console,
    target: Path,
    key: str,
    old_record: dict,
    old_revision: Optional[str],
    pre_drift_dirty_tracked: bool,
    consent: dict,
    git_exe: Optional[str],
) -> None:
    """Fail closed after a declined drift review: restore or disable the live tree."""
    from hermes_cli.plugins_cmd import PluginOperationError, _git_or_raise, _set_plugin_enabled
    restored = False
    if git_exe and old_revision and not pre_drift_dirty_tracked:
        try:
            _git_or_raise(
                git_exe, target, "reset", "--hard", old_revision,
                failure_prefix=(
                    f"Could not restore '{target.name}' to the previously "
                    f"consented revision {old_revision}:\n"))
            restored = True
        except PluginOperationError:
            restored = False
    if restored:
        restored = _consent_artifact_matches(target, consent, git_exe=git_exe) is True
    if restored:
        console.print(
            f"[yellow]✗ Update declined.[/yellow] Plugin [bold]{key}[/bold] was restored to "
            f"the previously consented revision {old_revision} — no changes were adopted.")
        return
    _set_plugin_enabled(key, enable=False)
    console.print(
        f"[red bold]✗ Update declined and '{key}' could not be restored to its consented "
        f"tree, so it has been DISABLED (fail closed).[/red bold] "
        f"[red]Inspect {target} before re-enabling: it holds unreviewed content. "
        f"Re-run `hermes plugins enable {key}` only after restoring the tree you trust.[/red]")


def _decline_staged_update(console, key: str, name: str, token: str, old_revision: Optional[str]) -> None:
    """Drop a declined staged candidate; the live tree was never touched."""
    _discard_staged_update(token)
    console.print(
        f"[yellow]✗ Update declined.[/yellow] Plugin [bold]{name}[/bold] stays on the "
        f"previously consented revision {old_revision or 'HEAD'} — no changes were adopted.")


# ── Mutable local state (the coordinate split) ───────────────────────────────


def _untracked_relpaths(target: Path, git_exe: Optional[str]) -> Optional[List[str]]:
    """Untracked (and not git-ignored) relpaths of a checkout, or None on failure.

    These are the plugin's *mutable local state* — the files a running plugin
    or the operator creates next to the immutable artifact. The commit step
    replays only the subset the artifact model permits.
    """
    from hermes_cli.plugins_cmd import _run_plugin_git
    if not git_exe:
        return None
    try:
        result = _run_plugin_git(
            git_exe, target, "ls-files", "--others", "--exclude-standard", "-z", timeout=20)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    return [p for p in result.stdout.split("\0") if p]


def _tracked_relpath_set(target: Path, git_exe: Optional[str]) -> Set[str]:
    """Set of tracked relpaths of a checkout (empty when git is unavailable)."""
    from hermes_cli.plugins_cmd import _run_plugin_git
    if not git_exe:
        return set()
    try:
        result = _run_plugin_git(git_exe, target, "ls-files", "-z", timeout=20)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()
    if result.returncode != 0:
        return set()
    return {p for p in result.stdout.split("\0") if p}


def _safe_relpath(rel: str) -> bool:
    """True when *rel* is a relative, non-escaping path (defense against hostile entries)."""
    path = PurePosixPath(rel)
    return not path.is_absolute() and ".." not in path.parts and rel != ""


def _copy_untracked_entry(source_root: Path, dest_root: Path, rel: str) -> None:
    """Copy one untracked file/symlink from *source_root* to *dest_root*, preserving symlinks."""
    src = source_root / rel
    dst = dest_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
    elif src.is_file():
        shutil.copy2(src, dst)


def _replay_allowed_untracked(source_root: Path, promoted: Path, git_exe: Optional[str], untracked_now: List[str]) -> List[str]:
    """Replay the artifact-model-permitted mutable state into the promoted tree.

    A live untracked file is *allowed* when the promoted artifact still models
    it: the candidate tracks ``<rel>.example`` (the ``_copy_example_files``
    class) and does not itself track ``<rel>``. Files the artifact does not
    model stay artifact-only (never replayed). Returns the replayed relpaths.
    """
    tracked = _tracked_relpath_set(promoted, git_exe)
    replayed: List[str] = []
    for rel in untracked_now:
        if not _safe_relpath(rel):
            logger.warning("Skipping unsafe untracked plugin path during update: %r", rel)
            continue
        if rel in tracked or f"{rel}.example" not in tracked:
            continue  # artifact-only: the candidate owns this path or does not model it
        try:
            _copy_untracked_entry(source_root, promoted, rel)
            replayed.append(rel)
        except OSError as exc:  # pragma: no cover - fs-level edge
            logger.warning("Could not replay untracked plugin file %s: %s", rel, exc)
    return replayed


def _clear_plugin_bytecode(target: Path) -> int:
    """Remove ``__pycache__`` dirs under a just-updated plugin checkout. Plugin dirs sit outside
    the repo, so the launch-time bytecode sweep never covers them and stale bytecode after a pull
    can ImportError in the next process. Never raises.

    See #60242, #6207.
    """
    removed = 0
    try:
        for cache_dir in target.rglob("__pycache__"):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)
                removed += 0 if cache_dir.exists() else 1
    except OSError:
        pass
    return removed


def _post_commit_housekeeping(target: Path, stage_live_untracked: List[str], console) -> None:
    """Freshly-promoted-tree housekeeping: stale bytecode + NEW ``*.example`` files only.

    ``*.example`` real-name copies present in the stage-time untracked snapshot
    are deliberately NOT recreated — deleting one after staging was a user
    decision and promotion must not resurrect it. Files the snapshot never
    listed are genuinely new (the update shipped a new example) and are copied
    as before.
    """
    if console is None:
        from hermes_cli.plugins_cmd import _console
        console = _console()
    _clear_plugin_bytecode(target)
    present_at_stage = set(stage_live_untracked)
    for example_file in target.glob("*.example"):
        real_name = example_file.stem
        real_path = target / real_name
        if real_path.exists():
            continue
        if real_name in present_at_stage:
            continue  # deleted from untracked state after staging → do not resurrect
        try:
            shutil.copy2(example_file, real_path)
            console.print(f"[dim]  Created {real_name} from {example_file.name}[/dim]")
        except OSError as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to copy {example_file.name}: {e}")


# ── Post-accept policy (scan + capability settlement) ────────────────────────
# (G4-2 caution-on-update: the scan verdict is carried into the commit decision;
# dangerous → refused pre-promotion; caution → refused unless the caller's
# accept carried an explicit keep decision. The CLI gate resolves that decision
# with the report visible and passes it in; surfaces only render the policy.)


def _scan_staged_candidate(staged: Path, name: str) -> Tuple[Optional[str], bool, str, List[dict]]:
    """Plugin-guard scan of the exact staged candidate.

    Returns ``(verdict, blocked, reason, findings)`` where *blocked* is True
    only for a ``dangerous`` verdict (the commit decision refuses it). A
    ``caution`` verdict is carried into the policy for the surfaces to render;
    the review-gate acceptance is the operator's confirmation. Scanning is
    skipped entirely (verdict None) when ``plugins.scan_on_install`` is off.
    """
    from hermes_cli.plugins_cmd import _scan_on_install_enabled
    from tools.plugin_guard import scan_plugin, should_allow_plugin_install
    if not _scan_on_install_enabled():
        return None, False, "", []
    result = scan_plugin(staged, source=name)
    allowed, reason = should_allow_plugin_install(result)
    fields = ("pattern_id", "severity", "category", "file", "line", "description")
    findings = [{k: getattr(f, k, None) for k in fields} for f in result.findings]
    return result.verdict, allowed is False, reason, findings


def _capability_delta(plugin_id: str, declared: List[str]) -> Tuple[bool, List[str]]:
    """``(declared_set_changed, pending_capabilities)`` for the candidate manifest."""
    from hermes_cli.plugin_capabilities import declared_set_changed, pending_capabilities
    pending = pending_capabilities(plugin_id, declared)
    return declared_set_changed(plugin_id, declared), pending


def _declared_capabilities_from_manifest(manifest: dict, plugin_name: str = "?") -> List[str]:
    """Extract + normalize the ``capabilities:`` declaration from a manifest."""
    from hermes_cli.plugin_capabilities import parse_declared_capabilities
    return parse_declared_capabilities((manifest or {}).get("capabilities"), plugin_name)


def _refresh_capability_consent_hash(plugin_id: str, declared: List[str]) -> None:
    """Re-record the declared-set hash when the set changed but nothing is pending.

    Bookkeeping only (no grants are made here — granting needs a human). The
    surfaces then settle identically: a declared-set change that adds no new
    capability must not re-prompt on the next update through either entrypoint.
    Best effort: a config-write failure must not roll back an already-durable
    plugin commit.
    """
    try:
        from hermes_cli.plugin_capabilities import record_consent
        record_consent(plugin_id, [], declared)
    except Exception:
        logger.warning(
            "Could not refresh the capability consent record for %s after update",
            plugin_id, exc_info=True)


# ── Commit (activate/promote/settle) — the single mutation authority ─────────


def _commit_staged_plugin_update(
    name: str, target: Path, review_token: str, *, accept_caution: bool = False, console=None,
) -> PluginUpdatePolicy:
    """Promote one reviewed staged candidate under the per-plugin commit lock.

    Everything that mutates the plugin namespace for one update runs under the
    per-plugin lock, in this order:

    1. re-read the stage metadata and revalidate the LIVE tree (still the old
       revision + old artifact, tracked-clean) and the STAGED tree (still the
       candidate revision + candidate artifact) — the review decision is bound
       to the exact artifacts it was made against, and a concurrent accept of a
       different candidate from the same old revision loses here (stale /
       superseded → refused);
    2. run the plugin security scan on the exact staged candidate and refuse
       (``outcome=refused``) a dangerous verdict BEFORE promotion;
    3. promote: live → ``backup-*``, staged → live (two ``os.replace`` renames
       on the same filesystem), replay the allowed mutable untracked state from
       the live tree into the promoted artifact, copy genuinely-new
       ``*.example`` files, clear stale bytecode;
    4. settle the metadata sidecar + consent record (the durable terminal
       outcome) and only then clean up the stage + backup and release the lock.

    Any failure before the metadata write rolls the tree and the sidecar back.
    Returns the surface-neutral :class:`PluginUpdatePolicy` for the caller to
    render. Raises :class:`PluginOperationError` when the accept itself is
    stale/superseded or the stage is invalid.
    """
    from hermes_cli.plugins_cmd import (
        PluginOperationError, _git_head_revision, _read_install_metadata,
        _resolve_git_executable, _write_install_metadata)
    if not _HEX64_RE.fullmatch(review_token):
        raise PluginOperationError("Invalid plugin update review token.")
    git_exe = _resolve_git_executable()
    if not git_exe:
        raise PluginOperationError("git is not installed or not in PATH.")
    root = _plugin_update_root()

    with _per_plugin_update_lock(name, root):
        stage_meta = _read_stage_metadata(review_token)
        if stage_meta.get("name") != name or stage_meta.get("target") != str(target.resolve()):
            raise PluginOperationError("The staged plugin update does not match this plugin.")
        staged_path = root / review_token

        # Revalidate live: still exactly what the review was based on. Under the
        # lock a concurrent accept of another candidate from the same old
        # revision fails HERE — exactly one commit survives.
        if _git_head_revision(target, git_exe) != stage_meta.get("old_revision"):
            _discard_staged_update(review_token)
            raise PluginOperationError(
                "The live plugin revision changed after review (another update may have "
                "been accepted); the staged update was discarded. Review a fresh update.")
        old_kind, old_artifact = _plugin_artifact_identity(target, is_git=True, git_exe=git_exe)
        if old_artifact != stage_meta.get("old_revision_artifact"):
            _discard_staged_update(review_token)
            raise PluginOperationError(
                "The live plugin content changed after review; the staged update was "
                "discarded. Review a fresh update.")
        if _plugin_checkout_tracked_dirty(target, git_exe):
            _discard_staged_update(review_token)
            raise PluginOperationError(
                "The live plugin has uncommitted tracked changes after review; the staged "
                "update was discarded. Review a fresh update.")

        # Revalidate staged: still exactly the reviewed candidate.
        if _git_head_revision(staged_path, git_exe) != stage_meta.get("candidate_revision"):
            _discard_staged_update(review_token)
            raise PluginOperationError("The staged plugin revision changed after review.")
        cand_kind, cand_artifact = _plugin_artifact_identity(staged_path, is_git=True, git_exe=git_exe)
        if (cand_artifact != stage_meta.get("candidate_artifact")
                or cand_kind != stage_meta.get("candidate_artifact_kind")):
            _discard_staged_update(review_token)
            raise PluginOperationError("The staged plugin content changed after review.")

        # Policy: scan the exact staged candidate BEFORE promotion. The verdict
        # is carried into the commit decision; a dangerous candidate is never
        # promoted (the live tree stays at the last consented revision).
        scan_verdict, scan_blocked, scan_reason, scan_findings = _scan_staged_candidate(
            staged_path, name)
        new_manifest = stage_meta.get("new_manifest") or {}
        plugin_id = new_manifest.get("name") or name
        declared_caps = _declared_capabilities_from_manifest(new_manifest, plugin_id)
        if declared_caps:
            caps_changed, pending = _capability_delta(plugin_id, declared_caps)
        else:
            caps_changed, pending = False, []
        if scan_blocked:
            _discard_staged_update(review_token)
            return PluginUpdatePolicy(
                outcome=_OUTCOME_REFUSED, committed=False,
                candidate_revision=stage_meta.get("candidate_revision"),
                candidate_artifact=cand_artifact,
                reason=f"Security scan blocked the plugin update: {scan_reason}",
                scan_verdict=scan_verdict, scan_blocked=True,
                scan_findings=scan_findings,
                pending_capabilities=pending, capabilities_changed=caps_changed)
        # Caution verdict on content being adopted (HookPry G4-2): promotion requires an
        # explicit keep-enabled decision. The CLI gate resolves it (--accept-caution / TTY y)
        # before calling commit and passes the outcome here; the Dashboard's accept carries no
        # such decision, so a caution candidate settles refused — the same surface-neutral
        # outcome the CLI's non-interactive path gets, never a silent adopt-and-keep. In the
        # staged model "fail closed" means the candidate is NOT promoted (no disable is
        # needed: unlike the old pull-then-scan shape, the caution tree never reached the
        # live namespace, which stays at the last consented revision).
        if scan_verdict == "caution" and not accept_caution:
            _discard_staged_update(review_token)
            return PluginUpdatePolicy(
                outcome=_OUTCOME_REFUSED, committed=False,
                candidate_revision=stage_meta.get("candidate_revision"),
                candidate_artifact=cand_artifact,
                reason=(
                    "Security scan returned a caution verdict on the update; adopting it "
                    "requires an explicit keep-enabled decision (--accept-caution). "
                    "Nothing was adopted — the live tree stays at the last consented "
                    "revision."),
                scan_verdict=scan_verdict, scan_blocked=False,
                scan_findings=scan_findings,
                pending_capabilities=pending, capabilities_changed=caps_changed)

        # The review bound to (key, old rev, candidate rev, candidate artifact);
        # the live old artifact is recorded at stage time and re-checked above.
        old_metadata = _read_install_metadata()
        old_record = dict(old_metadata.get(name) or {})
        new_metadata = {
            **old_metadata,
            name: {
                **old_record,
                "revision": stage_meta["candidate_revision"],
                "consent": _consent_record(
                    cand_kind, cand_artifact, stage_meta["candidate_revision"], scope="update"),
            },
        }

        backup = root / f"backup-{review_token}"
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        replaced_existing = target.exists()
        untracked_now: List[str] = []
        try:
            # Mutable-state snapshot of the LIVE tree taken under the lock,
            # before the renames move it to backup-*: exactly the current bytes
            # the operator/plugin left there after staging.
            if replaced_existing:
                untracked_now = _untracked_relpaths(target, git_exe) or []
                os.replace(target, backup)
            os.replace(staged_path, target)
            if replaced_existing:
                _replay_allowed_untracked(backup, target, git_exe, untracked_now)
            _post_commit_housekeeping(
                target, stage_meta.get("live_untracked") or [], console)
            _write_install_metadata(new_metadata)
        except Exception:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if replaced_existing and backup.exists():
                os.replace(backup, target)
            if old_record:
                _write_install_metadata(old_metadata)
            raise
        finally:
            _discard_staged_update(review_token)
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)

        # Declared-set bookkeeping that involves no human decision settles in
        # the owner so CLI and Dashboard leave the same consent state. Only
        # genuinely pending (declared-but-ungranted) capabilities make the
        # outcome ``pending`` — a changed-but-covered set is granted.
        if declared_caps and caps_changed and not pending:
            _refresh_capability_consent_hash(plugin_id, declared_caps)

        outcome = _OUTCOME_PENDING if pending else _OUTCOME_GRANTED
        return PluginUpdatePolicy(
            outcome=outcome, committed=True,
            candidate_revision=stage_meta["candidate_revision"],
            candidate_artifact=cand_artifact,
            scan_verdict=scan_verdict, scan_blocked=False,
            scan_findings=scan_findings,
            pending_capabilities=pending, capabilities_changed=caps_changed)
