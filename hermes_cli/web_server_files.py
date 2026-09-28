"""Managed-files policy for the dashboard file browser: root resolution, path
containment, entry metadata, plus the sensitive-path denylist and canonical
write-guard seam shared by the ``/api/files/*`` and ``/api/fs/*`` routes.
"""

import mimetypes
import os
import urllib.request
from dataclasses import dataclass
from fastapi import HTTPException, Request
from pathlib import Path
from typing import Any, Dict


_MANAGED_FILES_ROOT_ENV = "HERMES_DASHBOARD_FILES_ROOT"
_HOSTED_MANAGED_FILES_ROOT = Path("/opt/data")


@dataclass(frozen=True)
class ManagedFilesPolicy:
    default_path: Path
    locked_root: Path | None
    can_change_path: bool


def _reject_namespace_path(raw: str) -> None:
    """400 on NT/device-namespace and UNC input.

    Checked on the raw string and again after ``file:`` unwrapping: merely
    resolving ``\\?\\UNC\\host\\share`` or a bare ``\\\\host\\share`` can trigger
    outbound SMB auth (NTLM leak) on Windows before any resolved-path denylist
    could fire. ``is_nt_namespace_path`` deliberately permits bare UNC for
    agent tools; this remote surface never needs it.
    """
    from agent.file_safety import is_nt_namespace_path
    if is_nt_namespace_path(raw) or raw.replace("/", "\\").startswith("\\\\"):
        raise HTTPException(status_code=400, detail="NT/device namespace paths are not allowed")


def _fs_path(raw_path: str, *, cwd: str | None = None) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Path is required")
    if "\0" in raw:
        raise HTTPException(status_code=400, detail="Invalid path")
    _reject_namespace_path(raw)
    try:
        if raw.lower().startswith("file:"):
            parsed = urllib.parse.urlparse(raw)
            uri_path = parsed.path
            if parsed.netloc and parsed.netloc.lower() != "localhost":
                if os.name != "nt":
                    raise ValueError
                uri_path = f"//{parsed.netloc}{uri_path}"
            raw = urllib.request.url2pathname(uri_path)
            # The netloc unwrap can manufacture a UNC path out of input the
            # raw check already cleared (file://host/share -> \\host\share).
            _reject_namespace_path(raw)
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            base = Path(cwd).expanduser() if cwd is not None else Path.cwd()
            if not base.is_absolute():
                raise HTTPException(status_code=400, detail="Session working directory is unavailable")
            candidate = base / candidate
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid path")


def _canonical_path(path: Path, *, require_exists: bool = False) -> Path:
    try:
        return path.expanduser().resolve(strict=require_exists)
    except FileNotFoundError:
        if require_exists:
            raise HTTPException(status_code=404, detail="Path not found")
        raise
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")


def _ensure_managed_root(raw_path: str | Path) -> Path:
    root = Path(raw_path).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Managed files root is unavailable: {exc}")
    if not resolved.is_dir():
        raise HTTPException(status_code=500, detail="Managed files root is not a directory")
    return resolved


def _path_is_under(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _path_text(raw_path: str | None) -> str:
    text = str(raw_path or "").strip()
    if "\x00" in text:
        raise HTTPException(status_code=400, detail="Invalid path")
    _reject_namespace_path(text)
    return text


# --- Sensitive-path denylist ------------------------------------------------
# Basenames denied wherever they appear on the managed and free-fs surfaces:
# credential stores that become live secrets in the browsable tree the moment an
# operator points the managed root at HERMES_HOME (#57505). Mirrors the canonical
# guards (agent.file_safety.get_read_block_error, gateway.platforms.base
# ._ROOT_CREDENTIAL_PATHS) so this surface never lags them.
_SENSITIVE_MANAGED_FILE_BASENAMES = frozenset({
    "auth.json", "auth.lock", "credentials", "config.yaml", ".anthropic_oauth.json",
    "google_token.json", "google_oauth_pending.json", "google_oauth.json",
    "webhook_subscriptions.json", "bws_cache.json", "bws_cache.enc.json",
    ".git-credentials",
})

# Directory names whose whole subtree is credential material, matched on ANY path
# component so the trees are blocked wherever they sit, no root resolution needed.
_SENSITIVE_MANAGED_DIR_NAMES = frozenset({"mcp-tokens", "pairing"})

# Credential trees and stores the canonical guards deny ONLY beneath a Hermes
# home: agent.file_safety._READ_DENIED_DIRS (vault/, browser-profile/, skills/.hub)
# and the delivery guard's _ROOT_CREDENTIAL_PATHS (sessions/, state.db, kanban.db
# + SQLite sidecars). Anchored to the first component below each credential home,
# matching canonical semantics: these are common names a user may legitimately
# browse elsewhere on disk, AND deeper inside the Hermes tree (a plugin's own
# state.db, a backup dir's sessions/).
_HERMES_SCOPED_DIR_RELS = (("vault",), ("browser-profile",), ("sessions",), ("skills", ".hub"))
_HERMES_SCOPED_FILE_BASENAMES = frozenset({
    name
    for db in ("state.db", "kanban.db")
    for name in (db, f"{db}-wal", f"{db}-shm", f"{db}-journal")
})


def _hermes_credential_roots() -> list[Path]:
    """Resolved Hermes homes whose credential stores this surface guards: the
    active home and shared root (``agent.file_safety._hermes_dirs``, already
    resolved, deduplicated and fail-soft) plus every ``profiles/<name>`` home,
    enumerated at check time like ``gateway.platforms.base._profile_dirs`` so a
    profile created mid-process is covered.
    """
    from agent.file_safety import _hermes_dirs, _resolve_each

    roots = _hermes_dirs()
    profiles: list[Path] = []
    for root in roots:
        try:
            profiles.extend(_resolve_each(p for p in (root / "profiles").iterdir() if p.is_dir()))
        except OSError:
            continue
    return list(dict.fromkeys(roots + profiles))


def _is_hermes_scoped_sensitive(target: Path, roots: list[Path] | None = None) -> bool:
    """True when ``target`` names (or sits under) a credential path anchored at
    the top of a Hermes home: ``vault/``, ``browser-profile/``, ``sessions/``,
    ``skills/.hub/``, or the session/kanban SQLite stores. ``target`` must
    already be resolved.

    Compares on lowercased parts: on case-insensitive filesystems (default
    macOS APFS) ``resolve()`` preserves the caller's typed case, so a
    case-variant of the root prefix must still match (over-denying a
    differently-cased directory on a case-sensitive FS is the safe direction).
    """
    tparts = tuple(part.lower() for part in target.parts)
    for root in _hermes_credential_roots() if roots is None else roots:
        rparts = tuple(part.lower() for part in root.parts)
        if tparts[: len(rparts)] != rparts:
            continue
        below = tparts[len(rparts):]
        if not below:
            continue  # the home dir itself is not credential material
        if any(below[: len(rel)] == rel for rel in _HERMES_SCOPED_DIR_RELS):
            return True
        if below[0] in _HERMES_SCOPED_FILE_BASENAMES:
            return True
        # kanban/boards/<board>/kanban.db* sit one level deeper than the
        # home-anchored stores (the delivery guard enumerates them via
        # _kanban_board_db_paths).
        if below[:2] == ("kanban", "boards") and below[-1] in _HERMES_SCOPED_FILE_BASENAMES:
            return True
    return False


def _is_sensitive_filename(name: str) -> bool:
    """Basename denylist: ``.env`` / ``.env.<suffix>`` / ``.envrc`` plus the
    credential-store basenames. Case-insensitive so ``.ENV`` / ``Auth.JSON``
    on case-insensitive mounts can't slip past. Basename-only, call sites use
    :func:`_is_sensitive_path`, which adds the credential-directory checks."""
    lowered = name.lower()
    if lowered == ".env" or lowered.startswith(".env.") or lowered == ".envrc":
        return True
    return lowered in _SENSITIVE_MANAGED_FILE_BASENAMES


def _is_sensitive_path(path: Path, roots: list[Path] | None = None) -> bool:
    """True when the basename is sensitive, any path component (case-insensitive)
    is a credential directory, or the path sits inside a Hermes-scoped credential
    tree. Read-side guard shared by the managed and free-fs routes; write
    endpoints additionally apply the canonical write guard via
    :func:`_raise_for_sensitive_target`.

    ``roots`` lets directory listings compute the credential-home set once per
    request instead of re-resolving it for every entry.
    """
    if _is_sensitive_filename(path.name):
        return True
    if any(part.lower() in _SENSITIVE_MANAGED_DIR_NAMES for part in path.parts):
        return True
    return _is_hermes_scoped_sensitive(path, roots)


def _raise_for_sensitive_target(target: Path, *, write: bool = False) -> None:
    """403 when ``target`` is a denied credential path.

    ``write=True`` additionally enforces the canonical write guard
    (``agent.file_safety``): home credential dirs (``~/.ssh``, ``~/.aws``, ...),
    system files, ``HERMES_WRITE_SAFE_ROOT``, and approval-gated paths
    (``~/.ssh/config``) fail closed because the dashboard has no approval
    channel. ``target`` must already be resolved; raw-string checks
    (NT/device namespace, NUL) belong to ``_fs_path`` / ``_path_text``.
    """
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    if not write:
        return
    from agent.file_safety import get_write_denied_error, is_write_approval_required
    denial = get_write_denied_error(str(target), verb="Write")
    if denial is None and is_write_approval_required(str(target)):
        denial = "Write denied: this path requires an approval the dashboard cannot prompt for."
    if denial:
        raise HTTPException(status_code=403, detail=denial)


def _raise_for_protected_tree_delete(target: Path) -> None:
    """403 when a directory delete would remove a credential store.

    The per-path denylist inspects the target itself, so deleting a CONTAINER
    (``~/.hermes``, ``~/.ssh``, or any ancestor like ``~``) would destroy
    ``vault/``, ``state.db`` and the home credential dirs it was written to
    protect. Denies when ``target`` equals or contains a Hermes credential
    home or a canonical write-denied directory.
    """
    protected: list[Path] = list(_hermes_credential_roots())
    try:
        from agent.file_safety import build_write_denied_prefixes
        protected.extend(
            Path(prefix.rstrip(os.sep))
            for prefix in build_write_denied_prefixes(str(Path.home()))
        )
    except (OSError, RuntimeError):
        pass
    tlower = tuple(part.lower() for part in target.parts)
    for path in protected:
        plower = tuple(part.lower() for part in path.parts)
        if len(tlower) <= len(plower) and plower[: len(tlower)] == tlower:
            raise HTTPException(
                status_code=403,
                detail="Cannot delete a directory that contains credential stores",
            )


def _default_hermes_root_is_opt_data() -> bool:
    raw = os.environ.get("HERMES_HOME", "").strip()
    if not raw:
        return False
    try:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root().expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        root = Path(raw).expanduser().resolve(strict=False)
    return root == _HOSTED_MANAGED_FILES_ROOT


def _dashboard_local_update_managed_externally() -> bool:
    """True when the dashboard should not offer ``hermes update``.

    Containerized dashboards are updated by the outer launcher/image — except a
    ``git`` install (bind-mounted checkout, e.g. the hermes-webui image), where
    the update button is the correct path. pip stays blocked in containers: its
    apply path mutates the running container filesystem.
    """
    from hermes_cli.web_server import PROJECT_ROOT
    from hermes_cli.config import detect_install_method
    if _default_hermes_root_is_opt_data():
        return True
    try:
        from hermes_constants import is_container

        if not is_container():
            return False
    except Exception:
        return False
    try:
        if detect_install_method(PROJECT_ROOT) == "git":
            return False
    except Exception:
        pass
    return True


def _managed_files_policy(request: Request, *, create_root: bool = True) -> ManagedFilesPolicy:
    raw_forced_root = os.environ.get(_MANAGED_FILES_ROOT_ENV, "").strip()
    if raw_forced_root:
        root = _ensure_managed_root(raw_forced_root) if create_root else _canonical_path(Path(raw_forced_root))
        return ManagedFilesPolicy(default_path=root, locked_root=root, can_change_path=False)

    # Remote/OAuth access does not imply a hosted container (a gated macOS launchd
    # install still browses its home). Lock to /opt/data only when the Hermes
    # root actually IS /opt/data or HERMES_DASHBOARD_FILES_ROOT is set.
    if _default_hermes_root_is_opt_data():
        root = _ensure_managed_root(_HOSTED_MANAGED_FILES_ROOT) if create_root else _HOSTED_MANAGED_FILES_ROOT
        return ManagedFilesPolicy(default_path=root, locked_root=root, can_change_path=False)

    home = _canonical_path(Path.home())
    return ManagedFilesPolicy(default_path=home, locked_root=None, can_change_path=True)


def _resolve_managed_path(
    raw_path: str | None, request: Request, *, for_write: bool = False
) -> tuple[ManagedFilesPolicy, Path, str]:
    policy = _managed_files_policy(request)
    text = _path_text(raw_path)
    root = policy.locked_root

    if root is not None and (not text or text in {".", "/"}):
        candidate = root
    elif not text:
        candidate = policy.default_path
    else:
        candidate = Path(text).expanduser()
        if root is not None and not candidate.is_absolute():
            if any(part == ".." for part in candidate.parts):
                raise HTTPException(status_code=400, detail="Path cannot contain '..'")
            candidate = root / candidate
        elif not candidate.is_absolute():
            raise HTTPException(status_code=400, detail="Path must be absolute")

    if ".." in candidate.parts:
        raise HTTPException(status_code=400, detail="Path cannot contain '..'")

    # resolve(strict=False) still follows every symlink that exists, including a
    # dangling leaf: a planted link must not launder a write past the denylist.
    resolved = _canonical_path(candidate, require_exists=not for_write)

    if root is not None and not _path_is_under(root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    # Write ops deny sensitive targets outright (uploading over state.db, planting
    # into mcp-tokens/, deleting credentials). Read ops keep the established
    # contract: list filters sensitive entries rather than 403-ing the directory,
    # and the file-read call sites 403 on sensitive targets themselves.
    if for_write:
        _raise_for_sensitive_target(resolved, write=True)
    return policy, resolved, str(resolved)


def _managed_response_meta(policy: ManagedFilesPolicy) -> Dict[str, Any]:
    locked_root = str(policy.locked_root) if policy.locked_root is not None else None
    return {"root": locked_root, "locked_root": locked_root, "can_change_path": policy.can_change_path}


def _managed_file_entry(policy: ManagedFilesPolicy, target: Path) -> Dict[str, Any]:
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")
    if policy.locked_root is not None and not _path_is_under(policy.locked_root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    try:
        st = resolved.stat()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat path: {exc}")

    is_dir = resolved.is_dir()
    mime_type = None if is_dir else (mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
    return {
        "name": target.name or resolved.name or str(resolved),
        "path": str(resolved),
        "is_directory": is_dir,
        "size": None if is_dir else st.st_size,
        "mtime": st.st_mtime,
        "mime_type": mime_type,
    }
