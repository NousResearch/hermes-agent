"""Managed-files policy for the dashboard file browser: root resolution, path containment, entry metadata.
"""

import mimetypes
import os
import stat
import urllib.request
from dataclasses import dataclass
from fastapi import HTTPException, Request
from pathlib import Path
from typing import Any, Dict, Optional


_MANAGED_FILES_ROOT_ENV = "HERMES_DASHBOARD_FILES_ROOT"
_HOSTED_MANAGED_FILES_ROOT = Path("/opt/data")


@dataclass(frozen=True)
class ManagedFilesPolicy:
    default_path: Path
    locked_root: Path | None
    can_change_path: bool


def _resolve_fs_candidate(raw: str, *, cwd: str | None = None) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        base = Path(cwd).expanduser() if cwd is not None else Path.cwd()
        if not base.is_absolute():
            raise HTTPException(status_code=400, detail="Session working directory is unavailable")
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _fs_path(raw_path: str, *, cwd: str | None = None, decode_fallback: bool = True) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Path is required")
    if "\0" in raw:
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        if raw.lower().startswith("file:"):
            parsed = urllib.parse.urlparse(raw)
            uri_path = parsed.path
            if parsed.netloc and parsed.netloc.lower() != "localhost":
                if os.name != "nt":
                    raise ValueError
                uri_path = f"//{parsed.netloc}{uri_path}"
            raw = urllib.request.url2pathname(uri_path)
        elif os.name == "nt":
            # MEDIA links can reuse Git Bash paths; native Path would read /c/ as C:\c\.
            from tools.environments.local import _msys_to_windows_path

            raw = _msys_to_windows_path(raw)
        candidate = _resolve_fs_candidate(raw, cwd=cwd)
        # A remote client hop may percent-encode a path on top of HTTP's own
        # decoding, so a non-ASCII name can arrive as a literal "%E5%8D%8A..."
        # string that stats as missing (issue #103425). The verbatim path wins
        # whenever it exists, so filenames that genuinely contain "%XX" keep
        # resolving as-is; the unquoted form only rescues the lookup.
        if decode_fallback and "%" in raw and not candidate.exists():
            decoded = _resolve_fs_candidate(urllib.parse.unquote(raw), cwd=cwd)
            if decoded.exists():
                candidate = decoded
        return candidate
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
    return text


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


def _managed_files_policy(request: Optional[Request], *, create_root: bool = True) -> ManagedFilesPolicy:
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

    if for_write and not candidate.exists():
        parent = _canonical_path(candidate.parent)
        resolved = parent / candidate.name
    else:
        resolved = _canonical_path(candidate, require_exists=not for_write)

    if root is not None and not _path_is_under(root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    return policy, resolved, str(resolved)


def _managed_response_meta(policy: ManagedFilesPolicy) -> Dict[str, Any]:
    locked_root = str(policy.locked_root) if policy.locked_root is not None else None
    return {"root": locked_root, "locked_root": locked_root, "can_change_path": policy.can_change_path}


def _hosted_fs_path_allowed(root: Path, target: Path) -> bool:
    from hermes_cli.web_routers.files import _is_sensitive_path

    resolved = _canonical_path(target)
    return (_path_is_under(root, resolved)
            and not _is_sensitive_path(target) and not _is_sensitive_path(resolved))


def _hosted_fs_read_guard(target: Path, request: Optional[Request] = None) -> Path | None:
    """Preview and Git reads share managed-file restrictions on locked deployments.

    ``request`` is unused (the policy is env/home driven) and optional so the
    route functions stay callable directly, e.g. from agent-side tests.
    """
    root = _managed_files_policy(request, create_root=False).locked_root
    if root is not None and not _hosted_fs_path_allowed(root, target):
        raise HTTPException(status_code=403, detail="Path is outside the managed read boundary")
    return root


def _managed_file_entry(
    policy: ManagedFilesPolicy, target: Path, *, skip_missing: bool = False
) -> Dict[str, Any] | None:
    """Describe an entry; listings may skip vanished files, while writes stay strict."""
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")

    # A dangling symlink is still a directory entry even when its missing
    # target resolves outside the managed root. Classify only a definite
    # missing target before checking the resolved-target boundary so one safe
    # placeholder does not abort the whole listing. Permission and other I/O
    # errors still pass through the existing boundary/error handling below.
    st = None
    if target.is_symlink():
        try:
            st = resolved.stat()
        except FileNotFoundError:
            # A placeholder may describe only an entry inside the managed root,
            # even when the missing destination is outside it.
            if policy.locked_root is not None and not _path_is_under(
                policy.locked_root, target.parent.resolve() / target.name
            ):
                raise HTTPException(status_code=403, detail="Path outside managed files root")
            return {
                "name": target.name or resolved.name or str(resolved),
                "path": str(target),
                "is_directory": False,
                "broken_link": True,
                "size": None,
                "mtime": None,
                "mime_type": None,
            }
        except OSError:
            pass

    if policy.locked_root is not None and not _path_is_under(policy.locked_root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    if st is None:
        try:
            st = resolved.stat()
        except FileNotFoundError as exc:
            if skip_missing:
                return None
            raise HTTPException(status_code=500, detail=f"Could not stat path: {exc}")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not stat path: {exc}")

    is_dir = stat.S_ISDIR(st.st_mode)
    mime_type = None if is_dir else (mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
    return {
        "name": target.name or resolved.name or str(resolved),
        "path": str(resolved),
        "is_directory": is_dir,
        "size": None if is_dir else st.st_size,
        "mtime": st.st_mtime,
        "mime_type": mime_type,
    }
