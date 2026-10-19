"""Managed-files policy for the dashboard file browser: root resolution, path containment, entry metadata.
"""

import logging
import mimetypes
import os
import stat
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from fastapi import HTTPException, Request
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

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


def _translate_agent_path(candidate: Path) -> Path | None:
    """One translation attempt under whatever scope is bound; None off the docker backend.

    The gate keeps a local-backend gateway with leftover ``docker_volumes`` from serving
    mount sources in place of real host files. Quiet: fs callers also see ordinary host
    paths, which are not MEDIA misses worth a warning."""
    from gateway.platforms.base import _docker_env_active, _translate_docker_container_media_path
    from tools.terminal_tool import _ensure_terminal_env_bridged

    _ensure_terminal_env_bridged()
    if not _docker_env_active():
        return None
    return _translate_docker_container_media_path(candidate, quiet=True)


@contextmanager
def _agent_path_scope(profile_home: Path | None = None) -> Iterator[None]:
    """Bind the profile whose sandbox a container path names: its home and its terminal policy.

    A requested ``profile_home`` other than the current home binds that home and its complete
    terminal policy, as MEDIA delivery does (#109024). A home already bound by the caller (a
    native profile ticket's override) gets its own terminal policy when no scope is bound, so
    translation never reads the launch profile's ambient ``TERMINAL_DOCKER_VOLUMES``. An
    override naming the launch home keeps the ambient env, which may carry env-only policy.
    Translation and the unmapped-path classification both run inside this scope, so they
    agree about the backend.
    """
    from hermes_cli.config import get_hermes_home
    from hermes_constants import (
        get_hermes_home_override, get_process_hermes_home, reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools.terminal_scope import get_terminal_scope, install_profile_terminal_scope, reset_terminal_scope

    home_token = scope_token = None
    if profile_home is not None and Path(profile_home).resolve() != get_hermes_home().resolve():
        home_token = set_hermes_home_override(str(profile_home))
    try:
        foreign_override = (get_hermes_home_override() is not None
                            and get_hermes_home().resolve() != get_process_hermes_home().resolve())
        if home_token is not None or (foreign_override and get_terminal_scope() is None):
            scope_token = install_profile_terminal_scope(get_hermes_home())
        yield
    finally:
        if scope_token is not None:
            reset_terminal_scope(scope_token)
        if home_token is not None:
            reset_hermes_home_override(home_token)


def _agent_container_path(raw: str, *, profile_home: Path | None = None) -> Path | None:
    """Map an agent-visible container path to its host bind mount, or None.

    Docker agents emit container paths (``/workspace/...``, ``/root/...``) in
    transcripts and MEDIA tags, while the dashboard fs routes stat the gateway
    HOST filesystem (#85238). Reuse the gateway's MEDIA volume translator inside
    ``_agent_path_scope(profile_home)``. When no terminal scope is bound even
    then (``serve --isolated`` after a failed early bridge), retry once under the
    current profile's complete terminal policy, never replacing an already-bound
    scope or refusal (retry from #104992).

    Runs on the raw string, before host normalization: on a Windows gateway
    ``Path("/workspace/x").resolve()`` anchors to the current drive and the
    container path is gone before translation sees it. ``..`` segments are never
    translated, so the managed routes keep refusing them.
    """
    if not raw.startswith("/") or raw.startswith("//") or ".." in raw.split("/"):
        return None
    try:
        from hermes_cli.config import get_hermes_home
        from tools.terminal_scope import (
            build_profile_terminal_scope, get_terminal_scope, reset_terminal_scope, set_terminal_scope,
        )

        candidate = Path(raw)
        with _agent_path_scope(profile_home):
            translated = _translate_agent_path(candidate)
            if translated is not None or get_terminal_scope() is not None:
                return translated
            token = set_terminal_scope(build_profile_terminal_scope(get_hermes_home()))
            try:
                return _translate_agent_path(candidate)
            finally:
                reset_terminal_scope(token)
    except Exception:  # fail open: translation is additive, the host path still serves
        logger.debug("agent container path translation failed for %s", raw, exc_info=True)
        return None


SANDBOX_NOT_FOUND = "File not found on the gateway host: the path may live inside the agent sandbox"


def _agent_fs_path(
    raw_path: str, *, cwd: str | None = None, profile_home: Path | None = None,
) -> Path:
    """``_fs_path`` for agent-facing reads; container paths resolve to their host
    mount, so callers' boundary guards see the real host target.

    POSIX: a readable host path wins, and translation only runs when the host
    lookup misses, so a root-run gateway can still browse its own ``/root``.
    Windows: a container-shaped path never names a host file (it would anchor to
    the current drive), so a translation miss is a 404 rather than a drive lookup.
    """
    raw = str(raw_path or "").strip()
    if os.name != "nt":
        host = _fs_path(raw_path, cwd=cwd)
        if os.access(host, os.R_OK):
            return host
        translated = _agent_container_path(raw, profile_home=profile_home)
        return translated if translated is not None else host
    translated = _agent_container_path(raw, profile_home=profile_home)
    if translated is not None:
        return translated
    if _is_unmapped_container_path(raw, profile_home=profile_home):
        raise HTTPException(status_code=404, detail=SANDBOX_NOT_FOUND)
    return _fs_path(raw_path, cwd=cwd)


def _is_unmapped_container_path(raw: str, *, profile_home: Path | None = None) -> bool:
    """A docker-backend container path (``/workspace``, ``/root`` or a configured
    volume target) rather than a host path or a typo elsewhere on the host. Reads
    the backend and volumes in the same profile scope translation used."""
    if not raw.startswith("/") or raw.startswith("//"):
        return False
    try:
        from gateway.platforms.base import _docker_env_active, _parse_docker_volume_mounts
        from tools.environments.local import _msys_to_windows_path

        with _agent_path_scope(profile_home):
            if not _docker_env_active() or _msys_to_windows_path(raw) != raw:
                return False
            roots = {"/workspace", "/root", *(c.as_posix() for _, c in _parse_docker_volume_mounts())}
    except Exception:  # a refusal scope or import failure: just skip the hint
        logger.debug("container path classification failed for %s", raw, exc_info=True)
        return False
    return any(raw == r or raw.startswith(r.rstrip("/") + "/") for r in roots)


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
    raw_path: str | None, request: Request, *, for_write: bool = False,
    for_agent_read: bool = False,
) -> tuple[ManagedFilesPolicy, Path, str]:
    policy = _managed_files_policy(request)
    text = _path_text(raw_path)
    root = policy.locked_root

    if for_agent_read and text and not text.startswith("file:") and (
            os.name == "nt" or not os.access(text, os.R_OK)):
        # Agent-emitted container paths (/workspace/...) resolve to their host
        # bind mount before the root checks, so the locked-root refusal below
        # still governs the translated target (#85238). A readable host path wins.
        translated = _agent_container_path(text)
        if translated is not None:
            text = str(translated)

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


def _managed_response_meta(policy: ManagedFilesPolicy) -> dict[str, Any]:
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
) -> dict[str, Any] | None:
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
