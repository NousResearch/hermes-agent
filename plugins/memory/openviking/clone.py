"""Offline clone preparation; remote account, user and optional peer stay shared.

The host binds the source profile scope and owns staging publication/rollback.
Do not import the runtime provider here: even importing it registers an atexit
handler, and activating it can probe or start a local OpenViking server.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from agent.secret_scope import get_secret

_CONFIG_ENV = "OPENVIKING_CLI_CONFIG_FILE"
_RECOVERY_DIRS = (Path("openviking/pending_sessions"), Path("openviking/runs"))


def _check_staging_component(path: Path, *, allow_leaf_link: bool = False) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    # Junctions are not symlinks on Python 3.11. Refuse every reparse point,
    # including leaves, before any writes rather than guessing unlink semantics.
    if (getattr(info, "st_reparse_tag", 0)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or (stat.S_ISLNK(info.st_mode) and not allow_leaf_link)):
        raise ValueError("OpenViking clone found a staging link/reparse point; replace it with a private file/directory and retry.")


def _safe_staging_path(home: Path, relative: Path, *, allow_leaf_link: bool = False) -> Path:
    """Check from the staging root, never following a copied parent link.

    ``home`` has a canonical host-owned parent, but its leaf remains unchecked
    until here. Leaf symlinks are allowed only for unlink/atomic replacement;
    every traversed component, including the staging root, must be link-free.
    """
    path = home
    parts = relative.parts
    _check_staging_component(home)
    for index, part in enumerate(parts):
        path = path / part
        _check_staging_component(path, allow_leaf_link=allow_leaf_link and index == len(parts) - 1)
    return path


def _relative_to_profile(path: Path, source: Path) -> Path | None:
    """Find the first profile boundary by identity, preserving its raw suffix.

    Resolve neither the suffix nor its parents before matching: a nested link
    back to the profile must remain untrusted staging payload. Filesystem
    identity also handles case aliases without conflating Linux directories.
    """
    for parent in (*reversed(path.parents), path):
        try:
            if parent.samefile(source):
                return path.relative_to(parent)
        except FileNotFoundError:
            return None  # A missing external ancestor cannot contain the source.
    return None


def _private_link(raw: str, source_home: Path) -> tuple[Path, Path] | None:
    if not isinstance(raw, str) or any(c in raw for c in ("\x00", "\r", "\n")):
        raise ValueError("OpenViking CLI config path is malformed; configure an absolute file path before cloning.")
    path = Path(raw).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("OpenViking CLI config path is ambiguous; configure an absolute file path before cloning.")
    source = source_home.resolve(strict=True)
    relative = _relative_to_profile(path, source)
    if relative is None:
        if _relative_to_profile(path.resolve(), source) is not None:
            raise ValueError("OpenViking CLI config aliases a private file; link its absolute profile-local path before cloning.")
        return None  # An external/global link is intentionally shared.
    resolved = path.resolve(strict=True)
    resolved_relative = _relative_to_profile(resolved, source)
    if resolved_relative is None or not resolved.is_file():
        raise ValueError("OpenViking private CLI config escapes its profile or is not a file; replace the link before cloning.")
    # Reserve these names case-insensitively on every host: staging may live on
    # a case-insensitive volume even when the source does not (and vice versa).
    for candidate in (relative, resolved_relative):
        folded = Path(*(part.casefold() for part in candidate.parts))
        if folded in (Path("config.yaml"), Path(".env")) or any(folded.is_relative_to(p) for p in _RECOVERY_DIRS):
            raise ValueError("OpenViking CLI config overlaps clone metadata; move it to a private config directory before cloning.")
    return resolved, relative


def _write_pointer(env_path: Path, value: str) -> None:
    # Use Hermes's dotenv parser/quoting/atomic writer, but never save_env_value:
    # that publishes to the SOURCE's scope/process environment as a side effect.
    from hermes_cli.config import _env_line_defines_key, _quote_env_value, _write_env_lines

    lines = env_path.read_text(encoding="utf-8-sig").splitlines(keepends=True) if env_path.exists() else []
    lines = [line for line in lines if not _env_line_defines_key(line, _CONFIG_ENV)]
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append(f"{_CONFIG_ENV}={_quote_env_value(value)}\n")
    _write_env_lines(env_path, lines, preserve_mode=False)


def prepare_clone(*, source_home: Path, source_name: str, staging_home: Path,
                  destination_home: Path, destination_name: str, clone_all: bool) -> dict | None:
    """Strip run ownership and relocate a linked source-private credential file.

    Both clone modes materialize private ovcli files. No server DB is copied and
    no remote identity is invented: absent peers deliberately use user memory.

    The host owns the staging directory's parent and publication transaction.
    Aliases above that boundary are host paths, not copied payload: canonicalize
    the parent once, never the staging leaf. Validate the root and every path
    component we access beneath it; source-private files must resolve inside the
    actual source profile. This is not a defense against concurrent host mutation.
    """
    from hermes_cli.config import _expand_env_vars, read_user_config_raw
    from utils import atomic_yaml_write

    try:
        staging_home = staging_home.parent.resolve(strict=True) / staging_home.name
        config_path = _safe_staging_path(staging_home, Path("config.yaml"))
        recovery = [_safe_staging_path(staging_home, path, allow_leaf_link=True) for path in _RECOVERY_DIRS]
        raw = read_user_config_raw(config_path) if config_path.exists() else {}
        memory = raw.get("memory", {})
        if memory is None:
            memory = {}
        if not isinstance(memory, dict):
            raise ValueError("OpenViking clone requires memory to be a mapping or null in config.yaml; repair it before cloning.")
        raw_config = memory.get("openviking", {})
        if raw_config is None:
            raw_config = {}
        if not isinstance(raw_config, dict):
            raise ValueError("OpenViking clone requires memory.openviking to be a mapping or null in config.yaml; repair it before cloning.")
        # Resolve behavior in the host's source scope without a loader's backup
        # writes; only the authoritative pointer is replaced in the raw YAML.
        config = _expand_env_vars(raw_config)
        private = None
        env_pointer = ""
        if config.get("use_ovcli_config"):
            env_pointer = (get_secret(_CONFIG_ENV, "") or "").strip()
            pointer = env_pointer or config.get("ovcli_config_path") or ""
            if pointer:
                private = _private_link(pointer, source_home)
        if private:
            source_file, relative = private
            target = _safe_staging_path(staging_home, relative, allow_leaf_link=True)
            env_path = _safe_staging_path(staging_home, Path(".env")) if env_pointer else None
            final_path = str(destination_home / relative)
            if any(c in final_path for c in ("\x00", "\r", "\n")):
                raise ValueError("OpenViking destination path is malformed; choose another profile location.")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".ovcli-clone-")
            try:
                with os.fdopen(fd, "wb") as output, source_file.open("rb") as original:
                    shutil.copyfileobj(original, output)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
            if env_path is not None:
                _write_pointer(env_path, final_path)
            else:
                raw_config["ovcli_config_path"] = final_path
                atomic_yaml_write(config_path, raw, sort_keys=False)
        for path in recovery:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
        return {"connection": "preserved", "private_config_materialized": private is not None}
    except (OSError, RuntimeError):
        raise ValueError(
            "OpenViking clone could not read its private CLI config or prepare local files; "
            "check that the linked file exists, is readable, and stays inside the source profile, then retry."
        ) from None
