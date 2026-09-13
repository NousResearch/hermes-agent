"""Directory initialization and storage diagnostics for the active Hermes home."""

import os
from pathlib import Path


# Link provenance is not proof of initialization: legacy homes must still run it.
_HOME_LINK_ALIASES: dict[Path, Path] = {}


class HomeInitializationError(RuntimeError):
    """The home skeleton is unavailable, not an invalid YAML document."""


def _directory_links(path: Path) -> list[Path]:
    return [part for part in (*reversed(path.parents), path) if part.is_symlink()]


def _home_directory_path(home: Path) -> Path:
    """Recover a live lexical link after plugin discovery resolves the home."""
    if _directory_links(home):
        _HOME_LINK_ALIASES[home.resolve()] = home.absolute()
        return home
    alias = _HOME_LINK_ALIASES.get(home)
    if alias is not None:
        try:
            if _directory_links(alias) and alias.resolve(strict=True) == home:
                return alias
        except (OSError, RuntimeError):
            pass  # A missing target or link loop cannot supply provenance.
        _HOME_LINK_ALIASES.pop(home, None)
    return home


def _ensure_directory(path: Path, *, create: bool, secure: bool, parents: bool = True) -> None:
    from hermes_cli.config import _secure_dir

    detail = ""
    try:
        links = _directory_links(path)
        detail = "; ".join(f"{link} -> {link.readlink()}" for link in links)
        # Never materialize a missing mount's target on the underlying disk.
        for link in links:
            if not link.is_dir():
                raise FileNotFoundError(f"Directory link is unavailable: {link}")
        if create:
            path.mkdir(parents=parents, exist_ok=True)
        elif not path.is_dir():
            raise FileNotFoundError(f"Required directory does not exist: {path}")
        # The operator owns permissions beyond a link, including logs/curator.
        if secure and not links:
            _secure_dir(path)
    except OSError as exc:
        raise HomeInitializationError(
            f"Cannot initialize Hermes directory {path}: {exc}. "
            + (f"Directory links: {detail}. " if detail else "")
            + "Check the directory/link target, mount availability and access permissions; "
            "restore the mount or repair the link before retrying. "
            "Hermes has not replaced the link or created its missing target."
        ) from exc


def initialize_home(
    home: Path, subdirs: tuple[str, ...], ensured: dict[str, tuple[int, int, int, str | None]],
) -> None:
    from hermes_cli.config import _ensure_default_soul_md, _hermes_home_identity, is_managed
    from hermes_constants import named_profile_home_is_unavailable, profile_deletion_marker_path

    named_profile = profile_deletion_marker_path(home) is not None
    if named_profile and not home.is_dir():
        raise FileNotFoundError(f"Named profile home disappeared during initialization: {home}")
    aliases = (home, home.resolve())
    initial_identities = {
        alias: _hermes_home_identity(alias, named_profile=True)
        for alias in aliases if profile_deletion_marker_path(alias) is not None
    }
    directory_home = _home_directory_path(home)
    managed = is_managed()
    old_umask = os.umask(0o007) if managed else None
    try:
        _ensure_directory(directory_home, create=not managed and not named_profile, secure=not managed)
        required = ("cron", "sessions", "logs", "memories") if managed else subdirs
        for subdir in required:
            if named_profile and named_profile_home_is_unavailable(home):
                raise FileNotFoundError(f"Named profile home is missing or being deleted: {home}")
            # A stale initializer must not recreate a deleted profile's parents.
            _ensure_directory(
                directory_home / subdir, create=not managed, secure=not managed, parents=not named_profile,
            )
        if managed:
            if named_profile and named_profile_home_is_unavailable(home):
                raise FileNotFoundError(f"Named profile home is missing or being deleted: {home}")
            _ensure_directory(
                directory_home / "logs" / "curator", create=True, secure=False, parents=not named_profile,
            )
        try:
            _ensure_default_soul_md(directory_home)
        except OSError as exc:
            raise HomeInitializationError(
                f"Cannot initialize Hermes home {home}: {exc}. "
                "Check storage availability and access permissions."
            ) from exc
    finally:
        if old_umask is not None:
            os.umask(old_umask)
    identities = {
        alias: _hermes_home_identity(
            alias, named_profile=profile_deletion_marker_path(alias) is not None,
        )
        for alias in aliases
    }
    # Initialization changes ctime, but must not credit a replacement generation.
    # A tokenless named home cannot prove completion for either spelling.
    for alias, before in initial_identities.items():
        after = identities[alias]
        if (before is None or after is None or before[:2] != after[:2]
                or before[3] != after[3]):
            return
    for alias, identity in identities.items():
        if identity is not None:
            ensured[str(alias)] = identity


def config_load_issue(exc: Exception):
    from hermes_cli.config import ConfigIssue

    if isinstance(exc, (HomeInitializationError, OSError)):
        return ConfigIssue(
            "error", f"Hermes storage is unavailable: {exc}",
            "Check the reported path, link target, mount and permissions; keep config.yaml unchanged.",
        )
    return ConfigIssue("error", "Could not load config.yaml", "Run 'hermes setup' to create a valid config")
