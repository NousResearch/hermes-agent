"""Directory initialization and storage diagnostics for the active Hermes home."""

import os
from contextlib import suppress
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


def _operator_owned_links(links: list[Path], home: Path) -> list[Path]:
    """Keep only the links that make the operator the owner of the home's permissions.

    That boundary is the home itself and anything under it. A link *above* the home
    (macOS ``/tmp`` -> ``/private/tmp``, ``/var`` -> ``/private/var``, or any aliased parent
    the user happens to live in) says nothing about who owns :data:`home`, and treating it as
    an operator-owned link left a fresh home and its ``cron``/``sessions``/``logs``/``memories``
    subdirectories at the default ``0o755`` instead of ``0o700``.
    """
    return [link for link in links if link == home or home in link.parents]


def _ensure_directory(path: Path, *, create: bool, secure: bool, home: Path) -> None:
    from hermes_cli.config import _secure_dir
    from hermes_constants import mkdir_under_hermes_home

    detail = ""
    try:
        links = _directory_links(path)
        detail = "; ".join(f"{link} -> {link.readlink()}" for link in links)
        # Never materialize a missing mount's target on the underlying disk.
        for link in links:
            if not link.is_dir():
                raise FileNotFoundError(f"Directory link is unavailable: {link}")
        if not path.is_dir():
            if create:
                mkdir_under_hermes_home(path)
            else:
                raise FileNotFoundError(f"Required directory does not exist: {path}")
        # The operator owns permissions beyond a link, including logs/curator.
        if secure and not _operator_owned_links(links, home):
            _secure_dir(path)
    except OSError as exc:
        raise HomeInitializationError(
            f"Cannot initialize Hermes directory {path}: {exc}. "
            + (f"Directory links: {detail}. " if detail else "")
            + "Check the directory/link target, mount availability and access permissions; "
            "restore the mount or repair the link before retrying. "
            "Hermes has not replaced the link or created its missing target."
        ) from exc


def _adopt_profile_incarnation(home: Path) -> None:
    """Give a pre-marker named home the generation marker its memo identity needs.

    Only TRIES the lifecycle lease: config loads run under gateway locks that
    lease holders take next, so a busy lease leaves this pass unmemoized (the
    next load retries) instead of waiting. Tombstoned homes are never backfilled.
    """
    from hermes_cli.profile_incarnation import ensure_profile_incarnation
    from hermes_cli.profile_lifecycle import profile_lifecycle_lease

    with suppress(OSError, RuntimeError), profile_lifecycle_lease(home, timeout=0):
        ensure_profile_incarnation(home)


def _hermes_home_identity(
    home: Path, *, named_profile: bool,
) -> tuple[int, int, str | None] | None:
    """The memo identity of one home skeleton pass: ``(st_dev, st_ino, incarnation)``.

    Named homes need a persisted generation: filesystems can reuse inode and ctime together.
    No ctime either: every create/import mints a fresh token, while every write in the profile
    root (config.yaml, auth.json, state.db -wal/-shm) moves ctime and would force a re-init.
    """
    try:
        value = home.stat()
        incarnation = None
        if named_profile:
            from hermes_cli.profile_incarnation import PROFILE_INCARNATION_FILENAME, read_incarnation_marker

            # Callers already derived named_profile, so read the marker directly (this runs on
            # every load). A tokenless home cannot prove a reusable cache identity until
            # initialize_home adopts a marker, which it does whenever the lifecycle lease is free.
            incarnation = read_incarnation_marker(home / PROFILE_INCARNATION_FILENAME)
            if incarnation is None:
                return None
    except OSError:
        return None
    return (value.st_dev, value.st_ino, incarnation)


def ensure_home(home: Path, subdirs: tuple[str, ...], ensured: dict[str, tuple[int, int, str | None]]) -> None:
    """``hermes_cli.config.ensure_hermes_home``: initialize *home* unless *ensured* already holds the
    identity of its live generation."""
    # Named profiles must be created explicitly. Check tombstones BEFORE the memo so a stale
    # empty shell cannot skip the deleted-profile guard.
    from hermes_constants import assert_named_profile_home_available, profile_deletion_marker_path
    marker = profile_deletion_marker_path(home)
    named_profile = marker is not None
    if named_profile:  # Only a named home can be unavailable; skip re-resolving every other one.
        assert_named_profile_home_available(home, marker=marker)
    current_identity = _hermes_home_identity(home, named_profile=named_profile)
    if current_identity is not None and ensured.get(str(home)) == current_identity:
        return
    initialize_home(home, subdirs, ensured)


def initialize_home(
    home: Path, subdirs: tuple[str, ...], ensured: dict[str, tuple[int, int, str | None]],
) -> None:
    from hermes_cli.config import _ensure_default_soul_md, is_managed
    from hermes_constants import assert_named_profile_home_available, profile_deletion_marker_path

    named_profile = profile_deletion_marker_path(home) is not None
    if named_profile and not home.is_dir():
        raise FileNotFoundError(f"Named profile home disappeared during initialization: {home}")
    if named_profile and _hermes_home_identity(home, named_profile=True) is None:
        _adopt_profile_incarnation(home)
    aliases = (home, home.resolve())
    initial_identities = {
        alias: _hermes_home_identity(alias, named_profile=True)
        for alias in aliases if profile_deletion_marker_path(alias) is not None
    }
    directory_home = _home_directory_path(home)
    managed = is_managed()
    old_umask = os.umask(0o007) if managed else None
    try:
        _ensure_directory(directory_home, create=not managed, secure=not managed, home=directory_home)
        required = ("cron", "sessions", "logs", "memories") if managed else subdirs
        for subdir in required:
            if named_profile:
                assert_named_profile_home_available(home)
            _ensure_directory(directory_home / subdir, create=not managed, secure=not managed, home=directory_home)
        if managed:
            if named_profile:
                assert_named_profile_home_available(home)
            _ensure_directory(directory_home / "logs" / "curator", create=True, secure=False, home=directory_home)
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
    # Never credit a generation that replaced the home during initialization.
    # A tokenless named home cannot prove completion for either spelling.
    for alias, before in initial_identities.items():
        if before is None or before != identities[alias]:
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
