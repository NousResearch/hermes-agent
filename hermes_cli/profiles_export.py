"""Profile export / import: tar.gz staging, credential exclusion, secret scrubbing.

Split out of :mod:`hermes_cli.profiles` (#79980). Facade helpers are imported INSIDE each
function so ``monkeypatch.setattr(hermes_cli.profiles, "<helper>", ...)`` stays the seam
the extraction inherited (AGENTS.md: "patch where production reads").
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

from hermes_cli.archive_safe import archive_root_dirs, make_targz, normalize_archive_parts, safe_extract_targz


# Allow-list for ``export_profile("default")``: when HERMES_HOME equals the cwd
# (Docker/custom deployments) the default home holds arbitrary user files that must NOT
# be bundled. Only known Hermes profile artifacts at the root survive; sensitive runtime
# infrastructure (``state.db``, ``logs/``, ``auth.*``, other profiles) is deliberately
# absent so the export stays a portable, credential-free snapshot. Add new artifacts here
# when introduced in ``hermes_constants``.
# See #58394.
_DEFAULT_EXPORT_INCLUDE_ROOT = frozenset({
    # Configuration / persona
    "config.yaml", "SOUL.md", "MEMORY.md", "USER.md", "todo.json",
    "system_prompt.md", "AGENTS.md", "CLAUDE.md", ".cursorrules",
    # Desktop appearance overlay (written/applied by the desktop app's export/import).
    "desktop.json",
    # User-facing skill, cron, and session artifacts
    "skills", "cron", "scripts", "sessions",
    # Plugin / memory surfaces (per-profile overrides live here)
    "plugins", "memories", "knowledge", "preferences",
})


# Export / Import

def _inside_git_checkout(path: Path) -> bool:
    """True when *path* lies inside a Git checkout. Walks the path's OWN resolved ancestry
    (not cwd) so the check holds when HERMES_HOME sits in a checkout but the process runs
    elsewhere (cron, service manager). Resolution failure reports True (fail closed)."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):  # RuntimeError: symlink loops on Python <= 3.12
        return True
    return any((candidate / ".git").exists() for candidate in (resolved, *resolved.parents))


def _profile_export_directory() -> Path:
    """Choose an export directory that cannot become source-tree input."""
    from hermes_cli.profiles import _get_default_hermes_home
    import tempfile
    export_dir = _get_default_hermes_home() / "profile-exports"
    if not _inside_git_checkout(export_dir):
        return export_dir

    # A custom deployment may point HERMES_HOME at its source checkout: use a sibling store,
    # falling back to the OS temp dir only when the user's home itself is a checkout (dotfiles
    # repo). Per-uid temp name: a fixed /tmp/hermes-profile-exports is a predictable shared
    # path another local user could pre-create (or symlink) first.
    uid_suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    candidates = (
        Path.home() / ".hermes-profile-exports", Path(tempfile.gettempdir()) / f"hermes-profile-exports{uid_suffix}"
    )
    for candidate in candidates:
        if not _inside_git_checkout(candidate):
            return candidate
    # Fail closed: writing a secret-bearing archive into a source tree is the incident this
    # helper prevents; a stderr warning would not stop a scripted export.
    raise ValueError(
        # See #92457.
        "No safe automatic export destination: every candidate directory is "
        "inside a Git checkout. Provide an explicit output path outside the "
        "checkout (CLI: -o /path/outside/repo/profile.tar.gz)."
    )


def get_profile_export_path(name: str, *, timestamp: Optional[str] = None) -> Path:
    """Managed destination for an export with no explicit output — outside the cwd and every
    profile, since a ``<name>.tar.gz`` default in a source checkout got committed by accident."""
    from hermes_cli.profiles import _canon_valid
    canon = _canon_valid(name)
    export_dir = _profile_export_directory()
    export_dir.mkdir(parents=True, exist_ok=True)
    # exist_ok=True silently accepts a directory (or symlink) another local user pre-created
    # at a predictable path; refuse to write a secret-bearing archive anywhere we don't own.
    if export_dir.is_symlink():
        raise ValueError(
            f"Export directory {export_dir} is a symlink; refusing to write "
            "a profile archive through it. Provide an explicit output path."
        )
    if hasattr(os, "getuid") and export_dir.stat().st_uid != os.getuid():
        raise ValueError(
            f"Export directory {export_dir} is owned by another user; "
            "refusing to write a profile archive there. Provide an explicit output path."
        )
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    return export_dir / f"{canon}-{stamp}.tar.gz"


def _default_export_ignore(root_dir: Path):
    """copytree ignore for the default-profile export: root-level allow-list
    (``_DEFAULT_EXPORT_INCLUDE_ROOT``) plus universal exclusions. Surviving text files are
    then force-redacted by :func:`_scrub_export_secrets`.

    * **Root-level allow-list** — only entries whose name appears in ``_DEFAULT_EXPORT_INCLUDE_ROOT``
    survive. Everything else (such as an unrelated ``x11-dev/`` directory in a Docker deployment where
    HERMES_HOME equals the cwd) is excluded. Blacklisting was tried first and proved unable to anticipate
    every non-Hermes file the user may have lying alongside HERMES_HOME (#58394). * **Universal exclusions
    at any depth** — ``__pycache__``, sockets and other special files, temp files
    (:func:`_non_exportable_entries`); plus npm lockfiles, which may appear at the root.
    """
    from hermes_cli.profiles import _non_exportable_entries

    def _ignore(directory: str, contents: list) -> set:
        # Universal exclusions (any depth) plus npm lockfiles that can appear at root.
        ignored = _non_exportable_entries(directory, contents)
        ignored.update({"package.json", "package-lock.json"} & set(contents))
        if Path(directory) == root_dir:
            ignored.update(entry for entry in contents if entry not in _DEFAULT_EXPORT_INCLUDE_ROOT)
        return ignored

    return _ignore


# Credential files dropped from named-profile exports.
_EXPORT_CREDENTIAL_FILES = frozenset({"auth.json", ".env"})

# Text/config suffixes secret-scrubbed on export; binary DBs, images etc. are left alone.
_EXPORT_REDACT_SUFFIXES = frozenset({
    ".md", ".txt", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".ini", ".cfg", ".conf", ".py", ".sh",
    ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".xml", ".csv",
})
# ``Path(".cursorrules").suffix`` is "" — name-match; ``*.env.example`` uses endswith.
_EXPORT_REDACT_NAMES = frozenset({".cursorrules"})


def _should_redact_export_file(path: Path) -> bool:
    name = path.name
    return (
        name in _EXPORT_REDACT_NAMES
        or name.lower().endswith(".env.example")
        or path.suffix.lower() in _EXPORT_REDACT_SUFFIXES
    )


def _scrub_export_secrets(staged: Path) -> None:
    """Force-redact secret-shaped strings in a staged export tree (same pass as ``hermes
    sessions export --redact``). Runs on the staged copy only; symlinks to text files are
    materialized when content changes so redaction never follows a link back into the source."""
    from agent.redact import redact_sensitive_text
    for path in staged.rglob("*"):
        try:
            is_link = path.is_symlink()
            if not path.is_file():  # broken links, symlinked dirs, non-files
                continue
        except OSError:
            continue
        if not _should_redact_export_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        redacted = redact_sensitive_text(text, force=True)
        if redacted == text:
            continue
        if is_link:
            path.unlink()
        path.write_text(redacted, encoding="utf-8")


def export_profile(name: str, output_path: str, extra_files: Optional[Dict[str, str]] = None) -> Path:
    """Export a profile to a tar.gz archive; credential files are excluded and staged text is
    force-redacted first. Returns the output file path."""
    from hermes_cli.profiles import _existing_profile_dir, _non_exportable_entries
    import tempfile
    canon, profile_dir = _existing_profile_dir(name)
    # Archive base name without extension (.tar.gz appended by the writer).
    base = str(Path(output_path)).removesuffix(".tar.gz").removesuffix(".tgz")

    # The default profile IS ~/.hermes (dir name ".hermes"), so both paths stage a filtered
    # copy under a temp dir named after the canonical id: root allow-list for default,
    # credential exclusion for named profiles.
    def _ignore_credentials(directory: str, contents: list) -> set:
        ignored = _non_exportable_entries(directory, contents)
        ignored.update(_EXPORT_CREDENTIAL_FILES & set(contents))
        return ignored

    ignore = _default_export_ignore(profile_dir) if canon == "default" else _ignore_credentials
    with tempfile.TemporaryDirectory() as tmpdir:
        staged = Path(tmpdir) / canon
        shutil.copytree(profile_dir, staged, symlinks=True, ignore=ignore)
        for rel, content in (extra_files or {}).items():
            target = staged.joinpath(*normalize_archive_parts(rel))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        _scrub_export_secrets(staged)
        return Path(make_targz(base, tmpdir, canon))


def import_profile(archive_path: str, name: Optional[str] = None) -> Path:
    """Import a profile from a tar.gz archive."""
    from hermes_cli.profiles import _canon_valid, _get_profiles_root, _profile_exists_error, get_profile_dir
    import tempfile
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    top_dirs = archive_root_dirs(archive)
    archive_root = top_dirs.pop() if len(top_dirs) == 1 else None
    inferred_name = name or archive_root
    if not inferred_name:
        raise ValueError(
            "Cannot determine profile name from archive. "
            "Specify it explicitly: hermes profile import <archive> --name <name>"
        )
    if archive_root is None:
        raise ValueError("Profile archive must contain exactly one top-level directory.")

    # Default-profile archives have "default/" at top level; importing as "default" would
    # target ~/.hermes itself.
    canon = _canon_valid(inferred_name)
    if canon == "default":
        raise ValueError(
            "Cannot import as 'default' — that is the built-in root profile (~/.hermes). "
            "Specify a different name: hermes profile import <archive> --name <name>"
        )
    profile_dir = get_profile_dir(canon)
    if profile_dir.exists():
        raise _profile_exists_error(canon)
    _get_profiles_root().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hermes_profile_import_") as tmpdir:
        staging_root = Path(tmpdir)
        safe_extract_targz(archive, staging_root)
        extracted = staging_root / archive_root
        if not extracted.is_dir():
            raise ValueError(f"Profile archive root is missing or invalid: {archive_root}")
        final_source = extracted
        if archive_root != canon:
            final_source = staging_root / canon
            extracted.rename(final_source)
        shutil.move(str(final_source), str(profile_dir))
    return profile_dir
