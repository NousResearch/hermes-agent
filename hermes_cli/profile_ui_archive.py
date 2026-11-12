"""Capture profile UI metadata and group sync state as one coherent archive unit."""
from pathlib import Path
import os

from tui_gateway.group_chat_storage import GROUP_CHAT_STATE


_PAIR = (Path("profile.yaml"), GROUP_CHAT_STATE)


def _optional_bytes(path: Path) -> bytes | None:
    if path.is_symlink() or (path.name == GROUP_CHAT_STATE.name and path.parent.is_symlink()):
        raise OSError("Cannot snapshot symlinked profile UI state")
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def capture_profile_ui_pair(home: Path) -> dict[Path, bytes]:
    """Capture immutable bytes, retrying if a cross-file commit crosses the metadata read.

    Never create writer locks or hold them during a potentially large archive operation.
    Preserve a pending journal verbatim: it is part of the coherent snapshot.
    """
    for _ in range(3):
        before = _optional_bytes(home / GROUP_CHAT_STATE)
        metadata = _optional_bytes(home / "profile.yaml")
        after = _optional_bytes(home / GROUP_CHAT_STATE)
        if before == after:
            return {path: data for path, data in zip(_PAIR, (metadata, after)) if data is not None}
    raise OSError("Profile UI state changed repeatedly during archive capture")


def stage_profile_ui_pair(home: Path, staged: Path) -> None:
    """Replace copytree's potentially torn pair before export redaction and publication."""
    pair = capture_profile_ui_pair(home)
    if (staged / GROUP_CHAT_STATE.parent).is_symlink():
        raise OSError("Cannot stage profile UI state through a symlink")
    for relative in _PAIR:
        target = staged / relative
        target.unlink(missing_ok=True)
        if relative in pair:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(pair[relative])


def named_profile_export_ignore(profile_dir: Path):
    """The named-profile export credential policy, unchanged by coherent UI capture."""
    from hermes_cli.profiles import (
        PM_RUNTIME_ROOT_DIRS, _export_credential_entries, _non_exportable_entries, profile_path_is_private,
    )

    def ignore(directory: str, contents: list) -> set:
        ignored = _non_exportable_entries(directory, contents)
        ignored.update(_export_credential_entries(directory, contents))
        rel = Path(directory).relative_to(profile_dir).parts
        ignored.update(e for e in contents if profile_path_is_private((*rel, e)))
        if Path(directory) == profile_dir:
            ignored |= PM_RUNTIME_ROOT_DIRS & set(contents)
        return ignored

    return ignore


def _profile_ui_home(absolute: Path, relative: Path):
    parts = relative.parts
    prefix = Path(*parts[:2]) if len(parts) >= 3 and parts[0] == "profiles" else Path()
    tail = relative.relative_to(prefix)
    if tail not in (*_PAIR, Path("config.yaml")):
        return None
    return prefix, absolute.parents[len(tail.parts) - 1]


def capture_backup_ui_pairs(files: list[tuple[Path, Path]]) -> dict[Path, bytes]:
    """Replace the scanned pair entries with immutable per-profile snapshots.

    Update the file inventory before callers count it: a group file first published
    between the directory walk and capture must be included along with its metadata.
    """
    homes = {}
    for absolute, relative in files:
        located = _profile_ui_home(absolute, relative)
        if located is not None:
            prefix, home = located
            homes[prefix] = home
    snapshots = {}
    captured_entries = []
    paired_paths = set()
    for prefix, home in homes.items():
        paired_paths.update(prefix / path for path in _PAIR)
        for relative, data in capture_profile_ui_pair(home).items():
            archived = prefix / relative
            snapshots[archived] = data
            captured_entries.append((home / relative, archived))
    files[:] = [(absolute, relative) for absolute, relative in files if relative not in paired_paths]
    files.extend(captured_entries)
    return snapshots
