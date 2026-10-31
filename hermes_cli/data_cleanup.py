"""Data-only deletion separates user state from the runtime that must survive."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat

from pm.filesystem import is_junction


@dataclass(frozen=True)
class DataRemovalPlan:
    home: Path
    remove: tuple[Path, ...]
    keep: tuple[Path, ...]


# State Hermes writes on purpose. Launching any ``hermes`` command seeds SOUL.md and the
# skeleton dirs into whatever HERMES_HOME names, and ``.env`` is common in project dirs,
# so neither proves a directory is a Hermes home.
_HOME_MARKERS = ("config.yaml", "state.db", "auth.json")
# Left by ``remove_data`` in the (already vetted) home it emptied: the markers are gone then,
# and the follow-up code uninstall must still recognise the directory Hermes itself cleaned.
_DATA_REMOVED_RECORD = ".hermes-data-removed"


def guard_data_removal_home(home: Path, project: Path) -> Path:
    """Return the canonical home, or refuse a home whose removal reaches data Hermes does not own.

    Refused: a filesystem root, the user's home or any ancestor of it, a home inside the
    install directory, and an existing directory carrying no Hermes marker file. A link
    anywhere in the path is followed: the checks run on the directory that would be emptied.
    """
    resolved_home = home.resolve()
    if resolved_home == Path(resolved_home.anchor) or Path.home().resolve().is_relative_to(resolved_home):
        raise ValueError(f"refusing to remove {resolved_home}: it is a filesystem root or contains your home folder")
    if resolved_home.is_relative_to(project.resolve()):
        raise ValueError(f"refusing to remove {resolved_home}: it is inside the installation directory")
    if resolved_home.is_dir() and not any(
            (resolved_home / name).exists() for name in (*_HOME_MARKERS, _DATA_REMOVED_RECORD)):
        raise ValueError(
            f"refusing to remove {resolved_home}: it has no {', '.join(_HOME_MARKERS)}, "
            "so it is not a Hermes home"
        )
    return resolved_home


def guard_whole_home_removal(home: Path, project: Path) -> Path:
    """:func:`guard_data_removal_home` for a wipe of the whole tree, which also refuses a home
    that is itself a link or junction: its target is a directory the user placed elsewhere,
    and ``rmtree`` refuses links, so the wipe would fail only after the code was gone.
    """
    entry = Path(os.path.abspath(home))
    if entry.is_symlink() or is_junction(entry):
        raise ValueError(
            f"refusing to remove {entry}: it is a link to {entry.resolve()}. "
            "Set HERMES_HOME to the real directory to remove it."
        )
    return guard_data_removal_home(home, project)


def plan_data_removal(home: Path, project: Path, userdata: Path | None = None) -> DataRemovalPlan:
    from hermes_constants import get_default_hermes_root
    from pm.environments import base_venv, installs_root, store_root
    from hermes_cli.steward import is_bundled_payload
    from tools.checkpoint_pruning import store_lock_path

    home = guard_data_removal_home(home, project)
    machine = get_default_hermes_root(home=home).resolve()
    protected = {
        project.resolve(), base_venv(project).resolve(), store_root(project).resolve(),
        installs_root().resolve(), machine / "tools", machine / "bin", machine / "cache",
        machine / "spawn-ledger.json", machine / "spawn-ledger.json.lock", home / "gateway.lock", home / ".backup.lock",
        home / "runtime" / "active_sessions.lock", home / "cron" / ".tick.lock",
        store_lock_path(home / "checkpoints"), home / _DATA_REMOVED_RECORD,
    }
    if home == machine:
        protected.add(machine / "profiles")
    if is_bundled_payload(project):
        from hermes_cli.bundled_app import NotBundledApp, resolve_bundle_layout

        try:
            protected.add(resolve_bundle_layout(project).app_root)
        except NotBundledApp as exc:
            raise ValueError(str(exc)) from exc
    # The canonical source installation may coexist with the invoking package.
    canonical = home / "hermes-agent"
    if (canonical / "hermes_cli" / "__init__.py").is_file():
        protected.add(canonical)
    remove: list[Path] = []
    keep: set[Path] = set()

    def visit(path: Path) -> None:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return
        resolved = path.resolve()
        if any(path == root or resolved == root or path.is_relative_to(root) for root in protected):
            keep.add(path)
        elif path.is_symlink() or is_junction(path):
            remove.append(path)  # Remove the directory entry, never its target.
        elif any(root.is_relative_to(path) for root in protected):
            if stat.S_ISDIR(mode):
                for child in sorted(path.iterdir()):
                    visit(child)
            else:
                keep.add(path)
        else:
            remove.append(path)

    if home.is_dir():
        for child in sorted(home.iterdir()):
            visit(child)
    if home != machine:
        userdata = None  # Desktop preferences belong to the app, not a named profile.
    if userdata is not None and not userdata.resolve().is_relative_to(home):
        if userdata.resolve() == Path.home().resolve() or home.is_relative_to(userdata.resolve()):
            raise ValueError(f"desktop data directory must not contain the Hermes or user home: {userdata}")
        visit(userdata.parent.resolve() / userdata.name)
    return DataRemovalPlan(home, tuple(remove), tuple(sorted(keep)))


def remove_data(plan: DataRemovalPlan) -> tuple[list[Path], list[tuple[Path, str]]]:
    import shutil

    removed: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for path in plan.remove:
        try:
            if path.parent.resolve() != path.parent:
                raise OSError("parent directory changed after confirmation; refusing to follow a link")
            mode = path.lstat().st_mode
            if path.is_symlink() or not stat.S_ISDIR(mode):
                path.unlink()
            elif is_junction(path):
                path.rmdir()
            else:
                shutil.rmtree(path)
            removed.append(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            failed.append((path, str(exc)))
    record = plan.home / _DATA_REMOVED_RECORD
    try:
        if record.is_symlink():
            record.unlink()  # never write through a link into a file elsewhere
        record.write_text("Hermes removed this home's user data; installed code may remain.\n", encoding="utf-8")
    except OSError as exc:
        failed.append((record, f"could not record that this is a Hermes home: {exc}"))
    return removed, failed
