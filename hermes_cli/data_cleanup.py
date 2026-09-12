"""Data-only deletion separates user state from the runtime that must survive."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat

from pm.filesystem import is_junction


@dataclass(frozen=True)
class DataRemovalPlan:
    home: Path
    remove: tuple[Path, ...]
    keep: tuple[Path, ...]


def plan_data_removal(home: Path, project: Path, userdata: Path | None = None) -> DataRemovalPlan:
    from hermes_constants import get_default_hermes_root
    from hermes_cli.runtime_paths import base_venv, installs_root, store_root
    from hermes_cli.steward import is_bundled_payload
    from tools.checkpoint_pruning import store_lock_path

    home = home.resolve()
    if home == Path(home.anchor) or home == Path.home().resolve():
        raise ValueError(f"refusing to erase an entire filesystem or user home: {home}")
    machine = get_default_hermes_root(home=home).resolve()
    protected = {
        project.resolve(), base_venv(project).resolve(), store_root(project).resolve(),
        installs_root().resolve(), machine / "tools", machine / "bin", machine / "cache",
        machine / "spawn-ledger.json", home / "gateway.lock", home / ".backup.lock",
        home / "runtime" / "active_sessions.lock", home / "cron" / ".tick.lock",
        store_lock_path(home / "checkpoints"),
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
    return removed, failed
