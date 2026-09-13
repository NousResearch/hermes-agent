"""Distribution snapshots, PM facts and portable link sealing."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def snapshot(repo: Path, ref: str, destination: Path) -> None:
    """Archive a resolved git revision without carrying checkout metadata."""
    repo, destination = repo.resolve(), destination.resolve()
    if repo == destination or repo.is_relative_to(destination):
        raise ValueError("the snapshot destination must not contain the source checkout")
    with tempfile.TemporaryDirectory(prefix="hermes-archive-") as temp:
        archive = Path(temp) / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), ref], cwd=repo, check=True)
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")


def record_tools(root: Path, lock_path: Path, target: str, entries: dict[str, str]) -> None:
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package
    from pm.store import tree_digest

    store = root / "tools"
    facts, lock = Facts(store / "facts.json"), Lockfile(lock_path)
    for name, entry_name in entries.items():
        entry = store / entry_name
        version, artifacts = lock.version(name), lock.artifacts(name, target)
        if not entry.is_dir() or not version or not artifacts:
            raise ValueError(f"incomplete payload tool: {name}")
        facts.record(name, version, entry_name, get_package(name).env(entry, target), store,
                     target=target, artifacts=[a["sha256"] for a in artifacts], digest=tree_digest(entry))


def rehash_tools(root: Path) -> int:
    """Record final tool bytes before the enclosing package is signed."""
    from pm.lock import Facts

    store = root / "tools"
    return Facts(store / "facts.json", strict=True).refresh_digests(store)


def seal_pm_runtime(root: Path, python: Path) -> dict:
    """Record a resident PM runtime without Windows' CWD-bound redirector.

    Sealed workers execute the base interpreter with -I -S and add only the
    recorded site directory. Its paths remain valid after the payload moves.
    """
    root, python = root.resolve(), python.resolve()
    if not python.is_relative_to(root) or not python.is_file():
        raise ValueError(f"PM interpreter must belong to the payload: {python}")
    runtime = root / "pm-runtime"
    sites = list(runtime.glob("lib/python*/site-packages")) + list(runtime.glob("Lib/site-packages"))
    if len(sites) != 1:
        raise ValueError(f"PM dependency directory missing or ambiguous: {runtime}")
    marker = {
        "python": Path(os.path.relpath(python, runtime)).as_posix(),
        "sitePackages": sites[0].relative_to(runtime).as_posix(),
    }
    cfg = runtime / "pyvenv.cfg"
    lines = cfg.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.partition("=")[0].strip() not in
             {"home", "executable", "base-executable", "base-prefix", "base-exec-prefix", "command"}]
    lines.insert(0, f"home = {os.path.relpath(python.parent, runtime)}")
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # A sealed runtime is not activated, and copied Windows redirectors cannot
    # follow its relative home from arbitrary working directories.
    bindir = runtime / ("Scripts" if os.name == "nt" else "bin")
    for entry in bindir.iterdir():
        if os.name == "nt" or not entry.is_symlink():
            if entry.is_file():
                entry.unlink()
    _relativize_bin_links(root, runtime / "bin")
    (runtime / "pm-runtime.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return marker


def relativize_links(root: Path) -> int:
    """Only dependency-venv links move; framework links belong to codesign."""
    root = root.resolve()
    return sum(_relativize_bin_links(root, root / name / "bin") for name in ("venv", "pm-runtime"))


def _relativize_bin_links(root: Path, directory: Path) -> int:
    count = 0
    if not directory.is_dir():
        return count
    for link in directory.iterdir():
        if not link.is_symlink():
            continue
        target = os.readlink(link)
        if not os.path.isabs(target):
            # Keep sibling chains intact; their absolute store link is rewritten separately.
            if not Path(os.path.abspath(directory / target)).is_relative_to(root):
                raise ValueError(f"link escapes payload: {link} -> {target}")
            continue
        resolved = (directory / target).resolve()
        if not resolved.is_relative_to(root):
            parts = Path(target).parts
            if "tools" not in parts:
                raise ValueError(f"link escapes payload: {link} -> {target}")
            resolved = root.joinpath(*parts[parts.index("tools"):]).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            raise ValueError(f"missing payload link target: {link} -> {target}")
        relative = os.path.relpath(resolved, directory)
        if relative != target:
            link.unlink()
            link.symlink_to(relative)
            count += 1
    for link in directory.iterdir():
        if link.is_symlink() and (not link.resolve().is_relative_to(root) or not link.exists()):
            raise ValueError(f"invalid relative payload link: {link}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["relocate", "rehash"])
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    if args.action == "rehash":
        print(f"rehashed {rehash_tools(args.payload)} payload tools")
    else:
        relativize_links(args.payload)


if __name__ == "__main__":
    main()
