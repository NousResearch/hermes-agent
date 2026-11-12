"""What a dependency generation copies of each plugin member, the key it shares, and the copy.

A generation snapshots every member (``pm/workspace.py::_workspace_member``). Three things made
each one cost gigabytes (#154442): a member was keyed by its path, so a profile clone's
byte-identical plugin copy was a new member and so a new generation; the snapshot of a plugin
checkout carried its git-ignored data; and the copy wrote new blocks where the filesystem can
share them.
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

from pm.plugin_declarations import read_python_declaration


def member_files(plugin_dir: Path) -> list[str]:
    """The build inputs below *plugin_dir*: sorted POSIX paths relative to it.

    A plugin dir that is a git checkout (its own ``.git``) gives what git keeps: tracked files and
    the untracked ones its ignore rules (``.gitignore``, ``.git/info/exclude``) admit. A
    development checkout's ignored notes, renders and worktrees are not build inputs. Any other
    dir, or a git that cannot answer, is walked. Either way the PM exclusions apply, and a symlink
    is one entry that is never followed.
    """
    from pm.workspace import _member_ignored

    listed = _git_listed(plugin_dir) if (plugin_dir / ".git").exists() else None
    files: set[str] = set()
    for relative in [""] if listed is None else listed:
        parts = Path(relative).parts
        if any(_member_ignored(plugin_dir, [part]) for part in parts):
            continue
        path = plugin_dir / relative
        if relative and (path.is_symlink() or path.is_file()):
            files.add(Path(relative).as_posix())
        elif path.is_dir():  # the walk, or a submodule or nested repository git lists as one entry
            files.update(_walked(plugin_dir, path))
    return sorted(files)


def _git_listed(plugin_dir: Path) -> list[str] | None:
    from hermes_cli._subprocess_compat import bounded_probe_run, noninteractive_repo_git_env

    env = noninteractive_repo_git_env(plugin_dir)
    if env is None:
        return None
    proc = bounded_probe_run(
        ["git", "-C", str(plugin_dir), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        timeout=60, env=env, errors="surrogateescape",
    )
    if proc is None or proc.returncode != 0:
        return None
    return [entry.rstrip("/") for entry in proc.stdout.split("\0") if entry]


def _walked(plugin_dir: Path, top: Path) -> Iterator[str]:
    from pm.workspace import _member_ignored

    for directory, dirs, files in os.walk(top):
        ignored = set(_member_ignored(directory, dirs))
        dirs[:] = [name for name in dirs if name not in ignored]
        links = [name for name in dirs if os.path.islink(os.path.join(directory, name))]
        for name in [*links, *(set(files) - set(_member_ignored(directory, files)))]:
            yield (Path(directory) / name).relative_to(plugin_dir).as_posix()


def uv_path_specs(document: dict) -> Iterator[dict]:
    """Every ``tool.uv.sources`` entry of *document* that names a ``path``."""
    for sources in document.get("tool", {}).get("uv", {}).get("sources", {}).values():
        for spec in sources if isinstance(sources, list) else [sources]:
            if isinstance(spec, dict) and "path" in spec:
                yield spec


def external_source(spec: dict, plugin_dir: Path, identity: Path) -> Path | None:
    """The out-of-tree path a relative uv ``path`` source names, resolved from the installed
    *identity*; None for an absolute path or a tree the member snapshot holds itself."""
    relative = Path(spec["path"])
    if relative.is_absolute() or (plugin_dir / relative).resolve().is_relative_to(plugin_dir.resolve()):
        return None
    return (identity / relative).resolve()


def member_digest(identity: Path, plugin_dir: Path) -> str:
    """Hash what a member's snapshot holds: its declaration files, its build inputs, and the
    out-of-tree paths its uv sources name. Where it is installed is left out, so byte-identical
    copies of one plugin in several profiles share a digest and are one member."""
    import tomllib

    h = hashlib.sha256()
    declaration = read_python_declaration(plugin_dir)
    for source in declaration.files:
        h.update(source.name.encode("utf-8"))
        h.update(source.read_bytes())
        h.update(b"\0")
    if declaration.pyproject is not None:
        for relative in member_files(plugin_dir):
            path = plugin_dir / relative
            h.update(relative.encode("utf-8", "surrogateescape"))
            h.update(b"\0")
            h.update(os.readlink(path).encode("utf-8", "surrogateescape") if path.is_symlink() else path.read_bytes())
            h.update(b"\0")
        document = tomllib.loads(declaration.pyproject.read_text(encoding="utf-8-sig"))
        for spec in uv_path_specs(document):
            external = external_source(spec, plugin_dir, identity)
            if external is not None:
                h.update(external.as_posix().encode("utf-8", "surrogateescape"))
                h.update(b"\0")
    return h.hexdigest()


def copy_member(plugin_dir: Path, member: Path) -> None:
    """Snapshot the build inputs of *plugin_dir* into the fresh dir *member*."""
    member.mkdir(parents=True)
    for relative in member_files(plugin_dir):
        source, target = plugin_dir / relative, member / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            os.symlink(os.readlink(source), target)
        else:
            clone_file(source, target)


def clone_file(source, target):
    """``shutil.copy2`` that shares the data blocks where the filesystem can: ``clonefile(2)`` on
    APFS, ``FICLONE`` on Btrfs and XFS. Another volume or a filesystem without clones gets a copy."""
    if sys.platform == "darwin":
        cloned = _clonefile()(os.fsencode(source), os.fsencode(target), 0) == 0
    elif sys.platform.startswith("linux"):
        cloned = _ficlone(source, target)
    else:
        cloned = False
    if not cloned:
        shutil.copy2(source, target)
    return target


@functools.cache
def _clonefile():
    import ctypes

    # Like copy2, clonefile(2) also carries the mode, times, extended attributes and ACLs.
    function = ctypes.CDLL(None, use_errno=True).clonefile
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
    function.restype = ctypes.c_int
    return function


_FICLONE = 0x40049409  # _IOW(0x94, 9, int), linux/fs.h


def _ficlone(source, target) -> bool:
    import fcntl

    with open(source, "rb") as reader, open(target, "wb") as writer:
        try:
            fcntl.ioctl(writer.fileno(), _FICLONE, reader.fileno())
        except OSError:
            return False  # the empty target is overwritten by the copy
    shutil.copystat(source, target)
    return True
