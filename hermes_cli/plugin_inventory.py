"""Installer-owned pristine payload inventory, not authentication or a code audit.

Capture while the selected repository is still available. Never infer ownership
from a used installation: unlisted files may be private runtime state.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

_RESERVED = {".git", ".install-metadata.json", ".hermes-catalog.json", ".hermes-snapshot.json"}


class PluginInventoryError(ValueError):
    """An inventory or its owned payload cannot be verified."""


def relative_file(name: str) -> PurePosixPath:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(c) < 32 for c in name)):
        raise PluginInventoryError("unsafe package path")
    parts = name.split("/")
    if any(p in {"", ".", ".."} or p.lower() in _RESERVED for p in parts):
        raise PluginInventoryError("unsafe or reserved package path")
    return PurePosixPath(name)


def checked_stat(path: Path):
    info = path.lstat()
    # Python 3.11 has no Path.is_junction. Inspect lstat before traversing
    # directories, including other Windows reparse-point kinds, not just links.
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or getattr(info, "st_reparse_tag", 0)):
        raise PluginInventoryError("symlink or reparse point in owned package path")
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise PluginInventoryError("special file in owned package path")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise PluginInventoryError("hard-linked owned package file")
    return info


def _ancestors(root: Path, relative: PurePosixPath) -> None:
    current = root
    for part in ("", *relative.parts[:-1]):
        current = current / part
        if not stat.S_ISDIR(checked_stat(current).st_mode):
            raise PluginInventoryError("owned package ancestor is not a directory")


def read_regular(path: Path) -> bytes:
    before = checked_stat(path)
    if not stat.S_ISREG(before.st_mode):
        raise PluginInventoryError("expected regular package file")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PluginInventoryError("owned package changed while opening")
        data = stream.read()
    after = checked_stat(path)
    keys = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(after, k) for k in keys):
        raise PluginInventoryError("owned package changed while reading")
    return data


def _git(root: Path, *args: str) -> bytes:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "--no-replace-objects", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}",
         "--git-dir=" + str(root / ".git"), "--work-tree=" + str(root), *args],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
    if result.returncode:
        raise PluginInventoryError("Git ownership verification failed")
    return result.stdout


def _checkout_install_inventory(clone_root: Path, package_root: Path) -> dict:
    """Read HEAD with Git's built-in checkout conversions, never process filters.

    The caller supplies a pristine checkout: attributes must not come from local
    edits. Git handles eol, ident and working-tree-encoding; raw object bytes are
    not the installed payload. Disable the path's named driver so verification
    cannot execute another repository-configured smudge/process command.
    """
    relative = package_root.relative_to(clone_root)
    if relative.parts:
        relative_file(relative.as_posix())
        _ancestors(clone_root, PurePosixPath(*relative.parts, "placeholder"))
    prefix = relative.as_posix() + "/" if relative.parts else ""
    entries = {}
    for entry in _git(clone_root, "ls-tree", "-rz", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        name = raw_name.decode("utf-8")
        if not name.startswith(prefix):
            continue
        name = name[len(prefix):]
        relative_file(name)
        mode, kind, oid = header.decode().split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise PluginInventoryError("tracked symlink, submodule or special file")
        repo_name = prefix + name
        driver = _git(clone_root, "check-attr", "-z", "filter", "--", repo_name).split(b"\0")[2].decode("utf-8")
        disabled = [option for setting in ("smudge=", "process=", "required=false")
                    for option in ("-c", f"filter.{driver}.{setting}")]
        data = _git(clone_root, *disabled, "cat-file", "--filters", "--path=" + repo_name, oid)
        git_mode = int(mode, 8) & 0o777
        # Git records executable intent, not the caller's umask. Capture the
        # pristine POSIX checkout without widening its permissions.
        _ancestors(package_root, relative_file(name))
        checkout_mode = stat.S_IMODE(checked_stat(package_root / name).st_mode)
        if os.name == "nt":
            owned_mode = git_mode
        else:
            if (not _valid_mode(checkout_mode)
                    or bool(checkout_mode & stat.S_IXUSR) != bool(git_mode & stat.S_IXUSR)
                    or (git_mode == 0o644 and checkout_mode & 0o111)):
                raise PluginInventoryError("checkout mode disagrees with Git executable intent")
            owned_mode = checkout_mode
        entries[name] = {"sha256": hashlib.sha256(data).hexdigest(), "mode": owned_mode}
    if not entries:
        raise PluginInventoryError("empty committed package inventory")
    return {"version": 1, "entries": entries}


def _valid_mode(mode: int) -> bool:
    # Owner-readable/writable regular permissions only; never special bits or
    # execute permission granted solely to group/other.
    return (type(mode) is int and 0 <= mode <= 0o777 and mode & 0o600 == 0o600
            and (bool(mode & stat.S_IXUSR) or not mode & 0o111))


def _entries(inventory: dict) -> dict:
    if (not isinstance(inventory, dict) or set(inventory) != {"version", "entries"}
            or type(inventory["version"]) is not int or inventory["version"] != 1
            or not isinstance(inventory["entries"], dict) or not inventory["entries"]):
        raise PluginInventoryError("unsupported or malformed file inventory")
    for name, record in inventory["entries"].items():
        relative_file(name)
        if (not isinstance(record, dict) or set(record) != {"sha256", "mode"}
                or not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or not _valid_mode(record["mode"])):
            raise PluginInventoryError("malformed owned file digest or mode")
    return inventory["entries"]


def _owned_bytes(source: Path, name: str, record: dict) -> bytes:
    relative = relative_file(name)
    _ancestors(source, relative)
    path = source / relative
    data = read_regular(path)
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise PluginInventoryError("owned package file differs from pristine inventory")
    _check_owned_mode(path, record["mode"])
    return data


def _check_owned_mode(path: Path, expected: int) -> None:
    info = checked_stat(path)
    if not stat.S_ISREG(info.st_mode):
        raise PluginInventoryError("expected regular package file")
    mode = stat.S_IMODE(info.st_mode)
    # Windows retains Git intent in metadata but exposes only readonly/writable.
    mode_matches = bool(mode & stat.S_IWRITE) if os.name == "nt" else mode == expected
    if not mode_matches:
        raise PluginInventoryError("owned package mode differs from pristine inventory")


def validate_install_modes(source: Path, inventory: dict) -> None:
    """Prevent a staged update from adopting chmod drift that Git cannot stash."""
    for name, record in _entries(inventory).items():
        relative = relative_file(name)
        _ancestors(source, relative)
        _check_owned_mode(source / relative, record["mode"])


def validate_install_inventory(source: Path, inventory: dict) -> None:
    for name, record in _entries(inventory).items():
        _owned_bytes(source, name, record)


def capture_install_inventory(clone_root: Path, package_root: Path) -> dict:
    """Capture pristine checkout BEFORE scan/examples/local-edit restoration."""
    inventory = _checkout_install_inventory(clone_root, package_root)
    validate_install_inventory(package_root, inventory)
    return inventory


def copy_install_inventory(source: Path, destination: Path, inventory: dict) -> None:
    """Copy only verified owned files into a new private directory, never links."""
    entries = _entries(inventory)
    destination.mkdir(parents=True, exist_ok=False)
    for name, record in entries.items():
        data = _owned_bytes(source, name, record)
        target = destination / relative_file(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        _ancestors(destination, relative_file(name))
        with target.open("xb") as stream:
            stream.write(data)
        target.chmod(stat.S_IREAD | stat.S_IWRITE if os.name == "nt" else record["mode"])
    validate_install_inventory(source, inventory)
