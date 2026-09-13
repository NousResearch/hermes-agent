"""Capture distribution files and bind publication to the approved bytes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile


@dataclass(frozen=True)
class SourceSnapshot:
    mode: int
    digest: bytes | None = None
    children: dict[str, SourceSnapshot] | None = None


def _check_mode(source, expected: SourceSnapshot | None) -> None:
    if expected is not None and source.mode != expected.mode:
        raise OSError("Distribution source mode changed after planning")


def _file_chunks(source, expected: SourceSnapshot | None = None):
    if source.is_dir:
        raise OSError("Distribution source is not a regular file")
    _check_mode(source, expected)
    digest = hashlib.sha256() if expected is not None else None
    while chunk := source.read(1024 * 1024):
        if digest is not None:
            digest.update(chunk)
        yield chunk
    if digest is not None and digest.digest() != expected.digest:
        raise OSError("Distribution source content changed after planning")


def read_source_bytes(source, *, expected: SourceSnapshot | None = None) -> bytes:
    return b"".join(_file_chunks(source, expected))


def seal_source_tree(source) -> SourceSnapshot:
    """Seal bytes as well as metadata: size and mtime can survive a replacement."""
    if source.is_dir:
        children = {}
        for name in source.names():
            with source.child(name) as entry:
                children[name] = seal_source_tree(entry)
        return SourceSnapshot(source.mode, children=children)
    digest = hashlib.sha256()
    for chunk in _file_chunks(source):
        digest.update(chunk)
    return SourceSnapshot(source.mode, digest=digest.digest())


def copy_source_file(
    source, destination: Path, *, expected_bytes: bytes | None = None,
    expected: SourceSnapshot | None = None,
) -> None:
    temporary = None
    try:
        # A sibling temporary keeps unapproved bytes away from an existing file,
        # and replace never follows a symlink at the final destination component.
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".hermes-dist-", dir=destination.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            if expected_bytes is not None:
                content = read_source_bytes(source, expected=expected)
                if content != expected_bytes:
                    raise OSError("Distribution manifest changed during staging")
                output.write(content)
            else:
                for chunk in _file_chunks(source, expected):
                    output.write(chunk)
        os.chmod(temporary, stat.S_IMODE(source.mode))
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError:
                # A read-only source also makes the temporary read-only on Windows.
                temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
                temporary.unlink(missing_ok=True)


def copy_source_tree(
    source, destination: Path, manifest_name: str = "", manifest_bytes: bytes = b"",
    *, expected: SourceSnapshot | None = None,
) -> None:
    if not source.is_dir:
        raise OSError("Distribution source is not a directory")
    _check_mode(source, expected)
    names = source.names()
    if expected is not None and (
        expected.children is None or set(names) != set(expected.children)
    ):
        raise OSError("Distribution source entries changed after planning")
    for name in names:
        target = destination / name
        with source.child(name) as entry:
            child_expected = expected.children[name] if expected is not None else None
            if entry.is_dir:
                # Populate writable scratch directories before restoring source modes.
                target.mkdir(mode=0o700)
                copy_source_tree(entry, target, expected=child_expected)
            else:
                approved_bytes = (
                    manifest_bytes
                    if manifest_name and os.path.normcase(name) == os.path.normcase(manifest_name)
                    else None
                )
                copy_source_file(entry, target, expected_bytes=approved_bytes, expected=child_expected)
    os.chmod(destination, stat.S_IMODE(source.mode))


@contextmanager
def owned_stage(path: Path):
    # A caller's pre-existing tree is never ours to remove on failure.
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        yield path
    except BaseException as exc:
        from hermes_cli.profiles import _rmtree_make_writable

        try:
            shutil.rmtree(path, onerror=_rmtree_make_writable)
        except OSError as cleanup_error:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                exc.add_note(f"Could not clean distribution stage {path}: {cleanup_error}")
            else:
                raise OSError(f"Could not clean distribution stage {path}: {cleanup_error}") from exc
        raise
