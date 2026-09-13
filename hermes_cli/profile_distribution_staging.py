"""Capture distribution files into a private tree before interpreting the plan."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import stat


def read_source_bytes(source) -> bytes:
    if source.is_dir:
        raise OSError("Distribution manifest is not a regular file")
    chunks = []
    while chunk := source.read(1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def copy_source_tree(source, destination: Path, manifest_name: str = "", manifest_bytes: bytes = b"") -> None:
    for name in source.names():
        target = destination / name
        with source.child(name) as entry:
            if entry.is_dir:
                # Scratch directories stay writable until publication/cleanup.
                target.mkdir(mode=0o700)
                copy_source_tree(entry, target)
            else:
                with target.open("xb") as output:
                    if manifest_name and os.path.normcase(name) == os.path.normcase(manifest_name):
                        content = read_source_bytes(entry)
                        if content != manifest_bytes:
                            raise OSError("Distribution manifest changed during staging")
                        output.write(content)
                    else:
                        while chunk := entry.read(1024 * 1024):
                            output.write(chunk)
                os.chmod(target, stat.S_IMODE(entry.mode))


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
