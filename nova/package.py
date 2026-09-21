"""Packaging a tenant bundle for transport, reproducibly.

Getting a bundle onto a runtime host was a shell recipe: ``tar czf`` on a laptop, an
object in a bucket, ``tar xzf ... --strip-components=1`` over SSM. Every step of that is
fine and one of them is a trap — drop ``--strip-components=1``, or extract one directory
too high, and the archive's tenant-named root becomes a second bundle directory that
nothing reads. The deployment then looks complete and serves the old configuration,
which is exactly what happened on the first field deployment (``/var/lib/nova/test``
beside ``/var/lib/nova/bundle``, both stale).

So the archive this writes is **rooted at the bundle's contents**: ``organization.yaml``
is at the top level, not ``<tenant>/organization.yaml``. There is no strip to forget, and
extracting into the wrong directory produces a visibly wrong result instead of a quiet
second copy.

**Deterministic.** Names sorted, mtime zeroed, uid/gid/uname/gname cleared, modes
normalised, gzip written without its timestamp field. The same bundle packages to the
same bytes on any machine, so ``sha256`` of the archive is a real identity an operator
can compare across a laptop, a bucket and a host — and a changed byte is a changed
bundle, not a changed clock.

Two digests, deliberately, because they answer different questions:

``bundle digest`` (``BundleSpec.digest()``)
    The hash of the *parsed* specification. Reformatting a YAML file does not change it.
    This is the one that answers "is the deployed configuration the one I reviewed".
``archive sha256``
    The hash of the bytes. This is the one that answers "did the file arrive intact".
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from pathlib import Path
from typing import Iterator

from nova.errors import SpecError

#: Never packaged. Editor droppings and VCS metadata are not tenant declarations, and a
#: ``.git`` directory in a bundle would ship history into a runtime host.
EXCLUDED_NAMES = frozenset({".git", ".DS_Store", "__pycache__", ".idea", ".vscode"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".swp", "~")

#: Fixed metadata for every member. The archive describes a configuration, not a
#: filesystem: whose uid wrote it and when is noise that would change the bytes.
_FIXED_MTIME = 0
_FILE_MODE = 0o644
_DIR_MODE = 0o755


def _included(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES or path.name.endswith(EXCLUDED_SUFFIXES):
        return False
    return not any(part in EXCLUDED_NAMES for part in path.parts)


def _members(root: Path) -> Iterator[tuple[Path, str]]:
    """Every packaged file, as ``(absolute path, archive name)``, in sorted order."""
    for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
        if path.is_symlink():
            # A symlink in a bundle either dangles on the host or escapes the bundle.
            # Neither is a tenant declaration; refuse rather than package a surprise.
            raise SpecError(
                f"{path} is a symbolic link. A bundle is the declarations themselves, "
                "so that either dangles on the runtime host or points outside the bundle"
            )
        if not path.is_file() or not _included(path.relative_to(root)):
            continue
        yield path, str(path.relative_to(root).as_posix())


def package(bundle_dir: Path, out: Path) -> dict[str, object]:
    """Write a deterministic archive of ``bundle_dir``. Returns what it wrote.

    The bundle is loaded first, so an archive is never produced from declarations that do
    not parse — shipping one would move the failure from a laptop to a runtime host.
    """
    from nova.spec import load_bundle

    root = Path(bundle_dir)
    bundle = load_bundle(root)

    names: list[str] = []
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path, name in _members(root):
            info = archive.gettarinfo(str(path), arcname=name)
            info.mtime = _FIXED_MTIME
            info.mode = _FILE_MODE
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)
            names.append(name)

    if not names:
        raise SpecError(f"{root} contains no files to package")

    # Two gzip header fields would otherwise leak the machine into the bytes: mtime, and
    # the *output filename*, which GzipFile takes from `fileobj.name` unless told
    # otherwise. Both are cleared. Without the second, packaging the same bundle to
    # `a.tgz` and `b.tgz` gives different hashes — caught here, not by an operator
    # comparing digests across a laptop and a bucket and finding they never match.
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = raw.getvalue()
    with open(out, "wb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as compressor:
            compressor.write(payload)

    return {
        "archive": str(out),
        "tenant_id": bundle.tenant_id,
        "bundle_digest": bundle.digest(),
        "archive_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "files": len(names),
        "members": names,
    }


def unpack(archive: Path, destination: Path) -> dict[str, object]:
    """Extract an archive written by :func:`package` into ``destination``.

    Refuses any member that would land outside ``destination``. The archive is written by
    NOVA, but it travels through a bucket and a host, and the one place a tar extraction
    must not trust its input is after it has been somewhere else.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()

    names: list[str] = []
    with tarfile.open(Path(archive), mode="r:gz") as handle:
        for info in handle.getmembers():
            if not info.isfile():
                raise SpecError(
                    f"{archive} contains {info.name!r}, which is not a regular file"
                )
            target = (destination / info.name).resolve()
            if not str(target).startswith(str(resolved_root) + "/"):
                raise SpecError(
                    f"{archive} contains {info.name!r}, which would extract outside "
                    f"{destination}"
                )
            names.append(info.name)
        handle.extractall(destination)  # noqa: S202 — every member checked above

    return {"destination": str(destination), "files": len(names), "members": sorted(names)}
