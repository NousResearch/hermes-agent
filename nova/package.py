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
from pathlib import Path, PurePosixPath
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


def _members_of(archive: Path) -> list[str]:
    """Every regular-file member of ``archive``, refusing anything that could escape.

    Checked against a notional root rather than the real destination, so the refusal
    happens before a single byte is written and cannot depend on where the caller is
    extracting to. The archive is written by NOVA, but it travels through a bucket and a
    host, and the one place a tar extraction must not trust its input is after it has been
    somewhere else.
    """
    names: list[str] = []
    with tarfile.open(Path(archive), mode="r:gz") as handle:
        for info in handle.getmembers():
            if not info.isfile():
                raise SpecError(
                    f"{archive} contains {info.name!r}, which is not a regular file"
                )
            name = info.name
            if name.startswith("/") or PurePosixPath(name).is_absolute():
                raise SpecError(f"{archive} contains an absolute path {name!r}")
            if ".." in PurePosixPath(name).parts:
                raise SpecError(
                    f"{archive} contains {name!r}, which would extract outside the "
                    "destination"
                )
            names.append(name)
    return names


def _is_empty(path: Path) -> bool:
    """True when ``path`` does not exist, or exists and holds nothing at all.

    Any entry counts, including a dotfile. A destination with something in it is one
    somebody else is using, and guessing which leftovers are harmless is how the stale
    ``channels.yaml`` survived a bundle that had deleted it.
    """
    if not path.exists():
        return True
    return not any(path.iterdir())


def unpack(
    archive: Path, destination: Path, *, replace: bool = False
) -> dict[str, object]:
    """Extract an archive written by :func:`package` into ``destination``.

    **Extraction replaces, it never merges.** A bundle is a complete statement of what a
    tenant declares, so a file deleted from it must disappear from the deployed copy —
    otherwise deleting a declaration locally does nothing on the host, and the bundle
    stops being the source of truth the whole design rests on. That is not hypothetical:
    a merging unpack left a removed ``channels.yaml`` in place on the first field
    deployment, which kept a channel alive, kept its channel-scoped agent profile
    materialized, and made the host's bundle digest disagree with the archive it had just
    been given — from identical bytes.

    Because replacing is destructive, it is never the default. A non-empty destination is
    refused unless ``replace=True``, so the dangerous act is the one somebody typed.

    **Ordered so a failure cannot leave a half-updated bundle.** Members are validated,
    then extracted to a staging directory beside the destination, then the result is
    loaded and parsed — and only once all of that has succeeded is anything in the
    destination touched. The swap itself is two renames on one filesystem: the old tree
    moves aside, the new one moves in. A failure before the swap leaves the destination
    untouched; a failure between the renames restores the old tree.
    """
    import shutil
    import uuid

    archive = Path(archive)
    destination = Path(destination)
    names = _members_of(archive)
    if not names:
        raise SpecError(f"{archive} contains no files")

    empty = _is_empty(destination)
    if not empty and not replace:
        existing = sorted(p.name for p in destination.iterdir())[:5]
        raise SpecError(
            f"{destination} is not empty (contains {', '.join(existing)}"
            f"{', …' if len(existing) == 5 else ''}). Unpacking would merge the archive "
            "into what is already there and leave behind any file this bundle has "
            "deleted, so the deployed bundle would stop matching the one you packaged. "
            "Re-run with --replace to replace the directory's contents, or unpack into "
            "an empty directory."
        )

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Beside the destination, so the swap below is a rename within one filesystem rather
    # than a copy that could half-finish.
    staging = parent / f".{destination.name}.incoming-{uuid.uuid4().hex[:12]}"
    retired = parent / f".{destination.name}.retired-{uuid.uuid4().hex[:12]}"

    try:
        staging.mkdir(parents=True)
        with tarfile.open(archive, mode="r:gz") as handle:
            handle.extractall(staging)  # noqa: S202 — every member checked by _members_of

        # Parsed before anything is replaced: a truncated or malformed archive must not be
        # able to destroy a working bundle on its way to failing.
        from nova.spec import load_bundle

        load_bundle(staging)

        removed: list[str] = []
        if empty:
            if destination.exists():
                destination.rmdir()
            staging.rename(destination)
        else:
            before = {
                str(p.relative_to(destination).as_posix())
                for p in destination.rglob("*") if p.is_file()
            }
            removed = sorted(before - set(names))
            destination.rename(retired)
            try:
                staging.rename(destination)
            except BaseException:
                # Put the old tree back rather than leaving nothing where the
                # authoritative bundle used to be.
                retired.rename(destination)
                raise
            shutil.rmtree(retired, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return {
        "destination": str(destination),
        "files": len(names),
        "members": sorted(names),
        "replaced": not empty,
        "removed": removed if not empty else [],
    }
