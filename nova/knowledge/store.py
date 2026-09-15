"""Putting documents into a corpus, from the Control Centre.

A knowledge source is a directory plus the rules for reading it. Until now the only way a
document got into one was somebody putting a file on the host's disk, which is fine for a
deployment an engineer runs and useless for a customer who has a handbook to upload.

Three things make this narrower than "write a file where the user says":

**The corpus decides what it accepts.** Every source declares ``include`` globs and a
``max_file_bytes``. A corpus of ``**/*.md`` does not receive an ``.exe`` — not because
this module keeps a list of dangerous extensions, but because the tenant already said what
belongs there and that declaration is the check.

**The name is rebuilt, not trusted.** An uploaded filename comes from a browser and is
attacker-controlled. It is reduced to a safe basename and the result is resolved against
the corpus root and refused if it escapes — a corpus root is often outside the bundle, so
the writer's own bundle confinement does not apply here.

**Indexing is part of the operation.** A document on disk that is not in the index is
invisible to every agent. The caller re-ingests the source and reports what the index
actually recorded, so "uploaded" never means "saved somewhere nothing reads".
"""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

from nova.errors import SpecError

#: Characters kept in a stored filename. Everything else becomes a hyphen. Deliberately
#: strict: this name is joined to a filesystem path and later shown in a UI, and the set of
#: things that are safe in both is small.
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Longest stored basename. Filesystems cap at 255 bytes; leaving room avoids a write that
#: fails only for a user whose language needs more bytes per character.
MAX_NAME = 120


def safe_name(raw: str) -> str:
    """A filename from untrusted input, or a refusal.

    Takes the basename of whatever was sent — a browser can send ``../../etc/passwd`` or a
    Windows path — normalises the unicode so visually identical names do not collide, and
    keeps only characters that are unambiguous on a filesystem and on a screen.
    """
    candidate = PurePosixPath(str(raw or "").replace("\\", "/")).name
    # NFKD then drop combining marks, so an accented name transliterates rather than being
    # punched full of hyphens: "ünïcode.md" should become "unicode.md", not "n-code.md".
    decomposed = unicodedata.normalize("NFKD", candidate.strip())
    candidate = "".join(c for c in decomposed if not unicodedata.combining(c))
    candidate = _SAFE.sub("-", candidate)
    # Tidy the seams the substitution leaves: runs of hyphens, and hyphens sitting either
    # side of the extension dot.
    candidate = re.sub(r"-{2,}", "-", candidate)
    candidate = re.sub(r"-*\.-*", ".", candidate).strip("-.")
    if not candidate or candidate in (".", ".."):
        raise SpecError(
            f"{raw!r} is not a usable filename. Give it a name made of letters, digits, "
            "dots, hyphens or underscores"
        )
    if len(candidate) > MAX_NAME:
        stem, dot, suffix = candidate.rpartition(".")
        keep = MAX_NAME - (len(suffix) + 1 if dot else 0)
        candidate = (stem[:keep] + dot + suffix) if dot else candidate[:MAX_NAME]
    return candidate


def accepts(source, name: str) -> bool:
    """Whether this corpus's own rules admit a file of this name.

    The source's ``include``/``exclude`` globs are the allowlist. NOVA keeps no separate
    list of permitted types: the tenant already declared what belongs in each corpus, and a
    second list would eventually disagree with the first.
    """
    included = any(fnmatch(name, pattern.rsplit("/", 1)[-1]) or fnmatch(name, pattern)
                   for pattern in (source.include or ("**/*",)))
    if not included:
        return False
    return not any(fnmatch(name, pattern.rsplit("/", 1)[-1]) or fnmatch(name, pattern)
                   for pattern in (source.exclude or ()))


def _resolved(source, name: str) -> Path:
    root = Path(source.root).resolve()
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise SpecError(f"{name!r} resolves outside the corpus directory") from None
    return target


def list_documents(source) -> tuple[dict[str, Any], ...]:
    """What is in the corpus now, as the ingester would see it.

    Walked with the source's own include/exclude rules, so a file sitting in the directory
    that the corpus does not admit is not listed as though an agent could read it.
    """
    root = Path(source.root)
    if not root.is_dir():
        return ()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover — rglob cannot produce this
            continue
        if not accepts(source, relative):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append({
            "name": relative,
            "bytes": stat.st_size,
            "modified_at": int(stat.st_mtime),
            # Whether it exceeds what the ingester will read. Shown rather than hidden: a
            # file too large to index is exactly the one somebody will wonder about.
            "too_large": stat.st_size > source.max_file_bytes,
        })
    return tuple(rows)


def store_document(source, *, filename: str, data: bytes, replace: bool = False) -> dict[str, Any]:
    """Write one document into the corpus. Returns what was stored.

    Refuses rather than overwrites unless ``replace`` is set: an upload that silently
    replaced a document of the same name would be a way to change what every agent reads
    without anyone seeing a change.
    """
    name = safe_name(filename)
    if not accepts(source, name):
        raise SpecError(
            f"{name!r} is not a document {source.id!r} accepts. This corpus takes "
            f"{', '.join(source.include)}"
            + (f" and excludes {', '.join(source.exclude)}" if source.exclude else "")
        )
    if not data:
        raise SpecError(f"{name!r} is empty")
    if len(data) > source.max_file_bytes:
        raise SpecError(
            f"{name!r} is {len(data) // 1000} kB; {source.id!r} accepts up to "
            f"{source.max_file_bytes // 1000} kB per document. The limit is the corpus's "
            "own `max_file_bytes`"
        )

    target = _resolved(source, name)
    if target.exists() and not replace:
        raise SpecError(
            f"{name!r} is already in {source.id!r}. Send it again with replace to overwrite "
            "it — an upload that silently replaced a document would change what every agent "
            "reads with nothing to see"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".nova-doc-")
    try:
        with os.fdopen(handle, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    return {"name": name, "bytes": len(data), "replaced": bool(replace and target.exists())}


def remove_document(source, name: str) -> bool:
    """Delete one document from the corpus. False when it was not there.

    The file goes; the index does not follow until the source is re-ingested, which is the
    caller's job and is why the caller reports the index result rather than this function.
    """
    target = _resolved(source, safe_name(name))
    if not target.is_file():
        return False
    target.unlink()
    return True
