"""Where a corpus's documents come from, when they do not come from the host.

A knowledge source is a directory. That is still true — ingest walks a directory and
nothing about chunking, indexing or retrieval changes here. What this adds is a declared
*origin* for that directory: a bucket the documents are mirrored from, so a customer whose
handbook already lives in S3 does not have to copy it onto the NOVA host by hand.

**The corpus model does not change.** An origin syncs into the same ``root`` the local
corpora use, and the existing ingest runs over the result. That is deliberate: a second
ingestion path for remote sources would eventually disagree with the first about chunking,
provenance or what counts as a document.

**A mirrored corpus is read-only from NOVA's side.** Uploading into one would work exactly
until the next sync deleted it, so the upload route refuses and says why. The place to add
a document to a mirrored corpus is the bucket.

**NOVA holds no credentials for this.** boto3 resolves them the way it always does — the
instance role on EC2, the environment elsewhere — which is the same posture as every other
integration: the runtime's identity is granted access, and NOVA never stores a key.
"""

from __future__ import annotations

import fnmatch
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from nova.errors import SpecError

#: Origin kinds. One today; the shape exists so a second (GCS, Azure, a git remote) is an
#: entry here rather than a second concept.
KINDS = ("s3",)

#: NOVA's own bookkeeping inside a mirrored corpus: which object was downloaded at which
#: ETag, so an unchanged one is not fetched again. It lives in the corpus root because that
#: is what it describes, which makes it the one file in there that is NOT a customer
#: document — ``nova.knowledge.sources`` excludes it from the walk by this name.
MARKER_NAME = ".nova-sync.json"


@dataclass(frozen=True)
class Origin:
    """Where a corpus is mirrored from."""

    kind: str
    bucket: str = ""
    prefix: str = ""
    region: str = ""
    #: Delete local files the remote no longer has. Defaults to True — a mirror that only
    #: ever added would keep answering from a document the customer deleted, which is the
    #: worse failure of the two.
    prune: bool = True
    #: Endpoint for an S3-compatible store. Empty means AWS.
    endpoint_url: str = ""

    @classmethod
    def parse(cls, doc) -> Optional["Origin"]:
        if doc is None:
            return None
        kind = doc.choice("type", KINDS, default="s3")
        spec = cls(
            kind=kind,
            bucket=doc.str_("bucket", required=True),
            prefix=doc.str_("prefix"),
            region=doc.str_("region"),
            prune=doc.bool_("prune", default=True),
            endpoint_url=doc.str_("endpoint_url"),
        )
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.kind, "bucket": self.bucket, "prune": self.prune}
        for key, value in (("prefix", self.prefix), ("region", self.region),
                           ("endpoint_url", self.endpoint_url)):
            if value:
                out[key] = value
        return out

    @property
    def location(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}".rstrip("/")


@dataclass
class SyncReport:
    """What one sync did. Counts, never contents."""

    source_id: str
    location: str = ""
    downloaded: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: list[dict[str, str]] = None  # type: ignore[assignment]
    error: str = ""

    def __post_init__(self) -> None:
        if self.skipped is None:
            self.skipped = []

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id, "location": self.location,
            "downloaded": self.downloaded, "unchanged": self.unchanged,
            "removed": self.removed, "skipped": list(self.skipped),
            "ok": self.ok, "error": self.error,
        }


def _client(origin: Origin):
    """A boto3 S3 client, or a refusal that names what is missing.

    Imported here rather than at module scope so NOVA keeps loading with the standard
    library and PyYAML. A deployment that never declares an S3 origin never needs boto3.
    """
    try:
        import boto3  # type: ignore
    except ImportError:
        raise SpecError(
            "this corpus is mirrored from S3, which needs boto3. Install it "
            "(`pip install boto3`) on the host running NOVA — it is not a NOVA dependency "
            "because a deployment with no S3 corpus does not need it"
        ) from None
    kwargs: dict[str, Any] = {}
    if origin.region:
        kwargs["region_name"] = origin.region
    if origin.endpoint_url:
        kwargs["endpoint_url"] = origin.endpoint_url
    return boto3.client("s3", **kwargs)


def _accepted(source, relative: str) -> bool:
    from nova.knowledge.store import accepts

    return accepts(source, relative)


def _safe_target(root: Path, relative: str) -> Optional[Path]:
    """Where an object lands locally, or None if the key will not stay inside the root.

    An object key comes from a bucket, and a bucket is not necessarily one NOVA's operator
    controls end to end. A key of ``../../etc/cron.d/x`` must not become a write there.
    """
    if not relative or relative.endswith("/"):
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def sync(source, *, dry_run: bool = False) -> SyncReport:
    """Mirror a corpus's origin into its root. Returns what changed.

    Compares by size and ETag recorded alongside each file, so an unchanged object is not
    re-downloaded. Objects the corpus does not accept are skipped and named — a bucket
    holding PDFs and a corpus declaring ``**/*.md`` is a normal, survivable mismatch, and
    silently ignoring it would leave somebody wondering where their documents went.
    """
    origin = getattr(source, "origin", None)
    if origin is None:
        raise SpecError(f"{source.id!r} has no declared origin to sync from")

    report = SyncReport(source_id=source.id, location=origin.location)
    root = Path(source.root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / MARKER_NAME

    try:
        client = _client(origin)
    except SpecError as exc:
        report.error = str(exc)
        return report

    known = _read_marker(marker)
    seen: dict[str, str] = {}

    try:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=origin.bucket, **({"Prefix": origin.prefix} if origin.prefix else {})
        )
        for page in pages:
            for entry in page.get("Contents", ()) or ():
                key = str(entry.get("Key") or "")
                relative = key[len(origin.prefix):].lstrip("/") if origin.prefix else key
                if not relative or relative.endswith("/"):
                    continue
                if not _accepted(source, relative):
                    report.skipped.append({"document": relative, "reason": "not accepted by this corpus"})
                    continue
                size = int(entry.get("Size") or 0)
                if size > source.max_file_bytes:
                    report.skipped.append({"document": relative, "reason": "larger than max_file_bytes"})
                    continue
                target = _safe_target(root, relative)
                if target is None:
                    report.skipped.append({"document": relative, "reason": "key resolves outside the corpus"})
                    continue

                etag = str(entry.get("ETag") or "").strip('"')
                seen[relative] = etag
                if known.get(relative) == etag and target.is_file():
                    report.unchanged += 1
                    continue
                if dry_run:
                    report.downloaded += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                handle, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".nova-s3-")
                os.close(handle)
                try:
                    client.download_file(origin.bucket, key, tmp)
                    os.replace(tmp, target)
                except BaseException:
                    Path(tmp).unlink(missing_ok=True)
                    raise
                report.downloaded += 1
    except SpecError:
        raise
    except Exception as exc:  # noqa: BLE001 — every boto3 failure mode is a report, not a crash
        report.error = f"{type(exc).__name__}: {exc}"
        return report

    if origin.prune and not dry_run:
        for relative in sorted(set(known) - set(seen)):
            target = _safe_target(root, relative)
            if target is not None and target.is_file():
                target.unlink()
                report.removed += 1

    if not dry_run:
        _write_marker(marker, seen)
    return report


def _read_marker(path: Path) -> dict[str, str]:
    """What the last sync downloaded, so an unchanged object is not fetched again."""
    import json

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}


def _write_marker(path: Path, seen: Mapping[str, str]) -> None:
    import json

    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".nova-marker-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(dict(seen), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
