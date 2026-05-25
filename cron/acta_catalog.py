"""Persistent catalog helpers for Acta output artifacts.

The catalog is intentionally small and dashboard-agnostic: it stores durable
metadata for first-class Acta outputs while preserving user-controlled state
across repeated imports/upserts.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from hermes_constants import get_hermes_home

CATALOG_VERSION = 1
USER_STATE_FIELDS = ("pinned", "read", "archived")
DEFAULT_CATALOG_FILENAME = "catalog.json"
DEFAULT_OUTPUTS_DIR = "acta-outputs"


@dataclass(frozen=True)
class ActaCatalogEntry:
    """A durable, public-safe record for an Acta output."""

    id: str
    title: str
    href: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    source_ref: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    pinned: bool = False
    read: bool = False
    archived: bool = False


def default_catalog_path(hermes_home: Path | None = None) -> Path:
    """Return the default persistent Acta output catalog path."""

    home = hermes_home or get_hermes_home()
    return home / "acta" / DEFAULT_CATALOG_FILENAME


def default_outputs_dir(hermes_home: Path | None = None) -> Path:
    """Return the default directory containing static Acta output HTML files."""

    home = hermes_home or get_hermes_home()
    return home / "artifacts" / DEFAULT_OUTPUTS_DIR


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with a stable timezone suffix."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str, *, fallback: str = "output") -> str:
    """Return a stable lower-case slug suitable for catalog IDs."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or fallback


def new_catalog() -> dict[str, Any]:
    """Return an empty normalized catalog document."""

    return {"version": CATALOG_VERSION, "outputs": []}


def load_catalog(path: Path | str) -> dict[str, Any]:
    """Load a catalog JSON file, returning an empty catalog when missing/malformed.

    Invalid top-level shapes are treated as empty instead of leaking malformed
    state into callers. Individual non-object entries are ignored. A hand-edited
    or partially-written catalog must not break Acta publishing.
    """

    catalog_path = Path(path)
    if not catalog_path.exists():
        return new_catalog()
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return new_catalog()
    if not isinstance(data, Mapping):
        return new_catalog()
    outputs = data.get("outputs", [])
    if not isinstance(outputs, list):
        outputs = []
    try:
        version = int(data.get("version") or CATALOG_VERSION)
    except (TypeError, ValueError):
        version = CATALOG_VERSION
    if version != CATALOG_VERSION:
        return new_catalog()
    return {
        "version": version,
        "outputs": [normalize_entry(entry) for entry in outputs if isinstance(entry, Mapping)],
    }


def save_catalog(catalog: Mapping[str, Any], path: Path | str) -> Path:
    """Atomically write a normalized catalog JSON file.

    The write uses a same-directory temporary file plus ``os.replace`` so readers
    never observe a partially-written JSON document.
    """

    catalog_path = Path(path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_catalog(catalog)
    payload = json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{catalog_path.name}.", suffix=".tmp", dir=str(catalog_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, catalog_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return catalog_path


def normalize_catalog(catalog: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize catalog shape and sort entries deterministically."""

    if not isinstance(catalog, Mapping):
        return new_catalog()
    outputs = catalog.get("outputs", [])
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes, bytearray)):
        outputs = []
    entries = [normalize_entry(entry) for entry in outputs if isinstance(entry, Mapping)]
    entries.sort(key=lambda entry: (not bool(entry.get("pinned")), entry.get("title", "").casefold(), entry.get("id", "")))
    return {"version": CATALOG_VERSION, "outputs": entries}


def normalize_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a single catalog entry and remove unsafe source detail."""

    raw_id = str(entry.get("id") or entry.get("slug") or entry.get("title") or "output")
    entry_id = slugify(raw_id)
    now = utc_now_iso()
    return asdict(
        ActaCatalogEntry(
            id=entry_id,
            title=str(entry.get("title") or entry_id.replace("-", " ").title()),
            href=_safe_href(entry.get("href")),
            summary=str(entry.get("summary") or ""),
            tags=_normalize_tags(entry.get("tags")),
            source_ref=redact_source_ref(entry.get("source_ref") or entry.get("source") or {}),
            created_at=str(entry.get("created_at") or now),
            updated_at=str(entry.get("updated_at") or now),
            pinned=bool(entry.get("pinned", False)),
            read=bool(entry.get("read", False)),
            archived=bool(entry.get("archived", False)),
        )
    )


def upsert_output(catalog: MutableMapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    """Insert or update an output entry, preserving durable user state fields."""

    normalized = normalize_entry(entry)
    outputs = catalog.setdefault("outputs", [])
    if not isinstance(outputs, list):
        outputs = []
        catalog["outputs"] = outputs

    for index, existing in enumerate(outputs):
        if isinstance(existing, Mapping) and slugify(str(existing.get("id") or "")) == normalized["id"]:
            merged = {**normalized}
            existing_norm = normalize_entry(existing)
            for field_name in USER_STATE_FIELDS:
                merged[field_name] = bool(existing_norm.get(field_name, False))
            merged["created_at"] = str(existing_norm.get("created_at") or normalized["created_at"])
            outputs[index] = merged
            catalog["version"] = CATALOG_VERSION
            return merged

    outputs.append(normalized)
    catalog["version"] = CATALOG_VERSION
    return normalized


def promote_output(catalog: MutableMapping[str, Any], output_id: str, *, read: bool | None = None) -> dict[str, Any]:
    """Promote an output to the live pinned shelf.

    Promotion keeps the entry, sets ``pinned=True`` and ``archived=False``. If
    ``read`` is supplied it explicitly updates the read state as well.
    """

    wanted = slugify(output_id)
    for entry in catalog.get("outputs", []):
        if isinstance(entry, MutableMapping) and slugify(str(entry.get("id") or "")) == wanted:
            entry["pinned"] = True
            entry["archived"] = False
            if read is not None:
                entry["read"] = bool(read)
            entry["updated_at"] = utc_now_iso()
            return normalize_entry(entry)
    raise KeyError(f"Acta output not found: {output_id}")


def import_acta_outputs(
    artifacts_dir: Path | str | None = None,
    *,
    catalog_path: Path | str | None = None,
    hermes_home: Path | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Import static HTML outputs from ``~/.hermes/artifacts/acta-outputs``.

    The importer is idempotent: each file maps to a stable slug ID derived from
    the filename, and repeated imports update metadata without duplicating rows
    or overwriting user state.
    """

    root = Path(artifacts_dir) if artifacts_dir is not None else default_outputs_dir(hermes_home)
    path = Path(catalog_path) if catalog_path is not None else default_catalog_path(hermes_home)
    catalog = load_catalog(path)
    if not root.exists():
        if save:
            save_catalog(catalog, path)
        return catalog

    index_metadata = _metadata_from_index(root / "index.html")
    for html_path in sorted(root.glob("*.html")):
        if html_path.name == "index.html":
            continue
        stem = slugify(html_path.stem)
        metadata = index_metadata.get(stem, {})
        entry = _entry_from_html_file(html_path, root=root, metadata=metadata)
        upsert_output(catalog, entry)

    normalized = normalize_catalog(catalog)
    if save:
        save_catalog(normalized, path)
    return normalized


def redact_source_ref(value: object) -> dict[str, str]:
    """Return a public-safe source reference with local paths and secrets removed."""

    if isinstance(value, (str, Path)):
        name = Path(str(value)).name
        return {"kind": "acta-output", "name": name} if name else {"kind": "acta-output"}
    if not isinstance(value, Mapping):
        return {"kind": "acta-output"}
    kind = str(value.get("kind") or value.get("type") or "acta-output")
    name = str(value.get("name") or Path(str(value.get("path") or value.get("file") or "")).name)
    redacted: dict[str, str] = {"kind": kind}
    if name:
        redacted["name"] = name
    label = value.get("label")
    if label:
        redacted["label"] = str(label)
    return redacted


def _safe_href(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value) or "\\" in value:
        return ""
    if value == "/outputs":
        return ""
    if re.fullmatch(r"/outputs/[a-z0-9][a-z0-9-]*/?", value):
        return value.rstrip("/")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        return f"/outputs/{value}"
    return ""


def _normalize_tags(value: object) -> list[str]:
    if isinstance(value, str):
        raw = re.split(r"[,\s]+", value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        raw = [str(item) for item in value]
    else:
        raw = []
    tags: list[str] = []
    for item in raw:
        tag = slugify(str(item), fallback="")
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _entry_from_html_file(html_path: Path, *, root: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    stem = slugify(html_path.stem)
    title = str(metadata.get("title") or _html_title(text) or stem.replace("-", " ").title())
    summary = str(metadata.get("summary") or _first_paragraph(text) or "")
    tags = metadata.get("tags") or _infer_tags(stem, title, summary)
    mtime = datetime.fromtimestamp(html_path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
    return {
        "id": stem,
        "title": title,
        "href": f"/outputs/{stem}",
        "summary": summary,
        "tags": tags,
        "source_ref": {"kind": "acta-output", "name": html_path.name},
        "created_at": str(metadata.get("created_at") or mtime),
        "updated_at": mtime,
    }


def _metadata_from_index(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        return {}
    text = index_path.read_text(encoding="utf-8", errors="replace")
    metadata: dict[str, dict[str, Any]] = {}
    for match in re.finditer(r"<article\b(?P<attrs>[^>]*)>(?P<body>.*?)</article>", text, flags=re.I | re.S):
        attrs = match.group("attrs")
        body = match.group("body")
        output_id = _attr(attrs, "data-id") or _href_slug(_attr(attrs, "data-href"))
        if not output_id:
            continue
        title = _attr(attrs, "data-title") or _tag_text(body, "h2")
        summary = _tag_text(body, "p")
        tags = _attr(attrs, "data-tags")
        meta_text = _tag_text_by_class(body, "meta")
        metadata[slugify(output_id)] = {
            "title": title,
            "summary": summary,
            "tags": _normalize_tags(tags),
            "created_at": _published_timestamp(meta_text),
        }
    return metadata


def _html_title(text: str) -> str:
    return _tag_text(text, "title")


def _first_paragraph(text: str) -> str:
    return _tag_text(text, "p")


def _tag_text(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, flags=re.I | re.S)
    return _clean_html_text(match.group(1)) if match else ""


def _tag_text_by_class(text: str, class_name: str) -> str:
    match = re.search(rf"<[^>]+class=[\"'][^\"']*\b{re.escape(class_name)}\b[^\"']*[\"'][^>]*>(.*?)</[^>]+>", text, flags=re.I | re.S)
    return _clean_html_text(match.group(1)) if match else ""


def _clean_html_text(value: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", value, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _attr(attrs: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}=[\"']([^\"']*)[\"']", attrs, flags=re.I)
    return match.group(1).strip() if match else ""


def _href_slug(href: str) -> str:
    if not href:
        return ""
    return slugify(href.rstrip("/").rsplit("/", 1)[-1])


def _published_timestamp(meta_text: str) -> str:
    match = re.search(r"Published\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+UTC", meta_text or "", flags=re.I)
    if not match:
        return ""
    return f"{match.group(1)}T{match.group(2)}:00+00:00"


def _infer_tags(*values: str) -> list[str]:
    words = " ".join(values).lower()
    tags = ["acta"]
    for keyword in ("hermes", "agents", "decision", "tree", "roadmap", "strategy"):
        if keyword in words:
            tags.append(keyword)
    return tags
