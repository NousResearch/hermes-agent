"""Stable HKI source manifest generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hki.inventory import Inventory, InventoryFile, load_inventory, utc_now_iso
from hki.paths import HkiScope, as_workspace_relative, ensure_output_dir


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    relative_path: str
    kind: str
    size: int
    mtime: int
    is_text: bool
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "source_id": self.source_id,
            "relative_path": self.relative_path,
            "kind": self.kind,
            "size": self.size,
            "mtime": self.mtime,
            "is_text": self.is_text,
        }
        if self.sha256:
            data["sha256"] = self.sha256
        return data


@dataclass(frozen=True)
class Manifest:
    root: str
    generated_at: str
    sources: tuple[SourceRecord, ...]
    inventory_path: str | None = None
    inventory_skipped_count: int = 0
    inventory_skipped_by_reason: dict[str, int] | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "root": self.root,
            "source_count": len(self.sources),
            "inventory_skipped_count": self.inventory_skipped_count,
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.inventory_skipped_by_reason:
            data["inventory_skipped_by_reason"] = dict(
                sorted(self.inventory_skipped_by_reason.items())
            )
        if self.inventory_path:
            data["inventory_path"] = self.inventory_path
        return data


def build_manifest(inventory: Inventory, *, inventory_path: Path | None = None) -> Manifest:
    """Create a stable source manifest from ``inventory``."""

    inventory_ref = None
    if inventory_path is not None:
        inventory_ref = as_workspace_relative(inventory_path, Path(inventory.root))

    records = tuple(
        SourceRecord(
            source_id=source_id_for(item.relative_path),
            relative_path=item.relative_path,
            kind=kind_for_inventory_file(item),
            size=item.size,
            mtime=item.mtime,
            is_text=item.is_text,
            sha256=item.sha256,
        )
        for item in sorted(inventory.files, key=lambda entry: entry.relative_path)
    )
    return Manifest(
        root=inventory.root,
        generated_at=utc_now_iso(),
        sources=records,
        inventory_path=inventory_ref,
        inventory_skipped_count=inventory.skipped_count,
        inventory_skipped_by_reason=dict(inventory.skipped_by_reason),
    )


def write_manifest(manifest: Manifest, scope: HkiScope) -> Path:
    """Write ``manifest`` to ``.hermes/hki/manifest.json``."""

    ensure_output_dir(scope)
    path = scope.manifest_path
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path: Path) -> Manifest:
    return manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_inventory_for_scope(scope: HkiScope) -> Inventory:
    return load_inventory(scope.inventory_path)


def manifest_from_dict(data: dict[str, Any]) -> Manifest:
    sources = tuple(
        SourceRecord(
            source_id=str(item["source_id"]),
            relative_path=str(item["relative_path"]),
            kind=str(item.get("kind", "unknown")),
            size=int(item.get("size", 0)),
            mtime=int(item.get("mtime", 0)),
            is_text=bool(item.get("is_text", False)),
            sha256=item.get("sha256"),
        )
        for item in data.get("sources", [])
    )
    return Manifest(
        root=str(data.get("root", "")),
        generated_at=str(data.get("generated_at", "")),
        sources=sources,
        inventory_path=data.get("inventory_path"),
        inventory_skipped_count=int(data.get("inventory_skipped_count", 0)),
        inventory_skipped_by_reason={
            str(k): int(v)
            for k, v in dict(data.get("inventory_skipped_by_reason", {})).items()
        },
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
    )


def source_id_for(relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:20]
    return f"src_{digest}"


def kind_for_inventory_file(item: InventoryFile) -> str:
    rel_name = Path(item.relative_path).name.lower()
    suffix = item.suffix.lower()

    special_names = {
        "dockerfile": "dockerfile",
        "makefile": "makefile",
        "readme": "documentation",
        "license": "license",
    }
    if rel_name in special_names:
        return special_names[rel_name]

    by_suffix = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".md": "markdown",
        ".mdx": "markdown",
        ".rst": "documentation",
        ".txt": "text",
        ".json": "json",
        ".jsonl": "jsonl",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "config",
        ".cfg": "config",
        ".csv": "csv",
        ".html": "html",
        ".css": "css",
        ".scss": "css",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".sql": "sql",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".gif": "image",
        ".webp": "image",
        ".pdf": "pdf",
    }
    if suffix in by_suffix:
        return by_suffix[suffix]
    return "text" if item.is_text else "binary"
