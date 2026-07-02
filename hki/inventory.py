"""Workspace inventory generation for HKI."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hki.paths import HkiScope, ensure_output_dir


SCHEMA_VERSION = 1
DEFAULT_HASH_LIMIT_BYTES = 1024 * 1024
SAMPLE_BYTES = 8192

EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
    }
)


@dataclass(frozen=True)
class InventoryFile:
    relative_path: str
    size: int
    mtime: int
    suffix: str
    is_text: bool
    classification: str
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "relative_path": self.relative_path,
            "size": self.size,
            "mtime": self.mtime,
            "suffix": self.suffix,
            "is_text": self.is_text,
            "classification": self.classification,
        }
        if self.sha256:
            data["sha256"] = self.sha256
        return data


@dataclass(frozen=True)
class Inventory:
    root: str
    generated_at: str
    files: tuple[InventoryFile, ...]
    skipped_count: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "root": self.root,
            "included_count": len(self.files),
            "skipped_count": self.skipped_count,
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items())),
            "files": [item.to_dict() for item in self.files],
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_inventory(
    scope: HkiScope,
    *,
    hash_limit_bytes: int = DEFAULT_HASH_LIMIT_BYTES,
) -> Inventory:
    """Scan ``scope.root`` and return an HKI inventory."""

    root = scope.root.resolve()
    files: list[InventoryFile] = []
    skipped: Counter[str] = Counter()

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            full_dir = current / dirname
            reason = _excluded_dir_reason(full_dir, root)
            if reason:
                skipped[reason] += 1
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = current / filename
            reason = _excluded_file_reason(path, root)
            if reason:
                skipped[reason] += 1
                continue
            try:
                item = _inventory_file(path, root, hash_limit_bytes=hash_limit_bytes)
            except OSError:
                skipped["read_error"] += 1
                continue
            if item is None:
                skipped["outside_root"] += 1
                continue
            files.append(item)

    files.sort(key=lambda item: item.relative_path)
    return Inventory(
        root=str(root),
        generated_at=utc_now_iso(),
        files=tuple(files),
        skipped_count=sum(skipped.values()),
        skipped_by_reason=dict(skipped),
    )


def write_inventory(inventory: Inventory, scope: HkiScope) -> Path:
    """Write ``inventory`` to ``.hermes/hki/inventory.json``."""

    ensure_output_dir(scope)
    path = scope.inventory_path
    path.write_text(
        json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_inventory(path: Path) -> Inventory:
    """Load an inventory JSON file."""

    return inventory_from_dict(json.loads(path.read_text(encoding="utf-8")))


def inventory_from_dict(data: dict[str, Any]) -> Inventory:
    files = tuple(
        InventoryFile(
            relative_path=str(item["relative_path"]),
            size=int(item.get("size", 0)),
            mtime=int(item.get("mtime", 0)),
            suffix=str(item.get("suffix", "")),
            is_text=bool(item.get("is_text", False)),
            classification=str(item.get("classification") or ("text" if item.get("is_text") else "binary")),
            sha256=item.get("sha256"),
        )
        for item in data.get("files", [])
    )
    return Inventory(
        root=str(data.get("root", "")),
        generated_at=str(data.get("generated_at", "")),
        files=files,
        skipped_count=int(data.get("skipped_count", 0)),
        skipped_by_reason={str(k): int(v) for k, v in dict(data.get("skipped_by_reason", {})).items()},
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
    )


def _excluded_dir_reason(path: Path, root: Path) -> str | None:
    name = path.name
    if name in EXCLUDED_DIR_NAMES:
        return "excluded_directory"
    try:
        rel = path.relative_to(root)
    except ValueError:
        return "outside_root"
    if _is_hki_output_path(rel):
        return "hki_output"
    if path.is_symlink():
        return "symlink_directory"
    return None


def _excluded_file_reason(path: Path, root: Path) -> str | None:
    name = path.name
    if name == ".env" or name.startswith(".env."):
        return "secret_env_file"
    try:
        rel = path.relative_to(root)
    except ValueError:
        return "outside_root"
    if _is_hki_output_path(rel):
        return "hki_output"
    return None


def _is_hki_output_path(rel: Path) -> bool:
    parts = rel.parts
    return len(parts) >= 2 and parts[0] == ".hermes" and parts[1] == "hki"


def _inventory_file(
    path: Path,
    root: Path,
    *,
    hash_limit_bytes: int,
) -> InventoryFile | None:
    if path.is_symlink():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            return None

    if not path.is_file():
        raise OSError(f"not a regular file: {path}")

    stat = path.stat()
    is_text = _looks_text(path)
    sha256 = _hash_file(path) if stat.st_size <= hash_limit_bytes else None
    rel = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    return InventoryFile(
        relative_path=rel,
        size=int(stat.st_size),
        mtime=int(stat.st_mtime),
        suffix=suffix,
        is_text=is_text,
        classification="text" if is_text else "binary",
        sha256=sha256,
    )


def _looks_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(SAMPLE_BYTES)
    except OSError:
        return False
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
