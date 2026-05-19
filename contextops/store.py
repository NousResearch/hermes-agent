"""File-backed local store helpers for ContextOps prototypes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def default_store_root() -> Path:
    """Return the default local/offline ContextOps data directory."""

    return Path(".data/contextops")


class ContextOpsStore:
    """Small JSONL/YAML store rooted under a single local directory.

    Names are logical store names such as ``events`` or ``packs/pack-1``. Path
    traversal and absolute paths are rejected so prototype writes cannot escape
    the configured ContextOps data root.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_store_root()

    def append_jsonl(self, name: str, model: BaseModel | dict[str, Any]) -> Path:
        """Atomically append one JSON-serializable record to ``<name>.jsonl``."""

        path = self._path_for(name, ".jsonl")
        record = self._dump_record(model)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        content = existing + json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        self._atomic_write_text(path, content)
        return path

    def read_jsonl(self, name: str, model_type: type[T]) -> list[T]:
        """Read ``<name>.jsonl`` and validate each row as ``model_type``."""

        path = self._path_for(name, ".jsonl")
        if not path.exists():
            return []
        rows: list[T] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(model_type.model_validate_json(line))
        return rows

    def write_yaml(self, name: str, model: BaseModel | dict[str, Any]) -> Path:
        """Atomically write one YAML document to ``<name>.yaml``."""

        path = self._path_for(name, ".yaml")
        record = self._dump_record(model)
        content = yaml.safe_dump(record, allow_unicode=True, sort_keys=True)
        self._atomic_write_text(path, content)
        return path

    def read_yaml(self, name: str, model_type: type[T]) -> T:
        """Read ``<name>.yaml`` and validate it as ``model_type``."""

        path = self._path_for(name, ".yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return model_type.model_validate(data)

    def _path_for(self, name: str, suffix: str) -> Path:
        logical = Path(name)
        if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
            raise ValueError("store name must be a relative path inside the ContextOps root")
        if logical.suffix:
            path = self.root / logical
        else:
            path = self.root / logical.with_suffix(suffix)
        root = self.root.resolve()
        resolved_parent = path.parent.resolve(strict=False)
        if root != resolved_parent and root not in resolved_parent.parents:
            raise ValueError("store name must stay inside the ContextOps root")
        return path

    @staticmethod
    def _dump_record(model: BaseModel | dict[str, Any]) -> dict[str, Any]:
        if isinstance(model, BaseModel):
            # Fail-closed write boundary: model_copy(update=...) and construct()
            # can produce model instances that never ran field/model validators.
            # Re-validate before persistence so invalid records cannot reach disk
            # and rely on read-time validation to catch them later.
            validated = type(model).model_validate(model.model_dump())
            return validated.model_dump(mode="json")
        return dict(model)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
