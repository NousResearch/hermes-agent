"""Benign: plain utf-8 on an internal package resource (not user-writable)."""
from pathlib import Path


def load_bundled_schema(root: Path) -> str:
    schema = root / "schemas" / "internal_v1.json"
    return schema.read_text(encoding="utf-8")
